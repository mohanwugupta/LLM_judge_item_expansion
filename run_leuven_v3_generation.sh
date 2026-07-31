#!/bin/bash
#SBATCH --job-name=leuven_v3_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v3_gen_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v3_gen_%j.err

# =============================================================================
# Leuven v3 free feature generation - production
#
# Standalone production job following the established Leuven cluster template.
# It uses one Qwen2.5-72B-Instruct model and collects 20 responses per word for
# prompt conditions A, B, and C: 293 x 3 x 20 = 17,580 planned calls.
#
# Submit only after run_leuven_v3_smoke_test.sh passes:
#   sbatch run_leuven_v3_generation.sh
# =============================================================================

set -eo pipefail

VLLM_PORT=8021

echo "=========================================="
echo " Leuven v3 Feature Generation"
echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Time:    $(date)"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo ""

# ------------------------------------------------------------------
# 1. Configuration (copied from the working Leuven production job)
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion
MODEL_DIR_NAME=Qwen--Qwen2.5-72B-Instruct
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/$MODEL_DIR_NAME
SERVED_MODEL_NAME=Qwen2.5-72B-Instruct
CONDA_ENV=PromptControlText
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.92

LEUVEN_FEATURES=data/leuven_combined_features_consolidated.csv

JOB_ID=leuven_v3_qwen2_5_72b
OUTPUT_DIR=artifacts/leuven_feature_generation/$JOB_ID

RESPONSES_PER_WORD=20
TEMPERATURE=0.8
BASE_SEED=20260801
MAX_WORKERS=64
MAX_TOKENS=500

# ------------------------------------------------------------------
# 2. Environment setup
# ------------------------------------------------------------------
cd "$PROJECT_DIR"

module load anaconda3/2025.6

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ------------------------------------------------------------------
# 3. Cache & offline settings
# ------------------------------------------------------------------
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_DISK_DIR=$HF_DATASETS_CACHE
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/usage_stats
export TRITON_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/triton
export XDG_CACHE_HOME=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache/xdg
export TIKTOKEN_CACHE_DIR=$HOME/.tiktoken_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
mkdir -p "$VLLM_CACHE_DIR" "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------------------------------------------------------------------
# 4. GPU and memory settings
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ------------------------------------------------------------------
# 5. Validate prerequisites
# ------------------------------------------------------------------
if [ -d "$MODEL_PATH" ]; then
    echo "✅ Model found: $MODEL_PATH"
else
    echo "❌ ERROR: Model not found at: $MODEL_PATH"
    exit 1
fi

if [ -f "$PROJECT_DIR/$LEUVEN_FEATURES" ]; then
    echo "✅ Leuven features found: $LEUVEN_FEATURES"
else
    echo "❌ ERROR: Leuven features not found: $PROJECT_DIR/$LEUVEN_FEATURES"
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/$OUTPUT_DIR"

# ------------------------------------------------------------------
# 6. Start vLLM server
# ------------------------------------------------------------------
echo ""
echo "Starting vLLM server ($SERVED_MODEL_NAME, TP=$TENSOR_PARALLEL_SIZE, port=$VLLM_PORT)..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --disable-custom-all-reduce \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 7. Wait for server readiness
# ------------------------------------------------------------------
echo "Waiting for vLLM server on port $VLLM_PORT..."
MAX_WAIT=600
WAIT_INTERVAL=15
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM server exited unexpectedly"
        exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM server ready after ${ELAPSED}s"
        break
    fi
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM server failed to start within ${MAX_WAIT}s"
    exit 1
fi

# ------------------------------------------------------------------
# 8. Run v3 production generation
# ------------------------------------------------------------------
echo ""
echo "Running full v3 generation: 293 words x 3 prompts x $RESPONSES_PER_WORD responses"

python -m leuven_expansion.generate_features \
    --input-csv          "$PROJECT_DIR/$LEUVEN_FEATURES" \
    --job-id             "$JOB_ID" \
    --output-dir         "$PROJECT_DIR/$OUTPUT_DIR" \
    --model              "$SERVED_MODEL_NAME" \
    --base-url           "http://localhost:${VLLM_PORT}/v1" \
    --responses-per-word "$RESPONSES_PER_WORD" \
    --temperature        "$TEMPERATURE" \
    --base-seed          "$BASE_SEED" \
    --max-workers        "$MAX_WORKERS" \
    --max-tokens         "$MAX_TOKENS" \
    --resume

# ------------------------------------------------------------------
# 9. Post-run checks
# ------------------------------------------------------------------
echo ""
echo "--- Post-run checks ---"
PASS=0
FAIL=0

check_file() {
    local f="$PROJECT_DIR/$OUTPUT_DIR/$1"
    local desc="$2"
    if [ -f "$f" ]; then
        local rows
        rows=$(wc -l < "$f")
        echo "  ✅ $desc: $f  ($rows lines)"
        PASS=$((PASS + 1))
    else
        echo "  ❌ MISSING $desc: $f"
        FAIL=$((FAIL + 1))
    fi
}

check_file "feature_generations.csv"           "participant responses"
check_file "generated_features_long.csv"       "long-form generated features"
check_file "generated_feature_frequencies.csv" "exact-string frequencies"
check_file "parse_errors.csv"                  "parse errors"
check_file "manifest.json"                     "manifest"
check_file "run.log"                           "run log"

MANIFEST="$PROJECT_DIR/$OUTPUT_DIR/manifest.json"
if [ -f "$MANIFEST" ]; then
    if python3 -c "import json,sys; d=json.load(open('$MANIFEST')); expected=d['word_count'] * d['responses_per_word_per_prompt']; counts=d.get('valid_responses_by_prompt', {}); ok=d.get('finished_at') and d.get('pending_after_run') == 0 and set(counts) == {'A','B','C'} and all(v == expected for v in counts.values()); sys.exit(0 if ok else 1)"; then
        echo "  ✅ manifest.json records a complete A/B/C run"
        PASS=$((PASS + 1))
    else
        echo "  ❌ manifest.json is incomplete or has missing prompt responses"
        FAIL=$((FAIL + 1))
    fi
fi

echo ""
echo "Production summary: $PASS checks passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo "❌ Production checks FAILED — inspect the SLURM and run logs"
    exit 1
fi

echo "✅ Production generation PASSED all completion checks"
echo "Completed at $(date)"
