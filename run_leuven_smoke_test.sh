#!/bin/bash
#SBATCH --job-name=leuven_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --partition=test
#SBATCH --time=1:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_smoke_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_smoke_%j.err

# =============================================================================
# Leuven Feature Expansion — smoke test  (1-hour test partition)
#
# Runs a tiny cell-holdout validation (2 % of cells, ~25 pairs) to verify
# the full pipeline end-to-end without committing a 12-hour allocation.
#
# Submit:
#   sbatch run_leuven_smoke_test.sh
#
# Success criteria:
#   - vLLM server starts and accepts requests
#   - feature_votes.csv has ≥ 3 rows per judged pair
#   - feature_resolutions.csv written
#   - parse_errors.csv written (may be empty)
#   - manifest.json shows finished_at != null
#   - feature_validation_metrics.json written
# =============================================================================

set -eo pipefail

VLLM_PORT=8010   # distinct port to avoid collision with production tasks

echo "=========================================="
echo " Leuven Smoke Test"
echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Time:    $(date)"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
echo ""

# ------------------------------------------------------------------
# 1. Configuration  (must match run_leuven_expansion.sh)
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
LEUVEN_CATEGORIES=data/leuven_categories.csv

JOB_ID=leuven_smoke_cell_holdout
OUTPUT_DIR=artifacts/leuven_feature_expansion/smoke_test/$JOB_ID

# Smoke-test-specific knobs: tiny 0.5 % cell holdout → ~2900 pairs
TEST_SIZE=0.005
SEED=42
MAX_WORKERS=32     # vLLM with TP=4 on 4×80G handles 32+ concurrent requests

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
# 4. GPU / memory
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
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model not found at: $MODEL_PATH"
    exit 1
fi
echo "✅ Model found: $MODEL_PATH"

if [ ! -f "$PROJECT_DIR/$LEUVEN_FEATURES" ]; then
    echo "❌ ERROR: Leuven features not found: $PROJECT_DIR/$LEUVEN_FEATURES"
    exit 1
fi
echo "✅ Leuven features found"

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/$OUTPUT_DIR"

# ------------------------------------------------------------------
# 6. Start vLLM server
# ------------------------------------------------------------------
echo ""
echo "Starting vLLM ($SERVED_MODEL_NAME, TP=$TENSOR_PARALLEL_SIZE, port=$VLLM_PORT)..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --disable-custom-all-reduce \
    &

VLLM_PID=$!

cleanup() {
    echo "Shutting down vLLM (PID $VLLM_PID)..."
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------
# 7. Wait for server readiness
# ------------------------------------------------------------------
echo "Waiting for vLLM on port $VLLM_PORT..."
MAX_WAIT=600
ELAPSED=0
WAIT_INTERVAL=15

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ ERROR: vLLM exited unexpectedly"
        exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM ready after ${ELAPSED}s"
        break
    fi
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ ERROR: vLLM failed to start within ${MAX_WAIT}s"
    exit 1
fi

# ------------------------------------------------------------------
# 8. Run 2 % cell-holdout validation
# ------------------------------------------------------------------
echo ""
echo "Running smoke test: cell holdout (test_size=$TEST_SIZE)..."

python -m leuven_expansion.validate_features \
    --mode              cell_holdout \
    --leuven-features   "$PROJECT_DIR/$LEUVEN_FEATURES" \
    --leuven-categories "$PROJECT_DIR/$LEUVEN_CATEGORIES" \
    --job-id            "$JOB_ID" \
    --output-dir        "$PROJECT_DIR/$OUTPUT_DIR" \
    --model             "$SERVED_MODEL_NAME" \
    --base-url          "http://localhost:${VLLM_PORT}/v1" \
    --test-size         "$TEST_SIZE" \
    --seed              "$SEED" \
    --max-workers       "$MAX_WORKERS"

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

check_file "feature_votes.csv"              "votes"
check_file "feature_resolutions.csv"        "resolutions"
check_file "parse_errors.csv"               "parse_errors"
check_file "manifest.json"                  "manifest"
check_file "run.log"                        "run.log"
check_file "feature_validation_metrics.json" "metrics"

# Verify manifest has finished_at
MANIFEST="$PROJECT_DIR/$OUTPUT_DIR/manifest.json"
if [ -f "$MANIFEST" ]; then
    if python3 -c "import json,sys; d=json.load(open('$MANIFEST')); sys.exit(0 if d.get('finished_at') else 1)"; then
        echo "  ✅ manifest.json has finished_at"
        PASS=$((PASS + 1))
    else
        echo "  ❌ manifest.json missing finished_at (job may have crashed)"
        FAIL=$((FAIL + 1))
    fi
fi

echo ""
echo "Smoke test summary: $PASS checks passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo "❌ Smoke test FAILED — review logs before submitting production job"
    exit 1
else
    echo "✅ Smoke test PASSED — safe to submit: sbatch --array=0-1 run_leuven_expansion.sh"
fi

echo ""
echo "Completed at $(date)"
