#!/bin/bash
#SBATCH --job-name=leuven_label
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --array=0-1
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_label_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_label_%A_%a.err

# =============================================================================
# Leuven v2 labeling and v3 feature generation
#
# V2 labels every existing word x feature cell. V3 recreates the original
# human free-generation task: each simulated participant sees one word and
# generates preferably 10 features without seeing the Leuven feature schema.
#
# SLURM array mapping
# -------------------
#   task 0 : v2 prompts (spontaneous-production framing)
#             → artifacts/leuven_full_labels/leuven_full_v2/
#   task 1 : v3 free generation with prompt conditions A, B, and C
#            (20 responses per word per prompt by default)
#             -> artifacts/leuven_feature_generation/leuven_v3_<run label>/
#
# Each task gets its own node allocation and can run independently.
#
# Submit both tasks:            sbatch run_leuven_full_labels.sh
# Submit a single task:         sbatch --array=0 run_leuven_full_labels.sh
# Resume a timed-out task:      sbatch --array=<N> run_leuven_full_labels.sh
#   (--resume is always passed; completed pairs or responses are skipped)
# =============================================================================

set -eo pipefail

TASK=${SLURM_ARRAY_TASK_ID}
V3_RUN_LABEL=${V3_RUN_LABEL:-qwen2_5_72b}
V3_RESPONSES_PER_WORD=${V3_RESPONSES_PER_WORD:-20}
V3_TEMPERATURE=${V3_TEMPERATURE:-0.8}
V3_BASE_SEED=${V3_BASE_SEED:-20260801}
V3_MAX_WORDS=${V3_MAX_WORDS:-0}

# ------------------------------------------------------------------
# Per-task configuration
# ------------------------------------------------------------------
case "$TASK" in
  0)
    PROMPT_VERSION="v2"
    JOB_ID="leuven_full_v2"
    OUTPUT_DIR="artifacts/leuven_full_labels/leuven_full_v2"
    EXPERIMENT_MODE="atomic_labels"
    ENABLE_VERIFICATION=false
    ;;
  1)
    PROMPT_VERSION="v3"
    JOB_ID="leuven_v3_${V3_RUN_LABEL}"
    OUTPUT_DIR="artifacts/leuven_feature_generation/$JOB_ID"
    EXPERIMENT_MODE="free_generation"
    ENABLE_VERIFICATION=false
    ;;
  *)
    echo "❌ ERROR: Unknown array task ID: $TASK (expected 0 or 1)"
    exit 1
    ;;
esac

# Each task gets its own vLLM port to allow co-scheduling on the same node
VLLM_PORT=$((8020 + TASK))

echo "============================================================"
echo " Leuven experiment — task ${TASK} / ${JOB_ID}"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array task:   $TASK"
echo "Node:         $SLURMD_NODENAME"
echo "Time:         $(date)"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Prompt ver:   $PROMPT_VERSION"
echo "Mode:         $EXPERIMENT_MODE"
echo "Verification: $ENABLE_VERIFICATION"
echo "Output dir:   $OUTPUT_DIR"
if [ "$EXPERIMENT_MODE" = "free_generation" ]; then
    echo "Prompts:      A, B, C"
    echo "Responses:    $V3_RESPONSES_PER_WORD per word per prompt"
    echo "Temperature:  $V3_TEMPERATURE"
    echo "Base seed:    $V3_BASE_SEED"
    if [ "$V3_MAX_WORDS" -gt 0 ]; then
        echo "Word limit:   $V3_MAX_WORDS (evenly spaced smoke subset)"
    else
        echo "Word limit:   all words"
    fi
fi
echo ""

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
MODEL_DIR_NAME=${MODEL_DIR_NAME:-Qwen--Qwen2.5-72B-Instruct}
MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/$MODEL_DIR_NAME}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen2.5-72B-Instruct}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}

# Data paths (relative to PROJECT_DIR)
LEUVEN_FEATURES=data/leuven_combined_features_consolidated.csv
LEUVEN_CATEGORIES=data/leuven_categories.csv

# Runtime settings
MAX_WORKERS=${MAX_WORKERS:-64}
MAX_TOKENS=${MAX_TOKENS:-500}

# Positive verification thresholds (only used when ENABLE_VERIFICATION=true)
POSITIVE_THRESHOLD=1.0
VERIFICATION_THRESHOLD=1.0

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
# 4. GPU / memory optimizations
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
echo "✅ Leuven features found: $LEUVEN_FEATURES"

if head -n 1 "$PROJECT_DIR/$LEUVEN_FEATURES" | grep -q "git-lfs.github.com/spec"; then
    echo "❌ ERROR: Leuven input is a Git LFS pointer. Run git lfs pull before submission."
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/$OUTPUT_DIR"

if [ "$EXPERIMENT_MODE" = "free_generation" ]; then
    V3_WORD_ARGS=()
    if [[ "$V3_MAX_WORDS" =~ ^[1-9][0-9]*$ ]]; then
        V3_WORD_ARGS+=(--max-words "$V3_MAX_WORDS")
    elif [ "$V3_MAX_WORDS" != "0" ]; then
        echo "❌ ERROR: V3_MAX_WORDS must be 0 or a positive integer"
        exit 1
    fi
    echo "Running no-model v3 preflight..."
    python -m leuven_expansion.generate_features \
        --input-csv          "$PROJECT_DIR/$LEUVEN_FEATURES" \
        --job-id             "$JOB_ID" \
        --output-dir         "$PROJECT_DIR/$OUTPUT_DIR" \
        --model              "$SERVED_MODEL_NAME" \
        --responses-per-word "$V3_RESPONSES_PER_WORD" \
        --temperature        "$V3_TEMPERATURE" \
        --base-seed          "$V3_BASE_SEED" \
        --max-tokens         "$MAX_TOKENS" \
        "${V3_WORD_ARGS[@]}" \
        --resume \
        --preflight-only
fi

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
echo "vLLM PID: $VLLM_PID"

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
WAIT_INTERVAL=15
ELAPSED=0

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
# 8. Count planned trials (informational)
# ------------------------------------------------------------------
echo ""
# Pass env vars needed by the heredoc below
export PROJECT_DIR_PY="$PROJECT_DIR"
export LEUVEN_FEATURES_PY="$LEUVEN_FEATURES"
export EXPERIMENT_MODE_PY="$EXPERIMENT_MODE"
export V3_RESPONSES_PER_WORD_PY="$V3_RESPONSES_PER_WORD"
export V3_MAX_WORDS_PY="$V3_MAX_WORDS"

python3 - <<'PYEOF'
import pandas as pd, pathlib, os
feat_csv = os.path.join(os.environ["PROJECT_DIR_PY"],
                        os.environ["LEUVEN_FEATURES_PY"])
df = pd.read_csv(feat_csv)
n_words = len(df)
n_feats = len(df.columns) - 1   # exclude word column
if os.environ["EXPERIMENT_MODE_PY"] == "free_generation":
    n_responses = int(os.environ["V3_RESPONSES_PER_WORD_PY"])
    max_words = int(os.environ["V3_MAX_WORDS_PY"])
    selected_words = min(n_words, max_words) if max_words else n_words
    n_prompts = 3
    calls = selected_words * n_prompts * n_responses
    print(f"  V3 generation: {selected_words} words x {n_prompts} prompts x {n_responses} responses = {calls:,} calls")
    print("  Existing feature columns are not sent to the model.")
else:
    print(f"  V2 labeling: {n_words} words x {n_feats} features = {n_words * n_feats:,} pairs")
PYEOF

# ------------------------------------------------------------------
# 9. Build python command
# ------------------------------------------------------------------
if [ "$EXPERIMENT_MODE" = "free_generation" ]; then
    PYTHON_CMD=(
        python -m leuven_expansion.generate_features
        --input-csv          "$PROJECT_DIR/$LEUVEN_FEATURES"
        --job-id             "$JOB_ID"
        --output-dir         "$PROJECT_DIR/$OUTPUT_DIR"
        --model              "$SERVED_MODEL_NAME"
        --base-url           "http://localhost:${VLLM_PORT}/v1"
        --responses-per-word "$V3_RESPONSES_PER_WORD"
        --temperature        "$V3_TEMPERATURE"
        --base-seed          "$V3_BASE_SEED"
        --max-workers        "$MAX_WORKERS"
        --max-tokens         "$MAX_TOKENS"
        --resume
    )
    PYTHON_CMD+=("${V3_WORD_ARGS[@]}")
else
    PYTHON_CMD=(
        python -m leuven_expansion.validate_features
        --mode              full_leuven
        --leuven-features   "$PROJECT_DIR/$LEUVEN_FEATURES"
        --leuven-categories "$PROJECT_DIR/$LEUVEN_CATEGORIES"
        --job-id            "$JOB_ID"
        --output-dir        "$PROJECT_DIR/$OUTPUT_DIR"
        --model             "$SERVED_MODEL_NAME"
        --base-url          "http://localhost:${VLLM_PORT}/v1"
        --max-workers       "$MAX_WORKERS"
        --prompt-version    "$PROMPT_VERSION"
        --resume
    )
fi

if [ "$ENABLE_VERIFICATION" = "true" ]; then
    PYTHON_CMD+=(
        --enable-positive-verification
        --positive-threshold      "$POSITIVE_THRESHOLD"
        --verification-threshold  "$VERIFICATION_THRESHOLD"
    )
fi

echo ""
echo "============================================================"
echo " Task ${TASK}: ${JOB_ID}"
echo " Prompt version: ${PROMPT_VERSION}"
echo " Verification:   ${ENABLE_VERIFICATION}"
echo " Command: ${PYTHON_CMD[*]}"
echo "============================================================"
echo ""

"${PYTHON_CMD[@]}"

# ------------------------------------------------------------------
# 10. Post-run checks
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
        echo "  ✅ $desc ($rows lines): $f"
        PASS=$((PASS + 1))
    else
        echo "  ❌ MISSING $desc: $f"
        FAIL=$((FAIL + 1))
    fi
}

if [ "$EXPERIMENT_MODE" = "free_generation" ]; then
    check_file "feature_generations.csv"          "participant responses"
    check_file "generated_features_long.csv"      "long-form generated features"
    check_file "generated_feature_frequencies.csv" "exact-string frequencies"
    check_file "parse_errors.csv"                 "parse errors"
else
    check_file "feature_votes.csv"       "votes"
    check_file "feature_resolutions.csv" "resolutions"
    check_file "parse_errors.csv"        "parse errors"
fi
check_file "manifest.json"           "manifest"
check_file "run.log"                 "run.log"

if [ "$ENABLE_VERIFICATION" = "true" ]; then
    check_file "positive_verification_votes.csv" "verifier_votes"
fi

# Verify manifest records a completed run
MANIFEST="$PROJECT_DIR/$OUTPUT_DIR/manifest.json"
if [ -f "$MANIFEST" ]; then
    if [ "$EXPERIMENT_MODE" = "free_generation" ]; then
        MANIFEST_CHECK="d.get('finished_at') and d.get('pending_after_run') == 0 and set(d.get('valid_responses_by_prompt', {})) == {'A','B','C'} and all(v == d['word_count'] * d['responses_per_word_per_prompt'] for v in d['valid_responses_by_prompt'].values())"
    else
        MANIFEST_CHECK="d.get('finished_at')"
    fi
    if python3 -c "import json,sys; d=json.load(open('$MANIFEST')); sys.exit(0 if $MANIFEST_CHECK else 1)"; then
        echo "  ✅ manifest.json records a complete run"
        PASS=$((PASS + 1))
    else
        echo "  ❌ manifest.json is incomplete or has pending responses"
        FAIL=$((FAIL + 1))
    fi
fi

echo ""
echo "Task ${TASK} (${JOB_ID}) summary: $PASS checks passed, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo "❌ Post-run checks FAILED — inspect logs:"
    echo "   $PROJECT_DIR/$OUTPUT_DIR/run.log"
    exit 1
else
    echo "✅ All checks passed"
fi

echo ""
echo "Completed at $(date)"
