#!/bin/bash
#SBATCH --job-name=leuven_expansion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --array=0-2
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_expansion_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_expansion_%A_%a.err

# =============================================================================
# Leuven Feature Expansion — ISC-CI item coverage (PRD)
#
# Runs as a 3-task SLURM array (--array=0-2):
#   0 = cell-level holdout validation  (validate_features --mode cell_holdout)
#   1 = word-level holdout validation  (validate_features --mode word_holdout)
#   2 = production DRM expansion       (expand_feature_matrix)
#
# Run tasks 0 and 1 first to validate the pipeline before running task 2.
#
# Model: Qwen2.5-72B-Instruct (requires TP=2 on 80G GPUs)
# Approximate throughput: ~100 req/s
#
# Submit all tasks:          sbatch run_leuven_expansion.sh
# Validate only:             sbatch --array=0-1 run_leuven_expansion.sh
# Production expansion only: sbatch --array=2   run_leuven_expansion.sh
# Resume a timed-out task:   sbatch --array=<N> run_leuven_expansion.sh  (--resume is always on)
# =============================================================================

set -eo pipefail

# ------------------------------------------------------------------
# 0. Per-task job registry  (array index → task config)
# ------------------------------------------------------------------
TASK=${SLURM_ARRAY_TASK_ID}

# Task 0: cell-level holdout validation
# Task 1: word-level holdout validation
# Task 2: production DRM feature expansion

case "$TASK" in
  0)
    PIPELINE="validate"
    VALIDATE_MODE="cell_holdout"
    JOB_ID="leuven_atomic_cell_validation_qwen"
    OUTPUT_DIR="artifacts/leuven_feature_expansion/leuven_atomic_cell_validation_qwen"
    ;;
  1)
    PIPELINE="validate"
    VALIDATE_MODE="word_holdout"
    JOB_ID="leuven_atomic_word_validation_qwen"
    OUTPUT_DIR="artifacts/leuven_feature_expansion/leuven_atomic_word_validation_qwen"
    ;;
  2)
    PIPELINE="expand"
    JOB_ID="drm_atomic_leuven_feature_expansion_qwen"
    OUTPUT_DIR="artifacts/leuven_feature_expansion/drm_atomic_leuven_feature_expansion_qwen"
    ;;
  *)
    echo "❌ ERROR: Unknown array task ID: $TASK (expected 0-2)"
    exit 1
    ;;
esac

# Each array task gets its own port to allow co-scheduling on the same node
VLLM_PORT=$((8000 + TASK))

echo "=========================================="
echo " Leuven Feature Expansion — task ${TASK} / ${JOB_ID}"
echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $TASK"
echo "Node:        $SLURMD_NODENAME"
echo "Time:        $(date)"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Pipeline:    $PIPELINE"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
PROJECT_DIR=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion
MODEL_DIR_NAME=Qwen--Qwen2.5-72B-Instruct
MODEL_PATH=/scratch/gpfs/JORDANAT/mg9965/models/$MODEL_DIR_NAME
SERVED_MODEL_NAME=Qwen2.5-72B-Instruct
CONDA_ENV=PromptControlText
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.92

# Data paths (relative to PROJECT_DIR)
LEUVEN_FEATURES=data/leuven_combined_features_consolidated.csv
LEUVEN_CATEGORIES=data/leuven_categories.csv
SINGULAR_PLURAL=data/leuven_singular_to_plural.csv
DRM_ITEMS=data/new_items/drm_items_to_classify.csv
DRM_WORD_OCCURRENCES=data/new_items/drm_word_occurrences_long.csv

# Judge call settings
MAX_WORKERS=64
TEMPERATURE=0.0
MAX_TOKENS=200
TEST_SIZE=0.20
SEED=42

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
# 4. GPU / Memory optimizations
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

if [ "$PIPELINE" = "expand" ] && [ ! -f "$PROJECT_DIR/$DRM_ITEMS" ]; then
    echo "❌ ERROR: DRM items file not found: $PROJECT_DIR/$DRM_ITEMS"
    exit 1
fi

if [ "$PIPELINE" = "expand" ] && [ ! -f "$PROJECT_DIR/$DRM_WORD_OCCURRENCES" ]; then
    echo "❌ ERROR: DRM word occurrences file not found: $PROJECT_DIR/$DRM_WORD_OCCURRENCES"
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
# 8. Run pipeline
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Task ${TASK}: ${JOB_ID}"
echo " Pipeline:   ${PIPELINE}"
echo " Output:     ${OUTPUT_DIR}"
echo " Model:      ${SERVED_MODEL_NAME}"
echo "=========================================="

if [ "$PIPELINE" = "validate" ]; then

    python -m leuven_expansion.validate_features \
        --mode              "$VALIDATE_MODE" \
        --leuven-features   "$PROJECT_DIR/$LEUVEN_FEATURES" \
        --leuven-categories "$PROJECT_DIR/$LEUVEN_CATEGORIES" \
        --job-id            "$JOB_ID" \
        --output-dir        "$PROJECT_DIR/$OUTPUT_DIR" \
        --model             "$SERVED_MODEL_NAME" \
        --base-url          "http://localhost:${VLLM_PORT}/v1" \
        --test-size         "$TEST_SIZE" \
        --seed              "$SEED" \
        --max-workers       "$MAX_WORKERS" \
        --resume

elif [ "$PIPELINE" = "expand" ]; then

    python -m leuven_expansion.expand_feature_matrix \
        --items             "$PROJECT_DIR/$DRM_ITEMS" \
        --leuven-features   "$PROJECT_DIR/$LEUVEN_FEATURES" \
        --singular-plural   "$PROJECT_DIR/$SINGULAR_PLURAL" \
        --word-occurrences  "$PROJECT_DIR/$DRM_WORD_OCCURRENCES" \
        --job-id            "$JOB_ID" \
        --output-dir        "$PROJECT_DIR/$OUTPUT_DIR" \
        --model             "$SERVED_MODEL_NAME" \
        --base-url          "http://localhost:${VLLM_PORT}/v1" \
        --max-workers       "$MAX_WORKERS" \
        --resume

fi

echo ""
echo "✅ Task ${TASK} (${JOB_ID}) completed at $(date)"
