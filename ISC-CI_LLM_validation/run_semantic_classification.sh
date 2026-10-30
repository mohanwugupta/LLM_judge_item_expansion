#!/bin/bash
#SBATCH --job-name=isc_task_classes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/isc_task_classes_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/isc_task_classes_%j.err

set -eo pipefail
PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen2.5-72B-Instruct}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
VLLM_PORT=${VLLM_PORT:-8024}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
SMOKE_TEST=${SMOKE_TEST:-1}
CLASSIFY_CONDITION=${CLASSIFY_CONDITION:-human}
INCLUDE_DEFERRED=${INCLUDE_DEFERRED:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/ISC-CI_LLM_validation/artifacts/semantic_retraining_20261028_32_corrected}

cd "$PROJECT_DIR"
module load anaconda3/2025.6
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
elif [ -f "$HOME/.conda/envs/$CONDA_ENV/bin/activate" ]; then
    source "$HOME/.conda/envs/$CONDA_ENV/bin/activate"
else
    source activate "$CONDA_ENV"
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export VLLM_USAGE_STATS_DIR="$VLLM_CACHE_DIR/usage_stats"
export TRITON_CACHE_DIR="$VLLM_CACHE_DIR/triton"
export XDG_CACHE_HOME="$VLLM_CACHE_DIR/xdg"
export TIKTOKEN_CACHE_DIR="$HOME/.tiktoken_cache"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$VLLM_CACHE_DIR" \
    "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$PROJECT_DIR/logs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: Slurm did not expose GPUs"
    exit 1
fi
VISIBLE_GPU_COUNT=$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")
if [ "$VISIBLE_GPU_COUNT" -ne "$TENSOR_PARALLEL_SIZE" ]; then
    echo "ERROR: expected $TENSOR_PARALLEL_SIZE visible GPUs, got $VISIBLE_GPU_COUNT"
    exit 1
fi

RUNNER=ISC-CI_LLM_validation/run_semantic_retraining.py
# Restore missing frozen inputs only; matching files are verified and conflicts stop the job.
python ISC-CI_LLM_validation/run_semantic_inputs.py --install
# Fail on missing matrices, upstream data, package/tests, or stale originals before loading Qwen.
python "$RUNNER" --stage prepare --output-dir "$OUTPUT_DIR"
python "$RUNNER" --stage preflight --output-dir "$OUTPUT_DIR"
# Hash all actual local weight shards, not an unverified model-name string.
python "$RUNNER" --stage fingerprint --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --disable-custom-all-reduce &
VLLM_PID=$!
cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
READY=0
for attempt in $(seq 1 180); do
    if curl -fsS "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited during startup"
        exit 1
    fi
    sleep 10
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM did not become ready"
    exit 1
fi

ARGS=(--output-dir "$OUTPUT_DIR" --endpoint "http://localhost:$VLLM_PORT/v1" --resume)
if [ "$INCLUDE_DEFERRED" = 1 ]; then
    ARGS+=(--include-deferred)
fi
if [ "$SMOKE_TEST" = 1 ]; then
    python "$RUNNER" --stage classify-smoke "${ARGS[@]}"
else
    if [ "$CLASSIFY_CONDITION" != all ]; then
        ARGS+=(--condition "$CLASSIFY_CONDITION")
    fi
    python "$RUNNER" --stage classify "${ARGS[@]}"
fi
