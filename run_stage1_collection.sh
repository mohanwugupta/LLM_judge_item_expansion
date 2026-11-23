#!/bin/bash
#SBATCH --job-name=isc_stage1_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/isc_stage1_collect_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/isc_stage1_collect_%A_%a.err

# New namespace and original-domain inputs; historical scripts are untouched.
set -eo pipefail
PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
COLLECTION_STAGE=${COLLECTION_STAGE:-smoke}
OUTPUT_DIR=${OUTPUT_DIR:-artifacts/isc_stage1_collection_20261122_v1}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
SHARD_COUNT=${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}
VLLM_PORT=${VLLM_PORT:-$((8025 + SHARD_INDEX))}
MAX_WORKERS=${MAX_WORKERS:-16}

case "$COLLECTION_STAGE" in
    smoke|generate|judge|classify) ;;
    *) echo "ERROR: unsupported collection stage: $COLLECTION_STAGE"; exit 1 ;;
esac
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
    "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [ "$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")" -ne 4 ]; then
    echo "ERROR: Slurm must expose exactly four GPUs; do not override CUDA_VISIBLE_DEVICES"
    exit 1
fi
ARGS=(--output-dir "$OUTPUT_DIR" --task "$COLLECTION_STAGE" --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT")
echo "Collection: $COLLECTION_STAGE; shard $SHARD_INDEX/$SHARD_COUNT; output $OUTPUT_DIR"
# All input, scope and array checks precede vLLM startup. No training gates run.
python -m leuven_expansion.stage1_collection --stage preflight "${ARGS[@]}"
python -m leuven_expansion.stage1_collection --stage fingerprint --model-path "$MODEL_PATH" "${ARGS[@]}"
if [ "$COLLECTION_STAGE" != smoke ]; then
    python -m leuven_expansion.stage1_collection --stage preflight --require-smoke "${ARGS[@]}"
fi
if curl -fsS "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
    echo "ERROR: port $VLLM_PORT already serves an API; choose a different VLLM_PORT"
    exit 1
fi
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --port "$VLLM_PORT" --tensor-parallel-size 4 --dtype auto --trust-remote-code \
    --max-model-len 8192 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
    --enable-chunked-prefill --max-num-batched-tokens 16384 --disable-custom-all-reduce &
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
    echo "ERROR: vLLM readiness timeout"
    exit 1
fi
python -m leuven_expansion.stage1_collection --stage "$COLLECTION_STAGE" \
    --endpoint "http://localhost:$VLLM_PORT/v1" --workers "$MAX_WORKERS" "${ARGS[@]}"
