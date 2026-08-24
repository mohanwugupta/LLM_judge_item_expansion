#!/bin/bash
#SBATCH --job-name=v4_atomic_smoke
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
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/v4_atomic_smoke_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/v4_atomic_smoke_%j.err

# Small V4 atomic shard using the locked 175-context bank.

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen2.5-72B-Instruct}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
VLLM_PORT=${VLLM_PORT:-8025}
SHARD_COUNT=4096
SHARD_INDEX=0

CANDIDATE_BANK="$PROJECT_DIR/artifacts/v4/discovery/candidate_bank_v3_1_b_175.csv"
LEUVEN_WORDS="$PROJECT_DIR/data/leuven_combined_features_consolidated.csv"
V2_MANIFEST="$PROJECT_DIR/artifacts/leuven_full_labels/leuven_full_v2/manifest.json"
OUTPUT_DIR="$PROJECT_DIR/artifacts/v4/judgments_smoke/${SLURM_JOB_ID:-local}"

echo "V4 atomic smoke job: $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"
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
    "$VLLM_USAGE_STATS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
    "$PROJECT_DIR/logs" "$OUTPUT_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

for required in "$MODEL_PATH" "$CANDIDATE_BANK" "$LEUVEN_WORDS" "$V2_MANIFEST"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: required path missing: $required"
        exit 1
    fi
done

python run_v4_judgments.py \
    --candidate-bank "$CANDIDATE_BANK" \
    --leuven-words "$LEUVEN_WORDS" \
    --v2-manifest "$V2_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --model "$SERVED_MODEL_NAME" \
    --shard-count "$SHARD_COUNT" \
    --execution-mode prompt-c-cascade \
    --cascade-confidence-threshold 0.80 \
    --dry-run

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 4 \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --disable-custom-all-reduce &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

elapsed=0
while [ "$elapsed" -lt 600 ]; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming ready"
        exit 1
    fi
    if curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 15
    elapsed=$((elapsed + 15))
done
if [ "$elapsed" -ge 600 ]; then
    echo "ERROR: vLLM did not become ready"
    exit 1
fi

python run_v4_judgments.py \
    --candidate-bank "$CANDIDATE_BANK" \
    --leuven-words "$LEUVEN_WORDS" \
    --v2-manifest "$V2_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --model "$SERVED_MODEL_NAME" \
    --base-url "http://localhost:${VLLM_PORT}/v1" \
    --max-workers 32 \
    --shard-count "$SHARD_COUNT" \
    --shard-index "$SHARD_INDEX" \
    --execution-mode prompt-c-cascade \
    --cascade-confidence-threshold 0.80 \
    --resume

MANIFEST="$OUTPUT_DIR/shards/0000/v4_shard_manifest.json"
python3 - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if not manifest.get("complete"):
    raise SystemExit("V4 smoke shard is incomplete")
if manifest["resolved_cells"] != manifest["expected_cells"]:
    raise SystemExit("V4 smoke shard cell count is wrong")
if manifest["resolved_cells"] < 1:
    raise SystemExit("V4 smoke shard selected no cells")
if manifest["full_panel_vote_cells"] + manifest["prompt_c_only_cells"] != manifest["resolved_cells"]:
    raise SystemExit("V4 smoke shard has invalid cascade vote coverage")
print(f"V4 atomic smoke passed: {manifest['resolved_cells']} cells")
PY
