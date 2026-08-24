#!/bin/bash
#SBATCH --job-name=leuven_v4_atomic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --array=0-31%8
#SBATCH --time=72:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v4_atomic_%A_%a.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v4_atomic_%A_%a.err

# Exhaustive V4 candidate x Leuven judgment using the executed V2 panel.
#
# Resume estimate (2026-08-24): the 133,081-candidate bank creates
# 38,992,733 cells, not ~179,000 calls. The interrupted full-panel run
# wrote 2,508,409 resolution rows (2,508,408 valid). Remaining cells receive
# prompt C once; only
# positives, ambiguous/low-confidence responses, and parse failures receive
# A/B plus the frozen V2 resolver. The posthoc V4 pilot routed 1.95% of cells
# and reduced calls by 64.7%. At the observed shard throughput, a resumed
# shard is expected to need roughly 40-50 hours; 72 hours leaves room for
# denser later V3/V3.1-derived candidates and cluster variability.

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen2.5-72B-Instruct}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
VLLM_PORT=${VLLM_PORT:-8024}
SHARD_COUNT=${SHARD_COUNT:-32}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID}
MAX_WORKERS=${MAX_WORKERS:-64}
CASCADE_CONFIDENCE_THRESHOLD=${CASCADE_CONFIDENCE_THRESHOLD:-0.80}

CANDIDATE_BANK="$PROJECT_DIR/artifacts/v4/discovery/candidate_bank.csv"
LEUVEN_WORDS="$PROJECT_DIR/data/leuven_combined_features_consolidated.csv"
V2_MANIFEST="$PROJECT_DIR/artifacts/leuven_full_labels/leuven_full_v2/manifest.json"
OUTPUT_DIR="$PROJECT_DIR/artifacts/v4/judgments"

echo "V4 atomic shard $SHARD_INDEX/$SHARD_COUNT"
echo "Job: $SLURM_JOB_ID node: $SLURMD_NODENAME time: $(date)"

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

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 4 \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 4096 \
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
    --max-workers "$MAX_WORKERS" \
    --shard-count "$SHARD_COUNT" \
    --shard-index "$SHARD_INDEX" \
    --execution-mode prompt-c-cascade \
    --cascade-confidence-threshold "$CASCADE_CONFIDENCE_THRESHOLD" \
    --resume

python3 - "$OUTPUT_DIR/shards/$(printf '%04d' "$SHARD_INDEX")/v4_shard_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
if not manifest.get("complete") or manifest["resolved_cells"] != manifest["expected_cells"]:
    raise SystemExit("V4 atomic shard failed completeness checks")
print(f"V4 atomic shard complete: {manifest['resolved_cells']} cells")
PY
