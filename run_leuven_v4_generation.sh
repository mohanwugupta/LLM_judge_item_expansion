#!/bin/bash
#SBATCH --job-name=leuven_v4_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=4:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v4_gen_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v4_gen_%j.err

# V4 round-1 high-recall candidate discovery. This preserves the established
# Qwen2.5-72B vLLM job structure and changes only the configured prompt ensemble.

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen2.5-72B-Instruct}
MODEL_REVISION=${MODEL_REVISION:-Qwen2.5-72B-Instruct-cluster-snapshot}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
VLLM_PORT=${VLLM_PORT:-8022}
TENSOR_PARALLEL_SIZE=4
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.92

INPUT_CSV="$PROJECT_DIR/data/leuven_combined_features_consolidated.csv"
PROMPT_CONFIG="$PROJECT_DIR/configs/v4_discovery.json"
JOB_ID=leuven_v4_qwen2_5_72b
OUTPUT_DIR="$PROJECT_DIR/artifacts/leuven_feature_generation/v4/round1/$JOB_ID"
RESPONSES_PER_WORD=20
TEMPERATURE=0.8
BASE_SEED=20260801
MAX_WORKERS=64
MAX_TOKENS=500

echo "V4 generation job: $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"
echo "Output: $OUTPUT_DIR"

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
    "$PROJECT_DIR/logs"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_LEVEL=NVL
export CUDA_DEVICE_MAX_CONNECTIONS=1

for required in "$MODEL_PATH" "$INPUT_CSV" "$PROMPT_CONFIG"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: required path missing: $required"
        exit 1
    fi
done
if head -n 3 "$INPUT_CSV" | grep -q '^version https://git-lfs.github.com/spec/v1$'; then
    echo "ERROR: Leuven input is a Git LFS pointer"
    exit 1
fi

GENERATION_ARGS=(
    --input-csv "$INPUT_CSV"
    --job-id "$JOB_ID"
    --output-dir "$OUTPUT_DIR"
    --model "$SERVED_MODEL_NAME"
    --model-revision "$MODEL_REVISION"
    --model-source-path "$MODEL_PATH"
    --prompt-version v4
    --prompt-config "$PROMPT_CONFIG"
    --base-url "http://localhost:${VLLM_PORT}/v1"
    --responses-per-word "$RESPONSES_PER_WORD"
    --temperature "$TEMPERATURE"
    --base-seed "$BASE_SEED"
    --max-workers "$MAX_WORKERS"
    --max-tokens "$MAX_TOKENS"
    --resume
)

python -m leuven_expansion.generate_features "${GENERATION_ARGS[@]}" --preflight-only
mkdir -p "$OUTPUT_DIR"

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

python -m leuven_expansion.generate_features "${GENERATION_ARGS[@]}"

python3 - "$OUTPUT_DIR/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
families = set(data.get("prompt_variants", []))
counts = data.get("valid_responses_by_prompt", {})
expected = data["word_count"] * data["responses_per_word_per_prompt"]
ok = (
    data.get("protocol_version") == "leuven_free_generation_v4_configured_prompt_ensemble"
    and data.get("prompt_version") == "v4"
    and len(families) == 7
    and set(counts) == families
    and all(value == expected for value in counts.values())
    and data.get("model_revision")
    and data.get("model_source_path")
    and data.get("pending_after_run") == 0
    and data.get("finished_at")
)
if not ok:
    raise SystemExit("V4 generation manifest is incomplete")
print(f"V4 generation complete: {sum(counts.values())} responses across {len(families)} prompts")
PY

for name in feature_generations.csv generated_features_long.csv \
    generated_feature_frequencies.csv parse_errors.csv manifest.json run.log; do
    test -f "$OUTPUT_DIR/$name" || { echo "ERROR: missing $name"; exit 1; }
done
echo "V4 production generation passed at $(date)"
