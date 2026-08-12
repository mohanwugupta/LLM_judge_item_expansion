#!/bin/bash
#SBATCH --job-name=leuven_v4_smoke
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
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v4_smoke_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v4_smoke_%j.err

# Faithful small version of run_leuven_v4_generation.sh: 3 words x 7 prompts
# x 2 responses = 42 model calls.

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
export PROJECT_DIR
export MODEL_PATH=${MODEL_PATH:-/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen2.5-72B-Instruct}
export MODEL_REVISION=${MODEL_REVISION:-Qwen2.5-72B-Instruct-cluster-snapshot}
export VLLM_PORT=${VLLM_PORT:-8012}

export V4_SMOKE_MODE=1
export V4_JOB_ID=leuven_v4_smoke
export V4_OUTPUT_DIR="$PROJECT_DIR/artifacts/leuven_feature_generation/v4/smoke_test/$V4_JOB_ID"

# Keep server and generation settings synchronized with production by invoking
# the same script after overriding only smoke-specific arguments below.
cd "$PROJECT_DIR"

module load anaconda3/2025.6
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate PromptControlText
else
    source activate PromptControlText
fi
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/gpfs/JORDANAT/mg9965/hf_cache
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_CACHE_DIR=/scratch/gpfs/JORDANAT/mg9965/vLLM-cache
export TRITON_CACHE_DIR="$VLLM_CACHE_DIR/triton"
export XDG_CACHE_HOME="$VLLM_CACHE_DIR/xdg"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=true
mkdir -p "$PROJECT_DIR/logs" "$V4_OUTPUT_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

INPUT_CSV="$PROJECT_DIR/data/leuven_combined_features_consolidated.csv"
PROMPT_CONFIG="$PROJECT_DIR/configs/v4_discovery.json"

python -m leuven_expansion.generate_features \
    --input-csv "$INPUT_CSV" --job-id "$V4_JOB_ID" --output-dir "$V4_OUTPUT_DIR" \
    --model Qwen2.5-72B-Instruct --model-revision "$MODEL_REVISION" \
    --model-source-path "$MODEL_PATH" --prompt-version v4 --prompt-config "$PROMPT_CONFIG" \
    --responses-per-word 2 --temperature 0.8 --base-seed 20260801 \
    --max-workers 32 --max-tokens 500 --max-words 3 --resume --preflight-only

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --port "$VLLM_PORT" --tensor-parallel-size 4 --dtype auto \
    --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --disable-custom-all-reduce &
VLLM_PID=$!
cleanup() { kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

elapsed=0
while [ "$elapsed" -lt 600 ]; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM exited"; exit 1; }
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && break
    sleep 15
    elapsed=$((elapsed + 15))
done
test "$elapsed" -lt 600 || { echo "ERROR: vLLM readiness timeout"; exit 1; }

python -m leuven_expansion.generate_features \
    --input-csv "$INPUT_CSV" --job-id "$V4_JOB_ID" --output-dir "$V4_OUTPUT_DIR" \
    --model Qwen2.5-72B-Instruct --model-revision "$MODEL_REVISION" \
    --model-source-path "$MODEL_PATH" --prompt-version v4 --prompt-config "$PROMPT_CONFIG" \
    --base-url "http://localhost:${VLLM_PORT}/v1" --responses-per-word 2 \
    --temperature 0.8 --base-seed 20260801 --max-workers 32 --max-tokens 500 \
    --max-words 3 --resume

python3 - "$V4_OUTPUT_DIR/manifest.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
counts = data.get("valid_responses_by_prompt", {})
assert len(counts) == 7 and all(value == 6 for value in counts.values())
assert data.get("model_revision") and data.get("model_source_path")
assert data.get("pending_after_run") == 0 and data.get("finished_at")
print("V4 smoke test passed: 42 valid responses")
PY
