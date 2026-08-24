#!/bin/bash
#SBATCH --job-name=v4_atomic_finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/v4_atomic_finalize_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/v4_atomic_finalize_%j.err

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
SHARD_COUNT=${SHARD_COUNT:-32}

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

python run_v4_judgments.py \
    --candidate-bank artifacts/v4/discovery/candidate_bank.csv \
    --leuven-words data/leuven_combined_features_consolidated.csv \
    --v2-manifest artifacts/leuven_full_labels/leuven_full_v2/manifest.json \
    --output-dir artifacts/v4/judgments \
    --model Qwen2.5-72B-Instruct \
    --shard-count "$SHARD_COUNT" \
    --execution-mode prompt-c-cascade \
    --cascade-confidence-threshold 0.80 \
    --finalize
