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
SHARD_COUNT=32
MAX_UNRESOLVED_CELLS=${MAX_UNRESOLVED_CELLS:-127}
RECOVERY_DIR=${RECOVERY_DIR:-ISC-CI_LLM_validation/reports/v4_exact_id_recovery_20261001}
RECOVERY_INPUTS_BUNDLE=${RECOVERY_INPUTS_BUNDLE:-configs/v4_exact_id_recovery_inputs.tar.gz}

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

# Apply saved-response formatting repairs in this CPU job; no GPU retry is needed.
if [ ! -f "$RECOVERY_DIR/recovery_plan.json" ] && [ -f "$RECOVERY_INPUTS_BUNDLE" ]; then
    mkdir -p "$RECOVERY_DIR"
    tar -xzf "$RECOVERY_INPUTS_BUNDLE" -C "$RECOVERY_DIR"
fi
if [ -f "$RECOVERY_DIR/recovery_plan.json" ]; then
    python -u ISC-CI_LLM_validation/recover_v4_exact_ids.py \
        --judgments-dir artifacts/v4/judgments \
        --output-dir "$RECOVERY_DIR" --apply
fi

python run_v4_judgments.py \
    --candidate-bank artifacts/v4/discovery/candidate_bank.csv \
    --leuven-words data/leuven_combined_features_consolidated.csv \
    --v2-manifest artifacts/leuven_full_labels/leuven_full_v2/manifest.json \
    --output-dir artifacts/v4/judgments \
    --model Qwen2.5-72B-Instruct \
    --shard-count "$SHARD_COUNT" \
    --execution-mode prompt-c-cascade \
    --cascade-confidence-threshold 0.80 \
    --max-unresolved-cells "$MAX_UNRESOLVED_CELLS" \
    --finalize
