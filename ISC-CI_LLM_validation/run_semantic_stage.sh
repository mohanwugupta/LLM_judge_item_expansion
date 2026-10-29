#!/bin/bash
#SBATCH --job-name=isc_semantic_stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/isc_semantic_stage_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/isc_semantic_stage_%j.err

set -eo pipefail
PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
CONDA_ENV=${CONDA_ENV:-PromptControlText}
PIPELINE_STAGE=${PIPELINE_STAGE:-smoke}
PHASE=${PHASE:-1}
WORKERS=${WORKERS:-4}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/ISC-CI_LLM_validation/artifacts/semantic_retraining_20261028_32_corrected}
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-}}

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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$OUTPUT_DIR/runtime/matplotlib"
ARGS=(--stage "$PIPELINE_STAGE" --phase "$PHASE" --workers "$WORKERS" --output-dir "$OUTPUT_DIR" --resume)
for entry in "CONDITION:--condition" "SEMANTIC_SOURCE:--semantic-source" "CONTEXT_SOURCE:--context-source"; do
    variable=${entry%%:*}
    option=${entry#*:}
    if [ -n "${!variable:-}" ]; then
        ARGS+=("$option" "${!variable}")
    fi
done
if [ -n "$SEED" ]; then
    if ! [[ "$SEED" =~ ^[0-3]$ ]]; then
        echo "ERROR: SEED / array index must be 0..3, not '$SEED'"
        exit 1
    fi
    ARGS+=(--seed "$SEED")
fi
if [ -n "${CONFIG_PATH:-}" ]; then
    ARGS+=(--config "$CONFIG_PATH")
fi
# One requested stage only. This launcher is for a CPU training configuration.
python ISC-CI_LLM_validation/run_semantic_retraining.py "${ARGS[@]}"
