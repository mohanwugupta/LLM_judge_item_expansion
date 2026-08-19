#!/bin/bash
#SBATCH --job-name=build_v4_bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/build_v4_bank_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/build_v4_bank_%j.err

# CPU-only job: builds the V4 candidate bank (phrase clustering + MiniLM
# embeddings). The embedding call in iscci_validation/consolidation.py is
# hardcoded to device="cpu", so this does not need a GPU allocation --
# it just needs dedicated CPU threads instead of a contended login node.

set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
CONDA_ENV=${CONDA_ENV:-PromptControlText}

echo "build_v4_candidate_bank job: $SLURM_JOB_ID on $SLURMD_NODENAME at $(date)"
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=true
mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/artifacts/v4/discovery"

CANDIDATE_BANK="$PROJECT_DIR/artifacts/v4/discovery/candidate_bank_v3_1_b_175.csv"

python build_v4_candidate_bank.py \
    --config configs/v4_discovery.json \
    --manual-review configs/v4_candidate_merge_review.csv \
    --output-dir artifacts/v4/discovery \
    --auto-approve-merges

if [ ! -e "$CANDIDATE_BANK" ]; then
    echo "ERROR: candidate bank was not produced at $CANDIDATE_BANK"
    exit 1
fi
echo "build_v4_candidate_bank passed at $(date): $CANDIDATE_BANK"
