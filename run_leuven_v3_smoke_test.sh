#!/bin/bash
#SBATCH --job-name=leuven_v3_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v3_smoke_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v3_smoke_%j.err

# Small end-to-end cluster check: three evenly spaced words, all three prompt
# conditions, and two simulated participant responses per condition (18 calls).
set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
export PROJECT_DIR
export SLURM_ARRAY_TASK_ID=1
export V3_RUN_LABEL=${V3_RUN_LABEL:-smoke_${SLURM_JOB_ID:-manual}}
export V3_MAX_WORDS=${V3_MAX_WORDS:-3}
export V3_RESPONSES_PER_WORD=${V3_RESPONSES_PER_WORD:-2}
export MAX_WORKERS=${MAX_WORKERS:-8}

exec bash "$PROJECT_DIR/run_leuven_full_labels.sh"
