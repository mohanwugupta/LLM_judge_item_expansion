#!/bin/bash
#SBATCH --job-name=leuven_v3_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=mg9965@princeton.edu
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v3_gen_%j.out
#SBATCH --error=/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/logs/leuven_v3_gen_%j.err

# Dedicated v3 entry point. The shared launcher uses task 1 for free generation;
# setting it here prevents a default submission from also starting v2 labeling.
set -eo pipefail

PROJECT_DIR=${PROJECT_DIR:-/scratch/gpfs/JORDANAT/mg9965/FalseMemoryISC-CI/LLM_judge_item_expansion}
export PROJECT_DIR
export SLURM_ARRAY_TASK_ID=1

exec bash "$PROJECT_DIR/run_leuven_full_labels.sh"
