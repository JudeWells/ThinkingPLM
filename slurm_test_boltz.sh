#!/bin/bash
#SBATCH --job-name=test_boltz
#SBATCH --output=logs/test_boltz_%j.out
#SBATCH --error=logs/test_boltz_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -eo pipefail

cd /projects/u6bz/jude/ThinkingPLM

# Activate conda environment
set +u
source /lus/lfs1aip2/projects/u6bz/jude/miniforge3/etc/profile.d/conda.sh
conda activate profam_bagel
set -u

mkdir -p logs

echo "Testing Boltz ensemble behavior"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"

python test_boltz_determinism.py

echo "End time: $(date)"
