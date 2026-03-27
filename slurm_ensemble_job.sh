#!/bin/bash
#SBATCH --job-name=ensemble_bench
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Ensemble benchmark job - runs 3x Boltz predictions per sequence
# Usage: sbatch slurm_ensemble_job.sh <config_path>

set -eo pipefail

CONFIG="${1:?Usage: sbatch slurm_ensemble_job.sh <config_path>}"

cd /projects/u6bz/jude/ThinkingPLM

# Activate conda environment
set +u
source /lus/lfs1aip2/projects/u6bz/jude/miniforge3/etc/profile.d/conda.sh
conda activate profam_bagel
set -u

mkdir -p logs

echo "Running ensemble benchmark with config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"

python run_profam_bagel_pipeline.py --config "${CONFIG}"

echo "End time: $(date)"
