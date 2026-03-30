#!/bin/bash
#SBATCH --job-name=2gdz_campaign
#SBATCH --output=logs/2gdz_%j.out
#SBATCH --error=logs/2gdz_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# 2GDZ Campaign job - max 24 hour runtime
# Usage: sbatch slurm_2gdz_campaign.sh <config_path>

set -eo pipefail

CONFIG="${1:?Usage: sbatch slurm_2gdz_campaign.sh <config_path>}"

cd /projects/u6bz/jude/ThinkingPLM

# Activate conda environment
set +u
source /lus/lfs1aip2/projects/u6bz/jude/miniforge3/etc/profile.d/conda.sh
conda activate profam_bagel
set -u

mkdir -p logs

echo "Running 2GDZ campaign with config: ${CONFIG}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"

python run_profam_bagel_pipeline.py --config "${CONFIG}"

echo "End time: $(date)"
