#!/bin/bash
#SBATCH --job-name=greedy_diverse
#SBATCH --output=logs/greedy_diverse_%j.out
#SBATCH --error=logs/greedy_diverse_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=u6bz

# Usage: sbatch slurm_greedy_diverse_job.sh <config_path>
# Example: sbatch slurm_greedy_diverse_job.sh configs/pipelines/bench_2GDZ_15PGDH_3helix_greedy_diverse.yaml

set -euo pipefail

CONFIG="${1:-}"
if [[ -z "${CONFIG}" ]]; then
    echo "Error: No config file specified"
    echo "Usage: sbatch slurm_greedy_diverse_job.sh <config_path>"
    exit 1
fi

echo "============================================"
echo "Job started on host $(hostname) at $(date)"
echo "Config: ${CONFIG}"
echo "============================================"

# Navigate to project directory
cd /projects/u6bz/jude/ThinkingPLM

# Activate conda environment
source /home/judewells/miniconda3/etc/profile.d/conda.sh
conda activate profam_bagel

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the pipeline
python run_profam_bagel_pipeline.py --config "${CONFIG}"

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"
