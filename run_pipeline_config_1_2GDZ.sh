#!/bin/bash
#SBATCH --job-name=Pipeline_2GDZ
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --account=u6bz

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to the working directory
cd /lus/lfs1aip2/projects/u6bz/jude/ThinkingPLM

# Initialize conda and activate environment
source ~/.bashrc
conda activate profam_bagel

# Run the pipeline
python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_berlin_hack_bio.yaml

echo "Job completed at $(date)"
