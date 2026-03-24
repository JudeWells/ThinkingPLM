#!/bin/bash
#SBATCH --job-name=bench_2GDZ_15PGDH_3helix_bandit
#SBATCH --output=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.out
#SBATCH --error=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00

WORKDIR=/projects/u6bz/jude/ThinkingPLM

# Create logs directory if it doesn't exist
mkdir -p "$WORKDIR/logs"

# Initialize conda and activate environment
source ~/.bashrc
conda activate profam_bagel

# Run the pipeline
cd "$WORKDIR"
python "$WORKDIR/run_profam_bagel_pipeline.py" --config "$WORKDIR/configs/pipelines/bench_2GDZ_15PGDH_3helix_proposal_bandit.yaml"

echo "Job completed at $(date)"
