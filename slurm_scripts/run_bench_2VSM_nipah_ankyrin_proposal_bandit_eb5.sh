#!/bin/bash
#SBATCH --job-name=bench_2VSM_nipah_ankyrin_proposal_bandit_eb5
#SBATCH --output=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.out
#SBATCH --error=/projects/u6bz/jude/ThinkingPLM/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00

WORKDIR=/projects/u6bz/jude/ThinkingPLM

mkdir -p "$WORKDIR/logs"

source ~/.bashrc
conda activate profam_bagel

cd "$WORKDIR"
python "$WORKDIR/run_profam_bagel_pipeline.py" --config "$WORKDIR/configs/pipelines/bench_2VSM_nipah_ankyrin_proposal_bandit_eb5.yaml"

echo "Job completed at $(date)"
