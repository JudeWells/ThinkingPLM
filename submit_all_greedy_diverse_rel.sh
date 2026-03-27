#!/bin/bash
# Submit all greedy_diverse_rel benchmark jobs to SLURM
#
# Usage: ./submit_all_greedy_diverse_rel.sh
#
# This script submits 17 jobs (4 targets x 4 scaffolds + 1 extra for 2GDZ)
# Uses relative reward for proposal bandit (improvement over parent)

set -euo pipefail

cd /projects/u6bz/jude/ThinkingPLM

# Create logs directory
mkdir -p logs

CONFIG_DIR="configs/pipelines"
SUBMIT_SCRIPT="slurm_greedy_diverse_job.sh"

# Find all greedy_diverse_rel configs and submit them
for config in "${CONFIG_DIR}"/bench_*_greedy_diverse_rel.yaml; do
    if [[ -f "${config}" ]]; then
        echo "Submitting: $(basename "${config}")"
        sbatch "${SUBMIT_SCRIPT}" "${config}"
    fi
done

echo ""
echo "All jobs submitted. Use 'squeue -u \$USER' to check status."
