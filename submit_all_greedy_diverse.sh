#!/bin/bash
# Submit all greedy_diverse benchmark jobs to SLURM
#
# Usage: ./submit_all_greedy_diverse.sh
#
# This script submits 17 jobs (4 targets x 4 scaffolds + 1 extra for 2GDZ)

set -euo pipefail

cd /projects/u6bz/jude/ThinkingPLM

# Create logs directory
mkdir -p logs

CONFIG_DIR="configs/pipelines"
SUBMIT_SCRIPT="slurm_greedy_diverse_job.sh"

# Find all greedy_diverse configs and submit them
for config in "${CONFIG_DIR}"/bench_*_greedy_diverse.yaml; do
    if [[ -f "${config}" ]]; then
        echo "Submitting: $(basename "${config}")"
        sbatch "${SUBMIT_SCRIPT}" "${config}"
    fi
done

echo ""
echo "All jobs submitted. Use 'squeue -u \$USER' to check status."
