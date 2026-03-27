#!/bin/bash
# Submit all ensemble benchmark jobs to SLURM
#
# Usage: ./submit_all_ensemble.sh
#
# This script submits 64 jobs (4 strategies x 4 targets x 4 scaffolds)
# All use boltz_ensemble_n=3 for noise reduction

set -euo pipefail

cd /projects/u6bz/jude/ThinkingPLM

# Create logs directory
mkdir -p logs

CONFIG_DIR="configs/pipelines"
SUBMIT_SCRIPT="slurm_ensemble_job.sh"

# Make submit script executable
chmod +x "${SUBMIT_SCRIPT}"

# Find all ensemble configs and submit them
count=0
for config in "${CONFIG_DIR}"/ens_*.yaml; do
    if [[ -f "${config}" ]]; then
        echo "Submitting: $(basename "${config}")"
        sbatch "${SUBMIT_SCRIPT}" "${config}"
        ((count++))
    fi
done

echo ""
echo "Submitted ${count} jobs. Use 'squeue -u \$USER' to check status."
