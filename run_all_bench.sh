#!/bin/bash
set -e
source /home/judewells/miniconda3/etc/profile.d/conda.sh
conda activate profam_bagel
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs/pipelines"

for config in "${CONFIG_DIR}"/bench_*.yaml; do
    echo "=========================================="
    echo "Running: $(basename "$config")"
    echo "=========================================="
    python "${SCRIPT_DIR}/run_profam_bagel_pipeline.py" --config "$config"
done

echo "All bench configs completed."
