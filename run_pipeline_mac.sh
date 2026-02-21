#!/bin/bash
# Convenience wrapper to run the ProFam + BAGEL generative pipeline on a MacPro
# (or any local machine).
#
# Usage:
#   ./run_pipeline_mac.sh path/to/config.yaml
#
# Any additional CLI flags are passed through to the Python CLI, so you can
# override YAML values if desired.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/config.yaml [extra CLI args...]"
  exit 1
fi

CONFIG="$1"
shift || true

# Optional: activate your Python environment here, e.g.:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate profam_bagel

python run_profam_bagel_pipeline.py --config "${CONFIG}" "$@"

