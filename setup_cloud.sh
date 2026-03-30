#!/usr/bin/env bash
# =============================================================================
# setup_cloud.sh
#
# One-command setup for fresh cloud instances (AWS, GCP, Azure, etc.)
# Handles: miniconda installation, conda env creation, model downloads
#
# Usage:
#   chmod +x setup_cloud.sh
#   ./setup_cloud.sh
#
# After completion:
#   source ~/.bashrc && conda activate profam_bagel
#   python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="profam_bagel"

echo "============================================="
echo " ThinkingPLM Cloud Setup"
echo "============================================="
echo ""

# -------------------------------------------------------------------------
# 1. Check/Install Miniconda
# -------------------------------------------------------------------------
install_miniconda() {
    echo "Installing Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    INSTALLER="/tmp/miniconda_installer.sh"

    wget -q "${MINICONDA_URL}" -O "${INSTALLER}"
    bash "${INSTALLER}" -b -p "${HOME}/miniconda3"
    rm "${INSTALLER}"

    # Initialize conda for bash
    "${HOME}/miniconda3/bin/conda" init bash

    # Source the updated bashrc
    eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"

    echo "Miniconda installed to ${HOME}/miniconda3"
}

if command -v conda &> /dev/null; then
    echo "✓ Conda found: $(conda --version)"
    eval "$(conda shell.bash hook)"
else
    echo "Conda not found. Installing Miniconda..."
    install_miniconda
fi

# -------------------------------------------------------------------------
# 2. Check GPU availability
# -------------------------------------------------------------------------
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [[ -n "${GPU_INFO}" ]]; then
        echo "✓ GPU detected: ${GPU_INFO}"
    else
        echo "⚠ nvidia-smi found but no GPU detected. Pipeline will use Modal for GPU work."
    fi
else
    echo "⚠ nvidia-smi not found. Pipeline will use Modal for GPU work (run_on_modal: true)."
fi

# -------------------------------------------------------------------------
# 3. Run the main environment setup
# -------------------------------------------------------------------------
echo ""
echo "Running environment setup..."
cd "${SCRIPT_DIR}"

if [[ -f "setup_environment.sh" ]]; then
    chmod +x setup_environment.sh
    # Run non-interactively - say 'n' to recreate prompt if env exists
    echo "n" | ./setup_environment.sh || {
        echo ""
        echo "Environment may already exist. Attempting to activate..."
    }
else
    echo "ERROR: setup_environment.sh not found in ${SCRIPT_DIR}"
    exit 1
fi

# -------------------------------------------------------------------------
# 4. Activate environment and download model
# -------------------------------------------------------------------------
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo ""
echo "Downloading ProFam model checkpoint (~3GB)..."
PROFAM_MODEL_DIR="${SCRIPT_DIR}/.profam_repo/model_checkpoints/profam-1"

if [[ -d "${PROFAM_MODEL_DIR}" ]] && [[ -f "${PROFAM_MODEL_DIR}/config.json" ]]; then
    echo "✓ ProFam model already downloaded at ${PROFAM_MODEL_DIR}"
else
    python -c "
from huggingface_hub import snapshot_download
print('Downloading from HuggingFace Hub...')
snapshot_download('alex-hh/profam-1', local_dir='${PROFAM_MODEL_DIR}')
print('Download complete!')
"
fi

# -------------------------------------------------------------------------
# 5. Verify installation
# -------------------------------------------------------------------------
echo ""
echo "Verifying installation..."
python -c "
import sys

checks = []

# Check core imports
try:
    import numpy as np
    checks.append(('numpy', np.__version__, True))
except Exception as e:
    checks.append(('numpy', str(e), False))

try:
    import torch
    cuda_status = f'CUDA={torch.cuda.is_available()}'
    if torch.cuda.is_available():
        cuda_status += f' ({torch.cuda.get_device_name(0)})'
    checks.append(('torch', f'{torch.__version__} {cuda_status}', True))
except Exception as e:
    checks.append(('torch', str(e), False))

try:
    import bagel
    checks.append(('bagel', bagel.__version__, True))
except Exception as e:
    checks.append(('bagel', str(e), False))

try:
    from src.models.inference import ProFamSampler
    checks.append(('profam', 'OK', True))
except Exception as e:
    checks.append(('profam', str(e), False))

try:
    import modal
    checks.append(('modal', 'OK', True))
except Exception as e:
    checks.append(('modal', str(e), False))

try:
    import boltz
    checks.append(('boltz', 'OK', True))
except Exception as e:
    checks.append(('boltz', str(e), False))

# Print results
all_ok = True
for name, version, ok in checks:
    status = '✓' if ok else '✗'
    print(f'  {status} {name}: {version}')
    if not ok:
        all_ok = False

if not all_ok:
    print()
    print('WARNING: Some checks failed. The pipeline may not work correctly.')
    sys.exit(1)
"

# -------------------------------------------------------------------------
# 6. Print next steps
# -------------------------------------------------------------------------
echo ""
echo "============================================="
echo " Setup Complete!"
echo "============================================="
echo ""
echo "Next steps:"
echo ""
echo "  1. Reload shell (if conda was just installed):"
echo "     source ~/.bashrc"
echo ""
echo "  2. Activate the environment:"
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  3. (Optional) Configure Modal for cloud GPU:"
echo "     modal token new"
echo "     modal secret create huggingface-secret HF_TOKEN=hf_xxxxx"
echo ""
echo "  4. Run the pipeline:"
echo "     python run_profam_bagel_pipeline.py --config configs/pipelines/pipeline_campaign6_hairpin_elite.yaml"
echo ""
echo "Config files are in configs/pipelines/. Key options:"
echo "  - run_on_modal: true   (use Modal cloud for GPU - recommended)"
echo "  - run_on_modal: false  (use local GPU - requires NVIDIA GPU)"
echo ""
