#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh
#
# Creates a clean conda environment with all dependencies needed to run:
#
#   python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml
#
# Both ProFam and BAGEL (biobagel) are installed from their GitHub repos
# as pip packages.  This script resolves known dependency conflicts between
# them (e.g. numpy, matplotlib, transformers version pins).
#
# Usage:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# After running:
#   conda activate profam_bagel
#   cd /Users/stefano/CodeGen/profam_bagel
#   python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml
# =============================================================================

set -euo pipefail

ENV_NAME="profam_bagel"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo " ProFam + BAGEL Environment Setup"
echo "============================================="
echo ""

# -------------------------------------------------------------------------
# 1. Create (or recreate) the conda environment
# -------------------------------------------------------------------------
if conda info --envs | grep -q "^${ENV_NAME} "; then
  echo "Conda environment '${ENV_NAME}' already exists."
  read -rp "Remove and recreate it? [y/N] " answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "Removing existing environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "Reusing existing environment."
  fi
fi

if ! conda info --envs | grep -q "^${ENV_NAME} "; then
  echo "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# -------------------------------------------------------------------------
# 2. Activate the environment
# -------------------------------------------------------------------------
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# -------------------------------------------------------------------------
# 3. Install BAGEL (biobagel) from GitHub — this pulls boileroom==0.2.2
#    which constrains torch, plus biotite, numpy>=2.2, pandas, pydantic,
#    matplotlib.  The [local] extra adds transformers>=4.49.0.
# -------------------------------------------------------------------------
echo ""
echo "Installing BAGEL (biobagel) from GitHub..."
pip install "biobagel[local] @ git+https://github.com/softnanolab/bagel.git"

# -------------------------------------------------------------------------
# 4. Install PyTorch with matching torchvision/torchaudio
#    boileroom pins torch; we must install matching vision/audio versions.
# -------------------------------------------------------------------------
echo ""
echo "Installing PyTorch ecosystem..."

# Get the torch version that boileroom pulled in
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "  torch version (from boileroom): ${TORCH_VER}"

OS="$(uname -s)"

# Map torch version to compatible torchvision/torchaudio versions.
# PyTorch keeps a strict compatibility matrix.
case "${TORCH_VER}" in
  2.6.*)  TV_VER="0.21.0" ; TA_VER="2.6.0" ;;
  2.5.*)  TV_VER="0.20.0" ; TA_VER="2.5.0" ;;
  2.4.*)  TV_VER="0.19.0" ; TA_VER="2.4.0" ;;
  2.3.*)  TV_VER="0.18.0" ; TA_VER="2.3.0" ;;
  *)
    echo "  Unknown torch version ${TORCH_VER}, installing torchvision/torchaudio without pinning..."
    TV_VER=""
    TA_VER=""
    ;;
esac

if [[ -n "${TV_VER}" ]]; then
  echo "  Installing torchvision==${TV_VER}, torchaudio==${TA_VER}"
  if [[ "$OS" == "Linux" ]]; then
    pip install "torchvision==${TV_VER}" "torchaudio==${TA_VER}" --index-url https://download.pytorch.org/whl/cu124
  else
    pip install "torchvision==${TV_VER}" "torchaudio==${TA_VER}"
  fi
else
  pip install torchvision torchaudio
fi

# -------------------------------------------------------------------------
# 5. Install ProFam from GitHub (editable mode to include all submodules)
# -------------------------------------------------------------------------
echo ""
echo "Installing ProFam from GitHub..."
# Clone and install in editable mode because pip install from git misses src.sequence
PROFAM_DIR="${SCRIPT_DIR}/.profam_repo"
if [[ -d "${PROFAM_DIR}" ]]; then
  echo "  ProFam repo already cloned, pulling latest..."
  git -C "${PROFAM_DIR}" pull --quiet
else
  echo "  Cloning ProFam repository..."
  git clone --quiet https://github.com/alex-hh/profam.git "${PROFAM_DIR}"
fi
pip install -e "${PROFAM_DIR}"

# Additional ProFam runtime dependencies not in its setup.py
pip install \
  "rootutils" \
  "safetensors" \
  "huggingface-hub" \
  "biopython" \
  "scipy" \
  "scikit-learn" \
  "numba" \
  "rich" \
  "tqdm"

# -------------------------------------------------------------------------
# 6. Install pipeline-specific dependencies
# -------------------------------------------------------------------------
echo ""
echo "Installing pipeline utilities..."

pip install \
  "pyyaml" \
  "modal"

# -------------------------------------------------------------------------
# 7. Verify key imports work
# -------------------------------------------------------------------------
echo ""
echo "Verifying imports..."

python -c "
print('Checking numpy...', end=' ')
import numpy as np
print(f'OK (v{np.__version__})')

print('Checking torch...', end=' ')
import torch
print(f'OK (v{torch.__version__})')

print('Checking biotite...', end=' ')
import biotite
print(f'OK (v{biotite.__version__})')

print('Checking transformers...', end=' ')
import transformers
print(f'OK (v{transformers.__version__})')

print('Checking lightning...', end=' ')
import lightning
print(f'OK (v{lightning.__version__})')

print('Checking hydra...', end=' ')
import hydra
print('OK')

print('Checking pydantic...', end=' ')
import pydantic
print(f'OK (v{pydantic.__version__})')

print('Checking modal...', end=' ')
import modal
print('OK')

print('Checking boileroom...', end=' ')
import boileroom
print('OK')

print('Checking bagel...', end=' ')
import bagel as bg
print(f'OK (v{bg.__version__})')

print('Checking bagel.oracles.ESMFold...', end=' ')
from bagel.oracles import ESMFold
print('OK')

print('Checking bagel.energies...', end=' ')
from bagel.energies import TemplateMatchEnergy
print('OK')

print('Checking profam fasta utils...', end=' ')
from src.sequence.fasta import read_fasta, output_fasta
print('OK')

print('Checking profam model inference...', end=' ')
from src.models.inference import ProFamSampler, PromptBuilder
print('OK')

print('Checking pyyaml...', end=' ')
import yaml
print('OK')

print()
print('All imports verified successfully!')
"

echo ""
echo "============================================="
echo " Environment setup complete!"
echo "============================================="
echo ""
echo "To use the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run the pipeline:"
echo "  cd ${SCRIPT_DIR}"
echo "  python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml"
echo ""
echo "Make sure you have:"
echo "  1. Modal configured (modal token set) if using run_on_modal: true"
echo "  2. ProFam checkpoint downloaded in model_checkpoints/"
echo "     (run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('alex-hh/profam-1', local_dir='model_checkpoints/profam-1')\")"
echo ""
