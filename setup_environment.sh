#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh
#
# Creates a clean conda environment with all dependencies needed to run:
#
#   python run_profam_bagel_pipeline.py --config example_pipeline_config.yaml
#
# This script resolves dependency conflicts between ProFam and BAGEL:
#   - BAGEL requires numpy>=2.2.0 ; ProFam's freeze pins numpy==1.26.4
#   - BAGEL requires matplotlib>=3.10.0 ; ProFam's freeze pins 3.9.4
#   - BAGEL requires transformers>=4.49.0 ; ProFam's freeze pins 4.48.3
#   - BAGEL requires boileroom, pydantic, modal (not in ProFam deps)
#   - ProFam's requirements.txt includes CUDA packages that don't exist on macOS
#
# The strategy: use BAGEL's stricter lower bounds (they are newer), install
# ProFam's core deps on top, and let pip resolve compatible versions.
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
# 3. Install BAGEL (editable) first - this pulls boileroom==0.2.2 which
#    constrains torch, plus biotite, numpy>=2.2, pandas, pydantic, matplotlib
# -------------------------------------------------------------------------
echo ""
echo "Installing BAGEL (editable)..."
pip install -e "${SCRIPT_DIR}/bagel"
pip install -e "${SCRIPT_DIR}/bagel[local]"

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
# 5. Install ProFam's core dependencies (not covered by BAGEL)
#    We use the package names from profam/setup.py rather than the pinned
#    requirements.txt (which has CUDA-only packages and old numpy pins).
# -------------------------------------------------------------------------
echo ""
echo "Installing ProFam dependencies..."

pip install \
  "transformers>=4.49.0" \
  "tokenizers" \
  "datasets" \
  "accelerate" \
  "lightning" \
  "pytorch-lightning" \
  "hydra-core" \
  "omegaconf"

# Additional ProFam runtime dependencies from requirements-cpu.txt
pip install \
  "biopython" \
  "biotraj" \
  "rootutils" \
  "safetensors" \
  "huggingface-hub" \
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
  "matplotlib>=3.10.0" \
  "modal"

# -------------------------------------------------------------------------
# 7. Verify key imports work
# -------------------------------------------------------------------------
echo ""
echo "Verifying imports..."

python -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}/bagel/src')
sys.path.insert(0, '${SCRIPT_DIR}/profam')

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
echo "  2. ProFam checkpoint downloaded in profam/model_checkpoints/"
echo "     (run: python profam/scripts/hf_download_checkpoint.py)"
echo ""
