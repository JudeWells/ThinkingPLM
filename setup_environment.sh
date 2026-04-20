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
echo "Pre-installing packages that require compilation from binary wheels..."
echo "(Avoids source builds that need GCC >= 9.3 on older clusters)"
pip install --only-binary=:all: \
  "numpy>=2.2,<2.5" \
  "scipy>=1.13" \
  "biotite>=1.0.1" \
  "Cython"

# BAGEL's pyproject.toml uses `hatchling` as its PEP 517 build backend.
# We install biobagel below with --no-build-isolation (to avoid pip
# downloading conflicting numpy versions into an isolated build env), so
# the build backend must be present in the current env up front.
echo "Installing build backends for --no-build-isolation pip installs..."
pip install "hatchling" "hatch-fancy-pypi-readme" "setuptools" "wheel"

echo ""
echo "Installing BAGEL (biobagel) from GitHub..."
pip install --no-build-isolation "biobagel[local] @ git+https://github.com/JudeWells/bagel.git"
# Pin transformers to 4.x — the 5.x series introduces MoE config attributes
# (_experts_implementation_internal) that break ProFam's LlamaConfig loading.
pip install "transformers>=4.49.0,<5.0.0"

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
  git clone --quiet https://github.com/JudeWells/profam_batched.git "${PROFAM_DIR}"
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

# modal is pinned to 0.73.45 — later versions (1.4+) raise DeprecationError
# at import time for container_idle_timeout inside boileroom (ESMFold init),
# which breaks any BAGEL oracle import.
pip install \
  "pyyaml" \
  "modal==0.73.45" \
  "boltz" \
  "optuna" \
  "wandb"

# chai-lab is the runtime dependency of bagel.oracles.folding.Chai1.  Install
# it even if you only plan to use ESMFold/Boltz — bagel/oracles/folding/__init__
# imports Chai1 at package-load time, so without chai-lab the whole BAGEL
# folding subpackage fails to import.
echo ""
echo "Installing chai-lab (Chai-1 oracle)..."
pip install "chai-lab"

# -------------------------------------------------------------------------
# 6b. Re-pin numpy to 1.26.4
#
# BAGEL declares numpy>=2.2.2, but the rest of the stack (numba, boltz,
# Chai-1's deps) either requires numpy<2.2 or is built against 1.x.  The
# only version that satisfies everything in practice is 1.26.4.  We install
# it *last* so pip doesn't re-resolve and break it.
# -------------------------------------------------------------------------
echo ""
echo "Pinning numpy to 1.26.4 for compatibility with numba/boltz/chai..."
pip install "numpy==1.26.4" "scipy==1.13.1" --no-deps

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

print('Checking bagel.oracles.folding (all 4)...', end=' ')
from bagel.oracles.folding import ESMFold, Boltz, Chai1, AF2BindCraft
print('OK')

import inspect
print('Checking Boltz.diffusion_samples kwarg...', end=' ')
assert 'diffusion_samples' in inspect.signature(Boltz.__init__).parameters, \
    'Boltz oracle is missing diffusion_samples — installed bagel may be out of date'
print('OK')

print('Checking bagel.energies...', end=' ')
from bagel.energies import TemplateMatchEnergy, ipSAEEnergy, iPTMEnergy, PLDDTEnergy
from bagel.energies import SolMPNNPerplexityEnergy
print('OK')

print('Checking SolMPNNPerplexityEnergy local backend...', end=' ')
assert 'proteinmpnn_env' in inspect.signature(SolMPNNPerplexityEnergy.__init__).parameters, \
    'SolMPNNPerplexityEnergy is missing the local subprocess backend — installed bagel may be out of date'
print('OK')

print('Checking bagel.scripts.proteinmpnn_scorer...', end=' ')
import os
import bagel
scorer_path = os.path.join(os.path.dirname(bagel.__file__), 'scripts', 'proteinmpnn_scorer.py')
assert os.path.isfile(scorer_path), f'Expected scorer at {scorer_path} — installed bagel may be out of date'
print('OK')

print('Checking profam fasta utils...', end=' ')
from src.sequence.fasta import read_fasta, output_fasta
print('OK')

print('Checking profam model inference...', end=' ')
from src.models.inference import ProFamSampler, PromptBuilder
print('OK')

print('Checking chai_lab (runtime dep of Chai1 oracle)...', end=' ')
import chai_lab
print(f'OK (v{chai_lab.__version__})')

print('Checking pyyaml...', end=' ')
import yaml
print('OK')

print('Checking boltz...', end=' ')
import boltz
print('OK')

print()
print('All imports verified successfully!')
"

# -------------------------------------------------------------------------
# 8. ProteinMPNN (for SolMPNNPerplexityEnergy — local subprocess backend)
#
# SolMPNNPerplexityEnergy scores binder designability by running SolubleMPNN
# in a subprocess via `conda run -n proteinmpnn`.  We need three things:
#
#   a) A clone of github.com/dauparas/ProteinMPNN at ${SCRIPT_DIR}/.proteinmpnn_repo
#      (includes `protein_mpnn_utils.py` AND the SolubleMPNN weights under
#      `soluble_model_weights/v_48_020.pt`).
#
#   b) A conda env (`proteinmpnn`) with torch + numpy that can import the
#      ProteinMPNN repo.  Python 3.10 is fine — the repo has no pinned deps
#      beyond torch + numpy, and keeping it separate isolates it from
#      BAGEL/Boltz/Chai1's numpy 1.26.4 pin.
#
#   c) A functional smoke test that runs the bundled scorer
#      (bagel/scripts/proteinmpnn_scorer.py) end-to-end on a dummy PDB
#      to catch any install issues BEFORE the pipeline launches.
#
# The multi-oracle energy configs reference this path as `.proteinmpnn_repo`
# (relative to the ThinkingPLM repo root), so the pipeline must be launched
# from ${SCRIPT_DIR}.
# -------------------------------------------------------------------------
echo ""
echo "============================================="
echo " ProteinMPNN subprocess environment"
echo "============================================="

PROTEINMPNN_ENV="proteinmpnn"
PROTEINMPNN_REPO="${SCRIPT_DIR}/.proteinmpnn_repo"
export PROTEINMPNN_ENV PROTEINMPNN_REPO

# --- a) Clone the ProteinMPNN repo ---
# Delete a dangling symlink if one exists (can happen when the repo was
# rsynced from a machine where .proteinmpnn_repo was a symlink to /mnt/...).
if [[ -L "${PROTEINMPNN_REPO}" && ! -e "${PROTEINMPNN_REPO}" ]]; then
  echo "Removing dangling symlink at ${PROTEINMPNN_REPO}..."
  rm -f "${PROTEINMPNN_REPO}"
fi

if [[ -d "${PROTEINMPNN_REPO}" ]]; then
  echo "ProteinMPNN repo already cloned at ${PROTEINMPNN_REPO}"
  git -C "${PROTEINMPNN_REPO}" pull --quiet || true
else
  echo "Cloning ProteinMPNN to ${PROTEINMPNN_REPO}..."
  git clone --quiet https://github.com/dauparas/ProteinMPNN.git "${PROTEINMPNN_REPO}"
fi

# Verify the SolubleMPNN weights are present (they ship with the repo)
SOLMPNN_CKPT="${PROTEINMPNN_REPO}/soluble_model_weights/v_48_020.pt"
if [[ ! -f "${SOLMPNN_CKPT}" ]]; then
  echo "ERROR: SolubleMPNN checkpoint not found at ${SOLMPNN_CKPT}"
  echo "       The ProteinMPNN repo clone appears to be incomplete."
  exit 1
fi
echo "  SolubleMPNN checkpoint: ${SOLMPNN_CKPT}"

# --- b) Create the proteinmpnn conda env with torch + numpy ---
if conda info --envs | grep -q "^${PROTEINMPNN_ENV} "; then
  echo "Conda env '${PROTEINMPNN_ENV}' already exists — verifying..."
  (
    eval "$(conda shell.bash hook)"
    conda activate "${PROTEINMPNN_ENV}"
    python -c "import torch, numpy; print(f'  torch={torch.__version__} numpy={numpy.__version__}')" \
      || { echo "  ERROR: torch/numpy missing from ${PROTEINMPNN_ENV} env — reinstalling"; pip install --quiet torch numpy; }
  )
else
  echo "Creating conda env '${PROTEINMPNN_ENV}' (Python 3.10 + torch + numpy)..."
  conda create -n "${PROTEINMPNN_ENV}" python=3.10 -y
  (
    eval "$(conda shell.bash hook)"
    conda activate "${PROTEINMPNN_ENV}"
    # torch installed via default index — whatever pip picks is fine since
    # this env is only used to run the MPNN scorer in a subprocess.
    pip install --quiet torch numpy
  )
fi

# --- c) End-to-end smoke test: run the bundled scorer on a dummy dimer ---
echo ""
echo "Running SolMPNN scorer smoke test..."
python 2>&1 <<'PYEOF'
import bagel, json, os, subprocess, sys, tempfile
bagel_dir = os.path.dirname(os.path.abspath(bagel.__file__))
scorer = os.path.join(bagel_dir, 'scripts', 'proteinmpnn_scorer.py')
assert os.path.isfile(scorer), f"scorer script missing at {scorer}"

# Minimal 3-residue chain A PDB (CA + backbone atoms, valid geometry).
pdb = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.332   1.548   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       4.012   2.847   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.525   2.627   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       5.954   1.478   0.000  1.00  0.00           O
ATOM      9  N   ALA A   3       6.296   3.720   0.000  1.00  0.00           N
ATOM     10  CA  ALA A   3       7.749   3.720   0.000  1.00  0.00           C
ATOM     11  C   ALA A   3       8.310   5.140   0.000  1.00  0.00           C
ATOM     12  O   ALA A   3       7.557   6.120   0.000  1.00  0.00           O
TER
END
"""

repo = os.environ.get('PROTEINMPNN_REPO')
env_name = os.environ.get('PROTEINMPNN_ENV', 'proteinmpnn')
with tempfile.TemporaryDirectory() as d:
    pdbp = os.path.join(d, 'in.pdb')
    outp = os.path.join(d, 'out.json')
    with open(pdbp, 'w') as f:
        f.write(pdb)
    cmd = [
        'conda', 'run', '-n', env_name, 'python', scorer,
        '--pdb', pdbp, '--chains_to_score', 'A',
        '--proteinmpnn_path', repo,
        '--backbone_noise', '0.1',
        '--ensemble_n', '3',
        '--decoding_order', 'random',
        '--output_json', outp,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  FAIL (exit {r.returncode})")
        print("  STDERR (last 1000 chars):")
        print("  " + r.stderr[-1000:].replace("\n", "\n  "))
        sys.exit(1)
    with open(outp) as f:
        result = json.load(f)
    print(f"  OK — perplexity={result['perplexity']:.3f} "
          f"mean_nll={result['mean_nll']:.3f} ensemble_n={result['ensemble_n']}")
PYEOF
SMOKE_STATUS=$?

# Reactivate the main env (the python call above left us in profam_bagel)
conda activate "${ENV_NAME}"

if [[ ${SMOKE_STATUS} -ne 0 ]]; then
  echo ""
  echo "ERROR: SolMPNN scorer smoke test failed — the proteinmpnn environment"
  echo "       is not usable. Check the stderr above for the root cause."
  exit 1
fi

echo ""
echo "SolMPNN scorer is functional."
echo "Multi-oracle energy configs can reference:"
echo "  use_modal: false"
echo "  proteinmpnn_env: ${PROTEINMPNN_ENV}"
echo "  proteinmpnn_path: .proteinmpnn_repo"
echo ""
echo "The relative path resolves against the ThinkingPLM repo root, so launch"
echo "the pipeline with cwd=${SCRIPT_DIR}."

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
echo "  1. ProFam checkpoint downloaded at .profam_repo/model_checkpoints/profam-1"
echo "     (run: python -c \"from huggingface_hub import snapshot_download; snapshot_download('alex-hh/profam-1', local_dir='.profam_repo/model_checkpoints/profam-1')\")"
echo ""
echo "  2. For the AF2BindCraft oracle (optional): install BindCraft separately"
echo "     git clone https://github.com/martinpacesa/BindCraft.git"
echo "     cd BindCraft && bash install_bindcraft.sh"
echo "     This creates a 'BindCraft' conda env and downloads AF2 params."
echo "     Reference the resulting paths in your energy YAML:"
echo "       folding_oracles:"
echo "         af2:"
echo "           type: AF2BindCraft"
echo "           kwargs:"
echo "             target_pdb: /path/to/target.pdb"
echo "             conda_env: BindCraft"
echo "             af_params_dir: /path/to/BindCraft/params"
echo ""
echo "  3. wandb login (optional): wandb login"
echo ""
