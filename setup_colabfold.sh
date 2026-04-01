#!/bin/bash
# =============================================================================
# Setup localcolabfold for Nebius / cloud instances
#
# Installs ColabFold with MMseqs2 and AlphaFold2 model weights.
# After installation, the binary will be at:
#   ./localcolabfold/colabfold-conda/bin/colabfold_batch
#
# Usage:
#   chmod +x setup_colabfold.sh && ./setup_colabfold.sh
#
# Optionally set COLABFOLD_INSTALL_DIR to change install location:
#   COLABFOLD_INSTALL_DIR=/opt/colabfold ./setup_colabfold.sh
# =============================================================================

set -euo pipefail

INSTALL_DIR="${COLABFOLD_INSTALL_DIR:-$(pwd)/localcolabfold_install}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Installing localcolabfold"
echo "Install directory: ${INSTALL_DIR}"
echo "=============================================="

# Check for CUDA (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected. ColabFold will run on CPU (very slow)."
fi

# Create install directory
mkdir -p "${INSTALL_DIR}"
cd "${INSTALL_DIR}"

# Download and run the localcolabfold installer
if [ ! -f "install_colabbatch_linux.sh" ]; then
    echo "Downloading localcolabfold installer..."
    wget -q https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
fi

echo "Running localcolabfold installer (this may take 15-30 minutes)..."
bash install_colabbatch_linux.sh

# Verify installation
COLABFOLD_BIN="${INSTALL_DIR}/localcolabfold/colabfold-conda/bin/colabfold_batch"
if [ -x "${COLABFOLD_BIN}" ]; then
    echo ""
    echo "=============================================="
    echo "ColabFold installed successfully!"
    echo "Binary: ${COLABFOLD_BIN}"
    echo "=============================================="

    # Test with --help
    "${COLABFOLD_BIN}" --help 2>&1 | head -5

    # Create symlink in the repo for convenience
    LINK_PATH="${SCRIPT_DIR}/colabfold_batch"
    if [ ! -e "${LINK_PATH}" ]; then
        ln -s "${COLABFOLD_BIN}" "${LINK_PATH}"
        echo "Symlink created: ${LINK_PATH} -> ${COLABFOLD_BIN}"
    fi

    # Update energy config with correct path
    echo ""
    echo "To use ColabFold in the pipeline, update your energy config:"
    echo ""
    echo "  folding_oracle:"
    echo "    type: ColabFold"
    echo "    kwargs:"
    echo "      colabfold_bin: \"${COLABFOLD_BIN}\""
    echo "      num_models: 1"
    echo "      num_recycle: 1"
    echo ""
else
    echo "ERROR: ColabFold binary not found at ${COLABFOLD_BIN}"
    echo "Installation may have failed. Check the output above for errors."
    exit 1
fi
