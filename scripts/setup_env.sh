#!/usr/bin/env bash
# setup_env.sh – Create and configure the al_ee conda environment
# Usage: bash scripts/setup_env.sh
set -e

ENV_NAME="active_learning_ee"
PYTHON_VERSION="3.8"
CUDA_VERSION="11.5"

echo "=============================================="
echo " EE-AL Pipeline Environment Setup"
echo " Conda env: ${ENV_NAME} | Python: ${PYTHON_VERSION}"
echo " CUDA: ${CUDA_VERSION}"
echo "=============================================="

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
fi

# Create environment if it doesn't exist
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "[setup] Environment '${ENV_NAME}' already exists."
else
    echo "[setup] Creating conda environment '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate and install
echo "[setup] Installing PyTorch (CUDA ${CUDA_VERSION})..."
conda run -n ${ENV_NAME} pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 \
    --extra-index-url https://download.pytorch.org/whl/cu115

echo "[setup] Installing remaining requirements..."
conda run -n ${ENV_NAME} pip install -r requirements.txt

echo ""
echo "✓ Environment '${ENV_NAME}' is ready."
echo "  Activate with:  conda activate ${ENV_NAME}"
echo "  Run pipeline:   python run_al_pipeline.py --config configs/pascal_voc.yaml --strategy ee_al"
