#!/bin/bash
# Install verl dependencies without installing the verl package itself
# Equivalent to the Dockerfile but as a standalone bash script
# Skips: vLLM, ArcticInference, Apex, Megatron
# Uses uv for venv creation and package installation

set -e  # Exit on error

# ----------
# Configuration
# ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON_VERSION="3.11"
PYTHON_VERSION_SHORT="311"
TORCH_VERSION="2.8"
FLASHINFER_VERSION="0.3.1"
VENV_DIR="${VENV_DIR:-.venv}"

# Third-party dependencies directory
THIRD_PARTY_DIR="${VERL_ROOT}/third_party"

# vLLM - cloned under third_party for co-development
VLLM_REPO="https://github.com/shadowpa0327/vllm.git"
VLLM_BRANCH="suffix_decode"
VLLM_DIR="${THIRD_PARTY_DIR}/vllm"

# ArcticInference - cloned under third_party for co-development
ARCTIC_REPO="https://github.com/shadowpa0327/ArcticInference_srt.git"
ARCTIC_BRANCH="dev/grpc_server"
ARCTIC_DIR="${THIRD_PARTY_DIR}/ArcticInference_srt"

# Proxy settings (set these if needed)
# export https_proxy="http://your-proxy:port"

# ----------
# Check uv installation
# ----------
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "=== Installing verl dependencies using uv ==="

# ----------
# Create venv
# ----------
if [ ! -d "${VENV_DIR}" ]; then
    echo ">>> Creating virtual environment at ${VENV_DIR} with Python ${PYTHON_VERSION}..."
    uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
else
    echo ">>> Using existing virtual environment at ${VENV_DIR}"
fi

# ----------
# PyTorch
# ----------
echo ">>> Installing PyTorch 2.8.0+cu128..."
uv pip install --python "${VENV_DIR}/bin/python" \
    "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128

# -------------------
# Flash Attention 2
# -------------------
echo ">>> Installing Flash Attention 2..."
uv pip install --python "${VENV_DIR}/bin/python" ninja==1.13.0 psutil

# Install prebuilt wheel
VERSION="2.8.3"
ABI_FLAG=$("${VENV_DIR}/bin/python" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FILE="flash_attn-${VERSION}+cu12torch${TORCH_VERSION}cxx11abi${ABI_FLAG}-cp${PYTHON_VERSION_SHORT}-cp${PYTHON_VERSION_SHORT}-linux_x86_64.whl"
REPO="Dao-AILab/flash-attention"
URL="https://github.com/${REPO}/releases/download/v${VERSION}/${FILE}"

echo ">>> Downloading flash-attn from: ${URL}"
uv pip install --python "${VENV_DIR}/bin/python" "${URL}"

# ------------
# FlashInfer
# ------------
echo ">>> Installing FlashInfer ${FLASHINFER_VERSION}..."
uv pip install --python "${VENV_DIR}/bin/python" \
    "flashinfer-python==${FLASHINFER_VERSION}" --prerelease=allow
# uv pip install --python "${VENV_DIR}/bin/python" \
#     "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu128" \
#     --index-url https://flashinfer.ai/whl/cu128

# ---------------
# Miscellaneous
# ---------------
echo ">>> Installing miscellaneous dependencies..."
uv pip install --python "${VENV_DIR}/bin/python" \
    transformers==4.57.1 \
    hydra-core \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "numpy<2.0.0" \
    pytest \
    pybind11 \
    codetiming \
    torchdata \
    datasets \
    peft \
    qwen_vl_utils \
    mathruler \
    pylatexenc \
    cupy-cuda12x

# Additional dependencies from setup.py
echo ">>> Installing additional dependencies from setup.py..."
uv pip install --python "${VENV_DIR}/bin/python" \
    accelerate \
    dill \
    pandas \
    "pyarrow>=19.0.0" \
    "ray[default]>=2.41.0" \
    wandb \
    "packaging>=20.0" \
    tensorboard

# ------
# vLLM
# ------
echo ">>> Setting up vLLM for co-development..."
mkdir -p "${THIRD_PARTY_DIR}"
if [ ! -d "${VLLM_DIR}" ]; then
    echo ">>> Cloning vLLM to ${VLLM_DIR}..."
    git clone --branch "${VLLM_BRANCH}" "${VLLM_REPO}" "${VLLM_DIR}"
else
    echo ">>> vLLM already exists at ${VLLM_DIR}, skipping clone"
fi

echo ">>> Installing vLLM in editable mode..."
export VLLM_USE_PRECOMPILED=1
uv pip install --python "${VENV_DIR}/bin/python" -e "${VLLM_DIR}"

# ----------------
# ArcticInference
# ----------------
echo ">>> Setting up ArcticInference for co-development..."
if [ ! -d "${ARCTIC_DIR}" ]; then
    echo ">>> Cloning ArcticInference to ${ARCTIC_DIR}..."
    git clone --branch "${ARCTIC_BRANCH}" "${ARCTIC_REPO}" "${ARCTIC_DIR}"
else
    echo ">>> ArcticInference already exists at ${ARCTIC_DIR}, skipping clone"
fi

echo ">>> Installing ArcticInference in editable mode..."
uv pip install --python "${VENV_DIR}/bin/python" -e "${ARCTIC_DIR}"

# ----------
# Epilogue
# ----------
echo ""
echo "=== Dependency installation complete ==="
echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
