#!/bin/bash
# Install C++ dependencies for specRL via conda

set -e

CONDA_ENV_NAME="${1:-syslibs}"

echo "=================================="
echo "specRL Dependency Installer"
echo "=================================="
echo "Conda environment: ${CONDA_ENV_NAME}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda first."
    exit 1
fi

# Initialize conda
eval "$(conda shell.bash hook)"

# Create environment if needed
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Environment '${CONDA_ENV_NAME}' exists, activating..."
else
    echo "Creating environment '${CONDA_ENV_NAME}'..."
    conda create -n "${CONDA_ENV_NAME}" -y
fi

conda activate "${CONDA_ENV_NAME}"

echo "Installing C++ dependencies..."
conda install -c conda-forge \
    protobuf \
    libprotobuf \
    grpc-cpp \
    xxhash \
    boost \
    cmake \
    pkg-config \
    ninja \
    -y

echo ""
echo "=================================="
echo "Done!"
echo "=================================="
echo ""
echo "Before building, run:"
echo "  conda activate ${CONDA_ENV_NAME}"
echo "  export CMAKE_PREFIX_PATH=\$CONDA_PREFIX"
echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  export CPATH=\$CONDA_PREFIX/include:\$CPATH"
echo "  export LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LIBRARY_PATH"
echo ""
