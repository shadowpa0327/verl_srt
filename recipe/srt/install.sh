#!/bin/bash
# SRT Plugin Installation Script
#
# This script installs the SRT plugin with all C++ extensions:
# - suffix_cache: In-process snapshot mode (requires nanobind, cmake)
# - shm_cache: Shared memory mode (requires protobuf, grpc, xxhash, boost)
#
# Usage:
#   ./install.sh              # Install using system dependencies
#   ./install.sh --conda      # Use conda for C++ dependencies (recommended)
#   ./install.sh --deps-only  # Only install system dependencies, don't build
#
# The script will:
# 1. Check/install system dependencies for shm_cache extensions
# 2. pip install -e the srt_plugin package
#
# Note: If shm_cache dependencies are missing, only suffix_cache will be built.
# This is fine if you only need snapshot mode.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRT_PLUGIN_DIR="${SCRIPT_DIR}/srt_plugin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}================================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Parse arguments
USE_CONDA=false
DEPS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --conda)
            USE_CONDA=true
            shift
            ;;
        --deps-only)
            DEPS_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--conda] [--deps-only]"
            echo ""
            echo "Options:"
            echo "  --conda      Use conda to install C++ dependencies (recommended)"
            echo "  --deps-only  Only install dependencies, don't build the package"
            echo ""
            echo "Without --conda, the script will check for system packages and"
            echo "provide instructions for installing missing ones."
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_header "SRT Plugin Installation"
echo ""
echo "This script will install the SRT plugin with C++ extensions."
echo "Plugin directory: ${SRT_PLUGIN_DIR}"
echo ""

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]] && [[ -z "${CONDA_PREFIX}" ]]; then
    print_warning "No virtual environment detected."
    print_warning "It's recommended to install in a virtual environment."
    echo ""
fi

# Function to check system dependencies
check_system_deps() {
    local missing_deps=()

    if ! command -v protoc &> /dev/null; then
        missing_deps+=("protobuf-compiler")
    fi

    if ! command -v grpc_cpp_plugin &> /dev/null; then
        missing_deps+=("protobuf-compiler-grpc")
    fi

    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    fi

    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi

    echo "${missing_deps[@]}"
}

# Function to install system dependencies via apt
install_apt_deps() {
    print_header "Installing system dependencies via apt"
    echo "This requires sudo access."
    echo ""

    sudo apt update
    sudo apt install -y \
        libprotobuf-dev \
        protobuf-compiler \
        libgrpc-dev \
        libgrpc++-dev \
        protobuf-compiler-grpc \
        libxxhash-dev \
        libboost-all-dev \
        cmake \
        pkg-config
}

# Function to install via conda
install_conda_deps() {
    print_header "Installing C++ dependencies via conda"

    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "conda not found. Please install Miniconda first."
        exit 1
    fi

    # Initialize conda
    eval "$(conda shell.bash hook)"

    # Install dependencies in current environment
    echo "Installing dependencies in current conda environment..."
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
    print_success "Conda dependencies installed."
    echo ""
    echo "Note: When building, you may need to set these environment variables:"
    echo "  export CMAKE_PREFIX_PATH=\$CONDA_PREFIX"
    echo "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
    echo ""
}

# Check for missing dependencies
MISSING_DEPS=$(check_system_deps)

if [[ -n "${MISSING_DEPS}" ]]; then
    print_warning "Missing system dependencies for shm_cache extensions:"
    for dep in ${MISSING_DEPS}; do
        echo "  - ${dep}"
    done
    echo ""

    if [[ "${USE_CONDA}" == true ]]; then
        install_conda_deps
    else
        echo "Options:"
        echo "  1. Install via apt (requires sudo):"
        echo "     $0 --deps-only  # Then manually: sudo apt install <packages>"
        echo ""
        echo "  2. Install via conda (recommended, no sudo needed):"
        echo "     $0 --conda"
        echo ""
        echo "  3. Continue without shm_cache:"
        echo "     Only suffix_cache will be available (snapshot mode)"
        echo ""

        read -p "Would you like to install via apt now? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_apt_deps
        else
            print_warning "Continuing without shm_cache dependencies."
            print_warning "Only suffix_cache (snapshot mode) will be available."
            echo ""
        fi
    fi
else
    print_success "All system dependencies for shm_cache are available."
    echo ""
fi

if [[ "${DEPS_ONLY}" == true ]]; then
    print_success "Dependencies check complete."
    echo "Run '$0' again without --deps-only to build the package."
    exit 0
fi

# Install pip dependencies
print_header "Installing pip dependencies"
pip install --upgrade pip setuptools wheel
pip install cmake ninja nanobind pybind11 numpy

# Build and install the package
print_header "Building and installing SRT plugin"
cd "${SRT_PLUGIN_DIR}"

# If using conda, set up environment
if [[ "${USE_CONDA}" == true ]] && [[ -n "${CONDA_PREFIX}" ]]; then
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    export CPATH="${CONDA_PREFIX}/include:${CPATH}"
    export LIBRARY_PATH="${CONDA_PREFIX}/lib:${LIBRARY_PATH}"
fi

# Install in editable mode
pip install -e .

print_header "Installation Complete"
echo ""
echo "You can now import the SRT plugin:"
echo ""
echo "  # Snapshot mode (always available)"
echo "  from srt_plugin.suffix_cache import _C"
echo ""

# Check if shm_cache was built (test each module separately due to protobuf conflict)
if python -c "from srt_plugin.shm_cache.suffix_cache import SuffixCache" 2>/dev/null; then
    print_success "shm_cache extensions are available!"
    echo ""
    echo "  # Shared memory mode (import one at a time due to protobuf conflict)"
    echo "  from srt_plugin.shm_cache.suffix_cache import SuffixCache, RolloutCacheServer"
    echo "  # OR"
    echo "  from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater"
    echo ""
    echo "NOTE: Both shm_cache modules register the same proto file."
    echo "      Import only one module per process to avoid protobuf conflicts."
else
    print_warning "shm_cache extensions are NOT available."
    echo "To enable shm_cache, install system dependencies and reinstall:"
    echo "  $0 --conda  # or install apt dependencies"
fi
echo ""
