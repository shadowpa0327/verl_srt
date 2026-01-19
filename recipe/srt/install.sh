#!/bin/bash
# SRT Plugin Installation Script
#
# This script installs the SRT plugin with all C++ extensions:
# - suffix_cache: In-process snapshot mode (requires nanobind, cmake)
# - shm_cache: Shared memory mode (requires protobuf, grpc, xxhash, boost)
#
# Usage:
#   ./install.sh              # Install everything (requires sudo for apt)
#   ./install.sh --deps-only  # Only install system dependencies, don't build
#
# The script will:
# 1. Install system dependencies for shm_cache extensions via apt
# 2. pip install -e the srt_plugin package

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
DEPS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --deps-only)
            DEPS_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--deps-only]"
            echo ""
            echo "Options:"
            echo "  --deps-only  Only install dependencies, don't build the package"
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

# Install system dependencies via apt
print_header "Installing system dependencies via apt"
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

print_success "System dependencies installed."
echo ""

if [[ "${DEPS_ONLY}" == true ]]; then
    print_success "Dependencies installation complete."
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
    print_error "shm_cache extensions failed to build."
    echo "Check the build output above for errors."
fi
echo ""