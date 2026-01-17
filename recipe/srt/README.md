# SRT Plugin Installation Guide

SRT (Speculative Rollout with Tree-Structured Cache) accelerates on-policy rollouts by using cached token sequences as a model-free draft model for speculative decoding.

## Overview

The SRT plugin includes two cache modes:

| Mode | Extension | Use Case |
|------|-----------|----------|
| **Snapshot** | `suffix_cache._C` | In-process suffix tree (always built) |
| **Shared Memory** | `shm_cache.*` | Cross-process cache via gRPC (requires system deps) |

## Quick Start

```bash
cd recipe/srt
./install.sh
```

The script will prompt you to install missing dependencies if needed.

## Installation Methods

### Method 1: System Dependencies (Default, Recommended)

This method uses `apt` to install C++ dependencies system-wide.

#### Step 1: Install System Dependencies

```bash
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
```

#### Step 2: Install the Plugin

```bash
cd recipe/srt
./install.sh
```

Or manually:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install Python build dependencies
uv pip install cmake ninja nanobind pybind11 numpy

# Install the plugin in editable mode
uv pip install -e recipe/srt/srt_plugin
```

### Method 2: Conda Dependencies (No sudo required)

Use this method if you don't have sudo access or prefer isolated dependencies.

#### Step 1: Create/Activate Conda Environment

```bash
# Create a new environment (optional)
conda create -n verl python=3.11
conda activate verl
```

#### Step 2: Install Dependencies via Conda

```bash
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
```

#### Step 3: Set Environment Variables

```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
```

#### Step 4: Install the Plugin

```bash
cd recipe/srt
./install.sh --conda
```

Or use the one-liner:

```bash
./install.sh --conda
```

This automatically sets up the environment variables and installs dependencies.

## Verifying Installation

```python
# Snapshot mode (always available)
from srt_plugin.suffix_cache import _C
print("Snapshot mode: OK")

# Shared memory mode (requires system dependencies)
from srt_plugin.shm_cache.suffix_cache import SuffixCache, RolloutCacheServer
print("Shared memory mode: OK")
```

## Partial Installation (Snapshot Mode Only)

If you only need snapshot mode and don't want to install gRPC/protobuf dependencies:

```bash
uv pip install -e recipe/srt/srt_plugin
```

The build will automatically skip `shm_cache` extensions if dependencies are missing. You'll see a warning but the snapshot mode will still work.

## Troubleshooting

### Missing protoc or grpc_cpp_plugin

```
CMake Error: Could not find protoc
```

**Solution**: Install protobuf and gRPC:
```bash
# Ubuntu/Debian
sudo apt install protobuf-compiler protobuf-compiler-grpc

# Conda
conda install -c conda-forge protobuf grpc-cpp
```

### Library not found at runtime

```
ImportError: libgrpc++.so: cannot open shared object file
```

**Solution**: Add library path:
```bash
# For conda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# For system install, libraries should be in standard paths
sudo ldconfig
```

### Protobuf version mismatch

```
TypeError: Descriptors cannot be created directly
```

**Solution**: This happens when Python protobuf version doesn't match the C++ version. Ensure consistency:
```bash
# Check versions
protoc --version
python -c "import google.protobuf; print(google.protobuf.__version__)"

# Reinstall matching version
uv pip install protobuf==4.25.0  # Match your protoc version
```

### CMake can't find packages (Conda)

**Solution**: Ensure CMAKE_PREFIX_PATH is set:
```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
```

## Directory Structure

```
recipe/srt/
├── install.sh                  # Installation script
├── srt_plugin/
│   ├── suffix_cache/           # Snapshot mode (nanobind)
│   │   └── csrc/               # C++ suffix tree implementation
│   ├── shm_cache/              # Shared memory mode (pybind11)
│   │   ├── suffix_cache/       # SuffixCache, RolloutCacheServer
│   │   ├── cache_updater/      # gRPC client for cache updates
│   │   └── proto/              # gRPC protobuf definitions
│   ├── proposers/              # Speculative decoding proposers
│   ├── patches/                # vLLM patches
│   ├── setup.py                # Build configuration
│   └── pyproject.toml          # Package metadata
└── README.md                   # This file
```

## Usage

After installation, the plugin is automatically enabled via vLLM's plugin system. To disable:

```bash
VERL_SRT_DISABLED=1 python your_training_script.py
```

See `CLAUDE.md` for detailed integration documentation.
