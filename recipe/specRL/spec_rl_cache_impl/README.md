# specRL - Speculative Decoding RL

The C++ modules for specRL (Suffix Tree-based Distributed Draft Worker).

## Overview

This package provides two main modules:

- **cache_updater**: For updating the rollout cache via gRPC communication
- **suffix_cache**: For suffix tree based cache management and speculative decoding

Both modules are implemented in C++ with Python bindings via pybind11.

## Requirements

- CMake >= 3.14
- C++17 compatible compiler (GCC 7+)
- Boost (system, thread, chrono, atomic)
- gRPC and Protocol Buffers
- xxHash
- Python >= 3.8

## Installation

### Step 1: Install C++ dependencies via Conda

```bash
# Create conda environment for system libraries
conda create -n syslibs -y
conda activate syslibs

# Install C++ dependencies
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

### Step 2: Set environment variables and build

```bash
conda activate syslibs
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH

# Install with uv (if using uv-managed venv)
uv pip install -e /path/to/spec_rl_cache_impl/

# Or with pip
pip install -e /path/to/spec_rl_cache_impl/ --no-build-isolation -v
```

### Step 3: Runtime setup

Always set `LD_LIBRARY_PATH` before using specrl:

```bash
conda activate syslibs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Verifying Installation

```bash
python test_drafting.py
```

## Usage

```python
from specrl.suffix_cache import RolloutCacheServer, SuffixCache

# Start server with configurable shared memory size (in GB)
server = RolloutCacheServer("[::]:6378", shared_memory_size_gb=100)  # 100GB
server.initialize()
server.start()

# In vLLM worker process:
cache = SuffixCache()  # Connects to shared memory
cache.fetch_responses_by_prompts_batch([req_id], [prompt_tokens])
drafts = cache.speculate([req_id], [pattern_tokens])
```

## Troubleshooting

### ImportError: libprotobuf.so.32: cannot open shared object file

Set `LD_LIBRARY_PATH`:
```bash
conda activate syslibs
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### RuntimeError: No such file or directory (when creating SuffixCache)

This is expected if `RolloutCacheServer` is not running. Start the server first.

### Cannot allocate memory (when initializing server)

The default shared memory size is 500GB. Use a smaller size:
```python
server = RolloutCacheServer("[::]:6378", shared_memory_size_gb=10)  # 10GB
```

## License

Apache License 2.0

## Acknowledgments

This project leverages the suffix tree implementation from Snowflake's [ArcticInference](https://github.com/snowflakedb/ArcticInference).
