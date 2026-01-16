# SpecRL Cache Implementation Analysis

## Quick Reference (for Claude Code Sessions)

### What is SpecRL?
**SpecRL accelerates RL rollout using suffix-based speculative decoding.** It caches historical responses and uses them to draft tokens that can be verified in parallel, achieving up to 2.1x speedup.

### Core Insight
> Responses to the same prompt across RL epochs are often similar. Use historical responses as a "model-free draft model".

### Three Main Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `RolloutCacheServer` | suffix_cache/ | Creates shared memory, manages suffix trees, runs gRPC server |
| `SuffixCache` | suffix_cache/ | Connects to shared memory (zero-copy), performs speculation |
| `SuffixCacheUpdater` | cache_updater/ | Sends prompt/response pairs to server via gRPC |

### Quick Usage

```python
# Server (one per node, creates shared memory)
from specrl.suffix_cache import RolloutCacheServer
server = RolloutCacheServer("[::]:6378", shared_memory_size_gb=100)
server.initialize()
server.start()

# Worker (vLLM process, reads shared memory)
from specrl.suffix_cache import SuffixCache
cache = SuffixCache()
cache.fetch_responses_by_prompts_batch([req_id], [prompt_tokens])
drafts = cache.speculate([req_id], [pattern_tokens])

# Trainer (sends updates)
from specrl.cache_updater import SuffixCacheUpdater
updater = SuffixCacheUpdater(["[::1]:6378"])
updater.update_response_cache(prompts, responses, ...)
```

### Known Issues
1. **Protobuf conflict**: `cache_updater` and `suffix_cache` cannot be imported in the same process (different proto registrations). In production, they run in separate processes anyway.
2. **Shared memory size**: Default is 500GB. Use `shared_memory_size_gb` parameter to reduce.

### Installation (Conda, for non-root users)

```bash
# Install C++ dependencies
conda create -n syslibs -y && conda activate syslibs
conda install -c conda-forge protobuf libprotobuf grpc-cpp xxhash boost cmake pkg-config ninja -y

# Build
export CMAKE_PREFIX_PATH=$CONDA_PREFIX LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
uv pip install -e ./recipe/specRL/spec_rl_cache_impl/

# Runtime (always needed)
conda activate syslibs && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python check.py  # Verify installation
```

---

## Overview

The library implements a **distributed suffix tree cache** using:
- **Boost.Interprocess** for shared memory management
- **gRPC** for cross-process/cross-node cache updates
- **Ukkonen's algorithm** for O(n) suffix tree construction

The key innovation is that suffix trees live **directly in shared memory**, allowing vLLM workers to access them with **zero-copy reads**.

## How Drafting Works

```
Historical responses for "What is 2+2?":
  1. "The answer is 4. Two plus two equals four."
  2. "The answer is 4. This is basic arithmetic."
  3. "The answer is 4. Let me explain..."

Suffix tree indexes ALL substrings:
  ("answer", "is") → {"4.": 3}     # All 3 responses continue with "4."
  ("is", "4.")     → {"Two": 1, "This": 1, "Let": 1}

During generation:
  1. Model generates: "The answer is"
  2. Lookup ("answer", "is") → draft ["4."]
  3. Model verifies "4." in ONE forward pass
  4. If accepted → skip 1 autoregressive step!
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Trainer Process                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │  SuffixCacheUpdater (C++ gRPC client)                                       ││
│  │    └── stubs_[] → gRPC channels to cache servers                            ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │ gRPC UpdateCache()
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│       NODE 0          │ │       NODE 1          │ │       NODE 2          │
│                       │ │                       │ │                       │
│  RolloutCacheServer   │ │  RolloutCacheServer   │ │  RolloutCacheServer   │
│    (gRPC server)      │ │    (gRPC server)      │ │    (gRPC server)      │
│          │            │ │          │            │ │          │            │
│          ▼            │ │          ▼            │ │          ▼            │
│  ┌─────────────────┐  │ │  ┌─────────────────┐  │ │  ┌─────────────────┐  │
│  │ Shared Memory   │  │ │  │ Shared Memory   │  │ │  │ Shared Memory   │  │
│  │ "SUFFIX_CACHE"  │  │ │  │ "SUFFIX_CACHE"  │  │ │  │ "SUFFIX_CACHE"  │  │
│  │ (configurable)  │  │ │  │ (configurable)  │  │ │  │ (configurable)  │  │
│  └────────┬────────┘  │ │  └────────┬────────┘  │ │  └────────┬────────┘  │
│           │ zero-copy │ │           │ zero-copy │ │           │ zero-copy │
│  ┌────────┴────────┐  │ │  ┌────────┴────────┐  │ │  ┌────────┴────────┐  │
│  │  vLLM Worker 0  │  │ │  │  vLLM Worker 2  │  │ │  │  vLLM Worker 4  │  │
│  │  SuffixCache    │  │ │  │  SuffixCache    │  │ │  │  SuffixCache    │  │
│  ├─────────────────┤  │ │  ├─────────────────┤  │ │  ├─────────────────┤  │
│  │  vLLM Worker 1  │  │ │  │  vLLM Worker 3  │  │ │  │  vLLM Worker 5  │  │
│  │  SuffixCache    │  │ │  │  SuffixCache    │  │ │  │  SuffixCache    │  │
│  └─────────────────┘  │ │  └─────────────────┘  │ │  └─────────────────┘  │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
```

## Directory Structure

```
spec_rl_cache_impl/
├── specrl/
│   ├── __init__.py
│   │
│   ├── suffix_cache/              # vLLM worker side + server
│   │   ├── __init__.py            # Exports: SuffixCache, SuffixSpecResult, RolloutCacheServer
│   │   ├── suffix_tree.h/cc       # Ukkonen's suffix tree (shared memory)
│   │   ├── suffix_cache.h/cc      # Client that opens shared memory
│   │   ├── rollout_cache_server.h/cc  # gRPC server + shared memory owner
│   │   ├── pybind.cc              # Python bindings
│   │   └── CMakeLists.txt
│   │
│   ├── cache_updater/             # Trainer side
│   │   ├── __init__.py            # Exports: SuffixCacheUpdater
│   │   ├── suffix_cache_updater.h/cc  # gRPC client for updates
│   │   ├── pybind.cc              # Python bindings
│   │   └── CMakeLists.txt
│   │
│   └── proto/
│       └── rollout-cache.proto    # gRPC service definition
│
├── setup.py                       # Build configuration (CMake + pybind11)
├── check.py                       # Quick installation check
├── test_drafting.py               # Full test with drafting demo
├── test_shm.py                    # Shared memory tests
└── README.md
```

## Component Details

### 1. RolloutCacheServer (Shared Memory Owner)

**File**: `specrl/suffix_cache/rollout_cache_server.cc`

Creates and owns the shared memory segment, runs gRPC server.

```cpp
// Configurable shared memory size (default 500GB)
RolloutCacheServer::RolloutCacheServer(const std::string& server_address,
                                       unsigned long long shared_memory_size_gb)
    : shared_memory_size_(shared_memory_size_gb > 0
                          ? shared_memory_size_gb * 1024ULL * 1024ULL * 1024ULL
                          : DEFAULT_SHARED_MEMORY_SIZE) {}

bool RolloutCacheServer::Initialize() {
    // Remove any existing shared memory
    shared_memory_object::remove(SHARED_MEMORY_NAME);

    // Create the shared memory segment
    segment_ = new managed_shared_memory(create_only, SHARED_MEMORY_NAME, shared_memory_size_);

    // Create synchronization mutex
    segment_->construct<interprocess_mutex>("mutex")();

    // Create the hash map for trees (prompt_hash → tree_offset)
    TreeMapAllocator alloc(segment_->get_segment_manager());
    tree_map_ = segment_->construct<SharedTreeMap>("tree_map")(std::less<uint64_t>(), alloc);
}
```

**gRPC Update Handler**:
```cpp
grpc::Status RolloutCacheServiceImpl::UpdateCache(...) {
    // Allocate new tree IN shared memory
    ShmemAllocator alloc(segment_->get_segment_manager());
    SuffixTree* tree = segment_->construct<SuffixTree>(anonymous_instance)(alloc);

    // Build tree from prompt + responses using Ukkonen's algorithm
    tree->extend(0, tokens);

    // Lock and update the map
    scoped_lock<interprocess_mutex> shm_lock(*mutex);
    uint64_t tree_ptr = (uint64_t)tree - segment_base;  // Store as offset
    tree_map_->emplace(prompt_hash, tree_ptr);
}
```

### 2. SuffixTree (Shared Memory Data Structure)

**File**: `specrl/suffix_cache/suffix_tree.h`

Uses Boost.Interprocess containers with position-independent pointers:

```cpp
// Offset pointers work across different process address spaces
using NodePtr = offset_ptr<Node>;
using ChildrenMap = map<int, NodePtr, std::less<int>, ChildrenMapAllocator>;

struct Node {
    int count;                // Suffix count (frequency)
    NodePtr parent;           // Parent node
    ChildrenMap children;     // Token → Child mapping
    int start, length;        // Position in sequence
    NodePtr suffix_link;      // For Ukkonen's algorithm
};

class SuffixTree {
    void extend(int seq_id, const std::vector<int>& tokens);  // Build tree
    Candidate speculate(const std::vector<int>& pattern, ...);  // Get drafts
};
```

### 3. SuffixCache (vLLM Worker Client)

**File**: `specrl/suffix_cache/suffix_cache.cc`

Opens existing shared memory (zero-copy reads):

```cpp
SuffixCache::SuffixCache() {
    // Open EXISTING shared memory (doesn't allocate)
    shared_memory_segment_ = new managed_shared_memory(open_only, SHARED_MEMORY_NAME);

    // Find the tree map created by RolloutCacheServer
    shared_tree_map_ = shared_memory_segment_->find<SharedTreeMap>("tree_map").first;
}

// Zero-copy tree lookup
void SuffixCache::fetch_responses_by_prompts_batch(...) {
    auto it = shared_tree_map_->find(prompt_hash);
    if (it != shared_tree_map_->end()) {
        // Convert offset to direct pointer - ZERO COPY!
        req_id_to_responses_[req_id] = (SuffixTree*)(it->second + shared_mem_base);
    }
}

// Parallel speculation with OpenMP
std::vector<std::vector<int>> SuffixCache::speculate(...) {
    #pragma omp parallel for
    for (size_t i = 0; i < req_ids.size(); ++i) {
        results[i] = suffix_tree->speculate(pattern, spec_len, min_token_prob);
    }
}
```

### 4. SuffixCacheUpdater (Trainer gRPC Client)

**File**: `specrl/cache_updater/suffix_cache_updater.cc`

Sends updates to all cache servers asynchronously:

```cpp
void SuffixCacheUpdater::update_response_cache(...) {
    // Prepare requests with prompt hashes (XXH64)
    std::vector<UpdateCacheRequest> requests;

    // Send ALL requests to ALL servers asynchronously
    grpc::CompletionQueue cq;
    for (auto& request : requests) {
        for (auto& stub : stubs_) {
            stub->AsyncUpdateCache(&context, request, &cq);
        }
    }
    // Wait for completions...
}
```

## Data Flow

### Write Path (Trainer → Cache)

```
Trainer completes rollout batch
    │
    ▼
SuffixCacheUpdater.update_response_cache()
    │  - Compute prompt hashes (XXH64)
    │  - Create UpdateCacheRequest protos
    │
    ▼ (Async gRPC to ALL servers)
RolloutCacheServer.UpdateCache()
    │  - Allocate SuffixTree in shared memory
    │  - Build tree with Ukkonen's algorithm
    │  - Lock mutex, update tree_map
    │
    ▼
Done (trees available to all workers on same node)
```

### Read Path (vLLM Worker)

```
New request arrives at vLLM
    │
    ▼
GPUModelRunner.execute_model()
    │
    ▼
SuffixCache.fetch_responses_by_prompts_batch()
    │  - Lock mutex, lookup tree_map
    │  - Convert offset to pointer (ZERO COPY)
    │
    ▼
SuffixCache.speculate() [parallel with OpenMP]
    │  - Direct pointer access to SuffixTree
    │  - Tree traversal for draft tokens
    │
    ▼
Return draft tokens for verification
```

## Shared Memory Layout

```
Shared Memory Segment "SUFFIX_CACHE"
│
├── interprocess_mutex "mutex"           # Synchronization
│
├── SharedTreeMap "tree_map"             # prompt_hash → tree_offset
│   ├── hash_1 → offset_1
│   ├── hash_2 → offset_2
│   └── ...
│
├── SuffixTree @ offset_1 (anonymous)
│   ├── _root (NodePtr)
│   ├── _seqs (stored sequences)
│   └── Node tree structure...
│
├── SuffixTree @ offset_2 (anonymous)
│   └── ...
│
└── [Free space]
```

## Python API Summary

### RolloutCacheServer

```python
server = RolloutCacheServer(
    server_address="[::]:6378",
    shared_memory_size_gb=100  # Default: 500GB, now configurable!
)
server.initialize()  # Create shared memory
server.start()       # Start gRPC server
server.wait()        # Block until shutdown
server.shutdown()    # Cleanup
```

### SuffixCache

```python
cache = SuffixCache()  # Connects to existing shared memory

# Fetch trees for requests (must call before speculate)
cache.fetch_responses_by_prompts_batch(req_ids, prompts)

# Get draft tokens
drafts = cache.speculate(req_ids, patterns, min_token_prob=0.1)

# Update adaptive speculation length
cache.update_spec_len(req_id, accepted_length)

# Cleanup finished requests
cache.evict_responses(req_id)
```

### SuffixCacheUpdater

```python
updater = SuffixCacheUpdater(["[::1]:6378", "[::2]:6378"])
# Or: SuffixCacheUpdater()  # Reads from ARNOLD_WORKER_HOSTS env var

updater.update_response_cache(
    prompts=[[101, 102, ...], ...],
    responses=[[201, 202, ...], ...],
    prompt_lengths=[5.0, ...],
    response_lengths=[100.0, ...],
    responses_per_prompt=8
)
```

## Memory Usage Estimates

```
Per tree estimate:
- Node size: ~128 bytes
- Nodes per response: ~O(n) where n = response tokens
- Average response: 256 tokens
- Responses per prompt: 8

Per prompt: 8 × 256 × 128 ≈ 256KB

Realistic scenarios:
- 10K unique prompts: ~2.5GB
- 100K unique prompts: ~25GB
- 1M unique prompts: ~250GB
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data structure | Suffix tree | O(n) construction, O(m) lookup |
| Memory | Boost.Interprocess shared memory | Zero-copy across processes |
| Pointers | `offset_ptr<T>` | Position-independent (works across address spaces) |
| Communication | gRPC async | Non-blocking, works across nodes |
| Hashing | XXH64 | Fast, good distribution |
| Parallelism | OpenMP | Simple parallel speculation |
| Synchronization | Single mutex | Simple, sufficient for read-heavy workload |

## Dependencies

- **Boost.Interprocess**: Shared memory management
- **gRPC + Protobuf**: Cross-process communication
- **xxHash**: Fast prompt hashing
- **OpenMP**: Parallel speculation
- **pybind11**: Python bindings

## Files Modified for Configurable Memory

Recent changes to make shared memory size configurable:

1. `rollout_cache_server.h`: Added `shared_memory_size_` member, updated constructor signature
2. `rollout_cache_server.cc`: Constructor takes `shared_memory_size_gb` (0 = default 500GB)
3. `pybind.cc`: Exposed `shared_memory_size_gb` parameter to Python

## References

- [Boost.Interprocess Documentation](https://www.boost.org/doc/libs/release/doc/html/interprocess.html)
- [Ukkonen's Algorithm](https://en.wikipedia.org/wiki/Ukkonen%27s_algorithm)
- [SRT Paper](https://arxiv.org/abs/2503.09275) - Speculative Rollout with Tree-Structured Cache
