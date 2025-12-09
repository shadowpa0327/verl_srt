# ArcticInference Cache Management Skill Guide

## Purpose
Quick-reference skill for suffix tree cache operations, serialization, and distributed patterns in ArcticInference_srt.

---

## Source Documentation

| Document | Path | Description |
|----------|------|-------------|
| **Primary** | `third_party/ArcticInference_srt/CLAUDE.md` | Full project overview |
| **Parallel Cache** | `third_party/ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md` | ParallelSuffixDecodingCache API |
| **Serialization** | `third_party/ArcticInference_srt/claude_docs/SUFFIX_TREE_SERIALIZATION.md` | Binary format spec |
| **Integration** | `third_party/claude_docs/role_of_third_party_lib.md` | Integration architecture overview |

---

## Quick Navigation Index

### By Task

| Task | Go To | Key File(s) |
|------|-------|-------------|
| Start/stop requests | [Request Lifecycle](#request-lifecycle) | `parallel_cache.py` |
| Batch speculation | [Batch Operations](#batch-operations) | `parallel_cache.py` |
| Serialize tree | [Serialization](#serialization) | `suffix_tree.cc` |
| Distribute cache | [Distributed Operations](#distributed-cache-operations) | `suffix_controller.py` (TODO) |
| Configure threading | [Thread Configuration](#thread-configuration) | `ParallelSuffixDecodingCache` |
| vLLM integration | [vLLM Proposer](#vllm-proposer-integration) | `suffix_decoding_parallel.py` |

### By Component

| Component | Section | Primary File |
|-----------|---------|--------------|
| SuffixTree (C++) | [Core Data Structures](#core-data-structures-c) | `csrc/suffix_decoding/suffix_tree.h` |
| SuffixForest (C++) | [Core Data Structures](#core-data-structures-c) | `csrc/suffix_decoding/suffix_forest.h` |
| ParallelSuffixDecodingCache | [Python API](#python-api) | `arctic_inference/suffix_decoding/parallel_cache.py` |
| Serialization | [Serialization](#serialization) | `suffix_tree.cc` |
| gRPC Server | [Remote Cache](#remote-cache-grpc) | `arctic_inference/suffix_decoding/server.py` |

---

## Architecture Overview

### Component Hierarchy

```
ParallelSuffixDecodingCache (Python)
    |
    +-- SuffixForest (C++)
            |
            +-- SuffixTree[] (C++)
                    |
                    +-- Node (path-compressed)
                            |
                            +-- Int32Map (children)
```

### Data Flow

```
1. start_request(req_id, prompt)
   +-> Creates SuffixTree, inserts prompt tokens

2. add_tokens(req_id, tokens)
   +-> Extends suffix tree with new tokens

3. speculate(req_id, context) / batch_speculate(req_ids, contexts)
   +-> Matches context in suffix tree
   +-> Finds most frequent continuation
   +-> Returns draft tokens with probabilities

4. stop_request(req_id)
   +-> Removes tree, frees memory
```

---

## Core Data Structures (C++)

### SuffixTree

**Location**: `csrc/suffix_decoding/suffix_tree.h`

```cpp
class SuffixTree {
    int _max_depth;                    // Max context window
    std::vector<Sequence> _seqs;       // Token sequences
    Node _root;                        // Root node
    // ...
};

struct Node {
    int64_t count;      // Frequency tracking
    int token;          // First token ID
    int length;         // Path-compressed length
    int ref_seq;        // Reference sequence ID
    int ref_idx;        // Starting index in ref sequence
    Int32Map<Node*> children;
    // Sibling linked list for fast iteration
};
```

**Key Methods:**
| Method | Line | Purpose |
|--------|------|---------|
| `extend(seq_id, tokens)` | - | Add tokens to tree |
| `remove_seq(seq_id)` | - | Remove sequence |
| `speculate(context, ...)` | - | Path-based speculation |
| `speculate_tree(context, ...)` | - | Tree-based speculation |
| `create_snapshot()` | - | Serialize to bytes |
| `restore_snapshot(bytes)` | - | Deserialize from bytes |
| `check_integrity()` | - | Validate tree structure |

### SuffixForest

**Location**: `csrc/suffix_decoding/suffix_forest.h`

```cpp
class SuffixForest {
    std::vector<std::unique_ptr<SuffixTree>> _trees;
    int _num_threads;           // OpenMP threads
    int _parallel_threshold;    // Min batch for parallelization
};
```

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `add_tree(max_depth)` | Create new tree, returns index |
| `remove_tree(index)` | Delete tree |
| `extend(index, seq_id, tokens)` | Add tokens to specific tree |
| `batch_speculate(indices, contexts, ...)` | Parallel speculation |
| `batch_extend(indices, seq_ids, tokens)` | Parallel token addition |

---

## Python API

### ParallelSuffixDecodingCache

**Location**: `arctic_inference/suffix_decoding/parallel_cache.py`

```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

cache = ParallelSuffixDecodingCache(
    max_tree_depth=64,      # Context window size
    num_threads=-1,         # -1=auto, 0=sequential, >0=specified
    parallel_threshold=4,   # Min batch for parallelization
    hash_token_count=128    # Hash last N tokens for tree sharing (0=disabled)
)
```

### Hash-Based Tree Sharing (NEW)

Multiple requests with the same prompt share a single tree:

```python
# Same prompt → same tree
prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)
cache.start_request("req_1", prompt)
cache.start_request("req_2", prompt)  # Shares tree with req_1

# Each request gets unique seq_id (prevents token interleaving)
# Tree only removed when last request stops
```

**Disable sharing:**
```python
cache = ParallelSuffixDecodingCache(hash_token_count=0)  # Per-cache
cache.start_request("req", prompt, hash_token_count=0)   # Per-request
```

### Request Lifecycle

```python
# Start request
cache.start_request(req_id, np.array([1,2,3], dtype=np.int32))

# Add generated tokens
cache.add_tokens(req_id, np.array([4,5,6], dtype=np.int32))

# Speculate
draft = cache.speculate(
    req_id=req_id,
    context=np.array([3,4,5], dtype=np.int32),
    max_spec_tokens=10,
    max_spec_factor=1.0,
    min_token_prob=0.1,
    use_tree_spec=False  # False=greedy, True=beam
)

# Results
print(draft.token_ids)   # List[int] - draft tokens
print(draft.parents)     # List[int] - tree structure
print(draft.probs)       # List[float] - probabilities
print(draft.score)       # float - overall score
print(draft.match_len)   # int - context match length

# Cleanup
cache.stop_request(req_id)
```

### Batch Operations

**Recommended for production** - parallelized with OpenMP:

```python
# Batch speculation (parallelized)
drafts = cache.batch_speculate(
    req_ids=["req_1", "req_2", "req_3"],
    contexts=[ctx1, ctx2, ctx3],
    max_spec_tokens=10
)

# Batch add tokens (parallelized)
cache.batch_add_tokens(
    req_ids=["req_1", "req_2", "req_3"],
    token_batches=[tokens1, tokens2, tokens3]
)
```

### Properties

```python
cache.max_tree_depth        # int - context window
cache.num_threads           # int - actual thread count
cache.parallel_threshold    # int - min batch for parallel
cache.hash_token_count      # int - tokens hashed for sharing (0=disabled)
cache.active_requests       # KeysView - active request IDs
cache.num_active_requests   # int - count
```

---

## Serialization

### Binary Format Overview

**Location**: `csrc/suffix_decoding/suffix_tree.cc`

| Section | Size | Description |
|---------|------|-------------|
| Header | 16B | version, max_depth, num_seqs, num_nodes |
| Sequences | Variable | seq_id, tokens[], active_indices[] |
| Nodes | Variable | BFS order, count, token, children indices |

### Usage

```python
from arctic_inference.suffix_decoding._C import SuffixTree

# Serialize
tree = SuffixTree(64)
tree.extend(0, [1, 2, 3, 4, 5])
snapshot = tree.create_snapshot()  # bytes

# Deserialize
restored = SuffixTree.restore_snapshot(snapshot)

# File I/O helpers
from arctic_inference.suffix_decoding import save_suffix_tree, load_suffix_tree
save_suffix_tree(tree, "tree.snapshot")
loaded = load_suffix_tree("tree.snapshot")
```

### Size Estimation

| Scenario | Estimated Size |
|----------|----------------|
| Empty tree | ~48 bytes |
| 1 seq x 100 tokens | ~5 KB |
| 10 seqs x 200 tokens | ~35 KB |
| 50 seqs x 500 tokens | ~150 KB |

---

## Distributed Cache Operations

For the distributed architecture (per-question trees, micro-batch assignment, snapshot push/pull), see:
- [Role of Third-Party Libraries - Integration Architecture](../role_of_third_party_lib.md#integration-architecture)
- [Prompt Hash Tree Mapping](prompt_hash_tree_mapping.md) - Hash-based tree sharing implementation

### Key APIs for Distribution

| Operation | Method | Description |
|-----------|--------|-------------|
| Serialize tree | `tree.create_snapshot()` | Returns bytes |
| Restore tree | `SuffixTree.restore_snapshot(bytes)` | Returns new tree |
| Serialize forest | `cache.create_snapshot(include_hash_mapping=True)` | Returns (snapshots, hash_mapping) |
| Restore forest | `cache.load_snapshot(snapshots, hash_to_tree=mapping)` | Reconstructs with hash lookup |

---

## Thread Configuration

### Auto-detect (Recommended)

```python
cache = ParallelSuffixDecodingCache(num_threads=-1)
```

### Manual Configuration

```python
import os

# Use physical cores (not logical)
cache = ParallelSuffixDecodingCache(num_threads=os.cpu_count() // 2)

# Disable parallelization (debugging)
cache = ParallelSuffixDecodingCache(num_threads=0)
```

### Parallel Threshold Tuning

| Batch Size | Recommended Threshold |
|------------|----------------------|
| 1-3 | 4 (sequential) |
| 4-16 | 4 (optimal) |
| 16+ | 8 (reduce overhead) |

```python
# Higher threshold for large batches
cache = ParallelSuffixDecodingCache(parallel_threshold=8)
```

---

## vLLM Proposer Integration

### ParallelSuffixDecodingProposer

**Location**: `vllm/v1/spec_decode/suffix_decoding_parallel.py`

```python
class ParallelSuffixDecodingProposer:
    def __init__(self, vllm_config):
        self.suffix_cache = ParallelSuffixDecodingCache(
            max_tree_depth=vllm_config.speculative_config.suffix_decoding_max_tree_depth,
            num_threads=-1,
            parallel_threshold=4
        )

    def propose(self, input_batch, sampled_token_ids) -> list[list[int]]:
        # Batch add tokens + batch speculate
        ...

    def load_snapshot(self, snapshot: bytes) -> None:
        """Load global tree snapshot from controller."""
        ...

    def create_snapshot(self) -> bytes:
        """Create snapshot of current cache state."""
        ...
```

### Configuration

```python
from vllm import LLM

llm = LLM(
    model="your-model",
    speculative_config={
        "method": "suffix",  # or "suffix_parallel"
        "num_speculative_tokens": 5,
        "suffix_decoding_max_tree_depth": 64,
        "suffix_decoding_max_spec_factor": 1.0,
        "suffix_decoding_min_token_prob": 0.1,
    }
)
```

---

## Remote Cache (gRPC)

### Server

```bash
python -m arctic_inference.suffix_decoding.server \
    --port 50051 \
    --max-workers 10 \
    --num-threads -1
```

### Client

```python
from arctic_inference.suffix_decoding.client import SuffixDecodingClient

client = SuffixDecodingClient(host='localhost', port=50051)

client.start_request(req_id, prompt_tokens)
client.add_tokens(req_id, new_tokens)
draft = client.speculate(req_id, context, max_spec_tokens=5)

# Batch operations
drafts = client.batch_speculate(req_ids, contexts)

client.close()
```

---

## Speculation Modes

### Path-based (Greedy)

```python
draft = cache.speculate(req_id, context, use_tree_spec=False)
# Returns: single path of most likely tokens
# Faster, simpler verification
```

### Tree-based (Beam Search)

```python
draft = cache.speculate(req_id, context, use_tree_spec=True)
# Returns: tree structure with multiple branches
# draft.parents encodes tree structure
# Higher acceptance rate, more complex verification
```

### Dynamic Speculation Limits

```python
draft = cache.speculate(
    req_id=req_id,
    context=context,
    max_spec_tokens=10,         # Hard limit
    max_spec_factor=1.0,        # Scale with match length
    max_spec_offset=0.0,        # Offset for scaling
    min_token_prob=0.1          # Quality threshold
)
# Actual tokens = min(max_spec_tokens, max_spec_factor * match_len + max_spec_offset)
```

---

## Directory Structure

```
ArcticInference_srt/
+-- arctic_inference/
|   +-- suffix_decoding/
|       +-- __init__.py
|       +-- cache.py              # SuffixDecodingCache (legacy)
|       +-- parallel_cache.py     # ParallelSuffixDecodingCache
|       +-- client.py             # gRPC client
|       +-- server.py             # gRPC server
|       +-- proto/                # Protocol buffers
+-- csrc/
|   +-- suffix_decoding/
|       +-- suffix_tree.h/.cc     # SuffixTree (C++)
|       +-- suffix_forest.h/.cc   # SuffixForest (C++)
|       +-- int32_map.h           # Fast hashmap
|       +-- bindings.cc           # Python bindings
+-- claude_docs/
    +-- PARALLEL_CACHE_GUIDE.md
    +-- SUFFIX_TREE_SERIALIZATION.md
```

---

## Common Pitfalls

| Issue | Problem | Solution |
|-------|---------|----------|
| Using list instead of np.int32 | Slow (copies data) | `np.array([...], dtype=np.int32)` |
| Not cleaning up requests | Memory leak | Always call `stop_request()` |
| Sequential calls in loops | Loses parallelism | Use `batch_speculate()` |
| Wrong speculation mode | Performance/accuracy tradeoff | Path for speed, tree for accuracy |
| num_threads=1 | No parallelism | Check OpenMP installation |
| Token interleaving | Cross-contamination | Use hash-based sharing (default) |
| Trees not reused after restore | Hash mapping not passed | Use `load_snapshot(snapshots, hash_to_tree=mapping)` |

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Snapshot create | < 1ms | For ~10K tokens |
| Snapshot restore | < 1ms | For ~10K tokens |
| Snapshot size | ~5KB/100 tokens | Path-compressed |
| Batch speculation | Near-linear scaling | Up to num_cores |

---

## Related Skills

- [vLLM V1 Architecture](vLLM_v1_architecture.md) - vLLM component reference
- [Role of Third-Party Libraries](../role_of_third_party_lib.md) - Integration architecture overview
