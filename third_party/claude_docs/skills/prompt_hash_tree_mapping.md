# Prompt Hash Tree Mapping - Reference Guide

## Purpose

Quick-reference for using hash-based tree sharing in `ParallelSuffixDecodingCache`. This feature allows multiple requests with the same prompt to share a single suffix tree, reducing memory and enabling tree reuse after snapshot loading.

---

## When to Use

| Scenario | Use Hash-Based Sharing? |
|----------|------------------------|
| Multiple requests with same prompt (e.g., same question, different samples) | Yes (default) |
| Distributed inference with snapshot transfer | Yes, with `include_hash_mapping=True` |
| Each request needs isolated tree | No, set `hash_token_count=0` |
| Debugging tree behavior per-request | No, disable per-request |

---

## Quick Start

### Enable Sharing (Default)

```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache
import numpy as np

cache = ParallelSuffixDecodingCache(
    max_tree_depth=64,
    hash_token_count=128  # Default: hash last 128 tokens
)

# Same prompt → same tree
prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)
cache.start_request("req_1", prompt)
cache.start_request("req_2", prompt)  # Shares tree with req_1

# Each request tracks tokens separately (no interleaving)
cache.add_tokens("req_1", np.array([100, 101], dtype=np.int32))
cache.add_tokens("req_2", np.array([200, 201], dtype=np.int32))
```

### Disable Sharing

```python
# Entire cache
cache = ParallelSuffixDecodingCache(hash_token_count=0)

# Per-request
cache.start_request("isolated", prompt, hash_token_count=0)
```

### Snapshot with Tree Reuse

```python
# Create snapshot WITH hash mapping
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

# Load on another worker - trees are reused via hash lookup
new_cache = ParallelSuffixDecodingCache(hash_token_count=128)
new_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)

# New request with same prompt reuses loaded tree
new_cache.start_request("new_req", prompt)
```

---

## API Reference

### Constructor

```python
ParallelSuffixDecodingCache(
    max_tree_depth: int = 64,
    num_threads: int = -1,
    parallel_threshold: int = 4,
    hash_token_count: int = 128  # 0 = disabled
)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `start_request` | `(req_id, prompt, hash_token_count=None)` | Start request; uses hash lookup if enabled |
| `stop_request` | `(req_id)` | Stop request; tree removed only when last user stops |
| `add_tokens` | `(req_id, tokens)` | Add tokens with per-request seq_id (thread-safe) |
| `create_snapshot` | `(include_hash_mapping=False)` | Returns `snapshots` or `(snapshots, hash_mapping)` |
| `load_snapshot` | `(snapshots, hash_to_tree=None)` | Load trees; pass mapping for hash-based reuse |

### Properties

```python
cache.hash_token_count  # int - tokens hashed (0 = disabled)
```

---

## Design

### Hash Function

- Hashes **last N tokens** (default N=128) using SHA-256, truncated to 64 bits
- Last tokens capture the unique question, ignoring common system prompts
- Collision probability: ~1/2^32 for 65K unique questions (negligible)

### Data Flow

```
prompt_tokens → hash(last 128 tokens) → tree_idx
```

### Tree Sharing Behavior

| Event | Behavior |
|-------|----------|
| `start_request` with existing hash | Reuse tree, assign new seq_id |
| `start_request` with new hash | Create new tree |
| `add_tokens` | Uses per-request seq_id (no interleaving) |
| `stop_request` | Tree removed only if no other requests use it |

### Concurrent Access

- C++ per-tree `std::shared_mutex` protects concurrent access
- Multiple readers (speculation) allowed simultaneously
- Writers (add_tokens) get exclusive access
- No Python-side locking needed

---

## Common Patterns

### Distributed Inference

```python
# Controller: create snapshot with hash mapping
snapshots, hash_mapping = controller_cache.create_snapshot(include_hash_mapping=True)
# Send to workers...

# Worker: load and reuse trees
worker_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)
# New requests automatically find matching trees via hash
```

### Mixed Sharing/Isolated Requests

```python
cache = ParallelSuffixDecodingCache(hash_token_count=128)

# These share a tree
cache.start_request("batch_1", common_prompt)
cache.start_request("batch_2", common_prompt)

# This gets its own tree
cache.start_request("special", common_prompt, hash_token_count=0)
```

---

## Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| Trees not reused after load | Missing hash mapping | Use `create_snapshot(include_hash_mapping=True)` and `load_snapshot(hash_to_tree=mapping)` |
| Unexpected tree sharing | Same prompt hash | Use `hash_token_count=0` for isolation |
| More trees than expected | Different prompts | Expected behavior; each unique hash gets a tree |

---

## Related Documentation

- [ArcticInference Cache Management](arctic_inference_cache_managements.md) - Full cache API reference
- [PARALLEL_CACHE_GUIDE.md](../../ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md) - Detailed API guide
