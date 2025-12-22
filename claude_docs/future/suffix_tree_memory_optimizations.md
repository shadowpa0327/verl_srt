# Suffix Tree Memory Optimizations

**Date**: 2025-12-20
**Status**: Implemented
**Priority**: High (Memory), Medium (Performance)

## Overview

This document describes the sequence eviction feature for suffix tree memory management. The implementation addresses unbounded **per-tree sequence growth** by evicting oldest sequences when a configurable limit is exceeded.

**Important Limitation**: This feature bounds the number of sequences *within each tree*, but does **not** limit the total number of trees (unique prompt hashes). For long training runs with many unique prompts, the number of trees can still grow. Tree-level eviction is a future enhancement (see [Future Enhancements](#future-enhancements-todo)).

---

## Problem Statement

### Root Cause

**Location**: [suffix_tree_manager.py:237-242](../../verl/trainer/ppo/suffix_tree_manager.py#L237-L242)

```python
# NOTE: We do NOT call stop_request() here!
# When processing requests sequentially, stop_request() would delete
# the tree since no other requests reference it. We want trees to
# persist for speculation, so we keep requests "active" in the cache.
```

This design choice enables tree reuse across batches but causes:
- Unbounded growth in number of sequences per tree
- Unbounded growth in `_req_to_tree_idx` dict
- Linear memory growth with training duration

### Impact

| Component | Growth Pattern | Risk |
|-----------|---------------|------|
| Tree sequences | O(total_responses) | HIGH - OOM on long runs |
| `_req_to_tree_idx` | O(total_requests) | MEDIUM - Dict overhead |

---

## Solution: Sequence Eviction

### Rationale

As RL training progresses, older sequences become stale due to:
1. **Policy drift**: Model's generation distribution changes after each actor update
2. **Reward shaping**: Patterns common early may be discouraged later
3. **Exploration → exploitation**: Early diverse samples become irrelevant as policy converges

### Implementation

The solution leverages the existing C++ `SuffixTree::remove(seq_id)` API to remove individual sequences from trees without destroying the entire tree.

#### Configuration

```python
@dataclass
class SuffixTreeManagerConfig:
    enable: bool = False
    max_tree_depth: int = 64
    hash_token_count: int = 128
    num_threads: int = -1
    parallel_threshold: int = 4

    # Eviction configuration (NEW)
    max_sequences_per_tree: int = 0   # 0 = disabled (unbounded)
```

#### Data Structure

```python
# Track sequences for eviction: tree_idx → deque of (batch_num, req_id, seq_id)
self._tree_sequence_history: Dict[int, Deque[Tuple[int, str, int]]] = {}
```

#### Algorithm

FIFO eviction: oldest sequences (by batch number) are removed first until the tree is within the configured limit.

```python
def _maybe_evict_old_sequences(self, tree_idx: int) -> int:
    """Evict oldest sequences from a tree if it exceeds max_sequences_per_tree."""
    if self.config.max_sequences_per_tree <= 0:
        return 0  # Eviction disabled

    history = self._tree_sequence_history.get(tree_idx)
    if not history:
        return 0

    evicted = 0
    while len(history) > self.config.max_sequences_per_tree:
        batch_num, req_id, seq_id = history.popleft()

        # Remove sequence from C++ tree
        tree = self._cache._forest.get_tree(tree_idx)
        tree.remove(seq_id)

        # Clean up Python tracking
        self._cache._req_to_tree_idx.pop(req_id, None)
        self._cache._req_to_seq_id.pop(req_id, None)

        evicted += 1

    return evicted
```

---

## Usage

### Enabling Eviction

Set `max_sequences_per_tree` to a positive value:

```python
config = SuffixTreeManagerConfig(
    enable=True,
    max_tree_depth=64,
    hash_token_count=128,
    max_sequences_per_tree=10,  # Keep last 10 sequences per tree
)
```

### Recommended Values

| Use Case | max_sequences_per_tree | Rationale |
|----------|------------------------|-----------|
| Short training (< 1K steps) | 0 (disabled) | Memory not a concern |
| Standard training | 10-20 | Balance memory vs coverage |
| Memory-constrained | 5-10 | Aggressive eviction |
| Maximum speculation quality | 50+ | Keep more history |

---

## Metrics

### New Metrics Added

| Metric | Description |
|--------|-------------|
| `suffix_tree/total_evicted` | Total sequences evicted since start |
| `suffix_tree/sequences_evicted` | Sequences evicted in current batch |
| `suffix_tree/total_sequences` | Sum of sequences across all trees |
| `suffix_tree/avg_sequences_per_tree` | Average sequences per tree |
| `suffix_tree/max_sequences_in_tree` | Maximum sequences in any tree |
| `suffix_tree/oldest_sequence_batch` | Batch number of oldest sequence |
| `suffix_tree/oldest_sequence_age` | Age of oldest sequence in batches |

### Monitoring

Watch for:
- `memory_mb` should plateau instead of growing linearly
- `sequences_evicted` should be stable (not increasing rapidly)
- `oldest_sequence_age` should stay bounded by `max_sequences_per_tree`

---

## Limitations & Caveats

### Tree Count Still Unbounded

Sequence eviction bounds per-tree memory but **not the number of trees**. Each unique prompt hash creates a tree that persists indefinitely. For datasets with many unique prompts, tree count (and metadata) can grow unbounded. Tree-level eviction is planned as a future enhancement.

### Enabling Eviction Mid-Run or on Old Checkpoints

If you enable eviction on a checkpoint that was saved without eviction, or enable it mid-run, existing sequences in trees won't be tracked for eviction. A warning is logged:

```
WARNING - Eviction enabled but sequence history missing for N/M trees.
Existing sequences in these trees won't be evicted until new sequences are added.
```

New sequences added after enabling will be tracked and evicted normally.

### Metrics Depend on History

Metrics like `suffix_tree/total_sequences` and `suffix_tree/oldest_sequence_age` are computed from `_tree_sequence_history`. If history is missing (old checkpoints, eviction previously disabled), these metrics will under-report. The `suffix_tree/memory_bytes` metric from the C++ forest remains accurate regardless.

---

## Checkpoint Support

Eviction state is fully persisted across checkpoints:
- `_tree_sequence_history` is saved/loaded
- `_total_evicted` counter is preserved
- Training can resume with correct eviction behavior

---

## Future Enhancements (TODO)

### Async Eviction

**Priority**: Medium - Consider if synchronous eviction becomes a bottleneck.

```python
eviction_mode: str = "sync"  # "sync" | "async"
```

### Memory Pressure Triggers

**Priority**: Low - Only if sequence-count eviction proves insufficient.

```python
max_total_memory_bytes: int = 0  # Hard memory cap
```

### Tree-Level Eviction

**Priority**: Low - For extreme memory pressure scenarios.

```python
max_total_trees: int = 0  # Evict entire trees LRU
```

---

## Files Modified

| File | Changes |
|------|---------|
| [verl/trainer/ppo/suffix_tree_manager.py](../../verl/trainer/ppo/suffix_tree_manager.py) | Config fields, eviction logic, metrics |

## Key C++ API Used

- `SuffixTree::remove(seq_id)` - [suffix_tree.cc:518-595](../../third_party/ArcticInference_srt/csrc/suffix_decoding/suffix_tree.cc#L518)
