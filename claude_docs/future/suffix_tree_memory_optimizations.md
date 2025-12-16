# Suffix Tree Memory Optimizations

**Date**: 2025-12-14 (Updated)
**Status**: Analysis Complete, Ready for Implementation
**Priority**: High (Memory), Medium (Performance)

## Overview

This document analyzes memory concerns in the suffix tree integration and proposes optimizations. The analysis covers:

1. Unbounded tree growth
2. Snapshot serialization overhead
3. Snapshot transfer costs
4. Request ID accumulation

---

## Memory Concerns

### Concern 1: Unbounded Tree Growth (HIGH PRIORITY)

**Location**: `verl/trainer/ppo/suffix_tree_manager.py:220-225`

**Problem**: Trees are never cleaned up. The code explicitly avoids calling `stop_request()`:

```python
# NOTE: We do NOT call stop_request() here!
# When processing requests sequentially, stop_request() would delete
# the tree since no other requests reference it. We want trees to
# persist for speculation, so we keep requests "active" in the cache.
```

**Impact**:
- Every unique prompt hash creates a new tree
- Trees accumulate responses but are never pruned
- Memory grows linearly with number of unique prompts
- Long training runs can exhaust memory

**Metrics to monitor**:
```
suffix_tree/num_trees - Should plateau, not grow indefinitely
suffix_tree/total_bytes - Total memory used by trees (not currently tracked)
```

---

### Recommended Solution: Age-Based Sequence Eviction (PREFERRED)

**Rationale**: As RL training progresses, older sequences become stale because:
1. **Policy drift**: Model's generation distribution changes after each actor update
2. **Reward shaping**: Patterns common early may be discouraged later
3. **Exploration → exploitation**: Early diverse samples become irrelevant as policy converges

**Key insight**: Evict sequences within trees, not entire trees. This preserves the tree structure while removing outdated patterns.

#### Configuration

```python
@dataclass
class SuffixTreeManagerConfig:
    enable: bool = False
    max_tree_depth: int = 64
    hash_token_count: int = 128

    # Age-based sequence eviction
    sequence_max_age_batches: int = 100   # Evict sequences older than N batches
    prune_interval_batches: int = 20      # How often to run pruning
    min_sequences_per_tree: int = 10      # Keep at least N sequences per tree
```

#### Implementation: Leveraging Existing ArcticInference Tracking

**Good news**: `ParallelSuffixDecodingCache` already tracks everything we need!

Looking at [parallel_cache.py](../../third_party/ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py):
- `self._req_to_tree_idx: Dict[Hashable, int]` - Maps req_id → tree_idx (line 76)
- `self._hash_to_tree_idx: Dict[str, int]` - Maps prompt_hash → tree_idx (line 79)
- `active_requests` property - Returns all active request IDs (line 106-113)

This means we can:
1. Get all active req_ids via `cache.active_requests`
2. Get tree index for any req_id via `cache._req_to_tree_idx[req_id]`
3. Get tree for any prompt hash via `cache._hash_to_tree_idx[prompt_hash]`

**The only missing piece**: We need to track **when** each req_id was added (batch number). This is a simple addition to `SuffixTreeManager`.

##### Implementation (Simple - Just Add Age Tracking)

```python
class SuffixTreeManager:
    def __init__(self, config, tokenizer):
        # ... existing init ...

        # Track when each request was added: req_id -> batch_number
        self._request_batch: Dict[str, int] = {}

    def update_from_rollout(self, batch) -> Dict[str, Any]:
        # ... existing code ...

        for i in range(batch_size):
            req_id = f"train_{self._batch_counter}_{i}"
            self._cache.start_request(req_id, prompt_array)
            self._cache.add_tokens(req_id, response_array)

            # Track when this request was added
            self._request_batch[req_id] = self._batch_counter

        # Periodic pruning
        if self._batch_counter % self.config.prune_interval_batches == 0:
            prune_stats = self._prune_old_sequences()
            stats.update(prune_stats)

        return stats

    def _prune_old_sequences(self) -> Dict[str, Any]:
        """Remove sequences older than max_age_batches."""
        cutoff_batch = self._batch_counter - self.config.sequence_max_age_batches

        # Get all active requests from cache (already tracked by ArcticInference)
        active_requests = set(self._cache.active_requests)

        # Find old requests to evict
        to_evict = [
            req_id for req_id in active_requests
            if self._request_batch.get(req_id, 0) < cutoff_batch
        ]

        # Optional: enforce min_sequences_per_tree using ArcticInference's tracking
        if self.config.min_sequences_per_tree > 0:
            to_evict = self._filter_respecting_min_per_tree(to_evict)

        total_pruned = 0
        for req_id in to_evict:
            try:
                self._cache.stop_request(req_id)  # ArcticInference handles tree cleanup
                self._request_batch.pop(req_id, None)
                total_pruned += 1
            except Exception as e:
                logger.debug(f"Failed to prune {req_id}: {e}")

        return {
            "suffix_tree/sequences_pruned": total_pruned,
            "suffix_tree/alive_requests": len(active_requests) - total_pruned,
        }

    def _filter_respecting_min_per_tree(self, to_evict: List[str]) -> List[str]:
        """Filter eviction list to respect min_sequences_per_tree."""
        # Use ArcticInference's public API to count requests per tree
        tree_request_count = self._cache.get_requests_per_tree()

        # Only evict if tree will still have enough sequences
        filtered = []
        for req_id in to_evict:
            tree_idx = self._cache.get_tree_idx_for_request(req_id)
            if tree_idx is not None:
                count = tree_request_count.get(tree_idx, 0)
                if count > self.config.min_sequences_per_tree:
                    filtered.append(req_id)
                    tree_request_count[tree_idx] = count - 1

        return filtered
```

**Pros**:
- Leverages existing ArcticInference tracking (no changes needed there)
- Full tree association available via `cache._req_to_tree_idx`
- Can enforce `min_sequences_per_tree` using existing data
- Low overhead (~100 bytes per req_id for age tracking)

**Memory overhead**: ~10MB for 100K sequences (negligible)

##### ArcticInference Public API (Added)

The following public methods have been added to `ParallelSuffixDecodingCache` for cleaner access:

```python
# In ParallelSuffixDecodingCache (ArcticInference) - NOW AVAILABLE
def get_tree_idx_for_request(self, req_id: Hashable) -> Optional[int]:
    """Get the tree index for a request, or None if not active."""
    return self._req_to_tree_idx.get(req_id)

def get_requests_per_tree(self) -> Dict[int, int]:
    """Get count of active requests per tree index."""
    counts: Dict[int, int] = {}
    for tree_idx in self._req_to_tree_idx.values():
        counts[tree_idx] = counts.get(tree_idx, 0) + 1
    return counts
```

These methods enable the `SuffixTreeManager` to access tree associations without accessing private attributes.

#### Why Age-Based is Best for RL

| Training Phase | Model Behavior | Old Sequences |
|----------------|----------------|---------------|
| Early (random) | High entropy, diverse | Mostly wrong patterns |
| Mid (learning) | Policy improving | Outdated, pre-improvement |
| Late (converged) | Stable, optimal | May conflict with current |

**Key principle**: The suffix tree should reflect what the *current* model generates, not what older model versions generated.

#### Tuning `sequence_max_age_batches`

- **Too small** (e.g., 10): Trees stay small, less speculation benefit
- **Too large** (e.g., 1000): Stale patterns pollute speculation
- **Recommended**: 50-200 batches (roughly 1-2 epochs of unique prompts)

```python
# Rule of thumb: evict after ~1 epoch
sequence_max_age_batches = num_unique_prompts / batch_size
```

---

### Alternative: Tree-Level LRU Eviction (Simpler)

If sequence-level eviction is too complex, fall back to tree-level:

```python
@dataclass
class SuffixTreeManagerConfig:
    max_trees: int = 10000          # Maximum trees to keep
    eviction_policy: str = "lru"    # "lru", "oldest", "smallest"

class SuffixTreeManager:
    def _maybe_evict_trees(self):
        """Evict entire trees if over limit."""
        num_trees = self._cache.get_stats().get("num_trees_in_forest", 0)
        if num_trees <= self.config.max_trees:
            return

        to_evict = num_trees - self.config.max_trees
        eviction_candidates = sorted(
            self._tree_last_used.items(),
            key=lambda x: x[1]
        )[:to_evict]

        for tree_idx, _ in eviction_candidates:
            self._cache.remove_tree(tree_idx)
```

**Downside**: Loses all patterns for a prompt, even recent ones.

---

### Concern 2: Snapshot Serialization Memory Spike

**Location**: `verl/trainer/ppo/suffix_tree_manager.py:289-293`

**Problem**: `create_snapshot()` serializes ALL trees into memory simultaneously.

```python
def get_snapshot(self):
    return self._cache.create_snapshot(include_hash_mapping=True)
```

**Impact**:
- With 1000 trees × 100KB average = 100MB allocation spike
- Can cause OOM on memory-constrained systems
- Happens every training step

**Proposed Solutions**:

#### Option A: Streaming Snapshot (Preferred)
```python
def get_snapshot_streaming(self) -> Iterator[Tuple[int, bytes]]:
    """Yield snapshots one tree at a time."""
    for tree_idx in self._cache.get_tree_indices():
        yield tree_idx, self._cache.serialize_tree(tree_idx)
```

#### Option B: Top-K Snapshot
```python
def get_snapshot(self, max_trees: int = 500) -> ...:
    """Only snapshot most frequently used trees."""
    top_k_trees = sorted(
        self._tree_usage.items(),
        key=lambda x: x[1],
        reverse=True
    )[:max_trees]

    return self._cache.create_snapshot(
        tree_indices=[t[0] for t in top_k_trees]
    )
```

#### Option C: Compressed Snapshots
```python
import zlib

def get_snapshot(self, compress: bool = True) -> ...:
    snapshots, hash_mapping = self._cache.create_snapshot(...)
    if compress:
        snapshots = [
            (idx, zlib.compress(data, level=1))  # Fast compression
            for idx, data in snapshots
        ]
    return snapshots, hash_mapping
```

---

### Concern 3: Snapshot Transfer Overhead

**Location**: `verl/trainer/ppo/ray_trainer.py:1188-1192`

**Problem**: Full snapshot transferred to ALL workers on EVERY step:

```python
with marked_timer("push_suffix_snapshot", timing_raw):
    snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
    if snapshots and not self.async_rollout_mode:
        self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)
```

**Impact**:
- With N workers and M bytes: N × M bytes in Ray object store
- Network transfer: M bytes × N times
- Blocking operation delays rollout

**Proposed Solutions**:

#### Option A: Incremental Snapshots (USER (chi-chih chang): I don't like this option)
Only send trees that changed since last snapshot:

```python
class SuffixTreeManager:
    def __init__(self, ...):
        self._dirty_trees: Set[int] = set()
        self._last_snapshot_batch: int = 0

    def update_from_rollout(self, batch):
        # ... existing code ...
        # Track which trees were modified
        for tree_idx in modified_trees:
            self._dirty_trees.add(tree_idx)

    def get_incremental_snapshot(self):
        """Only return trees modified since last call."""
        if not self._dirty_trees:
            return [], {}

        snapshots = self._cache.create_snapshot(
            tree_indices=list(self._dirty_trees)
        )
        # Get hash mappings only for dirty trees
        hash_mapping = {
            h: idx for h, idx in self._full_hash_mapping.items()
            if idx in self._dirty_trees
        }

        self._dirty_trees.clear()
        return snapshots, hash_mapping
```

#### Option B: Shared Memory (USER (chi-chih chang): I don't like this option)
Use Ray's shared memory for large objects:

```python
import ray

def push_snapshot_shared(self, snapshots, hash_mapping):
    """Push snapshot via shared memory."""
    # Put in object store once
    snapshot_ref = ray.put(snapshots)
    hash_ref = ray.put(hash_mapping)

    # Workers get reference (zero-copy)
    self.actor_rollout_wg.load_suffix_snapshot_ref(snapshot_ref, hash_ref)
```

#### Option C: Async Loading
Load snapshots while previous step runs:

```python
# In training loop
if pending_snapshot_future is not None:
    ray.get(pending_snapshot_future)  # Wait for previous

# Start loading asynchronously
snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
pending_snapshot_future = self.actor_rollout_wg.load_suffix_snapshot.remote(
    snapshots, hash_mapping
)

# Continue with generation (loading happens in parallel)
```

---

### Concern 4: Request ID Memory Leak

**Location**: `verl/trainer/ppo/suffix_tree_manager.py:203`

**Problem**: Request IDs accumulate without cleanup:

```python
req_id = f"train_{self._batch_counter}_{i}"
self._cache.start_request(req_id, prompt_array)
# ... add tokens ...
# stop_request() is NEVER called
```

**Impact**:
- Request metadata accumulates in cache (`_req_to_tree_idx`, `_req_to_seq_id`)
- Memory grows with total requests processed
- May slow down cache operations

**Solution**: This is **automatically solved by age-based sequence eviction** (Concern 1).

When `stop_request(req_id)` is called during pruning:
1. ArcticInference removes from `_req_to_tree_idx` and `_req_to_seq_id`
2. If no other requests reference the tree, the tree is removed (unless protected)
3. Hash mapping is cleaned up

No additional implementation needed beyond what's in Concern 1.

---

## Performance Optimizations

### Optimization 1: Lazy Metrics Collection

**Location**: `verl/workers/rollout/vllm_rollout/spec_decode_metrics.py`

**Current**: `get_metrics()` called twice per rollout (start + end).

**Proposed**: Cache metrics with invalidation:

```python
class SpecDecodeMetricsTracker:
    def __init__(self, engine):
        self._cached_snapshot: Optional[SpecDecodeSnapshot] = None
        self._cache_valid = False

    def _get_snapshot(self) -> SpecDecodeSnapshot:
        if self._cache_valid and self._cached_snapshot:
            return self._cached_snapshot

        # Fetch fresh metrics
        snapshot = self._fetch_metrics_from_engine()
        self._cached_snapshot = snapshot
        self._cache_valid = True
        return snapshot

    def start_rollout(self):
        self._cache_valid = False  # Invalidate on new rollout
        self._last_snapshot = self._get_snapshot()
```

---

### Optimization 2: Reduce Per-Position Metrics

**Location**: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:497-499`

**Current**: Creates one wandb metric per position:
```python
for i, rate in enumerate(rollout_stats.per_position_rates):
    spec_decode_metrics[f"spec_decode/acceptance_rate_pos_{i}"] = rate
```

**Problem**: With 10+ speculative tokens, creates many metrics.

**Proposed**: Aggregate:
```python
if rollout_stats.per_position_rates:
    rates = rollout_stats.per_position_rates
    spec_decode_metrics["spec_decode/acceptance_rate_pos_0"] = rates[0]
    spec_decode_metrics["spec_decode/acceptance_rate_pos_last"] = rates[-1]
    spec_decode_metrics["spec_decode/acceptance_rate_pos_mean"] = sum(rates) / len(rates)
    # Only log first 3 positions individually
    for i in range(min(3, len(rates))):
        spec_decode_metrics[f"spec_decode/acceptance_rate_pos_{i}"] = rates[i]
```

---

### Optimization 3: Tree Pruning by Usage

Add metrics to track tree effectiveness:

```python
class SuffixTreeManager:
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get per-tree statistics for pruning decisions."""
        return {
            "tree_sizes": self._cache.get_tree_sizes(),
            "tree_hit_counts": dict(self._tree_usage),
            "tree_ages": {
                idx: self._batch_counter - last_used
                for idx, last_used in self._tree_last_used.items()
            }
        }

    def prune_ineffective_trees(
        self,
        max_age_batches: int = 100,
        min_usage: int = 5
    ):
        """Remove trees that aren't providing value."""
        stats = self.get_tree_stats()

        to_prune = []
        for tree_idx, age in stats["tree_ages"].items():
            usage = stats["tree_hit_counts"].get(tree_idx, 0)
            if age > max_age_batches and usage < min_usage:
                to_prune.append(tree_idx)

        for tree_idx in to_prune:
            self._cache.remove_tree(tree_idx)

        return len(to_prune)
```

---

## Implementation Priority

| Optimization | Memory Impact | Performance Impact | Effort | Priority |
|--------------|--------------|-------------------|--------|----------|
| Age-based sequence eviction | **High** | Low | **Easy** | **P0** |
| Compressed snapshots | Medium | Medium | Easy | P1 |
| Async loading | Low | High | Medium | P2 |
| Tree-level LRU (fallback) | High | Low | Easy | P3 |
| Lazy metrics | Low | Low | Easy | P3 |
| Reduce per-pos metrics | Low | Low | Easy | P3 |

### Why Age-Based Eviction is P0 and Easy

1. **RL-specific**: Old sequences from previous policy versions hurt speculation accuracy
2. **Memory bounded**: Naturally limits growth based on training window
3. **Uses existing API**: `stop_request()` already removes sequences from trees
4. **Leverages existing tracking**: ArcticInference already tracks `req_id → tree_idx`
5. **Minimal code changes**: Only need to add `_request_batch` dict in `SuffixTreeManager`
6. **No ArcticInference changes required**: All necessary APIs already exist

---

## Metrics to Add

Track these metrics in wandb for monitoring:

```python
metrics = {
    # Memory metrics
    "suffix_tree/total_bytes": total_snapshot_bytes,
    "suffix_tree/avg_tree_size_bytes": total_bytes / num_trees,
    "suffix_tree/num_trees": num_trees,

    # Usage metrics
    "suffix_tree/trees_hit_this_batch": trees_with_hits,
    "suffix_tree/avg_tree_usage": mean(tree_hit_counts),

    # Transfer metrics
    "suffix_tree/snapshot_transfer_bytes": len(serialized_snapshot),
    "suffix_tree/snapshot_transfer_time_ms": transfer_time * 1000,

    # Effectiveness metrics
    "suffix_tree/cache_hit_rate": hits / (hits + misses),
}
```

---

## Related Documents

- [Code Review](../task_plans/suffix_tree_code_review.md) - Code quality findings
- [Suffix Tree VERL Integration](../task_plans/suffix_tree_verl_integration.md) - Original integration
- [Async Options](suffix_tree_async_options.md) - Async update patterns
