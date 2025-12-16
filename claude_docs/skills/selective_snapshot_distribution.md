# Skill: Selective Snapshot Distribution

**Type**: Implementation Reference
**Status**: Implemented
**Last Updated**: 2025-12-16

---

## Overview

Selective snapshot distribution optimizes suffix tree transfer by only sending trees needed for the current batch, rather than the entire forest.

## Problem Solved

```
Before (full snapshot):
  Controller → get_snapshot() → ALL 100 trees
  Workers: Load all 100 trees, use only 10

After (selective snapshot):
  Controller → get_selective_snapshot(hashes) → 10 trees
  Workers: Load exactly the 10 trees they need
```

## Key Components

### 1. C++ Selective Snapshot API

**File**: `third_party/ArcticInference_srt/csrc/suffix_decoding/suffix_forest.cc`

```cpp
std::vector<std::pair<int, std::vector<uint8_t>>> SuffixForest::create_selective_snapshot(
    const std::vector<int>& tree_indices
) const {
    std::vector<std::pair<int, std::vector<uint8_t>>> results;
    results.reserve(tree_indices.size());
    for (int tree_idx : tree_indices) {
        auto it = _trees.find(tree_idx);
        if (it != _trees.end() && it->second) {
            auto lock = get_read_lock(tree_idx);
            results.emplace_back(tree_idx, it->second->create_snapshot());
        }
    }
    return results;
}
```

### 2. Python Wrapper

**File**: `third_party/ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py`

```python
def create_selective_snapshot(
    self,
    tree_indices: List[int],
    include_hash_mapping: bool = False,
) -> Union[List[Tuple[int, bytes]], Tuple[List[Tuple[int, bytes]], Dict[str, int]]]:
    """Create snapshot of specific trees only."""
    snapshots = self._forest.create_selective_snapshot(tree_indices)
    if include_hash_mapping:
        tree_set = set(tree_indices)
        filtered_mapping = {h: idx for h, idx in self._hash_to_tree_idx.items() if idx in tree_set}
        return snapshots, filtered_mapping
    return snapshots
```

### 3. SuffixTreeManager API

**File**: `verl/trainer/ppo/suffix_tree_manager.py`

```python
def get_selective_snapshot(
    self,
    hashes: List[str],
) -> Tuple[List[Tuple[int, bytes]], Dict[str, int]]:
    """Get snapshots for specific prompt hashes only."""
    # Map hashes to tree indices
    tree_indices = []
    hash_mapping = {}
    for prompt_hash in hashes:
        if prompt_hash in self._cache._hash_to_tree_idx:
            tree_idx = self._cache._hash_to_tree_idx[prompt_hash]
            if tree_idx not in tree_indices:
                tree_indices.append(tree_idx)
            hash_mapping[prompt_hash] = tree_idx

    if not tree_indices:
        return [], {}

    snapshots = self._cache.create_selective_snapshot(tree_indices)
    return snapshots, hash_mapping

def extract_batch_hashes(
    self,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> List[str]:
    """Extract unique prompt hashes from batch."""
    # Computes hash for each prompt in batch
    # Returns deduplicated list of hashes
```

## Usage

```python
# In ray_trainer.py

# Extract hashes from batch
batch_hashes = self._extract_batch_hashes(gen_batch)

if batch_hashes:
    # Selective: only trees for this batch
    snapshots, hash_mapping = self.suffix_tree_manager.get_selective_snapshot(batch_hashes)
else:
    # Fallback: full snapshot
    snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()

# Push to workers
self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)

# Log metrics
metrics["suffix_tree/trees_transferred"] = len(snapshots)
metrics["suffix_tree/transfer_bytes"] = sum(len(s[1]) for s in snapshots)
```

## Metrics

| Metric | Description |
|--------|-------------|
| `suffix_tree/trees_transferred` | Number of trees sent (should be << total) |
| `suffix_tree/transfer_bytes` | Total bytes transferred |
| `suffix_tree/total_trees` | Total trees in forest (for comparison) |

## Performance Impact

| Scenario | Full Snapshot | Selective |
|----------|---------------|-----------|
| 100 trees, batch needs 10 | 100 trees | 10 trees |
| Transfer savings | - | 90% reduction |

## Limitations

Current implementation sends same snapshot to ALL workers. For per-worker optimization (each worker gets only its DP partition's trees), see:
- [`../task_plans/per_worker_snapshot_analysis.md`](../task_plans/per_worker_snapshot_analysis.md)

## Related

- [`suffix_tree_verl_integration.md`](suffix_tree_verl_integration.md) - Base integration
- [`../future/suffix_tree_memory_optimizations.md`](../future/suffix_tree_memory_optimizations.md) - Memory optimizations
