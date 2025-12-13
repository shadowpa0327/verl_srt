# Future: Async Suffix Tree Update Options

> **Status**: Future enhancement (test simple version first)
> **Prerequisite**: Validate basic synchronous integration works end-to-end

## Problem Statement

The current synchronous `update_from_rollout()` blocks the training loop:

```
gen → update_suffix(blocked) → reward → train → gen → ...
      ^^^^^^^^^^^^^^^^
      Potential bottleneck
```

## Option 1: ThreadPoolExecutor (Recommended First)

**Why it works**: `ParallelSuffixDecodingCache` is C++ and releases the GIL.

```python
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

class SuffixTreeManager:
    def __init__(self, config, tokenizer):
        # ... existing init ...
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_update: Optional[Future] = None

    def update_from_rollout_async(self, batch) -> Future:
        """Start async update, returns Future."""
        self._wait_for_pending()

        # Extract data before submitting (batch may be modified later)
        prompt_ids = batch.batch["prompts"].cpu().numpy()
        response_ids = batch.batch["responses"].cpu().numpy()

        self._pending_update = self._executor.submit(
            self._do_update, prompt_ids, response_ids
        )
        return self._pending_update

    def _do_update(self, prompt_ids, response_ids):
        """Actual update logic (runs in background thread)."""
        # ... existing update logic ...

    def _wait_for_pending(self):
        """Wait for pending update to complete."""
        if self._pending_update is not None:
            self._pending_update.result()
            self._pending_update = None

    def get_snapshot(self):
        """Get snapshot (waits for pending update first)."""
        self._wait_for_pending()
        return self._cache.create_snapshot(include_hash_mapping=True)
```

**Training loop changes**:
```python
# After rollout - fire and forget
if self.suffix_tree_manager.enabled:
    self.suffix_tree_manager.update_from_rollout_async(batch)

# ... reward, training (overlapped with suffix update) ...

# Before next rollout - get_snapshot() auto-waits
snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
```

**Timeline**:
```
gen → reward → train ──────────────────→ gen → ...
      └── update_suffix (background) ──┘
```

**Pros**:
- Zero serialization overhead (in-process)
- Simple implementation
- True parallelism (C++ releases GIL)

**Cons**:
- Shares driver process memory
- No fault isolation

---

## Option 2: Ray Actor (For Scale/Isolation)

Only needed if:
- Driver node is CPU-constrained
- Need memory isolation on different node
- Very large suffix forests (100M+ patterns)

```python
@ray.remote(num_cpus=2, memory=8*1024*1024*1024)
class SuffixTreeActor:
    def __init__(self, config_dict, tokenizer_path):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.manager = SuffixTreeManager(
            SuffixTreeManagerConfig(**config_dict),
            tokenizer
        )

    def update_from_rollout(self, prompt_ids, response_ids):
        """Accept numpy arrays to minimize serialization."""
        return self.manager.update_from_ids(prompt_ids, response_ids)

    def get_snapshot(self):
        return self.manager.get_snapshot()

    def save(self, path):
        self.manager.save(path)

    def load(self, path):
        return self.manager.load(path)
```

**Trainer integration**:
```python
class RayPPOTrainer:
    def _init_suffix_tree_actor(self):
        config_dict = {...}  # Serializable config
        self.suffix_tree_actor = SuffixTreeActor.remote(
            config_dict,
            self.config.model.path
        )
        self._pending_update_ref = None

    def fit(self):
        # Extract only needed data (minimize serialization)
        prompt_ids = batch.batch["prompts"].cpu().numpy()
        response_ids = batch.batch["responses"].cpu().numpy()

        # Async update
        self._pending_update_ref = self.suffix_tree_actor.update_from_rollout.remote(
            prompt_ids, response_ids
        )

        # ... training ...

        # Before next rollout
        if self._pending_update_ref:
            ray.get(self._pending_update_ref)
        snapshots, hash_mapping = ray.get(
            self.suffix_tree_actor.get_snapshot.remote()
        )
```

**Challenges**:
1. **Serialization overhead**: Batch data goes through Ray object store
2. **Coordination**: Need to manage actor lifecycle, handle failures
3. **Checkpoint sync**: Actor state must be coordinated with trainer checkpoints
4. **Latency**: Extra round-trips through object store

---

## Comparison

| Aspect | Synchronous | ThreadPool | Ray Actor |
|--------|-------------|------------|-----------|
| Complexity | Simple | Low | High |
| Serialization | None | None | Full batch |
| Overlap with training | No | Yes | Yes |
| Memory isolation | No | No | Yes |
| Different node | No | No | Yes |
| Fault isolation | No | No | Yes |
| Latency overhead | None | ~0 | Object store |

---

## Recommendation

1. **First**: Test synchronous version (current implementation)
2. **If bottleneck observed**: Add ThreadPoolExecutor async
3. **If memory pressure on driver**: Consider Ray Actor

---

## Metrics to Watch

Before optimizing, measure:
- `update_suffix_tree` timing in training logs
- Driver process memory usage
- Overall training throughput

Only optimize if `update_suffix_tree` is a significant fraction of step time.
