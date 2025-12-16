# Skill: Suffix Tree VERL Integration

**Type**: Implementation Reference
**Status**: Implemented
**Last Updated**: 2025-12-16

---

## Overview

This skill documents how suffix tree speculation is integrated into VERL's training loop for speculative decoding during rollout.

## Architecture

```
RayPPOTrainer (Driver)
├── SuffixTreeManager              # Wraps ParallelSuffixDecodingCache
│   └── ParallelSuffixDecodingCache (ArcticInference)
│
└── Training Loop
    1. Push snapshot to workers (before rollout)
    2. generate_sequences() with speculation
    3. Update trees from rollout results
    4. Save checkpoint (includes suffix tree state)

vLLMRollout Workers
└── load_suffix_snapshot()         # Receives snapshots from trainer
    └── inference_engine.load_snapshot(snapshots, hash_mapping)
```

## Key Files

| File | Purpose |
|------|---------|
| `verl/trainer/ppo/suffix_tree_manager.py` | SuffixTreeManager class |
| `verl/trainer/ppo/ray_trainer.py` | Training loop integration |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Worker snapshot loading |
| `verl/workers/fsdp_workers.py` | Worker dispatch |

## SuffixTreeManager API

```python
from verl.trainer.ppo.suffix_tree_manager import SuffixTreeManager, SuffixTreeManagerConfig

# Initialize
config = SuffixTreeManagerConfig(
    enable=True,
    max_tree_depth=64,
    hash_token_count=128,  # For tree sharing
    num_threads=-1,
)
manager = SuffixTreeManager(config, tokenizer)

# Update from rollout results
stats = manager.update_from_rollout(batch)  # batch: DataProto with responses

# Get snapshot for workers
snapshots, hash_mapping = manager.get_snapshot()

# Get selective snapshot (batch-specific trees only)
snapshots, hash_mapping = manager.get_selective_snapshot(hashes=batch_hashes)

# Checkpoint
manager.save("/path/to/checkpoint/suffix_tree")
manager.load("/path/to/checkpoint/suffix_tree")
```

## Training Loop Integration

```python
# In ray_trainer.py

# 1. Before rollout - push snapshot to workers
if self.suffix_tree_manager.enabled:
    batch_hashes = self._extract_batch_hashes(gen_batch)
    if batch_hashes:
        snapshots, hash_mapping = self.suffix_tree_manager.get_selective_snapshot(batch_hashes)
    else:
        snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()

    if snapshots:
        self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)

# 2. Run generation with speculation
output = self.actor_rollout_wg.generate_sequences(prompts)

# 3. After rollout - update trees
if self.suffix_tree_manager.enabled:
    suffix_stats = self.suffix_tree_manager.update_from_rollout(batch)
    metrics.update(suffix_stats)
```

## Worker-Side Loading

```python
# In vllm_rollout_spmd.py
async def load_suffix_snapshot(
    self,
    snapshots: List[Tuple[int, bytes]],
    hash_mapping: Dict[str, int],
) -> None:
    """Load suffix tree snapshot for speculative decoding."""
    if snapshots:
        self.inference_engine.load_snapshot(snapshots, hash_mapping)
```

## Configuration

```yaml
# In training config
suffix_tree:
  enable: true
  max_tree_depth: 64
  hash_token_count: 128
  num_threads: -1
  parallel_threshold: 4
```

## Metrics

| Metric | Description |
|--------|-------------|
| `suffix_tree/num_trees` | Total trees in forest |
| `suffix_tree/requests_started` | New requests started this batch |
| `suffix_tree/tokens_added` | Tokens added this batch |
| `suffix_tree/trees_transferred` | Trees sent to workers |
| `suffix_tree/transfer_bytes` | Bytes transferred |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mode | Local (no gRPC server) | Simpler for training |
| Tree sharing | Hash-based | Same prompt shares tree |
| Tokenization | `add_special_tokens=True` | Must match vLLM's BOS |
| State location | Driver process (CPU) | Trees accumulate centrally |
| Push timing | Before every rollout | Fresh speculation patterns |

## Related

- [`selective_snapshot_distribution.md`](selective_snapshot_distribution.md) - Batch-specific snapshots
- [`third_party/ArcticInference_srt/CLAUDE.md`](../../third_party/ArcticInference_srt/CLAUDE.md) - ArcticInference API
