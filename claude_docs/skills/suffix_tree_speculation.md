# Skill: Suffix Tree Speculation System

**Type**: Implementation Reference
**Status**: Implemented
**Last Updated**: 2025-12-21

---

## Overview

This document covers the complete suffix tree speculation system, from VERL training integration to vLLM proposer internals.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ VERL Training (Driver)                                                       │
│                                                                              │
│  RayPPOTrainer                                                               │
│  ├── SuffixTreeManager              # Wraps ParallelSuffixDecodingCache      │
│  │   └── ParallelSuffixDecodingCache (ArcticInference)                       │
│  │                                                                           │
│  └── Training Loop                                                           │
│      1. Push snapshot to workers (before rollout)                            │
│      2. generate_sequences() with speculation                                │
│      3. Update trees from rollout results                                    │
│      4. Save checkpoint (includes suffix tree state)                         │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ snapshots via Ray
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ vLLM Rollout Workers                                                         │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ ParallelSuffixDecodingProposer                                        │  │
│  │ verl/workers/rollout/vllm_rollout/patches/proposers/                  │  │
│  │                                                                       │  │
│  │  - Receives InputBatch from vLLM scheduler                            │  │
│  │  - Manages request lifecycle (start/stop)                             │  │
│  │  - Calls suffix_cache.propose_from_batch() for speculation            │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ uses                                      │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ArcticInference (Suffix Tree Library)                                        │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ ParallelSuffixDecodingCache                              [Python]     │  │
│  │ third_party/ArcticInference_srt/arctic_inference/suffix_decoding/     │  │
│  │                                                                       │  │
│  │  - Maps request_id → tree_index                                       │  │
│  │  - Handles hash-based tree sharing                                    │  │
│  │  - Wraps C++ SuffixForest                                             │  │
│  │  - propose_from_batch(): zero-copy batch speculation                  │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ wraps                                     │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ SuffixForest                                             [C++]        │  │
│  │ third_party/ArcticInference_srt/csrc/suffix_decoding/                 │  │
│  │                                                                       │  │
│  │  - Manages collection of SuffixTree instances                         │  │
│  │  - propose_from_batch(): OpenMP-parallelized batch processing         │  │
│  │  - batch_speculate() / batch_extend() for token operations            │  │
│  └───────────────────────────────┬───────────────────────────────────────┘  │
│                                  │ contains                                  │
│                                  ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ SuffixTree                                               [C++]        │  │
│  │ third_party/ArcticInference_srt/csrc/suffix_decoding/                 │  │
│  │                                                                       │  │
│  │  - Single suffix tree data structure                                  │  │
│  │  - extend(): add tokens to tree                                       │  │
│  │  - speculate(): find matching patterns, return drafts                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Python ↔ C++ Bindings: bindings.cc (nanobind)                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| **VERL Driver** | |
| `verl/trainer/ppo/suffix_tree_manager.py` | SuffixTreeManager class |
| `verl/trainer/ppo/ray_trainer.py` | Training loop integration |
| `verl/workers/fsdp_workers.py` | Worker dispatch |
| **vLLM Workers** | |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Worker snapshot loading |
| `verl/workers/rollout/vllm_rollout/patches/proposers/suffix_decoding_parallel.py` | Proposer |
| **ArcticInference** | |
| `third_party/ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py` | Python cache wrapper |
| `third_party/ArcticInference_srt/csrc/suffix_decoding/suffix_forest.{h,cc}` | C++ forest |
| `third_party/ArcticInference_srt/csrc/suffix_decoding/suffix_tree.{h,cc}` | C++ tree |
| `third_party/ArcticInference_srt/csrc/suffix_decoding/bindings.cc` | nanobind bindings |

---

## VERL Integration

### SuffixTreeManager API

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

### Training Loop Integration

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

### Worker-Side Loading

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

---

## Proposer Data Flow

The proposer uses `propose_from_batch()` for efficient zero-copy batch processing:

```
InputBatch (from vLLM)
│
│  token_ids_cpu: np.ndarray[batch, max_seq_len]
│  num_tokens_no_spec: np.ndarray[batch]
│  num_prompt_tokens: np.ndarray[batch]
│  req_ids: List[str]
│
▼
┌────────────────────────────────────────────────────────────────┐
│ Proposer.propose()                                  [Python]   │
│                                                                │
│  # Minimal Python: just prepare indices                        │
│  tree_indices = precompute_tree_indices(req_ids)               │
│                                                                │
│  drafts = suffix_cache.propose_from_batch(                     │
│      token_ids_cpu,        # Pass 2D array directly            │
│      num_tokens,           # Pass metadata arrays              │
│      tree_indices,         # Pre-computed mapping              │
│      ...                                                       │
│  )                                                             │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ SuffixForest::propose_from_batch()                 [C++]       │
│                                                                │
│  #pragma omp parallel for              ◄── ALL WORK HERE       │
│  for (i = 0; i < batch_size; i++) {                            │
│      // Direct pointer arithmetic - no copy                    │
│      int32_t* context = &token_ids[i * stride + start];        │
│      trees[i]->extend(sampled_token);                          │
│      results[i] = trees[i]->speculate(context, ...);           │
│  }                                                             │
└────────────────────────────────────────────────────────────────┘
```

---

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
| Batch processing | `propose_from_batch` | Zero-copy, OpenMP parallelized |

## Related

- [`selective_snapshot_distribution.md`](selective_snapshot_distribution.md) - Batch-specific snapshots
- [`../../third_party/ArcticInference_srt/CLAUDE.md`](../../third_party/ArcticInference_srt/CLAUDE.md) - ArcticInference API
