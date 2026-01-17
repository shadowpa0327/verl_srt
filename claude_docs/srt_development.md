# SRT Development Guide

## Overview

**Speculative Rollout with Tree-Structured Cache (SRT)** accelerates RL training rollouts by 1.5-2x through:
1. **Suffix Tree Caching** - Store previously generated token sequences per prompt
2. **Speculative Decoding** - Use cached sequences as a model-free draft
3. **Runahead** - Fill GPU bubbles with speculative pre-generation of future batches

---

## Feature Checklist

### Core SRT Features

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| Suffix tree data structure | ✅ Done | `srt_plugin/suffix_cache/_C.so` | C++ SuffixTree/SuffixForest |
| Hash-based tree sharing | ✅ Done | `parallel_cache.py` | Same prompt → same tree |
| Parallel batch speculation | ✅ Done | `parallel_cache.py` | OpenMP parallelization |
| Snapshot serialization | ✅ Done | `parallel_cache.py` | `create_snapshot()` / `load_snapshot()` |
| Selective snapshots | ✅ Done | `parallel_cache.py` | Only trees for current batch |
| Memory estimation | ✅ Done | `parallel_cache.py` | `estimate_memory()` |

### Cache Modes

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| **Snapshot Mode** | ✅ Done | `suffix_tree_manager.py` | Default, simpler |
| - Trainer-side tree management | ✅ Done | `suffix_tree_manager.py` | `SuffixTreeManager` |
| - Worker snapshot loading | ✅ Done | `worker_extension.py` | `load_suffix_snapshot()` |
| - vLLM proposer | ✅ Done | `suffix_decoding_parallel.py` | `SuffixDecodingParallelProposer` |
| **Shared Memory Mode** | ✅ Done | `shared_memory_cache_manager.py` | Zero-copy, multi-node |
| - Cache server deployment | ✅ Done | `shared_memory_cache_manager.py` | Per-node `CacheWorker` |
| - gRPC cache updates | ✅ Done | `shared_memory_cache_manager.py` | Async updates |
| - IPv4/IPv6 address handling | ✅ Done | `shared_memory_cache_manager.py` | `_get_routable_ip()` |
| - vLLM proposer | ✅ Done | `suffix_decoding_shm.py` | `SharedMemorySuffixDecodingProposer` |

### Trainer Integration

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| SRT config injection | ✅ Done | `ray_trainer.py` | `_inject_srt_engine_kwargs()` |
| Standard training loop | ✅ Done | `ray_trainer.py` | `_fit_standard()` |
| Runahead training loop | ✅ Done | `ray_trainer.py` | `_fit_runahead()` |
| Suffix tree update from rollout | ✅ Done | `ray_trainer.py` | `_update_suffix_trees()` |
| Secondary output feedback | ✅ Done | `ray_trainer.py` | `_update_suffix_trees_from_secondary()` |
| Checkpoint save/load | ✅ Done | `suffix_tree_manager.py` | `save()` / `load()` |

### Runahead Features

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| Runahead config | ✅ Done | `runahead/config.py` | `RunaheadConfig` |
| Primary/secondary batching | ✅ Done | `agent_loop.py` | `generate_sequences_with_runahead()` |
| Admission control | ✅ Done | `router.py` | Load threshold gating |
| Primary reservation | ✅ Done | `router.py` | Prevent startup race |
| Targeted abort | ✅ Done | `router.py` | Per-request abort |
| Secondary output collection | ✅ Done | `agent_loop.py` | Partial outputs on abort |

### Metrics & Observability

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| Suffix tree metrics | ✅ Done | `suffix_tree_manager.py` | `num_trees`, `tokens_added`, etc. |
| Spec decode metrics | ✅ Done | `spec_decode_metrics.py` | From vLLM Prometheus |
| Runahead metrics | ✅ Done | `ray_trainer.py` | `secondary_started`, `completed`, etc. |
| WandB logging | ✅ Done | `ray_trainer.py` | All metrics logged |

### Testing

| Feature | Status | File | Notes |
|---------|--------|------|-------|
| Unit tests | ⚠️ Partial | `test_suffix_tree_manager.py` | Needs more coverage |
| Integration test | ✅ Done | `test_runahead_suffix_effectiveness.py` | Both cache modes |
| E2E test (snapshot) | ✅ Done | `scripts/grpo/run_grpo_e2e_test.sh` | Standard mode, no runahead |
| E2E test (shm) | ✅ Done | `scripts/grpo/run_grpo_e2e_test_shm.sh` | Shared memory, no runahead |
| E2E test (runahead) | ⚠️ New | `scripts/grpo/run_grpo_e2e_test_runahead.sh` | Runahead + snapshot mode |
| E2E test (runahead+shm) | ⚠️ New | `scripts/grpo/run_grpo_e2e_test_runahead_shm.sh` | Runahead + shared memory |

---

## What's Missing / Future Work

| Feature | Priority | Notes |
|---------|----------|-------|
| Multi-epoch tree pruning | Medium | Trees grow unbounded; need eviction policy |
| Tree size limits | Medium | Cap memory usage per tree |
| Async snapshot loading | Low | Non-blocking worker snapshot load |
| Incremental snapshots | Low | Delta updates instead of full snapshot |
| Benchmark suite | Medium | Standardized perf comparison |
| Documentation | Low | User-facing docs |

---

## Architecture

### File Structure

```
recipe/srt/
├── ray_trainer.py                 # SRTRayPPOTrainer (main entry)
├── suffix_tree_manager.py         # Trainer-side tree management (snapshot mode)
├── shared_memory_cache_manager.py # Trainer-side cache management (shm mode)
├── vllm_server.py                 # SRT-enabled vLLM server classes
│
└── srt_plugin/                    # vLLM patches and extensions
    ├── suffix_cache/
    │   ├── _C.cpython*.so         # C++ extension (SuffixTree, SuffixForest)
    │   ├── parallel_cache.py      # ParallelSuffixDecodingCache
    │   ├── cache.py               # SuffixDecodingCache (single-threaded)
    │   └── hash_utils.py          # Prompt hashing
    │
    ├── proposers/
    │   ├── suffix_decoding_parallel.py  # Snapshot mode proposer
    │   └── suffix_decoding_shm.py       # Shared memory proposer
    │
    ├── worker_extension.py        # vLLM worker extension
    ├── config.py                  # SRT config classes
    └── patches/                   # vLLM monkey patches
```

### Two Cache Modes

| Aspect | Snapshot (Default) | Shared Memory |
|--------|-------------------|---------------|
| **Trainer** | `SuffixTreeManager` | `SharedMemoryCacheManager` |
| **Worker** | `ParallelSuffixDecodingCache` | `SuffixCache` (SpecRL) |
| **Transfer** | Ray serialization | gRPC |
| **Push Timing** | BEFORE rollout | AFTER rollout |
| **Best For** | Single-node, simpler | Multi-node, high-frequency |

---

## Data Flow

### Snapshot Mode

```
┌─────────────────────────────────────────────────────────────────┐
│ SRTRayPPOTrainer                                                │
│                                                                  │
│  SuffixTreeManager                                              │
│  └── ParallelSuffixDecodingCache                                │
│      └── SuffixForest [Tree₀, Tree₁, ..., Treeₙ]                │
│                 │                                                │
│         create_snapshot()                                        │
│                 │                                                │
│                 ▼                                                │
│    [(tree_idx, bytes), ...] + hash_mapping                      │
└─────────────────┼───────────────────────────────────────────────┘
                  │ via Ray
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ vLLM Worker                                                      │
│                                                                  │
│  SuffixTreeWorkerExtension.load_suffix_snapshot()               │
│      └── ParallelSuffixDecodingCache.load_snapshot()            │
│          └── SuffixForest.from_snapshots()                      │
│                 │                                                │
│         batch_speculate()                                        │
│                 │                                                │
│                 ▼                                                │
│    [SuffixDecodingDraft, ...]                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Shared Memory Mode

```
┌─────────────────────────────────────────────────────────────────┐
│ SRTRayPPOTrainer                                                │
│                                                                  │
│  SharedMemoryCacheManager                                       │
│      └── SuffixCacheUpdater (gRPC client)                       │
│                 │                                                │
│     update_response_cache()                                      │
└─────────────────┼───────────────────────────────────────────────┘
                  │ via gRPC
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│ GPU Node                                                         │
│                                                                  │
│  CacheWorker (Ray Actor)                                        │
│      └── RolloutCacheServer                                     │
│          └── SharedTreeMap (shared memory)                      │
│                 │                                                │
│          zero-copy read                                          │
│                 │                                                │
│                 ▼                                                │
│  vLLM Worker                                                     │
│      └── SharedMemorySuffixDecodingProposer                              │
│          └── SuffixCache                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Loops

### Standard Mode (No Runahead)

```python
for step in range(total_steps):
    batch = data_loader.next()

    # 1. Push suffix snapshots (snapshot mode only)
    if suffix_tree_manager.enabled:
        snapshots, hash_mapping = suffix_tree_manager.get_snapshot()
        rollout_manager.load_suffix_snapshot(snapshots, hash_mapping)

    # 2. Generate
    outputs = rollout_manager.generate_sequences(batch)

    # 3. Update suffix trees
    suffix_tree_manager.update_from_rollout(batch)

    # 4. Train
    train_step(batch, outputs)
```

### Runahead Mode (Sliding Window)

```python
batch_current = data_loader.next()
batch_next = data_loader.next()

for step in range(total_steps):
    # 1. Push snapshots
    snapshots, hash_mapping = suffix_tree_manager.get_snapshot()
    rollout_manager.load_suffix_snapshot(snapshots, hash_mapping)

    # 2. Generate with runahead
    result = rollout_manager.generate_sequences_with_runahead(
        primary_prompts=batch_current,
        secondary_prompts=batch_next,
    )

    # 3. Update from primary
    suffix_tree_manager.update_from_rollout(batch_current)

    # 4. KEY: Update from secondary (for next tick)
    _update_suffix_trees_from_secondary(result.secondary_outputs)

    # 5. Train on primary
    train_step(batch_current, result.primary_outputs)

    # 6. Slide window
    batch_current = batch_next
    batch_next = data_loader.next()
```

---

## Configuration

### Snapshot Mode (Default)

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    mode: async
    enable_srt: true
    srt_cache_mode: snapshot
    srt_max_tree_depth: 64
    srt_hash_token_count: 128
    srt_num_speculative_tokens: 24
```

### Shared Memory Mode

```yaml
actor_rollout_ref:
  rollout:
    enable_srt: true
    srt_cache_mode: shared_memory
    srt_shared_memory:
      port: 6378
      memory_size_gb: 100          # Shared memory segment size (default: 100GB)
      shared_memory_name: ""       # Custom name (default: "SUFFIX_CACHE")
      spec_start_len: 2            # Initial/minimum speculation length
      spec_max_len: 16             # Maximum speculation length
```

**Note:** `shared_memory_name` allows running multiple SRT jobs on the same node by giving each a unique name.

### Runahead

```yaml
trainer:
  enable_runahead: true
  runahead:
    load_threshold: 32
    max_queue_size: 256
    secondary_priority: 10
```

---

## Testing

```bash
# Unit tests
pytest tests/workers/rollout/test_suffix_tree_manager.py -v

# Integration test
python tests/workers/rollout/rollout_vllm/test_runahead_suffix_effectiveness.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --batch-size 8 \
    --cache-mode snapshot

# E2E tests (standard mode - no runahead)
bash recipe/srt/scripts/grpo/run_grpo_e2e_test.sh      # Snapshot mode
bash recipe/srt/scripts/grpo/run_grpo_e2e_test_shm.sh  # Shared memory mode

# E2E tests (runahead mode)
bash recipe/srt/scripts/grpo/run_grpo_e2e_test_runahead.sh      # Runahead + Snapshot mode
bash recipe/srt/scripts/grpo/run_grpo_e2e_test_runahead_shm.sh  # Runahead + Shared memory mode
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| `suffix_tree/num_trees` | Total trees in forest |
| `suffix_tree/tokens_added` | Tokens added this step |
| `suffix_tree/memory_mb` | Tree memory usage |
| `spec_decode/acceptance_rate` | Drafted tokens accepted |
| `spec_decode/tokens_per_step` | Tokens per forward pass |
| `runahead/primary_time_s` | Primary batch generation time |
| `runahead/secondary_started` | Secondaries admitted |
| `runahead/secondary_completed` | Secondaries finished |
| `runahead/secondary_aborted` | Secondaries aborted (primary done) |
| `runahead/secondary_rejected` | Secondaries rejected (load too high) |
| `runahead/completion_rate` | Fraction of secondaries completed |

---

## Key Design Decisions

1. **Hash-based tree sharing** - Same prompt shares one tree (important for GRPO/DAPO with n_samples > 1)
2. **Never call `stop_request()`** - Trees persist for future speculation
3. **Push BEFORE rollout (snapshot)** - Workers need trees before generating
4. **Push AFTER rollout (shm)** - Async gRPC doesn't block generation
5. **Secondary output feedback** - Runahead outputs populate cache for next tick


## Todos
- ~~For shm mode, change the spec len to be adjustable~~ ✅ Done - C++ and Python interfaces now support configurable `spec_start_len` and `spec_max_len`

## Recently Completed

### Configurable Shared Memory Parameters (Jan 2026)

The shared memory mode now supports configurable parameters via YAML config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shared_memory_name` | `"SUFFIX_CACHE"` | Unique name for the shm segment (enables multiple jobs per node) |
| `spec_start_len` | `2` | Initial/minimum speculation length |
| `spec_max_len` | `16` | Maximum speculation length |
| `memory_size_gb` | `100` | Shared memory segment size in GB |

**Data flow (config-based):**
1. `ray_trainer.py` reads config and adds to `speculative_config` dict with `srt_` prefix
2. `SRTSuffixConfig.extract_from_dict()` extracts params into dataclass fields
3. `SharedMemoryCacheManager` passes `shared_memory_name` to `CacheWorker` → `RolloutCacheServer`
4. `runner_patches.py` reads from `srt_config` and creates `SuffixCache(shared_memory_name, spec_start_len, spec_max_len)`

**CLI usage:**
```bash
# Pass shared memory config from command line
+actor_rollout_ref.rollout.srt_shared_memory.shared_memory_name=my_cache \
+actor_rollout_ref.rollout.srt_shared_memory.spec_start_len=4 \
+actor_rollout_ref.rollout.srt_shared_memory.spec_max_len=32 \
```

### Known Issues Fixed

1. **IPv6 Link-Local Address Issue** - Fixed by using `ray._private.services.get_node_ip_address()` for routable IPv4
2. **IPv4 Address with Brackets** - Fixed to only bracket IPv6 addresses in gRPC addresses
3. **Early SuffixCache Initialization** - Fixed with lazy initialization pattern