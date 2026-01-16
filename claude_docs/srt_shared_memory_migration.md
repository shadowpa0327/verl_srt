# SRT Shared Memory Migration Plan

This document outlines the migration plan for SRT (Speculative Rollout with Tree-Structured Cache) from snapshot-based cache loading to shared-memory based SuffixCache, similar to SpecRL's approach.

## Background

### Current SRT Architecture (Snapshot-based)

```
┌──────────────────┐       serialize       ┌────────────────┐
│ SuffixTreeManager│ ────────────────────► │ vLLM Workers   │
│ (CPU, driver)    │   load_snapshot()     │ (deserialize)  │
└──────────────────┘                       └────────────────┘
```

- Uses `ParallelSuffixDecodingCache` from ArcticInference
- Trees are serialized and sent to workers via `load_snapshot()` API
- Each worker deserializes and holds its own copy

### Target Architecture (Shared Memory)

```
┌──────────────────┐       gRPC            ┌────────────────┐
│ SuffixCacheUpdater│ ──────────────────► │RolloutCacheServer│
│ (trainer)        │   UpdateCache()       │ (owns shm)     │
└──────────────────┘                       └───────┬────────┘
                                                   │ zero-copy
                                           ┌───────▼────────┐
                                           │ SuffixCache    │
                                           │ (vLLM workers) │
                                           └────────────────┘
```

In `/home/ubuntu/verl_srt/recipe/specRL`, it also contain the SuffixTree-based Rollout Cache for performing speculative decoding. Different to our snapshot pushing structure, the specRL allocated the shared memory on each servers. At verl side, it directly update the contents in shared memory and the cache owned in the vLLM workers side directly read through it. 


### Target Requirements

- migrate this styles of implementation as an other options into our srt Recipe. 
- If possible, I wish to have both style (e.g., snapshot-based and shared memory) supported and we can toggled over them. 

## Implementation (Completed)

The shared memory migration has been implemented with a configuration toggle between both approaches.

### New Files Created

| File | Purpose |
|------|---------|
| `recipe/srt/shared_memory_cache_manager.py` | Wraps SpecRL's CacheManager for SRT trainer |
| `recipe/srt/srt_plugin/patches/shm_patches.py` | Entry point for shared memory mode patches |
| `recipe/srt/srt_plugin/proposers/suffix_decoding_shm.py` | Proposer wrapping SuffixCache for speculation |

### Modified Files

| File | Changes |
|------|---------|
| `recipe/srt/ray_trainer.py` | Dual-mode init, mode routing, metrics, cleanup |
| `recipe/srt/vllm_server.py` | Cache mode detection, conditional patching |
| `recipe/srt/srt_plugin/patches/runner_patches.py` | Dual-mode GPUModelRunner patches, execute_model hooks |
| `recipe/srt/srt_plugin/config.py` | Added `cache_mode` field to SRTSuffixConfig |

### Configuration

```yaml
actor_rollout_ref:
  rollout:
    enable_srt: true
    srt_cache_mode: "shared_memory"  # or "snapshot" (default)
    srt_max_tree_depth: 64
    srt_hash_token_count: 128
    srt_num_speculative_tokens: 24
    srt_shared_memory:
      port: 6378
      memory_size_gb: 100
```

### Architecture Comparison

#### Snapshot Mode (Default)
```
Before rollout:
  Trainer: get_snapshot() → serialize trees
  Trainer: collective_rpc → push to workers
  Workers: load_suffix_snapshot() → deserialize

During rollout:
  Workers: ParallelSuffixDecodingProposer.propose() → use local cache

After rollout:
  Trainer: update_from_rollout() → add to local trees
```

#### Shared Memory Mode
```
Before rollout:
  (nothing - trees already in shared memory from previous batch)

During rollout:
  Workers: SuffixCache.fetch_responses_by_prompts_batch() → read from shm
  Workers: SuffixCache.speculate() → get drafts from shm
  Workers: SuffixCache.evict_responses() → cleanup on finish

After rollout:
  Trainer: update_from_rollout() → async gRPC to RolloutCacheServer
  Server: builds trees in shared memory for next batch
```

### Key Differences

| Aspect | Snapshot (SRT Default) | Shared Memory (SpecRL) |
|--------|------------------------|------------------------|
| Patch mechanism | `worker_extension_cls` | Unified `runner_patches.py` |
| Cache class | `ParallelSuffixDecodingCache` | `SuffixCache` (C++ shared mem) |
| Loading | `load_snapshot()` before rollout | No loading, direct shm access |
| Speculation | Via `Proposer.propose()` | Via `Proposer.propose()` with lifecycle hooks |
| Update timing | After rollout (for next) | After rollout (async gRPC) |
| Serialization | Yes (pickle-like) | No (zero-copy) |

### GPUModelRunner Lifecycle Hooks (Shared Memory Mode)

The `runner_patches.py` module extends GPUModelRunner with lifecycle hooks following SpecRL patterns:

```
┌─────────────────────────────────────────────────────────────┐
│ execute_model(scheduler_output)                             │
├─────────────────────────────────────────────────────────────┤
│ 1. CLEANUP FINISHED                                         │
│    for req_id in scheduler_output.finished_req_ids:         │
│        _suffix_cache.evict_responses(req_id)                │
│                                                             │
│ 2. ASYNC FETCH NEW REQUESTS                                 │
│    if scheduler_output.scheduled_new_reqs:                  │
│        future = _cache_updater.submit(                      │
│            fetch_responses_by_prompts_batch(...)            │
│        )                                                    │
│                                                             │
│ 3. RUN MODEL (original execute_model)                       │
│    → fetch happens in background via ThreadPoolExecutor     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ propose_draft_token_ids(...)                                │
├─────────────────────────────────────────────────────────────┤
│ 1. WAIT FOR FETCH (shared_memory mode only)                 │
│    _fetch_future.result()                                   │
│                                                             │
│ 2. UPDATE SPEC_LEN (shared_memory mode only)                │
│    for each sampled token:                                  │
│        _suffix_cache.update_spec_len(req_id, len(tokens))   │
│                                                             │
│ 3. PROPOSE DRAFTS                                           │
│    drafter.propose(input_batch, sampled_token_ids)          │
│    → SharedMemorySuffixDecodingProposer.speculate()         │
└─────────────────────────────────────────────────────────────┘
```

Key implementation details:
- `ThreadPoolExecutor(max_workers=1)` for async cache fetches
- Fetch submitted in `execute_model()`, awaited in `propose_draft_token_ids()`
- `update_spec_len()` called for cache coherency before speculation
- Fallback to snapshot mode if SuffixCache import fails

### Metrics

#### Snapshot Mode
- `timing/push_suffix_snapshot` - Time to serialize and push
- `suffix_tree/trees_transferred` - Trees sent per batch
- `suffix_tree/transfer_bytes` - Bytes transferred

#### Shared Memory Mode
- `timing/update_cache_shm` - Time to send gRPC update
- `shm_cache/update_submitted` - Updates sent
- `shm_cache/batch_size` - Prompts per update
- `shm_cache/response_tokens` - Tokens per update
- `shm_cache/num_servers` - Active cache servers
- `shm_cache/total_updates` - Cumulative updates
- `shm_cache/pending_futures` - Async updates in flight

### Usage Notes

1. **Shared memory mode requires `specrl` package** - The SuffixCache C++ extension must be installed

2. **Environment variable** - When shared_memory mode is configured, the trainer sets `SRT_CACHE_MODE=shared_memory` for worker processes to detect

3. **Cache servers** - In shared_memory mode, one `RolloutCacheServer` is deployed per GPU node via Ray actors

4. **Runahead support** - Both modes fully support runahead generation with `_update_suffix_trees_from_secondary()`

## Test Support

The `test_runahead_suffix_effectiveness.py` test file has been updated to support both cache modes.

### Test Usage

```bash
# Run with snapshot mode (default, existing behavior)
python tests/workers/rollout/rollout_vllm/test_runahead_suffix_effectiveness.py --cache-mode snapshot

# Run with shared memory mode
python tests/workers/rollout/rollout_vllm/test_runahead_suffix_effectiveness.py --cache-mode shared_memory --shm-port 6378

# Compare both modes
for mode in snapshot shared_memory; do
    python tests/workers/rollout/rollout_vllm/test_runahead_suffix_effectiveness.py \
        --cache-mode $mode \
        --output "results_${mode}.json"
done
```

### Test Architecture

The test uses a `CacheManagerInterface` abstraction to support both modes:

```
CacheManagerInterface (abstract)
├── SnapshotCacheManager       # Wraps SuffixTreeManager
│   ├── add_sequence()         # Adds to local trees
│   ├── push_to_workers()      # Serializes and pushes snapshot
│   └── get_metrics()          # Returns tree stats
│
└── SharedMemoryCacheManagerTest  # Wraps SpecRL infrastructure
    ├── add_sequence()         # Queues for batch update
    ├── flush_pending_sequences()  # Sends gRPC to cache server
    ├── push_to_workers()      # No-op (shm access direct)
    └── get_metrics()          # Returns shm stats
```

### Key Differences in Test Flow

| Aspect | Snapshot Mode | Shared Memory Mode |
|--------|--------------|-------------------|
| Cache init | `SuffixTreeManager` | Deploy `CacheWorker` actor |
| Add sequence | `add_sequence()` immediately | Queue, then `flush_pending_sequences()` |
| Push to workers | `load_suffix_snapshot()` | No-op (workers read from shm) |
| Cleanup | None needed | Shutdown cache server actor |
| Worker patches | `worker_extension_cls` | `SRT_CACHE_MODE` env var |

## Related Documentation

- [SpecRL Cache Implementation Analysis](../recipe/specRL/SPEC_RL_CACHE_IMPL_ANALYSIS.md) - Detailed C++ implementation analysis
- [SpecRL Architecture](../recipe/specRL/) - Reference implementation
