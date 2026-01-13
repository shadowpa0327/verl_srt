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

- One `RolloutCacheServer` per node owns shared memory segment
- `SuffixCacheUpdater` (trainer) sends updates via gRPC
- `SuffixCache` (vLLM workers) attaches to shared memory for zero-copy reads
- Trees built directly in shared memory using Ukkonen's algorithm

### Benefits of Migration

1. **Zero-copy reads**: Workers read directly from shared memory, no deserialization
2. **Memory efficiency**: Single copy of trees shared across all workers on a node
3. **Lower latency**: No serialization/transfer overhead for updates
4. **Proven implementation**: SpecRL's C++ library is production-tested

### Concerns

1. **500GB pre-allocation**: SpecRL hardcodes 500GB shared memory (virtual, not physical)
2. **C++ complexity**: Requires building and maintaining native extensions
3. **Multi-node coordination**: gRPC adds complexity for distributed setups

---

## Migration Phases

### Phase 1: C++ Infrastructure

| Task | Description | Effort | Notes |
|------|-------------|--------|-------|
| **Decision: Reuse vs Custom** | Decide whether to reuse SpecRL's `spec_rl_cache_impl` or write custom implementation | Low | SpecRL's is production-ready but has 500GB hardcoded |
| **Adapt Shared Memory Size** | If reusing, modify `rollout_cache_server.h` to make `SHARED_MEMORY_SIZE` configurable (env var or constructor param) | Low | Change constant to runtime config |
| **Build C++ Library** | Build `specrl_cache` wheel with pybind bindings for `SuffixCache`, `RolloutCacheServer`, `SuffixCacheUpdater` | Medium | Requires Boost, gRPC, protobuf |

#### Key Files (SpecRL Reference)

```
recipe/specRL/spec_rl_cache_impl/
├── specrl/
│   ├── suffix_cache/           # vLLM worker side
│   │   ├── suffix_tree.h/cc    # Ukkonen's suffix tree
│   │   ├── suffix_cache.h/cc   # Shared memory reader
│   │   ├── rollout_cache_server.h/cc  # gRPC server + shm owner
│   │   └── pybind.cc
│   ├── cache_updater/          # Trainer side
│   │   ├── suffix_cache_updater.h/cc  # gRPC client
│   │   └── pybind.cc
│   └── proto/
│       └── rollout-cache.proto
```

#### Configuration Change Example

```cpp
// Current (hardcoded in rollout_cache_server.h)
const unsigned long long SHARED_MEMORY_SIZE = 500ULL * 1024ULL * 1024ULL * 1024ULL;

// Proposed (configurable)
unsigned long long get_shared_memory_size() {
    const char* env = std::getenv("SRT_SHARED_MEMORY_SIZE_GB");
    return env ? std::stoull(env) * 1024ULL * 1024ULL * 1024ULL
               : 50ULL * 1024ULL * 1024ULL * 1024ULL;  // Default 50GB
}
```

---

### Phase 2: Server Component

| Task | Description | Files Affected |
|------|-------------|----------------|
| **Create Cache Server Launcher** | Add code to spawn `RolloutCacheServer` per node (one per machine) | New: `recipe/srt/cache_server.py` |
| **Integrate with vLLM Server** | Start cache server before vLLM server, ensure proper cleanup | `recipe/srt/vllm_server.py` |

#### Example: Cache Server Launcher

```python
# recipe/srt/cache_server.py
import os
from specrl_cache import RolloutCacheServer

class CacheServerManager:
    """Manages the RolloutCacheServer lifecycle."""

    def __init__(self, port: int = 50051):
        self.port = port
        self.server = None

    def start(self):
        """Start the cache server (should be called once per node)."""
        self.server = RolloutCacheServer()
        self.server.Initialize()
        self.server.Run(f"0.0.0.0:{self.port}")
        return self

    def stop(self):
        """Stop the cache server and cleanup shared memory."""
        if self.server:
            self.server.Shutdown()
            self.server = None
```

#### Integration with vLLM Server

```python
# In recipe/srt/vllm_server.py
class SRTvLLMServer:
    def __init__(self, ...):
        # Start cache server before vLLM
        self.cache_server = CacheServerManager(port=config.cache_server_port)
        self.cache_server.start()

        # Then start vLLM server
        self.vllm_server = ...

    def shutdown(self):
        self.vllm_server.shutdown()
        self.cache_server.stop()
```

---

### Phase 3: Writer Side (Trainer)

| Task | Description | Files Affected |
|------|-------------|----------------|
| **Replace SuffixTreeManager Backend** | Swap `ParallelSuffixDecodingCache` with `SuffixCacheUpdater` for writes | `recipe/srt/suffix_tree_manager.py` |
| **Update Training Loop** | Remove `load_suffix_snapshot()` calls; updater pushes via gRPC automatically | `recipe/srt/ray_trainer.py` |
| **Adapt update_from_rollout()** | Call `SuffixCacheUpdater.UpdateCache()` instead of building local trees | `recipe/srt/suffix_tree_manager.py` |

#### Example: Modified SuffixTreeManager

```python
# recipe/srt/suffix_tree_manager.py (after migration)
from specrl_cache import SuffixCacheUpdater

class SuffixTreeManager:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Replace ParallelSuffixDecodingCache with SuffixCacheUpdater
        self.updater = SuffixCacheUpdater()

        # Connect to all cache servers (one per node)
        for server_addr in config.cache_server_addresses:
            self.updater.AddServer(server_addr)

    def update_from_rollout(self, batch) -> dict:
        """Update cache with new rollout results."""
        updates = []
        for i in range(len(batch)):
            prompt_tokens = batch.input_ids[i]
            response_tokens = batch.responses[i]

            # Compute hash for tree lookup
            hash_tokens = prompt_tokens[:self.config.hash_token_count]
            prompt_hash = self._compute_hash(hash_tokens)

            updates.append({
                "hash": prompt_hash,
                "tokens": response_tokens.tolist(),
            })

        # Send updates via gRPC to all servers
        self.updater.UpdateCache(updates)

        return {"num_updates": len(updates)}

    # No more get_snapshot() or load_snapshot() needed!
```

#### Training Loop Changes

```python
# recipe/srt/ray_trainer.py (after migration)

# BEFORE (snapshot-based):
# if self.suffix_tree_manager.enabled:
#     snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
#     self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)

# AFTER (shared memory):
# Nothing needed before rollout - workers already have access via shared memory

# Generate sequences (workers read from shared memory automatically)
output = self.actor_rollout_wg.generate_sequences(prompts)

# Update cache (gRPC pushes to all servers)
if self.suffix_tree_manager.enabled:
    suffix_stats = self.suffix_tree_manager.update_from_rollout(batch)
```

---

### Phase 4: Reader Side (vLLM Workers)

| Task | Description | Files Affected |
|------|-------------|----------------|
| **Create SuffixCache-based Proposer** | New proposer that reads from shared memory via `SuffixCache` | `recipe/srt/vllm_plugin/proposers/suffix_decoding_shm.py` |
| **Remove Snapshot Loading** | Delete `worker_extension.py` or replace with no-op | `recipe/srt/vllm_plugin/worker_extension.py` |
| **Update vLLM Patches** | Modify patches to initialize `SuffixCache` reader | `recipe/srt/vllm_plugin/patches/` |

#### Example: Shared Memory Proposer

```python
# recipe/srt/vllm_plugin/proposers/suffix_decoding_shm.py
from specrl_cache import SuffixCache

class SuffixDecodingSHMProposer:
    """Suffix decoding proposer using shared memory cache."""

    def __init__(self, ...):
        # Attach to existing shared memory (created by RolloutCacheServer)
        self.suffix_cache = SuffixCache()  # Opens "SUFFIX_CACHE" segment

    def get_spec_proposals(
        self,
        request_ids: list[str],
        prompt_hashes: list[int],
        seq_lens: list[int],
        query_lens: list[int],
        ...
    ):
        """Get speculation proposals from shared memory cache."""
        # Register requests with cache
        for req_id, prompt_hash in zip(request_ids, prompt_hashes):
            self.suffix_cache.StartRequest(req_id, prompt_hash)

        # Get draft tokens from cache (zero-copy read from shared memory)
        draft_tokens = self.suffix_cache.Speculate(
            request_ids,
            current_tokens,  # Last N tokens for tree traversal
            max_spec_len=self.num_speculative_tokens
        )

        return draft_tokens

    def update_from_accepted(self, request_id: str, accepted_tokens: list[int]):
        """Update cache position after verification."""
        self.suffix_cache.UpdatePosition(request_id, len(accepted_tokens))

    def finish_request(self, request_id: str):
        """Cleanup when request completes."""
        self.suffix_cache.FinishRequest(request_id)
```

#### Worker Extension Removal

```python
# recipe/srt/vllm_plugin/worker_extension.py

# BEFORE: Complex snapshot loading logic
# class SuffixTreeWorkerExtension:
#     def load_suffix_snapshot(self, snapshots, hash_mapping):
#         ...

# AFTER: No-op or remove entirely
class SuffixTreeWorkerExtension:
    """No longer needed - workers read directly from shared memory."""

    def load_suffix_snapshot(self, *args, **kwargs):
        # No-op for backwards compatibility
        return {"status": "skipped", "reason": "using_shared_memory"}
```

---

### Phase 5: Testing & Validation

| Task | Description | Priority |
|------|-------------|----------|
| **Integration Tests** | Test server startup, gRPC updates, worker reads | High |
| **Multi-Process Tests** | Verify multiple vLLM workers can read concurrently | High |
| **Stress Tests** | Test with many concurrent updates and reads | Medium |
| **Benchmark** | Compare latency/throughput vs snapshot-based approach | Medium |
| **Memory Tests** | Verify shared memory cleanup on shutdown | High |

#### Test Scenarios

```python
# tests/srt/test_shared_memory_cache.py

def test_server_startup():
    """Test RolloutCacheServer creates shared memory correctly."""
    server = RolloutCacheServer()
    server.Initialize()
    # Verify shared memory segment exists
    cache = SuffixCache()  # Should attach successfully
    server.Shutdown()

def test_update_and_read():
    """Test trainer can update and workers can read."""
    # Start server
    server = RolloutCacheServer()
    server.Initialize()

    # Trainer updates
    updater = SuffixCacheUpdater()
    updater.AddServer("localhost:50051")
    updater.UpdateCache([{"hash": 12345, "tokens": [1, 2, 3, 4, 5]}])

    # Worker reads
    cache = SuffixCache()
    cache.StartRequest("req1", 12345)
    drafts = cache.Speculate(["req1"], [[1, 2]], max_spec_len=3)
    assert drafts[0] == [3, 4, 5]

def test_concurrent_reads():
    """Test multiple processes can read simultaneously."""
    # Spawn multiple reader processes
    # Verify no crashes or data corruption
```

---

## Key Decisions Required

Before starting implementation, clarify:

1. **Reuse SpecRL's C++?**
   - Yes: Copy `spec_rl_cache_impl/`, adapt configuration
   - No: Write custom implementation (significant effort)

2. **Shared Memory Size**
   - Configurable via env var? (Recommended)
   - Fixed value? What size? (50GB suggested default)

3. **Single-node or Multi-node?**
   - Single-node only: Can simplify by removing gRPC, using direct shm writes
   - Multi-node: Need gRPC for cross-node updates

4. **Backwards Compatibility**
   - Keep snapshot-based code path as fallback?
   - Clean removal of old code?

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `recipe/srt/cache_server.py` | **NEW** | Cache server launcher/manager |
| `recipe/srt/suffix_tree_manager.py` | **MODIFY** | Replace backend with SuffixCacheUpdater |
| `recipe/srt/ray_trainer.py` | **MODIFY** | Remove snapshot loading calls |
| `recipe/srt/vllm_server.py` | **MODIFY** | Integrate cache server lifecycle |
| `recipe/srt/vllm_plugin/proposers/suffix_decoding_shm.py` | **NEW** | Shared memory proposer |
| `recipe/srt/vllm_plugin/worker_extension.py` | **DELETE/MODIFY** | Remove or no-op |
| `recipe/srt/vllm_plugin/patches/runner_patches.py` | **MODIFY** | Initialize SuffixCache |

---

## Related Documentation

- [SpecRL Cache Implementation Analysis](../recipe/specRL/SPEC_RL_CACHE_IMPL_ANALYSIS.md) - Detailed C++ implementation analysis
- [SpecRL Architecture](../recipe/specRL/) - Reference implementation
