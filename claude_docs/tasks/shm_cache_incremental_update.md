# SHM Cache Incremental Update - Task Tracking

**Created**: 2025-01-18
**Status**: ✅ Validated
**Branch**: `dev/run_ahead_0117`

---

## Problem Statement

### Observed Behavior
When `update_from_secondary()` sends multiple responses for the same prompt via sequential `update_response_cache()` calls with `responses_per_prompt=1`, only the **last response survives** in the cache.

### Impact
- **75-87.5% data loss** when runahead produces 4-8 responses per prompt
- Speculative decoding effectiveness severely degraded
- Cache populated but mostly empty due to overwrites

### Reproduction
```bash
python recipe/srt/demo_cache_overwrite_bug.py
```

**Before fix:**
- Sequential updates: 1/4 responses retained (25%)
- Batch updates: 4/4 responses retained (100%)

---

## Root Cause Analysis

### Location
`recipe/srt/srt_plugin/shm_cache/suffix_cache/rollout_cache_server.cc:21-91`

### Original Code Behavior
```cpp
grpc::Status RolloutCacheServiceImpl::UpdateCache(...) {
    // ALWAYS creates a new tree
    SuffixTree* tree = segment_->construct<SuffixTree>(...);
    tree->extend(0, tokens);

    // REPLACES existing tree if found
    if (it != tree_map_->end()) {
        uint64_t existing_tree = it->second;
        it->second = tree_ptr;  // Overwrite pointer
        existing_tree_ptr = (SuffixTree*)(existing_tree + segment_base);
    }

    // DELETES old tree
    if (existing_tree_ptr) {
        segment_->destroy_ptr(existing_tree_ptr);
    }
}
```

### Why This Happens
1. Each gRPC `UpdateCache` call creates a **new** `SuffixTree`
2. If a tree exists for the prompt hash, it's **replaced** and **destroyed**
3. The `SuffixTree::extend()` method supports multiple sequences (via `seq_id`), but this capability was unused
4. Result: Each update overwrites previous data instead of accumulating

---

## Solutions

### Option A: C++ Incremental Update (Implemented)

**Status**: ✅ Implemented and tested

**File**: `recipe/srt/srt_plugin/shm_cache/suffix_cache/rollout_cache_server.cc`

**Changes**:
```cpp
grpc::Status RolloutCacheServiceImpl::UpdateCache(...) {
    SuffixTree* tree = nullptr;
    int next_seq_id = 0;

    // Check if tree already exists
    auto it = tree_map_->find(prompt_hash);
    if (it != tree_map_->end()) {
        // REUSE existing tree
        tree = (SuffixTree*)(it->second + segment_base);
        next_seq_id = tree->num_seqs();  // Get next available seq_id
    }

    // Only create new tree if none exists
    if (tree == nullptr) {
        tree = segment_->construct<SuffixTree>(...);
        is_new_tree = true;
    }

    // EXTEND tree with new sequence (incremental)
    tree->extend(next_seq_id, tokens);

    // Register new tree only if we created one
    if (is_new_tree) {
        tree_map_->emplace(prompt_hash, tree_ptr);
    }
}
```

**Pros**:
- Transparent to Python code - no changes needed
- Supports true streaming/incremental updates
- Order and timing of updates doesn't matter
- More robust API design

**Cons**:
- Requires C++ rebuild
- Trees grow unboundedly (may need eviction strategy later)

**Test Results**:
```
Sequential updates: 4/4 responses retained (100%) ✅
Batch updates: 4/4 responses retained (100%) ✅
```

**Critical Bug Found & Fixed (2025-01-19)**:
The initial implementation had a **race condition** that caused SIGSEGV on the second cache update:
```cpp
// BUG: Lock released before extend()
{
    lock();
    tree = find_tree();
    next_seq_id = tree->num_seqs();
    unlock();  // ← Lock released too early!
}
tree->extend(...);  // ← CRASH! Concurrent access without lock
```

**Fix**: Hold lock during entire tree operation (lookup + `num_seqs()` + `extend()`):
```cpp
// Build tokens BEFORE lock (minimize hold time)
std::vector<int> tokens = build_tokens(...);

// Lock held during all tree operations
lock();
tree = find_tree();
next_seq_id = tree->num_seqs();
tree->extend(next_seq_id, tokens);  // ← Now protected
if (is_new_tree) tree_map_->emplace(...);
unlock();
```

**Stress Test**: `recipe/srt/stress_test_cache_server.py` validates thread safety

---

### Option B: Python-Side Grouping (Alternative)

**Status**: ⏳ Not implemented (superseded by Option A)

**File**: `recipe/srt/shared_memory_cache_manager.py:420-464`

**Concept**: Group `SecondaryOutput` objects by `prompt_hash` before calling `update_response_cache()`, then send all responses for the same prompt in a single batch call.

**Changes** (conceptual):
```python
def update_from_secondary(self, secondary_outputs: List[SecondaryOutput], ...):
    # Group outputs by prompt hash
    groups = defaultdict(list)
    for out in usable_outputs:
        groups[out.prompt_hash].append(out)

    # Send each group as a batch
    for prompt_hash, outputs in groups.items():
        prompts = [out.prompt_tokens for out in outputs]
        responses = [out.response_tokens for out in outputs]
        self._cache_updater.update_response_cache(
            prompts=prompts,
            responses=responses,
            responses_per_prompt=len(outputs),  # All responses for this prompt
            precomputed_hashes=[prompt_hash],
        )
```

**Pros**:
- No C++ changes needed
- Works with existing cache server

**Cons**:
- Requires all responses to be available simultaneously
- Caller must know to batch correctly
- Doesn't support true streaming updates
- More complex Python code

---

### Option C: [Your Idea Here]

**Status**: ⏳ To be tested

**Description**:
_Add your alternative approach here_

**Changes**:
_Describe the implementation_

**Pros**:
- _List advantages_

**Cons**:
- _List disadvantages_

---

## Testing

### Unit Tests
- `tests/workers/rollout/rollout_vllm/test_shm_cache_maintenance.py` - 14 tests passing
- `recipe/srt/demo_cache_overwrite_bug.py` - Visual demonstration

### E2E Training
**Status**: ✅ Validated

**Configuration**: Runahead loop (`_fit_runahead`)

**Metrics to watch**:
- `spec_decode/acceptance_rate` - Should improve with proper cache population
- `shm_cache/secondary_outputs_processed` - Confirms secondary updates happening
- `spec_decode/tokens_per_step` - Higher = better speculation

**Results**:
- ✅ Acceptance rate increased after fix (confirmed by user)
- ✅ No server crashes (race condition fixed)
- ✅ Runahead secondary outputs successfully populating cache for next tick

---

## Files Modified

| File | Change |
|------|--------|
| `recipe/srt/srt_plugin/shm_cache/suffix_cache/rollout_cache_server.cc` | Incremental tree update logic |

## Files Created (for testing)

| File | Purpose |
|------|---------|
| `recipe/srt/demo_cache_overwrite_bug.py` | Visual demonstration of bug and fix |
| `recipe/srt/stress_test_cache_server.py` | Thread safety stress test |
| `tests/workers/rollout/rollout_vllm/test_shm_cache_maintenance.py` | Unit tests for cache maintenance |

---

## Open Questions

1. **Memory growth**: With incremental updates, trees can grow unboundedly. Need eviction strategy?
2. **Staleness**: Old sequences may not be relevant. Generation-based eviction?
3. ~~**Concurrency**: Thread-safety when multiple updates hit same tree?~~ **RESOLVED**: Fixed race condition by holding lock during `extend()`.

---

## Next Steps

- [x] Wait for e2e training results - ✅ Acceptance rate improved
- [x] Fix race condition causing SIGSEGV - ✅ Fixed
- [x] Add stress test for thread safety - ✅ `stress_test_cache_server.py`
- [ ] Consider memory management strategy for long-running training
- [ ] Merge to main branch after more validation
