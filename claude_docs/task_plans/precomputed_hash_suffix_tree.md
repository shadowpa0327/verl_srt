# Pre-computed Hash for Suffix Tree Integration

> **Status**: Partially Implemented
> **Created**: 2025-12-12
> **Related**: [`suffix_tree_verl_integration.md`](suffix_tree_verl_integration.md)

## Implementation Status

| Step | Component | Status | Notes |
|------|-----------|--------|-------|
| 1 | ArcticInference `pre_computed_hash` param | ✅ Done | `parallel_cache.py:start_request()` |
| 2 | vLLM InputBatch `prompt_hashes` dict | ❌ Pending | Need to add to `gpu_input_batch.py` |
| 3 | vLLM Spec Decoder use pre-computed hash | ❌ Pending | Need to modify `suffix_decoding_parallel.py` |
| 4 | Verl Rollout compute and pass hash | ❌ Pending | Need to modify `vllm_rollout_spmd.py` |
| 5 | SuffixTreeManager use pre-computed hash | ✅ Done | Already checks `prompt_hashes` in batch |

**Current State**: The trainer side (`suffix_tree_manager.py`) is ready to accept pre-computed hashes, but the rollout worker doesn't compute or pass them yet. Hash computation still happens separately in trainer and vLLM.

## Problem Statement

Currently, hash computation for suffix tree matching happens in two places:
1. **Trainer side** (`suffix_tree_manager.py`): Computes hash from `vllm_prompt_tokens`
2. **vLLM side** (`suffix_decoding_parallel.py`): Computes hash from `input_batch.token_ids_cpu`

This creates potential for hash mismatch if tokens differ between the two locations.

## Key Findings

### Current Architecture

| Component | Behavior | Location |
|-----------|----------|----------|
| ArcticInference | `start_request()` does NOT accept pre-computed hashes | `parallel_cache.py:178` |
| vLLM generate() | Accepts `sampling_params` as list (one per prompt) with `extra_args` | `llm.py:335` |
| verl rollout | Uses single `SamplingParams` for entire batch | `vllm_rollout_spmd.py:443` |
| verl tokens | Stores `vllm_prompt_tokens` in `non_tensor_batch` | `vllm_rollout_spmd.py:557` |

### Hash Computation (ArcticInference parallel_cache.py:124-155)
```python
def _hash_prompt(self, prompt_tokens, hash_token_count=None):
    # Uses LAST N tokens (default: 128)
    tokens_to_hash = prompt_tokens[-hash_token_count:]
    token_bytes = np.array(tokens_to_hash, dtype=np.int32).tobytes()
    return hashlib.sha256(token_bytes).hexdigest()[:16]
```

## Solution: Compute Hash Once, Pass Through Pipeline

### Architecture

```
Rollout Worker (verl)
    ├── Compute hash from prompt tokens (SINGLE SOURCE OF TRUTH)
    ├── Create per-request SamplingParams with extra_args["prompt_hash"]
    ├── Store hashes in non_tensor_batch["prompt_hashes"]
    └── Return hash with batch for trainer

vLLM InputBatch (gpu_input_batch.py)
    ├── Add self.prompt_hashes: dict[str, str]
    ├── In add_request(): extract from sampling_params.extra_args
    └── In remove_request(): cleanup prompt_hashes[req_id]

Spec Decoder (suffix_decoding_parallel.py)
    ├── Read hash from input_batch.prompt_hashes.get(req_id)
    └── Pass to suffix_cache.start_request(..., pre_computed_hash=hash)

ArcticInference (parallel_cache.py)
    └── Add pre_computed_hash param to start_request()

Trainer (SuffixTreeManager)
    └── Use hash from non_tensor_batch["prompt_hashes"] directly
```

## Files to Modify

| File | Change | Lines |
|------|--------|-------|
| `third_party/ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py` | Add `pre_computed_hash` param | ~178-234 |
| `third_party/vllm/vllm/v1/worker/gpu_input_batch.py` | Add `prompt_hashes` dict | ~276, ~352, ~491 |
| `third_party/vllm/vllm/v1/spec_decode/suffix_decoding_parallel.py` | Pass pre-computed hash | ~116-122 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Compute hash, create per-request params | ~108, ~411, ~443 |
| `verl/trainer/ppo/suffix_tree_manager.py` | Use pre-computed hash | ~190-209 |

## Implementation Details

### Step 1: ArcticInference - Accept Pre-computed Hash

**File:** `third_party/ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py`

```python
def start_request(
    self,
    req_id: Hashable,
    prompt_token_ids: np.ndarray | Sequence[int],
    hash_token_count: Optional[int] = None,
    pre_computed_hash: Optional[str] = None,  # NEW
):
    if req_id in self._req_to_tree_idx:
        raise ValueError(f"Request '{req_id}' is already active")

    effective_hash_count = hash_token_count if hash_token_count is not None else self._hash_token_count
    use_hash_lookup = effective_hash_count > 0 or pre_computed_hash is not None

    if use_hash_lookup:
        # Use pre-computed hash if provided, otherwise compute
        if pre_computed_hash is not None:
            prompt_hash = pre_computed_hash
        else:
            prompt_hash = self._hash_prompt(prompt_token_ids, effective_hash_count)
        # ... rest unchanged (tree reuse logic)
```

### Step 2: vLLM InputBatch - Store Hashes

**File:** `third_party/vllm/vllm/v1/worker/gpu_input_batch.py`

```python
# In __init__() after line 276:
self.prompt_hashes: dict[str, str] = {}

# In add_request() after line 351:
if sampling_params.extra_args:
    prompt_hash = sampling_params.extra_args.get("prompt_hash")
    if prompt_hash:
        self.prompt_hashes[req_id] = prompt_hash

# In remove_request() after line 491:
self.prompt_hashes.pop(req_id, None)
```

### Step 3: vLLM Spec Decoder - Use Pre-computed Hash

**File:** `third_party/vllm/vllm/v1/spec_decode/suffix_decoding_parallel.py`

```python
if req_id not in self.suffix_cache.active_requests:
    num_prompt_tokens = input_batch.num_prompt_tokens[index]
    prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]

    # Use pre-computed hash if available
    pre_computed_hash = input_batch.prompt_hashes.get(req_id)
    self.suffix_cache.start_request(
        req_id,
        prompt_token_ids,
        pre_computed_hash=pre_computed_hash
    )
```

### Step 4: Verl Rollout - Compute and Pass Hash

**File:** `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

```python
# Add hash computation helper after line 108:
def _compute_prompt_hash(prompt_tokens: list[int], hash_token_count: int = 128) -> str:
    """Compute hash from last N tokens, matching ArcticInference logic."""
    import hashlib
    if len(prompt_tokens) > hash_token_count:
        tokens_to_hash = prompt_tokens[-hash_token_count:]
    else:
        tokens_to_hash = prompt_tokens
    token_bytes = np.array(tokens_to_hash, dtype=np.int32).tobytes()
    return hashlib.sha256(token_bytes).hexdigest()[:16]

# In generate_sequences(), after line 411:
prompt_hashes = []
for input_data in vllm_inputs:
    prompt_hash = _compute_prompt_hash(input_data["prompt_token_ids"])
    prompt_hashes.append(prompt_hash)

# Create per-request SamplingParams (around line 443):
sampling_params_list = []
for i, prompt_hash in enumerate(prompt_hashes):
    params = copy.copy(self.sampling_params)
    params.extra_args = {"prompt_hash": prompt_hash}
    sampling_params_list.append(params)

outputs = self.inference_engine.generate(
    prompts=vllm_inputs,
    sampling_params=sampling_params_list,  # List instead of single
    ...
)

# Store hashes in non_tensor_batch (after line 550):
non_tensor_batch["prompt_hashes"] = np.array(prompt_hashes, dtype=object)
```

### Step 5: SuffixTreeManager - Use Pre-computed Hash

**File:** `verl/trainer/ppo/suffix_tree_manager.py`

```python
# In update_from_rollout():
prompt_hashes = batch.non_tensor_batch.get("prompt_hashes")

for i in range(batch_size):
    # ... existing prompt_tokens extraction ...

    pre_computed_hash = prompt_hashes[i] if prompt_hashes is not None else None
    self._cache.start_request(req_id, prompt_array, pre_computed_hash=pre_computed_hash)
```

## Benefits

| Benefit | Description |
|---------|-------------|
| Single source of truth | Hash computed once in rollout worker |
| No mismatch possible | Same hash used everywhere |
| Backward compatible | `pre_computed_hash=None` falls back to original behavior |
| Minimal changes | Leverages existing `extra_args` mechanism |

## Testing Strategy

### Level 1: Unit Tests (No GPU Required)

These tests validate hash computation logic in isolation.

**Test 1.1: Hash Function Consistency**
```python
# File: third_party/examples/test_precomputed_hash_unit.py
def test_hash_consistency():
    """Verify our hash function matches ArcticInference's _hash_prompt exactly."""
    from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

    cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

    test_tokens = [
        [1, 2, 3, 4, 5],                    # Short (< 128)
        list(range(128)),                    # Exactly 128
        list(range(200)),                    # Long (> 128, tests truncation)
    ]

    for tokens in test_tokens:
        our_hash = _compute_prompt_hash(tokens, hash_token_count=128)
        arctic_hash = cache._hash_prompt(np.array(tokens, dtype=np.int32), 128)
        assert our_hash == arctic_hash, f"Hash mismatch for {len(tokens)} tokens"
```

**Test 1.2: Edge Cases**
```python
def test_hash_edge_cases():
    """Test boundary conditions for hash computation."""
    # Empty tokens (just BOS)
    assert _compute_prompt_hash([1]) == expected_hash_for_bos

    # Exactly hash_token_count tokens
    tokens = list(range(128))
    hash1 = _compute_prompt_hash(tokens, hash_token_count=128)

    # hash_token_count + 1 tokens (should truncate)
    tokens_plus = [999] + list(range(128))  # First token should be ignored
    hash2 = _compute_prompt_hash(tokens_plus, hash_token_count=128)
    assert hash1 == hash2, "Truncation should ignore prefix tokens"
```

**Test 1.3: Tree Lookup with Pre-computed Hash**
```python
def test_tree_lookup_precomputed():
    """Verify tree reuse works when pre_computed_hash is provided."""
    cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

    prompt_tokens = np.array(list(range(100)), dtype=np.int32)
    pre_hash = cache._hash_prompt(prompt_tokens, 128)

    # Build tree with computed hash
    cache.start_request("req1", prompt_tokens)
    cache.add_tokens("req1", np.array([200, 201, 202], dtype=np.int32))

    # Look up with pre-computed hash (simulating different prompt with same hash)
    cache.start_request("req2", prompt_tokens, pre_computed_hash=pre_hash)

    # Should reuse the same tree
    assert cache._req_to_tree_idx["req1"] == cache._req_to_tree_idx["req2"]
```

### Level 2: vLLM Integration Tests (GPU Required, No verl)

These tests verify the `extra_args` flow through vLLM without involving verl.

**Test 2.1: extra_args Passthrough**
```python
# File: third_party/examples/test_precomputed_hash_vllm.py
def test_extra_args_passthrough():
    """Verify prompt_hash flows through SamplingParams to InputBatch."""
    llm = LLM(model="...", speculative_config={"method": "suffix", ...})

    prompts = ["Hello", "World"]
    params_list = [
        SamplingParams(extra_args={"prompt_hash": "hash_a"}),
        SamplingParams(extra_args={"prompt_hash": "hash_b"}),
    ]

    # Should not raise - validates extra_args are accepted
    outputs = llm.generate(prompts, params_list)
    assert len(outputs) == 2
```

**Test 2.2: Hash-Based Tree Reuse (vLLM Only)**
```python
def test_hash_tree_reuse_vllm_only():
    """
    Full vLLM-only test: build trees, load snapshot, verify tree reuse via hash.

    This is the key integration test before adding verl complexity.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 1: Build trees externally
    cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

    prompt = "What is machine learning?"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_hash = _compute_prompt_hash(prompt_tokens)

    cache.start_request("build", np.array(prompt_tokens, dtype=np.int32))
    cache.add_tokens("build", np.array([100, 101, 102], dtype=np.int32))  # Fake response

    snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

    # Step 2: Load into vLLM
    llm = LLM(model=model_name, speculative_config={...})
    llm.load_snapshot(snapshots, hash_mapping)

    # Step 3: Generate with pre-computed hash
    params = SamplingParams(extra_args={"prompt_hash": prompt_hash})
    outputs = llm.generate([prompt], [params])

    # Step 4: Verify tree was found via metrics or logging
    # (Check acceptance rate is non-zero, or add debug logging)
```

### Level 3: Full Integration Tests (verl + vLLM)

**Test 3.1: End-to-End Hash Consistency**
```python
def test_e2e_hash_consistency():
    """
    Full integration: verify hash computed in rollout matches trainer expectation.

    Run a single training step and verify:
    1. Hash is computed in rollout worker
    2. Hash is stored in non_tensor_batch["prompt_hashes"]
    3. Hash matches what trainer computes independently
    """
    # This would be a pytest integration test using a small model
    pass
```

### Test Files to Create

| File | Purpose | GPU Required |
|------|---------|--------------|
| `third_party/examples/test_precomputed_hash_unit.py` | Unit tests for hash computation | No |
| `third_party/examples/test_precomputed_hash_vllm.py` | vLLM-only integration tests | Yes |
| `third_party/examples/example_precomputed_hash.py` | Runnable demo showing full flow | Yes |

### Important: Tree Garbage Collection

When testing hash passthrough, be aware that **trees are garbage-collected** after their requests complete:

```
Request lifecycle:
1. start_request() → tree created, hash added to _hash_to_tree_idx
2. add_tokens() → tokens added to tree
3. speculate() → draft tokens generated
4. stop_request() → if no other requests reference tree AND tree is not protected:
   - tree is removed from forest
   - hash is removed from _hash_to_tree_idx
```

**Impact on testing**: If you check `_hash_to_tree_idx` after generation completes, some hashes may already be gone because their requests finished and got cleaned up.

**Solution**: Use monkey-patching to capture `pre_computed_hash` values at the moment `start_request()` is called. This is more rigorous than checking post-generation state.

### Debug Logging Points

Add temporary logging at these locations to verify hash flow:

```python
# In vllm_rollout_spmd.py (verl):
logger.debug(f"Computed prompt_hash={prompt_hash} for prompt len={len(tokens)}")

# In gpu_input_batch.py (vLLM):
logger.debug(f"Stored prompt_hash={prompt_hash} for req_id={req_id}")

# In suffix_decoding_parallel.py (vLLM):
logger.debug(f"Using pre_computed_hash={pre_hash} for req_id={req_id}")

# In parallel_cache.py (ArcticInference):
logger.debug(f"start_request: pre_computed={pre_computed_hash}, computed={computed_hash}")
```

### Validation Checklist

Before merging:

- [ ] Unit tests pass without GPU
- [ ] vLLM-only test shows tree reuse works
- [ ] Debug logging confirms hash matches at all 4 points
- [ ] `pre_computed_hash=None` still works (backward compat)
- [ ] No performance regression (hash computation is O(1))
- [ ] Full verl training run completes without hash mismatch errors
