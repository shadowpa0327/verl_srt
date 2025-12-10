# Hash Tree vLLM Integration

**Status**: Complete
**Created**: 2025-12-10
**Last Updated**: 2025-12-10

---

## Goal

Enable vLLM to leverage hash-based tree sharing so requests with identical prompts automatically reuse suffix trees, reducing memory and improving speculation quality.

---

## Background

`ParallelSuffixDecodingCache` in ArcticInference supports:
- `create_snapshot(include_hash_mapping=True)` - Returns `(snapshots, hash_mapping)`
- `load_snapshot(snapshots, hash_to_tree=mapping)` - Restores hash→tree lookup

**Reference**: [`skills/prompt_hash_tree_mapping.md`](../skills/prompt_hash_tree_mapping.md)

---

## Implementation Summary

### MVP Approach: Direct Access for verl SPMD

Simplified implementation focused on verl SPMD mode with direct drafter access, bypassing the RPC chain.

**Key design decision**: The drafter's `load_snapshot()` accepts both formats:
- `bytes` - Legacy RPC path (converted internally to list format)
- `(list, dict)` - New verl direct access path

---

## Critical Learning: BOS Token Matching

**Problem discovered during testing**: Hash mismatch between tree building and inference.

When building suffix trees externally:
```python
# WRONG - hash won't match vLLM's inference
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

# CORRECT - matches vLLM's tokenization
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
```

**Root cause**: vLLM's `input_batch.token_ids_cpu` includes the BOS token prepended during tokenization. The hash is computed on prompt tokens, so if tree building excludes BOS but inference includes it, the hashes won't match and no tree reuse occurs.

**Fix**: Always use `add_special_tokens=True` when building trees externally to match vLLM's tokenization.

---

## Files Modified

| File | Change |
|------|--------|
| `vllm/vllm/entrypoints/llm.py` | `load_snapshot(snapshots, hash_mapping)` - direct access to drafter |
| `vllm/vllm/v1/spec_decode/suffix_decoding_parallel.py` | `load_snapshot()` handles both `bytes` and `(list, dict)` via type detection |
| `examples/example_suffix_tree_speculation.py` | Updated to use `load_snapshot()` with hash-based tree sharing |

---

## How It Works

### Tree Building (External)

```python
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

for prompt, response in data:
    # CRITICAL: Use add_special_tokens=True to match vLLM
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    cache.start_request(req_id, np.array(prompt_tokens, dtype=np.int32))

    # Add response tokens
    full_tokens = tokenizer.encode(prompt + response, add_special_tokens=True)
    response_tokens = full_tokens[len(prompt_tokens):]
    cache.add_tokens(req_id, np.array(response_tokens, dtype=np.int32))

# Create snapshot with hash mapping
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)
```

### Loading into vLLM

```python
# Load into vLLM
llm.load_snapshot(snapshots, hash_mapping)
```

### During Inference (automatic)

1. New request arrives with prompt tokens (including BOS)
2. `propose()` checks if request is in `suffix_cache.active_requests`
3. If not, calls `suffix_cache.start_request(req_id, prompt_tokens)`
4. Cache hashes prompt, looks up in `_hash_to_tree_idx`, assigns to matching tree
5. `batch_speculate()` uses the assigned tree for proposals

---

## verl Integration Pattern

Similar to `update_weights` in `vllm_rollout_spmd.py`:

```python
# In verl rollout worker:
async def load_suffix_snapshot(self, snapshots, hash_mapping):
    self.inference_engine.load_snapshot(snapshots, hash_mapping)
```

---

## Testing

Run the example:
```bash
cd third_party
source ../.venv/bin/activate
python examples/example_suffix_tree_speculation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-rounds 3 \
    --num-prompts 10
```

Expected: Enhanced inference shows higher acceptance length than baseline after loading patterns.

---

## Troubleshooting

### No improvement in acceptance length

1. **Check hash matching**: Ensure tree building uses `add_special_tokens=True`
2. **Verify snapshot loaded**: Look for log `"Loaded N suffix trees (X bytes) with M hash mappings"`
3. **Check tree assignment**: When `_has_global_tree=False`, each request uses its hash-matched tree

### GPU OOM when creating second LLM instance

vLLM/PyTorch doesn't fully release GPU memory after `del`. Solutions:
- Use single LLM instance (recommended for verl)
- Use subprocess for isolated memory
- Reduce `gpu_memory_utilization` for second instance
