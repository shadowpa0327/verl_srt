# vLLM Suffix Tree Integration

## Design Philosophy

**Goal**: Enable pre-loaded suffix trees with hash-based prompt matching for verl SPMD mode.

**Approach**: Direct access to drafter via `model_runner`. Drafter detects input type internally.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL (verl/user)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────┐                                           │
│   │ ParallelSuffixDecodingCache │  ← Build trees from training data         │
│   │   (ArcticInference)         │                                           │
│   └──────────────┬──────────────┘                                           │
│                  │                                                          │
│                  │ cache.create_snapshot(include_hash_mapping=True)         │
│                  ▼                                                          │
│         (snapshots, hash_mapping)                                           │
│                  │                                                          │
└──────────────────┼──────────────────────────────────────────────────────────┘
                   │
                   │ llm.load_snapshot(snapshots, hash_mapping)
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              vLLM                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LLM ─── load_snapshot() ───► GPUModelRunner.drafter                       │
│                                        │                                    │
│                                        ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              ParallelSuffixDecodingProposer                         │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │                    suffix_cache                              │    │   │
│   │  │              (ParallelSuffixDecodingCache)                   │    │   │
│   │  │                                                              │    │   │
│   │  │   _hash_to_tree_idx: {"abc123": 0, "def456": 1, ...}        │    │   │
│   │  │   forest: [Tree0, Tree1, Tree2, ...]                        │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   During inference (propose):                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  1. New request with prompt_tokens (includes BOS)                   │   │
│   │  2. suffix_cache.start_request(req_id, prompt_tokens)               │   │
│   │       → hash(prompt_tokens) → lookup _hash_to_tree_idx → assign tree│   │
│   │  3. batch_speculate() → query assigned tree → draft_token_ids       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API

```python
# Build trees externally
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

# Load into vLLM
llm.load_snapshot(snapshots, hash_mapping)
```

## Files Modified

| File | Change |
|------|--------|
| `vllm/entrypoints/llm.py` | `load_snapshot(snapshots, hash_mapping)` |
| `vllm/v1/spec_decode/suffix_decoding_parallel.py` | `load_snapshot()` handles both bytes and tuple |

## Critical: BOS Token Matching

Use `tokenizer.encode(prompt, add_special_tokens=True)` when building trees. vLLM includes BOS in prompts - mismatched tokens = mismatched hash = no tree reuse.
