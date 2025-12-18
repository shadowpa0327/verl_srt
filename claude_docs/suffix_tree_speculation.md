# Suffix Tree Speculation (vLLM Integration)

Suffix tree speculation enables faster inference by predicting likely token sequences based on patterns learned from training data.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL (verl/user)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌─────────────────────────────┐                                           │
│   │ ParallelSuffixDecodingCache │  ← Build trees from training data         │
│   │   (ArcticInference)         │                                           │
│   └──────────────┬──────────────┘                                           │
│                  │ cache.create_snapshot(include_hash_mapping=True)         │
│                  ▼                                                          │
│         (snapshots, hash_mapping)                                           │
└──────────────────┼──────────────────────────────────────────────────────────┘
                   │ llm.load_snapshot(snapshots, hash_mapping)
                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              vLLM                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│   LLM.load_snapshot() ──► GPUModelRunner.drafter ──► suffix_cache           │
│                                                                             │
│   During inference:                                                         │
│   1. New request with prompt_tokens (includes BOS)                          │
│   2. suffix_cache.start_request(req_id, prompt_tokens)                      │
│        ├── hash(prompt_tokens) → "abc123"                                   │
│        ├── lookup _hash_to_tree_idx["abc123"] → tree_idx=0                  │
│        └── assign request to Tree0                                          │
│   3. batch_speculate() queries assigned tree for draft tokens               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
# Build trees during training
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)
for prompt, response in training_data:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)  # MUST include BOS
    cache.start_request(req_id, tokens)
    cache.add_tokens(req_id, response_tokens)

# Create snapshot with hash mapping
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

# Load into vLLM (in verl rollout worker)
self.inference_engine.load_snapshot(snapshots, hash_mapping)
```

## Integration Point

In `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`, follow the `update_weights()` pattern:

```python
# Similar to existing weight sync (lines 580-603):
self.inference_engine.load_snapshot(snapshots, hash_mapping)
```

## Key Files

| Component | Location |
|-----------|----------|
| vLLM patches | `verl/workers/rollout/vllm_rollout/patches/` → runtime monkey patches |
| vLLM API | `verl/workers/rollout/vllm_rollout/patches/llm_patches.py` → `load_snapshot()` |
| Cache API | `third_party/ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py` |
| verl rollout | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` |

## Critical Notes

1. **Tokenization**: Use `add_special_tokens=True` to include BOS token (must match vLLM)
2. **SPMD mode**: Requires `VLLM_ENABLE_V1_MULTIPROCESSING=0`
3. **Tree persistence**: Loaded trees are protected from GC until next `load_snapshot()` call

## Detailed Documentation

| Topic | Document |
|-------|----------|
| vLLM integration architecture | [`third_party/claude_docs/skills/vllm_suffix_tree_integration.md`](../third_party/claude_docs/skills/vllm_suffix_tree_integration.md) |
| Hash-based tree mapping | [`third_party/claude_docs/skills/prompt_hash_tree_mapping.md`](../third_party/claude_docs/skills/prompt_hash_tree_mapping.md) |
| Cache management API | [`third_party/ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md`](../third_party/ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md) |
| Task plan | [`third_party/claude_docs/task_plans/hash_tree_vllm_integration.md`](../third_party/claude_docs/task_plans/hash_tree_vllm_integration.md) |
