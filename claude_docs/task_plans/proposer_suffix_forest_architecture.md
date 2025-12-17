# Proposer ↔ SuffixForest Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ vLLM (Inference Engine)                                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ParallelSuffixDecodingProposer                              │   │
│  │ third_party/vllm/vllm/v1/spec_decode/                       │   │
│  │              suffix_decoding_parallel.py                    │   │
│  │                                                             │   │
│  │  - Receives InputBatch from vLLM scheduler                  │   │
│  │  - Manages request lifecycle (start/stop)                   │   │
│  │  - Calls suffix_cache for speculation                       │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ArcticInference (Suffix Tree Library)                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ ParallelSuffixDecodingCache                    [Python]     │   │
│  │ third_party/ArcticInference_srt/arctic_inference/           │   │
│  │              suffix_decoding/parallel_cache.py              │   │
│  │                                                             │   │
│  │  - Maps request_id → tree_index                             │   │
│  │  - Handles hash-based tree sharing                          │   │
│  │  - Wraps C++ SuffixForest                                   │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │ wraps                                 │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ SuffixForest                                   [C++]        │   │
│  │ third_party/ArcticInference_srt/csrc/suffix_decoding/       │   │
│  │              suffix_forest.{h,cc}                           │   │
│  │                                                             │   │
│  │  - Manages collection of SuffixTree instances               │   │
│  │  - Provides batch_speculate() with OpenMP parallelization   │   │
│  │  - Provides batch_extend() for adding tokens                │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                             │ contains                              │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ SuffixTree                                     [C++]        │   │
│  │ third_party/ArcticInference_srt/csrc/suffix_decoding/       │   │
│  │              suffix_tree.{h,cc}                             │   │
│  │                                                             │   │
│  │  - Single suffix tree data structure                        │   │
│  │  - extend(): add tokens to tree                             │   │
│  │  - speculate(): find matching patterns, return drafts       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Python ↔ C++ Bindings: bindings.cc (nanobind)                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow (Current)

```
InputBatch (from vLLM)
│
│  token_ids_cpu: np.ndarray[batch, max_seq_len]  # All tokens
│  num_tokens_no_spec: np.ndarray[batch]          # Current length per request
│  num_prompt_tokens: np.ndarray[batch]           # Prompt length per request
│  req_ids: List[str]                             # Request identifiers
│
▼
┌────────────────────────────────────────────────────────────────┐
│ Proposer.propose()                                  [Python]   │
│                                                                │
│  for i in range(batch_size):           ◄── BOTTLENECK (O(N))  │
│      req_id = input_batch.req_ids[i]                          │
│      context = token_ids_cpu[i, start:end]   # numpy slice    │
│      contexts_list.append(context)           # build list     │
│                                                                │
│  suffix_cache.batch_add_tokens(req_ids, tokens)               │
│  drafts = suffix_cache.batch_speculate(req_ids, contexts)     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ ParallelSuffixDecodingCache                         [Python]   │
│                                                                │
│  - Convert req_ids → tree_indices                             │
│  - Call forest.batch_speculate_ndarray(tree_indices, ...)     │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ bindings.cc: batch_speculate_forest_ndarray()       [C++]     │
│                                                                │
│  - Convert Python list → std::vector      ◄── COPY            │
│  - Release GIL                                                │
│  - Call forest.batch_speculate()                              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ SuffixForest::batch_speculate()                     [C++]     │
│                                                                │
│  #pragma omp parallel for              ◄── FAST (parallelized)│
│  for (i = 0; i < batch_size; i++) {                           │
│      results[i] = trees[i]->speculate(contexts[i], ...);      │
│  }                                                             │
└────────────────────────────────────────────────────────────────┘
```

## Optimization Target

**Move the Python loop into C++:**

```
InputBatch (from vLLM)
│
▼
┌────────────────────────────────────────────────────────────────┐
│ Proposer.propose()                                  [Python]   │
│                                                                │
│  # Minimal Python: just prepare indices                       │
│  tree_indices = precompute_tree_indices(req_ids)              │
│                                                                │
│  drafts = suffix_cache.propose_from_batch(                    │
│      token_ids_cpu,        # Pass 2D array directly           │
│      num_tokens,           # Pass metadata arrays             │
│      tree_indices,         # Pre-computed mapping             │
│      ...                                                       │
│  )                                                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│ SuffixForest::propose_from_batch()          [C++ - NEW]       │
│                                                                │
│  #pragma omp parallel for              ◄── ALL WORK HERE      │
│  for (i = 0; i < batch_size; i++) {                           │
│      // Direct pointer arithmetic - no copy                   │
│      int32_t* context = &token_ids[i * stride + start];       │
│      trees[i]->extend(sampled_token);                         │
│      results[i] = trees[i]->speculate(context, ...);          │
│  }                                                             │
└────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Language | Role |
|------|----------|------|
| `vllm/v1/spec_decode/suffix_decoding_parallel.py` | Python | Proposer - vLLM integration |
| `arctic_inference/suffix_decoding/parallel_cache.py` | Python | Cache wrapper - request management |
| `csrc/suffix_decoding/suffix_forest.{h,cc}` | C++ | Forest - batch operations |
| `csrc/suffix_decoding/suffix_tree.{h,cc}` | C++ | Tree - core data structure |
| `csrc/suffix_decoding/bindings.cc` | C++ | Python ↔ C++ bindings (nanobind) |

## Why This Matters

| Batch Size | Python Loop | C++ Speculate | Python % |
|------------|-------------|---------------|----------|
| 32         | 0.10 ms     | 0.05 ms       | **54%**  |
| 64         | 0.20 ms     | 0.09 ms       | **58%**  |

Moving the loop to C++ eliminates:
- Python interpreter overhead per iteration
- Numpy slice copies
- Python list building
- std::vector conversion in bindings
