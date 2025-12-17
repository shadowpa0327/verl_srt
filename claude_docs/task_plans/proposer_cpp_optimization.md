# Direct C++ InputBatch Consumption for Proposer

## Problem Statement

Profiling of `ParallelSuffixDecodingProposer.propose()` reveals that **Python setup overhead dominates latency** at higher batch sizes:

| Batch Size | Setup % | Setup (ms) | Speculate (ms) |
|------------|---------|------------|----------------|
| 1          | 11.7%   | 0.004      | 0.019          |
| 16         | 47.9%   | 0.052      | 0.028          |
| 32         | 54.2%   | 0.102      | 0.047          |
| 64         | 58.1%   | 0.202      | 0.086          |

The setup phase scales **O(N)** linearly with batch size, while `batch_speculate` scales sub-linearly thanks to C++ parallelization.

## Root Cause

The current flow has significant Python interpreter overhead:

```
Python (propose method)                          Time Cost
─────────────────────────────────────────────────────────────
for i in range(batch_size):                      # O(N) loop
    req_id = input_batch.req_ids[i]              # dict lookup
    if req_id in active_requests:                # dict lookup
        ...
    num_tokens = input_batch.num_tokens_no_spec[i]  # numpy index
    context = token_ids_cpu[i, start:end]        # numpy slice (copy)
    req_ids_list.append(req_id)                  # list append
    contexts_list.append(context)                # list append
─────────────────────────────────────────────────────────────
                    ↓
Python → C++ boundary crossing
                    ↓
─────────────────────────────────────────────────────────────
C++ (batch_speculate_forest_ndarray in bindings.cc:162-199)
  for i in range(len(contexts_list)):            # Convert again
      contexts_vec.emplace_back(...)             # Copy to vector
─────────────────────────────────────────────────────────────
                    ↓
C++ (SuffixForest::batch_speculate)              # Actual work
  #pragma omp parallel for                       # Parallelized
```

**Key inefficiencies:**
1. Python `for` loop over batch (interpreter overhead per iteration)
2. Multiple dict lookups per request (`req_id_to_index`, `active_requests`)
3. Numpy array slicing creates copies
4. Building Python lists with `.append()` (memory allocation per item)
5. C++ binding converts Python lists to `std::vector` (second copy)

## Proposed Solution: Direct C++ Batch Processing

Create a new C++ API that directly consumes the numpy arrays from `InputBatch`:

```
Python (minimal wrapper)                         Time Cost
─────────────────────────────────────────────────────────────
forest.propose_from_batch(                       # Single call
    token_ids_cpu,          # 2D numpy array ptr
    num_tokens_no_spec,     # 1D numpy array ptr
    num_prompt_tokens,      # 1D numpy array ptr
    sampled_token_ids,      # 1D numpy array ptr
    req_indices,            # 1D numpy array (active request indices)
    tree_indices,           # 1D numpy array (tree idx per request)
    max_spec_tokens,
    ...
)
─────────────────────────────────────────────────────────────
                    ↓
C++ (propose_from_batch) - ALL work here
─────────────────────────────────────────────────────────────
  #pragma omp parallel for                       # Parallel setup!
  for (int i = 0; i < batch_size; i++) {
      int req_idx = req_indices[i];
      int num_tokens = num_tokens_no_spec[req_idx];
      int start = max(0, num_tokens - max_tree_depth);
      // Direct pointer access - no copy
      int32_t* context = &token_ids_cpu[req_idx * max_len + start];
      int context_len = num_tokens - start;

      // Add tokens (if provided)
      if (sampled_token_ids[i] > 0) {
          trees[tree_indices[i]]->extend(...);
      }

      // Speculate
      results[i] = trees[tree_indices[i]]->speculate(
          span<const int32_t>(context, context_len), ...);
  }
─────────────────────────────────────────────────────────────
```

## Implementation Components

### 1. C++ Implementation

**Files to modify:**
- `csrc/suffix_decoding/suffix_forest.h` - Add new method declaration
- `csrc/suffix_decoding/suffix_forest.cc` - Implement `propose_from_batch`
- `csrc/suffix_decoding/bindings.cc` - Add nanobind wrapper

**New C++ method signature:**
```cpp
struct ProposeBatchResult {
    std::vector<Draft> drafts;
    std::vector<int> draft_indices;  // Maps result back to input index
};

ProposeBatchResult SuffixForest::propose_from_batch(
    // 2D token array [batch_size, max_seq_len] - row-major
    std::span<const int32_t> token_ids_cpu,
    int max_seq_len,  // stride for 2D indexing

    // Per-request metadata [batch_size]
    std::span<const int32_t> num_tokens,
    std::span<const int32_t> num_prompt_tokens,
    std::span<const int32_t> sampled_token_ids,  // Single token per request

    // Tree mapping [batch_size]
    std::span<const int32_t> tree_indices,
    std::span<const int32_t> seq_ids,

    // Speculation parameters
    int max_spec_tokens,
    float max_spec_factor,
    float min_token_prob
);
```

**Complexity:** Medium
- Implement C++ method with OpenMP parallelization
- Handle edge cases (invalid tree indices, empty batches)
- Add proper locking for concurrent tree access

### 2. Nanobind Wrapper

**New binding in `bindings.cc`:**
```cpp
using Int32Array2D = nb::ndarray<int32_t, nb::numpy, nb::shape<-1, -1>,
                                 nb::device::cpu, nb::c_contig>;

m.def("propose_from_batch", [](
    SuffixForest& forest,
    const Int32Array2D& token_ids_cpu,
    const Int32Array1D& num_tokens,
    const Int32Array1D& num_prompt_tokens,
    const Int32Array1D& sampled_token_ids,
    const Int32Array1D& tree_indices,
    const Int32Array1D& seq_ids,
    int max_spec_tokens,
    float max_spec_factor,
    float min_token_prob
) {
    nb::gil_scoped_release release;

    return forest.propose_from_batch(
        std::span<const int32_t>(token_ids_cpu.data(),
                                  token_ids_cpu.shape(0) * token_ids_cpu.shape(1)),
        token_ids_cpu.shape(1),  // max_seq_len stride
        std::span<const int32_t>(num_tokens.data(), num_tokens.size()),
        // ... etc
    );
});
```

**Complexity:** Low

### 3. Python Wrapper

**New method in `parallel_cache.py`:**
```python
def propose_from_batch(
    self,
    token_ids_cpu: np.ndarray,      # [batch, max_seq_len], int32
    num_tokens: np.ndarray,         # [batch], int32
    num_prompt_tokens: np.ndarray,  # [batch], int32
    sampled_token_ids: np.ndarray,  # [batch], int32
    tree_indices: np.ndarray,       # [batch], int32 (pre-computed)
    seq_ids: np.ndarray,            # [batch], int32 (pre-computed)
    max_spec_tokens: int,
    max_spec_factor: float = 1.0,
    min_token_prob: float = 0.1,
) -> List[SuffixDecodingDraft]:
    """
    Propose draft tokens directly from InputBatch arrays.

    This bypasses Python iteration overhead by processing the
    entire batch in C++ with OpenMP parallelization.
    """
    # Validate inputs
    assert token_ids_cpu.dtype == np.int32
    assert token_ids_cpu.flags['C_CONTIGUOUS']

    # Single C++ call - all work happens there
    drafts = self._forest.propose_from_batch(
        token_ids_cpu,
        num_tokens,
        num_prompt_tokens,
        sampled_token_ids,
        tree_indices,
        seq_ids,
        max_spec_tokens,
        max_spec_factor,
        min_token_prob,
    )

    return [SuffixDecodingDraft.from_native(d) for d in drafts]
```

**Complexity:** Low

### 4. Integration with vLLM Proposer

The challenge is that the current proposer handles:
1. Starting new requests (Python dict management)
2. Adding sampled tokens to trees
3. Speculation
4. Cleanup of finished requests

**Two options:**

**Option A: Move request lifecycle to C++**
- Requires C++ hash maps for `req_to_tree_idx`, `active_requests`
- Complex: need to sync Python and C++ state
- Maximum performance but high complexity

**Option B: Keep request lifecycle in Python, optimize only the hot path** (Recommended)
- Keep `start_request`, `stop_request` in Python (infrequent operations)
- Move `add_tokens` + `speculate` to single C++ call (hot path)
- Moderate complexity, significant speedup

```python
class ParallelSuffixDecodingProposer:
    def propose(self, input_batch, sampled_token_ids):
        # Pre-compute tree indices (still Python, but simple O(N))
        tree_indices = np.zeros(len(sampled_token_ids), dtype=np.int32)
        seq_ids = np.zeros(len(sampled_token_ids), dtype=np.int32)

        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = input_batch.req_ids[i]
            if req_id not in self.suffix_cache.active_requests:
                self._start_request(req_id, input_batch, i)

            tree_indices[i] = self.suffix_cache.get_tree_idx(req_id)
            seq_ids[i] = self.suffix_cache.get_seq_id(req_id)

        # HOT PATH: Single C++ call
        sampled_flat = np.array([s[0] if s else 0 for s in sampled_token_ids], dtype=np.int32)

        drafts = self.suffix_cache.propose_from_batch(
            token_ids_cpu=input_batch.token_ids_cpu,
            num_tokens=input_batch.num_tokens_no_spec,
            num_prompt_tokens=input_batch.num_prompt_tokens,
            sampled_token_ids=sampled_flat,
            tree_indices=tree_indices,
            seq_ids=seq_ids,
            max_spec_tokens=self.num_speculative_tokens,
        )

        return [d.token_ids for d in drafts]
```

**Complexity:** Medium

## Expected Performance Improvement

| Batch Size | Current (ms) | Expected (ms) | Speedup |
|------------|--------------|---------------|---------|
| 32         | 0.189        | ~0.060        | ~3x     |
| 64         | 0.347        | ~0.100        | ~3.5x   |
| 128        | ~0.700       | ~0.180        | ~4x     |

**Assumptions:**
- Setup overhead reduced by ~90% (parallel C++ vs sequential Python)
- `batch_speculate` time unchanged (already optimized)
- Small constant overhead for Python→C++ boundary (~0.01ms)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Memory layout mismatch (`token_ids_cpu` not C-contiguous) | Runtime check, fall back to Python path |
| Request lifecycle complexity | Keep in Python (Option B) |
| Thread safety during concurrent tree access | Use existing per-tree locking in SuffixForest |
| API stability (InputBatch changes) | Version checks, compatibility layer |

## Files to Modify

```
third_party/ArcticInference_srt/
├── csrc/suffix_decoding/
│   ├── suffix_forest.h      # Add propose_from_batch declaration
│   ├── suffix_forest.cc     # Implement propose_from_batch
│   └── bindings.cc          # Add nanobind wrapper
├── arctic_inference/suffix_decoding/
│   └── parallel_cache.py    # Add Python wrapper method

third_party/vllm/vllm/v1/spec_decode/
└── suffix_decoding_parallel.py  # Update to use new API
```

## Success Criteria

1. Proposer setup time reduced by >80% at batch_size=64
2. All existing tests pass
3. No regression in speculation quality (same drafts produced)
4. Memory usage unchanged or reduced
