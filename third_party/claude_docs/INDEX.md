# Third-Party Integration Documentation Index

## Quick Start

**What are you trying to do?**

| Goal | Start Here |
|------|------------|
| Understand project goals | [Role of Third-Party Libraries](role_of_third_party_lib.md) |
| Understand vLLM architecture | [vLLM V1 Architecture Skill](skills/vLLM_v1_architecture.md) |
| Work with suffix tree cache | [ArcticInference Cache Skill](skills/arctic_inference_cache_managements.md) |
| Measure acceptance lengths | [Acceptance Length Measurement Skill](skills/acceptance_length_measurement.md) |
| Prompt hash tree mapping | [Prompt Hash Implementation Guide](skills/prompt_hash_tree_mapping.md) |
| Load trees into vLLM | [vLLM Suffix Tree Integration](skills/vllm_suffix_tree_integration.md) |

---

## Documentation Map

```
third_party/
+-- CLAUDE.md                    # High-level third_party overview
+-- claude_docs/
|   +-- INDEX.md                 # THIS FILE - master navigation
|   +-- role_of_third_party_lib.md  # Why these libs, what they enable
|   +-- skills/
|       +-- vLLM_v1_architecture.md       # vLLM quick reference
|       +-- arctic_inference_cache_managements.md  # Cache quick reference
|       +-- acceptance_length_measurement.md  # Metrics quick reference
|       +-- prompt_hash_tree_mapping.md  # Hash-based tree sharing
|       +-- vllm_suffix_tree_integration.md  # Load trees into vLLM
|
+-- vllm/
|   +-- VLLM_V1_ARCHITECTURE.md  # Full vLLM architecture docs
|
+-- ArcticInference_srt/
    +-- CLAUDE.md                # Full ArcticInference overview
    +-- claude_docs/
        +-- PARALLEL_CACHE_GUIDE.md        # ParallelSuffixDecodingCache API
        +-- SUFFIX_TREE_SERIALIZATION.md   # Binary format spec
```

---

## Document Summaries

### Core Documents

| Document | Path | Summary |
|----------|------|---------|
| **vLLM V1 Architecture** | `vllm/VLLM_V1_ARCHITECTURE.md` | Complete vLLM V1 architecture including request flow, component responsibilities, speculative decoding proposer system, and modification points |
| **ArcticInference CLAUDE.md** | `ArcticInference_srt/CLAUDE.md` | Suffix decoding overview, parallel cache API, gRPC server, usage examples, and configuration guidelines |

### Skill Guides

| Document | Path | Summary |
|----------|------|---------|
| **vLLM Architecture Skill** | `skills/vLLM_v1_architecture.md` | Quick-reference for vLLM: request flow, component lookup, adding proposers, RPC chain |
| **Cache Management Skill** | `skills/arctic_inference_cache_managements.md` | Quick-reference for suffix cache: request lifecycle, batch operations, serialization, threading |
| **Acceptance Length Skill** | `skills/acceptance_length_measurement.md` | Quick-reference for measuring speculation quality: metrics extraction, formulas, Prometheus queries |
| **Prompt Hash Implementation** | `skills/prompt_hash_tree_mapping.md` | Implementation guide for prompt hash → tree mapping, concurrent write protection |
| **vLLM Suffix Integration** | `skills/vllm_suffix_tree_integration.md` | Load pre-built trees into vLLM, direct access API, BOS token gotcha |

### Technical References

| Document | Path | Summary |
|----------|------|---------|
| **Parallel Cache Guide** | `ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md` | Detailed ParallelSuffixDecodingCache API, migration from SuffixDecodingCache, vLLM integration example |
| **Serialization Spec** | `ArcticInference_srt/claude_docs/SUFFIX_TREE_SERIALIZATION.md` | Binary format specification, size estimates, Python/C++ API |

---

## Quick Reference: Key Files

### vLLM V1 (Speculative Decoding)

| Component | File | Line | Description |
|-----------|------|------|-------------|
| LLM Entry | `vllm/entrypoints/llm.py` | 66 | Main inference class |
| LLMEngine | `vllm/v1/engine/llm_engine.py` | 45 | Engine orchestration |
| EngineCore | `vllm/v1/engine/core.py` | 63 | Core scheduling loop |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | 43 | Request scheduling |
| Executor | `vllm/v1/executor/abstract.py` | 24 | Worker orchestration |
| Worker | `vllm/v1/worker/gpu_worker.py` | 45 | GPU execution |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py` | 176 | **OWNS DRAFTER** |
| Drafter Init | `gpu_model_runner.py` | 276-317 | Proposer instantiation |
| Draft Dispatch | `gpu_model_runner.py` | 2480-2648 | `propose_draft_token_ids()` |

### ArcticInference

| Component | File | Description |
|-----------|------|-------------|
| SuffixTree (C++) | `csrc/suffix_decoding/suffix_tree.h` | Core data structure |
| SuffixForest (C++) | `csrc/suffix_decoding/suffix_forest.h` | Batched tree container |
| Bindings | `csrc/suffix_decoding/bindings.cc` | Python/C++ bridge |
| ParallelCache | `arctic_inference/suffix_decoding/parallel_cache.py` | Python API |
| gRPC Server | `arctic_inference/suffix_decoding/server.py` | Remote suffix service |
| gRPC Client | `arctic_inference/suffix_decoding/client.py` | Client for remote |

### vLLM Proposers

| Proposer | File | Type |
|----------|------|------|
| Suffix | `vllm/v1/spec_decode/suffix_decoding.py` | Heuristic |
| ParallelSuffix | `vllm/v1/spec_decode/suffix_decoding_parallel.py` | Heuristic |
| RemoteSuffix | `vllm/v1/spec_decode/suffix_decoding_remote.py` | Heuristic |
| N-gram | `vllm/v1/spec_decode/ngram_proposer.py` | Heuristic |
| EAGLE | `vllm/v1/spec_decode/eagle.py` | Model-based |
| Medusa | `vllm/v1/spec_decode/medusa.py` | Model-based |

---

## Common Workflows

### Add New Speculative Decoding Proposer

1. Read: [Adding a New Proposer](skills/vLLM_v1_architecture.md#adding-a-new-proposer)
2. Files:
   - Create: `vllm/v1/spec_decode/your_proposer.py`
   - Modify: `gpu_model_runner.py:276` (init)
   - Modify: `gpu_model_runner.py:2480` (dispatch)

### Implement Distributed Cache

1. Read: [Role of Third-Party Libs - Integration Architecture](role_of_third_party_lib.md#integration-architecture)
2. Key concepts:
   - Per-question SuffixTrees maintained by VERL
   - Micro-batch assignment determines which trees go to which worker
   - Workers reconstruct local Forest from received snapshots

### Debug Suffix Tree Operations

1. Read: [Cache Management Skill](skills/arctic_inference_cache_managements.md#request-lifecycle)
2. Key methods:
   - `cache.get_stats()` - cache statistics
   - `tree.check_integrity()` - validate structure
   - `num_threads=0` - disable parallelism for debugging

---

## Cross-Reference Table

| If you need... | See... |
|----------------|--------|
| vLLM request flow diagram | [vLLM Skill - Request Flow](skills/vLLM_v1_architecture.md#high-level-request-flow) |
| Cache data flow diagram | [Cache Skill - Data Flow](skills/arctic_inference_cache_managements.md#data-flow) |
| Distributed architecture diagram | [Role of Third-Party Libs](role_of_third_party_lib.md#integration-architecture) |
| Serialization binary format | [Serialization Spec](../ArcticInference_srt/claude_docs/SUFFIX_TREE_SERIALIZATION.md#binary-format-v1) |
| ParallelCache full API | [Parallel Cache Guide](../ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md#api-reference) |
| RPC chain implementation | [vLLM Skill - RPC Chain](skills/vLLM_v1_architecture.md#rpc-chain-for-snapshot-loading) |
| Speculation modes | [Cache Skill - Speculation Modes](skills/arctic_inference_cache_managements.md#speculation-modes) |
| Thread configuration | [Cache Skill - Threading](skills/arctic_inference_cache_managements.md#thread-configuration) |
| vLLM speculative config | [vLLM Skill - Config](skills/vLLM_v1_architecture.md#configuration-reference) |
| Measure acceptance length | [Acceptance Skill - Methods](skills/acceptance_length_measurement.md#how-to-measure-acceptance-length) |
| Prometheus metrics queries | [Acceptance Skill - Prometheus](skills/acceptance_length_measurement.md#method-3-prometheus-queries-production) |
| Compare before/after snapshot | [Acceptance Skill - Comparison](skills/acceptance_length_measurement.md#comparing-beforeafter-snapshot-loading) |
