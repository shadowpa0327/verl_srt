# Role of Third-Party Libraries

## Overview

This directory contains two dependency libraries that enable speculative decoding for LLM inference:

- **vLLM** - Inference engine with speculative decoding support
- **ArcticInference_srt** - Suffix tree implementation for draft token generation

## ArcticInference_srt

**Provides**:
- SuffixTree/SuffixForest data structures for pattern matching
- Snapshot serialization for persisting and restoring historical token patterns across sessions

**Used by**:
- vLLM (as the speculation backend)
- Verl (parent project, outside `third_party/`)

## vLLM

**Provides**:
- LLM inference engine with continuous batching
- Speculative decoding proposers that leverage SuffixTree/SuffixForest from ArcticInference_srt
- Interface for loading SuffixTree/Forest snapshots at runtime

**Used by**:
- Verl (parent project, outside `third_party/`)

---

## Integration Architecture

### High-Level Flow

VERL maintains **per-question SuffixTrees** that store historical response patterns. Before each rollout batch, VERL:
1. Determines micro-batch assignment (which questions go to which worker)
2. Collects snapshots of trees for the assigned questions
3. Pushes snapshots to each worker, which reconstructs a local Forest
4. After rollout, workers return completed Q/A pairs to update the trees

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VERL (Controller/Trainer)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Per-Question SuffixTrees (one tree per question)       │    │
│  │     Q1:Tree1    Q2:Tree2    Q3:Tree3    ...    Qn:TreeN             │    │
│  │         (each stores historical response patterns for that question) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Micro-batch assignment: Worker1 ← [Q1,Q2], Worker2 ← [Q3,Q4], ...         │
└─────────────────────────────────────────────────────────────────────────────┘
        │                                              ▲
        │ PUSH snapshots for                           │ Updated Q/A pairs
        │ assigned questions                           │ (batch end)
        │ (batch start)                                │
        ▼                                              │
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│   vLLM Worker 1   │   │   vLLM Worker 2   │   │   vLLM Worker N   │
│  ┌─────────────┐  │   │  ┌─────────────┐  │   │  ┌─────────────┐  │
│  │Local Forest │  │   │  │Local Forest │  │   │  │Local Forest │  │
│  │ [T1, T2]    │  │   │  │ [T3, T4]    │  │   │  │ [Tn-1, Tn]  │  │
│  └─────────────┘  │   │  └─────────────┘  │   │  └─────────────┘  │
│  ParallelSuffix   │   │  ParallelSuffix   │   │  ParallelSuffix   │
│  DecodingProposer │   │  DecodingProposer │   │  DecodingProposer │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

### Component Mapping

| Component | Location | Purpose |
|-----------|----------|---------|
| SuffixTree (C++) | `ArcticInference_srt/csrc/suffix_decoding/suffix_tree.h` | Core data structure |
| SuffixForest (C++) | `ArcticInference_srt/csrc/suffix_decoding/suffix_forest.h` | Batched tree container |
| ParallelSuffixDecodingCache | `ArcticInference_srt/arctic_inference/suffix_decoding/parallel_cache.py` | Python API |
| Serialization | `ArcticInference_srt/csrc/suffix_decoding/suffix_tree.cc` | `create_snapshot()` / `restore_snapshot()` |
| SuffixDecodingProposer | `vllm/v1/spec_decode/suffix_decoding.py` | Single-request proposer |
| ParallelSuffixDecodingProposer | `vllm/v1/spec_decode/suffix_decoding_parallel.py` | Batched proposer |

### vLLM Internal Structure

The drafter (proposer) lives deep in vLLM's stack:

```
LLM() / AsyncLLM
    └── LLMEngine
        └── EngineCoreClient
            └── EngineCore
                ├── Scheduler
                └── Executor (collective_rpc)
                    └── Worker(s)
                        └── GPUModelRunner  ← OWNS the drafter
                            └── drafter (SuffixDecodingProposer)
```

**Key insight**: To load snapshots into the drafter, must use RPC chain through `Executor.collective_rpc()` since there's no direct path from `LLM()` entry point.

---

## References
- [vLLM Architecture Skill](skills/vLLM_v1_architecture.md)
- [Cache Management Skill](skills/arctic_inference_cache_managements.md)
