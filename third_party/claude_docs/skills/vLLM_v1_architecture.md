# vLLM V1 Architecture Skill Guide

## Purpose
Quick-reference skill for navigating vLLM V1's architecture when working on speculative decoding integration.

---

## Source Documentation

| Document | Path | Description |
|----------|------|-------------|
| **Primary** | `third_party/claude_docs/VLLM_V1_ARCHITECTURE.md` | Full architecture documentation |
| **Integration** | `third_party/claude_docs/role_of_third_party_lib.md` | Integration architecture overview |

---

## Quick Navigation Index

### By Task

| Task | Go To | Key File(s) |
|------|-------|-------------|
| Add new proposer | [Adding a New Proposer](#adding-a-new-proposer) | `gpu_model_runner.py:276-317` |
| Modify scheduling | [Scheduler](#3-scheduler) | `vllm/v1/core/sched/scheduler.py` |
| Add RPC method | [RPC Chain](#rpc-chain-for-snapshot-loading) | `abstract.py`, `gpu_worker.py` |
| Modify sampling | [Sampler](#8-sampling) | `vllm/v1/sample/sampler.py` |
| Understand request flow | [Request Flow](#high-level-request-flow) | - |
| Speculative decoding | [Spec Decode](#speculative-decoding-overview) | `vllm/v1/spec_decode/` |

### By Component

| Component | Section | Primary File |
|-----------|---------|--------------|
| Entry Point | [Entry Points](#1-entry-points) | `vllm/entrypoints/llm.py:66` |
| LLMEngine | [Engine Layer](#2-engine-layer) | `vllm/v1/engine/llm_engine.py:45` |
| EngineCore | [Core Scheduling](#3-scheduler) | `vllm/v1/engine/core.py:63` |
| Scheduler | [Core Scheduling](#3-scheduler) | `vllm/v1/core/sched/scheduler.py:43` |
| Executor | [Executor Layer](#4-executor-layer) | `vllm/v1/executor/abstract.py:24` |
| Worker | [Worker Layer](#5-worker-layer) | `vllm/v1/worker/gpu_worker.py:45` |
| GPUModelRunner | [Worker Layer](#5-worker-layer) | `vllm/v1/worker/gpu_model_runner.py:176` |
| Drafter | [Proposers](#supported-proposers) | `vllm/v1/spec_decode/` |

---

## Architecture Summary

### High-Level Request Flow

```
User Request
      |
      v
LLM() -----> LLMEngine -----> EngineCoreClient -----> EngineCore
                                                          |
                                    +---------------------+
                                    |
                                    v
                              Executor -----> Worker(s) -----> GPUModelRunner
                                                                    |
                                                         +----------+----------+
                                                         |          |          |
                                                      Model     Sampler    Drafter
```

### Key Ownership Chain

```
LLM() / AsyncLLM
    +-- LLMEngine
        +-- EngineCoreClient
            +-- EngineCore
                +-- Scheduler (request management)
                +-- Executor (worker orchestration)
                    +-- Worker(s)
                        +-- GPUModelRunner  <-- OWNS drafter
                            +-- drafter (proposer)
```

**Critical Insight**: The drafter lives deep in the stack. Cannot directly access from `LLM()`. Must use RPC chain.

---

## Component Reference

### 1. Entry Points

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `LLM` | `vllm/entrypoints/llm.py` | 66 | Main offline inference |
| `AsyncLLMEngine` | `vllm/v1/engine/async_llm.py` | - | Async serving |

**Key Methods:**
- `LLM.__init__()` - Creates LLMEngine
- `LLM.generate()` - Main generation (line 335)
- `LLM._run_engine()` - Execution loop (line 1578)

### 2. Engine Layer

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `LLMEngine` | `vllm/v1/engine/llm_engine.py` | 45 | Processing orchestration |
| `Processor` | `vllm/v1/engine/processor.py` | - | Input processing |
| `OutputProcessor` | `vllm/v1/engine/output_processor.py` | - | Output detokenization |

**Key Methods:**
- `from_engine_args()` - Factory (line 158)
- `add_request()` - Queue request (line 213)
- `step()` - Execute one step (line 257)

### 3. Scheduler

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `EngineCore` | `vllm/v1/engine/core.py` | 63 | Main scheduling loop |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py` | 43 | Request scheduling |
| `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` | - | KV cache allocation |

**Key Methods:**
- `Scheduler.schedule()` - Schedule batch (line 179)
- `Scheduler.add_request()` - Add to queue (line 1097)
- `Scheduler.update_from_output()` - Process outputs (line 861)

### 4. Executor Layer

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `Executor` | `vllm/v1/executor/abstract.py` | 24 | Abstract executor |
| `MultiprocExecutor` | `vllm/v1/executor/multiproc_executor.py` | - | Multi-process |
| `RayDistributedExecutor` | `vllm/v1/executor/ray_distributed_executor.py` | - | Ray distributed |

**Key Methods:**
- `execute_model()` - Execute on scheduler output (line 98)
- `collective_rpc()` - RPC to all workers (line 90)
- `initialize_from_config()` - Init KV cache (line 67)

### 5. Worker Layer

| Class | File | Line | Purpose |
|-------|------|------|---------|
| `Worker` | `vllm/v1/worker/gpu_worker.py` | 45 | GPU worker |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py` | 176 | Model execution |
| `InputBatch` | `vllm/v1/worker/gpu_input_batch.py` | - | Batch preparation |

**Key Methods in GPUModelRunner:**
- `execute_model()` - Forward + sampling (line 2253)
- `_prepare_inputs()` - Prepare inputs
- `_sample()` - Sample tokens
- `propose_draft_token_ids()` - Generate drafts (line 2480)

---

## Speculative Decoding Overview

### Drafter Initialization

**Location**: `vllm/v1/worker/gpu_model_runner.py:276-317`

```python
if self.speculative_config:
    if method == "ngram":
        self.drafter = NgramProposer(self.vllm_config)
    elif method == "suffix":
        self.drafter = SuffixDecodingProposer(self.vllm_config)
    elif method == "suffix_parallel":
        self.drafter = ParallelSuffixDecodingProposer(self.vllm_config)
    elif method == "suffix_remote":
        self.drafter = RemoteSuffixDecodingProposer(self.vllm_config)
    elif self.speculative_config.use_eagle():
        self.drafter = EagleProposer(...)
    elif method == "medusa":
        self.drafter = MedusaProposer(...)
```

### Supported Proposers

| Proposer | File | Type | Description |
|----------|------|------|-------------|
| **EAGLE/EAGLE3** | `vllm/v1/spec_decode/eagle.py:42` | Model-based | Hidden state draft model |
| **Medusa** | `vllm/v1/spec_decode/medusa.py:17` | Model-based | Multi-head prediction |
| **N-gram** | `vllm/v1/spec_decode/ngram_proposer.py:11` | Heuristic | Context pattern match |
| **Suffix** | `vllm/v1/spec_decode/suffix_decoding.py:8` | Heuristic | Suffix tree matching |
| **ParallelSuffix** | `vllm/v1/spec_decode/suffix_decoding_parallel.py` | Heuristic | Batched suffix ops |
| **RemoteSuffix** | `vllm/v1/spec_decode/suffix_decoding_remote.py` | Heuristic | gRPC suffix server |

### Draft Token Flow

**Location**: `gpu_model_runner.py:2480-2648`

```
execute_model()
    |
    +-> Target Model Forward
    |
    +-> Sampler (sample tokens)
    |
    +-> propose_draft_token_ids()
            |
            +-> drafter.propose(...)
            |
            +-> Return draft tokens to scheduler
```

---

## RPC Chain for Snapshot Loading

For distributed cache operations (loading suffix tree snapshots):

```
External Call
    |
    v
LLMEngine.load_suffix_snapshot(snapshot)
    |
    v
EngineCore.load_suffix_snapshot(snapshot)
    |
    v
Executor.collective_rpc("load_suffix_snapshot", args=(snapshot,))
    |
    v
Worker.load_suffix_snapshot(snapshot)  [each worker]
    |
    v
GPUModelRunner.load_suffix_snapshot(snapshot)
    |
    v
drafter.load_snapshot(snapshot)
```

---

## Adding a New Proposer

### Step 1: Create Proposer Class

```python
# vllm/v1/spec_decode/your_proposer.py

class YourProposer:
    def __init__(self, vllm_config: VllmConfig):
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens

    def propose(self, input_batch, sampled_token_ids, ...) -> list[list[int]]:
        """Return draft tokens for each request."""
        ...

    def load_model(self, target_model):
        """Load any required models."""
        pass
```

### Step 2: Register in GPUModelRunner

**File**: `gpu_model_runner.py:276`

```python
elif self.speculative_config.method == "your_method":
    from vllm.v1.spec_decode.your_proposer import YourProposer
    self.drafter = YourProposer(self.vllm_config)
```

### Step 3: Add Dispatch

**File**: `gpu_model_runner.py:2480`

```python
elif self.speculative_config.method == "your_method":
    draft_token_ids = self.drafter.propose(
        input_batch=self.input_batch,
        sampled_token_ids=sampled_token_ids,
        ...)
```

### Step 4: Update Configuration

Add new options in `vllm/config.py` for `SpeculativeConfig` if needed.

---

## Configuration Reference

### Speculative Config Options

```python
speculative_config:
  method: "eagle" | "ngram" | "medusa" | "suffix" | "suffix_remote"
  num_speculative_tokens: int
  draft_model_config: ModelConfig  # For model-based

  # N-gram
  prompt_lookup_min: int
  prompt_lookup_max: int

  # Suffix
  suffix_decoding_max_tree_depth: int
  suffix_decoding_max_spec_factor: float
  suffix_decoding_min_token_prob: float
  suffix_decoding_max_cached_requests: int
  suffix_decoding_use_parallel: bool

  # Runtime
  disable_by_batch_size: int  # Disable above threshold
```

---

## Directory Structure

```
vllm/v1/
+-- engine/
|   +-- llm_engine.py       # LLMEngine
|   +-- core.py             # EngineCore
|   +-- core_client.py      # EngineCoreClient
|   +-- processor.py        # Input processing
+-- core/
|   +-- sched/
|   |   +-- scheduler.py    # Scheduler
|   +-- kv_cache_manager.py
+-- executor/
|   +-- abstract.py         # Base Executor
|   +-- multiproc_executor.py
+-- worker/
|   +-- gpu_worker.py       # Worker
|   +-- gpu_model_runner.py # GPUModelRunner (OWNS DRAFTER)
+-- spec_decode/
|   +-- eagle.py            # EagleProposer
|   +-- medusa.py           # MedusaProposer
|   +-- ngram_proposer.py   # NgramProposer
|   +-- suffix_decoding.py  # SuffixDecodingProposer
|   +-- suffix_decoding_parallel.py
|   +-- suffix_decoding_remote.py
|   +-- rejection_sampler.py
|   +-- metadata.py         # SpecDecodeMetadata
+-- sample/
|   +-- sampler.py
|   +-- rejection_sampler.py
```

---

## Related Skills

- [ArcticInference Cache Management](arctic_inference_cache_managements.md) - Suffix tree cache operations
- [Role of Third-Party Libraries](../role_of_third_party_lib.md) - Integration architecture overview
