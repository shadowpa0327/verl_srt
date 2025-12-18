# vLLM V1 Architecture Documentation

This document provides a high-level overview of vLLM V1's architecture, including the speculative decoding proposer system owned by GPUModelRunner.

## Table of Contents

1. [Overview](#overview)
2. [High-Level Request Flow](#high-level-request-flow)
3. [Key Components and File Locations](#key-components-and-file-locations)
4. [Speculative Decoding Architecture](#speculative-decoding-architecture)
5. [Key Modification Points](#key-modification-points)
6. [Directory Structure](#directory-structure)

---

## Overview

vLLM V1 is a high-performance LLM inference engine with a clean separation between scheduling, execution, and output processing. The architecture follows a producer-consumer pattern with asynchronous processing.

Key design principles:
- **Scheduler** decides what to run
- **Executor** coordinates workers
- **ModelRunner** executes the model
- **Sampler** generates tokens
- **Drafter** (optional) proposes speculative tokens

---

## High-Level Request Flow

```
User Request
      │
      ▼
┌─────────────┐
│   LLM()     │  ← Entry point (vllm/entrypoints/llm.py:66)
│ .generate() │
└─────┬───────┘
      │
      ▼
┌─────────────────┐
│   LLMEngine     │  ← Orchestrates processing (vllm/v1/engine/llm_engine.py:45)
│   - Processor   │     Converts inputs to EngineCoreRequests
│   - OutputProc  │     Converts outputs to RequestOutput
└─────┬───────────┘
      │
      ▼
┌──────────────────┐
│  EngineCoreClient│  ← Communication layer (vllm/v1/engine/core_client.py:49)
│   - InprocClient │     In-process (for LLMEngine)
│   - SyncMPClient │     ZMQ multiprocess (for LLM)
│   - AsyncMPClient│     ZMQ async multiprocess (for AsyncLLM)
└─────┬────────────┘
      │
      ▼
┌─────────────────┐
│   EngineCore    │  ← Inner loop (vllm/v1/engine/core.py:63)
│   - Scheduler   │     Schedules requests, manages KV cache
│   - Executor    │     Coordinates worker execution
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│    Executor     │  ← Worker orchestration (vllm/v1/executor/abstract.py:24)
│   - UniProc     │     Single process
│   - Multiproc   │     Multiple processes
│   - Ray         │     Ray distributed
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│     Worker      │  ← GPU execution (vllm/v1/worker/gpu_worker.py:45)
│  - init_device  │     Initialize CUDA device
│  - ModelRunner  │     Execute model forward
└─────┬───────────┘
      │
      ▼
┌──────────────────┐
│  GPUModelRunner  │  ← Model execution (vllm/v1/worker/gpu_model_runner.py:176)
│  - execute_model │     Run forward pass
│  - Sampler       │     Sample tokens
│  - Drafter       │     Propose speculative tokens (optional)
│  - KV Cache      │     Manage attention cache
└──────────────────┘
```

---

## Key Components and File Locations

### 1. Entry Points

| Component | File | Purpose |
|-----------|------|---------|
| `LLM` | `vllm/entrypoints/llm.py:66` | Main offline inference class |
| `AsyncLLMEngine` | `vllm/v1/engine/async_llm.py` | Async inference for serving |

**Key Methods in `LLM`:**
- `__init__()` - Creates LLMEngine from EngineArgs
- `generate()` - Main generation method (line 335)
- `_validate_and_add_requests()` - Prepares and queues requests (line 1466)
- `_run_engine()` - Main execution loop (line 1578)

### 2. Engine Layer

| Component | File | Purpose |
|-----------|------|---------|
| `LLMEngine` | `vllm/v1/engine/llm_engine.py:45` | Coordinates processing |
| `Processor` | `vllm/v1/engine/processor.py` | Input processing |
| `OutputProcessor` | `vllm/v1/engine/output_processor.py` | Output detokenization |

**Key Methods in `LLMEngine`:**
- `from_engine_args()` - Factory method (line 158)
- `add_request()` - Queue a new request (line 213)
- `step()` - Execute one scheduling step (line 257)

### 3. Core Scheduling

| Component | File | Purpose |
|-----------|------|---------|
| `EngineCore` | `vllm/v1/engine/core.py:63` | Main scheduling loop |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py:43` | Request scheduling |
| `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` | KV cache allocation |
| `BlockPool` | `vllm/v1/core/block_pool.py` | Block memory management |

**Key Methods in `Scheduler`:**
- `schedule()` - Schedule next batch (line 179)
- `add_request()` - Add request to waiting queue (line 1097)
- `update_from_output()` - Process model outputs (line 861)
- `_try_schedule_encoder_inputs()` - Handle multimodal (line 709)

### 4. Executor Layer

| Component | File | Purpose |
|-----------|------|---------|
| `Executor` | `vllm/v1/executor/abstract.py:24` | Abstract executor |
| `MultiprocExecutor` | `vllm/v1/executor/multiproc_executor.py` | Multi-process execution |
| `RayDistributedExecutor` | `vllm/v1/executor/ray_distributed_executor.py` | Ray-based distributed |

**Key Methods in `Executor`:**
- `execute_model()` - Execute model on scheduler output (line 98)
- `collective_rpc()` - RPC to all workers (line 90)
- `initialize_from_config()` - Initialize KV cache (line 67)

### 5. Worker Layer

| Component | File | Purpose |
|-----------|------|---------|
| `Worker` | `vllm/v1/worker/gpu_worker.py:45` | GPU worker |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py:176` | Model execution |
| `InputBatch` | `vllm/v1/worker/gpu_input_batch.py` | Batch preparation |

**Key Methods in `Worker`:**
- `init_device()` - Initialize CUDA device (line 156)
- `initialize_cache()` - Set up KV cache (line 151)
- `execute_model()` - Run model forward (delegated to ModelRunner)

**Key Methods in `GPUModelRunner`:**
- `execute_model()` - Run forward pass + sampling (line 2253)
- `_prepare_inputs()` - Prepare model inputs
- `_sample()` - Sample next tokens
- `propose_draft_token_ids()` - Generate draft tokens (line 2480)

### 6. Data Structures

| Component | File | Purpose |
|-----------|------|---------|
| `Request` | `vllm/v1/request.py:26` | Internal request representation |
| `EngineCoreRequest` | `vllm/v1/engine/__init__.py` | Request for EngineCore |
| `SchedulerOutput` | `vllm/v1/core/sched/output.py` | Scheduler's batch output |
| `ModelRunnerOutput` | `vllm/v1/outputs.py:98` | Model execution output |

**Key Fields in `Request`:**
- `request_id` - Unique identifier
- `prompt_token_ids` - Input tokens
- `sampling_params` - Sampling configuration
- `num_computed_tokens` - Tokens already processed
- `status` - RequestStatus enum

### 7. Attention & KV Cache

| Component | File | Purpose |
|-----------|------|---------|
| `AttentionBackend` | `vllm/v1/attention/backends/` | Attention implementations |
| `FlashAttention` | `vllm/v1/attention/backends/flash_attn.py` | Flash attention |
| `FlashInfer` | `vllm/v1/attention/backends/flashinfer.py` | FlashInfer backend |
| `KVCacheConfig` | `vllm/v1/kv_cache_interface.py` | KV cache configuration |

### 8. Sampling

| Component | File | Purpose |
|-----------|------|---------|
| `Sampler` | `vllm/v1/sample/sampler.py` | Token sampling |
| `SamplingMetadata` | `vllm/v1/sample/metadata.py` | Sampling state |
| `LogitsProcessor` | `vllm/v1/sample/logits_processor/` | Logits manipulation |

---

## Speculative Decoding Architecture

### Overview

Speculative decoding accelerates inference by using a fast "drafter" to propose multiple tokens, then verifying them in parallel with the target model. The GPUModelRunner owns the drafter and manages the entire speculative decoding flow.

### Complete Flow with Speculative Decoding

```
┌──────────────────────────────────────────────────────────────┐
│                        Worker                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   GPUModelRunner                        │  │
│  │                                                         │  │
│  │  execute_model()                                        │  │
│  │       │                                                 │  │
│  │       ▼                                                 │  │
│  │  ┌──────────────┐                                       │  │
│  │  │ Target Model │ ← Run forward pass                    │  │
│  │  │   Forward    │                                       │  │
│  │  └──────┬───────┘                                       │  │
│  │         │                                               │  │
│  │         ▼                                               │  │
│  │  ┌──────────────┐                                       │  │
│  │  │   Sampler    │ ← Sample tokens from logits           │  │
│  │  └──────┬───────┘                                       │  │
│  │         │                                               │  │
│  │         ▼                                               │  │
│  │  ┌──────────────────────────────┐                       │  │
│  │  │        Drafter               │                       │  │
│  │  │  (if spec decode enabled)    │                       │  │
│  │  │                              │                       │  │
│  │  │  ┌─────────────────────────┐ │                       │  │
│  │  │  │  EagleProposer          │ │ ← Model-based         │  │
│  │  │  │  MedusaProposer         │ │ ← Model-based         │  │
│  │  │  │  NgramProposer          │ │ ← Heuristic           │  │
│  │  │  │  SuffixDecodingProposer │ │ ← Heuristic           │  │
│  │  │  └─────────────────────────┘ │                       │  │
│  │  └──────────────────────────────┘                       │  │
│  │                                                         │  │
│  │  ┌──────────────────┐                                   │  │
│  │  │ RejectionSampler │ ← Verify draft tokens             │  │
│  │  │  (next step)     │                                   │  │
│  │  └──────────────────┘                                   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Supported Proposer Methods

| Method | File | Type | Description |
|--------|------|------|-------------|
| **EAGLE/EAGLE3** | `vllm/v1/spec_decode/eagle.py:42` | Model-based | Uses draft model with hidden states |
| **Medusa** | `vllm/v1/spec_decode/medusa.py:17` | Model-based | Multi-head prediction |
| **N-gram** | `vllm/v1/spec_decode/ngram_proposer.py:11` | Heuristic | Pattern matching in context |
| **Suffix Decoding** | `vllm/v1/spec_decode/suffix_decoding.py:8` | Heuristic | Suffix tree matching |
| **Parallel Suffix** | `vllm/v1/spec_decode/suffix_decoding_parallel.py` | Heuristic | Batch suffix operations |
| **Remote Suffix** | `vllm/v1/spec_decode/suffix_decoding_remote.py` | Heuristic | gRPC-based suffix server |

### Drafter Initialization

Location: `vllm/v1/worker/gpu_model_runner.py:276-317`

```python
# In GPUModelRunner.__init__()
if self.speculative_config and get_pp_group().is_last_rank:
    if self.speculative_config.method == "ngram":
        self.drafter = NgramProposer(self.vllm_config)
    elif self.speculative_config.method == "suffix":
        if use_parallel:
            self.drafter = ParallelSuffixDecodingProposer(self.vllm_config)
        else:
            self.drafter = SuffixDecodingProposer(self.vllm_config)
    elif self.speculative_config.method == "suffix_remote":
        self.drafter = RemoteSuffixDecodingProposer(self.vllm_config)
    elif self.speculative_config.use_eagle():
        self.drafter = EagleProposer(self.vllm_config, self.device, self)
    elif self.speculative_config.method == "medusa":
        self.drafter = MedusaProposer(vllm_config, device)
    self.rejection_sampler = RejectionSampler()
```

### Proposer Interface

All proposers share a common interface pattern:

```python
class Proposer:
    def __init__(self, vllm_config: VllmConfig):
        """Initialize proposer with configuration"""
        ...

    def propose(self, ...) -> list[list[int]]:
        """Returns draft token IDs for each request in batch"""
        ...

    def load_model(self, target_model: nn.Module) -> None:
        """Load draft model (model-based proposers only)"""
        ...
```

### Draft Token Invocation Flow

Location: `vllm/v1/worker/gpu_model_runner.py:2480-2648`

```python
def propose_draft_token_ids(
    self,
    scheduler_output,
    sampled_token_ids,      # Either torch.Tensor or list[list[int]]
    sampling_metadata,
    hidden_states,          # From target model
    sample_hidden_states,
    aux_hidden_states,      # For EAGLE3
    spec_decode_metadata,
    common_attn_metadata,
) -> Union[list[list[int]], torch.Tensor]:

    # Check batch size threshold
    if disable_by_batch_size and batch_size > threshold:
        return [[] for _ in sampled_token_ids]

    if self.speculative_config.method == "ngram":
        draft_token_ids = self.drafter.propose(
            sampled_token_ids,
            self.input_batch.req_ids,
            self.input_batch.num_tokens_no_spec,
            self.input_batch.token_ids_cpu,
            self.input_batch.spec_decode_unsupported_reqs)

    elif self.speculative_config.method == "suffix":
        draft_token_ids = self.drafter.propose(
            input_batch=self.input_batch,
            sampled_token_ids=sampled_token_ids)

    elif self.speculative_config.method == "medusa":
        draft_token_ids = self.drafter.propose(
            target_hidden_states=hidden_states,
            sampling_metadata=sampling_metadata)

    elif self.speculative_config.use_eagle():
        # Prepare inputs based on padded vs non-padded batch
        next_token_ids = self.drafter.prepare_next_token_ids_*(...)

        # Prepare attention metadata for draft model
        common_attn_metadata, token_indices = self.drafter.prepare_inputs*(...)

        # Run draft model
        draft_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            next_token_ids=next_token_ids,
            last_token_indices=token_indices_to_sample,
            sampling_metadata=sampling_metadata,
            common_attn_metadata=common_attn_metadata,
            mm_embeds=mm_embeds)

    return draft_token_ids
```

### EagleProposer Details

Location: `vllm/v1/spec_decode/eagle.py:42`

EAGLE uses a lightweight draft model that takes hidden states from the target model.

**Key Fields:**
```python
self.model              # Draft model (loaded in load_model())
self.num_speculative_tokens  # Number of draft tokens to generate
self.tree_choices       # Token tree structure for tree attention
self.hidden_states      # Persistent buffer for CUDA graphs
self.input_ids          # Persistent buffer for CUDA graphs
self.positions          # Persistent buffer for CUDA graphs
```

**Key Methods:**

| Method | Line | Purpose |
|--------|------|---------|
| `propose()` | 155 | Main proposal method |
| `propose_tree()` | 539 | Tree-based drafting for multi-token speculation |
| `prepare_inputs()` | 706 | Prepare inputs accounting for rejected tokens |
| `prepare_inputs_padded()` | 486 | Prepare inputs for padded batch (GPU-only) |
| `prepare_next_token_ids_cpu()` | 387 | Prepare next tokens from CPU |
| `prepare_next_token_ids_padded()` | 419 | Prepare next tokens (GPU tensor) |
| `load_model()` | 809 | Load and share embeddings with target |

**Proposal Algorithm:**
```python
def propose(self, target_token_ids, target_positions,
            target_hidden_states, next_token_ids, ...):

    # 1. Shift input ids by one token
    # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
    self.input_ids[:num_tokens - 1] = target_token_ids[1:]

    # Replace last with next token
    # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
    self.input_ids[last_token_indices] = next_token_ids

    # 2. Build attention metadata for draft model
    attn_metadata = attn_metadata_builder.build_for_drafting(...)

    # 3. Run draft model forward
    ret_hidden_states = self.model(
        input_ids=input_ids,
        positions=self.positions,
        hidden_states=self.hidden_states,
    )

    # 4. Sample first draft token
    logits = self.model.compute_logits(sample_hidden_states)
    draft_token_ids = logits.argmax(dim=-1)

    # 5. Generate remaining draft tokens iteratively
    for token_index in range(self.num_speculative_tokens - 1):
        # Update positions, slot mapping
        positions += 1

        # Rebuild attention metadata
        attn_metadata = attn_metadata_builder.build_for_drafting(...)

        # Run model, sample next draft token
        ...

    return draft_token_ids  # [batch_size, num_speculative_tokens]
```

### NgramProposer Details

Location: `vllm/v1/spec_decode/ngram_proposer.py:11`

N-gram proposer finds matching patterns in the context and proposes continuation tokens.

**Key Fields:**
```python
self.min_n   # Minimum n-gram length to match
self.max_n   # Maximum n-gram length to match
self.k       # Number of tokens to propose after match
```

**Key Methods:**

| Method | Line | Purpose |
|--------|------|---------|
| `propose()` | 124 | Main proposal method |
| `batch_propose()` | 62 | Numba-accelerated batch processing |
| `_find_longest_matched_ngram_and_propose_tokens()` | 191 | Core KMP-based algorithm |

**Algorithm:**
```
1. Find longest n-gram in [min_n, max_n] that matches suffix of context
2. Use KMP algorithm for efficient pattern matching
3. Extract k tokens following the match
4. Return as draft tokens
```

### MedusaProposer Details

Location: `vllm/v1/spec_decode/medusa.py:17`

Medusa uses multiple prediction heads to generate draft tokens in parallel.

**Key Methods:**

| Method | Line | Purpose |
|--------|------|---------|
| `propose()` | 37 | Forward through heads, argmax each |
| `load_model()` | 52 | Load Medusa heads |
| `dummy_run()` | 59 | Warmup run |

**Proposal Flow:**
```python
def propose(self, target_hidden_states, sampling_metadata):
    # Run through Medusa heads (each head predicts one position)
    blocks = self.model(target_hidden_states)
    logits = self.model.compute_logits(blocks)

    # Argmax for each head
    draft_tokens = [logit.argmax(dim=-1).tolist() for logit in logits]

    return [list(row) for row in zip(*draft_tokens)]
```

### SuffixDecodingProposer Details

Location: `vllm/v1/spec_decode/suffix_decoding.py:8`

Uses suffix trees from Arctic Inference to find common patterns and propose continuations.

**Key Fields:**
```python
self.suffix_cache       # SuffixDecodingCache from Arctic Inference
self.max_tree_depth     # Maximum depth for suffix matching
self.max_spec_factor    # Maximum speculation factor
self.min_token_prob     # Minimum token probability threshold
```

**Key Methods:**

| Method | Line | Purpose |
|--------|------|---------|
| `propose()` | 33 | Main proposal using suffix trees |
| `load_model()` | 100 | No-op (no model to load) |

**Proposal Flow:**
```python
def propose(self, input_batch, sampled_token_ids):
    for i, sampled_ids in enumerate(sampled_token_ids):
        req_id = input_batch.req_ids[i]

        # Start/update request in suffix cache
        if req_id not in self.suffix_cache.active_requests:
            prompt_token_ids = input_batch.token_ids_cpu[...]
            self.suffix_cache.start_request(req_id, prompt_token_ids)

        # Add newly sampled tokens to cache
        self.suffix_cache.add_active_response(req_id, sampled_ids)

        # Extract recent pattern for matching
        pattern = input_batch.token_ids_cpu[i, -max_tree_depth:num_tokens]

        # Speculate using suffix tree
        draft = self.suffix_cache.speculate(
            req_id, pattern,
            max_spec_tokens=...,
            max_spec_factor=...,
            min_token_prob=...)

        draft_token_ids.append(draft.token_ids)

    # Clean up finished requests
    for req_id in (active_requests - current_requests):
        self.suffix_cache.stop_request(req_id)

    return draft_token_ids
```

### Rejection Sampling

Location: `vllm/v1/sample/rejection_sampler.py:23`

After target model verification, draft tokens are accepted/rejected using the algorithm from [Leviathan et al.](https://arxiv.org/abs/2211.17192)

**Terminology:**
- **Accepted tokens**: Tokens accepted based on draft vs target probability ratio
- **Recovered tokens**: Tokens sampled from adjusted distribution when rejected
- **Bonus tokens**: Extra token added when all drafts accepted

**Flow:**
```
Target Model Output (with draft tokens in context)
         │
         ▼
┌─────────────────────────────────────┐
│       RejectionSampler.forward()    │
│                                     │
│  For each draft position:           │
│    p_target = target_probs[token]   │
│    p_draft = draft_probs[token]     │
│    u = uniform(0, 1)                │
│                                     │
│    if p_target / p_draft >= u:      │
│      ACCEPT draft token             │
│    else:                            │
│      REJECT → sample from           │
│      max(p_target - p_draft, 0)     │
│                                     │
│  If all accepted → add bonus token  │
└─────────────────────────────────────┘
         │
         ▼
   Final token IDs (accepted + recovered + bonus)
```

**Key Classes:**

| Class | File | Purpose |
|-------|------|---------|
| `RejectionSampler` | `rejection_sampler.py:23` | Main rejection sampling logic |
| `SpecDecodeMetadata` | `metadata.py:10` | Tracks draft token positions |

### SpecDecodeMetadata

Location: `vllm/v1/spec_decode/metadata.py:10`

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor      # [num_tokens] - flattened draft tokens
    num_draft_tokens: list[int]        # [batch_size] - drafts per request
    cu_num_draft_tokens: torch.Tensor  # [batch_size] - cumulative counts
    target_logits_indices: torch.Tensor # [num_tokens] - for verification
    bonus_logits_indices: torch.Tensor  # [batch_size] - for bonus tokens
    logits_indices: torch.Tensor        # [num_tokens + batch_size] - combined
```

### Speculative Decoding Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Step N (with drafts)                        │
│                                                                 │
│  SchedulerOutput                                                │
│    └── num_scheduled_tokens includes draft tokens               │
│                                                                 │
│  GPUModelRunner.execute_model()                                 │
│    │                                                            │
│    ├── _prepare_inputs() → spec_decode_metadata                 │
│    │     └── SpecDecodeMetadata:                                │
│    │           - draft_token_ids: [d1, d2, d3, ...]            │
│    │           - num_draft_tokens: [2, 3, 1, ...]              │
│    │           - cu_num_draft_tokens: [2, 5, 6, ...]           │
│    │                                                            │
│    ├── Target Model Forward (includes draft tokens)             │
│    │                                                            │
│    ├── _sample() with spec_decode_metadata                      │
│    │     └── RejectionSampler.forward()                         │
│    │           - Compares target vs draft probs                 │
│    │           - Returns accepted/recovered tokens              │
│    │                                                            │
│    └── propose_draft_token_ids() → self._draft_token_ids        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Scheduler                                   │
│                                                                 │
│  take_draft_token_ids() → DraftTokenIds                         │
│    - req_ids: ["req1", "req2", ...]                            │
│    - draft_token_ids: [[d1, d2], [d3, d4, d5], ...]            │
│                                                                 │
│  Used in next schedule() to include draft tokens                │
└─────────────────────────────────────────────────────────────────┘
```

### Speculative Decoding Configuration

Via `VllmConfig.speculative_config`:

```python
speculative_config:
  method: "eagle" | "ngram" | "medusa" | "suffix" | "suffix_remote"
  num_speculative_tokens: int  # Number of draft tokens
  draft_model_config: ModelConfig  # For model-based proposers

  # N-gram specific
  prompt_lookup_min: int
  prompt_lookup_max: int

  # Suffix specific
  suffix_decoding_max_tree_depth: int
  suffix_decoding_max_spec_factor: float
  suffix_decoding_min_token_prob: float
  suffix_decoding_max_cached_requests: int
  suffix_decoding_use_parallel: bool

  # EAGLE specific
  speculative_token_tree: str  # Tree structure like "[(0,), (0,0), ...]"
  disable_padded_drafter_batch: bool
  enforce_eager: bool

  # Runtime control
  disable_by_batch_size: int  # Disable spec decode above this batch size
```

---

## Key Modification Points

### General Architecture

| Task | File | Location |
|------|------|----------|
| Add a new scheduling policy | `vllm/v1/core/sched/scheduler.py` | `schedule()` method |
| Modify KV cache behavior | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots()`, `get_computed_blocks()` |
| Add a new attention backend | `vllm/v1/attention/backends/` | Implement `AttentionBackend` interface |
| Modify token sampling | `vllm/v1/sample/sampler.py` | `forward()` method |
| Modify request processing | `vllm/v1/engine/processor.py` | `process_inputs()` method |
| Add worker-level features | `vllm/v1/worker/gpu_worker.py` | `execute_model()` or new RPC methods |

### Speculative Decoding

| Task | File | Location |
|------|------|----------|
| Add new proposer type | `gpu_model_runner.py` | `__init__` (line 276-317) |
| Add new proposer dispatch | `gpu_model_runner.py` | `propose_draft_token_ids` (line 2480) |
| Modify draft generation | `vllm/v1/spec_decode/<proposer>.py` | `propose()` method |
| Modify rejection logic | `rejection_sampler.py` | `forward()` (line 46) |
| Modify verification kernels | `rejection_sampler.py` | `rejection_*_sample_kernel` |
| Change spec decode metadata | `metadata.py` | `SpecDecodeMetadata` class |
| Add unsupported sampling params | `utils.py` | `is_spec_decode_unsupported()` |

---

## Directory Structure

### V1 Core Structure

```
vllm/v1/
├── __init__.py
├── attention/              # Attention implementations
│   ├── backends/           # FlashAttn, FlashInfer, etc.
│   └── __init__.py
├── core/                   # Core scheduling components
│   ├── block_pool.py       # Block memory pool
│   ├── kv_cache_manager.py # KV cache allocation
│   ├── kv_cache_utils.py   # KV cache utilities
│   └── sched/              # Scheduler
│       ├── scheduler.py    # Main scheduler
│       ├── output.py       # SchedulerOutput
│       └── request_queue.py # Request queues
├── engine/                 # Engine components
│   ├── async_llm.py        # Async LLM engine
│   ├── core.py             # EngineCore (inner loop)
│   ├── core_client.py      # EngineCoreClient
│   ├── llm_engine.py       # LLMEngine
│   ├── processor.py        # Input processing
│   └── output_processor.py # Output processing
├── executor/               # Execution backends
│   ├── abstract.py         # Executor interface
│   ├── multiproc_executor.py # Multi-process
│   └── ray_distributed_executor.py # Ray
├── worker/                 # Workers
│   ├── gpu_worker.py       # GPU worker
│   ├── gpu_model_runner.py # Model execution (OWNS DRAFTER)
│   ├── gpu_input_batch.py  # Input batch handling
│   └── worker_base.py      # Worker interface
├── sample/                 # Sampling
│   ├── sampler.py          # Token sampler
│   ├── rejection_sampler.py # Spec decode verification
│   └── metadata.py         # SamplingMetadata
├── spec_decode/            # Speculative decoding proposers
│   ├── __init__.py
│   ├── eagle.py            # EagleProposer
│   ├── medusa.py           # MedusaProposer
│   ├── ngram_proposer.py   # NgramProposer
│   ├── suffix_decoding.py  # SuffixDecodingProposer
│   ├── suffix_decoding_parallel.py  # ParallelSuffixDecodingProposer
│   ├── suffix_decoding_remote.py    # RemoteSuffixDecodingProposer
│   ├── metadata.py         # SpecDecodeMetadata
│   ├── metrics.py          # Spec decode metrics
│   └── utils.py            # is_spec_decode_unsupported()
├── kv_cache_interface.py   # KV cache specs
├── outputs.py              # Output data structures
└── request.py              # Request class
```

### Important Patterns

#### Request Lifecycle
```
WAITING → RUNNING → FINISHED_*
   │         │
   └── WAITING_FOR_FSM (structured output)
   └── WAITING_FOR_REMOTE_KVS (P/D disaggregation)
   └── PREEMPTED → (back to WAITING)
```

#### Execution Flow
```python
# In EngineCore.step()
scheduler_output = self.scheduler.schedule()
model_output = self.model_executor.execute_model(scheduler_output)
engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

#### Communication Pattern (Multiprocess)
```
EngineCoreClient ←→ [ZMQ] ←→ EngineCoreProc
     │                            │
     └── add_request()            └── process_input_queue()
     └── get_output()             └── step()
```

---

## Quick Reference: Adding a New Proposer

1. **Create proposer class** in `vllm/v1/spec_decode/your_proposer.py`:
```python
class YourProposer:
    def __init__(self, vllm_config: VllmConfig):
        self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        ...

    def propose(self, input_batch, sampled_token_ids, ...) -> list[list[int]]:
        """Return draft tokens for each request"""
        ...

    def load_model(self, target_model):
        """Load any required models"""
        pass
```

2. **Register in GPUModelRunner.__init__** (`gpu_model_runner.py:276`):
```python
elif self.speculative_config.method == "your_method":
    from vllm.v1.spec_decode.your_proposer import YourProposer
    self.drafter = YourProposer(self.vllm_config)
```

3. **Add dispatch in propose_draft_token_ids** (`gpu_model_runner.py:2480`):
```python
elif self.speculative_config.method == "your_method":
    draft_token_ids = self.drafter.propose(
        input_batch=self.input_batch,
        sampled_token_ids=sampled_token_ids,
        ...)
```

4. **Update configuration** if needed in `vllm/config.py` for `SpeculativeConfig`.
