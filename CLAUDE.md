## Guidelines for this project.
This project is a modified version of Verl, a reinforcement learning framework.
Specifically, I want to implement the run-ahead rollout strategy and SRT (Speculative Rollout with Tree-Structured Cache) in Verl's rollout.

The idea behind is that common rollout will have a lot of GPU bubble time, steps from the waiting for a subset of samples in a batch that have not finished yet.


## Project Goals

### 1. Run-Ahead Rollout Strategy (Converged)
Have a proper implementation of run-ahead rollout strategy in Verl's rollout.

### 2. SRT Integration (Current Focus)
Integrate suffix decoding (speculative decoding with tree-structured cache) into vLLM Server mode.

---

## Core Problem the Paper Addresses

The paper tackles a computational bottleneck in reinforcement learning (RL) training for language models: the rollout generation phase consumes ~65-70% of total training time. This happens because:

- Token generation is auto-regressive and memory-bound
- Response lengths have a "long-tail" distribution—a few lengthy rollouts stall entire batches, leaving GPUs idle
- Algorithms like GRPO and DAPO require multiple responses per prompt, amplifying the cost

## The Proposed Solution: SRT

Speculative Rollout with Tree-Structured Cache (SRT) accelerates on-policy rollouts by exploiting a key observation: responses to the same prompt across different training epochs are often similar.

The system has three main components:

1. **Per-Prompt Tree-Structured Cache**: Stores previously generated token sequences for each prompt in a tree structure, where paths represent all substrings seen in earlier generations

2. **Speculative Decoding with the Cache**: Uses the cached sequences as a "model-free draft model"—the current policy verifies drafted tokens and accepts them up to the first mismatch

3. **Cache Maintenance Strategies**:
   - **Online updates**: Insert tokens from ongoing rollouts into the cache
   - **Run-ahead generation**: Use idle GPU time (when some sequences finish early) to pre-generate rollouts for upcoming prompts

---

## Previous Integration (SPMD Mode - Retired)

The suffix tree speculation was previously integrated into VERL's training loop for speculative decoding during rollout in SPMD mode.

### Architecture (Old SPMD Mode)

```
RayPPOTrainer (Driver)
├── SuffixTreeManager              # Wraps ParallelSuffixDecodingCache
│   └── ParallelSuffixDecodingCache (ArcticInference)
│
└── Training Loop
    1. Push snapshot to workers (before rollout)
    2. generate_sequences() with speculation
    3. Update trees from rollout results
    4. Save checkpoint (includes suffix tree state)

vLLMRollout Workers
└── load_suffix_snapshot()         # Receives snapshots from trainer
    └── inference_engine.load_snapshot(snapshots, hash_mapping)
```

### Key Files (Old Integration)

| File | Purpose |
|------|---------|
| `verl/trainer/ppo/suffix_tree_manager.py` | SuffixTreeManager class |
| `verl/trainer/ppo/ray_trainer.py` | Training loop integration |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Worker snapshot loading (RETIRED) |
| `verl/workers/fsdp_workers.py` | Worker dispatch |

### SuffixTreeManager API

```python
from verl.trainer.ppo.suffix_tree_manager import SuffixTreeManager, SuffixTreeManagerConfig

# Initialize
config = SuffixTreeManagerConfig(
    enable=True,
    max_tree_depth=64,
    hash_token_count=128,  # For tree sharing
    num_threads=-1,
)
manager = SuffixTreeManager(config, tokenizer)

# Update from rollout results
stats = manager.update_from_rollout(batch)  # batch: DataProto with responses

# Get snapshot for workers
snapshots, hash_mapping = manager.get_snapshot()

# Get selective snapshot (batch-specific trees only)
snapshots, hash_mapping = manager.get_selective_snapshot(hashes=batch_hashes)

# Checkpoint
manager.save("/path/to/checkpoint/suffix_tree")
manager.load("/path/to/checkpoint/suffix_tree")
```

### Training Loop Integration (Old)

```python
# In ray_trainer.py

# 1. Before rollout - push snapshot to workers
if self.suffix_tree_manager.enabled:
    batch_hashes = self._extract_batch_hashes(gen_batch)
    if batch_hashes:
        snapshots, hash_mapping = self.suffix_tree_manager.get_selective_snapshot(batch_hashes)
    else:
        snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()

    if snapshots:
        self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)

# 2. Run generation with speculation
output = self.actor_rollout_wg.generate_sequences(prompts)

# 3. After rollout - update trees
if self.suffix_tree_manager.enabled:
    suffix_stats = self.suffix_tree_manager.update_from_rollout(batch)
    metrics.update(suffix_stats)
```

### Configuration

```yaml
# In training config
suffix_tree:
  enable: true
  max_tree_depth: 64
  hash_token_count: 128
  num_threads: -1
  parallel_threshold: 4
```

### Metrics

| Metric | Description |
|--------|-------------|
| `suffix_tree/num_trees` | Total trees in forest |
| `suffix_tree/requests_started` | New requests started this batch |
| `suffix_tree/tokens_added` | Tokens added this batch |
| `suffix_tree/trees_transferred` | Trees sent to workers |
| `suffix_tree/transfer_bytes` | Bytes transferred |

### Key Design Decisions (Old)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mode | Local (no gRPC server) | Simpler for training |
| Tree sharing | Hash-based | Same prompt shares tree |
| Tokenization | `add_special_tokens=True` | Must match vLLM's BOS |
| State location | Driver process (CPU) | Trees accumulate centrally |
| Push timing | Before every rollout | Fresh speculation patterns |

---

## Current Implementation: Server Mode Integration

**Status**: Core Implementation Complete (Testing/Validation Phase)

The SPMD mode of vLLM has been retired. SRT support has been implemented for **Server mode** vLLM Rollout.

### Implemented Components

| Component | File | Status |
|-----------|------|--------|
| SRT vLLM Server | `recipe/srt/vllm_server.py` | Done |
| vLLM Plugin (patches, proposers) | `recipe/srt/vllm_plugin/` | Done |
| Suffix Tree Manager | `recipe/srt/suffix_tree_manager.py` | Done |
| Standard Training Loop | `ray_trainer.py:_fit_standard()` | Done |
| Runahead Training Loop | `ray_trainer.py:_fit_runahead()` | Done |
| Spec Decode Metrics | `spec_decode_metrics.py` | Done |
| Server Suffix Snapshot Loading | `vllm_rollout.py:load_suffix_snapshot()` | Done |

### Architecture (Server Mode)

```
SRTRayPPOTrainer (Driver)
├── SuffixTreeManager              # Manages tree state on CPU
│   └── ParallelSuffixDecodingCache (ArcticInference)
│
├── _fit_standard() or _fit_runahead()
│   ├── _push_suffix_snapshots()   # Transfer trees before rollout
│   ├── generate_sequences[_with_runahead]()
│   ├── _update_suffix_trees()     # From primary outputs
│   └── _update_suffix_trees_from_secondary()  # From runahead
│
└── AgentLoopManager
    └── vLLMReplica
        └── SRTvLLMHttpServer      # Extended server with SRT patches
            └── load_suffix_snapshot()
```

### Runahead Training Loop

The `_fit_runahead()` method implements a sliding window pattern:
- **Primary batch (tick N)**: Full training, must complete
- **Secondary batch (tick N+1)**: Pre-generation during GPU bubbles

Key features:
- Secondary outputs populate suffix cache for next tick
- Spec decode metrics logged to wandb
- Configurable via `trainer.runahead.*` settings

### Spec Decode Metrics

Metrics collected via Prometheus `/metrics` endpoint:
- `spec_decode/acceptance_rate` - Fraction of drafted tokens accepted
- `spec_decode/tokens_per_step` - Average tokens per forward pass
- `spec_decode/mean_accepted_length` - Tokens accepted per draft round
- `spec_decode/acceptance_rate_pos_N` - Per-position acceptance rates

### Remaining Work

| Task | Priority | Notes |
|------|----------|-------|
| End-to-end validation | High | Test full training loop with SRT enabled |
| Suffix snapshot HTTP transfer | Medium | Currently uses direct Ray calls; may need HTTP API for true server mode isolation |
| Performance benchmarking | Medium | Measure actual speedup from speculation |

### Key Files (Current)

| File | Purpose |
|------|---------|
| `recipe/srt/ray_trainer.py` | SRTRayPPOTrainer with runahead loop |
| `recipe/srt/suffix_tree_manager.py` | Tree management and snapshot API |
| `recipe/srt/vllm_server.py` | SRT-enabled vLLM server classes |
| `recipe/srt/vllm_plugin/` | vLLM patches and proposers |
| `verl/experimental/agent_loop/spec_decode_metrics.py` | Prometheus metrics parsing |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `get_spec_decode_metrics()` |

---

## Reference
See `./claude_docs` for some background information of agentLoop and how the vLLM server rollout works.
See `./vllm/` for vLLM-specific modifications and scheduling policy documentation.

## Guide for executing the implemented function in our environment
+ All the dependency has been installed. Get the virtual environments by `source .venv/bin/activate`
