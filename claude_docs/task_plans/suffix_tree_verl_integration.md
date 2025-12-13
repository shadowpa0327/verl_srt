# Suffix Tree Speculation Integration Plan

> **Status**: Implemented
> **Created**: 2025-12-10
> **Related**: [`../suffix_tree_speculation.md`](../suffix_tree_speculation.md)

## Overview

Integrate suffix tree speculation into VERL's training loop. The trainer maintains a live `SuffixTreeManager` that accumulates Q/A patterns from rollouts and pushes snapshots to vLLM workers for speculative decoding.

## Architecture

```
RayPPOTrainer (Driver)
├── SuffixTreeManager          ← NEW: Wraps ParallelSuffixDecodingCache
│   └── ParallelSuffixDecodingCache (ArcticInference)
│
└── Training Loop
    1. Push snapshot to workers (before rollout)
    2. generate_sequences()
    3. Update trees from rollout (after line 1175)
    4. Save checkpoint (includes suffix tree state)

vLLMRollout Workers
└── load_suffix_snapshot()     ← NEW: Receives snapshots from trainer
    └── self.inference_engine.load_snapshot(snapshots, hash_mapping)
```

## Files to Create/Modify

### 1. NEW: `verl/trainer/ppo/suffix_tree_manager.py`

New class wrapping `ParallelSuffixDecodingCache`:

```python
class SuffixTreeManager:
    def __init__(self, config, tokenizer): ...
    def update_from_rollout(self, batch: DataProto) -> Dict[str, Any]: ...
    def get_snapshot(self) -> Tuple[List[Tuple[int, bytes]], Dict[str, int]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> bool: ...
```

Key responsibilities:
- Initialize `ParallelSuffixDecodingCache` with config params
- Extract prompts/responses from `DataProto` batch
- Tokenize with BOS token (matching vLLM)
- Call `cache.start_request()` / `cache.add_tokens()` / `cache.stop_request()`
- Serialize/deserialize for checkpointing

### 2. MODIFY: `verl/trainer/ppo/ray_trainer.py`

**Initialization** (after line ~351):
```python
self.suffix_tree_manager = SuffixTreeManager(suffix_config, self.tokenizer)
```

**Before rollout** (before line 1138):
```python
if self.suffix_tree_manager.enabled:
    snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
    if snapshots:
        self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)
```

**After rollout** (after line 1175):
```python
if self.suffix_tree_manager.enabled:
    self.suffix_tree_manager.update_from_rollout(batch)
```

**Checkpoint save/load**:
```python
# In _save_checkpoint() ~line 926
self.suffix_tree_manager.save(os.path.join(folder, "suffix_tree"))

# In _load_checkpoint() ~line 990
self.suffix_tree_manager.load(os.path.join(folder, "suffix_tree"))
```

### 3. MODIFY: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

**Add method** (after line 603):
```python
async def load_suffix_snapshot(
    self,
    snapshots: List[Tuple[int, bytes]],
    hash_mapping: Dict[str, int],
) -> None:
    """Load suffix tree snapshot for speculative decoding."""
    if snapshots:
        self.inference_engine.load_snapshot(snapshots, hash_mapping)
```

### 4. MODIFY: `verl/workers/config/rollout.py`

**Extend SuffixDecodingConfig** (lines 113-124):
```python
@dataclass
class SuffixDecodingConfig(BaseConfig):
    enable: bool = False
    server_host: str = "localhost"
    server_port: int = 50051
    num_speculative_tokens: int = 5
    max_tree_depth: int = 64          # Changed from 24
    auto_manage_server: bool = True
    server_mode: bool = True
    # NEW fields:
    hash_token_count: int = 128       # Tokens to hash for tree sharing
    update_frequency: int = 1         # Update trees every N batches
    push_frequency: int = 1           # Push to workers every N batches
```

### 5. MODIFY: Worker dispatch (fsdp_workers.py or similar)

Add `load_suffix_snapshot` to the worker dispatch so it can be called from trainer via Ray.

## Implementation Sequence

1. **Create `SuffixTreeManager`** - Core class with cache management
2. **Update config** - Add new fields to `SuffixDecodingConfig`
3. **Add worker method** - `load_suffix_snapshot()` in vLLMRollout
4. **Integrate trainer** - Init, pre-rollout push, post-rollout update, checkpoint

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mode | Local (`server_mode=false`) | Avoids gRPC server complexity |
| Tree sharing | Hash-based (`hash_token_count=128`) | Same prompt shares tree |
| Tokenization | `add_special_tokens=True` | Must match vLLM's BOS token |
| State location | Driver process (CPU) | Trees accumulate centrally |
| Quality filter | All responses | Include every Q/A pair (no advantage filtering) |
| Push timing | Before every rollout | Freshest speculation patterns for each batch |

## Critical API References

**ArcticInference (parallel_cache.py)**:
```python
cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)
cache.start_request(req_id, prompt_tokens)  # With BOS
cache.add_tokens(req_id, response_tokens)
cache.stop_request(req_id)
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)
```

**vLLM (llm.py:1415-1444)**:
```python
llm.load_snapshot(snapshots, hash_mapping)
# Propagates to: model_runner.drafter.load_snapshot()
```

## Data Flow

```
1. Trainer: batch = dataloader.next()           # Prompts only
2. Trainer: push_snapshot() →                   # Before rollout
   Workers: load_suffix_snapshot()
3. Workers: generate_sequences() →              # With speculation
4. Trainer: batch.union(gen_output)             # Q/A pairs ready (line 1175)
5. Trainer: suffix_manager.update_from_rollout(batch)  # Add to trees
6. Trainer: _save_checkpoint()                  # Persist trees
```

## Testing Strategy

1. Unit test `SuffixTreeManager` with mock tokenizer
2. Integration test snapshot creation/loading
3. End-to-end test with small training run
