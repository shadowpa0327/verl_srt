# Design: Multi-Strategy Secondary Data Feeding for SRT Run-Ahead

**For External Review**

---

## Background: SRT and Run-Ahead Concepts

This section provides context for reviewers unfamiliar with the SRT system.

### The Problem: GPU Bubbles in RL Rollouts

Reinforcement learning training for language models spends 65-70% of time on rollout generation. A key inefficiency is **GPU bubbles**: when some requests in a batch complete early, the GPU sits idle waiting for longer requests.

```
WITHOUT RUNAHEAD
════════════════

Primary Batch (8 requests):
  P0 ████████░░░░░░░░░░░░░░░░░░░░░░░░░░  (16 tokens)
  P1 ████████░░░░░░░░░░░░░░░░░░░░░░░░░░  (16 tokens)
  P2 ████████████░░░░░░░░░░░░░░░░░░░░░░  (24 tokens)
  P3 ████████████░░░░░░░░░░░░░░░░░░░░░░  (24 tokens)
  P4 ████████████████████████████████████  (200 tokens)
  P5 ██████████████████████████████████░░  (180 tokens)
  P6 ████████████████████████████████░░░░  (160 tokens)
  P7 ██████████████████████████████░░░░░░  (140 tokens)

     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           WASTED GPU CAPACITY
```

### Solution 1: Run-Ahead Rollout

**Run-ahead** fills these bubbles by speculatively generating responses for **future batches** while the current batch is still running.

```
WITH RUNAHEAD
═════════════

Primary Batch (tick N):
  P0-P7 as above...

Secondary Batch (tick N+1, opportunistic):
  S0         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (completed!)
  S1         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (completed!)
  S2         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████  (ABORTED when primary done)
  S3         ▓▓▓▓▓▓▓▓▓▓▓▓████████████  (ABORTED when primary done)

GPU Utilization: ~85% (vs ~50% without runahead)
```

Key points:
- Primary batch **must complete** (used for training)
- Secondary batch is **opportunistic** (fills bubbles, may be aborted)
- Secondary outputs populate the **suffix cache** for when they become primary

> For detailed run-ahead architecture, see: `claude_docs/runahead_rollout_design.md`

### Solution 2: SRT (Speculative Rollout with Tree-Structured Cache)

**SRT** accelerates generation by caching previously generated token sequences per prompt:

1. **Suffix Tree Cache**: Store all token sequences seen for each prompt in a tree structure
2. **Speculative Decoding**: Use cached sequences as a "model-free draft" - verify with current policy
3. **Online Updates**: Insert new responses into cache immediately after generation

The key insight: responses to the same prompt across training epochs are often similar, so speculation has high acceptance rates.

```
Suffix Tree for prompt "What is 2+2?":
                    [ROOT]
                   /      \
                 "The"   "2+2"
                 /          \
              "answer"     "equals"
               /              \
             "is"           "four"
              |
            "four"

When generating for this prompt, SRT drafts token sequences from this tree
and verifies them in parallel with the model.
```

> For detailed SRT architecture, see: `claude_docs/srt_development.md`

### Current Data Feeding: Sliding Window

The current implementation uses a **sliding window** pattern:

```
Tick 1: Batch_1 PRIMARY  + Batch_2 SECONDARY
        └─ Batch_1: full training
        └─ Batch_2: ~50% generated → cache populated

Tick 2: Batch_2 PRIMARY  + Batch_3 SECONDARY
        └─ Batch_2: STARTS WITH CACHE from Tick 1 → faster!
        └─ Batch_3: generate speculatively

Tick 3: Batch_3 PRIMARY  + Batch_4 SECONDARY
        └─ Batch_3: STARTS WITH CACHE from Tick 2 → faster!
        └─ Batch_4: generate speculatively
```

With DAPO/GRPO training (`n=16` responses per prompt):
- Each batch of 32 prompts becomes 512 samples (32 × 16)
- Secondary uses the same `n` as primary

---

## Problem Statement

If run-ahead can process ~256 extra secondary samples per tick, should we:

- **Depth**: 16 unique prompts × 16 repetitions (current approach)
- **Breadth**: 256 unique prompts × 1 repetition (exploration-first)
- **Hybrid**: 64 unique prompts × 4 repetitions (balanced)

This design proposes a configurable system to experiment with these strategies.

---

## Design Goals

1. **Configurability**: Support all three strategies via config parameters
2. **Multi-batch lookahead**: Enable sampling from batches N+1, N+2, N+3, ...
3. **Decay-based allocation**: Prioritize nearer batches in breadth/hybrid modes
4. **Independent n values**: Decouple secondary repetition from primary training `n`
5. **Observability**: Track per-batch cache coverage and utilization metrics

---

## Configuration Schema

### New Parameters

```yaml
trainer:
  enable_runahead: true
  runahead:
    # === EXISTING PARAMETERS ===
    load_threshold: 32          # Admit secondary when server_load < threshold
    max_queue_size: 256         # Max pending secondary items
    secondary_priority: 10      # Lower priority than primary (0)
    abort_grace_s: 1.0          # Wait time for partial outputs after abort

    # === NEW PARAMETERS ===

    # Strategy selection
    secondary_strategy: "depth"  # Options: "depth" | "breadth" | "hybrid"

    # Budget control
    secondary_budget: 256        # Total secondary samples per tick
                                 # If 0 or null, uses full next batch (current behavior)

    # Independent n for secondary
    secondary_n: 1               # Repetitions per unique prompt in secondary
                                 # Ignored when strategy="depth" (uses primary n)

    # Multi-batch lookahead (for breadth/hybrid)
    max_lookahead_batches: 4     # How many future batches to peek
    decay_factor: 0.5            # Allocation decay per batch distance
                                 # 0.5 → [128, 64, 32, 32] for budget=256

    # Optional: minimum samples per batch
    min_samples_per_batch: 8     # Don't include batch if allocation < this
```

### Strategy Behavior Matrix

| Strategy | secondary_n | Lookahead | Allocation |
|----------|-------------|-----------|------------|
| `depth` | = primary n | 1 batch | All budget to N+1 |
| `breadth` | = configured (default 1) | max_lookahead_batches | Decay distribution |
| `hybrid` | = configured (default 4) | max_lookahead_batches | Decay distribution |

---

## Allocation Algorithm

### Core Distribution Function

```python
def compute_secondary_allocation(
    strategy: str,
    budget: int,
    primary_n: int,
    secondary_n: int,
    max_lookahead: int,
    decay: float,
    available_batches: int,  # Actual future batches available
) -> dict[int, tuple[int, int]]:
    """
    Compute sample allocation for secondary batches.

    Returns:
        Dict mapping batch_offset -> (num_unique_prompts, repetitions)
        e.g., {1: (16, 16), 2: (8, 4), 3: (4, 4)}
    """
    if strategy == "depth":
        # All budget to immediate next batch with primary n
        num_unique = budget // primary_n
        return {1: (num_unique, primary_n)}

    # breadth or hybrid
    n = secondary_n if strategy in ("breadth", "hybrid") else 1
    total_unique = budget // n

    # Compute decay allocation across batches
    num_batches = min(max_lookahead, available_batches)
    allocations = _decay_allocate(total_unique, num_batches, decay)

    return {i + 1: (alloc, n) for i, alloc in enumerate(allocations) if alloc > 0}


def _decay_allocate(total: int, num_buckets: int, decay: float) -> list[int]:
    """
    Distribute total across buckets with exponential decay.

    Example: total=256, num_buckets=4, decay=0.5
    → [128, 64, 32, 32]
    """
    if num_buckets == 1:
        return [total]

    allocations = []
    remaining = total

    for i in range(num_buckets - 1):
        # First bucket gets (1 - decay) of remaining
        alloc = max(1, int(remaining * (1 - decay)))
        allocations.append(alloc)
        remaining -= alloc

        if remaining <= 0:
            break

    if remaining > 0:
        allocations.append(remaining)

    # Pad with zeros if we ran out early
    while len(allocations) < num_buckets:
        allocations.append(0)

    return allocations
```

### Example Allocations

```
Budget = 256, primary_n = 16

DEPTH (secondary_n = 16):
  Batch N+1: 16 prompts × 16 reps = 256 samples
  Total unique prompts: 16

BREADTH (secondary_n = 1, decay = 0.5):
  Batch N+1: 128 prompts × 1 rep = 128 samples
  Batch N+2: 64 prompts × 1 rep = 64 samples
  Batch N+3: 32 prompts × 1 rep = 32 samples
  Batch N+4: 32 prompts × 1 rep = 32 samples (remainder)
  Total unique prompts: 256

HYBRID (secondary_n = 4, decay = 0.5):
  Batch N+1: 32 prompts × 4 reps = 128 samples
  Batch N+2: 16 prompts × 4 reps = 64 samples
  Batch N+3: 8 prompts × 4 reps = 32 samples
  Batch N+4: 8 prompts × 4 reps = 32 samples (remainder)
  Total unique prompts: 64
```

---

## Implementation Changes

### 1. Lookahead Buffer

Current implementation peeks only at the next batch:

```python
batch_iter = iter(self.train_dataloader)
next_batch_dict = next(batch_iter, None)
```

New implementation maintains a buffer of future batches:

```python
class LookaheadBuffer:
    """Maintains a buffer of future batches for multi-batch sampling."""

    def __init__(self, dataloader, max_lookahead: int):
        self.dataloader = dataloader
        self.max_lookahead = max_lookahead
        self.buffer: deque[dict] = deque(maxlen=max_lookahead)
        self._iter = iter(dataloader)
        self._fill_buffer()

    def _fill_buffer(self):
        """Fill buffer up to max_lookahead."""
        while len(self.buffer) < self.max_lookahead:
            try:
                batch = next(self._iter)
                self.buffer.append(batch)
            except StopIteration:
                break

    def pop_primary(self) -> Optional[dict]:
        """Get next primary batch and refill buffer."""
        if not self.buffer:
            return None
        primary = self.buffer.popleft()
        self._fill_buffer()
        return primary

    def peek_secondary(self, count: int) -> list[dict]:
        """Peek at next `count` batches without consuming."""
        return list(self.buffer)[:count]
```

### 2. Secondary Batch Composition

```python
def _compose_secondary_batch(
    self,
    lookahead_buffer: LookaheadBuffer,
    allocation: dict[int, tuple[int, int]],  # batch_offset -> (num_prompts, n)
) -> Optional[DataProto]:
    """
    Compose secondary batch from multiple future batches.
    """
    future_batches = lookahead_buffer.peek_secondary(max(allocation.keys()))

    combined_prompts = []
    batch_origins = []  # Track which batch each sample came from

    for offset, (num_prompts, n) in sorted(allocation.items()):
        if offset > len(future_batches):
            break

        batch_dict = future_batches[offset - 1]
        batch = DataProto.from_single_dict(batch_dict)
        gen_batch = self._get_gen_batch(batch)

        # Sample num_prompts from this batch
        if num_prompts < len(gen_batch.batch):
            indices = np.random.choice(
                len(gen_batch.batch), num_prompts, replace=False
            )
            gen_batch = gen_batch.select(indices)

        # Repeat with secondary n (may differ from primary n)
        repeated = gen_batch.repeat(repeat_times=n, interleave=True)
        combined_prompts.append(repeated)
        batch_origins.extend([offset] * len(repeated.batch))

    if not combined_prompts:
        return None

    # Merge all prompts into single DataProto
    secondary = DataProto.concat(combined_prompts)
    secondary.non_tensor_batch["batch_origin"] = np.array(batch_origins)

    return secondary
```

### 3. Updated SecondaryOutput

```python
@dataclass
class SecondaryOutput:
    sample_id: str
    output: Optional[TokenOutput]
    status: Literal["completed", "aborted", "rejected", "pending"]
    tokens_generated: int
    prompt_ids: list[int]
    prompt_hash: int
    batch_origin: int = 1  # NEW: Which future batch (1 = N+1, 2 = N+2, etc.)
```

---

## Trade-off Analysis

### Depth (Current Default)

**Pros:**
- Maximum cache benefit for immediate next batch
- Matches DAPO/GRPO training dynamics (n responses per prompt)
- Simpler implementation (no multi-batch handling)

**Cons:**
- Zero pre-caching for batches N+2, N+3, ...
- May waste cache potential if prompt responses are already similar

**Best for:**
- Short training runs
- High response similarity within prompts
- When next-batch performance is critical

### Breadth (Maximum Exploration)

**Pros:**
- Pre-populates cache across many future batches
- **Cumulative benefit**: After warm-up, each batch receives cache updates from multiple preceding ticks
  ```
  With max_lookahead=4, decay=0.5:
  Batch K receives: tick K-4 (32) + tick K-3 (32) + tick K-2 (64) + tick K-1 (128)
  Steady-state: ~256 unique prompts cached per batch (full budget!)
  ```
- Better for diverse prompt distributions
- May catch "hard" prompts early

**Cons:**
- Cold-start period: First few batches have incomplete cache coverage
- n=1 may miss common response patterns (each prompt only gets one response variant cached)
- More complex batch composition logic

**Best for:**
- Long training runs (amortizes cold-start cost)
- Diverse/evolving prompt distributions
- Early exploration phases

### Hybrid (Balanced)

**Pros:**
- Balances depth and breadth benefits
- Still captures some repetition patterns
- Configurable trade-off point

**Cons:**
- May not excel at either extreme
- Requires tuning secondary_n parameter

**Best for:**
- General-purpose usage
- When optimal strategy is unknown
- A/B testing baseline

---

## Metrics and Observability

### New Metrics

| Metric | Description |
|--------|-------------|
| `runahead/strategy` | Current strategy (categorical: depth/breadth/hybrid) |
| `runahead/secondary_n` | Repetition factor used for secondary |
| `runahead/unique_prompts_total` | Total unique prompts in secondary batch |
| `runahead/batch_{N}_count` | Samples attempted from batch N+offset |
| `runahead/batch_{N}_completed` | Samples completed from batch N+offset |
| `runahead/batch_{N}_tokens` | Tokens added to cache from batch N+offset |
| `runahead/cache_coverage_{N}` | % of batch N+offset prompts with cache entries |

---

## Experimental Design

### Suggested Experiments

1. **Baseline Comparison**
   ```
   A: depth, secondary_n=16 (current)
   B: breadth, secondary_n=1, decay=0.5
   C: hybrid, secondary_n=4, decay=0.5
   ```
   Measure: Wall-clock time, spec decode acceptance rate, training loss

2. **Decay Factor Sensitivity**
   ```
   decay ∈ {0.3, 0.5, 0.7}
   ```
   Measure: Per-batch cache coverage, utilization efficiency

3. **Budget Sensitivity**
   ```
   budget ∈ {128, 256, 512}
   ```
   Measure: Trade-off between exploration and primary batch impact

4. **Epoch-dependent Strategy**
   - Early epochs: breadth (exploration)
   - Later epochs: depth (exploitation)

---

## Implementation Plan

### Phase 1: Core Infrastructure
1. Add configuration parameters to config schema
2. Implement `LookaheadBuffer` class
3. Implement `compute_secondary_allocation()` function

### Phase 2: Batch Composition
1. Implement `_compose_secondary_batch()` method
2. Update `SecondaryOutput` with `batch_origin` field
3. Modify secondary batch preparation in `_fit_runahead()`

### Phase 3: Cache Updates
1. Update `_update_suffix_trees_from_secondary()` for multi-batch tracking
2. Add `CacheCoverageTracker` for observability
3. Add new metrics to wandb logging

### Phase 4: Testing & Validation
1. Unit tests for allocation algorithm
2. Integration tests for batch composition
3. End-to-end validation on sample training run

---

## Open Questions

1. **Sampling without replacement**: When selecting prompts from a future batch, should we track which prompts have been sampled to avoid re-sampling across ticks?

   **Important consideration**: For breadth mode to achieve its cumulative benefit, we likely need **coordinated sampling** to ensure different prompts are sampled each tick. Options:
   - Random sampling (current): May re-sample same prompts, reducing effective coverage
   - Round-robin: Deterministically cycle through prompts across ticks
   - Coverage-aware: Track cached prompts, prioritize uncached ones

2. **Adaptive strategy switching**: Should we support automatic strategy switching based on observed cache hit rates?

3. **Budget overflow handling**: If available future batches have fewer prompts than budget allows, should we increase n or accept lower utilization?

4. **Partial batch handling**: When a batch is partially pre-cached from previous ticks, should we skip already-cached prompts or regenerate for fresh patterns?

---

## References

- Run-ahead architecture: `claude_docs/runahead_rollout_design.md`
- SRT development guide: `claude_docs/srt_development.md`
- Current implementation: `recipe/srt/ray_trainer.py:_fit_runahead()`
- Task context: `claude_docs/tasks/different_data_feeding_logics.md`
- DAPO/GRPO sample repetition: controlled by `actor_rollout_ref.rollout.n`
