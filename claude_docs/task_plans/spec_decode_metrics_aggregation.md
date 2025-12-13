# Spec Decode Metrics Aggregation

**Date**: 2025-12-13
**Status**: Planning Complete, Ready for Implementation
**Priority**: High (enables accurate measurement for other optimizations)

## Problem Statement

In multi-GPU scenarios, each vLLM rollout worker computes spec decode metrics locally. Currently:

1. Each worker returns its own `spec_decode_metrics` in `DataProto.meta_info`
2. After dispatch collection, only one rank's metrics survive (others are dropped/overwritten)
3. Logged metrics are incomplete/inaccurate for the full batch

### Example of Incorrect Aggregation

```
Worker 0: 100 drafts, 80 accepted → rate = 0.80
Worker 1: 10 drafts, 9 accepted  → rate = 0.90

Current (only rank 0): rate = 0.80  ❌ Incomplete
Naive AVG:            rate = 0.85  ❌ Statistically incorrect
Correct (sum counts): rate = 89/110 = 0.809  ✓
```

## Solution Design

Follow the existing `reduce_timing()` pattern in verl:

1. Workers compute local metrics (unchanged)
2. After rollout returns, aggregate metrics using `all_reduce` with SUM
3. Recompute rates from aggregated counts

### Key Insight

- **Counts** (`num_drafts`, `num_draft_tokens`, `num_accepted_tokens`) → SUM across ranks
- **Rates** (`acceptance_rate`, `mean_acceptance_length`) → Recompute from summed counts (NOT averaged)

## Implementation Plan

### Step 1: Add `reduce_spec_decode_metrics()` to `performance.py`

**File**: `verl/utils/profiler/performance.py`

```python
def reduce_spec_decode_metrics(
    metrics: dict[str, float],
) -> dict[str, float]:
    """
    Reduce spec decode metrics across all processes.

    Counts are summed, rates are recomputed from aggregated counts.
    Unlike timing (which uses AVG), spec decode counts should be summed
    and rates recomputed for statistical correctness.
    """
    if not dist.is_initialized():
        return metrics

    if not metrics:
        return metrics

    # Extract counts to sum
    count_keys = ["num_drafts", "num_draft_tokens", "num_accepted_tokens"]
    counts = torch.tensor(
        [metrics.get(f"spec_decode/{k}", 0) for k in count_keys],
        dtype=torch.float64,
        device=get_device_id(),
    )

    # Sum counts across all ranks
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    total_drafts = int(counts[0].item())
    total_draft_tokens = int(counts[1].item())
    total_accepted = int(counts[2].item())

    # Recompute rates from aggregated counts
    acceptance_rate = total_accepted / total_draft_tokens if total_draft_tokens > 0 else 0.0
    mean_acceptance_length = 1.0 + (total_accepted / total_drafts) if total_drafts > 0 else 1.0

    return {
        "spec_decode/num_drafts": total_drafts,
        "spec_decode/num_draft_tokens": total_draft_tokens,
        "spec_decode/num_accepted_tokens": total_accepted,
        "spec_decode/acceptance_rate": acceptance_rate,
        "spec_decode/mean_acceptance_length": mean_acceptance_length,
    }
```

### Step 2: Call reduction in `fsdp_workers.py`

**File**: `verl/workers/fsdp_workers.py`
**Location**: In `generate_sequences()`, after line 949 (after timing reduction)

```python
# Reduce spec decode metrics across ranks (similar to timing)
if "spec_decode_metrics" in output.meta_info:
    from verl.utils.profiler.performance import reduce_spec_decode_metrics
    output.meta_info["spec_decode_metrics"] = reduce_spec_decode_metrics(
        output.meta_info["spec_decode_metrics"]
    )
```

### Step 3: Simplify metrics in `vllm_rollout_spmd.py`

**File**: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
**Location**: Lines 525-534

Remove per-position rates for simplicity:

```python
# Before (lines 525-534):
spec_decode_metrics = {
    "spec_decode/num_drafts": rollout_stats.num_drafts,
    "spec_decode/num_draft_tokens": rollout_stats.num_draft_tokens,
    "spec_decode/num_accepted_tokens": rollout_stats.num_accepted_tokens,
    "spec_decode/acceptance_rate": rollout_stats.acceptance_rate,
    "spec_decode/mean_acceptance_length": rollout_stats.mean_acceptance_length,
}
# Add per-position acceptance rates  ← REMOVE THIS LOOP
for i, rate in enumerate(rollout_stats.per_position_rates):
    spec_decode_metrics[f"spec_decode/acceptance_rate_pos_{i}"] = rate

# After:
spec_decode_metrics = {
    "spec_decode/num_drafts": rollout_stats.num_drafts,
    "spec_decode/num_draft_tokens": rollout_stats.num_draft_tokens,
    "spec_decode/num_accepted_tokens": rollout_stats.num_accepted_tokens,
    "spec_decode/acceptance_rate": rollout_stats.acceptance_rate,
    "spec_decode/mean_acceptance_length": rollout_stats.mean_acceptance_length,
}
```

## Files Changed Summary

| File | Change |
|------|--------|
| `verl/utils/profiler/performance.py` | Add `reduce_spec_decode_metrics()` function |
| `verl/workers/fsdp_workers.py` | Call reduction after rollout returns (~line 949) |
| `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Remove per-position rates (lines 532-534) |

## Testing

1. Run with 2+ GPUs and verify metrics are aggregated
2. Check wandb logs show combined counts (not just rank 0)
3. Verify acceptance_rate is correctly recomputed from sums

## Known Gaps (Future Work)

### Gap 1: Megatron Workers Not Covered

**File**: `verl/workers/megatron_workers.py`
**Issue**: Has similar `generate_sequences()` pattern (line 703) but this implementation only covers FSDP workers.
**Action**: If Megatron backend is used with spec decode in the future, add the same reduction call there.

### Gap 2: vLLMAsyncRollout Not Covered

**File**: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
**Issue**: `vLLMAsyncRollout` class does not have spec decode metrics tracking (only `vLLMRollout` sync mode does).
**Action**: If async rollout mode is used with spec decode, add `SpecDecodeMetricsTracker` initialization and usage to `vLLMAsyncRollout`.

### Gap 3: Per-Position Rates Dropped

**Issue**: Per-position acceptance rates (e.g., `acceptance_rate_pos_0`, `acceptance_rate_pos_1`) are removed for simplicity.
**Rationale**: Aggregating per-position rates requires tracking per-position counts (not just rates), which would require changes to `SpecDecodeSnapshot`. The overall acceptance rate is sufficient for most monitoring needs.
**Action**: If per-position analysis is needed, track `per_pos_accepted_counts` in the snapshot and sum them across ranks before computing rates.

## Related Documents

- [Suffix Tree Memory Optimizations](../future/suffix_tree_memory_optimizations.md) - Memory concerns
- [Suffix Tree VERL Integration](suffix_tree_verl_integration.md) - Original integration
