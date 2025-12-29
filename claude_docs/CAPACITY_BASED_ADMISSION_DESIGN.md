# Capacity-Based Admission Design for Server-Side Runahead Control

## Problem Statement

### Current Fixed-Limit Approach
The current Server-Side Admission uses a fixed `max_runahead_inflight` limit per server:

```python
# Current: Fixed limit regardless of server state
max_runahead_inflight = 2  # Always 2, even if server is idle
```

### Under-Subscription Issue
With multiple DP workers, this causes **under-subscription**:

```
Scenario: 4 DP workers, 2 servers, budget=2 per server

Server capacity: Can handle ~32 concurrent requests
Current runahead slots: 2 per server × 2 servers = 4 total

When all workers finish primary and want runahead:
- Worker 1: Gets 1 slot ✓
- Worker 2: Gets 1 slot ✓
- Worker 3: Gets 1 slot ✓
- Worker 4: Gets 1 slot ✓
- Workers want more: BLOCKED (only 4 slots total)

Result: Server has capacity for 32, but only 4 runahead running
→ Wasted slack capacity!
```

### The Trade-off
| Approach | Under-subscription | Over-subscription |
|----------|-------------------|-------------------|
| Slack Detection (racing) | Low | High risk |
| Server-Side (fixed limit) | High risk | Low |
| **Capacity-Based (proposed)** | **Low** | **Low** |

## Proposed Solution: Capacity-Based Admission

Instead of a fixed limit, dynamically calculate available capacity based on server state.

### Core Formula

```python
def get_available_runahead_capacity(workload, config):
    current_load = workload["num_requests_running"] + workload["num_requests_waiting"]

    available = config.max_server_capacity - current_load - config.reserved_for_primary

    return max(0, available)
```

### Configuration Parameters

```python
@dataclass
class AdmissionGateConfig:
    # Existing parameters
    max_runahead_inflight: int = 1      # Used when capacity-based is disabled
    enforce_slack: bool = True
    load_threshold: int = 32
    kv_cache_threshold: float = 0.85

    # NEW: Capacity-based admission
    use_capacity_based_admission: bool = False  # Enable dynamic capacity
    max_server_capacity: int = 64               # Max concurrent requests server can handle
    reserved_for_primary: int = 8               # Reserve slots for incoming primary
```

### Modified Admission Logic

```python
# In AdmissionControlledServer.generate():

async def _get_runahead_limit(self) -> int:
    """Calculate dynamic runahead limit based on server capacity."""
    if not self._cfg.use_capacity_based_admission:
        return self._cfg.max_runahead_inflight  # Use fixed limit

    workload = await self._get_cached_workload()
    if workload.get("error"):
        return 0  # Conservative: no runahead if can't get metrics

    current_load = (
        int(workload.get("num_requests_running", 0)) +
        int(workload.get("num_requests_waiting", 0))
    )

    available = (
        self._cfg.max_server_capacity -
        current_load -
        self._cfg.reserved_for_primary
    )

    return max(0, available)

async def generate(self, *, request_id, prompt_ids, sampling_params, image_data=None):
    params = dict(sampling_params)
    kind = params.pop("_verl_request_kind", "primary")

    acquired_slot = False
    if kind == "runahead":
        # Get dynamic limit instead of fixed
        current_limit = await self._get_runahead_limit()

        async with self._lock:
            if self._runahead_inflight >= current_limit:
                self._runahead_rejected_total += 1
                return TokenOutput(token_ids=[], stop_reason="rejected")

            self._runahead_inflight += 1
            acquired_slot = True

    # ... rest of generation logic
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  AdmissionControlledServer                       │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Capacity Calculator (runs on each admission decision)     │ │
│  │                                                             │ │
│  │  Inputs:                                                    │ │
│  │    - workload.num_requests_running                         │ │
│  │    - workload.num_requests_waiting                         │ │
│  │    - config.max_server_capacity (e.g., 64)                 │ │
│  │    - config.reserved_for_primary (e.g., 8)                 │ │
│  │                                                             │ │
│  │  Formula:                                                   │ │
│  │    available = max_capacity - current_load - reserved      │ │
│  │                                                             │ │
│  │  Example:                                                   │ │
│  │    64 - 10 - 8 = 46 runahead slots available               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Admission Decision:                                             │
│    if runahead_inflight < available_capacity:                   │
│        ADMIT → increment inflight counter                        │
│    else:                                                         │
│        REJECT → return stop_reason="rejected"                   │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Worker Behavior

### Before (Fixed Limit)
```
4 Workers, Server capacity=64, Fixed limit=2

Worker 1: runahead → admitted (1/2)
Worker 2: runahead → admitted (2/2)
Worker 3: runahead → REJECTED
Worker 4: runahead → REJECTED

Total runahead: 2 (under-utilizing 62 slots!)
```

### After (Capacity-Based)
```
4 Workers, Server capacity=64, Reserved=8, Current load=0

Available = 64 - 0 - 8 = 56 slots for runahead

Worker 1: 8 runahead → all admitted
Worker 2: 8 runahead → all admitted
Worker 3: 8 runahead → all admitted
Worker 4: 8 runahead → all admitted

Total runahead: 32 (much better utilization!)
```

## Key Benefits

1. **Adapts to load**: When server is busy, admits less runahead
2. **Maximizes slack utilization**: When server is idle, admits more runahead
3. **Safety buffer**: `reserved_for_primary` ensures incoming primary requests have capacity
4. **Global enforcement**: Still uses server-side lock, preventing racing

## Configuration Recommendations

| Workload Type | max_server_capacity | reserved_for_primary |
|---------------|--------------------|--------------------|
| Small models (0.5B) | 128 | 16 |
| Medium models (7B) | 64 | 8 |
| Large models (70B) | 32 | 4 |

## Files to Modify

| File | Changes |
|------|---------|
| `test_vllm_run_ahead_server_side_admission.py` | Add capacity config, modify admission logic |
| `benchmark_admission_comparison.py` | Add `--capacity-based` flag for testing |

## Testing Plan

1. **Single-worker**: Verify capacity calculation is correct
2. **Multi-worker**: Verify better utilization vs fixed limit
3. **Under load**: Verify primary requests get reserved capacity
4. **KV cache pressure**: Verify rejection when cache is full

## Open Questions

1. **How to determine `max_server_capacity`?**
   - Could query vLLM for max_num_seqs config
   - Could use heuristics based on model size

2. **Should `reserved_for_primary` be dynamic?**
   - Could scale based on expected incoming primary batch size
   - Could use historical arrival rate

3. **Cache TTL for workload stats?**
   - Currently 200ms - may need tuning for responsiveness vs overhead

## Related Files

- Current implementation: `tests/workers/rollout/rollout_vllm/test_vllm_run_ahead_server_side_admission.py`
- Benchmark: `tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py`
- vLLM metrics: `verl/workers/rollout/vllm_rollout/vllm_async_server.py:632-672`
