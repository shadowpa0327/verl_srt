# Runahead Design History

This document archives the evolution of the runahead rollout design. For the **current design**, see `runahead_rollout_design.md`.

> **Note (2026-01-02):** The base routing class has been refactored from per-worker `AsyncLLMServerManager` to a single `CentralRouter` Ray actor shared by all workers. This provides global load visibility essential for run-ahead. Historical references below describe the original per-worker architecture.

---

## Version Timeline

| Version | Name | Key Contribution | Status |
|---------|------|------------------|--------|
| V0 | Original Design | Basic runahead concept, `abort_all_requests()` | Superseded |
| V0.1 | Design Review | Identified critical issues (targeted abort, local counters) | Superseded |
| V0.2 | Trigger Strategies | Documented completion-based vs capacity-based triggers | Superseded |
| V0.3 | Architecture Diagrams | Visual diagrams of components | Superseded |
| V1 | MVP Architecture | 4 invariants, 4 primitives, server-gated, StepBarrier | Superseded |
| V1.1 | Workload Aware | Server-side admission control, retry logic | Superseded |
| V2 | Unified Proposal | AgentLoop integration attempt (too broad) | Superseded |
| **V3** | **Rollout Design** | **Focused rollout-only, single-turn** | **CURRENT** |

---

## V0: Original Design

**File**: `verl/experimental/agent_loop/runahead.py`

**Key Ideas:**
- `AsyncLLMServerManagerWithRunahead` extends base server manager
- `RunaheadController` orchestrates batch1 → batch2 injection
- Trigger based on completion ratio (50%)

**Problems Identified:**
1. Used `abort_all_requests()` - kills ALL requests including other workers' primary
2. Per-worker local counters - doesn't reflect global server state
3. No server_request_id tracking - can't do targeted abort

```python
# V0 abort approach (WRONG for multi-worker)
async def abort_all_requests(self) -> dict[str, Any]:
    """Abort all requests across all managed servers."""
    # This kills EVERYTHING - unsafe in multi-worker!
```

---

## V0.1: Design Review

**Summary**: Analysis of external feedback, categorized as VALID/ACCEPTABLE/OVER-ENGINEERED.

### Critical Issues Identified

**MUST FIX:**
1. **Request-ID based abort** - Track `server_request_id`, use `abort_requests([ids])`
2. **Expose server_request_id** - Return/store the ID sent to vLLM

**ACCEPTABLE FOR V1:**
3. Local capacity tracking - Works with conservative thresholds (0.5)
4. Sequence-based capacity - Reasonable proxy
5. Burst injection - vLLM handles queuing

**SKIP (Over-engineered):**
6. Replica-level coordinator - Too complex
7. Lease-based admission - Solves wrong problem
8. Slot-filling injection - Adds latency without benefit

### Key Insight
> ChatGPT's suggestions (coordinator, leases, slot-filling) are excellent for **online serving** but VERL's runahead is **batch training** with fixed sizes and controlled submission.

---

## V0.2: Trigger Strategies

**Summary**: Documented three trigger approaches.

### Option 1: Completion-Based (V1 Implementation)
```python
# Trigger when X% of batch1 completes
if batch1_tracker.completion_ratio >= 0.5:
    trigger_runahead()
```
- Simple, predictable
- Ignores actual server capacity

### Option 2: Server Workload Tracking
```python
# Trigger when servers have spare capacity
if server_manager.capacity_ratio < 0.5:
    trigger_runahead()
```
- Measures actual capacity
- May trigger earlier when short requests finish

### Option 3: Hybrid Approach
```python
# Combine multiple signals
if (completion_ratio >= 0.3 and
    capacity_ratio < 0.6 and
    in_flight >= 2):
    trigger_runahead()
```
- Most robust
- More complex

**Recommendation**: Hybrid for production, but completion-based acceptable for V1.

---

## V0.3: Architecture Diagrams

**Summary**: Visual documentation of the runahead architecture.

### GPU Bubble Problem
```
Without Runahead:
─────────────────────────────────────────────────────────────
│ Batch1 Short │████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ Batch1 Long  │████████████████████████████████████████████│
│ Batch2       │                                    │███████│ <- Waits
─────────────────────────────────────────────────────────────
                                           GPU Bubble ↑

With Runahead:
─────────────────────────────────────────────────────────────
│ Batch1 Short │████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ Batch1 Long  │████████████████████████████████████████████│
│ Batch2       │            │██████████████████│ <- Fills bubble
─────────────────────────────────────────────────────────────
```

### State Machine
```
          ┌──────────────┐     capacity check
START ───►│   MONITOR    │─────────────────────┐
          │  CAPACITY    │                     │
          └──────┬───────┘                     │
                 │ batch1 empty?               │
          ┌──────▼──────┐                      │
          │   BATCH1    │                      │
          │  COMPLETE   │                      │
          └──────┬──────┘                      │
                 │ batch2 started?             │
          ┌──────▼──────┐     ┌──────────┐    │
          │    ABORT    │────►│ COLLECT  │────►│ DONE
          │   BATCH2    │     │  BATCH2  │
          └─────────────┘     └──────────┘
```

---

## V1: MVP Architecture (Server-Gated Slot Filling)

**Summary**: Production-ready architecture with 4 invariants and 4 primitives.

### Four Non-Negotiable Invariants

1. **Primary never harmed** - Secondary cannot slow down or affect primary
2. **Targeted & preemptible** - Never `abort_all_requests()` in shared environment
3. **Global capacity** - Server-side counters, not per-worker
4. **Bounded** - Caps on concurrency, tokens, memory

### Four Minimal Primitives

**A. Request Tagging**
```python
@dataclass
class RequestMeta:
    kind: str  # "primary" | "secondary"
    step_id: str
    worker_id: int
```

**B. Targeted Abort**
```python
async def abort_requests(self, request_ids: list[str]) -> dict:
    """Abort specific requests by ID."""
```

**C. Global Server Workload Signal**
```python
# Inside vLLMHttpServer actor
self.in_flight_total: int = 0
self.in_flight_secondary: int = 0
```

**D. Global Stop Condition (StepBarrier)**
```python
@ray.remote
class StepBarrier:
    def start(self, step_id: str, num_workers: int): ...
    def mark_primary_done(self, step_id: str, worker_id: int): ...
    def is_done(self, step_id: str) -> bool: ...
```

### Server-Gated vs Threshold-Based

| Threshold-Based | Server-Gated |
|-----------------|--------------|
| Client decides "now is good time" | Server decides "I have room" |
| Race conditions between workers | No races (server has truth) |
| Brittle trigger point | Naturally adapts |
| Per-worker capacity (wrong) | Global capacity (correct) |

### 4-Phase Execution
1. **Phase 1**: Primary + opportunistic secondary filling
2. **Phase 2**: Notify barrier (local primary done)
3. **Phase 3**: Keep filling until global done
4. **Phase 4**: Abort remaining secondary

---

## V1.1: Workload-Aware Design

**Summary**: Server-side admission control with retry logic.

### Problem: Per-Process Local Counter
```python
class GatedServerManager:
    def __init__(self, ...):
        self.secondary_in_flight = 0  # LOCAL - invisible to other workers!
```

Worker A can't see requests from Worker B → over-admission or under-utilization.

### Solution: Server-Side Admission Control

```python
# In vLLMHttpServer (shared across all workers)
def should_admit_runahead(self) -> bool:
    # Check 1: Runahead capacity limit
    runahead_cap = int(max_num_seqs * self.runahead_config.runahead_frac)
    if self.in_flight_runahead >= runahead_cap:
        return False

    # Check 2: Primary headroom
    available_slots = max_num_seqs - self.in_flight_total
    if available_slots <= self.runahead_config.primary_headroom:
        return False

    return True

async def generate(self, ..., meta=None):
    kind = (meta or {}).get("kind", "primary")

    if kind == "runahead" and not self.should_admit_runahead():
        return TokenOutput(token_ids=[], stop_reason="runahead_rejected")

    # ... proceed with generation
```

### Retry Logic
```python
class RunaheadWorkQueue:
    async def complete(self, request_id: str, success: bool):
        item = self.in_flight.pop(request_id, None)
        if success:
            self.completed.append(item)
        else:
            if item.retry_count < item.max_retries:
                item.retry_count += 1
                self.pending.appendleft(item)  # Priority retry
```

---

## V2: Unified Proposal (Superseded)

**Summary**: Attempted to unify all designs with trainer integration.

**Scope was too broad:**
- Included BatchScheduler for future batch management
- Included result caching
- Included trainer integration
- Included multi-turn agent support

**Why superseded**: User clarified scope:
1. Ignore trainer - focus on rollout only
2. User handles runahead results
3. Single-turn only
4. vLLM priority scheduling deferred

---

## V3: Rollout Design (CURRENT)

**See**: `runahead_rollout_design.md`

**Focused scope:**
- Single-turn rollout with runahead speculation
- Server-side admission control
- Multi-worker coordination via StepBarrier
- Targeted abort
- Returns `(primary_results, secondary_results)` to caller

**API:**
```python
result = await manager.generate_sequences_with_runahead(
    primary_prompts,    # Current batch
    spec_prompts,       # Future batch
    config,
)

# Returns:
result.primary_outputs   # All completed
result.secondary_outputs # List[SecondaryOutput] - caller handles caching
result.metrics           # Observability
```

---

## V3.1: API Pattern Updates (Superseded Patterns)

These patterns from V3 have been superseded based on prototype implementation experience.
See `test_vllm_runahead_server_side_admission_prototype.py` for reference implementation.

| Pattern | Original (V3) | Current (Prototype) | Why Changed |
|---------|---------------|---------------------|-------------|
| Request kind | `meta={"kind": "secondary"}` param | `sampling_params["_verl_request_kind"]` | Avoids modifying vLLM server signature; wrapper pops key before forwarding |
| Abort | Batch `abort_requests([ids])` on server | Per-id `abort_request(id)` with manager-side grouping | vLLM exposes per-request abort, not batch |
| Admission location | Inside vLLMHttpServer | Ray actor wrapper (`AdmissionControlledServer`) | Avoids Verl library modifications; headroom-based inside vLLM is end-state |

> **Critical note:** The wrapper pattern only fixes multi-worker races if all workers share
> the **same named/detached admission gate actor per server**. If each worker instantiates
> its own wrapper, you're back to per-worker local counters and races.

---

## Key Lessons Learned

1. **Start simple**: Completion-based trigger is fine for V1
2. **Server is truth**: Global capacity must live in server, not workers
3. **Targeted abort is critical**: Never `abort_all_requests()` in multi-worker
4. **4-phase execution works**: Primary → barrier → fill → abort
5. **Scope matters**: Focused rollout-only design is cleaner than unified proposal

---

## References

### Test Files (Prototypes)
- `tests/workers/rollout/rollout_vllm/test_vllm_runahead_agentloop_standalone.py` - AgentLoop integration prototype
- `tests/workers/rollout/rollout_vllm/test_vllm_runahead_server_side_admission_prototype.py` - **V3.1 server-side admission (current)**
- `tests/workers/rollout/rollout_vllm/test_vllm_runahead_slack_filling.py` - Continuous slack-filling pattern

### Implementation
- `verl/experimental/agent_loop/runahead.py` - V0 implementation (needs update)
