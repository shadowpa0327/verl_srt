# Runahead Rollout: Focused Design (Rollout-Only)

## Scope

**In scope:**
- Single-turn rollout with runahead speculation
- Router-side admission control (prototype)
- Manager-level coordination via `ray.wait()`
- Targeted abort (never abort_all)
- Return `RunaheadResult(primary_outputs, secondary_outputs)` to caller

**Out of scope (handled by caller):**
- Result caching and reuse
- Trainer integration
- Multi-turn agent speculation

**Implemented but optional:**
- vLLM priority scheduling (requires `scheduler_policy: "priority"` in vLLM config)

---

## Execution Timeline

### The Problem: GPU Bubbles

In a typical RL training batch, requests have varying output lengths. Short requests complete early, leaving GPU capacity idle ("bubbles") while waiting for long requests:

```
Time ──────────────────────────────────────────────────────────────────────────>

                    WITHOUT RUNAHEAD
                    ════════════════

Primary Batch (8 requests):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  P0 ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (16 tokens)  │
│  P1 ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (16 tokens)  │
│  P2 ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (16 tokens)  │
│  P3 ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (16 tokens)  │
│  P4 ████████████████████████████████████████████████████████  (200 tokens) │
│  P5 ██████████████████████████████████████████████████████░░  (200 tokens) │
│  P6 ████████████████████████████████████████████████████░░░░  (200 tokens) │
│  P7 ██████████████████████████████████████████████████░░░░░░  (200 tokens) │
│                                                                             │
│     ▲              ▲                                      ▲                 │
│     │              │                                      │                 │
│  t=0.0s         t=0.5s                                 t=3.0s               │
│  Start       Short requests                         All complete            │
│              complete (P0-P3)                                               │
│                                                                             │
│              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                │
│              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^               │
│                    WASTED GPU CAPACITY ("Bubble")                           │
│                    ~2.5 seconds of idle time                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

GPU Utilization:
100% ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 50% ────────────────████████████████████████████████████████████
  0% ─────────────────────────────────────────────────────────────
     t=0           t=0.5s                                    t=3.0s
```

### The Solution: Runahead Fills Bubbles

Runahead detects when capacity becomes available and speculatively starts future batch (secondary) requests.

**Important: Startup Race Prevention**

A critical timing issue exists: when primaries are dispatched via Ray async actors, there's a
~1.5s window before they arrive at `router.generate()` and increment `server_load`. During this
window, the router sees `server_load=0` and would admit secondaries prematurely.

**Primary Reservation** solves this by blocking ALL secondary admission until all primaries have
arrived at the router. The manager calls `reserve_primary_load(N)` before dispatching primaries,
and each primary arriving at `generate()` releases one reservation. Secondaries are only admitted
when `_primary_reserved_total = 0`.

```
Time ──────────────────────────────────────────────────────────────────────────>

                    WITH RUNAHEAD (4-Phase Execution)
                    ═════════════════════════════════

Primary Batch (8 requests):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  P0 ████████                                                   (completed)  │
│  P1 ████████                                                   (completed)  │
│  P2 ████████████                                               (completed)  │
│  P3 ████████████                                               (completed)  │
│  P4 ████████████████████████████████████████████████████████   (completed)  │
│  P5 ██████████████████████████████████████████████████████     (completed)  │
│  P6 ████████████████████████████████████████████████████       (completed)  │
│  P7 ██████████████████████████████████████████████████         (completed)  │
│                                                                             │
│     ▲              ▲                                      ▲                 │
│     │              │                                      │                 │
│  t=0.0s         t=0.5s                                 t=3.0s               │
│  Start       Short done                              All complete           │
│              (P0-P3)                                                        │
│                                                                             │
│  Secondary Batch (4 requests from NEXT iteration):                          │
│                                                                             │
│  S0         ░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (completed!) │
│  S1         ░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (completed!) │
│  S2         ░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████  (ABORTED)  │
│  S3         ░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████████  (ABORTED)  │
│                   ▲                                   ▲                     │
│                   │                                   │                     │
│               PHASE 1                             PHASE 4                   │
│            (opportunistic                      (abort remaining)            │
│             secondary)                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  ████  Primary request execution
  ▓▓▓▓  Secondary (runahead) request execution
  ░░░░  Waiting / not started

GPU Utilization:
100% ████████████████████████████████████████████████████████████
 50% ────────────────────────────────────────────────────────────
  0% ────────────────────────────────────────────────────────────
     t=0           t=0.5s                                    t=3.0s
                   ▲
                   Secondary fills the bubble!
```

### 4-Phase Execution Timeline (Single Worker)

```
Time ──────────────────────────────────────────────────────────────────────────>
     │
     │  ╔═══════════════════════════════════════════════════════════════════════╗
     │  ║ PHASE 1: Primary execution + opportunistic secondary                 ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │
t=0.00s  ┌─ PRIMARY BATCH STARTS ─────────────────────────────────────────────
     │   │  • Launch P0-P7 (8 primary requests)
     │   │  • Check server capacity: can we start secondary?
     │   │  • Server: in_flight=8, headroom=8 → NO secondary yet
     │   │
t=0.30s  ├─ SHORT REQUESTS COMPLETING ────────────────────────────────────────
     │   │  • P0, P1 complete (16 tokens each)
     │   │  • Server: in_flight=6, headroom=8 → secondary_frac allows 2
     │   │  • Pop S0, S1 from the manager secondary queue
     │   │  • Launch S0, S1 (opportunistic)
     │   │
t=0.50s  ├─ MORE PRIMARY COMPLETING ──────────────────────────────────────────
     │   │  • P2, P3 complete
     │   │  • Server: in_flight=6 (4 primary + 2 secondary)
     │   │  • Pull S2, S3 from queue, launch them
     │   │
t=1.50s  ├─ LOCAL PRIMARY DONE ───────────────────────────────────────────────
     │   │  • P4-P7 all complete on THIS worker
     │   │  • S0, S1 complete (short outputs)
     │   │  • S2, S3 still running (long outputs)
     │   │
     │  ╔═══════════════════════════════════════════════════════════════════════╗
     │  ║ PHASE 2: Continue busy loop                                           ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │   │
t=1.51s  ├─ SOME PRIMARY DONE ────────────────────────────────────────────────
     │   │  • A primary chunk completes
     │   │  • Other primary chunks may still be running
     │   │
     │  ╔═══════════════════════════════════════════════════════════════════════╗
     │  ║ PHASE 3: Keep filling until all primary done                         ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │   │
t=1.52s  ├─ CONTINUE SECONDARY ───────────────────────────────────────────────
     │   │  • while primary still running:
     │   │  •     _try_launch_secondary()  # Pull more from queue
     │   │  •     _collect_secondary()     # Gather completed
     │   │  • S2 at 150/300 tokens, S3 at 120/300 tokens
     │   │
t=2.80s  ├─ GLOBAL PRIMARY DONE ──────────────────────────────────────────────
     │   │  • primary_refs becomes empty
     │   │  • All workers have finished their primary batches
     │   │
     │  ╔═══════════════════════════════════════════════════════════════════════╗
     │  ║ PHASE 4: Abort remaining secondary                                   ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │   │
t=2.81s  ├─ TARGETED ABORT ───────────────────────────────────────────────────
     │   │  • running_ids = [S2.server_request_id, S3.server_request_id]
     │   │  • await router.abort_requests(running_ids)
     │   │  • S2: aborted at 250 tokens (partial)
     │   │  • S3: aborted at 220 tokens (partial)
     │   │
t=2.82s  └─ RESULTS ──────────────────────────────────────────────────────────
         │
         │  Primary Results (8/8 completed):
         │    P0-P7: All completed successfully
         │
         │  Secondary Results (2 completed, 2 aborted):
         │    S0: completed (32 tokens) → REUSABLE in next iteration!
         │    S1: completed (32 tokens) → REUSABLE in next iteration!
         │    S2: aborted at 250 tokens → partial, caller decides
         │    S3: aborted at 220 tokens → partial, caller decides
         │
         │  Bubble Utilization: ~85% (vs 0% without runahead)
```

### Multi-Worker Timeline (Manager Busy Loop)

```
Time ──────────────────────────────────────────────────────────────────────────>

Worker 0:                                    Worker 1:
┌────────────────────────────────────┐       ┌────────────────────────────────────┐
│                                    │       │                                    │
│ PHASE 1: Primary + Opp. Secondary  │       │ PHASE 1: Primary + Opp. Secondary  │
│ ───────────────────────────────────│       │ ───────────────────────────────────│
│ P0 ████████                        │       │ P4 ████████████████████████████████│
│ P1 ████████                        │       │ P5 ██████████████████████████████  │
│ P2 ████████████                    │       │ P6 ████████████████████████████    │
│ P3 ████████████                    │       │ P7 ██████████████████████████      │
│                                    │       │                                    │
│ S0     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓           │       │ (no capacity for secondary yet)    │
│ S1     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓           │       │                                    │
│        ▲                           │       │                                    │
│        │                           │       │                                    │
│    Pulled from                     │       │                                    │
│    manager queue                   │       │                                    │
│                                    │       │                                    │
│ t=1.0s: LOCAL PRIMARY DONE         │       │                                    │
│         ▼                          │       │                                    │
│ PHASE 2: primary chunk done        │       │                                    │
│         │                          │       │                                    │
│         ▼                          │       │                                    │
│ PHASE 3: Keep filling              │       │                                    │
│ ───────────────────────────────────│       │                                    │
│ S2     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │       │                                    │
│ S3     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │       │                                    │
│                                    │       │                                    │
│ (Worker 0 keeps working on sec.)   │       │ t=2.8s: LOCAL PRIMARY DONE         │
│                                    │       │         ▼                          │
│                                    │       │ PHASE 2: primary chunk done        │
│                                    │       │                                    │
│ t=2.8s: all primary done           │       │ t=2.8s: all primary done           │
│         ▼                          │       │         ▼                          │
│ PHASE 4: Abort S2, S3              │       │ PHASE 4: (no secondary running)    │
│                                    │       │                                    │
└────────────────────────────────────┘       └────────────────────────────────────┘
         │                                            │
         │              ┌─────────────────────────────┘
         │              │
         ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Manager Busy Loop                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  primary_refs: empty  ← All primaries done                             │  │
│  │  secondary_refs: {S2, S3} (may still be running)                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Timeline:
─────────────────────────────────────────────────────────────────────────────────
t=0.0s   │ Both workers start primary batch
t=0.3s   │ Worker 0: P0,P1 done → starts S0,S1 (manager queue)
t=0.5s   │ Worker 0: P2,P3 done → starts S2,S3
t=1.0s   │ Worker 0: All primary done (manager observes primary ref done)
         │ Manager keeps drip-feeding secondary while any primary remains
t=1.5s   │ Worker 0: S0,S1 complete
t=2.8s   │ Worker 1: All primary done (manager observes primary ref done)
         │ Manager sees all primary done → PHASE 4
t=2.81s  │ Worker 0: Aborts S2,S3 by server_request_id
─────────────────────────────────────────────────────────────────────────────────
```

### Server-Side Admission Control

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    vLLM Server Admission Control                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  max_num_seqs = 32                                                          │
│  secondary_frac = 0.20  →  max_secondary = 6                                │
│  primary_headroom = 8                                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Capacity: 32 slots                           │    │
│  │                                                                     │    │
│  │  ████████████████████████  │  ░░░░░░  │  ▓▓▓▓▓▓▓▓                  │    │
│  │  PRIMARY (18 running)      │ HEADROOM │  SECONDARY (6 max)         │    │
│  │                            │   (8)    │                            │    │
│  │                            │          │                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  should_admit_secondary() logic:                                            │
│  ─────────────────────────────────                                          │
│  1. Check secondary_cap: in_flight_secondary < max_secondary (6)?           │
│  2. Check headroom: (max_num_seqs - in_flight_total) > primary_headroom?   │
│                                                                             │
│  Example states:                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ State        │ in_flight │ secondary │ available │ admit? │ reason   │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │ Full primary │    24     │     0     │     8     │   NO   │ headroom │  │
│  │ Draining     │    16     │     4     │    16     │  YES   │          │  │
│  │ Sec. cap hit │    12     │     6     │    20     │   NO   │ sec_frac │  │
│  │ Available    │    10     │     2     │    22     │  YES   │          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Design

### AgentLoopManager API

```python
class AgentLoopManager:
    async def generate_sequences_with_runahead(
        self,
        primary_prompts: list[PromptData],      # Current batch
        secondary_prompts: list[PromptData],   # Future batch (for runahead)
        runahead_config: RunaheadConfig,
    ) -> RunaheadResult:
        """
        Run primary batch with runahead on future batch.

        Args:
            primary_prompts: Current batch prompts (must complete)
            secondary_prompts: Future batch prompts (opportunistic, may be aborted)
            runahead_config: Runahead configuration

        Returns:
            RunaheadResult containing both primary and secondary outputs
        """
        ...

@dataclass
class RunaheadResult:
    """Result of runahead rollout."""
    primary_outputs: list[TokenOutput]          # All completed
    secondary_outputs: list[SecondaryOutput]    # May be partial/aborted
    metrics: RunaheadMetrics

@dataclass
class SecondaryOutput:
    """Single secondary (runahead) output."""
    sample_id: str
    output: Optional[TokenOutput]               # None if not started
    status: Literal["completed", "aborted", "rejected", "pending"]
    tokens_generated: int                       # Partial progress if aborted

@dataclass
class RunaheadMetrics:
    """Observability metrics."""
    primary_time: float
    secondary_started: int
    secondary_completed: int
    secondary_aborted: int
    secondary_rejected: int
    bubble_utilization: float                   # % of bubble time used
```

### Configuration

```python
@dataclass
class RunaheadConfig:
    """Runahead configuration."""

    enabled: bool = False                     # Whether run-ahead is enabled

    # Admission control (per-server gating naturally limits total secondaries)
    load_threshold: int = 32                  # Admit secondary when server_load < threshold

    # Router queue settings
    max_queue_size: int = 256                 # Max pending secondary items in queue
    admit_loop_poll_s: float = 0.05           # How often router polls for slack (seconds)

    # Optional workload-aware admission (kv cache as a coarse safety valve)
    use_kv_cache_admission: bool = False
    kv_cache_threshold: float = 0.85          # Reject secondary when kv_cache >= threshold
    workload_poll_interval_s: float = 0.5     # Background polling interval for workload
    workload_staleness_threshold_s: float = 2.0
    require_fresh_workload: bool = False

    # Abort handling
    abort_grace_s: float = 1.0                # Grace period for aborted secondaries

    # Startup race prevention
    wait_for_primary_start: bool = True       # Block secondaries until primaries registered

    # Priority scheduling (vLLM scheduler must be set to "priority" policy)
    primary_priority: int = 0                 # Primary batch requests get highest priority
    secondary_priority: int = 10              # Runahead requests get lower priority
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            CALLER (User Code)                                 │
│                                                                               │
│   primary_prompts = [...]      # Current batch                               │
│   secondary_prompts = [...]    # Future batch                                │
│                                                                               │
│   result = await manager.generate_sequences_with_runahead(                   │
│       primary_prompts, secondary_prompts, config                             │
│   )                                                                           │
│                                                                               │
│   # User handles results:                                                     │
│   # - result.primary_outputs → train on these                                │
│   # - result.secondary_outputs → cache for next iteration                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          AgentLoopManager                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  1. Configure router threshold                                         │  │
│  │  2. Split primary_prompts across workers                               │  │
│  │  3. Build local secondary queue from secondary_prompts                 │  │
│  │  4. Busy loop: ray.wait() + drip-feed secondary via router             │  │
│  │  5. Abort/cancel remaining secondary, then merge results               │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  AgentLoopWorker 0 │  │  AgentLoopWorker 1 │  │  AgentLoopWorker N │
│                    │  │                    │  │                    │
│  4-Phase Execution │  │  4-Phase Execution │  │  4-Phase Execution │
│  ────────────────  │  │  ────────────────  │  │  ────────────────  │
│  Phase 1: Primary  │  │  Phase 1: Primary  │  │  Phase 1: Primary  │
│    + Opp. Sec.     │  │    + Opp. Sec.     │  │    + Opp. Sec.     │
│  Phase 2: Notify   │  │  Phase 2: Notify   │  │  Phase 2: Notify   │
│    Barrier         │  │    Barrier         │  │    Barrier         │
│  Phase 3: Fill     │  │  Phase 3: Fill     │  │  Phase 3: Fill     │
│    Until Done      │  │    Until Done      │  │    Until Done      │
│  Phase 4: Abort    │  │  Phase 4: Abort    │  │  Phase 4: Abort    │
│    Remaining       │  │    Remaining       │  │    Remaining       │
└─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
          │                       │                       │
          │    ┌──────────────────┴──────────────────┐    │
          │    │        Manager Busy Loop            │    │
          │    │  (local coordination)               │    │
          │    │                                     │    │
          │    │  • ray.wait(primary+secondary)      │    │
          │    │  • drip-feed secondary via router   │    │
          │    │  • abort remaining secondary        │    │
          │    └──────────────────┬──────────────────┘    │
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CentralRouter (Extended for Runahead)                      │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Primary routing: LRU-based sticky sessions (existing)                 │  │
│  │  Secondary routing: Round-robin (don't pollute LRU cache)              │  │
│  │  abort_requests([ids]): Targeted abort by server_request_id           │  │
│  │  _request_to_server: dict[server_request_id → server_idx]             │  │
│  │  server_load: Global visibility of active requests per server         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│    vLLM Server 0   │  │    vLLM Server 1   │  │    vLLM Server N   │
│                    │  │                    │  │                    │
│  in_flight_total   │  │  in_flight_total   │  │  in_flight_total   │
│  in_flight_secondary│ │  in_flight_secondary│ │  in_flight_secondary│
│                    │  │                    │  │                    │
│  should_admit_sec  │  │  should_admit_sec  │  │  should_admit_sec  │
│  abort_request(id) │  │  abort_request(id) │  │  abort_request(id) │
└────────────────────┘  └────────────────────┘  └────────────────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                          GLOBAL TRUTH
                   (server-side counters)
```

---

## Component Specifications

### 1. RunaheadCentralRouter

Extension to `CentralRouter` with:

- **Primary reservation**: Blocks ALL secondary admission until all primaries have arrived
  at the router. The manager calls `reserve_primary_load(N)` before dispatching, and each
  primary arriving at `generate()` releases one reservation. Uses global tracking (not
  per-server) to avoid distribution mismatch when primaries don't spread evenly.
- **Secondary admission**: Only admit when `server_load < load_threshold` AND
  `_primary_reserved_total == 0`
- **Secondary routing**: Least-loaded server (doesn't touch primary sticky session cache)
- **Request tracking**: Maps `server_request_id → server_idx` for targeted abort
- **Targeted abort**: Groups requests by server, issues per-id `abort_request()` calls
  in parallel (vLLM exposes per-request abort, not batch)

### 2. AgentLoopManager Busy Loop

The manager executes the busy loop (see timeline diagrams above):

1. **Phase 1**: Run primary batch + opportunistically launch secondary when slack detected
2. **Phase 2**: Keep launching secondary while any primary is still running
3. **Phase 3**: Abort remaining secondary by `server_request_id`, collect partial results

Key behaviors:
- Polls every ~50ms to check for completed primary and available slack
- Pulls secondary work from a manager-local queue
- Tracks `sample_id → (task, server_request_id)` for targeted abort
- Requeues rejected secondary for retry (up to `max_retries`)

### 3. Server-Side Admission

#### Current Prototype: Ray Wrapper Pattern

See `test_vllm_runahead_server_side_admission_prototype.py` for reference implementation.

The prototype implements admission as a **Ray actor wrapper** (`AdmissionControlledServer`)
around the vLLM server, avoiding Verl library modifications:

- **Global counters**: `runahead_inflight` shared across all workers using this actor
- **Admission logic**: Check `max_runahead_inflight` limit + optional workload metrics (waiting queue, KV cache)
- **Request kind**: Pops `sampling_params["_verl_request_kind"]` before forwarding to vLLM
- **Rejection**: Returns `TokenOutput(stop_reason="rejected")` when capacity full

> **Multi-Worker Race Condition Note**: This approach only fixes multi-worker races
> if all workers share the **same named/detached admission gate actor per server**.
> If each worker instantiates its own wrapper, you're back to per-worker local
> counters and races. Always use `name=f"admission_gate_{server_idx}"` when creating
> the wrapper actors.

#### Intended End-State: Headroom-Based Admission in vLLM Server

The long-term design moves admission logic **inside the vLLM server** for tighter
scheduler integration:

- **Rule 1 - Secondary capacity**: `in_flight_secondary < max_num_seqs * secondary_frac`
- **Rule 2 - Primary headroom**: `(max_num_seqs - in_flight_total) > primary_headroom`

**Why headroom-based is better:**
- Integrated with vLLM scheduler (can check actual KV cache, waiting queue)
- No extra Ray actor hop per request
- Natural integration with vLLM priority scheduling (future)

---

## Sequence Diagram

```
Caller          Manager          Worker[0..N]            Router              vLLM Server
  │                │                │                │                    │                │
  │ generate_with_ │                │                │                    │                │
  │ runahead(      │                │                │                    │                │
  │   primary,     │                │                │                    │                │
  │   secondary,   │                │                │                    │                │
  │   config)      │                │                │                    │                │
  │───────────────>│                │                │                    │                │
  │                │                │                │                    │                │
  │                │ start(step_id, │                │                    │                │
  │                │   num_workers) │                │                    │                │
  │                │───────────────>│                │                    │                │
  │                │                │                │                    │                │
  │                │ build secondary queue (local)                        │                │
  │                │─────────────────────────────────────────────────────>│                │
  │                │                │                │                    │                │
  │                │ run_with_runahead(primary_chunk, step_id, ...)       │                │
  │                │───────────────────────────────>│                    │                │
  │                │                │                │                    │                │
  │                │                │    ┌──────────┴──────────┐          │                │
  │                │                │    │ PHASE 1: Primary +  │          │                │
  │                │                │    │ opportunistic sec   │          │                │
  │                │                │    └──────────┬──────────┘          │                │
  │                │                │                │                    │                │
  │                │                │                │ generate(primary)  │                │
  │                │                │                │───────────────────>│                │
  │                │                │                │                    │ generate(kind=│
  │                │                │                │                    │   primary)    │
  │                │                │                │                    │──────────────>│
  │                │                │                │                    │               │
  │                │                │                │ generate_secondary()│              │
  │                │                │                │───────────────────>│               │
  │                │                │                │                    │ generate(kind=│
  │                │                │                │                    │   secondary)  │
  │                │                │                │                    │──────────────>│
  │                │                │                │                    │               │
  │                │                │                │                    │  admit_sec()? │
  │                │                │                │                    │<─ ─ ─ ─ ─ ─ ─│
  │                │                │                │                    │               │
  │                │                │    ┌──────────┴──────────┐          │               │
  │                │                │    │ PHASE 2: Notify     │          │               │
  │                │                │    │ barrier             │          │               │
  │                │                │    └──────────┬──────────┘          │               │
  │                │                │                │                    │               │
  │                │                │ mark_primary_  │                    │               │
  │                │                │ done(step_id,  │                    │               │
  │                │                │   worker_id)   │                    │               │
  │                │                │<───────────────│                    │               │
  │                │                │                │                    │               │
  │                │                │    ┌──────────┴──────────┐          │               │
  │                │                │    │ PHASE 3: Fill until │          │               │
  │                │                │    │ global done         │          │               │
  │                │                │    └──────────┬──────────┘          │               │
  │                │                │                │                    │               │
  │                │                │ primary_done()? │                    │               │
  │                │                │<───────────────│                    │               │
  │                │                │───────────────>│ (continue sec.)    │               │
  │                │                │                │                    │               │
  │                │                │    ┌──────────┴──────────┐          │               │
  │                │                │    │ PHASE 4: Abort      │          │               │
  │                │                │    │ remaining secondary │          │               │
  │                │                │    └──────────┬──────────┘          │               │
  │                │                │                │                    │               │
  │                │                │                │ abort_requests(ids)│               │
  │                │                │                │───────────────────>│               │
  │                │                │                │                    │ abort_request │
  │                │                │                │                    │──────────────>│
  │                │                │                │                    │               │
  │                │                │                │<───────────────────│               │
  │                │                │                │ WorkerRunaheadResult               │
  │                │<───────────────────────────────│                    │               │
  │                │                │                │                    │               │
  │ RunaheadResult │                │                │                    │               │
  │<───────────────│                │                │                    │               │
  │                │                │                │                    │               │
```

---

## File Structure

```
verl/experimental/agent_loop/
├── agent_loop.py                    # Add: generate_sequences_with_runahead() (manager busy loop)
├── router.py                        # Add: RunaheadCentralRouter (secondary admission + targeted abort)
└── runahead/
    ├── __init__.py
    ├── config.py                    # RunaheadConfig
    ├── types.py                     # SecondaryOutput, RunaheadResult, etc.
```

---

## Implementation Checklist

> Note: The current prototype uses a **manager-level Ray-native busy loop** (`ray.wait`) plus a shared
> **CentralRouter extension** for secondary admission + targeted abort. Server-side admission inside
> `vllm_async_server.py` is deferred.

### Phase 1: Router Infrastructure
- [ ] Add `RunaheadCentralRouter.generate_secondary(server_request_id, ...)` with admission control
- [ ] Add `RunaheadCentralRouter.abort_requests([ids])` for targeted abort
- [ ] Add `RunaheadCentralRouter.set_load_threshold()` for per-run configuration

### Phase 3: Client Components
- [ ] Create `RunaheadConfig` dataclass
- [ ] Create result types: `SecondaryOutput`, `RunaheadResult`

### Phase 4: Manager Integration
- [ ] Add `generate_sequences_with_runahead()` to AgentLoopManager (busy loop + drip-feed)
- [ ] Preserve primary output order deterministically (match `generate_sequences()`)
- [ ] Abort/cancel remaining secondary when primary completes

### Phase 5: Testing
- [ ] Unit tests for each component
- [ ] Integration test with mock servers
- [ ] End-to-end test with real vLLM

---

## Invariants

1. **Primary never harmed**: Secondary cannot slow down or affect primary
2. **Targeted abort only**: Never `abort_all_requests()` in multi-worker
3. **Single source of truth**: Admission is enforced in one shared component (router for prototype; server for end-state)
4. **Bounded**: Max concurrent, max tokens, max retries all capped

---

## Slack-Filling Submission Logic

This section details the continuous slack-filling approach implemented in
`test_vllm_runahead_slack_filling.py`.

### Submission Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SLACK-FILLING RUNAHEAD LOGIC                         │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │  START: Primary Batch │
                    │  (all submitted       │
                    │   immediately)        │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  MAIN LOOP (while primary_tasks)│
              │  Poll interval: 100ms + jitter  │
              └────────────────┬───────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐
   │ Collect     │    │ Collect done │    │ FEEDER TICK:      │
   │ completed   │    │ runahead     │    │ maybe_submit_     │
   │ primary     │    │ (non-block)  │    │ runahead()        │
   └─────────────┘    └──────────────┘    └─────────┬─────────┘
                                                    │
                                                    ▼
                               ┌────────────────────────────────┐
                               │  Query server workloads        │
                               │  (cached for 300ms + jitter)   │
                               └────────────────┬───────────────┘
                                                │
                                                ▼
                      ┌─────────────────────────────────────────┐
                      │  FOR EACH SERVER: Check slack condition │
                      └─────────────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
                    ▼                                               ▼
        ┌───────────────────────┐                    ┌──────────────────────┐
        │  HAS SLACK?           │                    │  NO SLACK            │
        │  ─────────────────    │                    │  (backpressure)      │
        │  total_load <= 32     │                    │                      │
        │  AND                  │                    │  Skip this server    │
        │  kv_cache <= 85%      │                    │  ++backpressure_     │
        │  AND                  │                    │    events            │
        │  runahead_inflight    │                    └──────────────────────┘
        │    < budget (1)       │
        └───────────┬───────────┘
                    │ YES
                    ▼
        ┌───────────────────────┐
        │  SUBMIT 1 RUNAHEAD    │
        │  to this server       │
        │  (preferred_server_   │
        │   idx = server_idx)   │
        └───────────────────────┘
```

### Slack Condition Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                    SLACK CHECK (per server)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Server has slack if ALL conditions are true:                   │
│                                                                  │
│   1. total_load = (running + waiting) <= load_threshold (32)     │
│      └── "Is the server busy?"                                   │
│                                                                  │
│   2. kv_cache_usage <= kv_cache_threshold (85%)                  │
│      └── "Does the server have memory?"                          │
│                                                                  │
│   3. runahead_inflight[server] < budget_per_server (1)           │
│      └── "Have we already sent runahead to this server?"         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `load_threshold` | 32 | Per-server gate: admit when server_load < threshold |
| `max_queue_size` | 256 | Max pending secondary items in router queue |
| `admit_loop_poll_s` | 0.05 | How often router polls for slack (50ms) |
| `kv_cache_threshold` | 0.85 | Max KV cache usage (85%) |
| `wait_for_primary_start` | True | Block secondaries until all primaries registered |
| `primary_priority` | 0 | Priority for primary requests (lower = higher) |
| `secondary_priority` | 10 | Priority for runahead requests |

### Timeline Example (2 servers, budget=1)

```
Time ──────────────────────────────────────────────────────────────────────────►

                    load_threshold = 32, budget_per_server = 1

SERVER 0:  ═══════════════════════════════════════════════════════════════════
           │ P0 ████████████████████████████████████████│
           │ P1 ██████████████████████│                 │
           │ P2 █████████████│                          │
           │ P3 ██████████████████████████████████████████████████████│ (LONG)
           │                                            │
           │    load=4  load=3  load=2  load=1  load=1  │

SERVER 1:  ═══════════════════════════════════════════════════════════════════
           │ P4 ████████████████████████████████████████│
           │ P5 ████████████████████│                   │
           │ P6 █████████████████│                      │
           │ P7 █████████████████████████████████████████████████████████│ (LONG)
           │                                            │
           │    load=4  load=3  load=2  load=1  load=1  │

RUNAHEAD:  ═══════════════════════════════════════════════════════════════════
           │                                            │
           │  No slack (load > 32? No, but budget=1)    │
           │                                            │
           t=0                                          │
           │                                            │
           │         ┌── P2 completes, Server 0: load=3 │
           │         │   Slack check: load=3 <= 32 ✓    │
           │         │   kv_cache < 85% ✓               │
           │         │   runahead_inflight[0] = 0 < 1 ✓ │
           │         │   → Submit R0 to Server 0        │
           │         ▼                                  │
           │         R0 ████████████████████████████████████│ (completed)
           │                                            │
           │              ┌── P6 completes, Server 1    │
           │              │   → Submit R1 to Server 1   │
           │              ▼                             │
           │              R1 ███████████████████████████████████│ (completed)
           │                                            │
           │                   ┌── P1/P5 complete       │
           │                   │   But R0/R1 still running
           │                   │   runahead_inflight = 1
           │                   │   Budget exhausted! ✗  │
           │                   │   (backpressure event) │
           │                   │                        │
           │                        ┌── R0 completes    │
           │                        │   runahead_inflight[0] = 0
           │                        │   → Submit R2 to Server 0
           │                        ▼                   │
           │                        R2 █████████████████│ (aborted when P3 done)
           │                                            │
           │                                     ┌── P3 done (last primary)
           │                                     │   PRIMARY COMPLETE!
           │                                     │   Cancel remaining runahead
           │                                     ▼
           └─────────────────────────────────────────────────────────────────►
                                                 t=end
```

### Key Timing Concepts

| Phase | Timing | Description |
|-------|--------|-------------|
| **Primary Submission** | t=0 (immediate) | All primary requests submitted in parallel at start |
| **Runahead Submission** | Continuous, opportunistic | Every 100ms poll + jitter, only when slack available |
| **Runahead Cancellation** | When primary completes | All in-flight runahead cancelled, server abort called |

### When Runahead Happens

| Condition | Runahead Submitted? |
|-----------|---------------------|
| Primary dispatch window (~1.5s) | No (primary reservation blocks) |
| All primaries arrived, servers busy | No (no slack) |
| Some primaries complete, server load drops | **Yes** (slack detected) |
| Runahead already in-flight on server | No (budget exhausted) |
| KV cache > 85% | No (memory pressure) |
| Primary batch completes | Cancel all runahead |

### Key Insight

**Runahead fills the "bubbles"** created when short primary requests complete before long ones,
utilizing otherwise idle GPU capacity. The continuous slack-filling approach (drip-feed) is more
efficient than one-shot batch triggers because:

1. Runahead starts as soon as ANY slack appears
2. Backpressure stops feeding automatically when servers get busy
3. Per-server budgets prevent overloading individual servers
4. Safe cancellation ensures no orphaned requests

---

## Primary Reservation (Startup Race Prevention)

### The Problem: Startup Race Condition

When primaries are dispatched via Ray async actors, there's a timing window (~1.5s for 2048
requests) before they arrive at `router.generate()` and increment `server_load`:

```
Time 0:      Manager dispatches 2048 primaries via Ray
             router.server_load = 0 (primaries still in Ray async queue)

Time 0-1.5s: Ray delivers primaries to workers
             Primaries enter router.generate() incrementally
             server_load increases: 0 → 100 → 500 → 1000 → ...

Problem:     At time 0, server_load=0 looks like slack!
             Secondaries get admitted before primaries even start.
```

### The Solution: Primary Reservation

The manager calls `reserve_primary_load(N)` before dispatching primaries, setting
`_primary_reserved_total = N`. Each primary arriving at `generate()` decrements this counter.
`pick_slack_server()` blocks ALL secondaries while `_primary_reserved_total > 0`.

```
Time 0:      reserve_primary_load(2048)
             _primary_reserved_total = 2048
             pick_slack_server() returns None → ALL secondaries blocked

Time 0-1.5s: Primaries arrive at router.generate()
             Each: _primary_reserved_total -= 1
             Still > 0 → Still blocked

Time ~1.5s:  Last primary arrives
             _primary_reserved_total = 0
             pick_slack_server() uses actual server_load
             Secondary admission begins based on real slack
```

### Why Global (Not Per-Server) Tracking

Primary reservation uses a single global counter rather than per-server distribution because:

1. Primaries don't distribute evenly across servers (load balancing, sticky sessions)
2. Per-server estimation (`total // num_servers`) creates mismatch when distribution is uneven
3. The goal is simple: block until ALL primaries have registered, then use actual `server_load`

```python
# In pick_slack_server():
if self._primary_reserved_total > 0:
    return None  # Block ALL secondaries

# Use actual server_load only (no reservation estimate)
load = self.server_load[idx] + self._secondary_reserved_load.get(idx, 0)
```
