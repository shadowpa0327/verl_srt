# Runahead Rollout: Focused Design (Rollout-Only)

## Scope

**In scope:**
- Single-turn rollout with runahead speculation
- Server-side admission control
- Multi-worker coordination via StepBarrier
- Targeted abort (never abort_all)
- Return `(primary_results, secondary_results)` to caller

**Out of scope (handled by caller):**
- Result caching and reuse
- Trainer integration
- Multi-turn agent speculation
- vLLM priority scheduling (defer)

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

Runahead detects when capacity becomes available and speculatively starts future batch (secondary) requests:

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
     │   │  • Pull S0, S1 from SecondaryWorkQueue
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
     │  ║ PHASE 2: Notify barrier                                              ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │   │
t=1.51s  ├─ MARK PRIMARY DONE ────────────────────────────────────────────────
     │   │  • await step_barrier.mark_primary_done(step_id, worker_id)
     │   │  • Other workers may still be running primary
     │   │
     │  ╔═══════════════════════════════════════════════════════════════════════╗
     │  ║ PHASE 3: Keep filling until global done                              ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │   │
t=1.52s  ├─ CONTINUE SECONDARY ───────────────────────────────────────────────
     │   │  • while not await step_barrier.is_done(step_id):
     │   │  •     _try_launch_secondary()  # Pull more from queue
     │   │  •     _collect_secondary()     # Gather completed
     │   │  • S2 at 150/300 tokens, S3 at 120/300 tokens
     │   │
t=2.80s  ├─ GLOBAL PRIMARY DONE ──────────────────────────────────────────────
     │   │  • step_barrier.is_done() returns True
     │   │  • All workers have finished their primary batches
     │   │
     │  ╔═══════════════════════════════════════════════════════════════════════╗
     │  ║ PHASE 4: Abort remaining secondary                                   ║
     │  ╚═══════════════════════════════════════════════════════════════════════╝
     │   │
t=2.81s  ├─ TARGETED ABORT ───────────────────────────────────────────────────
     │   │  • running_ids = [S2.server_request_id, S3.server_request_id]
     │   │  • await server_manager.abort_requests(running_ids)
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

### Multi-Worker Timeline (2 Workers, StepBarrier Coordination)

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
│    SecondaryWorkQueue              │       │                                    │
│                                    │       │                                    │
│ t=1.0s: LOCAL PRIMARY DONE         │       │                                    │
│         ▼                          │       │                                    │
│ PHASE 2: mark_primary_done()       │       │                                    │
│         │                          │       │                                    │
│         ▼                          │       │                                    │
│ PHASE 3: Keep filling              │       │                                    │
│ ───────────────────────────────────│       │                                    │
│ S2     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │       │                                    │
│ S3     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   │       │                                    │
│                                    │       │                                    │
│ (Worker 0 keeps working on sec.)   │       │ t=2.8s: LOCAL PRIMARY DONE         │
│                                    │       │         ▼                          │
│                                    │       │ PHASE 2: mark_primary_done()       │
│                                    │       │                                    │
│ t=2.8s: is_done() → TRUE           │       │ t=2.8s: is_done() → TRUE           │
│         ▼                          │       │         ▼                          │
│ PHASE 4: Abort S2, S3              │       │ PHASE 4: (no secondary running)    │
│                                    │       │                                    │
└────────────────────────────────────┘       └────────────────────────────────────┘
         │                                            │
         │              ┌─────────────────────────────┘
         │              │
         ▼              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            StepBarrier                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  step_id: "step_42"                                                   │  │
│  │  num_workers: 2                                                       │  │
│  │  completed: {0, 1}  ← Both workers done                               │  │
│  │  done_event: SET                                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Timeline:
─────────────────────────────────────────────────────────────────────────────────
t=0.0s   │ Both workers start primary batch
t=0.3s   │ Worker 0: P0,P1 done → starts S0,S1 (pulls from shared queue)
t=0.5s   │ Worker 0: P2,P3 done → starts S2,S3
t=1.0s   │ Worker 0: All primary done → PHASE 2 (mark_primary_done)
         │ Worker 0: Enters PHASE 3 (keep filling)
t=1.5s   │ Worker 0: S0,S1 complete
t=2.8s   │ Worker 1: All primary done → PHASE 2 (mark_primary_done)
         │ StepBarrier: done_event.set() (all workers done)
         │ Both workers: is_done() → True → PHASE 4
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

    # Server-side limits
    secondary_frac: float = 0.20              # Max 20% of server capacity for secondary
    primary_headroom: int = 8                 # Always reserve 8 slots for primary

    # Per-worker limits
    max_secondary_concurrent: int = 4         # Max secondary tasks per worker
    max_secondary_tokens: int = 64            # Truncate secondary output (reduce abort waste)

    # Retry behavior
    rejection_backoff_ms: int = 50            # Backoff on server rejection
    max_retries: int = 3                      # Max retry attempts per secondary request
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
│  │  1. Create StepBarrier for this step                                   │  │
│  │  2. Split primary_prompts across workers                               │  │
│  │  3. Create SecondaryWorkQueue from secondary_prompts                   │  │
│  │  4. Dispatch to workers with runahead enabled                          │  │
│  │  5. Collect and merge results                                          │  │
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
          │    │           StepBarrier               │    │
          │    │  (Ray actor - coordination)         │    │
          │    │                                     │    │
          │    │  • start(step_id, num_workers)      │    │
          │    │  • mark_primary_done(step_id, wid)  │    │
          │    │  • is_done(step_id) → bool          │    │
          │    └──────────────────┬──────────────────┘    │
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    AsyncLLMServerManager (Extended)                           │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Primary routing: LRU-based sticky sessions (existing)                 │  │
│  │  Secondary routing: Round-robin (don't pollute LRU cache)              │  │
│  │  abort_requests([ids]): Targeted abort by server_request_id           │  │
│  │  _request_to_server: dict[server_request_id → server_idx]             │  │
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

### 1. StepBarrier

```python
@ray.remote
class StepBarrier:
    """Coordinates multi-worker primary completion."""

    def __init__(self):
        self.steps: dict[str, StepState] = {}

    async def start(self, step_id: str, num_workers: int):
        """Initialize barrier for a step."""
        self.steps[step_id] = StepState(
            num_workers=num_workers,
            completed=set(),
            done_event=asyncio.Event(),
        )

    async def mark_primary_done(self, step_id: str, worker_id: int):
        """Worker reports primary completion."""
        state = self.steps[step_id]
        state.completed.add(worker_id)
        if len(state.completed) >= state.num_workers:
            state.done_event.set()

    async def is_done(self, step_id: str) -> bool:
        """Check if all workers done (non-blocking)."""
        return self.steps[step_id].done_event.is_set()

    async def wait_done(self, step_id: str, timeout: float = None) -> bool:
        """Wait for all workers to complete."""
        try:
            await asyncio.wait_for(
                self.steps[step_id].done_event.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    async def cleanup(self, step_id: str):
        """Remove step state."""
        self.steps.pop(step_id, None)
```

### 2. SecondaryWorkQueue

Shared queue that workers pull from to get secondary work:

```python
@ray.remote
class SecondaryWorkQueue:
    """Distributes secondary work to workers."""

    def __init__(self, secondary_prompts: list[PromptData]):
        self.queue = asyncio.Queue()
        for prompt in secondary_prompts:
            self.queue.put_nowait(SecondaryWorkItem(
                sample_id=prompt.sample_id,
                prompt_ids=prompt.prompt_ids,
                max_tokens=prompt.max_tokens,
            ))
        self.results: dict[str, SecondaryOutput] = {}
        self._lock = asyncio.Lock()

    async def get_work(self) -> Optional[SecondaryWorkItem]:
        """Get next secondary work item (non-blocking)."""
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def requeue(self, item: SecondaryWorkItem):
        """Requeue rejected item for retry."""
        if item.retry_count < item.max_retries:
            item.retry_count += 1
            await self.queue.put(item)

    async def submit_result(self, sample_id: str, result: SecondaryOutput):
        """Submit completed/aborted result."""
        async with self._lock:
            self.results[sample_id] = result

    async def get_all_results(self) -> dict[str, SecondaryOutput]:
        """Get all collected results."""
        async with self._lock:
            return dict(self.results)
```

### 3. RunaheadServerManager

Extension to `AsyncLLMServerManager` with:

- **Secondary routing**: Round-robin (don't pollute primary LRU cache)
- **Request tracking**: Maps `server_request_id → server_idx` for targeted abort
- **Request kind plumbing**: Sets `sampling_params["_verl_request_kind"] = "secondary"`;
  the admission wrapper pops this key before forwarding to vLLM
- **Targeted abort**: Groups requests by server, issues per-id `abort_request()` calls
  in parallel (vLLM exposes per-request abort, not batch)

### 4. RunaheadWorker (4-Phase Execution)

Each worker executes the 4-phase pattern (see timeline diagrams above):

1. **Phase 1**: Run primary batch + opportunistically launch secondary when slack detected
2. **Phase 2**: When local primary done, notify `StepBarrier.mark_primary_done()`
3. **Phase 3**: Keep launching secondary until `StepBarrier.is_done()` returns true
4. **Phase 4**: Abort remaining secondary by `server_request_id`, collect partial results

Key behaviors:
- Polls every ~50ms to check for completed primary and available slack
- Pulls secondary work from shared `SecondaryWorkQueue`
- Tracks `sample_id → (task, server_request_id)` for targeted abort
- Requeues rejected secondary for retry (up to `max_retries`)

### 5. Server-Side Admission

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
Caller          Manager         StepBarrier      Worker[0..N]       ServerManager     vLLM Server
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
  │                │ create SecondaryWorkQueue(secondary_prompts)         │                │
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
  │                │                │ is_done()?     │                    │               │
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
├── agent_loop.py                    # Existing (minimal changes)
├── runahead/
│   ├── __init__.py
│   ├── config.py                    # RunaheadConfig
│   ├── barrier.py                   # StepBarrier Ray actor
│   ├── work_queue.py                # SecondaryWorkQueue Ray actor
│   ├── server_manager.py            # RunaheadServerManager
│   ├── worker_mixin.py              # RunaheadWorkerMixin
│   └── types.py                     # SecondaryOutput, RunaheadResult, etc.

verl/workers/rollout/vllm_rollout/
└── vllm_async_server.py             # Add: in_flight counters, should_admit_secondary, abort_requests
```

---

## Implementation Checklist

### Phase 1: Server Infrastructure
- [ ] Add `in_flight_total`, `in_flight_secondary` counters to vLLMHttpServer
- [ ] Add `meta` parameter to `generate()` with `kind` field
- [ ] Implement `should_admit_secondary()` logic
- [ ] Return `stop_reason="rejected"` when capacity full
- [ ] Implement `abort_requests([ids])` method

### Phase 2: Coordination
- [ ] Create `StepBarrier` Ray actor
- [ ] Create `SecondaryWorkQueue` Ray actor
- [ ] Add tests for barrier coordination

### Phase 3: Client Components
- [ ] Create `RunaheadConfig` dataclass
- [ ] Create `RunaheadServerManager` with secondary routing
- [ ] Create `RunaheadWorkerMixin` with 4-phase execution
- [ ] Create result types: `SecondaryOutput`, `RunaheadResult`

### Phase 4: Manager Integration
- [ ] Add `generate_sequences_with_runahead()` to AgentLoopManager
- [ ] Wire up barrier and work queue
- [ ] Add metrics collection

### Phase 5: Testing
- [ ] Unit tests for each component
- [ ] Integration test with mock servers
- [ ] End-to-end test with real vLLM

---

## Invariants

1. **Primary never harmed**: Secondary cannot slow down or affect primary
2. **Targeted abort only**: Never `abort_all_requests()` in multi-worker
3. **Server is truth**: Global capacity tracked in server, not workers
4. **Bounded**: Max concurrent, max tokens, max retries all capped
