# Actor, Worker Group, and AgentLoop Architecture

This document explains the relationship between Ray Actors, Worker Groups, and the AgentLoop system in VERL's distributed training framework.

---

## Table of Contents

1. [Ray Actor Concept](#1-ray-actor-concept)
2. [Worker Base Class](#2-worker-base-class)
3. [WorkerGroup Abstraction](#3-workergroup-abstraction)
4. [RayWorkerGroup Implementation](#4-rayworkergroup-implementation)
5. [The @register Decorator](#5-the-register-decorator)
6. [ActorRolloutRefWorker](#6-actorrolloutrefworker)
7. [AsyncActorRolloutRefWorker and AgentLoop Relationship](#7-asyncactorrolloutrefworker-and-agentloop-relationship)
8. [Complete Architecture Diagram](#8-complete-architecture-diagram)

---

## 1. Ray Actor Concept

In Ray, an **Actor** is a stateful worker process. When you decorate a class with `@ray.remote`, instances become distributed actors that:
- Run in separate processes (potentially on different machines)
- Maintain their own state
- Execute methods via `.remote()` calls that return futures

```python
# Ray actor creation (verl/trainer/main_ppo.py)
actor_rollout_cls = AsyncActorRolloutRefWorker
self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
```

---

## 2. Worker Base Class

**Location**: `verl/single_controller/base/worker.py:76`

```python
class Worker(WorkerHelper):
    """A distributed worker that handles initialization and configuration."""

    def _register_dispatch_collect_info(self, mesh_name: str, dp_rank: int, is_collect: bool):
        """Register which dp_rank this worker belongs to for dispatch/collect."""
```

The `Worker` base class provides:
- Distributed communication setup
- Dispatch/collect registration for data parallelism
- Helper methods for master address/port discovery

---

## 3. WorkerGroup Abstraction

**Location**: `verl/single_controller/base/worker_group.py:123`

```python
class WorkerGroup:
    """Manages a group of workers in a distributed system."""

    def __init__(self, resource_pool: ResourcePool, **kwargs):
        self._workers = []        # List of Ray actor handles
        self._worker_names = []
        self._dispatch_info = {}  # How to split data across workers
        self._collect_info = {}   # How to gather results
```

**WorkerGroup** is an abstraction that:
- Holds references to multiple Ray actors
- Provides unified method calls across all workers
- Handles data dispatch (splitting) and collection (gathering)

---

## 4. RayWorkerGroup Implementation

**Location**: `verl/single_controller/ray/base.py:334`

```python
class RayWorkerGroup(WorkerGroup):
    def __init__(self, resource_pool, ray_cls_with_init, ...):
        # Creates Ray actors and binds worker methods
        self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)
```

**Key mechanism**: When you call a method on `RayWorkerGroup`, it:
1. **Dispatches** data to workers (splits batch)
2. **Executes** the method on each worker via Ray `.remote()`
3. **Collects** results back (gathers outputs)

### Method Binding Flow

```
func_generator (base.py:46)
    │
    ├── dispatch_fn(self, *args, **kwargs)  → Split data for each worker
    │
    ├── execute_fn(method_name, *args, **kwargs)  → Call worker.method.remote()
    │
    ├── ray.get(output) if blocking  → Wait for results
    │
    └── collect_fn(self, output)  → Gather results from workers
```

---

## 5. The @register Decorator

**Location**: `verl/single_controller/base/decorator.py:423`

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    ...

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: DataProto):
    ...

@register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
async def wake_up(self):
    ...
```

### Dispatch Modes

| Dispatch Mode | Behavior |
|---------------|----------|
| `ONE_TO_ALL` | Same data sent to all workers |
| `ALL_TO_ALL` | Data split evenly across workers |
| `DP_COMPUTE` | Data split by data-parallel rank |
| `DIRECT_ROLLOUT_METHOD` | Direct call (no dispatch logic, used for async server mode) |

---

## 6. ActorRolloutRefWorker

**Location**: `verl/workers/fsdp_workers.py:134`

```python
class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout
    or a standalone reference policy or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.role = role  # "actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"

        # Register dispatch info for data parallelism
        self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
```

### Roles

| Role | Capabilities |
|------|--------------|
| `actor` | Policy training only |
| `rollout` | Sequence generation only |
| `ref` | Reference policy computation only |
| `actor_rollout` | Training + generation |
| `actor_rollout_ref` | Training + generation + reference (full hybrid) |

### Key Registered Methods

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
    """Initialize model on all workers."""

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: DataProto):
    """Update actor weights (data-parallel dispatch)."""

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
def generate_sequences(self, prompts: DataProto):
    """Generate sequences (data-parallel dispatch)."""
```

---

## 7. AsyncActorRolloutRefWorker and AgentLoop Relationship

### Mode Selection

**Location**: `verl/trainer/main_ppo.py:146-169`

```python
if config.actor_rollout_ref.rollout.mode == "async":
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    actor_rollout_cls = AsyncActorRolloutRefWorker
```

**Location**: `verl/trainer/ppo/ray_trainer.py:919-937`

```python
if self.config.actor_rollout_ref.rollout.mode == "async":
    from verl.experimental.agent_loop import AgentLoopManager
    self.async_rollout_manager = AgentLoopManager(
        config=self.config,
        worker_group=self.actor_rollout_wg,
    )
```

### AsyncActorRolloutRefWorker

**Location**: `verl/workers/fsdp_workers.py:1935`

```python
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()  # Sync weights from FSDP → vLLM engine

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()  # Free rollout memory for training

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(self, prompt_ids, sampling_params, request_id, ...):
        return await self.rollout.generate(...)  # Token generation
```

### Architecture Connection

```
AgentLoopManager
    │
    ├── RolloutReplica (per GPU group)
    │       │
    │       ├── vLLMHttpServer (Ray actor, CPU)
    │       │       │
    │       │       └── workers[] ──────────────► AsyncActorRolloutRefWorker (Ray actors, GPU)
    │       │                                      ├── wake_up()  → rollout_mode()
    │       │                                      ├── sleep()    → trainer_mode()
    │       │                                      └── generate() → rollout.generate()
    │       │
    │       └── wake_up() calls server.wake_up.remote()
    │                          ↓
    │                     worker.wake_up.remote()
    │
    └── AgentLoopWorker[] (Ray actors, CPU)
            └── Routes requests via CentralRouter → vLLMHttpServer
```

### Key Connection Points

**1. Worker Group Binding** (`verl/workers/rollout/replica.py:115-125`)
```python
async def init_hybrid(self, worker_group: RayWorkerGroup):
    self.workers = worker_group.workers[...]  # Gets AsyncActorRolloutRefWorker handles
    await self.launch_servers()
```

**2. Server → Worker Communication** (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:530-533`)
```python
async def wake_up(self):
    if self.rollout_mode == RolloutMode.HYBRID:
        # Calls AsyncActorRolloutRefWorker.wake_up()
        await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
```

**3. Data Flow for Generation**

1. `AgentLoopManager.generate_sequences(prompts)` → calls `wake_up()`
2. `wake_up()` propagates: `RolloutReplica` → `vLLMHttpServer` → `AsyncActorRolloutRefWorker.wake_up()`
3. Worker syncs FSDP weights to vLLM engine via `rollout_mode()`
4. Requests flow: `AgentLoopWorker` → `CentralRouter` → `vLLMHttpServer` → ZMQ → GPU workers
5. After generation: `sleep()` frees rollout memory for training

---

## 8. Complete Architecture Diagram

```
RayPPOTrainer (Driver)
    │
    ├── ResourcePoolManager
    │       └── Creates Ray Placement Groups (GPU allocation)
    │
    ├── RayWorkerGroup (actor_rollout_wg)
    │       │
    │       ├── _workers[] ─────────────► Ray Actor handles
    │       │                             (AsyncActorRolloutRefWorker instances)
    │       │
    │       ├── _dispatch_info ──────────► {mesh_name: dp_rank_mapping}
    │       │
    │       └── Bound Methods:
    │               ├── wg.init_model()         [ONE_TO_ALL]
    │               ├── wg.update_actor(data)   [DP_COMPUTE]
    │               └── wg.generate_sequences() [DP_COMPUTE]
    │
    └── AgentLoopManager (when mode="async")
            │
            ├── RolloutReplica[]
            │       ├── vLLMHttpServer (scheduling, ZMQ dispatch)
            │       └── workers[] → AsyncActorRolloutRefWorker (GPU execution)
            │
            ├── CentralRouter (load balancing)
            │
            └── AgentLoopWorker[] (request processing)
```

### Usage in Training Loop

```python
# In ray_trainer.py
self.actor_rollout_wg = RayWorkerGroup(
    resource_pool=global_resource_pool,
    ray_cls_with_init=RayClassWithInitArgs(cls=AsyncActorRolloutRefWorker, ...),
)

# Sync mode: Direct worker group calls
self.actor_rollout_wg.init_model()                        # ONE_TO_ALL
output = self.actor_rollout_wg.generate_sequences(prompts)  # DP_COMPUTE

# Async mode: AgentLoopManager orchestrates
self.async_rollout_manager.generate_sequences(prompts)    # Routes through servers
```

---

## Summary Table

| Concept | Location | Definition |
|---------|----------|------------|
| **Ray Actor** | Ray framework | Stateful distributed process created via `@ray.remote` |
| **Worker** | `single_controller/base/worker.py` | Base class providing distributed setup and dispatch registration |
| **WorkerGroup** | `single_controller/base/worker_group.py` | Collection of actors with unified method dispatch/collect |
| **RayWorkerGroup** | `single_controller/ray/base.py` | Ray-specific WorkerGroup with placement group support |
| **@register** | `single_controller/base/decorator.py` | Decorator specifying how data is dispatched/collected |
| **ActorRolloutRefWorker** | `workers/fsdp_workers.py` | Concrete worker combining training + inference capabilities |
| **AsyncActorRolloutRefWorker** | `workers/fsdp_workers.py` | Async extension with `wake_up`/`sleep`/`generate` for server mode |
| **AgentLoopManager** | `experimental/agent_loop/agent_loop.py` | Top-level orchestrator for server-based rollout |
