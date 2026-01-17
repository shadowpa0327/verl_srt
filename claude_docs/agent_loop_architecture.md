# Agent Loop Architecture with vLLM Rollout

This document explains the complete architecture of VERL's agent loop system with vLLM async rollout, covering class hierarchy, data distribution, load balancing, and GPU communication.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SYSTEM LAYERS                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │ ORCHESTRATION LAYER                                                                │  │
│  │   AgentLoopManager                                                                 │  │
│  │   - Creates worker pool and vLLM servers                                           │  │
│  │   - Chunks batch across workers                                                    │  │
│  │   - Manages server lifecycle (wake_up/sleep)                                       │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │ WORKER LAYER (CPU)                                                                 │  │
│  │   AgentLoopWorker (Ray actors)                                                     │  │
│  │   - Each worker handles a chunk of the batch                                       │  │
│  │   - Creates async tasks per sample                                                 │  │
│  │   - Dynamically instantiates AgentLoopBase subclasses                              │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │ ROUTING LAYER                                                                      │  │
│  │   CentralRouter (single Ray actor, shared by all workers)                          │  │
│  │   - Global least-requests load balancing (min-heap)                                │  │
│  │   - Sticky sessions for prefix caching (LRU cache)                                 │  │
│  │   - Routes requests to vLLM servers                                                │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │ SERVER LAYER (GPU)                                                                 │  │
│  │   vLLMHttpServer (Ray actors, one per DP group)                                    │  │
│  │   - AsyncLLM engine for scheduling/batching                                        │  │
│  │   - ZMQ communication to GPU workers                                               │  │
│  │   - Continuous batching for variable completion times                              │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                           │                                              │
│                                           ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │ GPU LAYER                                                                          │  │
│  │   vLLMAsyncRollout workers (one per GPU)                                           │  │
│  │   - Hold model weight shards                                                       │  │
│  │   - Execute forward passes                                                         │  │
│  │   - NCCL all-reduce for tensor parallelism                                         │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLASS RELATIONSHIPS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CentralRouter (@ray.remote)        (Global load balancing, server routing)  │
│       ↑                                                                      │
│       └── Used by: AgentLoopBase (via self.router.generate.remote())         │
│                                                                              │
│  AgentLoopBase (ABC)                (Abstract interface for agent logic)     │
│       │                                                                      │
│       ├── SingleTurnAgentLoop       @register("single_turn_agent")           │
│       └── ToolAgentLoop             @register("tool_agent")                  │
│                                                                              │
│  AgentLoopWorkerBase                (Batch processing, post-processing)      │
│       │                                                                      │
│       └── AgentLoopWorker           @ray.remote wrapper                      │
│                                                                              │
│  AgentLoopManager                   (Top-level orchestrator)                 │
│       │                                                                      │
│       ├── Creates: CentralRouter (single Ray actor)                          │
│       ├── Creates: AgentLoopWorker pool (CPU)                                │
│       └── Creates: vLLMHttpServer replicas (GPU)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Class | Resource | Responsibility |
|-------|-------|----------|----------------|
| **Orchestration** | AgentLoopManager | - | Worker pool, router creation, batch distribution, lifecycle |
| **Worker** | AgentLoopWorker | CPU | Async task creation, agent instantiation |
| **Routing** | CentralRouter | CPU (Ray actor) | Global load balancing, sticky sessions |
| **Server** | vLLMHttpServer | CPU | AsyncLLM scheduling, ZMQ dispatch |
| **GPU** | vLLMAsyncRollout | GPU | Model weights, forward pass |

---

## Data Flow: Batch to GPU

```
                              Batch (N samples)
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  AgentLoopManager.generate_sequences()                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  chunks = prompts.chunk(len(self.agent_loop_workers))                        │
│  outputs = ray.get([worker.generate_sequences.remote(chunk) for ...])       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  AgentLoopWorker  │     │  AgentLoopWorker  │     │  AgentLoopWorker  │
│      (CPU)        │     │      (CPU)        │     │      (CPU)        │
│                   │     │                   │     │                   │
│ for sample in     │     │ for sample in     │     │ for sample in     │
│   chunk:          │     │   chunk:          │     │   chunk:          │
│   asyncio.create_ │     │   asyncio.create_ │     │   asyncio.create_ │
│   task(agent.run) │     │   task(agent.run) │     │   task(agent.run) │
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CentralRouter.generate() → _choose_server()                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  - Sticky session check (LRU cache)                                          │
│  - Min-heap load balancing (global view across all workers)                  │
│  - Returns least-loaded vLLM server                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  vLLMHttpServer   │    │  vLLMHttpServer   │    │  vLLMHttpServer   │
│     (DP=0)        │    │     (DP=1)        │    │     (DP=N)        │
│                   │    │                   │    │                   │
│   AsyncLLM        │    │   AsyncLLM        │    │   AsyncLLM        │
│  (scheduler)      │    │  (scheduler)      │    │  (scheduler)      │
└─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
          │                        │                        │
          ▼                        ▼                        ▼
    ┌─────┬─────┐            ┌─────┬─────┐            ┌─────┬─────┐
    │GPU 0│GPU 1│            │GPU 2│GPU 3│            │GPU N│GPU N+1│
    │TP=0 │TP=1 │            │TP=0 │TP=1 │            │TP=0 │TP=1 │
    └──┬──┴──┬──┘            └──┬──┴──┬──┘            └──┬──┴──┬──┘
       └─NCCL─┘                 └─NCCL─┘                 └─NCCL─┘
```

---

## Centralized Routing Architecture

All CPU workers route requests through a single CentralRouter that provides global load balancing:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              AgentLoopManager                                         │
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         LLM Rollout Replicas (GPU)                             │  │
│  │                                                                                │  │
│  │    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │  │
│  │    │   Server 0   │   │   Server 1   │   │   Server 2   │   │   Server 3   │   │  │
│  │    │  (vLLM DP=0) │   │  (vLLM DP=1) │   │  (vLLM DP=2) │   │  (vLLM DP=3) │   │  │
│  │    │    [GPUs]    │   │    [GPUs]    │   │    [GPUs]    │   │    [GPUs]    │   │  │
│  │    └──────▲───────┘   └──────▲───────┘   └──────▲───────┘   └──────▲───────┘   │  │
│  │           │                  │                  │                  │           │  │
│  └───────────┼──────────────────┼──────────────────┼──────────────────┼───────────┘  │
│              │                  │                  │                  │              │
│              └──────────────────┼──────────────────┼──────────────────┘              │
│                                 │                  │                                 │
│                        ┌────────┴──────────────────┴────────┐                        │
│                        │         CentralRouter              │                        │
│                        │      (Single Ray Actor)            │                        │
│                        │  - Global load visibility          │                        │
│                        │  - Least-requests routing          │                        │
│                        │  - Sticky sessions (LRU)           │                        │
│                        └────────┬──────────────────┬────────┘                        │
│                                 │                  │                                 │
│              ┌──────────────────┼──────────────────┼──────────────────┐              │
│              │                  │                  │                  │              │
│  ┌───────────┼──────────────────┼──────────────────┼──────────────────┼───────────┐  │
│  │           │                  │                  │                  │           │  │
│  │    ┌──────┴───────┐   ┌──────┴───────┐   ┌──────┴───────┐   ┌──────┴───────┐   │  │
│  │    │   Worker 0   │   │   Worker 1   │   │   Worker 2   │   │   Worker 3   │   │  │
│  │    │              │   │              │   │              │   │              │   │  │
│  │    │ self.router  │   │ self.router  │   │ self.router  │   │ self.router  │   │  │
│  │    │  .generate   │   │  .generate   │   │  .generate   │   │  .generate   │   │  │
│  │    │  .remote()   │   │  .remote()   │   │  .remote()   │   │  .remote()   │   │  │
│  │    │    [CPU]     │   │    [CPU]     │   │    [CPU]     │   │    [CPU]     │   │  │
│  │    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   │  │
│  │                                                                                │  │
│  │                         AgentLoopWorkers (CPU)                                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Properties:**
- Single `CentralRouter` Ray actor shared by all workers
- Global visibility of server loads enables optimal load balancing
- Essential for run-ahead rollout strategy (needs global view of queue depths)
- Workers call `self.router.generate.remote()` directly

---

## Load Balancing

### Min-Heap Algorithm

```python
@ray.remote
class CentralRouter:
    def __init__(self, server_handles, max_cache_size=10000):
        self.server_handles = server_handles

        # Min-heap: [num_sessions, server_index, server_handle]
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache for sticky sessions (stores server_idx only)
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # Track active load per server
        self.server_load = {i: 0 for i in range(len(server_handles))}

    def _choose_server(self, request_id: str):
        # 1. Check sticky session cache
        if request_id in self.request_id_to_server:
            server_idx = self.request_id_to_server[request_id]
            return server_idx, self.server_handles[server_idx]

        # 2. Pick server with minimum load (heap root)
        _, server_idx, server = self.weighted_serveres[0]

        # 3. Increment and rebalance heap
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        # 4. Cache for sticky sessions
        self.request_id_to_server[request_id] = server_idx
        return server_idx, server

    async def generate(self, request_id, *, prompt_ids, sampling_params, ...):
        server_idx, server = self._choose_server(request_id)
        self.server_load[server_idx] += 1
        try:
            output = await server.generate.remote(...)
            return output
        finally:
            self.server_load[server_idx] -= 1
```

**Complexity:** O(log N) per request, where N = number of servers

### Sticky Sessions for Multi-Turn

For tool-calling agents with multiple LLM turns:

```
Turn 1: request_id="conv_123" → assigned to Server0, cached
Turn 2: request_id="conv_123" → cache HIT → Server0 (prefix cache reused)
Turn 3: request_id="conv_123" → cache HIT → Server0 (prefix cache reused)
```

**Benefits:** Enables vLLM prefix caching - no re-encoding of conversation history.

---

## vLLM Server Architecture

### Engine vs Workers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLMHttpServer (Ray Actor)                    │
│                              (NO GPU)                                │
│                                                                     │
│   self.engine = AsyncLLM.from_vllm_config(...)                      │
│                    │                                                │
│                    │ ExternalZeroMQDistributedExecutor              │
│                    │                                                │
│                    ▼                                                │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  collective_rpc() sends commands to workers via ZMQ         │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ Worker R=0   │  │ Worker R=1   │  │ Worker R=2   │  ...
  │ (GPU 0)      │  │ (GPU 1)      │  │ (GPU 2)      │
  │              │  │              │  │              │
  │ ZMQ REP sock │  │ ZMQ REP sock │  │ ZMQ REP sock │
  │ Model shard  │  │ Model shard  │  │ Model shard  │
  └──────────────┘  └──────────────┘  └──────────────┘
```

| Component | GPU? | Purpose |
|-----------|------|---------|
| `AsyncLLM` engine | No | Scheduling, continuous batching |
| ZMQ communication layer | No | Route commands to workers |
| `vLLMAsyncRollout` workers | Yes | Model weights, forward pass |

### ZMQ Socket Topology (TP=2, DP=4)

```
Workers (8 GPUs):
  R=0 → ipc://zmq_0.ipc ─┐
  R=1 → ipc://zmq_1.ipc ─┼─ Engine DP=0
  R=2 → ipc://zmq_2.ipc ─┐
  R=3 → ipc://zmq_3.ipc ─┼─ Engine DP=1
  R=4 → ipc://zmq_4.ipc ─┐
  R=5 → ipc://zmq_5.ipc ─┼─ Engine DP=2
  R=6 → ipc://zmq_6.ipc ─┐
  R=7 → ipc://zmq_7.ipc ─┼─ Engine DP=3
```

Each DP group's engine only connects to its own TP workers.

---

## Agent Registration System

Agent loops are registered via decorator and instantiated dynamically:

```python
# Registration
_agent_loop_registry = {}

def register(agent_name: str):
    def decorator(cls):
        _agent_loop_registry[agent_name] = f"{cls.__module__}.{cls.__qualname__}"
        return cls
    return decorator

@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase): ...

@register("tool_agent")
class ToolAgentLoop(AgentLoopBase): ...

# Dynamic instantiation in AgentLoopWorkerBase
agent_loop_class = get_agent_loop_class(config.agent.agent_name)
agent_loop = agent_loop_class(config, router, tokenizer, ...)
output = await agent_loop.run(sampling_params, **kwargs)
```

---

## Configuration

### Key Parameters

| Config Path | Default | Description |
|-------------|---------|-------------|
| `actor_rollout_ref.rollout.mode` | `sync` | `async` enables server mode |
| `actor_rollout_ref.rollout.agent.num_workers` | 8 | CPU workers for agent loops |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 1 | GPUs per model shard |
| `actor_rollout_ref.rollout.data_parallel_size` | 1 | Number of model replicas |

### Derived Values

```
num_servers = (n_gpus_per_node * nnodes) / (TP * DP * PP)
max_concurrency = num_workers * samples_per_worker * turns_per_sample
```

---

## Extensibility

Custom implementations at each layer:

```python
# Custom router (extend CentralRouter for run-ahead, etc.)
@ray.remote
class MyRouter(CentralRouter):
    async def generate_with_runahead(...): ...

# Custom worker
@ray.remote
class MyAgentLoopWorker(AgentLoopWorkerBase):
    # Custom batch processing

# Custom manager
class MyAgentLoopManager(AgentLoopManager):
    async def generate_sequences_async(...): ...
```

Config-based loading:
```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: "mypackage.MyManager"  # FQDN
```

---

## Code Index

### Core Agent Loop Classes

| Class | File | Lines |
|-------|------|-------|
| `CentralRouter` | `verl/experimental/agent_loop/router.py` | 40-170 |
| `AgentLoopBase` | `verl/experimental/agent_loop/agent_loop.py` | 120-160 |
| `AgentLoopWorkerBase` | `verl/experimental/agent_loop/agent_loop.py` | 178-600 |
| `AgentLoopWorker` | `verl/experimental/agent_loop/agent_loop.py` | 596-610 |
| `AgentLoopManager` | `verl/experimental/agent_loop/agent_loop.py` | 630-800 |
| `register()` | `verl/experimental/agent_loop/agent_loop.py` | 55-70 |
| `AgentLoopConfig` | `verl/workers/config/rollout.py` | 68+ |

### Agent Implementations

| Class | File | Registered Name |
|-------|------|-----------------|
| `SingleTurnAgentLoop` | `verl/experimental/agent_loop/single_turn_agent_loop.py` | `"single_turn_agent"` |
| `ToolAgentLoop` | `verl/experimental/agent_loop/tool_agent_loop.py` | `"tool_agent"` |

### vLLM Server Components

| Class | File | Lines |
|-------|------|-------|
| `vLLMHttpServerBase` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 163+ |
| `vLLMHttpServer` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 846+ |
| `vLLMAsyncRollout` | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 112+ |
| ZMQ socket init | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 137+ |
| `get_zeromq_address()` | `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 273+ |

### Ray Infrastructure

| Component | File | Lines |
|-----------|------|-------|
| Worker spawn | `verl/single_controller/ray/base.py` | 362-443 |
| Rank assignment | `verl/single_controller/ray/base.py` | 394-401 |
| Mode selection | `verl/trainer/main_ppo.py` | 127-130 |

### Runahead Extensions

*Note: Run-ahead functionality is being developed. CentralRouter provides the global visibility needed for run-ahead by tracking `server_load` across all workers.*

---

## See Also

- [vLLM Documentation](https://docs.vllm.ai/) - AsyncLLM engine details
- [Ray Documentation](https://docs.ray.io/) - Actor model and placement groups
