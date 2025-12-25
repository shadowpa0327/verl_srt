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
│  │   AsyncLLMServerManager (per worker)                                               │  │
│  │   - Least-requests load balancing (min-heap)                                       │  │
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
│  AsyncLLMServerManager              (Load balancing, server routing)         │
│       ↑                                                                      │
│       └── Used by: AgentLoopBase, AgentLoopWorkerBase                        │
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
│       ├── Creates: AgentLoopWorker pool (CPU)                                │
│       └── Creates: vLLMHttpServer replicas (GPU)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer Summary

| Layer | Class | Resource | Responsibility |
|-------|-------|----------|----------------|
| **Orchestration** | AgentLoopManager | - | Worker pool, batch distribution, lifecycle |
| **Worker** | AgentLoopWorker | CPU | Async task creation, agent instantiation |
| **Routing** | AsyncLLMServerManager | CPU | Load balancing, sticky sessions |
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
│  AsyncLLMServerManager._choose_server()                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  - Sticky session check (LRU cache)                                          │
│  - Min-heap load balancing                                                   │
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

## Many-to-Many Architecture

Multiple CPU workers can send requests to multiple GPU servers concurrently:

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
│              │    ┌─────────────┴──────────────────┴─────────────┐    │              │
│              │    │                                              │    │              │
│              │    │         Many-to-Many Connections             │    │              │
│              │    │    (Each worker can reach any server)        │    │              │
│              │    │                                              │    │              │
│              │    └─────────────┬──────────────────┬─────────────┘    │              │
│              │                  │                  │                  │              │
│  ┌───────────┼──────────────────┼──────────────────┼──────────────────┼───────────┐  │
│  │           │                  │                  │                  │           │  │
│  │    ┌──────┴───────┐   ┌──────┴───────┐   ┌──────┴───────┐   ┌──────┴───────┐   │  │
│  │    │   Worker 0   │   │   Worker 1   │   │   Worker 2   │   │   Worker 3   │   │  │
│  │    │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │   │  │
│  │    │ │ AsyncLLM │ │   │ │ AsyncLLM │ │   │ │ AsyncLLM │ │   │ │ AsyncLLM │ │   │  │
│  │    │ │ ServerMgr│ │   │ │ ServerMgr│ │   │ │ ServerMgr│ │   │ │ ServerMgr│ │   │  │
│  │    │ └──────────┘ │   │ └──────────┘ │   │ └──────────┘ │   │ └──────────┘ │   │  │
│  │    │    [CPU]     │   │    [CPU]     │   │    [CPU]     │   │    [CPU]     │   │  │
│  │    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   │  │
│  │                                                                                │  │
│  │                         AgentLoopWorkers (CPU)                                 │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Key Properties:**
- Each worker has its own `AsyncLLMServerManager` with handles to ALL servers
- Workers make independent load balancing decisions (decentralized)
- No central bottleneck or coordination required

---

## Load Balancing

### Min-Heap Algorithm

```python
class AsyncLLMServerManager:
    def __init__(self, config, server_handles, max_cache_size=10000):
        # Min-heap: [request_count, server_index, server_handle]
        self.weighted_servers = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_servers)

        # LRU cache for sticky sessions
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str):
        # 1. Check sticky session cache
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        # 2. Pick server with minimum load (heap root)
        _, _, server = self.weighted_servers[0]

        # 3. Increment and rebalance heap
        self.weighted_servers[0][0] += 1
        heapq.heapreplace(self.weighted_servers, self.weighted_servers[0])

        # 4. Cache for sticky sessions
        self.request_id_to_server[request_id] = server
        return server
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
| `ExternalZeroMQDistributedExecutor` | No | Route commands to workers |
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
agent_loop = agent_loop_class(config, server_manager, tokenizer, ...)
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
# Custom server manager
class MyServerManager(AsyncLLMServerManager):
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
| `AsyncLLMServerManager` | `verl/experimental/agent_loop/agent_loop.py` | 54-118 |
| `AgentLoopBase` | `verl/experimental/agent_loop/agent_loop.py` | 186-223 |
| `AgentLoopWorkerBase` | `verl/experimental/agent_loop/agent_loop.py` | 245-664 |
| `AgentLoopWorker` | `verl/experimental/agent_loop/agent_loop.py` | 668-680 |
| `AgentLoopManager` | `verl/experimental/agent_loop/agent_loop.py` | 705-871 |
| `register()` | `verl/experimental/agent_loop/agent_loop.py` | 122-135 |
| `AgentLoopConfig` | `verl/workers/config/rollout.py` | 68+ |

### Agent Implementations

| Class | File | Registered Name |
|-------|------|-----------------|
| `SingleTurnAgentLoop` | `verl/experimental/agent_loop/single_turn_agent_loop.py` | `"single_turn_agent"` |
| `ToolAgentLoop` | `verl/experimental/agent_loop/tool_agent_loop.py` | `"tool_agent"` |

### vLLM Server Components

| Class | File | Lines |
|-------|------|-------|
| `vLLMHttpServer` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 123-442 |
| `ExternalZeroMQDistributedExecutor` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 61-118 |
| `vLLMAsyncRollout` | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 702-884 |
| ZMQ socket init | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | 727-754 |
| `generate()` | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 445-513 |
| Abort mechanisms | `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 545-625 |

### Ray Infrastructure

| Component | File | Lines |
|-----------|------|-------|
| Worker spawn | `verl/single_controller/ray/base.py` | 362-443 |
| Rank assignment | `verl/single_controller/ray/base.py` | 394-401 |
| Mode selection | `verl/trainer/main_ppo.py` | 127-130 |

### Runahead Extensions

| Class | File | Lines |
|-------|------|-------|
| `AsyncLLMServerManagerWithRunahead` | `verl/experimental/agent_loop/runahead.py` | 203-423 |
| `RunaheadController` | `verl/experimental/agent_loop/runahead.py` | 431-582 |

---

## See Also

- [vLLM Documentation](https://docs.vllm.ai/) - AsyncLLM engine details
- [Ray Documentation](https://docs.ray.io/) - Actor model and placement groups
