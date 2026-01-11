# vLLM Scheduling Policy Guide

## Overview

This document outlines how to use the scheduling policy feature in vLLM v1 engine. The feature allows you to control how requests are ordered and processed by the scheduler.

**Availability:** Introduced in vLLM v0.10.0, available in v0.11.0+

## Scheduling Policies

vLLM supports two scheduling policies:

| Policy | Description | Use Case |
|--------|-------------|----------|
| `fcfs` | First Come, First Served (default) | Fair queuing, simple workloads |
| `priority` | Priority-based with arrival time tiebreaker | SLA requirements, tiered service |

## Configuration

### CLI Flag

```bash
# Start server with priority scheduling
python -m vllm.entrypoints.openai.api_server \
    --model <model-name> \
    --scheduling-policy priority

# Default (FCFS)
python -m vllm.entrypoints.openai.api_server \
    --model <model-name> \
    --scheduling-policy fcfs
```

### Python API (Offline Inference)

```python
from vllm import LLM, SamplingParams

# Initialize with priority scheduling
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", scheduling_policy="priority")

# Generate with per-request priorities
prompts = ["High priority prompt", "Medium priority", "Low priority"]
priorities = [0, 5, 10]  # Lower value = higher priority

sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(prompts, sampling_params, priority=priorities)
```

## Per-Request Priority

### OpenAI-Compatible API

When using the OpenAI-compatible server with `--scheduling-policy priority`:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

# High priority request (priority=0)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Urgent request"}],
    extra_body={"priority": 0}  # Highest priority
)

# Low priority request (priority=10)
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Background task"}],
    extra_body={"priority": 10}  # Lower priority
)
```

### Direct HTTP Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "priority": 0
  }'
```

## Priority Ordering Logic

Requests are ordered by the following criteria (in order of precedence):

1. **Priority value** - Lower value = higher priority
2. **Arrival time** - Earlier arrival wins for same priority
3. **Request ID** - Lexicographic ordering for same arrival time

```python
# From vllm/v1/request.py
def __lt__(self, other: "Request") -> bool:
    if self.priority != other.priority:
        return self.priority < other.priority
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    if self.request_id != other.request_id:
        return self.request_id < other.request_id
    return id(self) < id(other)
```

### Example Ordering

```
Request A: priority=1, arrival_time=100
Request B: priority=0, arrival_time=101  <- Processed first (lowest priority value)
Request C: priority=1, arrival_time=99   <- Processed second (same priority as A, earlier arrival)
Request D: priority=1, arrival_time=100  <- Processed after A (same priority/time, request_id tiebreaker)

Processing order: B, C, A, D
```

## Preemption Behavior

When KV cache is exhausted, the scheduler preempts requests differently based on policy:

| Policy | Preemption Strategy |
|--------|---------------------|
| `fcfs` | Preempts the last request added to running queue |
| `priority` | Preempts the lowest-priority running request |

```python
# From vllm/v1/core/sched/scheduler.py:319-323
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )
else:
    preempted_req = self.running.pop()
```

## Implementation Details

### Key Files

| File | Purpose |
|------|---------|
| `vllm/v1/core/sched/request_queue.py` | Queue implementations (FCFS, Priority) |
| `vllm/v1/core/sched/scheduler.py` | Main scheduler with policy-aware logic |
| `vllm/v1/request.py` | Request class with priority field and comparison |
| `vllm/config/scheduler.py` | Configuration definition |
| `vllm/engine/arg_utils.py` | CLI argument parsing |
| `vllm/entrypoints/openai/protocol.py` | API protocol with priority field |

### Queue Implementations

```
┌─────────────────────────────────────────────────────────┐
│                    RequestQueue (ABC)                    │
│  - add_request()    - pop_request()    - peek_request() │
│  - prepend_request()  - remove_request()  - __iter__()  │
└─────────────────────────────────────────────────────────┘
                          ▲
            ┌─────────────┴─────────────┐
            │                           │
┌───────────────────────┐   ┌───────────────────────┐
│   FCFSRequestQueue    │   │  PriorityRequestQueue │
│   (extends deque)     │   │   (uses heapq)        │
│                       │   │                       │
│ - append() to add     │   │ - heappush() to add   │
│ - popleft() to pop    │   │ - heappop() to pop    │
│ - O(1) operations     │   │ - O(log n) operations │
└───────────────────────┘   └───────────────────────┘
```

## Use Cases

### 1. Tiered Service Levels

```python
# Premium users get priority 0
# Standard users get priority 5
# Free tier gets priority 10

def get_priority(user_tier: str) -> int:
    return {"premium": 0, "standard": 5, "free": 10}.get(user_tier, 5)
```

### 2. Latency-Sensitive vs Batch Workloads

```python
# Interactive requests: priority 0
# Batch/background jobs: priority 100
```

### 3. Request Size-Based Priority

```python
# Shorter requests get higher priority for better throughput
def priority_by_length(prompt: str) -> int:
    return min(len(prompt) // 100, 10)
```

## Important Notes

1. **Default priority is 0** - All requests have equal priority by default

2. **Priority only works with priority policy** - Setting `priority != 0` with FCFS policy raises an error:
   ```
   "Any priority other than 0 will raise an error if the served model
   does not use priority scheduling."
   ```

3. **Lower value = higher priority** - Priority 0 is processed before priority 10

4. **Preempted requests rejoin the queue** - They are re-ordered based on their priority

## Relevance to Run-Ahead Rollout

For the run-ahead rollout strategy in Verl, the scheduling policy can be used to:

1. **Prioritize primary batch requests** over run-ahead requests
2. **Allow run-ahead requests to be preempted** when primary work arrives
3. **Control admission** based on request priority levels

Example integration:
```python
# Primary batch requests: priority 0 (highest)
# Run-ahead speculative requests: priority 10 (lower)

# This ensures run-ahead work doesn't delay primary batch completion
primary_priority = 0
runahead_priority = 10
```

## References

- PR #19057: `[Core] feat: Implement Priority Scheduling in V1 Engine`
- PR #29764: Bugfix for `--scheduling-policy=priority` with `n>1`
- Config: `vllm/config/scheduler.py:103-108`
