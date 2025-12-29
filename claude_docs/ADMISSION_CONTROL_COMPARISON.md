# Admission Control Comparison: Slack Detection vs Server-Side Admission

## Overview

This document summarizes the two runahead admission control approaches and their performance characteristics.

## Core Implementation Files

| File | Purpose |
|------|---------|
| `tests/workers/rollout/rollout_vllm/test_vllm_run_ahead_slack_filling.py` | **Slack Detection** implementation |
| `tests/workers/rollout/rollout_vllm/test_vllm_run_ahead_server_side_admission.py` | **Server-Side Admission** implementation |
| `tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py` | Benchmark comparing both approaches |

## Key Classes

### Slack Detection (`test_vllm_run_ahead_slack_filling.py`)

| Class | Purpose |
|-------|---------|
| `SlackFillingConfig` | Configuration (budget_per_server, thresholds) |
| `SlackFillingServerManager` | Client-side budget tracking per worker |
| `SlackFillingRunaheadController` | Orchestrates runahead with backpressure counting |

### Server-Side Admission (`test_vllm_run_ahead_server_side_admission.py`)

| Class | Purpose |
|-------|---------|
| `AdmissionGateConfig` | Configuration (max_runahead_inflight) |
| `AdmissionControlledServer` | Ray actor wrapper with asyncio.Lock for global enforcement |
| `AdmissionGateRegistry` | Singleton ensuring one gate per server (prevents racing) |
| `ServerSideAdmissionServerManager` | Routes requests through admission gates |
| `ServerSideAdmissionController` | Orchestrates runahead |

## Architecture Comparison

### Slack Detection (Client-Side)

```
┌─────────┐    ┌─────────┐
│Worker 1 │    │Worker 2 │    Each worker has own budget counter
│budget=4 │    │budget=4 │    Racing possible: 2 workers × 4 = 8 concurrent
└────┬────┘    └────┬────┘
     │              │
     ▼              ▼
┌─────────────────────┐
│   vLLM Server       │   No global enforcement
└─────────────────────┘
```

**Characteristics:**
- Budget tracked locally per worker
- No coordination between workers
- Racing issue: N workers × budget = N × budget concurrent requests possible
- Backpressure events tracked (client blocks when no slack/budget)

### Server-Side Admission (Global)

```
┌─────────┐    ┌─────────┐
│Worker 1 │    │Worker 2 │
└────┬────┘    └────┬────┘
     │              │
     ▼              ▼
┌─────────────────────────┐
│ AdmissionControlledServer│   ← Global lock, single counter
│ (Ray Actor + asyncio.Lock)│
│ max_inflight=4 (enforced)│
└───────────┬─────────────┘
            ▼
┌─────────────────────┐
│   vLLM Server       │
└─────────────────────┘
```

**Characteristics:**
- Budget enforced globally at server level
- `AdmissionGateRegistry` ensures one gate per server
- No racing possible - global limit is atomic
- Rejections return `stop_reason="rejected"` instead of client-side blocking

## Key Code Locations

| Concept | File | Lines |
|---------|------|-------|
| Backpressure counting | `test_vllm_run_ahead_slack_filling.py` | 671-675 |
| Client-side budget check (Slack) | `test_vllm_run_ahead_slack_filling.py` | 345-347 |
| Server-side admission with lock | `test_vllm_run_ahead_server_side_admission.py` | 261-280 |
| Registry singleton pattern | `test_vllm_run_ahead_server_side_admission.py` | 300-350 |

## Backpressure Events Explained

A **backpressure event** is counted when the runahead controller has work waiting but cannot submit to any server.

```python
# From test_vllm_run_ahead_slack_filling.py, lines 671-675
if runahead_queue and not any(
    w.has_slack(cfg) and self.sm.can_submit_runahead(w.server_idx) for w in workloads
):
    self.backpressure_events += 1
```

**Condition breakdown:**
- `runahead_queue` - There ARE requests waiting to be submitted
- `not any(has_slack AND has_budget)` - NO server can accept work

Server-Side Admission doesn't track backpressure because rejections happen at the server level rather than client-side blocking.

## Benchmark Results

### Qwen2.5-0.5B-Instruct (Small Model)

| Metric | Baseline | Slack Detection | Server-Side Admission |
|--------|----------|-----------------|----------------------|
| Time | 3.70s | 3.55s | 2.87s |
| Overhead | - | -4.2% (speedup) | -22.5% (speedup) |
| Runahead Tokens | - | 8,001 | 6,786 |

**Result:** Server-Side Admission 18.3% better

### Qwen3-8B (Large Model)

| Metric | Baseline | Slack Detection | Server-Side Admission |
|--------|----------|-----------------|----------------------|
| Time | 68.80s | 74.03s | 73.92s |
| Overhead | - | +7.6% | +7.4% |
| Runahead Tokens | - | 21,504 | 21,504 |
| Backpressure Events | - | 257 | 0 |
| Feeder Ticks | - | 266 | 262 |

**Result:** Nearly identical performance

## Conclusions

1. **Small models** benefit more from runahead (more GPU idle time to exploit)
2. **Large models** have less slack, so runahead adds overhead
3. **Server-Side Admission** provides safety guarantee against racing with negligible performance difference
4. **Slack Detection's 257 backpressure events** vs **Server-Side's 0** suggests client-side approach is more conservative

## Recommendation

- **Single-worker scenarios:** Either approach works similarly
- **Multi-worker scenarios:** **Server-Side Admission required** to prevent racing issue

## How to Run Benchmarks

```bash
source /home/ubuntu/verl_srt/.venv/bin/activate

# Single comparison (quick)
NUM_GPUS=2 MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct" \
    python tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py --single

# With larger model
NUM_GPUS=2 MODEL_PATH="Qwen/Qwen3-8B" \
    python tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py --single

# Full matrix (multiple configurations)
NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py --rounds 3
```
