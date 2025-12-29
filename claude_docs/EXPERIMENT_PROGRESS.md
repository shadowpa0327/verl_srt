# Runahead Rollout Experiment Progress

**Last Updated:** 2025-12-29
**Status:** v7 experiments interrupted (5/54 completed)

## Overview

We are benchmarking the runahead/slack-filling strategy for vLLM rollout in the veRL framework. The goal is to measure the trade-off between primary task overhead and additional tokens generated during GPU idle time.

## Key Concept

- **Runahead/Slack-filling**: Speculatively start future batch requests during GPU idle time (when waiting for long-tail requests to complete)
- **Long-tail distribution**: Mix of short requests (1-2K tokens) and long requests (16K tokens)
- **load_threshold**: Maximum `total_load` before runahead submission is paused (backpressure)
- **Primary overhead**: Extra time caused by runahead competing for GPU resources

## Experiment Versions

| Version | Model | Batch Sizes | Thresholds | Long-Tail Ratios | Status |
|---------|-------|-------------|------------|------------------|--------|
| v1-v4 | Qwen2.5-0.5B | Various | Various | 20% fixed | Completed |
| v5 | Qwen2.5-0.5B | 32, 64 | 16, 24 | 10%, 20%, 30%, 40%, 50% | **Completed** |
| v6 | Qwen2.5-0.5B | 128, 256 | 16, 24 | 10%-50% | Interrupted |
| v7 | Qwen3-8B | 16, 32, 64 | 8, 16 | 20%, 40%, 50% | **Partial (5/54)** |

---

## V5 Results: Qwen2.5-0.5B-Instruct (Completed)

**File:** `/home/ubuntu/verl_srt/results/runahead_experiment_20251229_025221.json`

### Configuration
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- GPUs: 2 (DP=2, TP=1)
- Short tokens: 1024, Long tokens: 16384
- Rounds per config: 5
- Total experiments: 100

### Results Summary (Averaged across 5 rounds)

| Batch | Thresh | Ratio | Baseline(s) | Runahead(s) | Overhead% | RA Tokens |
|-------|--------|-------|-------------|-------------|-----------|-----------|
| 32 | 16 | 10% | 3.52 | 3.61 | +13.29% | 14,099 |
| 32 | 16 | 20% | 3.54 | 4.17 | +17.17% | 22,082 |
| 32 | 16 | 30% | 3.43 | 4.00 | +16.65% | 21,280 |
| 32 | 16 | 40% | 4.11 | 4.04 | -0.05% | 21,752 |
| 32 | 16 | 50% | 4.23 | 3.95 | -4.77% | 21,035 |
| 32 | 24 | 10% | 3.04 | 3.53 | +20.00% | 24,790 |
| 32 | 24 | 20% | 3.44 | 3.77 | +9.91% | 25,895 |
| 32 | 24 | 30% | 4.02 | 4.69 | +19.13% | 28,111 |
| 32 | 24 | 40% | 3.34 | 4.56 | +37.72% | 28,528 |
| 32 | 24 | 50% | 7.59 | 3.99 | -11.35% | 27,051 |
| 64 | 16 | 10% | 3.66 | 5.40 | +49.82% | 6,126 |
| 64 | 16 | 20% | 4.03 | 5.43 | +36.97% | 11,261 |
| 64 | 16 | 30% | 4.47 | 5.07 | +13.23% | 15,435 |
| 64 | 16 | 40% | 4.47 | 4.71 | +6.60% | 15,791 |
| 64 | 16 | 50% | 5.80 | 5.31 | -4.01% | 17,806 |
| 64 | 24 | 10% | 3.82 | 4.36 | +14.65% | 13,032 |
| 64 | 24 | 20% | 3.82 | 4.68 | +23.20% | 17,671 |
| 64 | 24 | 30% | 4.48 | 6.67 | +53.83% | 25,434 |
| 64 | 24 | 40% | 4.19 | 5.44 | +32.39% | 23,058 |
| 64 | 24 | 50% | 4.96 | 5.14 | +12.53% | 22,592 |

### V5 Key Findings
1. **Break-even at ~50% long-tail ratio** for batch=32, threshold=16
2. Small model (0.5B) generates relatively few runahead tokens (6K-28K)
3. High variance in results suggests model/workload sensitivity

---

## V7 Results: Qwen3-8B (Partial - 5/54 experiments)

**File:** Not saved (interrupted before completion)

### Configuration
- Model: `Qwen/Qwen3-8B`
- GPUs: 2 (DP=2, TP=1)
- Short tokens: 2048, Long tokens: 16384
- Rounds per config: 3
- Total planned: 54 experiments (3 batch × 2 thresh × 3 ratios × 3 rounds)

### Partial Results (Round 1 only, captured from logs)

| Batch | Thresh | Ratio | Baseline(s) | Runahead(s) | Overhead% | RA Tokens | Effective Gain |
|-------|--------|-------|-------------|-------------|-----------|-----------|----------------|
| 16 | 8 | 20% | 140.72 | 158.69 | +12.77% | 75,717 | **+78%** |
| 16 | 8 | 40% | 150.13 | 183.65 | +22.33% | 118,670 | **+63%** |
| 16 | 8 | 50% | 165.29 | 208.04 | +25.87% | 147,185 | **+59%** |
| 16 | 16 | 20% | 149.95 | 166.03 | +10.72% | 75,729 | **+81%** |
| 16 | 16 | 40% | 156.25 | 183.57 | +17.49% | 118,691 | **+69%** |

### V7 Key Findings (Preliminary)
1. **Much higher runahead token generation** (75K-147K vs 6K-28K for 0.5B)
2. Consistent overhead (+10-26%) but **massive effective throughput gain (+59-81%)**
3. Nearly 1:1 ratio of runahead tokens to primary tokens
4. Larger model = more GPU idle time = more opportunity for runahead

### Effective Gain Calculation
```
Baseline throughput = primary_tokens / baseline_time
Runahead throughput = (primary_tokens + ra_tokens) / runahead_time
Effective gain = (runahead_throughput / baseline_throughput) - 1
```

---

## How to Resume Experiments

### Resume V7 (Qwen3-8B)

The benchmark script is already configured for v7. Just run:

```bash
source /home/ubuntu/verl_srt/.venv/bin/activate
NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/benchmark_runahead_tradeoff.py --rounds 3
```

Current v7 configuration in the script:
```python
# In run_experiment_matrix():
primary_sizes = [16, 32, 64]
load_thresholds = [8, 16]
long_tail_ratios = [0.20, 0.40, 0.50]
```

### Modify Experiment Parameters

Edit `/home/ubuntu/verl_srt/tests/workers/rollout/rollout_vllm/benchmark_runahead_tradeoff.py`:

1. **Change model**: Set `MODEL_PATH` env var or modify default in `main()`
2. **Change batch sizes**: Modify `primary_sizes` list in `run_experiment_matrix()`
3. **Change thresholds**: Modify `load_thresholds` list
4. **Change ratios**: Modify `long_tail_ratios` list
5. **Change token lengths**: Set `SHORT_MAX_TOKENS` and `LONG_MAX_TOKENS` env vars

### Environment Variables

```bash
# Model selection
MODEL_PATH="Qwen/Qwen3-8B"  # or Qwen/Qwen2.5-0.5B-Instruct

# Token lengths
SHORT_MAX_TOKENS=2048  # Short request max tokens
LONG_MAX_TOKENS=16384  # Long request max tokens

# GPU configuration
NUM_GPUS=2  # Number of GPUs (DP size)
```

---

## Results File Location

All results are saved to: `/home/ubuntu/verl_srt/results/`

Filename format: `runahead_experiment_YYYYMMDD_HHMMSS.json`

### Parse Results

```python
import json
with open('/home/ubuntu/verl_srt/results/runahead_experiment_XXXXXXXX_XXXXXX.json', 'r') as f:
    data = json.load(f)

for r in data['results']:
    cfg = r['config']
    print(f"Batch={cfg['primary_size']}, Thresh={cfg['load_threshold']}, "
          f"Ratio={cfg['long_tail_ratio']:.0%}, Overhead={r['primary_overhead_pct']:+.2f}%")
```

---

## Suggested Next Experiments

### Priority 1: Complete V7 with Qwen3-8B
- Finish remaining 49/54 experiments
- Expected runtime: ~4-5 hours

### Priority 2: Test Larger Batch Sizes with 8B Model
- Batch sizes: 32, 64, 128 (if memory allows)
- Hypothesis: Larger batches may show even better runahead efficiency

### Priority 3: Test with Even Larger Models
- Qwen3-14B or similar
- Hypothesis: Even more GPU idle time = even better runahead benefit

---

## Architecture Notes

### Key Files

| File | Description |
|------|-------------|
| `tests/workers/rollout/rollout_vllm/benchmark_runahead_tradeoff.py` | Main benchmark script |
| `verl/workers/rollout/async_server.py` | AsyncLLMServerManager with runahead logic |
| `verl/workers/rollout/vllm_server/vllm_http_server.py` | vLLM HTTP server wrapper |
| `claude_docs/runahead_rollout_design.md` | Design document |

### Runahead Flow
1. Primary batch submitted to vLLM servers
2. Monitor `running_load` and `pending_load` on each server
3. When `total_load < threshold`, submit runahead requests
4. Abort runahead requests when primary batch completes
5. Completed runahead tokens = "free" work for next batch

---

## Contact

This experiment is part of the veRL runahead rollout implementation. See `claude_docs/` for design details.
