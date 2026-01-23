# Scaling Experiments for SRT Speculation Analysis

This document describes the tools and workflow for measuring how speculative decoding performance scales with the number of samples used to populate suffix trees.

## Overview

The scaling experiment measures the relationship between:
- **Input**: Number of samples per prompt (k) used to populate the suffix tree cache
- **Output**: Speculation performance metrics (tokens/step, acceptance rate, hit rate)

Two modes are compared:
- **Offline**: Cache populated before simulation, not updated during
- **Online**: Cache updated with newly generated tokens during simulation

## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Generate Expanded Rollouts (run_scaling_sweep.py)         │
├─────────────────────────────────────────────────────────────────────┤
│  For each training step:                                            │
│  1. Merge FSDP checkpoint → vLLM-compatible model                   │
│  2. Select top-k longest response prompts from rollout data         │
│  3. Generate 32 samples per prompt with vLLM                        │
│  4. Save to expanded_rollouts/step_N/scaling_rollouts.jsonl         │
│  5. Delete merged model to save disk space                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: Replay Simulation (scaling_replay_simulator.py)           │
├─────────────────────────────────────────────────────────────────────┤
│  For each k value (1, 2, 4, 8, 16, 32):                             │
│  1. Create fresh suffix tree cache                                  │
│  2. Populate with k samples per prompt from expanded_rollouts       │
│  3. Simulate speculation on actual rollout data                     │
│  4. Measure: tokens/step, acceptance_rate, hit_rate, draft_contrib  │
│  5. (Optional) Online mode: update cache during simulation          │
└─────────────────────────────────────────────────────────────────────┘
```

## Scripts Reference

### Data Generation

| Script | Location | Purpose |
|--------|----------|---------|
| `suffix_tree_scaling_experiment.py` | `recipe/srt/scripts/` | Generate expanded rollouts for a single step |
| `run_scaling_sweep.py` | `recipe/srt/scripts/` | Sweep across multiple training steps |

### Simulation & Analysis

| Script | Location | Purpose |
|--------|----------|---------|
| `scaling_replay_simulator.py` | `recipe/srt/` | Core simulator for measuring speculation metrics |
| `run_scaling_simulation.sh` | Repository root | Run simulation for a single step |
| `run_scaling_sweep.sh` | Repository root | Sweep all steps (offline mode) |
| `run_scaling_sweep_online.sh` | Repository root | Sweep all steps (online mode) |

## Usage

### Phase 1: Generate Expanded Rollouts

```bash
# Generate for specific steps
python -m recipe.srt.scripts.run_scaling_sweep \
    --steps 10,20,50,100 \
    --output-dir expanded_rollouts

# Generate for step range
python -m recipe.srt.scripts.run_scaling_sweep \
    --step-range 10:100:10 \
    --output-dir expanded_rollouts
```

### Phase 2: Run Scaling Simulation

```bash
# Single step simulation
./run_scaling_simulation.sh <step> [max_samples]
./run_scaling_simulation.sh 100 100

# Sweep all steps (offline mode)
./run_scaling_sweep.sh [max_samples]
./run_scaling_sweep.sh 100

# Sweep all steps (online mode)
./run_scaling_sweep_online.sh [max_samples]
./run_scaling_sweep_online.sh 100
```

## Output Structure

```
scaling_simulation_results/           # Offline results
├── step_1/results.json
├── step_5/results.json
├── ...
├── step_100/results.json
└── sweep_report.json                 # Consolidated report

scaling_simulation_results_online/    # Online results
├── step_1/results.json
├── ...
└── sweep_report.json
```

### Result JSON Format

```json
{
  "config": { "k_values": [1, 2, 4, 8, 16, 32] },
  "steps": {
    "1": {
      "results_by_k": {
        "1": { "acceptance_rate": 0.25, "hit_rate": 0.75, "tokens_per_step": 1.39 },
        "32": { "acceptance_rate": 0.30, "hit_rate": 0.93, "tokens_per_step": 1.97 }
      }
    }
  },
  "summary_by_k": {
    "1": { "mean_acceptance_rate": 0.25, "mean_tokens_per_step": 1.39 },
    "32": { "mean_acceptance_rate": 0.30, "mean_tokens_per_step": 1.97 }
  }
}
```

## Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| `tokens_per_step` | Output tokens per model forward pass | > 1.5 |
| `acceptance_rate` | Fraction of drafted tokens accepted | > 25% |
| `hit_rate` | Fraction of steps with cache hits | > 80% |
| `draft_contribution` | Fraction of tokens from cache (free) | > 40% |

## Experimental Results Summary

### Offline Mode (No Online Updates)

| k | Tokens/Step | Acceptance Rate | Hit Rate |
|---|-------------|-----------------|----------|
| 1 | 1.39× | 25.1% | 75.6% |
| 32 | 1.97× | 30.4% | 93.0% |

### Online Mode (With Online Updates)

| k | Tokens/Step | Acceptance Rate | Hit Rate |
|---|-------------|-----------------|----------|
| 1 | 2.20× | 39.6% | 92.7% |
| 32 | 2.44× | 39.1% | 96.4% |

### Key Findings

1. **Online updates provide ~1.6× speedup at k=1**, reducing to ~1.2× at k=32
2. **Online k=1 (2.20×) outperforms Offline k=32 (1.97×)**
3. **Diminishing returns**: Most gains from k=1→8; k=16→32 adds ~6%
4. **Later training steps (20+)** show better acceptance rates

## Related Documentation

- [SPECULATION_METRICS.md](./SPECULATION_METRICS.md) - Detailed metric definitions
- [CLAUDE.md](../../../../CLAUDE.md) - Project overview and SRT architecture
