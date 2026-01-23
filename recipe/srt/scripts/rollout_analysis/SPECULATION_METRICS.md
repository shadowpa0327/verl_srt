# Speculation Metrics Documentation

This document defines the metrics used to evaluate speculative decoding performance in SRT (Speculative Rollout with Tree-Structured Cache).

## Table of Contents

1. [Metric Definitions](#metric-definitions)
2. [Metric Relationships](#metric-relationships)
3. [Concrete Example](#concrete-example)
4. [Analysis Results](#analysis-results)
5. [Key Findings](#key-findings)
6. [Reproducing the Analysis](#reproducing-the-analysis)
7. [Glossary](#glossary)

---

## Metric Definitions

### Primary Metrics

#### 1. Tokens per Step (E2E Speedup)

```
tokens_per_step = total_output_tokens / total_decoding_steps
```

**Meaning**: The average number of tokens produced per model forward pass.

- Without speculation: `tokens_per_step = 1.0` (baseline)
- With speculation: `tokens_per_step > 1.0` indicates speedup
- **This is THE metric for end-to-end speedup**

**Example**: If `tokens_per_step = 2.5`, you get 2.5x speedup compared to standard autoregressive decoding.

---

#### 2. Hit Rate

```
hit_rate = steps_with_any_drafts / total_steps
```

**Meaning**: The fraction of decoding steps where the cache returned at least one draft token.

- High hit rate (>80%): Cache frequently has matching patterns
- Low hit rate (<50%): Cache often empty or no matching patterns

**Important**: Hit rate does NOT indicate quality of drafts, only availability.

---

#### 3. Acceptance Rate

```
acceptance_rate = total_accepted_tokens / total_speculated_tokens
```

**Meaning**: Of all draft tokens returned by the cache, what fraction matched the ground truth (or model's actual output)?

- High acceptance rate (>50%): Drafts are high quality
- Low acceptance rate (<30%): Drafts often don't match

**Note**: Only computed over steps where speculation occurred (denominator excludes cache misses).

---

#### 4. Draft Contribution

```
draft_contribution = total_accepted_tokens / total_output_tokens
```

**Meaning**: What fraction of the final output came "for free" from accepted draft tokens (vs. bonus tokens from model verification)?

- High draft contribution (>50%): Most output from speculation
- Low draft contribution (<30%): Most output from model forward passes

**This metric directly reflects compute savings.**

---

### Secondary Metrics

#### 5. Tokens per Hit Step (Ceiling)

```
tokens_per_hit_step = output_tokens_on_hit_steps / steps_with_drafts
```

**Meaning**: Average tokens produced per step, counting only steps where cache returned drafts.

**Use case**: Shows the "ceiling" performance if hit rate were 100%.

---

#### 6. Steps with Drafts

```
steps_with_drafts = count(steps where num_speculated_tokens > 0)
```

**Meaning**: Absolute count of steps where speculation occurred.

---

## Metric Relationships

### The Key Formula

```
tokens_per_step = 1 + hit_rate × acceptance_rate × avg_drafts_when_hit
```

Or approximately:

```
draft_contribution ≈ (tokens_per_step - 1) / tokens_per_step
```

### Why Hit Rate Can Be High But Draft Contribution Low

| Scenario | Hit Rate | Accept Rate | Draft Contrib | Explanation |
|----------|----------|-------------|---------------|-------------|
| Cache returns junk | 90% | 10% | 15% | Drafts available but wrong |
| Cache returns gold | 90% | 80% | 70% | Drafts available and correct |
| Cache often empty | 30% | 80% | 25% | Good drafts but rare |

**Bottom line**:
- **Hit rate** tells you: "Does cache have patterns?"
- **Draft contribution** tells you: "Are those patterns useful?"

---

## Concrete Example

### Single Decoding Step

```
Ground truth next tokens: [the, answer, is, 42, .]

Step execution:
├── Cache returns drafts: [the, answer, was, wrong]  (4 tokens)
├── Verification against ground truth:
│   ├── "the"    ✓ accepted (matches)
│   ├── "answer" ✓ accepted (matches)
│   ├── "was"    ✗ rejected (ground truth is "is")
│   └── "wrong"  (not checked, stopped at first mismatch)
├── Bonus token: "is" (from model forward pass)
└── Output this step: [the, answer, is] = 3 tokens

Metrics for this step:
- Speculated tokens: 4
- Accepted tokens: 2
- Output tokens: 3 (2 accepted + 1 bonus)
- Hit: YES
```

### Full Response Example (10 steps)

| Step | Drafts | Accepted | Bonus | Output | Hit? |
|------|--------|----------|-------|--------|------|
| 1 | 4 | 2 | 1 | 3 | Yes |
| 2 | 0 | 0 | 1 | 1 | No |
| 3 | 3 | 1 | 1 | 2 | Yes |
| 4 | 5 | 0 | 1 | 1 | Yes |
| 5 | 0 | 0 | 1 | 1 | No |
| 6 | 2 | 2 | 1 | 3 | Yes |
| 7 | 6 | 3 | 1 | 4 | Yes |
| 8 | 0 | 0 | 1 | 1 | No |
| 9 | 4 | 1 | 1 | 2 | Yes |
| 10 | 3 | 2 | 1 | 3 | Yes |
| **Total** | **27** | **11** | **10** | **21** | **7** |

### Metric Calculations

```
Hit Rate         = 7 / 10 = 70%
Acceptance Rate  = 11 / 27 = 40.7%
Draft Contrib    = 11 / 21 = 52.4%
Tokens per Step  = 21 / 10 = 2.1x
```

### Interpretation

- **70% hit rate**: Cache returned drafts in 7 of 10 steps
- **40.7% acceptance**: Less than half of suggested drafts were correct
- **52.4% draft contribution**: Over half of output came free from cache
- **2.1x speedup**: Produced 21 tokens in only 10 forward passes

---

## Analysis Results

### Experimental Setup

- **Model**: Qwen2.5-7B
- **Dataset**: DAPO-Qwen2.5-7b-MATH-SRT-Runahead rollout data
- **Training ticks analyzed**: 1-42 (step=10)
- **Focus**: Long sequences (≥4K tokens)

### Three Modes Compared

1. **Prefill Only**: Cache pre-populated from secondary outputs, no online updates
2. **Online Only**: Cache starts empty, only online updates during generation
3. **Prefill + Online**: Both prefill and online updates (best of both)

### Results Summary (Long Sequences ≥4K tokens)

| Mode | E2E Speedup | Hit Rate | Accept Rate | Draft Contrib |
|------|-------------|----------|-------------|---------------|
| Prefill Only | 1.77x | 83.6% | 30.9% | 41.4% |
| Online Only | 2.36x | 77.7% | 51.9% | 54.5% |
| Prefill + Online | 2.66x | 88.6% | 49.5% | 59.5% |

### Speedup Decomposition

| Component | Speedup Gain |
|-----------|--------------|
| No speculation (baseline) | 1.00x |
| Prefill contribution | +0.77x |
| Online contribution | +1.36x |
| Combined | +1.66x |

**Online updates alone provide ~1.8x more speedup gain than prefill alone!**

---

## Key Findings

### 1. Online Updates Are More Valuable Than Prefill for Long Sequences

For sequences ≥4K tokens:
- **Prefill only**: 1.77x speedup
- **Online only**: 2.36x speedup (+33% better)
- **Combined**: 2.66x speedup

### 2. Acceptance Rate Is the Key Differentiator

| Mode | Hit Rate | Accept Rate |
|------|----------|-------------|
| Prefill Only | 84% (higher) | 31% (lower) |
| Online Only | 78% (lower) | 52% (higher) |

Prefill has higher hit rate but lower acceptance because:
- Cached patterns are from a **previous training epoch**
- Model behavior has changed, so old patterns don't match well

Online has lower hit rate but higher acceptance because:
- Patterns come from the **same response being generated**
- Self-consistency within a response is high

### 3. Long Sequences Benefit Most from Online Updates

Without online updates, long sequences suffer from "pattern staleness":
- Early tokens use fresh cached patterns (good acceptance)
- Later tokens go beyond cached patterns (poor acceptance)
- Draft contribution degrades as response length increases

With online updates:
- Each generated token is added to cache
- Later tokens can match patterns from earlier in the same response
- "Self-bootstrapping" effect maintains high acceptance throughout

### 4. Prioritization for E2E Speedup

To maximize `tokens_per_step` (the actual speedup):

1. **Focus on acceptance rate** over hit rate
2. **Enable online updates** especially for long sequences
3. **Combine prefill + online** for best results
4. **Monitor draft contribution** as the key quality metric

---

## Reproducing the Analysis

This section provides complete instructions for reproducing all analysis figures.

### Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Required data location
DATA_DIR="/home/ubuntu/verl_srt/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead"

# Verify data exists
ls $DATA_DIR/secondary/  # Should show tick directories (1, 2, 3, ...)
ls $DATA_DIR/rollout/    # Should show tick directories
```

### File Locations

```
recipe/srt/
├── replay_simulator.py                    # Core simulation script
└── scripts/rollout_analysis/
    ├── SPECULATION_METRICS.md             # This document
    ├── srt_analyze.py                     # Unified CLI tool
    ├── sweep_runner.py                    # Sweep execution logic
    ├── figure_generator.py                # Figure generation
    ├── analysis_config.py                 # Configuration classes
    ├── data_discovery.py                  # Data directory utilities
    ├── reproduce.sh                       # 1-click reproduction script
    │
    │   # Output files (generated in output directory):
    ├── per_request_data.csv               # Generated per-request data
    ├── sweep_summary.json                 # Aggregated results
    └── figures/                           # Generated figures directory
        ├── three_mode_comparison.png      # Fig 1: 4 metrics over training ticks
        ├── three_mode_bars.png            # Fig 2: Bar chart comparing 3 modes
        ├── speedup_decomposition.png      # Fig 3: Speedup contribution breakdown
        ├── draft_contribution_over_ticks.png  # Fig 4: Draft contrib + accept rate trends
        ├── hit_vs_acceptance_tradeoff.png # Fig 5: Scatter showing the trade-off
        ├── metrics_by_length.png          # Fig 6: All metrics by response length
        ├── long_seq_heatmap.png           # Fig 7: Heatmap (tick × length)
        └── online_update_insight.png      # Fig 8: Key insights summary
```

---

### Step 1: Run the Full Sweep (Generate Data)

This runs simulation across multiple training ticks with all three modes.

```bash
cd /home/ubuntu/verl_srt
source .venv/bin/activate

# Using the unified CLI tool
python -m recipe.srt.scripts.rollout_analysis.srt_analyze sweep \
    /home/ubuntu/verl_srt/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \
    -o ./sweep_results \
    --model Qwen/Qwen2.5-7B \
    --tick-start 1 \
    --tick-end 46 \
    --tick-step 10 \
    --min-token-prob 0.3 \
    --min-response-len 4000
```

**Parameters explained**:
- `--tick-start 1 --tick-end 46 --tick-step 10`: Analyze ticks 1, 11, 21, 31, 41
- `--min-response-len 4000`: Focus on long sequences in summary
- All three modes (prefill_only, online_only, prefill_plus_online) run by default

**Output**: Creates `per_request_data.csv` and `sweep_summary.json` in the output directory

---

### Step 2: Generate Three-Mode Comparison Figure

After running the sweep, generate the comparison visualizations using the CLI:

```bash
# Generate all figures from the per-request CSV
python -m recipe.srt.scripts.rollout_analysis.srt_analyze plot \
    --data ./sweep_results/per_request_data.csv \
    -o ./sweep_results/figures \
    --min-response-len 4000
```

Alternatively, for custom figures using Python directly:

```python
#!/usr/bin/env python3
"""
Generate three-mode comparison figures.
Run from project root: python this_script.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
OUTPUT_DIR = "./sweep_results"  # Your output directory from sweep
MIN_RESPONSE_LEN = 4000  # Focus on long sequences

# Load data
df = pd.read_csv(f"{OUTPUT_DIR}/per_request_data.csv")
df_4k = df[df['response_len'] >= MIN_RESPONSE_LEN].copy()

print(f"Loaded {len(df)} total records, {len(df_4k)} with response_len >= {MIN_RESPONSE_LEN}")
print(f"Modes: {df_4k['mode'].unique()}")

# Define mode colors and labels
mode_config = {
    'prefill_only': {'color': 'r', 'marker': 's', 'label': 'Prefill Only'},
    'online_only': {'color': 'g', 'marker': '^', 'label': 'Online Only'},
    'prefill_plus_online': {'color': 'b', 'marker': 'o', 'label': 'Prefill + Online'},
}

# =========================================================================
# Figure 1: E2E Speedup Metrics Over Training Ticks (3 modes)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ("tokens_per_step", "Tokens per Step (E2E Speedup)", axes[0, 0]),
    ("hit_rate", "Hit Rate", axes[0, 1]),
    ("tokens_per_hit_step", "Tokens per Hit Step (Ceiling)", axes[1, 0]),
    ("acceptance_rate", "Acceptance Rate", axes[1, 1]),
]

for metric, title, ax in metrics:
    for mode, cfg in mode_config.items():
        subset = df_4k[df_4k['mode'] == mode]
        if len(subset) > 0:
            grouped = subset.groupby('sim_tick')[metric].mean()
            ax.plot(grouped.index, grouped.values, f'{cfg["color"]}{cfg["marker"]}-',
                   label=cfg['label'], alpha=0.8, linewidth=2, markersize=8)

    ax.set_xlabel("Training Tick", fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(f"{title}\n(Long Sequences >= {MIN_RESPONSE_LEN})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/three_mode_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved three_mode_comparison.png")

# =========================================================================
# Figure 2: Bar chart comparing 3 modes
# =========================================================================
fig, ax = plt.subplots(figsize=(12, 6))

metrics_to_plot = ['hit_rate', 'acceptance_rate', 'tokens_per_step', 'tokens_per_hit_step']
metric_labels = ['Hit Rate', 'Accept Rate', 'Toks/Step\n(Speedup)', 'Toks/Hit Step\n(Ceiling)']
x = np.arange(len(metrics_to_plot))
width = 0.25

colors = {'prefill_only': 'red', 'online_only': 'green', 'prefill_plus_online': 'blue'}
labels = {'prefill_only': 'Prefill Only', 'online_only': 'Online Only', 'prefill_plus_online': 'Prefill + Online'}

for i, mode in enumerate(['prefill_only', 'online_only', 'prefill_plus_online']):
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        means = [subset[m].mean() for m in metrics_to_plot]
        bars = ax.bar(x + (i - 1) * width, means, width, label=labels[mode], color=colors[mode], alpha=0.8)

        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel("Value", fontsize=12)
ax.set_title(f"Speculation Quality Comparison: 3 Modes\n(Long Sequences >= {MIN_RESPONSE_LEN} tokens)", fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 3.2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/three_mode_bars.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved three_mode_bars.png")

# =========================================================================
# Figure 3: Speedup decomposition showing contribution of each component
# =========================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Get means for each mode
mode_means = {}
for mode in ['prefill_only', 'online_only', 'prefill_plus_online']:
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        mode_means[mode] = {
            'tps': subset['tokens_per_step'].mean(),
            'hr': subset['hit_rate'].mean(),
            'tphs': subset['tokens_per_hit_step'].mean(),
            'ar': subset['acceptance_rate'].mean(),
        }

# Create bar chart showing speedup decomposition
modes = ['No Speculation\n(Baseline)', 'Prefill Only', 'Online Only', 'Prefill + Online']
speedups = [1.0, mode_means['prefill_only']['tps'], mode_means['online_only']['tps'], mode_means['prefill_plus_online']['tps']]
bar_colors = ['gray', 'red', 'green', 'blue']

bars = ax.bar(modes, speedups, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add speedup labels
for bar, speedup in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
           f'{speedup:.2f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add horizontal lines for comparison
ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
ax.axhline(y=speedups[1], color='red', linestyle=':', alpha=0.5)
ax.axhline(y=speedups[2], color='green', linestyle=':', alpha=0.5)

ax.set_ylabel("E2E Speedup (tokens/step)", fontsize=12)
ax.set_title(f"E2E Speedup: Contribution of Prefill vs Online Updates\n(Long Sequences >= {MIN_RESPONSE_LEN} tokens)", fontsize=14)
ax.set_ylim(0, 3.2)
ax.grid(True, alpha=0.3, axis='y')

# Add gain annotations
gain_prefill = speedups[1] - 1.0
gain_online = speedups[2] - 1.0
gain_combined = speedups[3] - 1.0

ax.annotate(f'Prefill gain:\n+{gain_prefill:.2f}x', xy=(1, speedups[1]/2 + 0.5), fontsize=10, ha='center', color='darkred')
ax.annotate(f'Online gain:\n+{gain_online:.2f}x', xy=(2, speedups[2]/2 + 0.5), fontsize=10, ha='center', color='darkgreen')
ax.annotate(f'Combined gain:\n+{gain_combined:.2f}x', xy=(3, speedups[3]/2 + 0.5), fontsize=10, ha='center', color='darkblue')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/speedup_decomposition.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved speedup_decomposition.png")

# =========================================================================
# Print Summary Statistics
# =========================================================================
print("\n" + "="*70)
print("SUMMARY: Three Mode Comparison (4K+ sequences)")
print("="*70)
for mode, label in [('prefill_only', 'Prefill Only'), ('online_only', 'Online Only'), ('prefill_plus_online', 'Prefill + Online')]:
    if mode in mode_means:
        m = mode_means[mode]
        print(f"\n{label}:")
        print(f"  E2E Speedup:    {m['tps']:.2f}x")
        print(f"  Hit Rate:       {m['hr']:.1%}")
        print(f"  Ceiling:        {m['tphs']:.2f}x")
        print(f"  Accept Rate:    {m['ar']:.1%}")
```

Save this as `generate_figures.py` and run:
```bash
python generate_figures.py
```

---

### Step 3: Generate Draft Contribution & Trade-off Figures

```python
#!/usr/bin/env python3
"""
Generate draft contribution trend and hit vs acceptance trade-off figures.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./sweep_results"  # Your output directory from sweep
MIN_RESPONSE_LEN = 4000

df = pd.read_csv(f"{OUTPUT_DIR}/per_request_data.csv")
df_4k = df[df['response_len'] >= MIN_RESPONSE_LEN].copy()

mode_config = {
    'prefill_only': {'color': 'r', 'marker': 's', 'label': 'Prefill Only'},
    'online_only': {'color': 'g', 'marker': '^', 'label': 'Online Only'},
    'prefill_plus_online': {'color': 'b', 'marker': 'o', 'label': 'Prefill + Online'},
}

# Figure 4: Draft Contribution Over Training Ticks
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Draft contribution trend
ax = axes[0]
for mode, cfg in mode_config.items():
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        grouped = subset.groupby('sim_tick')['draft_contribution'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   fmt=f'{cfg["color"]}{cfg["marker"]}-', label=cfg['label'],
                   capsize=3, alpha=0.8, linewidth=2, markersize=8)

ax.set_xlabel("Training Tick", fontsize=11)
ax.set_ylabel("Draft Contribution", fontsize=11)
ax.set_title(f"Draft Contribution Over Training\n(Long Sequences >= {MIN_RESPONSE_LEN})", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)

# Right: Acceptance rate trend
ax = axes[1]
for mode, cfg in mode_config.items():
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        grouped = subset.groupby('sim_tick')['acceptance_rate'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   fmt=f'{cfg["color"]}{cfg["marker"]}-', label=cfg['label'],
                   capsize=3, alpha=0.8, linewidth=2, markersize=8)

ax.set_xlabel("Training Tick", fontsize=11)
ax.set_ylabel("Acceptance Rate", fontsize=11)
ax.set_title(f"Acceptance Rate Over Training\n(Higher = Better Draft Quality)", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/draft_contribution_over_ticks.png", dpi=150)
plt.close()
print("Saved draft_contribution_over_ticks.png")

# Figure 5: Hit Rate vs Acceptance Rate Trade-off (Scatter)
fig, ax = plt.subplots(figsize=(10, 8))

for mode, cfg in mode_config.items():
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        sample = subset.sample(min(200, len(subset)), random_state=42)
        ax.scatter(sample['hit_rate'], sample['acceptance_rate'],
                  c=cfg['color'], marker=cfg['marker'], label=cfg['label'],
                  alpha=0.5, s=50)

        # Add mean point with larger marker
        mean_hr = subset['hit_rate'].mean()
        mean_ar = subset['acceptance_rate'].mean()
        ax.scatter([mean_hr], [mean_ar], c=cfg['color'], marker=cfg['marker'],
                  s=300, edgecolors='black', linewidths=2, zorder=10)
        ax.annotate(f'{cfg["label"]}\n({mean_hr:.0%}, {mean_ar:.0%})',
                   xy=(mean_hr, mean_ar), xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlabel("Hit Rate (Cache Availability)", fontsize=12)
ax.set_ylabel("Acceptance Rate (Draft Quality)", fontsize=12)
ax.set_title(f"Hit Rate vs Acceptance Rate Trade-off\n(Long Sequences >= {MIN_RESPONSE_LEN})", fontsize=14)
ax.legend(loc='lower left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

# Add quadrant labels
ax.text(0.25, 0.85, "Low Hit, High Accept", ha='center', fontsize=9, alpha=0.7)
ax.text(0.75, 0.85, "High Hit, High Accept\n(BEST)", ha='center', fontsize=9, alpha=0.7,
       bbox=dict(facecolor='lightgreen', alpha=0.3))
ax.text(0.25, 0.15, "Low Hit, Low Accept\n(WORST)", ha='center', fontsize=9, alpha=0.7)
ax.text(0.75, 0.15, "High Hit, Low Accept\n(Stale patterns)", ha='center', fontsize=9, alpha=0.7)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/hit_vs_acceptance_tradeoff.png", dpi=150)
plt.close()
print("Saved hit_vs_acceptance_tradeoff.png")
```

---

### Step 4: Generate Metrics by Response Length Figure

```python
#!/usr/bin/env python3
"""
Generate metrics by response length figure.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./sweep_results"  # Your output directory from sweep
df = pd.read_csv(f"{OUTPUT_DIR}/per_request_data.csv")

# Define length bins with focus on long sequences
bins = [0, 500, 1000, 2000, 4000, 8000, 20000]
labels = ['0-500', '500-1K', '1K-2K', '2K-4K', '4K-8K', '8K+']
df['length_bucket'] = pd.cut(df['response_len'], bins=bins, labels=labels)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

mode_config = {
    'prefill_only': {'color': 'r', 'marker': 's', 'label': 'Prefill Only'},
    'online_only': {'color': 'g', 'marker': '^', 'label': 'Online Only'},
    'prefill_plus_online': {'color': 'b', 'marker': 'o', 'label': 'Prefill + Online'},
}

metrics_by_len = [
    ("tokens_per_step", "Tokens per Step (E2E Speedup)", axes[0, 0]),
    ("hit_rate", "Hit Rate", axes[0, 1]),
    ("tokens_per_hit_step", "Tokens per Hit Step", axes[1, 0]),
    ("acceptance_rate", "Acceptance Rate", axes[1, 1]),
]

for metric, title, ax in metrics_by_len:
    for mode, cfg in mode_config.items():
        subset = df[df['mode'] == mode]
        if len(subset) > 0:
            grouped = subset.groupby('length_bucket', observed=True)[metric]
            means = grouped.mean()
            stds = grouped.std()

            valid_labels = [l for l in labels if l in means.index]
            x = [labels.index(l) for l in valid_labels]
            y = [means[l] for l in valid_labels]
            yerr = [stds[l] for l in valid_labels]

            ax.errorbar(x, y, yerr=yerr, fmt=f'{cfg["color"]}{cfg["marker"]}-',
                       label=cfg['label'], capsize=3, alpha=0.8, linewidth=2)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Response Length", fontsize=11)
    ax.set_ylabel(title, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight 4K+ region
    ax.axvspan(4, len(labels), alpha=0.1, color='yellow')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_by_length.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved metrics_by_length.png")
```

---

### Step 5: Generate Long Sequence Heatmap

```python
#!/usr/bin/env python3
"""
Generate heatmap for long sequences showing speedup by tick and length.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./sweep_results"  # Your output directory from sweep
df = pd.read_csv(f"{OUTPUT_DIR}/per_request_data.csv")

# Filter to long sequences
df_long = df[df['response_len'] >= 4000].copy()

if len(df_long) < 50:
    print(f"Not enough long sequences: {len(df_long)}")
    exit()

# Create length categories
len_bins = [4000, 6000, 8000, 12000, 20000]
len_labels = ['4K-6K', '6K-8K', '8K-12K', '12K+']
df_long['len_cat'] = pd.cut(df_long['response_len'], bins=len_bins, labels=len_labels)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

mode_titles = [
    ('prefill_only', 'PREFILL ONLY'),
    ('online_only', 'ONLINE ONLY'),
    ('prefill_plus_online', 'PREFILL + ONLINE'),
]

for ax_idx, (mode, title) in enumerate(mode_titles):
    subset = df_long[df_long['mode'] == mode]
    if len(subset) > 0:
        pivot = subset.pivot_table(
            values='tokens_per_step',
            index='len_cat',
            columns='sim_tick',
            aggfunc='mean',
            observed=True
        )

        if pivot.size > 0:
            im = axes[ax_idx].imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=1.0, vmax=3.0)
            axes[ax_idx].set_xticks(np.arange(len(pivot.columns)))
            axes[ax_idx].set_xticklabels([f'{int(t)}' for t in pivot.columns], rotation=45)
            axes[ax_idx].set_yticks(np.arange(len(pivot.index)))
            axes[ax_idx].set_yticklabels(pivot.index)
            axes[ax_idx].set_xlabel("Training Tick", fontsize=12)
            axes[ax_idx].set_ylabel("Response Length", fontsize=12)
            axes[ax_idx].set_title(f"E2E Speedup: {title}\n(Long Sequences 4K+)", fontsize=12)

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val < 1.5 else 'black'
                        axes[ax_idx].text(j, i, f'{val:.2f}x', ha='center', va='center',
                                        color=text_color, fontsize=10, fontweight='bold')

            plt.colorbar(im, ax=axes[ax_idx], label='Tokens per Step (Speedup)')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/long_seq_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved long_seq_heatmap.png")
```

---

### Step 6: Generate Key Insights Summary Figure

```python
#!/usr/bin/env python3
"""
Generate visual summary of key insights from the analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./sweep_results"  # Your output directory from sweep
MIN_RESPONSE_LEN = 4000

df = pd.read_csv(f"{OUTPUT_DIR}/per_request_data.csv")
df_4k = df[df['response_len'] >= MIN_RESPONSE_LEN].copy()

# Calculate mode means
mode_means = {}
for mode in ['prefill_only', 'online_only', 'prefill_plus_online']:
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        mode_means[mode] = {
            'tps': subset['tokens_per_step'].mean(),
            'hr': subset['hit_rate'].mean(),
            'ar': subset['acceptance_rate'].mean(),
            'dc': subset['draft_contribution'].mean(),
        }

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: The trade-off summary
ax = axes[0]
modes_list = ['prefill_only', 'online_only', 'prefill_plus_online']
mode_labels_short = ['Prefill\nOnly', 'Online\nOnly', 'Prefill +\nOnline']

hit_rates = [mode_means.get(m, {}).get('hr', 0) for m in modes_list]
accept_rates = [mode_means.get(m, {}).get('ar', 0) for m in modes_list]

x = np.arange(len(modes_list))
width = 0.35

bars1 = ax.bar(x - width/2, hit_rates, width, label='Hit Rate', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, accept_rates, width, label='Accept Rate', color='coral', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(mode_labels_short)
ax.set_ylabel("Rate", fontsize=11)
ax.set_title("Hit Rate vs Acceptance Rate\n(The Trade-off)", fontsize=12)
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, hit_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.0%}', ha='center', fontsize=9)
for bar, val in zip(bars2, accept_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.0%}', ha='center', fontsize=9)

# Panel 2: Speedup and draft contribution
ax = axes[1]
speedups = [mode_means.get(m, {}).get('tps', 1) for m in modes_list]
draft_contribs = [mode_means.get(m, {}).get('dc', 0) for m in modes_list]

ax.bar(x - width/2, speedups, width, label='Tokens/Step (Speedup)', color='green', alpha=0.8)
ax.bar(x + width/2, draft_contribs, width, label='Draft Contribution', color='purple', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(mode_labels_short)
ax.set_ylabel("Value", fontsize=11)
ax.set_title("Speedup & Draft Contribution\n(What Matters for E2E)", fontsize=12)
ax.legend()
ax.set_ylim(0, 3.0)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Text summary of key insights
ax = axes[2]
ax.axis('off')

insight_text = f"""
KEY INSIGHTS

1. Online Updates > Prefill for Long Sequences
   • Prefill: {mode_means.get('prefill_only', {}).get('hr', 0):.0%} hit rate but {mode_means.get('prefill_only', {}).get('ar', 0):.0%} acceptance
   • Online:  {mode_means.get('online_only', {}).get('hr', 0):.0%} hit rate but {mode_means.get('online_only', {}).get('ar', 0):.0%} acceptance

2. Why Acceptance Rate Matters More
   • High hit rate + low acceptance = wasted speculation
   • Prefill patterns from previous epoch (stale)
   • Online patterns from same response (fresh)

3. Self-Bootstrapping Effect
   • Online updates add tokens as generated
   • Later tokens match earlier patterns
   • High acceptance maintained throughout

4. Recommendation
   • Always enable online updates for long seqs
   • Combined: {mode_means.get('prefill_plus_online', {}).get('tps', 1):.2f}x vs {mode_means.get('prefill_only', {}).get('tps', 1):.2f}x (prefill only)
"""

ax.text(0.05, 0.95, insight_text, transform=ax.transAxes, fontsize=11,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/online_update_insight.png", dpi=150)
plt.close()
print("Saved online_update_insight.png")
```

---

### Quick Single-Tick Analysis (Optional)

For quick testing without running the full sweep:

```bash
# Run single tick simulation with metrics output
python recipe/srt/replay_simulator.py \
    --model_path Qwen/Qwen2.5-7B \
    --data_dir /home/ubuntu/verl_srt/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \
    --mode parallel \
    --cache_tick 5 \
    --sim_tick 6 \
    --min_token_prob 0.3 \
    --online_update \
    --max_samples 100
```

**Output metrics**:
- `Mean hit rate`: Fraction of steps with drafts
- `Mean acceptance rate`: Draft quality
- `Mean tokens/step`: E2E speedup
- `Mean tokens/hit step`: Ceiling speedup
- `Mean draft contrib`: Fraction of output from drafts

---

### Complete Reproduction Script

A complete script `reproduce.sh` is available in the `rollout_analysis/` directory.

To run all analysis and generate all 8 figures:

```bash
cd /home/ubuntu/verl_srt
./recipe/srt/scripts/rollout_analysis/reproduce.sh [DATA_DIR] [OUTPUT_DIR]

# Or with defaults:
./recipe/srt/scripts/rollout_analysis/reproduce.sh
```

The script generates all 8 figures:
1. `three_mode_comparison.png` - 4 metrics over training ticks
2. `three_mode_bars.png` - Bar chart comparing 3 modes
3. `speedup_decomposition.png` - Speedup contribution breakdown
4. `draft_contribution_over_ticks.png` - Draft contrib + accept rate trends
5. `hit_vs_acceptance_tradeoff.png` - Scatter showing the trade-off
6. `metrics_by_length.png` - All metrics by response length
7. `long_seq_heatmap.png` - Heatmap (tick × length)
8. `online_update_insight.png` - Key insights summary

View the full script:
```bash
cat recipe/srt/scripts/rollout_analysis/reproduce.sh
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Speculation** | Using cached patterns to predict future tokens |
| **Draft tokens** | Tokens suggested by the cache |
| **Accepted tokens** | Draft tokens that matched ground truth |
| **Bonus token** | The token from model's verification step |
| **Prefill** | Pre-populating cache with previous rollout data |
| **Online update** | Adding generated tokens to cache during decoding |
| **Hit** | A step where cache returned ≥1 draft token |
| **Miss** | A step where cache returned 0 drafts |
| **E2E Speedup** | End-to-end speedup factor (tokens_per_step) |
| **Ceiling** | Maximum possible speedup if hit rate were 100% |

---

## SRT Analysis CLI Tool

A unified CLI tool `srt_analyze` is available for all analysis tasks. It provides auto-detection of data directories, simulation sweeps, figure generation, and report creation.

### Installation

The CLI is available as a Python module in the codebase:

```bash
cd /home/ubuntu/verl_srt
source .venv/bin/activate

# Run via module
python -m recipe.srt.scripts.rollout_analysis.srt_analyze --help

# Or run the script directly
python recipe/srt/scripts/rollout_analysis/srt_analyze.py --help
```

### Quick Start

```bash
# 1. Show info about a data directory (auto-detect ticks, structure)
python -m recipe.srt.scripts.rollout_analysis.srt_analyze info \
    /path/to/rollout_data

# 2. Run full analysis with auto-detection
python -m recipe.srt.scripts.rollout_analysis.srt_analyze full \
    /path/to/rollout_data -o ./results

# 3. Generate figures from existing CSV data (no re-run needed)
python -m recipe.srt.scripts.rollout_analysis.srt_analyze plot \
    --data ./results/per_request_data.csv -o ./figures
```

### Commands

#### `info` - Show Data Directory Info

```bash
# Basic info
srt_analyze info /path/to/rollout_data

# Detailed statistics including response length analysis
srt_analyze info /path/to/rollout_data --detailed

# Include sample data preview
srt_analyze info /path/to/rollout_data --detailed --preview
```

Output example:
```
Data Directory: /path/to/rollout_data
  Valid: True
  Rollout ticks: 66 files (1-66)
  Secondary ticks: 65 files
  Rollout tick list: [1, 2, 3]...[64, 65, 66]
  Valid tick pairs for simulation: 65
    Range: (1, 2) to (65, 66)
```

#### `sweep` - Run Simulation Sweep

```bash
# Auto-detect tick range, run all modes
srt_analyze sweep /path/to/rollout_data -o ./sweep_results

# Custom tick range
srt_analyze sweep /path/to/rollout_data \
    --tick-start 1 --tick-end 50 --tick-step 10

# Run only specific mode
srt_analyze sweep /path/to/rollout_data --online-only

# With custom parameters
srt_analyze sweep /path/to/rollout_data \
    --min-token-prob 0.5 \
    --hash-token-count 256 \
    --max-samples 500
```

#### `plot` - Generate Figures

```bash
# Generate all figures from existing CSV
srt_analyze plot --data ./results/per_request_data.csv -o ./figures

# Generate specific figures
srt_analyze plot --data ./results/per_request_data.csv \
    --figures "three_mode_comparison,speedup_decomposition"

# List available figures
srt_analyze plot --list --data ./results/per_request_data.csv

# Custom settings
srt_analyze plot --data ./results/per_request_data.csv \
    --min-response-len 2000 \
    --dpi 300 \
    --format pdf
```

Available figures:
1. `three_mode_comparison` - 4 metrics over training ticks
2. `three_mode_bars` - Bar chart comparing 3 modes
3. `speedup_decomposition` - Speedup contribution breakdown
4. `draft_contribution_over_ticks` - Draft contrib + accept rate trends
5. `hit_vs_acceptance_tradeoff` - Scatter showing the trade-off
6. `metrics_by_length` - All metrics by response length
7. `long_seq_heatmap` - Heatmap (tick x length)
8. `online_update_insight` - Key insights summary

#### `full` - Complete Analysis Pipeline

Runs sweep + plot + report generation in one command:

```bash
srt_analyze full /path/to/rollout_data -o ./analysis_results \
    --tick-step 10 \
    --min-response-len 4000
```

Output structure:
```
analysis_results/
├── sweep_summary.json       # Aggregated metrics by mode
├── per_request_data.csv     # Per-request detailed data
├── figures/
│   ├── three_mode_comparison.png
│   ├── three_mode_bars.png
│   ├── speedup_decomposition.png
│   ├── draft_contribution_over_ticks.png
│   ├── hit_vs_acceptance_tradeoff.png
│   ├── metrics_by_length.png
│   ├── long_seq_heatmap.png
│   └── online_update_insight.png
└── ANALYSIS_REPORT.md       # Summary report
```

#### `single` - Single Tick Simulation

For quick testing or debugging:

```bash
# Basic single tick simulation
srt_analyze single /path/to/rollout_data --cache-tick 5 --sim-tick 6

# With online updates enabled
srt_analyze single /path/to/rollout_data \
    --cache-tick 5 --sim-tick 6 \
    --online-update

# Online-only mode (no prefill)
srt_analyze single /path/to/rollout_data \
    --cache-tick 5 --sim-tick 6 \
    --online-update --skip-prefill

# Save result to file
srt_analyze single /path/to/rollout_data \
    --cache-tick 5 --sim-tick 6 \
    -o ./result.json
```

### Python API

For programmatic use:

```python
from recipe.srt.scripts.rollout_analysis import (
    discover_data_directory,
    SweepConfig,
    SweepRunner,
    FigureConfig,
    FigureGenerator,
)

# Discover data directory
info = discover_data_directory("/path/to/data")
print(info.summary())

# Run sweep
config = SweepConfig(
    data_dir=Path("/path/to/data"),
    output_dir=Path("./results"),
    tick_step=5,
)
runner = SweepRunner(config)
results = runner.run()
runner.save_results(results)

# Generate figures
fig_config = FigureConfig(
    data_csv=Path("./results/per_request_data.csv"),
    output_dir=Path("./figures"),
)
generator = FigureGenerator(fig_config)
generator.generate_all()
```

---

## Changelog

- **2025-01-21**: Added unified CLI tool `srt_analyze`
  - Auto-detection of data directories
  - Subcommands: info, sweep, plot, full, single
  - Python API for programmatic use
  - Complete analysis pipeline in one command

- **2025-01-21**: Expanded documentation with complete reproduction
  - Added reproduce_analysis.sh script generating all 8 figures
  - Added draft contribution trends and trade-off analysis
  - Added metrics by response length visualization
  - Added key insights summary figure

- **2025-01-21**: Initial documentation created
  - Metric definitions and relationships
  - Three-mode analysis (prefill only, online only, prefill + online)
  - Basic reproduction instructions
