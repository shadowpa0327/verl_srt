# Acceptance Length Measurement Skill Guide

## Purpose

Quick-reference skill for measuring and interpreting speculative decoding acceptance lengths in vLLM. Acceptance length is the primary metric for evaluating speculation quality.

---

## Source Documentation

| Document | Path | Description |
|----------|------|-------------|
| **Spec Decode Metrics** | `vllm/vllm/v1/spec_decode/metrics.py` | Metric collection and logging |
| **Metrics Reader** | `vllm/vllm/v1/metrics/reader.py` | Prometheus metrics API |
| **Example** | `vllm/examples/offline_inference/spec_decode.py` | Reference implementation |
| **Local Example** | `third_party/examples/example_suffix_tree_speculation.py` | Suffix tree example |

---

## Key Concepts

### What is Acceptance Length?

**Acceptance length** measures how many draft tokens are accepted by the target model before a rejection occurs.

```
Mean Acceptance Length = 1 + (num_accepted_tokens / num_drafts)
```

The "+1" accounts for the **bonus token**: even when all draft tokens are rejected, the target model still produces one token.

### Interpretation

| Mean Acceptance Length | Interpretation |
|------------------------|----------------|
| 1.0 | All drafts rejected (baseline performance) |
| 1.5 - 2.0 | Moderate speculation success |
| 2.0 - 3.0 | Good speculation accuracy |
| 3.0+ | Excellent pattern matching |

### Key Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:spec_decode_num_drafts` | Counter | Total speculation attempts |
| `vllm:spec_decode_num_draft_tokens` | Counter | Total tokens proposed |
| `vllm:spec_decode_num_accepted_tokens` | Counter | Total tokens accepted |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Vector | Acceptance count at each position |

---

## How to Measure Acceptance Length

### Method 1: Using llm.get_metrics() (Recommended)

```python
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

# Initialize vLLM with speculative decoding
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "suffix",  # or "eagle", "ngram", etc.
        "num_speculative_tokens": 5,
    },
    disable_log_stats=False,  # Required for metrics
)

# Run inference
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=256))

# Extract metrics
metrics = llm.get_metrics()

num_drafts = 0
num_draft_tokens = 0
num_accepted_tokens = 0
acceptance_counts = []

for metric in metrics:
    if metric.name == "vllm:spec_decode_num_drafts":
        assert isinstance(metric, Counter)
        num_drafts += metric.value
    elif metric.name == "vllm:spec_decode_num_draft_tokens":
        assert isinstance(metric, Counter)
        num_draft_tokens += metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens":
        assert isinstance(metric, Counter)
        num_accepted_tokens += metric.value
    elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
        assert isinstance(metric, Vector)
        if not acceptance_counts:
            acceptance_counts = [0] * len(metric.values)
        for pos in range(len(metric.values)):
            acceptance_counts[pos] += metric.values[pos]

# Calculate acceptance length
mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1

print(f"Mean acceptance length: {mean_acceptance_length:.2f}")
```

### Method 2: Per-Position Analysis

```python
# Per-position acceptance rates show where speculation fails
for i, count in enumerate(acceptance_counts):
    rate = count / num_drafts if num_drafts > 0 else 0
    print(f"Position {i}: {rate:.3f} ({count}/{num_drafts})")
```

**Typical pattern:**
- Position 0: High acceptance (0.8-0.95)
- Position 1: Moderate (0.5-0.8)
- Position 2+: Decreasing rapidly

### Method 3: Prometheus Queries (Production)

```promql
# Mean acceptance length over time window
1 + (
  rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
  rate(vllm:spec_decode_num_drafts_total[5m])
)

# Acceptance rate (accepted / proposed)
rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
rate(vllm:spec_decode_num_draft_tokens_total[5m])

# Per-position acceptance rate
rate(vllm:spec_decode_num_accepted_tokens_per_pos_total[5m]) /
rate(vllm:spec_decode_num_drafts_total[5m])
```

---

## Complete Helper Function

```python
def extract_acceptance_metrics(metrics: list) -> dict:
    """
    Extract speculative decoding metrics from vLLM metrics.

    Args:
        metrics: List of Metric objects from llm.get_metrics()

    Returns:
        Dictionary with:
        - num_drafts: Total speculation attempts
        - num_draft_tokens: Total tokens proposed
        - num_accepted_tokens: Total tokens accepted
        - mean_acceptance_length: 1 + (accepted/drafts)
        - per_position_acceptance_rates: List of rates per position
    """
    from vllm.v1.metrics.reader import Counter, Vector

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = []

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            if not acceptance_counts:
                acceptance_counts = [0] * len(metric.values)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    mean_acceptance_length = 1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0

    per_pos_rates = []
    if acceptance_counts and num_drafts > 0:
        per_pos_rates = [count / num_drafts for count in acceptance_counts]

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": mean_acceptance_length,
        "per_position_acceptance_rates": per_pos_rates,
    }
```

---

## Comparing Before/After Snapshot Loading

```python
from vllm import LLM, SamplingParams
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

# Phase 1: Baseline (no patterns)
llm = LLM(
    model="your-model",
    speculative_config={"method": "suffix", "num_speculative_tokens": 5},
    disable_log_stats=False,
)

outputs_baseline = llm.generate(prompts, sampling_params)
metrics_baseline = llm.get_metrics()
stats_baseline = extract_acceptance_metrics(metrics_baseline)
print(f"Baseline AL: {stats_baseline['mean_acceptance_length']:.3f}")

# Phase 2: Build patterns from historical data
cache = ParallelSuffixDecodingCache(max_tree_depth=64)
for i, (prompt, response) in enumerate(historical_data):
    req_id = f"train_{i}"
    cache.start_request(req_id, tokenize(prompt))
    cache.add_tokens(req_id, tokenize(response))

# Phase 3: Load snapshot
snapshots = cache.create_snapshot()
if snapshots:
    llm.load_snapshot(snapshots[0][1])

# Phase 4: Inference with patterns
outputs_enhanced = llm.generate(prompts, sampling_params)
metrics_enhanced = llm.get_metrics()
stats_enhanced = extract_acceptance_metrics(metrics_enhanced)
print(f"Enhanced AL: {stats_enhanced['mean_acceptance_length']:.3f}")

# Calculate improvement
improvement = (
    (stats_enhanced['mean_acceptance_length'] - stats_baseline['mean_acceptance_length']) /
    stats_baseline['mean_acceptance_length'] * 100
)
print(f"Improvement: {improvement:+.1f}%")
```

---

## Interpreting Per-Position Rates

```
Position 0: 0.85  ← First draft token accepted 85% of time
Position 1: 0.62  ← Second token if first accepted: 62%
Position 2: 0.41  ← Third token if second accepted: 41%
Position 3: 0.28  ← Fourth token if third accepted: 28%
Position 4: 0.15  ← Fifth token if fourth accepted: 15%
```

**Key insights:**
- Steep drops indicate poor pattern matching
- Flat or gradual drops indicate good patterns
- Position 0 rate ≈ overall speculation quality

---

## Configuration for Metrics

### vLLM Initialization

```python
llm = LLM(
    model="your-model",
    speculative_config={
        "method": "suffix",  # or "eagle", "ngram", "mtp"
        "num_speculative_tokens": 5,
    },
    disable_log_stats=False,  # REQUIRED for metrics
)
```

### Speculative Config Options

| Option | Description | Default |
|--------|-------------|---------|
| `method` | Proposer type | Required |
| `num_speculative_tokens` | Max tokens per draft | Required |
| `suffix_decoding_max_tree_depth` | Context window | 64 |
| `suffix_decoding_max_spec_factor` | Scale speculation | 1.0 |
| `suffix_decoding_min_token_prob` | Quality threshold | 0.1 |

---

## Troubleshooting

### "Metrics are not supported in the V0 engine"

**Cause:** Using V0 engine or metrics not enabled

**Solution:**
```python
# Ensure using V1 engine (default)
# Ensure disable_log_stats=False
llm = LLM(model="...", disable_log_stats=False)
```

### get_metrics() returns empty list

**Cause:** No inference run yet or stats disabled

**Solution:**
1. Run at least one `llm.generate()` call
2. Ensure `disable_log_stats=False`

### num_drafts is 0

**Cause:** Speculative decoding not active

**Possible reasons:**
- Batch size too large (`disable_by_batch_size` threshold exceeded)
- Request hit `max_model_len`
- Request uses unsupported sampling params

---

## File Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| SpecDecodingStats | `vllm/v1/spec_decode/metrics.py` | 17-43 |
| Mean AL formula | `vllm/v1/spec_decode/metrics.py` | 88-90 |
| Prometheus metrics | `vllm/v1/spec_decode/metrics.py` | 116-203 |
| Metrics reader | `vllm/v1/metrics/reader.py` | 66-135 |
| Vector type | `vllm/v1/metrics/reader.py` | 31-37 |
| Example usage | `vllm/examples/offline_inference/spec_decode.py` | 161-201 |

---

## Related Skills

- [vLLM V1 Architecture](vLLM_v1_architecture.md) - Speculative decoding integration
- [ArcticInference Cache Management](arctic_inference_cache_managements.md) - Suffix tree operations
- [Role of Third-Party Libraries](../role_of_third_party_lib.md) - Integration overview
