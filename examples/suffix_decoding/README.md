# Suffix Decoding Examples

This directory contains examples demonstrating suffix tree speculative decoding with vLLM.

## Overview

Suffix decoding is a speculative decoding technique that uses historical response patterns to improve token prediction. These examples show how to:

1. Build suffix trees from collected responses
2. Load suffix tree snapshots into vLLM
3. Measure acceptance length improvement

## Prerequisites

```bash
# Install verl with vLLM support
pip install -e .[vllm]

# ArcticInference is required for suffix tree operations
pip install arctic-inference
```

## Examples

### 1. Simple Load Snapshot (`example_load_snapshot.py`)

A minimal example showing how to load a suffix tree snapshot into vLLM.

```bash
python example_load_snapshot.py
```

### 2. Full Speculation Workflow (`example_suffix_tree_speculation.py`)

Comprehensive example demonstrating the realistic verl workflow:
- Initialize vLLM with suffix decoding
- Collect baseline responses
- Build suffix trees from responses
- Load snapshot and measure improvement

```bash
# Basic run
python example_suffix_tree_speculation.py --model meta-llama/Llama-3.1-8B-Instruct

# Quick test with fewer samples
python example_suffix_tree_speculation.py --model meta-llama/Llama-3.1-8B-Instruct \
    --num-rounds 2 --num-prompts 5

# Show output responses
python example_suffix_tree_speculation.py --model meta-llama/Llama-3.1-8B-Instruct \
    --print-output
```

### 3. Hash Consistency Tests (`test_precomputed_hash.py`)

Tests that validate hash-based tree matching works correctly.

```bash
# CPU-only tests (no GPU required)
python test_precomputed_hash.py

# Include vLLM integration tests (requires GPU)
python test_precomputed_hash.py --with-vllm --model meta-llama/Llama-3.1-8B-Instruct
```

## Architecture

These examples use vLLM **monkey patches** instead of a forked vLLM. The patches are applied at runtime before importing vLLM:

```python
from verl.workers.rollout.vllm_rollout.patches import apply_all_patches
apply_all_patches()

from vllm import LLM, SamplingParams
# Now LLM has suffix decoding support
```

The patches add:
- `speculative_config` with `method="suffix"` support
- `LLM.load_snapshot()` method for loading suffix trees
- Hash-based tree mapping for efficient pattern reuse

## Key Concepts

### Hash-Based Tree Mapping

When building suffix trees, each unique prompt creates a tree. The prompt is hashed (using the last N tokens) to create a mapping:

```python
# Building trees
cache = ParallelSuffixDecodingCache(hash_token_count=128)
cache.start_request("req0", prompt_tokens)
cache.add_tokens("req0", response_tokens)

# Create snapshot with hash mapping
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

# Load into vLLM - requests with matching hashes reuse the same tree
llm.load_snapshot(snapshots, hash_mapping)
```

### Acceptance Length

Acceptance length measures how many speculated tokens are accepted on average:
- **Baseline**: ~1.0 (only the original token accepted)
- **With patterns**: 2.0+ (multiple speculated tokens accepted)

Higher acceptance length = faster inference.

## Benchmarks

The `benchmarks/` subdirectory contains profiling and comparison tools for evaluating suffix decoding performance.

### Proposer Profiling (`benchmarks/profile_suffix_decoding_proposer.py`)

Detailed latency breakdown of the propose() method, measuring individual operations:

```bash
# Basic profiling with latency breakdown
python benchmarks/profile_suffix_decoding_proposer.py

# With Chrome trace export (viewable at chrome://tracing/)
python benchmarks/profile_suffix_decoding_proposer.py --chrome-trace

# Batch size comparison
python benchmarks/profile_suffix_decoding_proposer.py --compare

# Custom configuration
python benchmarks/profile_suffix_decoding_proposer.py \
    --batch-size 64 \
    --num-trees 200 \
    --num-threads 8
```

### Sequential vs Parallel Comparison (`benchmarks/compare_proposer_implementations.py`)

Compares latency between:
- **Sequential**: SuffixDecodingCache (dual-tree architecture)
- **Parallel**: ParallelSuffixDecodingCache (forest architecture with OpenMP)

```bash
# Default comparison across batch sizes
python benchmarks/compare_proposer_implementations.py

# Custom configuration
python benchmarks/compare_proposer_implementations.py \
    --num-trees 200 \
    --num-threads 8 \
    --iterations 100
```

Output includes:
- Per-batch latency comparison
- Speedup ratios
- Crossover point (where parallel becomes faster)
- Per-request latency breakdown
