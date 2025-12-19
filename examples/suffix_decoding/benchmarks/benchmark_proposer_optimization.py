#!/usr/bin/env python3
"""
Benchmark script comparing Python loop vs zero-copy propose_from_batch implementations.

This script reproduces the performance comparison from GitHub issue #9,
showing the speedup achieved by eliminating Python loop overhead.

Usage:
    python benchmark_proposer_optimization.py
    python benchmark_proposer_optimization.py --batch-sizes 1,16,32,64
    python benchmark_proposer_optimization.py --detailed
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache


@dataclass
class LatencyBreakdown:
    """Latency breakdown for different phases."""
    setup_ms: float = 0.0
    batch_add_tokens_ms: float = 0.0
    batch_speculate_ms: float = 0.0
    total_ms: float = 0.0


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed_ms = 0.0
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


class PythonLoopProposer:
    """
    Python loop implementation using batch_speculate.

    This simulates the bottleneck identified in issue #9 where:
    1. Python loop iterates over sampled_token_ids
    2. Extracts context via numpy slice (COPY 1)
    3. Appends to list (COPY 2)
    4. Passes to batch_speculate (COPY 3 in C++ binding)
    """

    def __init__(self, cache: ParallelSuffixDecodingCache, max_tree_depth: int):
        self.cache = cache
        self.max_tree_depth = max_tree_depth

    def propose(
        self,
        token_ids_cpu: np.ndarray,
        num_tokens: np.ndarray,
        req_ids: List[str],
        sampled_token_ids: List[List[int]],
        max_spec_tokens: int = 5,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ) -> tuple[List[List[int]], LatencyBreakdown]:
        """Python loop implementation with batch_speculate."""
        breakdown = LatencyBreakdown()

        with Timer() as total_timer:
            # Phase 1: Setup (THE BOTTLENECK)
            with Timer() as setup_timer:
                req_ids_to_add_tokens = []
                tokens_to_add = []
                req_ids_to_speculate = []
                contexts_to_speculate = []

                for i, sampled_ids in enumerate(sampled_token_ids):
                    if not sampled_ids:
                        continue

                    req_id = req_ids[i]

                    # Collect tokens to add
                    req_ids_to_add_tokens.append(req_id)
                    tokens_to_add.append(sampled_ids)

                    # Extract context - THIS CREATES A COPY
                    num_tok = num_tokens[i]
                    start = max(0, num_tok - self.max_tree_depth)
                    pattern = token_ids_cpu[i, start:num_tok].copy()  # numpy slice = COPY

                    req_ids_to_speculate.append(req_id)
                    contexts_to_speculate.append(pattern)  # list append = COPY 2

            breakdown.setup_ms = setup_timer.elapsed_ms

            # Phase 2: Batch add tokens (fast - C++)
            with Timer() as add_timer:
                if req_ids_to_add_tokens:
                    self.cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)

            breakdown.batch_add_tokens_ms = add_timer.elapsed_ms

            # Phase 3: Batch speculate (fast - C++ with OpenMP)
            with Timer() as spec_timer:
                if req_ids_to_speculate:
                    drafts = self.cache.batch_speculate(
                        req_ids=req_ids_to_speculate,
                        contexts=contexts_to_speculate,
                        max_spec_tokens=max_spec_tokens,
                        max_spec_factor=max_spec_factor,
                        min_token_prob=min_token_prob,
                    )
                else:
                    drafts = []

            breakdown.batch_speculate_ms = spec_timer.elapsed_ms

        breakdown.total_ms = total_timer.elapsed_ms

        # Build result list
        draft_token_ids = []
        draft_idx = 0
        for sampled_ids in sampled_token_ids:
            if sampled_ids:
                draft_token_ids.append(drafts[draft_idx].token_ids)
                draft_idx += 1
            else:
                draft_token_ids.append([])

        return draft_token_ids, breakdown


class ZeroCopyProposer:
    """
    Zero-copy implementation using propose_from_batch.

    This eliminates Python loop overhead by:
    1. Passing arrays directly to C++
    2. Zero-copy context extraction via pointer arithmetic
    3. Parallelizing setup AND speculation in one OpenMP region
    """

    def __init__(self, cache: ParallelSuffixDecodingCache):
        self.cache = cache

    def propose(
        self,
        token_ids_cpu: np.ndarray,
        num_tokens: np.ndarray,
        tree_indices: np.ndarray,
        req_ids: List[str],
        sampled_token_ids: List[List[int]],
        max_spec_tokens: int = 5,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ) -> tuple[List[List[int]], LatencyBreakdown]:
        """Zero-copy propose_from_batch implementation."""
        breakdown = LatencyBreakdown()

        with Timer() as total_timer:
            # Phase 1: Minimal setup (just prepare indices)
            with Timer() as setup_timer:
                req_ids_to_add_tokens = []
                tokens_to_add = []
                batch_indices = []

                for i, sampled_ids in enumerate(sampled_token_ids):
                    if not sampled_ids:
                        continue

                    req_ids_to_add_tokens.append(req_ids[i])
                    tokens_to_add.append(sampled_ids)
                    batch_indices.append(i)

            breakdown.setup_ms = setup_timer.elapsed_ms

            # Phase 2: Batch add tokens (fast - C++)
            with Timer() as add_timer:
                if req_ids_to_add_tokens:
                    self.cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)

            breakdown.batch_add_tokens_ms = add_timer.elapsed_ms

            # Phase 3: Optimized speculation (zero-copy, parallelized)
            with Timer() as spec_timer:
                if batch_indices:
                    batch_indices_arr = np.array(batch_indices, dtype=np.int32)
                    drafts = self.cache.propose_from_batch(
                        token_ids_cpu=token_ids_cpu[batch_indices_arr],
                        num_tokens=num_tokens[batch_indices_arr],
                        tree_indices=tree_indices[batch_indices_arr],
                        max_spec_tokens=max_spec_tokens,
                        max_spec_factor=max_spec_factor,
                        min_token_prob=min_token_prob,
                    )
                else:
                    drafts = []

            breakdown.batch_speculate_ms = spec_timer.elapsed_ms

        breakdown.total_ms = total_timer.elapsed_ms

        # Build result list
        draft_token_ids = []
        draft_idx = 0
        for sampled_ids in sampled_token_ids:
            if sampled_ids:
                draft_token_ids.append(drafts[draft_idx].token_ids)
                draft_idx += 1
            else:
                draft_token_ids.append([])

        return draft_token_ids, breakdown


def create_test_batch(
    batch_size: int,
    max_seq_len: int = 256,
    max_tree_depth: int = 64,
    prompt_len: int = 50,
    history_steps: int = 10,
) -> tuple:
    """Create a realistic test batch with prompt and generation history."""
    cache = ParallelSuffixDecodingCache(
        max_tree_depth=max_tree_depth,
        num_threads=8,
        parallel_threshold=4,
    )

    token_ids_cpu = np.zeros((batch_size, max_seq_len), dtype=np.int32)
    num_tokens = np.zeros(batch_size, dtype=np.int32)
    tree_indices = np.zeros(batch_size, dtype=np.int32)
    req_ids = []
    sampled_token_ids = []

    for i in range(batch_size):
        req_id = f"req_{i}"
        req_ids.append(req_id)

        # Create prompt
        prompt = np.arange(1, prompt_len + 1, dtype=np.int32) + i * 10000
        cache.start_request(req_id, prompt)

        # Add generation history
        all_tokens = prompt.tolist()
        for j in range(history_steps):
            tokens = [prompt_len + j * 10 + k + i * 10000 for k in range(5)]
            cache.add_tokens(req_id, np.array(tokens, dtype=np.int32))
            all_tokens.extend(tokens)

        # Populate batch arrays
        token_ids_cpu[i, :len(all_tokens)] = all_tokens
        num_tokens[i] = len(all_tokens)
        tree_indices[i] = cache.get_tree_idx_for_request(req_id)

        # Simulate newly sampled token
        sampled_token_ids.append([len(all_tokens) + 100 + i * 10000])

    return cache, token_ids_cpu, num_tokens, tree_indices, req_ids, sampled_token_ids


def benchmark_batch_size(
    batch_size: int,
    num_warmup: int = 5,
    num_runs: int = 50,
    max_tree_depth: int = 64,
) -> tuple:
    """Benchmark both implementations for a given batch size."""
    # Create test batch
    cache, token_ids_cpu, num_tokens, tree_indices, req_ids, sampled_token_ids = \
        create_test_batch(batch_size, max_tree_depth=max_tree_depth)

    python_loop_proposer = PythonLoopProposer(cache, max_tree_depth)
    zero_copy_proposer = ZeroCopyProposer(cache)

    # Warmup
    for _ in range(num_warmup):
        python_loop_proposer.propose(token_ids_cpu, num_tokens, req_ids, sampled_token_ids)
        zero_copy_proposer.propose(token_ids_cpu, num_tokens, tree_indices, req_ids, sampled_token_ids)

    # Benchmark old implementation
    python_loop_breakdowns = []
    for _ in range(num_runs):
        _, breakdown = python_loop_proposer.propose(
            token_ids_cpu, num_tokens, req_ids, sampled_token_ids
        )
        python_loop_breakdowns.append(breakdown)

    # Benchmark new implementation
    zero_copy_breakdowns = []
    for _ in range(num_runs):
        _, breakdown = zero_copy_proposer.propose(
            token_ids_cpu, num_tokens, tree_indices, req_ids, sampled_token_ids
        )
        zero_copy_breakdowns.append(breakdown)

    # Calculate averages
    python_loop_avg = LatencyBreakdown(
        setup_ms=np.mean([b.setup_ms for b in python_loop_breakdowns]),
        batch_add_tokens_ms=np.mean([b.batch_add_tokens_ms for b in python_loop_breakdowns]),
        batch_speculate_ms=np.mean([b.batch_speculate_ms for b in python_loop_breakdowns]),
        total_ms=np.mean([b.total_ms for b in python_loop_breakdowns]),
    )

    zero_copy_avg = LatencyBreakdown(
        setup_ms=np.mean([b.setup_ms for b in zero_copy_breakdowns]),
        batch_add_tokens_ms=np.mean([b.batch_add_tokens_ms for b in zero_copy_breakdowns]),
        batch_speculate_ms=np.mean([b.batch_speculate_ms for b in zero_copy_breakdowns]),
        total_ms=np.mean([b.total_ms for b in zero_copy_breakdowns]),
    )

    return python_loop_avg, zero_copy_avg


def print_comparison_table(results: List[tuple], detailed: bool = False):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS: Python Loop vs Zero-Copy propose_from_batch")
    print("=" * 100)

    if detailed:
        print(f"\n{'Batch':>6} | {'Phase':>20} | {'Python Loop (ms)':>18} | {'Zero-Copy (ms)':>16} | {'Speedup':>8}")
        print("-" * 100)

        for batch_size, python_loop_avg, zero_copy_avg in results:
            # Setup phase
            setup_speedup = python_loop_avg.setup_ms / zero_copy_avg.setup_ms if zero_copy_avg.setup_ms > 0 else 0
            print(f"{batch_size:>6} | {'Setup':>20} | {python_loop_avg.setup_ms:>10.4f} | {zero_copy_avg.setup_ms:>10.4f} | {setup_speedup:>7.2f}x")

            # Add tokens phase
            add_speedup = python_loop_avg.batch_add_tokens_ms / zero_copy_avg.batch_add_tokens_ms if zero_copy_avg.batch_add_tokens_ms > 0 else 1.0
            print(f"{'':>6} | {'Add Tokens':>20} | {python_loop_avg.batch_add_tokens_ms:>10.4f} | {zero_copy_avg.batch_add_tokens_ms:>10.4f} | {add_speedup:>7.2f}x")

            # Speculate phase
            spec_speedup = python_loop_avg.batch_speculate_ms / zero_copy_avg.batch_speculate_ms if zero_copy_avg.batch_speculate_ms > 0 else 0
            print(f"{'':>6} | {'Speculate':>20} | {python_loop_avg.batch_speculate_ms:>10.4f} | {zero_copy_avg.batch_speculate_ms:>10.4f} | {spec_speedup:>7.2f}x")

            # Total
            total_speedup = python_loop_avg.total_ms / zero_copy_avg.total_ms if zero_copy_avg.total_ms > 0 else 0
            status = "✓" if total_speedup > 1.0 else " "
            print(f"{'':>6} | {'TOTAL':>20} | {python_loop_avg.total_ms:>10.4f} | {zero_copy_avg.total_ms:>10.4f} | {total_speedup:>7.2f}x {status}")
            print("-" * 100)
    else:
        print(f"\n{'Batch':>6} | {'Python Loop (ms)':>18} | {'Zero-Copy (ms)':>16} | {'Speedup':>8} | {'Setup %':>10}")
        print("-" * 100)

        for batch_size, python_loop_avg, zero_copy_avg in results:
            total_speedup = python_loop_avg.total_ms / zero_copy_avg.total_ms if zero_copy_avg.total_ms > 0 else 0
            setup_pct = (python_loop_avg.setup_ms / python_loop_avg.total_ms * 100) if python_loop_avg.total_ms > 0 else 0
            status = "✓" if total_speedup > 1.0 else " "
            print(f"{batch_size:>6} | {python_loop_avg.total_ms:>15.4f} | {zero_copy_avg.total_ms:>15.4f} | {total_speedup:>7.2f}x {status} | {setup_pct:>13.1f}%")


def visualize_setup_overhead(results: List[tuple]):
    """Print ASCII bar chart showing setup overhead growth."""
    print("\n" + "=" * 100)
    print("SETUP OVERHEAD VISUALIZATION (Python Loop Implementation)")
    print("=" * 100)
    print("\nSetup % of Total Time:")

    for batch_size, python_loop_avg, _ in results:
        setup_pct = (python_loop_avg.setup_ms / python_loop_avg.total_ms * 100) if python_loop_avg.total_ms > 0 else 0
        bar_len = int(setup_pct / 2)  # Scale for display
        bar = "█" * bar_len
        print(f"  batch={batch_size:<3}:  {setup_pct:>5.1f}%  {bar}")

    print("\nKey Insight: Setup overhead grows linearly with batch size in Python loop")
    print("             while zero-copy implementation has minimal setup overhead")


def main():
    parser = argparse.ArgumentParser(description="Benchmark propose_from_batch optimization")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32,64",
        help="Comma-separated batch sizes to test (default: 1,4,8,16,32,64)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=50,
        help="Number of benchmark runs per batch size (default: 50)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed per-phase breakdown"
    )
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=64,
        help="Maximum tree depth (default: 64)"
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print(f"\nRunning benchmarks with:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Runs per size: {args.num_runs}")
    print(f"  Max tree depth: {args.max_tree_depth}")
    print(f"  Threads: 8, Threshold: 4")

    # Run benchmarks
    results = []
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}...", end=" ", flush=True)
        python_loop_avg, zero_copy_avg = benchmark_batch_size(
            batch_size,
            num_runs=args.num_runs,
            max_tree_depth=args.max_tree_depth,
        )
        results.append((batch_size, python_loop_avg, zero_copy_avg))
        print("done")

    # Print results
    print_comparison_table(results, detailed=args.detailed)
    visualize_setup_overhead(results)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Find best speedup
    best_batch, best_old, best_new = max(results, key=lambda x: x[1].total_ms / x[2].total_ms)
    best_speedup = best_old.total_ms / best_new.total_ms

    print(f"\nBest speedup: {best_speedup:.2f}x at batch_size={best_batch}")
    print(f"  Python loop: {best_old.total_ms:.4f}ms")
    print(f"  Zero-copy:   {best_new.total_ms:.4f}ms")

    print("\nWhy the speedup?")
    print("  1. Zero-copy context extraction (direct pointer arithmetic)")
    print("  2. Eliminates 3 copies: numpy slice → list append → C++ binding")
    print("  3. Parallelizes setup AND speculation in one OpenMP region")
    print("  4. Minimal Python overhead (just array indexing)")

    print("\nExpected speedup (from issue #9):")
    print("  - batch=32: ~3.0x")
    print("  - batch=64: ~3.5x")
    print("\nNote: Actual speedup depends on CPU cores, tree depth, and context length")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
