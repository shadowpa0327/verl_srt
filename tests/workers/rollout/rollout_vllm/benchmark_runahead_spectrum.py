#!/usr/bin/env python3
"""
Benchmark script to compare runahead vs no-runahead across different workloads.
Shows the spectrum of performance improvements.
"""

import asyncio
import os
import subprocess
import sys
import json
from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class BenchmarkConfig:
    name: str
    primary_size: int
    runahead_size: int
    max_tokens: int
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    total_time: float
    primary_completed: int
    runahead_completed: int
    runahead_aborted: int
    total_tokens: int
    throughput: float  # tokens/second


def run_test(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark test and parse results."""
    env = os.environ.copy()
    env.update({
        "NUM_GPUS": "2",
        "MODEL_PATH": os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"),
        "PRIMARY_SIZE": str(config.primary_size),
        "RUNAHEAD_SIZE": str(config.runahead_size),
        "MAX_TOKENS": str(config.max_tokens),
        "WAITING_THRESHOLD": "4",
        "BUDGET_PER_SERVER": "2",
        "POLL_INTERVAL": "0.05",
    })

    cmd = [
        sys.executable,
        "tests/workers/rollout/rollout_vllm/test_vllm_run_ahead_slack_filling.py"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd="/home/ubuntu/verl_srt"
    )

    output = result.stdout + result.stderr

    # Parse results from output
    total_time = 0.0
    primary_completed = 0
    runahead_completed = 0
    runahead_aborted = 0

    # Extract total time
    time_match = re.search(r"Total time: ([\d.]+)s", output)
    if time_match:
        total_time = float(time_match.group(1))

    # Extract primary completed
    primary_match = re.search(r"Primary: (\d+)/(\d+) completed", output)
    if primary_match:
        primary_completed = int(primary_match.group(1))

    # Extract runahead stats
    runahead_match = re.search(r"Runahead: (\d+) completed, (\d+) aborted", output)
    if runahead_match:
        runahead_completed = int(runahead_match.group(1))
        runahead_aborted = int(runahead_match.group(2))

    total_tokens = (primary_completed + runahead_completed) * config.max_tokens
    throughput = total_tokens / total_time if total_time > 0 else 0

    return BenchmarkResult(
        config=config,
        total_time=total_time,
        primary_completed=primary_completed,
        runahead_completed=runahead_completed,
        runahead_aborted=runahead_aborted,
        total_tokens=total_tokens,
        throughput=throughput
    )


def print_comparison_table(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print("RUNAHEAD PERFORMANCE SPECTRUM")
    print("=" * 120)

    print(f"\n{'Workload':<30} | {'No Runahead':<25} | {'With Runahead':<30} | {'Improvement':<20}")
    print(f"{'':<30} | {'Time / Tokens / Tput':<25} | {'Time / Tokens / Tput':<30} | {'Throughput':<20}")
    print("-" * 120)

    for no_ra, with_ra in results:
        config = no_ra.config
        workload_name = f"{config.primary_size}p/{config.runahead_size}r @ {config.max_tokens}tok"

        no_ra_stats = f"{no_ra.total_time:.1f}s / {no_ra.total_tokens:,} / {no_ra.throughput:.0f}"
        with_ra_stats = f"{with_ra.total_time:.1f}s / {with_ra.total_tokens:,} / {with_ra.throughput:.0f}"

        if no_ra.throughput > 0:
            improvement = ((with_ra.throughput - no_ra.throughput) / no_ra.throughput) * 100
            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        else:
            improvement_str = "N/A"

        print(f"{workload_name:<30} | {no_ra_stats:<25} | {with_ra_stats:<30} | {improvement_str:<20}")

    print("=" * 120)


def print_detailed_results(results: List[Tuple[BenchmarkResult, BenchmarkResult]]):
    """Print detailed results for each configuration."""
    print("\n" + "=" * 120)
    print("DETAILED RESULTS")
    print("=" * 120)

    for i, (no_ra, with_ra) in enumerate(results):
        config = no_ra.config
        print(f"\n--- {config.name}: {config.description} ---")
        print(f"Configuration: {config.primary_size} primary, {config.runahead_size} runahead, {config.max_tokens} max tokens")
        print()

        print(f"  {'Metric':<25} | {'No Runahead':<20} | {'With Runahead':<20} | {'Delta':<15}")
        print(f"  {'-'*85}")

        # Time
        time_delta = with_ra.total_time - no_ra.total_time
        time_pct = (time_delta / no_ra.total_time * 100) if no_ra.total_time > 0 else 0
        print(f"  {'Total Time':<25} | {no_ra.total_time:<20.2f} | {with_ra.total_time:<20.2f} | {time_delta:+.2f}s ({time_pct:+.1f}%)")

        # Primary completed
        print(f"  {'Primary Completed':<25} | {no_ra.primary_completed:<20} | {with_ra.primary_completed:<20} | {with_ra.primary_completed - no_ra.primary_completed:+d}")

        # Runahead completed
        print(f"  {'Runahead Completed':<25} | {no_ra.runahead_completed:<20} | {with_ra.runahead_completed:<20} | {with_ra.runahead_completed - no_ra.runahead_completed:+d}")

        # Runahead aborted
        print(f"  {'Runahead Aborted':<25} | {no_ra.runahead_aborted:<20} | {with_ra.runahead_aborted:<20} | ")

        # Total tokens
        token_delta = with_ra.total_tokens - no_ra.total_tokens
        token_pct = (token_delta / no_ra.total_tokens * 100) if no_ra.total_tokens > 0 else 0
        print(f"  {'Total Tokens':<25} | {no_ra.total_tokens:<20,} | {with_ra.total_tokens:<20,} | {token_delta:+,} ({token_pct:+.1f}%)")

        # Throughput
        tput_delta = with_ra.throughput - no_ra.throughput
        tput_pct = (tput_delta / no_ra.throughput * 100) if no_ra.throughput > 0 else 0
        print(f"  {'Throughput (tok/s)':<25} | {no_ra.throughput:<20,.0f} | {with_ra.throughput:<20,.0f} | {tput_delta:+,.0f} ({tput_pct:+.1f}%)")

        # Effective speedup for combined workload
        if with_ra.runahead_completed > 0:
            # Time to do primary + runahead sequentially (estimate)
            sequential_time = no_ra.total_time + (no_ra.total_time / no_ra.primary_completed * with_ra.runahead_completed)
            speedup = sequential_time / with_ra.total_time
            print(f"  {'Effective Speedup':<25} | {'(sequential)':<20} | {speedup:<20.2f}x | (vs doing {with_ra.runahead_completed} extra sequentially)")


def main():
    print("=" * 120)
    print("RUNAHEAD ROLLOUT BENCHMARK")
    print("=" * 120)
    print(f"Model: {os.environ.get('MODEL_PATH', 'Qwen/Qwen3-8B')}")
    print(f"GPUs: 2 (TP=1, DP=2)")
    print("=" * 120)

    # Define benchmark configurations - spectrum of workloads
    configs = [
        # Small batches, short generation - runahead has less opportunity
        BenchmarkConfig("small_short", 4, 2, 512, "Small batch, short generation"),

        # Small batches, medium generation
        BenchmarkConfig("small_medium", 4, 2, 1024, "Small batch, medium generation"),

        # Medium batches, medium generation
        BenchmarkConfig("medium_medium", 6, 3, 1024, "Medium batch, medium generation"),

        # Medium batches, long generation - more opportunity for runahead
        BenchmarkConfig("medium_long", 6, 3, 2048, "Medium batch, long generation"),

        # Large batches, long generation
        BenchmarkConfig("large_long", 8, 4, 2048, "Large batch, long generation"),

        # Large batches, very long generation - maximum runahead opportunity
        BenchmarkConfig("large_vlong", 8, 4, 4096, "Large batch, very long generation"),
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Running: {config.name} - {config.description}")
        print(f"  Primary: {config.primary_size}, Runahead: {config.runahead_size}, Max Tokens: {config.max_tokens}")
        print("="*80)

        # Run without runahead
        print(f"\n  [1/2] Running WITHOUT runahead...")
        no_runahead_config = BenchmarkConfig(
            config.name, config.primary_size, 0, config.max_tokens, config.description
        )
        no_ra_result = run_test(no_runahead_config)
        print(f"        Time: {no_ra_result.total_time:.2f}s, Tokens: {no_ra_result.total_tokens:,}, Throughput: {no_ra_result.throughput:.0f} tok/s")

        # Run with runahead
        print(f"\n  [2/2] Running WITH runahead...")
        with_ra_result = run_test(config)
        print(f"        Time: {with_ra_result.total_time:.2f}s, Tokens: {with_ra_result.total_tokens:,}, Throughput: {with_ra_result.throughput:.0f} tok/s")
        print(f"        Runahead: {with_ra_result.runahead_completed} completed, {with_ra_result.runahead_aborted} aborted")

        results.append((no_ra_result, with_ra_result))

    # Print summary tables
    print_comparison_table(results)
    print_detailed_results(results)

    # Print summary insights
    print("\n" + "=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)

    improvements = []
    for no_ra, with_ra in results:
        if no_ra.throughput > 0:
            imp = ((with_ra.throughput - no_ra.throughput) / no_ra.throughput) * 100
            improvements.append((with_ra.config.name, imp, with_ra.runahead_completed))

    if improvements:
        best = max(improvements, key=lambda x: x[1])
        worst = min(improvements, key=lambda x: x[1])
        avg = sum(x[1] for x in improvements) / len(improvements)

        print(f"\n1. Average throughput improvement: {avg:.1f}%")
        print(f"2. Best case: {best[0]} with {best[1]:.1f}% improvement ({best[2]} runahead completed)")
        print(f"3. Worst case: {worst[0]} with {worst[1]:.1f}% improvement ({worst[2]} runahead completed)")
        print(f"\n4. Runahead is most effective with:")
        print(f"   - Longer generation lengths (more time for runahead to complete)")
        print(f"   - Larger batches (more primary requests = more variation in completion times)")
        print(f"   - Sufficient runahead queue (to fill slack as it appears)")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
