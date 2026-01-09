#!/usr/bin/env python3
"""
Benchmark Results Analyzer

Analyzes and compares baseline vs runahead benchmark results from JSON files.
Shows extra time overhead and runahead token gains for different configurations.

Usage:
    python scripts/analyze_benchmark_results.py <results_directory>
    python scripts/analyze_benchmark_results.py results/benchmark_sweep_20260108_074917
    python scripts/analyze_benchmark_results.py results/benchmark_sweep_20260108_074917 --csv output.csv
    python scripts/analyze_benchmark_results.py results/benchmark_sweep_20260108_074917 --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunMetrics:
    """Parsed metrics from a benchmark run."""
    time_seconds: float = 0.0
    primary_tokens: int = 0
    primary_completed: int = 0
    runahead_tokens_total: int = 0
    runahead_tokens_completed: int = 0
    runahead_tokens_aborted: int = 0
    runahead_completed_count: int = 0
    runahead_aborted_count: int = 0
    runahead_rejected_count: int = 0

    # Multi-round stats (optional)
    time_mean: float = 0.0
    time_std: float = 0.0
    throughput_mean: float = 0.0
    throughput_std: float = 0.0


@dataclass
class BenchmarkRun:
    """A single benchmark run (baseline or runahead)."""
    mode: str
    primary_size: int
    long_tail_ratio: float
    load_threshold: int = 0
    max_secondary_concurrent: int = 0
    metrics: RunMetrics = field(default_factory=RunMetrics)
    filepath: str = ""

    @property
    def config_key(self) -> tuple:
        """Key to match baseline with runahead (primary_size, long_tail_ratio)."""
        return (self.primary_size, self.long_tail_ratio)

    @property
    def runahead_config_key(self) -> tuple:
        """Full key for runahead configs."""
        return (self.primary_size, self.long_tail_ratio,
                self.load_threshold, self.max_secondary_concurrent)


@dataclass
class ComparisonResult:
    """Comparison between a baseline and runahead run."""
    primary_size: int
    long_tail_ratio: float
    load_threshold: int
    max_secondary_concurrent: int

    # Baseline metrics
    baseline_time: float = 0.0
    baseline_time_std: float = 0.0
    baseline_throughput: float = 0.0
    baseline_primary_tokens: int = 0

    # Runahead metrics
    runahead_time: float = 0.0
    runahead_time_std: float = 0.0
    runahead_throughput: float = 0.0
    runahead_primary_tokens: int = 0
    runahead_tokens_total: int = 0
    runahead_tokens_completed: int = 0
    runahead_tokens_aborted: int = 0
    runahead_completed_count: int = 0
    runahead_aborted_count: int = 0

    @property
    def extra_time_seconds(self) -> float:
        """Extra time spent with runahead vs baseline."""
        return self.runahead_time - self.baseline_time

    @property
    def extra_time_percent(self) -> float:
        """Extra time as percentage of baseline."""
        if self.baseline_time == 0:
            return 0.0
        return (self.extra_time_seconds / self.baseline_time) * 100

    @property
    def runahead_total_throughput(self) -> float:
        """Total throughput with runahead: (primary + completed runahead) / time."""
        if self.runahead_time <= 0:
            return 0.0
        return (self.runahead_primary_tokens + self.runahead_tokens_completed) / self.runahead_time

    @property
    def throughput_gain(self) -> float:
        """Absolute throughput gain: runahead_total_throughput - baseline_throughput."""
        return self.runahead_total_throughput - self.baseline_throughput

    @property
    def throughput_gain_percent(self) -> float:
        """Throughput gain as percentage of baseline."""
        if self.baseline_throughput == 0:
            return 0.0
        return (self.throughput_gain / self.baseline_throughput) * 100

    @property
    def completion_rate(self) -> float:
        """Percentage of runahead tokens that completed (not aborted)."""
        if self.runahead_tokens_total == 0:
            return 0.0
        return (self.runahead_tokens_completed / self.runahead_tokens_total) * 100

    @property
    def abort_rate(self) -> float:
        """Percentage of runahead requests that were aborted."""
        total = self.runahead_completed_count + self.runahead_aborted_count
        if total == 0:
            return 0.0
        return (self.runahead_aborted_count / total) * 100


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results comparing baseline vs runahead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing benchmark JSON files"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--format",
        choices=["table", "markdown", "detailed"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--sort-by",
        choices=["throughput_gain", "extra_time", "runahead_tokens", "completion_rate", "config"],
        default="config",
        help="Sort results by (default: config)"
    )
    parser.add_argument(
        "--filter-ps",
        type=int,
        nargs="+",
        default=None,
        help="Filter by primary sizes"
    )
    parser.add_argument(
        "--filter-ltr",
        type=float,
        nargs="+",
        default=None,
        help="Filter by long-tail ratios"
    )
    return parser.parse_args()


def load_benchmark_run(filepath: Path) -> Optional[BenchmarkRun]:
    """Load a benchmark run from a JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        config = data.get("config", {})
        metrics_data = data.get("metrics", {})
        multi_round = data.get("multi_round", {})

        metrics = RunMetrics(
            time_seconds=metrics_data.get("time_seconds", 0.0),
            primary_tokens=metrics_data.get("primary_tokens", 0),
            primary_completed=metrics_data.get("primary_completed", 0),
            runahead_tokens_total=metrics_data.get("runahead_tokens_total", 0),
            runahead_tokens_completed=metrics_data.get("runahead_tokens_completed", 0),
            runahead_tokens_aborted=metrics_data.get("runahead_tokens_aborted", 0),
            runahead_completed_count=metrics_data.get("runahead_completed_count", 0),
            runahead_aborted_count=metrics_data.get("runahead_aborted_count", 0),
            runahead_rejected_count=metrics_data.get("runahead_rejected_count", 0),
        )

        # Add multi-round stats if available
        if multi_round:
            time_stats = multi_round.get("time_stats", {})
            throughput_stats = multi_round.get("throughput_stats", {})
            metrics.time_mean = time_stats.get("mean", metrics.time_seconds)
            metrics.time_std = time_stats.get("std", 0.0)
            metrics.throughput_mean = throughput_stats.get("mean", 0.0)
            metrics.throughput_std = throughput_stats.get("std", 0.0)
        else:
            metrics.time_mean = metrics.time_seconds
            if metrics.time_seconds > 0:
                metrics.throughput_mean = metrics.primary_tokens / metrics.time_seconds

        return BenchmarkRun(
            mode=data.get("mode", "unknown"),
            primary_size=config.get("primary_size", 0),
            long_tail_ratio=config.get("long_tail_ratio", 0.0),
            load_threshold=config.get("load_threshold", 0),
            max_secondary_concurrent=config.get("max_secondary_concurrent", 0),
            metrics=metrics,
            filepath=str(filepath),
        )
    except Exception as e:
        print(f"Warning: Failed to load {filepath}: {e}", file=sys.stderr)
        return None


def load_all_runs(results_dir: Path) -> list[BenchmarkRun]:
    """Load all benchmark runs from a directory."""
    runs = []
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"Error: No JSON files found in {results_dir}", file=sys.stderr)
        return runs

    for filepath in sorted(json_files):
        run = load_benchmark_run(filepath)
        if run:
            runs.append(run)

    return runs


def build_comparisons(runs: list[BenchmarkRun]) -> list[ComparisonResult]:
    """Build comparison results matching baselines with runahead runs."""
    # Group baselines by config
    baselines: dict[tuple, BenchmarkRun] = {}
    runahead_runs: list[BenchmarkRun] = []

    for run in runs:
        if run.mode == "baseline":
            baselines[run.config_key] = run
        elif run.mode == "runahead":
            runahead_runs.append(run)

    comparisons = []

    for ra in runahead_runs:
        baseline = baselines.get(ra.config_key)
        if baseline is None:
            print(f"Warning: No baseline found for ps={ra.primary_size}, "
                  f"ltr={ra.long_tail_ratio}", file=sys.stderr)
            continue

        comp = ComparisonResult(
            primary_size=ra.primary_size,
            long_tail_ratio=ra.long_tail_ratio,
            load_threshold=ra.load_threshold,
            max_secondary_concurrent=ra.max_secondary_concurrent,
            baseline_time=baseline.metrics.time_mean,
            baseline_time_std=baseline.metrics.time_std,
            baseline_throughput=baseline.metrics.throughput_mean,
            baseline_primary_tokens=baseline.metrics.primary_tokens,
            runahead_time=ra.metrics.time_mean,
            runahead_time_std=ra.metrics.time_std,
            runahead_throughput=ra.metrics.throughput_mean,
            runahead_primary_tokens=ra.metrics.primary_tokens,
            runahead_tokens_total=ra.metrics.runahead_tokens_total,
            runahead_tokens_completed=ra.metrics.runahead_tokens_completed,
            runahead_tokens_aborted=ra.metrics.runahead_tokens_aborted,
            runahead_completed_count=ra.metrics.runahead_completed_count,
            runahead_aborted_count=ra.metrics.runahead_aborted_count,
        )
        comparisons.append(comp)

    return comparisons


def sort_comparisons(comparisons: list[ComparisonResult],
                     sort_by: str) -> list[ComparisonResult]:
    """Sort comparisons by specified key."""
    if sort_by == "throughput_gain":
        return sorted(comparisons, key=lambda c: -c.throughput_gain_percent)
    elif sort_by == "extra_time":
        return sorted(comparisons, key=lambda c: c.extra_time_percent)
    elif sort_by == "runahead_tokens":
        return sorted(comparisons, key=lambda c: -c.runahead_tokens_total)
    elif sort_by == "completion_rate":
        return sorted(comparisons, key=lambda c: -c.completion_rate)
    else:  # config
        return sorted(comparisons,
                     key=lambda c: (c.primary_size, c.long_tail_ratio,
                                   c.load_threshold, c.max_secondary_concurrent))


def filter_comparisons(comparisons: list[ComparisonResult],
                       filter_ps: Optional[list[int]],
                       filter_ltr: Optional[list[float]]) -> list[ComparisonResult]:
    """Filter comparisons by specified criteria."""
    result = comparisons
    if filter_ps:
        result = [c for c in result if c.primary_size in filter_ps]
    if filter_ltr:
        result = [c for c in result if c.long_tail_ratio in filter_ltr]
    return result


def format_number(n: float, precision: int = 1) -> str:
    """Format a number with K/M suffix for large values."""
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.{precision}f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def print_summary(comparisons: list[ComparisonResult], baselines: dict) -> None:
    """Print summary statistics."""
    if not comparisons:
        print("No comparisons available.")
        return

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Baseline summary
    print("\n--- Baselines ---")
    for (ps, ltr), baseline in sorted(baselines.items()):
        print(f"  ps={ps:4d}, ltr={ltr:.2f}: time={baseline.metrics.time_mean:.1f}s, "
              f"throughput={baseline.metrics.throughput_mean:.0f} tok/s")

    # Overall statistics
    avg_extra_time = sum(c.extra_time_percent for c in comparisons) / len(comparisons)
    avg_throughput_gain = sum(c.throughput_gain_percent for c in comparisons) / len(comparisons)
    avg_runahead_tokens = sum(c.runahead_tokens_total for c in comparisons) / len(comparisons)
    avg_completion_rate = sum(c.completion_rate for c in comparisons) / len(comparisons)

    print("\n--- Runahead Overall ---")
    print(f"  Configurations analyzed: {len(comparisons)}")
    print(f"  Avg extra time: {avg_extra_time:.1f}%")
    print(f"  Avg throughput gain: {avg_throughput_gain:+.1f}%")
    print(f"  Avg runahead tokens: {format_number(avg_runahead_tokens)}")
    print(f"  Avg completion rate: {avg_completion_rate:.1f}%")

    # Best configurations
    best_throughput = max(comparisons, key=lambda c: c.throughput_gain_percent)
    best_tokens = max(comparisons, key=lambda c: c.runahead_tokens_completed)
    lowest_overhead = min(comparisons, key=lambda c: c.extra_time_percent if c.extra_time_percent > 0 else float('inf'))

    print("\n--- Best Configurations ---")
    print(f"  Highest throughput gain:")
    print(f"    ps={best_throughput.primary_size}, ltr={best_throughput.long_tail_ratio}, "
          f"lt={best_throughput.load_threshold}, msc={best_throughput.max_secondary_concurrent}")
    print(f"    -> {best_throughput.throughput_gain_percent:+.1f}% throughput, "
          f"{best_throughput.extra_time_percent:.1f}% extra time")

    print(f"  Most completed runahead tokens:")
    print(f"    ps={best_tokens.primary_size}, ltr={best_tokens.long_tail_ratio}, "
          f"lt={best_tokens.load_threshold}, msc={best_tokens.max_secondary_concurrent}")
    print(f"    -> {format_number(best_tokens.runahead_tokens_completed)} completed, "
          f"{best_tokens.throughput_gain_percent:+.1f}% throughput")

    print(f"  Lowest time overhead:")
    print(f"    ps={lowest_overhead.primary_size}, ltr={lowest_overhead.long_tail_ratio}, "
          f"lt={lowest_overhead.load_threshold}, msc={lowest_overhead.max_secondary_concurrent}")
    print(f"    -> {lowest_overhead.extra_time_percent:.1f}% extra time, "
          f"{lowest_overhead.throughput_gain_percent:+.1f}% throughput")


def print_table(comparisons: list[ComparisonResult]) -> None:
    """Print results as a table."""
    if not comparisons:
        print("No comparison data available.")
        return

    print("\n" + "=" * 155)
    print("DETAILED COMPARISON TABLE")
    print("=" * 155)

    # Header
    header = (f"{'PS':>5} {'LTR':>5} {'LT':>4} {'MSC':>4} | "
              f"{'Base(s)':>8} {'RA(s)':>8} {'Extra%':>7} | "
              f"{'Pri Tok':>8} {'RA Tok':>8} {'Compl':>8} {'Abort':>7} {'Compl%':>6} | "
              f"{'BaseTP':>7} {'RA TP':>7} {'Gain%':>7}")
    print(header)
    print("-" * 155)

    for c in comparisons:
        row = (f"{c.primary_size:>5} {c.long_tail_ratio:>5.2f} "
               f"{c.load_threshold:>4} {c.max_secondary_concurrent:>4} | "
               f"{c.baseline_time:>8.1f} {c.runahead_time:>8.1f} "
               f"{c.extra_time_percent:>6.1f}% | "
               f"{format_number(c.runahead_primary_tokens):>8} "
               f"{format_number(c.runahead_tokens_total):>8} "
               f"{format_number(c.runahead_tokens_completed):>8} "
               f"{format_number(c.runahead_tokens_aborted):>7} "
               f"{c.completion_rate:>5.1f}% | "
               f"{c.baseline_throughput:>7.0f} {c.runahead_total_throughput:>7.0f} "
               f"{c.throughput_gain_percent:>+6.1f}%")
        print(row)

    print("=" * 155)
    print("Legend: PS=Primary Size, LTR=Long-Tail Ratio, LT=Load Threshold, MSC=Max Secondary Concurrent")
    print("        BaseTP=Baseline Throughput (tok/s), RA TP=Runahead Total Throughput (pri+completed)/time, Gain%=Throughput Gain %")


def print_markdown(comparisons: list[ComparisonResult]) -> None:
    """Print results as markdown table."""
    if not comparisons:
        print("No comparison data available.")
        return

    print("\n## Benchmark Results: Runahead vs Baseline\n")

    # Header
    print("| PS | LTR | LT | MSC | Base(s) | RA(s) | Extra% | Pri Tok | RA Tok | Compl | Abort | Compl% | Base TP | RA TP | Gain% |")
    print("|---:|----:|---:|----:|--------:|------:|-------:|--------:|-------:|------:|------:|-------:|--------:|------:|------:|")

    for c in comparisons:
        print(f"| {c.primary_size} | {c.long_tail_ratio:.2f} | {c.load_threshold} | {c.max_secondary_concurrent} | "
              f"{c.baseline_time:.1f} | {c.runahead_time:.1f} | {c.extra_time_percent:.1f}% | "
              f"{format_number(c.runahead_primary_tokens)} | {format_number(c.runahead_tokens_total)} | "
              f"{format_number(c.runahead_tokens_completed)} | {format_number(c.runahead_tokens_aborted)} | "
              f"{c.completion_rate:.1f}% | {c.baseline_throughput:.0f} | {c.runahead_total_throughput:.0f} | "
              f"{c.throughput_gain_percent:+.1f}% |")


def print_detailed(comparisons: list[ComparisonResult]) -> None:
    """Print detailed results for each comparison."""
    if not comparisons:
        print("No comparison data available.")
        return

    # Group by primary_size and long_tail_ratio
    grouped = defaultdict(list)
    for c in comparisons:
        grouped[(c.primary_size, c.long_tail_ratio)].append(c)

    for (ps, ltr), group in sorted(grouped.items()):
        print("\n" + "=" * 80)
        print(f"PRIMARY SIZE: {ps}, LONG-TAIL RATIO: {ltr:.0%}")
        print("=" * 80)

        # Get baseline info from first comparison
        baseline_time = group[0].baseline_time
        baseline_throughput = group[0].baseline_throughput
        baseline_tokens = group[0].baseline_primary_tokens
        print(f"\nBaseline: {baseline_time:.1f}s, {baseline_throughput:.0f} tok/s, {format_number(baseline_tokens)} primary tokens")
        print()

        print(f"{'LT':>4} {'MSC':>4} | {'Time(s)':>8} {'Extra%':>7} | "
              f"{'Pri Tok':>10} {'RA Total':>10} {'Completed':>10} {'Aborted':>10} | "
              f"{'Compl%':>6} | {'RA TP':>7} {'Gain%':>7}")
        print("-" * 115)

        for c in sorted(group, key=lambda x: (x.load_threshold, x.max_secondary_concurrent)):
            print(f"{c.load_threshold:>4} {c.max_secondary_concurrent:>4} | "
                  f"{c.runahead_time:>8.1f} {c.extra_time_percent:>6.1f}% | "
                  f"{format_number(c.runahead_primary_tokens):>10} "
                  f"{format_number(c.runahead_tokens_total):>10} "
                  f"{format_number(c.runahead_tokens_completed):>10} "
                  f"{format_number(c.runahead_tokens_aborted):>10} | "
                  f"{c.completion_rate:>5.1f}% | "
                  f"{c.runahead_total_throughput:>7.0f} {c.throughput_gain_percent:>+6.1f}%")


def export_csv(comparisons: list[ComparisonResult], filepath: str) -> None:
    """Export results to CSV file."""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "primary_size", "long_tail_ratio", "load_threshold", "max_secondary_concurrent",
            "baseline_time_s", "baseline_throughput_tok_s", "baseline_primary_tokens",
            "runahead_time_s", "extra_time_s", "extra_time_percent", "runahead_primary_tokens",
            "runahead_tokens_total", "runahead_tokens_completed", "runahead_tokens_aborted",
            "completion_rate_percent", "runahead_total_throughput_tok_s", "throughput_gain_percent"
        ])

        # Data rows
        for c in comparisons:
            writer.writerow([
                c.primary_size, c.long_tail_ratio, c.load_threshold, c.max_secondary_concurrent,
                f"{c.baseline_time:.2f}", f"{c.baseline_throughput:.1f}", c.baseline_primary_tokens,
                f"{c.runahead_time:.2f}", f"{c.extra_time_seconds:.2f}", f"{c.extra_time_percent:.2f}",
                c.runahead_primary_tokens,
                c.runahead_tokens_total, c.runahead_tokens_completed, c.runahead_tokens_aborted,
                f"{c.completion_rate:.2f}", f"{c.runahead_total_throughput:.2f}",
                f"{c.throughput_gain_percent:.2f}"
            ])

    print(f"\nResults exported to: {filepath}")


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading benchmark results from: {results_dir}")

    # Load all runs
    runs = load_all_runs(results_dir)
    if not runs:
        sys.exit(1)

    print(f"Loaded {len(runs)} benchmark files")

    # Separate baselines and runahead runs
    baselines = {r.config_key: r for r in runs if r.mode == "baseline"}
    runahead_runs = [r for r in runs if r.mode == "runahead"]

    print(f"  - {len(baselines)} baseline configurations")
    print(f"  - {len(runahead_runs)} runahead configurations")

    # Build comparisons
    comparisons = build_comparisons(runs)

    # Apply filters
    comparisons = filter_comparisons(comparisons, args.filter_ps, args.filter_ltr)

    # Sort
    comparisons = sort_comparisons(comparisons, args.sort_by)

    # Print summary
    print_summary(comparisons, baselines)

    # Print in requested format
    if args.format == "table":
        print_table(comparisons)
    elif args.format == "markdown":
        print_markdown(comparisons)
    elif args.format == "detailed":
        print_detailed(comparisons)

    # Export to CSV if requested
    if args.csv:
        export_csv(comparisons, args.csv)


if __name__ == "__main__":
    main()
