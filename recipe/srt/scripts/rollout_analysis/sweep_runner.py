#!/usr/bin/env python3
"""
Sweep Runner for SRT Speculation Analysis.

This module provides the SweepRunner class that runs replay simulations across
multiple training ticks and modes, collecting per-request metrics for analysis.
"""

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .analysis_config import MODE_CONFIG, SweepConfig
from .data_discovery import DataDirectoryInfo, discover_data_directory, validate_tick_range


@dataclass
class SweepResults:
    """Results from a simulation sweep."""

    # Results by mode
    prefill_only: List[Dict] = field(default_factory=list)
    online_only: List[Dict] = field(default_factory=list)
    prefill_plus_online: List[Dict] = field(default_factory=list)

    # Metadata
    tick_pairs: List[Tuple[int, int]] = field(default_factory=list)
    config: Optional[SweepConfig] = None
    errors: List[str] = field(default_factory=list)

    @property
    def total_simulations(self) -> int:
        """Total number of successful simulations."""
        return len(self.prefill_only) + len(self.online_only) + len(self.prefill_plus_online)

    def get_all_results(self) -> List[Dict]:
        """Get all results with mode labels."""
        all_results = []
        for r in self.prefill_only:
            r["mode"] = "prefill_only"
            all_results.append(r)
        for r in self.online_only:
            r["mode"] = "online_only"
            all_results.append(r)
        for r in self.prefill_plus_online:
            r["mode"] = "prefill_plus_online"
            all_results.append(r)
        return all_results


class SweepRunner:
    """
    Run replay simulations across multiple ticks and modes.

    Usage:
        config = SweepConfig(data_dir=Path(...), output_dir=Path(...))
        runner = SweepRunner(config)
        results = runner.run()
        runner.save_results(results)
    """

    def __init__(self, config: SweepConfig):
        """
        Initialize SweepRunner.

        Args:
            config: SweepConfig with paths and parameters.
        """
        self.config = config
        self._data_info: Optional[DataDirectoryInfo] = None
        self._simulator = None

    @property
    def data_info(self) -> DataDirectoryInfo:
        """Get data directory info, discovering if needed."""
        if self._data_info is None:
            self._data_info = discover_data_directory(self.config.data_dir)
            if not self._data_info.valid:
                raise ValueError(f"Invalid data directory: {self._data_info.error}")
        return self._data_info

    def _get_tick_pairs(self) -> List[Tuple[int, int]]:
        """Get list of (cache_tick, sim_tick) pairs to run."""
        # Validate and adjust tick range
        tick_start, tick_end, warnings = validate_tick_range(
            self.data_info,
            self.config.tick_start,
            self.config.tick_end,
            self.config.tick_step,
        )

        for w in warnings:
            print(f"Warning: {w}")

        # Generate tick pairs
        ticks = list(range(tick_start, tick_end, self.config.tick_step))
        tick_pairs = []

        for cache_tick in ticks:
            sim_tick = cache_tick + 1
            if self.data_info.has_tick_pair(cache_tick, sim_tick):
                tick_pairs.append((cache_tick, sim_tick))
            else:
                print(f"Skipping tick pair ({cache_tick}, {sim_tick}): missing data")

        return tick_pairs

    def _run_single_simulation(
        self,
        cache_tick: int,
        sim_tick: int,
        online_update: bool,
        skip_prefill: bool,
    ) -> Optional[Dict]:
        """
        Run a single simulation.

        Args:
            cache_tick: Tick to populate cache from.
            sim_tick: Tick to simulate.
            online_update: Whether to enable online updates.
            skip_prefill: Whether to skip prefill from secondary.

        Returns:
            Result dictionary or None on error.
        """
        try:
            # Import here to avoid import at module load time
            # This is necessary because the simulator may not be available in all environments
            sys.path.insert(0, str(Path(__file__).parents[3]))
            from recipe.srt.replay_simulator import SimulatorConfig, run_simulation

            config = SimulatorConfig(
                model_path=self.config.model_path,
                data_dir=str(self.config.data_dir),
                mode="parallel",
                cache_tick=cache_tick,
                sim_tick=sim_tick,
                min_token_prob=self.config.min_token_prob,
                hash_token_count=self.config.hash_token_count,
                max_tree_depth=self.config.max_tree_depth,
                spec_prefix_len=self.config.spec_prefix_len,
                spec_max_len=self.config.spec_max_len,
                online_update=online_update,
                skip_prefill=skip_prefill,
                max_samples=self.config.max_samples,
                verbose=self.config.verbose,
            )

            result = run_simulation(config)

            # Add tick info
            result["cache_tick"] = cache_tick
            result["sim_tick"] = sim_tick
            result["online_update"] = online_update
            result["skip_prefill"] = skip_prefill

            return result

        except Exception as e:
            print(f"  Error for tick {cache_tick}->{sim_tick}: {e}")
            if self.config.verbose:
                traceback.print_exc()
            return None

    def run(self) -> SweepResults:
        """
        Run the full sweep.

        Returns:
            SweepResults with all simulation results.
        """
        results = SweepResults(config=self.config)
        tick_pairs = self._get_tick_pairs()
        results.tick_pairs = tick_pairs

        if not tick_pairs:
            results.errors.append("No valid tick pairs found")
            return results

        print(f"\nSweep configuration:")
        print(f"  Data directory: {self.config.data_dir}")
        print(f"  Tick pairs: {len(tick_pairs)}")
        print(f"  Tick range: {tick_pairs[0]} to {tick_pairs[-1]}")
        print(f"  min_token_prob: {self.config.min_token_prob}")
        print(f"  Run prefill only: {self.config.run_prefill_only}")
        print(f"  Run online only: {self.config.run_online_only}")
        print(f"  Run prefill + online: {self.config.run_prefill_plus_online}")

        # Run prefill only mode (online_update=False, skip_prefill=False)
        if self.config.run_prefill_only:
            print(f"\n{'='*60}")
            print("Running PREFILL ONLY (no online updates)...")
            print("=" * 60)

            for i, (cache_tick, sim_tick) in enumerate(tick_pairs):
                print(f"\n[{i+1}/{len(tick_pairs)}] Tick {cache_tick} -> {sim_tick}...")
                result = self._run_single_simulation(
                    cache_tick, sim_tick, online_update=False, skip_prefill=False
                )
                if result:
                    results.prefill_only.append(result)
                    print(
                        f"  hit_rate={result.get('mean_hit_rate', 0):.3f}, "
                        f"accept_rate={result['mean_acceptance_rate']:.3f}, "
                        f"toks/step={result['mean_tokens_per_step']:.3f}"
                    )

        # Run prefill + online mode (online_update=True, skip_prefill=False)
        if self.config.run_prefill_plus_online:
            print(f"\n{'='*60}")
            print("Running PREFILL + ONLINE updates...")
            print("=" * 60)

            for i, (cache_tick, sim_tick) in enumerate(tick_pairs):
                print(f"\n[{i+1}/{len(tick_pairs)}] Tick {cache_tick} -> {sim_tick}...")
                result = self._run_single_simulation(
                    cache_tick, sim_tick, online_update=True, skip_prefill=False
                )
                if result:
                    results.prefill_plus_online.append(result)
                    print(
                        f"  hit_rate={result.get('mean_hit_rate', 0):.3f}, "
                        f"accept_rate={result['mean_acceptance_rate']:.3f}, "
                        f"toks/step={result['mean_tokens_per_step']:.3f}"
                    )

        # Run online only mode (online_update=True, skip_prefill=True)
        if self.config.run_online_only:
            print(f"\n{'='*60}")
            print("Running ONLINE ONLY (no prefill from secondary)...")
            print("=" * 60)

            for i, (cache_tick, sim_tick) in enumerate(tick_pairs):
                print(f"\n[{i+1}/{len(tick_pairs)}] Tick {cache_tick} -> {sim_tick}...")
                result = self._run_single_simulation(
                    cache_tick, sim_tick, online_update=True, skip_prefill=True
                )
                if result:
                    results.online_only.append(result)
                    print(
                        f"  hit_rate={result.get('mean_hit_rate', 0):.3f}, "
                        f"accept_rate={result['mean_acceptance_rate']:.3f}, "
                        f"toks/step={result['mean_tokens_per_step']:.3f}"
                    )

        return results

    def save_results(self, results: SweepResults, output_dir: Optional[Path] = None):
        """
        Save sweep results to JSON and CSV files.

        Args:
            results: SweepResults to save.
            output_dir: Directory to save to (uses config.output_dir if None).
        """
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        def make_summary_entry(r: Dict) -> Dict:
            return {
                "cache_tick": r["cache_tick"],
                "sim_tick": r["sim_tick"],
                "num_requests": r["num_requests"],
                "total_steps": r["total_steps"],
                "total_steps_with_drafts": r.get("total_steps_with_drafts", 0),
                "hit_rate": r.get("mean_hit_rate", 0),
                "acceptance_rate": r["mean_acceptance_rate"],
                "tokens_per_step": r["mean_tokens_per_step"],
                "tokens_per_hit_step": r.get("mean_tokens_per_hit_step", 0),
                "draft_contribution": r.get("mean_draft_contribution", 0),
            }

        # Save summary JSON
        summary_data = {
            "prefill_only": [make_summary_entry(r) for r in results.prefill_only],
            "online_only": [make_summary_entry(r) for r in results.online_only],
            "prefill_plus_online": [make_summary_entry(r) for r in results.prefill_plus_online],
            # Keep old keys for backwards compatibility
            "with_online_update": [make_summary_entry(r) for r in results.prefill_plus_online],
            "without_online_update": [make_summary_entry(r) for r in results.prefill_only],
        }

        summary_path = output_dir / "sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved summary to {summary_path}")

        # Save per-request CSV
        rows = []

        def add_rows(result_list: List[Dict], mode: str, online_update: bool, skip_prefill: bool):
            for r in result_list:
                for req in r.get("requests", []):
                    rows.append({
                        "cache_tick": r["cache_tick"],
                        "sim_tick": r["sim_tick"],
                        "mode": mode,
                        "online_update": online_update,
                        "skip_prefill": skip_prefill,
                        "request_id": req.get("request_id", ""),
                        "prompt_len": req.get("prompt_len", 0),
                        "response_len": req.get("response_len", 0),
                        "total_steps": req.get("total_steps", 0),
                        "steps_with_drafts": req.get("steps_with_drafts", 0),
                        "hit_rate": req.get("hit_rate", 0),
                        "acceptance_rate": req.get("acceptance_rate", 0),
                        "tokens_per_step": req.get("tokens_per_step", 0),
                        "tokens_per_hit_step": req.get("tokens_per_hit_step", 0),
                        "draft_contribution": req.get("draft_contribution", 0),
                    })

        add_rows(results.prefill_only, "prefill_only", False, False)
        add_rows(results.prefill_plus_online, "prefill_plus_online", True, False)
        add_rows(results.online_only, "online_only", True, True)

        if rows:
            df = pd.DataFrame(rows)
            csv_path = output_dir / "per_request_data.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved {len(rows)} per-request records to {csv_path}")

    def print_summary(self, results: SweepResults, min_response_len: int = 0):
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("SUMMARY: E2E Speedup Analysis")
        print("=" * 70)

        # Collect all per-request data
        all_rows = []

        def collect_rows(result_list: List[Dict], mode: str):
            for r in result_list:
                for req in r.get("requests", []):
                    row = {
                        "mode": mode,
                        "response_len": req.get("response_len", 0),
                        "hit_rate": req.get("hit_rate", 0),
                        "acceptance_rate": req.get("acceptance_rate", 0),
                        "tokens_per_step": req.get("tokens_per_step", 0),
                        "tokens_per_hit_step": req.get("tokens_per_hit_step", 0),
                        "draft_contribution": req.get("draft_contribution", 0),
                    }
                    all_rows.append(row)

        collect_rows(results.prefill_only, "prefill_only")
        collect_rows(results.prefill_plus_online, "prefill_plus_online")
        collect_rows(results.online_only, "online_only")

        if not all_rows:
            print("No data to summarize")
            return

        df = pd.DataFrame(all_rows)

        # Apply filter
        if min_response_len > 0:
            df = df[df["response_len"] >= min_response_len]
            filter_label = f"(response_len >= {min_response_len})"
        else:
            filter_label = "(all samples)"

        mode_labels = {
            "prefill_only": "PREFILL ONLY (no online)",
            "online_only": "ONLINE ONLY (no prefill)",
            "prefill_plus_online": "PREFILL + ONLINE",
        }

        for mode, label in mode_labels.items():
            subset = df[df["mode"] == mode]
            if len(subset) > 0:
                print(f"\n{label} {filter_label}:")
                print(f"  Samples: {len(subset)}")
                print(f"  Mean tokens/step (E2E speedup): {subset['tokens_per_step'].mean():.3f}x")
                print(f"  Mean hit rate:                  {subset['hit_rate'].mean():.3f} ({subset['hit_rate'].mean()*100:.1f}%)")
                print(f"  Mean tokens/hit step (ceiling): {subset['tokens_per_hit_step'].mean():.3f}x")
                print(f"  Mean acceptance rate:           {subset['acceptance_rate'].mean():.3f} ({subset['acceptance_rate'].mean()*100:.1f}%)")
                print(f"  Mean draft contribution:        {subset['draft_contribution'].mean():.3f} ({subset['draft_contribution'].mean()*100:.1f}%)")


def run_sweep(
    data_dir: Path,
    output_dir: Path,
    model_path: str = "Qwen/Qwen2.5-7B",
    tick_start: Optional[int] = None,
    tick_end: Optional[int] = None,
    tick_step: int = 5,
    run_prefill_only: bool = True,
    run_online_only: bool = True,
    run_prefill_plus_online: bool = True,
    min_token_prob: float = 0.3,
    verbose: bool = False,
) -> SweepResults:
    """
    Convenience function to run a sweep.

    Args:
        data_dir: Path to rollout data directory.
        output_dir: Path to save results.
        model_path: HuggingFace model path for tokenizer.
        tick_start: Start tick (None = auto-detect).
        tick_end: End tick (None = auto-detect).
        tick_step: Step between ticks.
        run_prefill_only: Run prefill-only mode.
        run_online_only: Run online-only mode.
        run_prefill_plus_online: Run combined mode.
        min_token_prob: Minimum probability for draft tokens.
        verbose: Enable verbose output.

    Returns:
        SweepResults with simulation results.
    """
    config = SweepConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_path=model_path,
        tick_start=tick_start,
        tick_end=tick_end,
        tick_step=tick_step,
        run_prefill_only=run_prefill_only,
        run_online_only=run_online_only,
        run_prefill_plus_online=run_prefill_plus_online,
        min_token_prob=min_token_prob,
        verbose=verbose,
    )

    runner = SweepRunner(config)
    results = runner.run()
    runner.save_results(results)
    runner.print_summary(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SRT speculation sweep")
    parser.add_argument("data_dir", type=str, help="Path to rollout data directory")
    parser.add_argument("-o", "--output", type=str, default="./sweep_results",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model path for tokenizer")
    parser.add_argument("--tick-start", type=int, default=None,
                        help="Start tick (auto-detect if not specified)")
    parser.add_argument("--tick-end", type=int, default=None,
                        help="End tick (auto-detect if not specified)")
    parser.add_argument("--tick-step", type=int, default=5,
                        help="Step between ticks")
    parser.add_argument("--prefill-only", action="store_true",
                        help="Only run prefill-only mode")
    parser.add_argument("--online-only", action="store_true",
                        help="Only run online-only mode")
    parser.add_argument("--combined-only", action="store_true",
                        help="Only run combined (prefill + online) mode")
    parser.add_argument("--all-modes", action="store_true", default=True,
                        help="Run all modes (default)")
    parser.add_argument("--min-token-prob", type=float, default=0.3,
                        help="Minimum probability for draft tokens")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")

    args = parser.parse_args()

    # Determine which modes to run
    if args.prefill_only or args.online_only or args.combined_only:
        run_prefill = args.prefill_only
        run_online = args.online_only
        run_combined = args.combined_only
    else:
        run_prefill = True
        run_online = True
        run_combined = True

    results = run_sweep(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        model_path=args.model,
        tick_start=args.tick_start,
        tick_end=args.tick_end,
        tick_step=args.tick_step,
        run_prefill_only=run_prefill,
        run_online_only=run_online,
        run_prefill_plus_online=run_combined,
        min_token_prob=args.min_token_prob,
        verbose=args.verbose,
    )

    print(f"\nSweep complete! Total simulations: {results.total_simulations}")
