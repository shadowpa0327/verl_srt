#!/usr/bin/env python3
"""
Plot replay simulator results across multiple training ticks.

This script runs the replay simulator for multiple cache_tick/sim_tick pairs
and generates line plots showing acceptance rate and tokens/step over training.
"""

import argparse
import subprocess
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def run_simulation(
    model_path: str,
    data_dir: str,
    cache_tick: int,
    sim_tick: int,
    mode: str = "parallel",
    min_token_prob: float = 0.3,
    online_update: bool = False,
) -> Dict:
    """Run a single simulation and return results."""
    cmd = [
        sys.executable,
        "recipe/srt/replay_simulator.py",
        "--model_path", model_path,
        "--data_dir", data_dir,
        "--mode", mode,
        "--cache_tick", str(cache_tick),
        "--sim_tick", str(sim_tick),
        "--min_token_prob", str(min_token_prob),
        "--max_samples", "0",
    ]
    if online_update:
        cmd.append("--online_update")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Parse results from output
    accept_match = re.search(r"Mean acceptance rate:\s+([\d.]+)", output)
    tps_match = re.search(r"Mean tokens/step:\s+([\d.]+)", output)
    steps_match = re.search(r"Total decoding steps:\s+([\d,]+)", output)

    if accept_match and tps_match:
        return {
            "cache_tick": cache_tick,
            "sim_tick": sim_tick,
            "acceptance_rate": float(accept_match.group(1)),
            "tokens_per_step": float(tps_match.group(1)),
            "total_steps": int(steps_match.group(1).replace(",", "")) if steps_match else 0,
        }
    else:
        print(f"  Warning: Could not parse results for tick {cache_tick}->{sim_tick}")
        print(f"  Output: {output[-500:]}")
        return None


def run_sweep(
    model_path: str,
    data_dir: str,
    tick_pairs: List[Tuple[int, int]],
    mode: str,
    min_token_prob: float,
    online_update: bool,
    label: str,
) -> List[Dict]:
    """Run simulations for all tick pairs."""
    results = []
    print(f"\nRunning: {label}")
    for cache_tick, sim_tick in tick_pairs:
        print(f"  Tick {cache_tick} -> {sim_tick}...", end=" ", flush=True)
        result = run_simulation(
            model_path, data_dir, cache_tick, sim_tick,
            mode, min_token_prob, online_update
        )
        if result:
            result["label"] = label
            results.append(result)
            print(f"accept={result['acceptance_rate']:.3f}, tps={result['tokens_per_step']:.3f}")
        else:
            print("FAILED")
    return results


def plot_results(all_results: Dict[str, List[Dict]], output_path: str):
    """Generate line plots from results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for idx, (label, results) in enumerate(all_results.items()):
        if not results:
            continue

        sim_ticks = [r["sim_tick"] for r in results]
        accept_rates = [r["acceptance_rate"] * 100 for r in results]  # Convert to percentage
        tokens_per_step = [r["tokens_per_step"] for r in results]

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        axes[0].plot(sim_ticks, accept_rates, marker=marker, color=color,
                     label=label, linewidth=2, markersize=6)
        axes[1].plot(sim_ticks, tokens_per_step, marker=marker, color=color,
                     label=label, linewidth=2, markersize=6)

    axes[0].set_xlabel("Training Tick (sim_tick)", fontsize=12)
    axes[0].set_ylabel("Acceptance Rate (%)", fontsize=12)
    axes[0].set_title("Speculation Acceptance Rate Over Training", fontsize=14)
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training Tick (sim_tick)", fontsize=12)
    axes[1].set_ylabel("Tokens per Step", fontsize=12)
    axes[1].set_title("Tokens per Decoding Step Over Training", fontsize=14)
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot replay simulator results over training ticks")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B",
                        help="Path to HuggingFace model")
    parser.add_argument("--data_dir", type=str,
                        default="/home/ubuntu/verl_srt/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead",
                        help="Path to rollout data directory")
    parser.add_argument("--start_tick", type=int, default=1,
                        help="Starting tick (default: 1)")
    parser.add_argument("--end_tick", type=int, default=10,
                        help="Ending tick for simulation (default: 10)")
    parser.add_argument("--output", type=str, default="replay_over_ticks.png",
                        help="Output plot filename")
    parser.add_argument("--mode", type=str, default="parallel",
                        choices=["shm", "parallel"],
                        help="Simulation mode (default: parallel)")

    args = parser.parse_args()

    # Generate tick pairs: (cache_tick=N, sim_tick=N+1)
    tick_pairs = [(i, i + 1) for i in range(args.start_tick, args.end_tick)]

    all_results = {}

    # Run different configurations
    configs = [
        {"min_token_prob": 0.3, "online_update": False, "label": "prob=0.3"},
        {"min_token_prob": 0.3, "online_update": True, "label": "prob=0.3 + online"},
        {"min_token_prob": 0.5, "online_update": False, "label": "prob=0.5"},
        {"min_token_prob": 0.5, "online_update": True, "label": "prob=0.5 + online"},
    ]

    for config in configs:
        results = run_sweep(
            args.model_path,
            args.data_dir,
            tick_pairs,
            args.mode,
            config["min_token_prob"],
            config["online_update"],
            config["label"],
        )
        all_results[config["label"]] = results

    # Plot results
    plot_results(all_results, args.output)

    # Also save raw data as JSON
    json_path = args.output.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    main()
