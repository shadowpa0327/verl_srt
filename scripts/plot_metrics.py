#!/usr/bin/env python3
"""
Visualize vLLM metrics collected during rollout.

Usage:
    python scripts/plot_metrics.py metrics.csv                    # Saves to metrics.png
    python scripts/plot_metrics.py metrics.csv -o output.png      # Custom output path
    python scripts/plot_metrics.py metrics.csv --aggregate        # Aggregate across servers
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot vLLM metrics from CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_file", type=str, help="Path to metrics CSV file")
    parser.add_argument("--output", "-o", type=str, default="metrics.png", help="Output image file")
    parser.add_argument("--aggregate", "-a", action="store_true", help="Show aggregate metrics instead of per-server")
    parser.add_argument("--figsize", type=str, default="14,10", help="Figure size as 'width,height'")
    return parser.parse_args()


def plot_per_server(df: pd.DataFrame, figsize: tuple[float, float]) -> plt.Figure:
    """Plot metrics with separate lines for each server."""
    servers = sorted(df["server_idx"].unique())
    num_servers = len(servers)

    # Check if optional columns are available
    has_itl = "itl_avg_ms" in df.columns and df["itl_avg_ms"].notna().any()
    has_secondary = "secondary_load" in df.columns and df["secondary_load"].notna().any()

    # Determine grid layout based on number of metrics (base 4 + optional)
    num_extra = int(has_itl) + int(has_secondary)
    if num_extra == 0:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    elif num_extra == 1:
        fig, axes = plt.subplots(2, 3, figsize=(figsize[0] * 1.3, figsize[1]))
    else:  # num_extra == 2
        fig, axes = plt.subplots(2, 3, figsize=(figsize[0] * 1.3, figsize[1]))

    fig.suptitle("vLLM Metrics During Rollout (Per Server)", fontsize=14, fontweight="bold")

    metrics = [
        ("num_requests_running", "Requests Running", "tab:blue"),
        ("num_requests_waiting", "Requests Waiting", "tab:orange"),
        ("kv_cache_usage", "KV Cache Usage", "tab:green"),
        ("server_load", "Server Load (Router)", "tab:red"),
    ]

    if has_secondary:
        metrics.append(("secondary_load", "Secondary (Runahead) Load", "tab:brown"))
    if has_itl:
        metrics.append(("itl_avg_ms", "Inter-Token Latency (ms)", "tab:purple"))

    for ax, (metric, title, base_color) in zip(axes.flat, metrics):
        for i, server_idx in enumerate(servers):
            server_df = df[df["server_idx"] == server_idx].sort_values("timestamp")
            alpha = 0.5 + 0.5 * (i / max(num_servers - 1, 1))  # Vary alpha for visibility
            ax.plot(
                server_df["timestamp"],
                server_df[metric],
                label=f"Server {server_idx}",
                alpha=alpha,
                linewidth=1.5,
            )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Format KV cache as percentage
        if metric == "kv_cache_usage":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.set_ylim(0, 1.05)

    # Hide unused axes
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


def plot_aggregate(df: pd.DataFrame, figsize: tuple[float, float]) -> plt.Figure:
    """Plot aggregate metrics across all servers."""
    # Check if optional columns are available
    has_itl = "itl_avg_ms" in df.columns and df["itl_avg_ms"].notna().any()
    has_secondary = "secondary_load" in df.columns and df["secondary_load"].notna().any()

    # Group by timestamp and aggregate
    agg_cols = {
        "num_requests_running": "sum",
        "num_requests_waiting": "sum",
        "kv_cache_usage": "mean",
        "server_load": "sum",
    }
    if has_secondary:
        agg_cols["secondary_load"] = "sum"  # Total secondary load across servers
    if has_itl:
        agg_cols["itl_avg_ms"] = "mean"  # Average across servers

    agg_df = df.groupby("timestamp").agg(agg_cols).reset_index().sort_values("timestamp")

    # Determine grid layout based on number of metrics (base 4 + optional)
    num_extra = int(has_itl) + int(has_secondary)
    if num_extra == 0:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
    elif num_extra == 1:
        fig, axes = plt.subplots(2, 3, figsize=(figsize[0] * 1.3, figsize[1]))
    else:  # num_extra == 2
        fig, axes = plt.subplots(2, 3, figsize=(figsize[0] * 1.3, figsize[1]))

    fig.suptitle("vLLM Metrics During Rollout (Aggregate)", fontsize=14, fontweight="bold")

    metrics = [
        ("num_requests_running", "Total Requests Running", "tab:blue"),
        ("num_requests_waiting", "Total Requests Waiting", "tab:orange"),
        ("kv_cache_usage", "Avg KV Cache Usage", "tab:green"),
        ("server_load", "Total Server Load", "tab:red"),
    ]

    if has_secondary:
        metrics.append(("secondary_load", "Total Secondary (Runahead) Load", "tab:brown"))
    if has_itl:
        metrics.append(("itl_avg_ms", "Avg Inter-Token Latency (ms)", "tab:purple"))

    for ax, (metric, title, color) in zip(axes.flat, metrics):
        ax.plot(agg_df["timestamp"], agg_df[metric], color=color, linewidth=2)
        ax.fill_between(agg_df["timestamp"], agg_df[metric], alpha=0.3, color=color)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Format KV cache as percentage
        if metric == "kv_cache_usage":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax.set_ylim(0, 1.05)

    # Hide unused axes
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    # Parse figsize
    figsize = tuple(float(x) for x in args.figsize.split(","))

    # Read CSV
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ["timestamp", "server_idx", "num_requests_running", "num_requests_waiting", "kv_cache_usage"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Fill missing server_load with 0 if not present
    if "server_load" not in df.columns:
        df["server_load"] = 0

    # Print summary
    num_samples = len(df)
    num_servers = df["server_idx"].nunique()
    duration = df["timestamp"].max() - df["timestamp"].min()
    print(f"Loaded {num_samples} samples from {num_servers} servers over {duration:.2f}s")

    # Plot
    if args.aggregate:
        fig = plot_aggregate(df, figsize)
    else:
        fig = plot_per_server(df, figsize)

    # Save to file
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
