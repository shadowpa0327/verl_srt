#!/usr/bin/env python3
"""
Figure Generator for SRT Speculation Analysis.

This module provides a FigureGenerator class that can create all analysis
visualizations from per-request CSV data. Figures can be generated individually
or all at once.

Available figures:
1. three_mode_comparison - 4 metrics over training ticks
2. three_mode_bars - Bar chart comparing 3 modes
3. speedup_decomposition - Speedup contribution breakdown
4. draft_contribution_over_ticks - Draft contrib + accept rate trends
5. hit_vs_acceptance_tradeoff - Scatter showing the trade-off
6. metrics_by_length - All metrics by response length
7. long_seq_heatmap - Heatmap (tick x length)
8. online_update_insight - Key insights summary
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_config import (
    FigureConfig,
    LENGTH_BINS,
    LENGTH_LABELS,
    LONG_SEQ_BINS,
    LONG_SEQ_LABELS,
    METRIC_DEFINITIONS,
    MODE_CONFIG,
)


class FigureGenerator:
    """
    Generate analysis figures from per-request data.

    Usage:
        config = FigureConfig(data_csv=Path("per_request_data.csv"))
        generator = FigureGenerator(config)
        generator.generate_all()  # or generator.generate("three_mode_comparison")
    """

    def __init__(self, config: FigureConfig):
        """
        Initialize FigureGenerator.

        Args:
            config: FigureConfig with paths and options.
        """
        self.config = config
        self._df: Optional[pd.DataFrame] = None
        self._df_filtered: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load and cache the per-request data."""
        if self._df is None:
            if self.config.data_csv is None or not self.config.data_csv.exists():
                raise FileNotFoundError(f"Data CSV not found: {self.config.data_csv}")

            self._df = pd.read_csv(self.config.data_csv)
            print(f"Loaded {len(self._df)} records from {self.config.data_csv}")

            # Create filtered view for long sequences
            self._df_filtered = self._df[
                self._df["response_len"] >= self.config.min_response_len
            ].copy()
            print(
                f"Filtered to {len(self._df_filtered)} records with "
                f"response_len >= {self.config.min_response_len}"
            )

        return self._df

    @property
    def df(self) -> pd.DataFrame:
        """Get the full dataframe."""
        if self._df is None:
            self.load_data()
        return self._df

    @property
    def df_filtered(self) -> pd.DataFrame:
        """Get the filtered (long sequences) dataframe."""
        if self._df_filtered is None:
            self.load_data()
        return self._df_filtered

    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure to output directory."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / f"{name}.{self.config.format}"
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    def _get_mode_means(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate mean metrics for each mode."""
        mode_means = {}
        for mode in ["prefill_only", "online_only", "prefill_plus_online"]:
            subset = df[df["mode"] == mode]
            if len(subset) > 0:
                mode_means[mode] = {
                    "tps": subset["tokens_per_step"].mean(),
                    "hr": subset["hit_rate"].mean(),
                    "ar": subset["acceptance_rate"].mean(),
                    "tphs": subset["tokens_per_hit_step"].mean(),
                    "dc": subset["draft_contribution"].mean(),
                }
        return mode_means

    def generate(self, figure_name: str) -> Path:
        """
        Generate a specific figure.

        Args:
            figure_name: Name of the figure to generate.

        Returns:
            Path to the generated figure.
        """
        method_name = f"_generate_{figure_name}"
        if not hasattr(self, method_name):
            raise ValueError(
                f"Unknown figure: {figure_name}. "
                f"Available: {FigureConfig.available_figures()}"
            )

        method = getattr(self, method_name)
        return method()

    def generate_all(self) -> List[Path]:
        """Generate all available figures."""
        figures = self.config.figures or FigureConfig.available_figures()
        paths = []
        for name in figures:
            try:
                path = self.generate(name)
                paths.append(path)
            except Exception as e:
                print(f"Error generating {name}: {e}")
        return paths

    # =========================================================================
    # Figure 1: Three Mode Comparison
    # =========================================================================
    def _generate_three_mode_comparison(self) -> Path:
        """Generate 4-panel plot showing metrics over training ticks."""
        df = self.df_filtered
        min_len = self.config.min_response_len

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize_double)

        metrics = [
            ("tokens_per_step", "Tokens per Step (E2E Speedup)", axes[0, 0]),
            ("hit_rate", "Hit Rate", axes[0, 1]),
            ("tokens_per_hit_step", "Tokens per Hit Step (Ceiling)", axes[1, 0]),
            ("acceptance_rate", "Acceptance Rate", axes[1, 1]),
        ]

        for metric, title, ax in metrics:
            for mode, cfg in MODE_CONFIG.items():
                subset = df[df["mode"] == mode]
                if len(subset) > 0:
                    grouped = subset.groupby("sim_tick")[metric].mean()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        f'{cfg["color"]}{cfg["marker"]}-',
                        label=cfg["label"],
                        alpha=0.8,
                        linewidth=2,
                        markersize=8,
                    )

            ax.set_xlabel("Training Tick", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f"{title}\n(Long Sequences >= {min_len})", fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, "three_mode_comparison")
        return self.config.output_dir / "three_mode_comparison.png"

    # =========================================================================
    # Figure 2: Three Mode Bars
    # =========================================================================
    def _generate_three_mode_bars(self) -> Path:
        """Generate bar chart comparing 3 modes."""
        df = self.df_filtered
        min_len = self.config.min_response_len

        fig, ax = plt.subplots(figsize=(12, 6))

        metrics_to_plot = [
            "hit_rate",
            "acceptance_rate",
            "tokens_per_step",
            "tokens_per_hit_step",
            "draft_contribution",
        ]
        metric_labels = [
            "Hit Rate",
            "Accept Rate",
            "Toks/Step\n(Speedup)",
            "Toks/Hit Step\n(Ceiling)",
            "Draft\nContribution",
        ]
        x = np.arange(len(metrics_to_plot))
        width = 0.25

        colors = {
            "prefill_only": "red",
            "online_only": "green",
            "prefill_plus_online": "blue",
        }
        labels = {
            "prefill_only": "Prefill Only",
            "online_only": "Online Only",
            "prefill_plus_online": "Prefill + Online",
        }

        for i, mode in enumerate(["prefill_only", "online_only", "prefill_plus_online"]):
            subset = df[df["mode"] == mode]
            if len(subset) > 0:
                means = [subset[m].mean() for m in metrics_to_plot]
                bars = ax.bar(
                    x + (i - 1) * width,
                    means,
                    width,
                    label=labels[mode],
                    color=colors[mode],
                    alpha=0.8,
                )

                # Add value labels
                for bar, val in zip(bars, means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.03,
                        f"{val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(
            f"Speculation Quality Comparison: 3 Modes\n"
            f"(Long Sequences >= {min_len} tokens)",
            fontsize=14,
        )
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 3.2)

        plt.tight_layout()
        self._save_figure(fig, "three_mode_bars")
        return self.config.output_dir / "three_mode_bars.png"

    # =========================================================================
    # Figure 3: Speedup Decomposition
    # =========================================================================
    def _generate_speedup_decomposition(self) -> Path:
        """Generate speedup decomposition showing contribution of each component."""
        df = self.df_filtered
        min_len = self.config.min_response_len
        mode_means = self._get_mode_means(df)

        fig, ax = plt.subplots(figsize=self.config.figsize_single)

        modes = [
            "No Speculation\n(Baseline)",
            "Prefill Only",
            "Online Only",
            "Prefill + Online",
        ]
        speedups = [
            1.0,
            mode_means.get("prefill_only", {}).get("tps", 1.0),
            mode_means.get("online_only", {}).get("tps", 1.0),
            mode_means.get("prefill_plus_online", {}).get("tps", 1.0),
        ]
        bar_colors = ["gray", "red", "green", "blue"]

        bars = ax.bar(
            modes, speedups, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=1.5
        )

        # Add speedup labels
        for bar, speedup in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        # Add horizontal lines
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        if speedups[1] > 1:
            ax.axhline(y=speedups[1], color="red", linestyle=":", alpha=0.5)
        if speedups[2] > 1:
            ax.axhline(y=speedups[2], color="green", linestyle=":", alpha=0.5)

        ax.set_ylabel("E2E Speedup (tokens/step)", fontsize=12)
        ax.set_title(
            f"E2E Speedup: Contribution of Prefill vs Online Updates\n"
            f"(Long Sequences >= {min_len} tokens)",
            fontsize=14,
        )
        ax.set_ylim(0, max(speedups) * 1.2)
        ax.grid(True, alpha=0.3, axis="y")

        # Add gain annotations
        gain_prefill = speedups[1] - 1.0
        gain_online = speedups[2] - 1.0
        gain_combined = speedups[3] - 1.0

        if gain_prefill > 0:
            ax.annotate(
                f"Prefill gain:\n+{gain_prefill:.2f}x",
                xy=(1, speedups[1] / 2 + 0.5),
                fontsize=10,
                ha="center",
                color="darkred",
            )
        if gain_online > 0:
            ax.annotate(
                f"Online gain:\n+{gain_online:.2f}x",
                xy=(2, speedups[2] / 2 + 0.5),
                fontsize=10,
                ha="center",
                color="darkgreen",
            )
        if gain_combined > 0:
            ax.annotate(
                f"Combined gain:\n+{gain_combined:.2f}x",
                xy=(3, speedups[3] / 2 + 0.5),
                fontsize=10,
                ha="center",
                color="darkblue",
            )

        plt.tight_layout()
        self._save_figure(fig, "speedup_decomposition")
        return self.config.output_dir / "speedup_decomposition.png"

    # =========================================================================
    # Figure 4: Draft Contribution Over Ticks
    # =========================================================================
    def _generate_draft_contribution_over_ticks(self) -> Path:
        """Generate draft contribution and acceptance rate trends."""
        df = self.df_filtered
        min_len = self.config.min_response_len

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Draft contribution trend
        ax = axes[0]
        for mode, cfg in MODE_CONFIG.items():
            subset = df[df["mode"] == mode]
            if len(subset) > 0:
                grouped = subset.groupby("sim_tick")["draft_contribution"].agg(
                    ["mean", "std"]
                )
                ax.errorbar(
                    grouped.index,
                    grouped["mean"],
                    yerr=grouped["std"],
                    fmt=f'{cfg["color"]}{cfg["marker"]}-',
                    label=cfg["label"],
                    capsize=3,
                    alpha=0.8,
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Training Tick", fontsize=11)
        ax.set_ylabel("Draft Contribution", fontsize=11)
        ax.set_title(
            f"Draft Contribution Over Training\n(Long Sequences >= {min_len})",
            fontsize=12,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        # Right: Acceptance rate trend
        ax = axes[1]
        for mode, cfg in MODE_CONFIG.items():
            subset = df[df["mode"] == mode]
            if len(subset) > 0:
                grouped = subset.groupby("sim_tick")["acceptance_rate"].agg(
                    ["mean", "std"]
                )
                ax.errorbar(
                    grouped.index,
                    grouped["mean"],
                    yerr=grouped["std"],
                    fmt=f'{cfg["color"]}{cfg["marker"]}-',
                    label=cfg["label"],
                    capsize=3,
                    alpha=0.8,
                    linewidth=2,
                    markersize=8,
                )

        ax.set_xlabel("Training Tick", fontsize=11)
        ax.set_ylabel("Acceptance Rate", fontsize=11)
        ax.set_title(
            "Acceptance Rate Over Training\n(Higher = Better Draft Quality)", fontsize=12
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        self._save_figure(fig, "draft_contribution_over_ticks")
        return self.config.output_dir / "draft_contribution_over_ticks.png"

    # =========================================================================
    # Figure 5: Hit vs Acceptance Trade-off
    # =========================================================================
    def _generate_hit_vs_acceptance_tradeoff(self) -> Path:
        """Generate scatter plot showing hit rate vs acceptance rate trade-off."""
        df = self.df_filtered
        min_len = self.config.min_response_len

        fig, ax = plt.subplots(figsize=self.config.figsize_single)

        for mode, cfg in MODE_CONFIG.items():
            subset = df[df["mode"] == mode]
            if len(subset) > 0:
                # Sample for scatter if too many points
                sample = subset.sample(min(200, len(subset)), random_state=42)
                ax.scatter(
                    sample["hit_rate"],
                    sample["acceptance_rate"],
                    c=cfg["color"],
                    marker=cfg["marker"],
                    label=cfg["label"],
                    alpha=0.5,
                    s=50,
                )

                # Add mean point with larger marker
                mean_hr = subset["hit_rate"].mean()
                mean_ar = subset["acceptance_rate"].mean()
                ax.scatter(
                    [mean_hr],
                    [mean_ar],
                    c=cfg["color"],
                    marker=cfg["marker"],
                    s=300,
                    edgecolors="black",
                    linewidths=2,
                    zorder=10,
                )
                ax.annotate(
                    f'{cfg["label"]}\n({mean_hr:.0%}, {mean_ar:.0%})',
                    xy=(mean_hr, mean_ar),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        ax.set_xlabel("Hit Rate (Cache Availability)", fontsize=12)
        ax.set_ylabel("Acceptance Rate (Draft Quality)", fontsize=12)
        ax.set_title(
            f"Hit Rate vs Acceptance Rate Trade-off\n(Long Sequences >= {min_len})",
            fontsize=14,
        )
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

        # Add quadrant labels
        ax.text(0.25, 0.85, "Low Hit, High Accept", ha="center", fontsize=9, alpha=0.7)
        ax.text(
            0.75,
            0.85,
            "High Hit, High Accept\n(BEST)",
            ha="center",
            fontsize=9,
            alpha=0.7,
            bbox=dict(facecolor="lightgreen", alpha=0.3),
        )
        ax.text(
            0.25, 0.15, "Low Hit, Low Accept\n(WORST)", ha="center", fontsize=9, alpha=0.7
        )
        ax.text(
            0.75,
            0.15,
            "High Hit, Low Accept\n(Stale patterns)",
            ha="center",
            fontsize=9,
            alpha=0.7,
        )

        plt.tight_layout()
        self._save_figure(fig, "hit_vs_acceptance_tradeoff")
        return self.config.output_dir / "hit_vs_acceptance_tradeoff.png"

    # =========================================================================
    # Figure 6: Metrics by Length
    # =========================================================================
    def _generate_metrics_by_length(self) -> Path:
        """Generate metrics by response length with 4K+ focus."""
        df = self.df.copy()

        # Add length bucket
        df["length_bucket"] = pd.cut(df["response_len"], bins=LENGTH_BINS, labels=LENGTH_LABELS)

        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize_double)

        metrics_by_len = [
            ("tokens_per_step", "Tokens per Step (E2E Speedup)", axes[0, 0]),
            ("hit_rate", "Hit Rate", axes[0, 1]),
            ("acceptance_rate", "Acceptance Rate", axes[1, 0]),
            ("draft_contribution", "Draft Contribution", axes[1, 1]),
        ]

        for metric, title, ax in metrics_by_len:
            for mode, cfg in MODE_CONFIG.items():
                subset = df[df["mode"] == mode]
                if len(subset) > 0:
                    grouped = subset.groupby("length_bucket", observed=True)[metric]
                    means = grouped.mean()
                    stds = grouped.std()

                    valid_labels = [l for l in LENGTH_LABELS if l in means.index]
                    x = [LENGTH_LABELS.index(l) for l in valid_labels]
                    y = [means[l] for l in valid_labels]
                    yerr = [stds[l] for l in valid_labels]

                    ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt=f'{cfg["color"]}{cfg["marker"]}-',
                        label=cfg["label"],
                        capsize=3,
                        alpha=0.8,
                        linewidth=2,
                    )

            ax.set_xticks(range(len(LENGTH_LABELS)))
            ax.set_xticklabels(LENGTH_LABELS, rotation=45)
            ax.set_xlabel("Response Length", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Highlight 4K+ region (starts after 2K-4K bucket at index 3)
            ax.axvspan(3.5, len(LENGTH_LABELS), alpha=0.1, color="yellow")

        plt.tight_layout()
        self._save_figure(fig, "metrics_by_length")
        return self.config.output_dir / "metrics_by_length.png"

    # =========================================================================
    # Figure 7: Long Sequence Heatmap
    # =========================================================================
    def _generate_long_seq_heatmap(self) -> Path:
        """Generate heatmap for long sequences showing speedup by tick and length."""
        df = self.df.copy()

        # Filter to long sequences
        df_long = df[df["response_len"] >= 4000].copy()

        if len(df_long) < 50:
            print(f"Not enough long sequences for heatmap: {len(df_long)}")
            # Create empty placeholder
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                f"Not enough data\n({len(df_long)} samples < 50 required)",
                ha="center",
                va="center",
                fontsize=14,
            )
            ax.axis("off")
            self._save_figure(fig, "long_seq_heatmap")
            return self.config.output_dir / "long_seq_heatmap.png"

        # Create length categories
        df_long["len_cat"] = pd.cut(
            df_long["response_len"], bins=LONG_SEQ_BINS, labels=LONG_SEQ_LABELS
        )

        # Count modes present
        modes_present = [
            m for m in ["prefill_only", "online_only", "prefill_plus_online"]
            if len(df_long[df_long["mode"] == m]) > 0
        ]
        n_modes = len(modes_present)

        if n_modes == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No data for any mode", ha="center", va="center", fontsize=14)
            ax.axis("off")
            self._save_figure(fig, "long_seq_heatmap")
            return self.config.output_dir / "long_seq_heatmap.png"

        fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 6))
        if n_modes == 1:
            axes = [axes]

        mode_titles = {
            "prefill_only": "PREFILL ONLY",
            "online_only": "ONLINE ONLY",
            "prefill_plus_online": "PREFILL + ONLINE",
        }

        for ax_idx, mode in enumerate(modes_present):
            subset = df_long[df_long["mode"] == mode]
            title = mode_titles[mode]

            pivot = subset.pivot_table(
                values="tokens_per_step",
                index="len_cat",
                columns="sim_tick",
                aggfunc="mean",
                observed=True,
            )

            if pivot.size > 0:
                im = axes[ax_idx].imshow(
                    pivot.values, aspect="auto", cmap="RdYlGn", vmin=1.0, vmax=3.5
                )
                axes[ax_idx].set_xticks(np.arange(len(pivot.columns)))
                axes[ax_idx].set_xticklabels(
                    [f"{int(t)}" for t in pivot.columns], rotation=45
                )
                axes[ax_idx].set_yticks(np.arange(len(pivot.index)))
                axes[ax_idx].set_yticklabels(pivot.index)
                axes[ax_idx].set_xlabel("Training Tick", fontsize=12)
                axes[ax_idx].set_ylabel("Response Length", fontsize=12)
                axes[ax_idx].set_title(
                    f"E2E Speedup: {title}\n(Long Sequences 4K+)", fontsize=12
                )

                # Add text annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            text_color = "white" if val < 1.8 else "black"
                            axes[ax_idx].text(
                                j,
                                i,
                                f"{val:.2f}x",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=10,
                                fontweight="bold",
                            )

                plt.colorbar(im, ax=axes[ax_idx], label="Tokens per Step (Speedup)")

        plt.tight_layout()
        self._save_figure(fig, "long_seq_heatmap")
        return self.config.output_dir / "long_seq_heatmap.png"

    # =========================================================================
    # Figure 8: Online Update Insight
    # =========================================================================
    def _generate_online_update_insight(self) -> Path:
        """Generate visual summary of key insights."""
        df = self.df_filtered
        mode_means = self._get_mode_means(df)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: The trade-off summary
        ax = axes[0]
        modes_list = ["prefill_only", "online_only", "prefill_plus_online"]
        mode_labels_short = ["Prefill\nOnly", "Online\nOnly", "Prefill +\nOnline"]

        hit_rates = [mode_means.get(m, {}).get("hr", 0) for m in modes_list]
        accept_rates = [mode_means.get(m, {}).get("ar", 0) for m in modes_list]

        x = np.arange(len(modes_list))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, hit_rates, width, label="Hit Rate", color="steelblue", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, accept_rates, width, label="Accept Rate", color="coral", alpha=0.8
        )

        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels_short)
        ax.set_ylabel("Rate", fontsize=11)
        ax.set_title("Hit Rate vs Acceptance Rate\n(The Trade-off)", fontsize=12)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars1, hit_rates):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.0%}",
                    ha="center",
                    fontsize=9,
                )
        for bar, val in zip(bars2, accept_rates):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.0%}",
                    ha="center",
                    fontsize=9,
                )

        # Panel 2: Speedup and draft contribution
        ax = axes[1]
        speedups = [mode_means.get(m, {}).get("tps", 1) for m in modes_list]
        draft_contribs = [mode_means.get(m, {}).get("dc", 0) for m in modes_list]

        ax.bar(
            x - width / 2,
            speedups,
            width,
            label="Tokens/Step (Speedup)",
            color="green",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            draft_contribs,
            width,
            label="Draft Contribution",
            color="purple",
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels_short)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title("Speedup & Draft Contribution\n(What Matters for E2E)", fontsize=12)
        ax.legend()
        ax.set_ylim(0, 3.0)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 3: Text summary of key insights
        ax = axes[2]
        ax.axis("off")

        prefill_hr = mode_means.get("prefill_only", {}).get("hr", 0)
        prefill_ar = mode_means.get("prefill_only", {}).get("ar", 0)
        prefill_tps = mode_means.get("prefill_only", {}).get("tps", 1)
        online_hr = mode_means.get("online_only", {}).get("hr", 0)
        online_ar = mode_means.get("online_only", {}).get("ar", 0)
        combined_tps = mode_means.get("prefill_plus_online", {}).get("tps", 1)

        insight_text = f"""
KEY INSIGHTS

1. Online Updates > Prefill for Long Sequences
   - Prefill: {prefill_hr:.0%} hit rate but {prefill_ar:.0%} acceptance
   - Online:  {online_hr:.0%} hit rate but {online_ar:.0%} acceptance

2. Why Acceptance Rate Matters More
   - High hit rate + low acceptance = wasted speculation
   - Prefill patterns from previous epoch (stale)
   - Online patterns from same response (fresh)

3. Self-Bootstrapping Effect
   - Online updates add tokens as generated
   - Later tokens match earlier patterns
   - High acceptance maintained throughout

4. Recommendation
   - Always enable online updates for long seqs
   - Combined: {combined_tps:.2f}x vs {prefill_tps:.2f}x (prefill only)
"""

        ax.text(
            0.05,
            0.95,
            insight_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        self._save_figure(fig, "online_update_insight")
        return self.config.output_dir / "online_update_insight.png"


def generate_figures_from_csv(
    data_csv: Path,
    output_dir: Path,
    figures: Optional[List[str]] = None,
    min_response_len: int = 4000,
) -> List[Path]:
    """
    Convenience function to generate figures from a CSV file.

    Args:
        data_csv: Path to per_request_data.csv.
        output_dir: Directory to save figures.
        figures: List of figure names to generate (None = all).
        min_response_len: Minimum response length for filtering.

    Returns:
        List of paths to generated figures.
    """
    config = FigureConfig(
        data_csv=data_csv,
        output_dir=output_dir,
        figures=figures,
        min_response_len=min_response_len,
    )
    generator = FigureGenerator(config)
    return generator.generate_all()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python figure_generator.py <data_csv> [output_dir]")
        sys.exit(1)

    data_csv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else data_csv.parent / "figures"

    paths = generate_figures_from_csv(data_csv, output_dir)
    print(f"\nGenerated {len(paths)} figures in {output_dir}")
