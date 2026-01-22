#!/usr/bin/env python3
"""
Configuration dataclasses for SRT Speculation Analysis.

This module defines configuration objects for various analysis tasks including
simulation sweeps, figure generation, and report generation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AnalysisConfig:
    """
    Main configuration for SRT speculation analysis.

    This configuration is used by both sweep_runner and figure_generator.
    """

    # Required paths
    data_dir: Path
    output_dir: Path

    # Model for tokenization
    model_path: str = "Qwen/Qwen2.5-7B"

    # Tick range (None = auto-detect from data)
    tick_start: Optional[int] = None
    tick_end: Optional[int] = None
    tick_step: int = 5

    # Simulation modes to run
    run_prefill_only: bool = True  # Prefill from secondary, no online updates
    run_online_only: bool = True  # Online updates only, no prefill
    run_prefill_plus_online: bool = True  # Both prefill and online updates

    # Response length filtering
    min_response_len: int = 0  # Filter for sequences >= this length (0 = no filter)

    # Simulation parameters
    min_token_prob: float = 0.3  # Minimum probability for draft tokens
    hash_token_count: int = 128  # Tokens to hash for tree sharing

    # Cache parameters
    max_tree_depth: int = 64
    spec_prefix_len: int = 7
    spec_max_len: int = 16

    # Limits
    max_samples: int = 0  # 0 = all samples

    # Output options
    verbose: bool = False
    save_per_request: bool = True  # Save per-request data to CSV

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @property
    def modes_to_run(self) -> List[str]:
        """Get list of mode names to run."""
        modes = []
        if self.run_prefill_only:
            modes.append("prefill_only")
        if self.run_online_only:
            modes.append("online_only")
        if self.run_prefill_plus_online:
            modes.append("prefill_plus_online")
        return modes


@dataclass
class FigureConfig:
    """Configuration for figure generation."""

    # Input data
    data_csv: Optional[Path] = None  # Path to per_request_data.csv
    summary_json: Optional[Path] = None  # Path to sweep_summary.json

    # Output
    output_dir: Path = Path("./figures")

    # Figure selection (if None, generate all)
    figures: Optional[List[str]] = None

    # Display options
    min_response_len: int = 4000  # Focus on long sequences
    dpi: int = 150
    format: str = "png"

    # Style options
    figsize_single: tuple = (10, 6)
    figsize_double: tuple = (14, 10)
    figsize_triple: tuple = (18, 6)

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_csv, str):
            self.data_csv = Path(self.data_csv)
        if isinstance(self.summary_json, str):
            self.summary_json = Path(self.summary_json)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def available_figures(cls) -> List[str]:
        """List of available figure names."""
        return [
            "three_mode_comparison",  # 4 metrics over training ticks
            "three_mode_bars",  # Bar chart comparing 3 modes
            "speedup_decomposition",  # Speedup contribution breakdown
            "draft_contribution_over_ticks",  # Draft contrib + accept rate trends
            "hit_vs_acceptance_tradeoff",  # Scatter showing the trade-off
            "metrics_by_length",  # All metrics by response length
            "long_seq_heatmap",  # Heatmap (tick x length)
            "online_update_insight",  # Key insights summary
        ]


@dataclass
class SweepConfig:
    """Configuration specifically for running simulation sweeps."""

    # Required
    data_dir: Path
    output_dir: Path
    model_path: str = "Qwen/Qwen2.5-7B"

    # Tick range
    tick_start: Optional[int] = None
    tick_end: Optional[int] = None
    tick_step: int = 5

    # Modes
    run_prefill_only: bool = True
    run_online_only: bool = True
    run_prefill_plus_online: bool = True

    # Simulation parameters
    min_token_prob: float = 0.3
    hash_token_count: int = 128
    max_tree_depth: int = 64
    spec_prefix_len: int = 7
    spec_max_len: int = 16

    # Limits
    max_samples: int = 0
    verbose: bool = False

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_analysis_config(cls, config: AnalysisConfig) -> "SweepConfig":
        """Create SweepConfig from AnalysisConfig."""
        return cls(
            data_dir=config.data_dir,
            output_dir=config.output_dir,
            model_path=config.model_path,
            tick_start=config.tick_start,
            tick_end=config.tick_end,
            tick_step=config.tick_step,
            run_prefill_only=config.run_prefill_only,
            run_online_only=config.run_online_only,
            run_prefill_plus_online=config.run_prefill_plus_online,
            min_token_prob=config.min_token_prob,
            hash_token_count=config.hash_token_count,
            max_tree_depth=config.max_tree_depth,
            spec_prefix_len=config.spec_prefix_len,
            spec_max_len=config.spec_max_len,
            max_samples=config.max_samples,
            verbose=config.verbose,
        )


@dataclass
class ReportConfig:
    """Configuration for generating analysis reports."""

    # Input data
    data_csv: Path
    summary_json: Optional[Path] = None

    # Output
    output_path: Path = Path("./ANALYSIS_REPORT.md")

    # Content options
    include_figures: bool = True
    include_tables: bool = True
    include_recommendations: bool = True

    # Focus
    min_response_len: int = 4000

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_csv, str):
            self.data_csv = Path(self.data_csv)
        if isinstance(self.summary_json, str):
            self.summary_json = Path(self.summary_json)
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)


# Mode configuration for plotting
MODE_CONFIG = {
    "prefill_only": {
        "color": "r",
        "marker": "s",
        "label": "Prefill Only",
        "description": "Cache pre-populated from secondary outputs, no online updates",
    },
    "online_only": {
        "color": "g",
        "marker": "^",
        "label": "Online Only",
        "description": "Cache starts empty, only online updates during generation",
    },
    "prefill_plus_online": {
        "color": "b",
        "marker": "o",
        "label": "Prefill + Online",
        "description": "Both prefill and online updates (best of both)",
    },
}

# Metric definitions for labels and descriptions
METRIC_DEFINITIONS = {
    "tokens_per_step": {
        "label": "Tokens per Step (E2E Speedup)",
        "short_label": "Toks/Step",
        "description": "Average tokens produced per model forward pass",
    },
    "hit_rate": {
        "label": "Hit Rate",
        "short_label": "Hit Rate",
        "description": "Fraction of steps where cache returned draft tokens",
    },
    "acceptance_rate": {
        "label": "Acceptance Rate",
        "short_label": "Accept Rate",
        "description": "Fraction of draft tokens that matched ground truth",
    },
    "tokens_per_hit_step": {
        "label": "Tokens per Hit Step (Ceiling)",
        "short_label": "Toks/Hit",
        "description": "Tokens per step, counting only steps with cache hits",
    },
    "draft_contribution": {
        "label": "Draft Contribution",
        "short_label": "Draft Contrib",
        "description": "Fraction of output tokens from accepted drafts",
    },
}

# Length bin definitions
LENGTH_BINS = [0, 500, 1000, 2000, 4000, 8000, 20000]
LENGTH_LABELS = ["0-500", "500-1K", "1K-2K", "2K-4K", "4K-8K", "8K+"]

# Long sequence bins
LONG_SEQ_BINS = [4000, 6000, 8000, 12000, 20000]
LONG_SEQ_LABELS = ["4K-6K", "6K-8K", "8K-12K", "12K+"]
