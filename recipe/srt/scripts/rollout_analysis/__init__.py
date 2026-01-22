# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Rollout data analysis scripts for SRT (Speculative Rollout with Tree-Structured Cache).

This package provides tools to analyze dumped rollout data:
- analyze_lengths.py: Analyze output token lengths and accuracy correlation
- analyze_runahead_prediction.py: Analyze secondary[N] -> primary[N+1] prediction

CLI Tool (srt_analyze):
    # Show info about a data directory
    python -m recipe.srt.scripts.rollout_analysis.srt_analyze info /path/to/data

    # Run full analysis with auto-detection
    python -m recipe.srt.scripts.rollout_analysis.srt_analyze full /path/to/data -o ./results

    # Run sweep with custom tick range
    python -m recipe.srt.scripts.rollout_analysis.srt_analyze sweep /path/to/data --tick-start 1 --tick-end 50

    # Generate figures from existing CSV
    python -m recipe.srt.scripts.rollout_analysis.srt_analyze plot --data ./results/per_request_data.csv
"""

from .analyze_lengths import analyze_primary_lengths, analyze_secondary_lengths
from .analyze_runahead_prediction import analyze_runahead_correlation

# Core analysis modules
from .data_discovery import discover_data_directory, DataDirectoryInfo
from .analysis_config import AnalysisConfig, SweepConfig, FigureConfig
from .figure_generator import FigureGenerator, generate_figures_from_csv
from .sweep_runner import SweepRunner, run_sweep, SweepResults

__all__ = [
    # Research script exports
    'analyze_primary_lengths',
    'analyze_secondary_lengths',
    'analyze_runahead_correlation',
    # Core exports
    'discover_data_directory',
    'DataDirectoryInfo',
    'AnalysisConfig',
    'SweepConfig',
    'FigureConfig',
    'FigureGenerator',
    'generate_figures_from_csv',
    'SweepRunner',
    'run_sweep',
    'SweepResults',
]
