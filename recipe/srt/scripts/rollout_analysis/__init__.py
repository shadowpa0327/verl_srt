# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Rollout data analysis scripts for SRT (Speculative Rollout with Tree-Structured Cache).

This package provides tools to analyze dumped rollout data:
- organize_by_prompt.py: Organize rollout data by prompt
- analyze_lengths.py: Analyze output token lengths and accuracy correlation
- analyze_runahead_prediction.py: Analyze secondary[N] -> primary[N+1] prediction

Usage:
    python -m recipe.srt.scripts.rollout_analysis.run_all --data_dir /path/to/data
"""

from .organize_by_prompt import organize_by_step, organize_by_prompt
from .analyze_lengths import analyze_primary_lengths, analyze_secondary_lengths
from .analyze_runahead_prediction import analyze_runahead_correlation

__all__ = [
    'organize_by_step',
    'organize_by_prompt',
    'analyze_primary_lengths',
    'analyze_secondary_lengths',
    'analyze_runahead_correlation',
]
