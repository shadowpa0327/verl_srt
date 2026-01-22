#!/usr/bin/env python3
"""
Data Discovery Utilities for SRT Speculation Analysis.

This module provides utilities to auto-detect and validate rollout data directories,
including finding available ticks, checking data integrity, and providing structured
information about the data layout.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DataDirectoryInfo:
    """Information about a rollout data directory."""

    path: Path
    rollout_ticks: List[int] = field(default_factory=list)
    secondary_ticks: List[int] = field(default_factory=list)
    min_tick: int = 0
    max_tick: int = 0
    has_rollout: bool = False
    has_secondary: bool = False
    valid: bool = False
    error: Optional[str] = None

    @property
    def total_ticks(self) -> int:
        """Total number of unique ticks available."""
        return len(set(self.rollout_ticks) | set(self.secondary_ticks))

    @property
    def common_ticks(self) -> List[int]:
        """Ticks that exist in both rollout and secondary."""
        return sorted(set(self.rollout_ticks) & set(self.secondary_ticks))

    @property
    def tick_range(self) -> Tuple[int, int]:
        """Return (min_tick, max_tick) tuple."""
        return (self.min_tick, self.max_tick)

    def has_tick_pair(self, cache_tick: int, sim_tick: int) -> bool:
        """Check if we have data for a cache_tick -> sim_tick simulation."""
        return cache_tick in self.secondary_ticks and sim_tick in self.rollout_ticks

    def get_valid_tick_pairs(self, tick_step: int = 1) -> List[Tuple[int, int]]:
        """
        Get list of valid (cache_tick, sim_tick) pairs.

        Args:
            tick_step: Step between ticks to sample.

        Returns:
            List of (cache_tick, sim_tick) tuples where:
            - cache_tick exists in secondary/
            - sim_tick = cache_tick + 1 exists in rollout/
        """
        pairs = []
        for cache_tick in sorted(self.secondary_ticks):
            sim_tick = cache_tick + 1
            if sim_tick in self.rollout_ticks:
                pairs.append((cache_tick, sim_tick))

        # Apply step filter
        if tick_step > 1:
            pairs = pairs[::tick_step]

        return pairs

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Data Directory: {self.path}",
            f"  Valid: {self.valid}",
        ]

        if self.error:
            lines.append(f"  Error: {self.error}")
            return "\n".join(lines)

        lines.extend([
            f"  Rollout ticks: {len(self.rollout_ticks)} files "
            f"({self.min_tick}-{self.max_tick})" if self.rollout_ticks else "  Rollout: NOT FOUND",
            f"  Secondary ticks: {len(self.secondary_ticks)} files" if self.secondary_ticks else "  Secondary: NOT FOUND",
        ])

        if self.rollout_ticks:
            # Show first and last few ticks
            if len(self.rollout_ticks) <= 10:
                lines.append(f"  Rollout tick list: {self.rollout_ticks}")
            else:
                lines.append(f"  Rollout tick list: {self.rollout_ticks[:3]}...{self.rollout_ticks[-3:]}")

        if self.secondary_ticks:
            if len(self.secondary_ticks) <= 10:
                lines.append(f"  Secondary tick list: {self.secondary_ticks}")
            else:
                lines.append(f"  Secondary tick list: {self.secondary_ticks[:3]}...{self.secondary_ticks[-3:]}")

        # Check for gaps
        if self.rollout_ticks:
            expected = set(range(self.min_tick, self.max_tick + 1))
            actual = set(self.rollout_ticks)
            gaps = expected - actual
            if gaps and len(gaps) < 20:
                lines.append(f"  Rollout gaps: {sorted(gaps)}")
            elif gaps:
                lines.append(f"  Rollout gaps: {len(gaps)} missing ticks")

        # Valid tick pairs
        valid_pairs = self.get_valid_tick_pairs()
        if valid_pairs:
            lines.append(f"  Valid tick pairs for simulation: {len(valid_pairs)}")
            if len(valid_pairs) <= 5:
                lines.append(f"    Pairs: {valid_pairs}")
            else:
                lines.append(f"    Range: {valid_pairs[0]} to {valid_pairs[-1]}")

        return "\n".join(lines)


def _extract_tick_number(filename: str) -> Optional[int]:
    """
    Extract tick number from a filename like '5.jsonl' or 'tick_5.jsonl'.

    Args:
        filename: Filename to parse.

    Returns:
        Integer tick number or None if not parseable.
    """
    # Try pattern: just number.jsonl
    match = re.match(r'^(\d+)\.jsonl$', filename)
    if match:
        return int(match.group(1))

    # Try pattern: tick_N.jsonl or similar
    match = re.match(r'^(?:tick[_-]?)?(\d+)\.jsonl$', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _scan_tick_directory(dir_path: Path) -> List[int]:
    """
    Scan a directory for tick files and return sorted list of tick numbers.

    Args:
        dir_path: Path to directory containing tick files.

    Returns:
        Sorted list of tick numbers found.
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []

    ticks = []
    for entry in dir_path.iterdir():
        if entry.is_file() and entry.suffix == '.jsonl':
            tick_num = _extract_tick_number(entry.name)
            if tick_num is not None:
                ticks.append(tick_num)

    return sorted(ticks)


def discover_data_directory(path: str | Path) -> DataDirectoryInfo:
    """
    Scan a data directory and return structured information about available data.

    Expected directory structure:
        data_dir/
        ├── rollout/
        │   ├── 1.jsonl
        │   ├── 2.jsonl
        │   └── ...
        └── secondary/
            ├── 1.jsonl
            ├── 2.jsonl
            └── ...

    Args:
        path: Path to the data directory.

    Returns:
        DataDirectoryInfo with discovered information.
    """
    path = Path(path)
    info = DataDirectoryInfo(path=path)

    # Check base directory exists
    if not path.exists():
        info.error = f"Directory does not exist: {path}"
        return info

    if not path.is_dir():
        info.error = f"Not a directory: {path}"
        return info

    # Scan rollout directory
    rollout_dir = path / "rollout"
    if rollout_dir.exists():
        info.rollout_ticks = _scan_tick_directory(rollout_dir)
        info.has_rollout = len(info.rollout_ticks) > 0

    # Scan secondary directory
    secondary_dir = path / "secondary"
    if secondary_dir.exists():
        info.secondary_ticks = _scan_tick_directory(secondary_dir)
        info.has_secondary = len(info.secondary_ticks) > 0

    # Compute min/max tick across both
    all_ticks = info.rollout_ticks + info.secondary_ticks
    if all_ticks:
        info.min_tick = min(all_ticks)
        info.max_tick = max(all_ticks)
        info.valid = True
    else:
        info.error = "No tick files found in rollout/ or secondary/"

    return info


def validate_tick_range(
    info: DataDirectoryInfo,
    tick_start: Optional[int] = None,
    tick_end: Optional[int] = None,
    tick_step: int = 1,
) -> Tuple[int, int, List[str]]:
    """
    Validate and adjust tick range based on available data.

    Args:
        info: DataDirectoryInfo from discover_data_directory().
        tick_start: Requested start tick (None = auto-detect).
        tick_end: Requested end tick (None = auto-detect).
        tick_step: Step between ticks.

    Returns:
        Tuple of (adjusted_start, adjusted_end, warnings).
    """
    warnings = []

    if not info.valid:
        return (0, 0, [f"Invalid data directory: {info.error}"])

    # Auto-detect start
    if tick_start is None:
        tick_start = info.min_tick
    elif tick_start < info.min_tick:
        warnings.append(f"tick_start ({tick_start}) < min available ({info.min_tick}), using {info.min_tick}")
        tick_start = info.min_tick

    # Auto-detect end
    if tick_end is None:
        tick_end = info.max_tick
    elif tick_end > info.max_tick:
        warnings.append(f"tick_end ({tick_end}) > max available ({info.max_tick}), using {info.max_tick}")
        tick_end = info.max_tick

    # Validate range
    if tick_start >= tick_end:
        warnings.append(f"tick_start ({tick_start}) >= tick_end ({tick_end})")
        tick_end = tick_start + tick_step

    return (tick_start, tick_end, warnings)


def get_sample_data(
    info: DataDirectoryInfo,
    tick: int,
    source: str = "rollout",
    max_samples: int = 5,
) -> List[Dict]:
    """
    Load sample data from a specific tick for preview.

    Args:
        info: DataDirectoryInfo.
        tick: Tick number to load.
        source: "rollout" or "secondary".
        max_samples: Maximum samples to return.

    Returns:
        List of sample records.
    """
    data_path = info.path / source / f"{tick}.jsonl"

    if not data_path.exists():
        return []

    samples = []
    try:
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    except Exception:
        pass

    return samples


def count_samples_in_tick(info: DataDirectoryInfo, tick: int, source: str = "rollout") -> int:
    """
    Count the number of samples in a tick file.

    Args:
        info: DataDirectoryInfo.
        tick: Tick number.
        source: "rollout" or "secondary".

    Returns:
        Number of samples (lines) in the file.
    """
    data_path = info.path / source / f"{tick}.jsonl"

    if not data_path.exists():
        return 0

    try:
        with open(data_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def analyze_response_lengths(
    info: DataDirectoryInfo,
    tick: int,
    source: str = "rollout",
    max_samples: int = 1000,
) -> Dict[str, float]:
    """
    Analyze response length distribution for a tick.

    Args:
        info: DataDirectoryInfo.
        tick: Tick number.
        source: "rollout" or "secondary".
        max_samples: Maximum samples to analyze.

    Returns:
        Dictionary with length statistics.
    """
    data_path = info.path / source / f"{tick}.jsonl"

    if not data_path.exists():
        return {}

    lengths = []
    try:
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                line = line.strip()
                if line:
                    record = json.loads(line)
                    # Handle both rollout (input/output) and secondary (prompt/response) formats
                    response = record.get('output') or record.get('response', '')
                    # Rough token estimate (chars / 4)
                    lengths.append(len(response) // 4)
    except Exception:
        return {}

    if not lengths:
        return {}

    import numpy as np
    lengths_arr = np.array(lengths)

    return {
        "count": len(lengths),
        "mean": float(np.mean(lengths_arr)),
        "median": float(np.median(lengths_arr)),
        "std": float(np.std(lengths_arr)),
        "min": int(np.min(lengths_arr)),
        "max": int(np.max(lengths_arr)),
        "p25": float(np.percentile(lengths_arr, 25)),
        "p75": float(np.percentile(lengths_arr, 75)),
        "p95": float(np.percentile(lengths_arr, 95)),
        "pct_over_4k": float(np.mean(lengths_arr >= 4000) * 100),
    }


if __name__ == "__main__":
    # Quick test
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/home/ubuntu/verl_srt/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead"

    info = discover_data_directory(data_dir)
    print(info.summary())

    if info.valid and info.rollout_ticks:
        print("\n--- Sample Response Length Analysis ---")
        tick = info.rollout_ticks[0]
        stats = analyze_response_lengths(info, tick)
        if stats:
            print(f"Tick {tick} response lengths (estimated tokens):")
            for k, v in stats.items():
                print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
