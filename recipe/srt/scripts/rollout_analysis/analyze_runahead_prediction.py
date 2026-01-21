#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Analyze runahead prediction: Can secondary[N] predict primary[N+1] output lengths?

This script analyzes:
1. Correlation between secondary and primary output lengths (same prompt)
2. Correlation between secondary[N] and primary[N+1] (next step prediction)
3. Quartile prediction accuracy
4. Long output prediction precision/recall

Usage:
    python analyze_runahead_prediction.py \\
        --data_dir /path/to/rollout_data \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --output_json results.json
"""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (first 16 hex chars)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def load_data_by_step(
    data_dir: Path,
    tokenizer,
    verbose: bool = False
) -> Tuple[Dict[int, Dict[str, List[int]]], Dict[int, Dict[str, List[int]]]]:
    """Load primary and secondary data organized by step.

    Returns:
        Tuple of (primary_by_step, secondary_by_step)
        Each is: step -> prompt_hash -> list of token lengths
    """
    primary_by_step = defaultdict(lambda: defaultdict(list))
    secondary_by_step = defaultdict(lambda: defaultdict(list))

    # Load primary
    rollout_dir = data_dir / "rollout"
    count = 0
    for step_file in sorted(rollout_dir.glob("*.jsonl"), key=lambda x: int(x.stem)):
        step = int(step_file.stem)
        with open(step_file) as f:
            for line in f:
                item = json.loads(line)
                prompt_hash = compute_prompt_hash(item['input'])
                tokens = tokenizer.encode(item['output'], add_special_tokens=False)
                primary_by_step[step][prompt_hash].append(len(tokens))
                count += 1
                if verbose and count % 5000 == 0:
                    print(f"  Primary: {count} samples...")

    if verbose:
        print(f"  Primary: {count} total samples, {len(primary_by_step)} steps")

    # Load secondary
    secondary_dir = data_dir / "secondary"
    count = 0
    skipped = 0
    for step_file in sorted(secondary_dir.glob("*.jsonl"), key=lambda x: int(x.stem)):
        step = int(step_file.stem)
        with open(step_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get('status') == 'rejected' or 'prompt' not in item:
                    skipped += 1
                    continue
                prompt_hash = compute_prompt_hash(item['prompt'])
                tokens = tokenizer.encode(item['response'], add_special_tokens=False)
                secondary_by_step[step][prompt_hash].append(len(tokens))
                count += 1
                if verbose and count % 5000 == 0:
                    print(f"  Secondary: {count} samples...")

    if verbose:
        print(f"  Secondary: {count} total samples, {len(secondary_by_step)} steps (skipped {skipped})")

    return dict(primary_by_step), dict(secondary_by_step)


def calculate_correlation(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate Pearson and Spearman correlation coefficients.

    Args:
        pairs: List of (x, y) tuples

    Returns:
        Tuple of (pearson, spearman)
    """
    if len(pairs) < 3:
        return 0.0, 0.0

    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]

    # Pearson
    x_mean = statistics.mean(x_vals)
    y_mean = statistics.mean(y_vals)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    x_std = (sum((x - x_mean)**2 for x in x_vals)) ** 0.5
    y_std = (sum((y - y_mean)**2 for y in y_vals)) ** 0.5

    pearson = numerator / (x_std * y_std) if x_std * y_std > 0 else 0

    # Spearman
    n = len(pairs)
    x_sorted = sorted(enumerate(pairs), key=lambda p: p[1][0])
    y_sorted = sorted(enumerate(pairs), key=lambda p: p[1][1])

    x_ranks = {idx: rank for rank, (idx, _) in enumerate(x_sorted)}
    y_ranks = {idx: rank for rank, (idx, _) in enumerate(y_sorted)}

    d_squared_sum = sum((x_ranks[i] - y_ranks[i])**2 for i in range(n))
    spearman = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))

    return pearson, spearman


def analyze_same_step_correlation(
    primary_by_step: Dict[int, Dict[str, List[int]]],
    secondary_by_step: Dict[int, Dict[str, List[int]]],
    min_samples: int = 8
) -> Dict:
    """Analyze correlation between secondary and primary for same prompts (any step)."""

    # Aggregate by prompt across all steps
    primary_agg = defaultdict(list)
    secondary_agg = defaultdict(list)

    for step, prompts in primary_by_step.items():
        for prompt_hash, lengths in prompts.items():
            primary_agg[prompt_hash].extend(lengths)

    for step, prompts in secondary_by_step.items():
        for prompt_hash, lengths in prompts.items():
            secondary_agg[prompt_hash].extend(lengths)

    # Find common prompts
    common = set(primary_agg.keys()) & set(secondary_agg.keys())

    # Build paired data
    paired_data = []
    for prompt_hash in common:
        p_lens = primary_agg[prompt_hash]
        s_lens = secondary_agg[prompt_hash]

        if len(p_lens) >= min_samples and len(s_lens) >= min_samples:
            paired_data.append({
                'hash': prompt_hash,
                'primary_mean': statistics.mean(p_lens),
                'secondary_mean': statistics.mean(s_lens),
            })

    if not paired_data:
        return {'error': 'No common prompts with sufficient samples'}

    # Calculate correlation
    pairs = [(d['secondary_mean'], d['primary_mean']) for d in paired_data]
    pearson, spearman = calculate_correlation(pairs)

    # Quartile analysis
    paired_data.sort(key=lambda x: x['secondary_mean'])
    q_size = len(paired_data) // 4

    paired_sorted_primary = sorted(paired_data, key=lambda x: x['primary_mean'])
    primary_quartile = {}
    for i, d in enumerate(paired_sorted_primary):
        if i < q_size:
            primary_quartile[d['hash']] = 'Q1'
        elif i < 2*q_size:
            primary_quartile[d['hash']] = 'Q2'
        elif i < 3*q_size:
            primary_quartile[d['hash']] = 'Q3'
        else:
            primary_quartile[d['hash']] = 'Q4'

    quartile_accuracy = {}
    for q_idx, q_name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        q_prompts = paired_data[q_idx*q_size:(q_idx+1)*q_size]
        same_q = sum(1 for p in q_prompts if primary_quartile.get(p['hash']) == q_name)
        quartile_accuracy[q_name] = {
            'count': len(q_prompts),
            'match_rate': same_q / len(q_prompts) if q_prompts else 0,
            'secondary_mean': statistics.mean([p['secondary_mean'] for p in q_prompts]) if q_prompts else 0,
            'primary_mean': statistics.mean([p['primary_mean'] for p in q_prompts]) if q_prompts else 0,
        }

    # Prediction accuracy
    q4_sec = set(p['hash'] for p in paired_data[3*q_size:])
    q4_pri = set(p['hash'] for p in paired_sorted_primary[3*q_size:])
    q4_overlap = q4_sec & q4_pri

    top10_size = len(paired_data) // 10
    top10_sec = set(p['hash'] for p in paired_data[-top10_size:])
    top10_pri = set(p['hash'] for p in paired_sorted_primary[-top10_size:])
    top10_overlap = top10_sec & top10_pri

    prediction_accuracy = {
        'top25': {
            'precision': len(q4_overlap) / len(q4_sec) if q4_sec else 0,
            'recall': len(q4_overlap) / len(q4_pri) if q4_pri else 0,
            'overlap': len(q4_overlap),
            'predicted': len(q4_sec),
            'actual': len(q4_pri),
        },
        'top10': {
            'precision': len(top10_overlap) / len(top10_sec) if top10_sec else 0,
            'recall': len(top10_overlap) / len(top10_pri) if top10_pri else 0,
            'overlap': len(top10_overlap),
            'predicted': len(top10_sec),
            'actual': len(top10_pri),
        },
    }

    return {
        'common_prompts': len(paired_data),
        'pearson_correlation': pearson,
        'spearman_correlation': spearman,
        'quartile_accuracy': quartile_accuracy,
        'prediction_accuracy': prediction_accuracy,
    }


def analyze_runahead_correlation(
    primary_by_step: Dict[int, Dict[str, List[int]]],
    secondary_by_step: Dict[int, Dict[str, List[int]]],
    min_samples: int = 4
) -> Dict:
    """Analyze correlation between secondary[N] and primary[N+1]."""

    step_correlations = []
    all_paired = []

    for sec_step in sorted(secondary_by_step.keys()):
        pri_step = sec_step + 1

        if pri_step not in primary_by_step:
            continue

        sec_prompts = set(secondary_by_step[sec_step].keys())
        pri_prompts = set(primary_by_step[pri_step].keys())
        common = sec_prompts & pri_prompts

        if len(common) < 10:
            continue

        # Build paired data for this step
        paired = []
        for prompt_hash in common:
            sec_lens = secondary_by_step[sec_step][prompt_hash]
            pri_lens = primary_by_step[pri_step][prompt_hash]

            if len(sec_lens) >= min_samples and len(pri_lens) >= min_samples:
                sec_mean = statistics.mean(sec_lens)
                pri_mean = statistics.mean(pri_lens)
                paired.append((sec_mean, pri_mean))
                all_paired.append({
                    'sec_step': sec_step,
                    'pri_step': pri_step,
                    'hash': prompt_hash,
                    'secondary_mean': sec_mean,
                    'primary_mean': pri_mean,
                })

        if len(paired) < 10:
            continue

        pearson, spearman = calculate_correlation(paired)

        step_correlations.append({
            'sec_step': sec_step,
            'pri_step': pri_step,
            'common_prompts': len(common),
            'paired_prompts': len(paired),
            'pearson': pearson,
            'secondary_mean': statistics.mean([p[0] for p in paired]),
            'primary_mean': statistics.mean([p[1] for p in paired]),
        })

    if not all_paired:
        return {'error': 'No common prompts between consecutive steps'}

    # Overall correlation
    pairs = [(d['secondary_mean'], d['primary_mean']) for d in all_paired]
    overall_pearson, overall_spearman = calculate_correlation(pairs)

    # Quartile analysis
    all_paired_sorted_sec = sorted(all_paired, key=lambda x: x['secondary_mean'])
    all_paired_sorted_pri = sorted(all_paired, key=lambda x: x['primary_mean'])

    q_size = len(all_paired) // 4

    primary_quartile = {}
    for i, d in enumerate(all_paired_sorted_pri):
        key = (d['sec_step'], d['hash'])
        if i < q_size:
            primary_quartile[key] = 'Q1'
        elif i < 2*q_size:
            primary_quartile[key] = 'Q2'
        elif i < 3*q_size:
            primary_quartile[key] = 'Q3'
        else:
            primary_quartile[key] = 'Q4'

    quartile_accuracy = {}
    for q_idx, q_name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
        q_data = all_paired_sorted_sec[q_idx*q_size:(q_idx+1)*q_size]
        same_q = sum(1 for d in q_data if primary_quartile.get((d['sec_step'], d['hash'])) == q_name)
        quartile_accuracy[q_name] = {
            'count': len(q_data),
            'match_rate': same_q / len(q_data) if q_data else 0,
            'secondary_mean': statistics.mean([d['secondary_mean'] for d in q_data]) if q_data else 0,
            'primary_mean': statistics.mean([d['primary_mean'] for d in q_data]) if q_data else 0,
        }

    # Prediction accuracy
    q4_sec = set((d['sec_step'], d['hash']) for d in all_paired_sorted_sec[3*q_size:])
    q4_pri = set((d['sec_step'], d['hash']) for d in all_paired_sorted_pri[3*q_size:])
    q4_overlap = q4_sec & q4_pri

    top10_size = len(all_paired) // 10
    top10_sec = set((d['sec_step'], d['hash']) for d in all_paired_sorted_sec[-top10_size:])
    top10_pri = set((d['sec_step'], d['hash']) for d in all_paired_sorted_pri[-top10_size:])
    top10_overlap = top10_sec & top10_pri

    prediction_accuracy = {
        'top25': {
            'precision': len(q4_overlap) / len(q4_sec) if q4_sec else 0,
            'recall': len(q4_overlap) / len(q4_pri) if q4_pri else 0,
            'overlap': len(q4_overlap),
            'predicted': len(q4_sec),
            'actual': len(q4_pri),
        },
        'top10': {
            'precision': len(top10_overlap) / len(top10_sec) if top10_sec else 0,
            'recall': len(top10_overlap) / len(top10_pri) if top10_pri else 0,
            'overlap': len(top10_overlap),
            'predicted': len(top10_sec),
            'actual': len(top10_pri),
        },
    }

    # Per-step correlation statistics
    correlations = [sc['pearson'] for sc in step_correlations]
    step_correlation_stats = {
        'min': min(correlations),
        'median': statistics.median(correlations),
        'mean': statistics.mean(correlations),
        'max': max(correlations),
    }

    # Correlation by training phase
    early = [sc['pearson'] for sc in step_correlations if sc['sec_step'] <= 20]
    mid = [sc['pearson'] for sc in step_correlations if 20 < sc['sec_step'] <= 45]
    late = [sc['pearson'] for sc in step_correlations if sc['sec_step'] > 45]

    phase_correlations = {}
    if early:
        phase_correlations['early'] = {'mean': statistics.mean(early), 'median': statistics.median(early)}
    if mid:
        phase_correlations['middle'] = {'mean': statistics.mean(mid), 'median': statistics.median(mid)}
    if late:
        phase_correlations['late'] = {'mean': statistics.mean(late), 'median': statistics.median(late)}

    return {
        'total_paired': len(all_paired),
        'pearson_correlation': overall_pearson,
        'spearman_correlation': overall_spearman,
        'quartile_accuracy': quartile_accuracy,
        'prediction_accuracy': prediction_accuracy,
        'step_correlations': step_correlations,
        'step_correlation_stats': step_correlation_stats,
        'phase_correlations': phase_correlations,
    }


def print_results(same_step: Dict, runahead: Dict):
    """Print analysis results to console."""

    print("=" * 80)
    print("SAME-STEP CORRELATION: SECONDARY vs PRIMARY (same prompts)")
    print("=" * 80)

    if 'error' not in same_step:
        print(f"\nCommon prompts: {same_step['common_prompts']}")
        print(f"Pearson correlation:  {same_step['pearson_correlation']:.4f}")
        print(f"Spearman correlation: {same_step['spearman_correlation']:.4f}")

        print("\nQuartile prediction accuracy:")
        for q_name, q_data in same_step['quartile_accuracy'].items():
            print(f"  {q_name}: {q_data['match_rate']*100:.1f}% match "
                  f"(sec={q_data['secondary_mean']:.0f}, pri={q_data['primary_mean']:.0f})")

        print("\nLong output prediction:")
        for level, data in same_step['prediction_accuracy'].items():
            print(f"  {level}: precision={data['precision']*100:.1f}%, "
                  f"recall={data['recall']*100:.1f}% ({data['overlap']}/{data['predicted']})")

    print("\n" + "=" * 80)
    print("RUNAHEAD PREDICTION: SECONDARY[N] -> PRIMARY[N+1]")
    print("=" * 80)

    if 'error' not in runahead:
        print(f"\nTotal paired prompt-step combinations: {runahead['total_paired']}")
        print(f"Pearson correlation:  {runahead['pearson_correlation']:.4f}")
        print(f"Spearman correlation: {runahead['spearman_correlation']:.4f}")

        print("\nQuartile prediction accuracy:")
        for q_name, q_data in runahead['quartile_accuracy'].items():
            print(f"  {q_name}: {q_data['match_rate']*100:.1f}% match "
                  f"(sec={q_data['secondary_mean']:.0f}, pri={q_data['primary_mean']:.0f})")

        print("\nLong output prediction:")
        for level, data in runahead['prediction_accuracy'].items():
            print(f"  {level}: precision={data['precision']*100:.1f}%, "
                  f"recall={data['recall']*100:.1f}% ({data['overlap']}/{data['predicted']})")

        print("\nPer-step correlation statistics:")
        stats = runahead['step_correlation_stats']
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")

        if runahead['phase_correlations']:
            print("\nCorrelation by training phase:")
            for phase, data in runahead['phase_correlations'].items():
                print(f"  {phase}: mean={data['mean']:.4f}, median={data['median']:.4f}")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│ RUNAHEAD PREDICTION EFFECTIVENESS                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Same-step Pearson:    {same_step['pearson_correlation']:.4f}                                          │
│ Next-step Pearson:    {runahead['pearson_correlation']:.4f}                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ Q1 (short) accuracy:  {runahead['quartile_accuracy']['Q1']['match_rate']*100:.1f}%                                            │
│ Q4 (long) accuracy:   {runahead['quartile_accuracy']['Q4']['match_rate']*100:.1f}%                                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Top 25% precision:    {runahead['prediction_accuracy']['top25']['precision']*100:.1f}%                                            │
│ Top 10% precision:    {runahead['prediction_accuracy']['top10']['precision']*100:.1f}%                                            │
└─────────────────────────────────────────────────────────────────────────┘
""")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze runahead prediction: secondary[N] -> primary[N+1]."
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to rollout data directory"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer model name or path"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path to save results as JSON"
    )
    parser.add_argument(
        "--min_samples", type=int, default=4,
        help="Minimum samples per prompt to include"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress information"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("\nLoading data by step...")
    primary_by_step, secondary_by_step = load_data_by_step(data_dir, tokenizer, args.verbose)

    print("\nAnalyzing same-step correlation...")
    same_step_analysis = analyze_same_step_correlation(
        primary_by_step, secondary_by_step, min_samples=8
    )

    print("Analyzing runahead prediction (secondary[N] -> primary[N+1])...")
    runahead_analysis = analyze_runahead_correlation(
        primary_by_step, secondary_by_step, min_samples=args.min_samples
    )

    print_results(same_step_analysis, runahead_analysis)

    if args.output_json:
        results = {
            'same_step_analysis': same_step_analysis,
            'runahead_analysis': {
                k: v for k, v in runahead_analysis.items()
                if k != 'step_correlations'  # Exclude detailed per-step data
            },
            'step_correlations': runahead_analysis.get('step_correlations', []),
        }

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
