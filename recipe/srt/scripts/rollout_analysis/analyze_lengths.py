#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Analyze rollout output token lengths and correlation with accuracy.

This script analyzes:
1. Per-prompt output token length distribution
2. Correlation between output length and accuracy (primary only)
3. Characteristics of short vs long output prompts

Usage:
    python analyze_lengths.py \\
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
from typing import Dict, List, Optional

from transformers import AutoTokenizer


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (first 16 hex chars)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def load_primary_data(
    data_dir: Path,
    tokenizer,
    verbose: bool = False
) -> Dict[str, Dict]:
    """Load primary rollout data and tokenize outputs."""
    rollout_dir = data_dir / "rollout"
    prompt_data = defaultdict(lambda: {
        'token_lengths': [],
        'char_lengths': [],
        'steps': [],
        'accs': [],
        'scores': [],
        'prompt_text': None,
        'gts': None,
    })

    count = 0
    for step_file in sorted(rollout_dir.glob("*.jsonl"), key=lambda x: int(x.stem)):
        step = int(step_file.stem)
        with open(step_file) as f:
            for line in f:
                item = json.loads(line)
                prompt_text = item['input']
                output_text = item['output']

                output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
                prompt_hash = compute_prompt_hash(prompt_text)

                prompt_data[prompt_hash]['token_lengths'].append(len(output_tokens))
                prompt_data[prompt_hash]['char_lengths'].append(len(output_text))
                prompt_data[prompt_hash]['steps'].append(step)
                prompt_data[prompt_hash]['accs'].append(item.get('acc', False))
                prompt_data[prompt_hash]['scores'].append(item.get('score', 0))

                if prompt_data[prompt_hash]['prompt_text'] is None:
                    prompt_data[prompt_hash]['prompt_text'] = prompt_text
                    prompt_data[prompt_hash]['gts'] = item.get('gts', '')

                count += 1
                if verbose and count % 5000 == 0:
                    print(f"  Primary: processed {count} samples...")

    if verbose:
        print(f"  Primary: total {count} samples, {len(prompt_data)} prompts")

    return dict(prompt_data)


def load_secondary_data(
    data_dir: Path,
    tokenizer,
    verbose: bool = False
) -> Dict[str, Dict]:
    """Load secondary (runahead) data and tokenize outputs."""
    secondary_dir = data_dir / "secondary"
    prompt_data = defaultdict(lambda: {
        'token_lengths': [],
        'char_lengths': [],
        'steps': [],
        'statuses': [],
        'prompt_text': None,
    })

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

                prompt_text = item['prompt']
                response_text = item['response']

                response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
                prompt_hash = compute_prompt_hash(prompt_text)

                prompt_data[prompt_hash]['token_lengths'].append(len(response_tokens))
                prompt_data[prompt_hash]['char_lengths'].append(len(response_text))
                prompt_data[prompt_hash]['steps'].append(step)
                prompt_data[prompt_hash]['statuses'].append(item.get('status', 'unknown'))

                if prompt_data[prompt_hash]['prompt_text'] is None:
                    prompt_data[prompt_hash]['prompt_text'] = prompt_text

                count += 1
                if verbose and count % 5000 == 0:
                    print(f"  Secondary: processed {count} samples...")

    if verbose:
        print(f"  Secondary: total {count} samples, {len(prompt_data)} prompts (skipped {skipped})")

    return dict(prompt_data)


def analyze_primary_lengths(
    primary_data: Dict[str, Dict],
    min_samples: int = 8
) -> Dict:
    """Analyze primary output length distribution and accuracy correlation."""

    prompt_stats = []
    for prompt_hash, data in primary_data.items():
        lengths = data['token_lengths']
        if len(lengths) >= min_samples:
            prompt_stats.append({
                'hash': prompt_hash,
                'count': len(lengths),
                'mean_tokens': statistics.mean(lengths),
                'median_tokens': statistics.median(lengths),
                'min_tokens': min(lengths),
                'max_tokens': max(lengths),
                'stdev_tokens': statistics.stdev(lengths) if len(lengths) > 1 else 0,
                'acc_rate': sum(data['accs']) / len(data['accs']) if data['accs'] else 0,
                'mean_score': statistics.mean(data['scores']) if data['scores'] else 0,
                'prompt_text': data['prompt_text'],
                'gts': data['gts'],
            })

    prompt_stats.sort(key=lambda x: x['mean_tokens'])
    all_means = [p['mean_tokens'] for p in prompt_stats]

    # Distribution stats
    distribution = {
        'total_prompts': len(prompt_stats),
        'min': min(all_means),
        'p10': sorted(all_means)[len(all_means)//10],
        'p25': sorted(all_means)[len(all_means)//4],
        'median': statistics.median(all_means),
        'p75': sorted(all_means)[3*len(all_means)//4],
        'p90': sorted(all_means)[9*len(all_means)//10],
        'max': max(all_means),
    }

    # Accuracy by token length bucket
    buckets = [(0, 300), (300, 600), (600, 1000), (1000, 1500),
               (1500, 2000), (2000, 3000), (3000, float('inf'))]

    accuracy_by_bucket = []
    for low, high in buckets:
        bucket_prompts = [p for p in prompt_stats if low <= p['mean_tokens'] < high]
        if bucket_prompts:
            accuracy_by_bucket.append({
                'bucket': f"{low}-{high}" if high != float('inf') else f"{low}+",
                'count': len(bucket_prompts),
                'accuracy': statistics.mean([p['acc_rate'] for p in bucket_prompts]),
                'median_tokens': statistics.median([p['mean_tokens'] for p in bucket_prompts]),
            })

    # High vs zero accuracy comparison
    high_acc = [p for p in prompt_stats if p['acc_rate'] >= 0.8]
    zero_acc = [p for p in prompt_stats if p['acc_rate'] == 0]

    accuracy_comparison = {
        'high_acc_count': len(high_acc),
        'high_acc_median_tokens': statistics.median([p['mean_tokens'] for p in high_acc]) if high_acc else 0,
        'zero_acc_count': len(zero_acc),
        'zero_acc_median_tokens': statistics.median([p['mean_tokens'] for p in zero_acc]) if zero_acc else 0,
    }

    # Short and long prompt examples
    short_threshold = distribution['p10']
    long_threshold = distribution['p90']

    short_prompts = [p for p in prompt_stats if p['mean_tokens'] <= short_threshold][:10]
    long_prompts = [p for p in prompt_stats if p['mean_tokens'] >= long_threshold][-10:]

    return {
        'distribution': distribution,
        'accuracy_by_bucket': accuracy_by_bucket,
        'accuracy_comparison': accuracy_comparison,
        'short_prompts': [{
            'hash': p['hash'],
            'mean_tokens': p['mean_tokens'],
            'acc_rate': p['acc_rate'],
            'gts': p['gts'],
        } for p in short_prompts],
        'long_prompts': [{
            'hash': p['hash'],
            'mean_tokens': p['mean_tokens'],
            'acc_rate': p['acc_rate'],
            'gts': p['gts'],
        } for p in long_prompts],
        'prompt_stats': prompt_stats,
    }


def analyze_secondary_lengths(
    secondary_data: Dict[str, Dict],
    min_samples: int = 8
) -> Dict:
    """Analyze secondary output length distribution."""

    prompt_stats = []
    for prompt_hash, data in secondary_data.items():
        lengths = data['token_lengths']
        if len(lengths) >= min_samples:
            completed_rate = sum(1 for s in data['statuses'] if s == 'completed') / len(data['statuses'])
            prompt_stats.append({
                'hash': prompt_hash,
                'count': len(lengths),
                'mean_tokens': statistics.mean(lengths),
                'median_tokens': statistics.median(lengths),
                'min_tokens': min(lengths),
                'max_tokens': max(lengths),
                'completed_rate': completed_rate,
            })

    prompt_stats.sort(key=lambda x: x['mean_tokens'])
    all_means = [p['mean_tokens'] for p in prompt_stats]

    distribution = {
        'total_prompts': len(prompt_stats),
        'min': min(all_means),
        'p10': sorted(all_means)[len(all_means)//10],
        'p25': sorted(all_means)[len(all_means)//4],
        'median': statistics.median(all_means),
        'p75': sorted(all_means)[3*len(all_means)//4],
        'p90': sorted(all_means)[9*len(all_means)//10],
        'max': max(all_means),
    }

    # Completion rate by bucket
    buckets = [(0, 300), (300, 600), (600, 1000), (1000, 1500),
               (1500, 2000), (2000, 3000), (3000, float('inf'))]

    completion_by_bucket = []
    for low, high in buckets:
        bucket_prompts = [p for p in prompt_stats if low <= p['mean_tokens'] < high]
        if bucket_prompts:
            completion_by_bucket.append({
                'bucket': f"{low}-{high}" if high != float('inf') else f"{low}+",
                'count': len(bucket_prompts),
                'completion_rate': statistics.mean([p['completed_rate'] for p in bucket_prompts]),
            })

    return {
        'distribution': distribution,
        'completion_by_bucket': completion_by_bucket,
        'prompt_stats': prompt_stats,
    }


def print_results(
    primary_analysis: Dict,
    secondary_analysis: Optional[Dict] = None,
):
    """Print analysis results to console."""

    print("=" * 80)
    print("PRIMARY ROLLOUT - OUTPUT TOKEN LENGTH ANALYSIS")
    print("=" * 80)

    dist = primary_analysis['distribution']
    print(f"\nDistribution of per-prompt mean token lengths ({dist['total_prompts']} prompts):")
    print(f"  Min:    {dist['min']:,.0f} tokens")
    print(f"  10th:   {dist['p10']:,.0f} tokens")
    print(f"  25th:   {dist['p25']:,.0f} tokens")
    print(f"  Median: {dist['median']:,.0f} tokens")
    print(f"  75th:   {dist['p75']:,.0f} tokens")
    print(f"  90th:   {dist['p90']:,.0f} tokens")
    print(f"  Max:    {dist['max']:,.0f} tokens")

    print("\nAccuracy by output token length bucket:")
    print(f"  {'Bucket':<16} {'Count':<8} {'Accuracy':<12} {'Median Tokens'}")
    print("  " + "-" * 55)
    for b in primary_analysis['accuracy_by_bucket']:
        print(f"  {b['bucket']:<16} {b['count']:<8} {b['accuracy']*100:>6.1f}%      {b['median_tokens']:>8,.0f}")

    acc_cmp = primary_analysis['accuracy_comparison']
    print(f"\nHigh accuracy (≥80%): {acc_cmp['high_acc_count']} prompts, median={acc_cmp['high_acc_median_tokens']:.0f} tokens")
    print(f"Zero accuracy (0%):   {acc_cmp['zero_acc_count']} prompts, median={acc_cmp['zero_acc_median_tokens']:.0f} tokens")

    print("\nShortest output prompts (bottom 10%):")
    for p in primary_analysis['short_prompts'][:5]:
        print(f"  [{p['hash']}] tokens={p['mean_tokens']:.0f}, acc={p['acc_rate']*100:.0f}%")

    print("\nLongest output prompts (top 10%):")
    for p in primary_analysis['long_prompts'][-5:]:
        print(f"  [{p['hash']}] tokens={p['mean_tokens']:.0f}, acc={p['acc_rate']*100:.0f}%")

    if secondary_analysis:
        print("\n" + "=" * 80)
        print("SECONDARY (RUNAHEAD) - OUTPUT TOKEN LENGTH ANALYSIS")
        print("=" * 80)

        dist = secondary_analysis['distribution']
        print(f"\nDistribution ({dist['total_prompts']} prompts):")
        print(f"  Min:    {dist['min']:,.0f} tokens")
        print(f"  Median: {dist['median']:,.0f} tokens")
        print(f"  Max:    {dist['max']:,.0f} tokens")

        print("\nCompletion rate by token length bucket:")
        for b in secondary_analysis['completion_by_bucket']:
            print(f"  {b['bucket']:<16} {b['count']:<8} {b['completion_rate']*100:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rollout output token lengths and accuracy correlation."
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
        "--min_samples", type=int, default=8,
        help="Minimum samples per prompt to include"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress information"
    )
    parser.add_argument(
        "--primary_only", action="store_true",
        help="Only analyze primary data"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print("\nLoading primary data...")
    primary_data = load_primary_data(data_dir, tokenizer, args.verbose)

    print("Analyzing primary data...")
    primary_analysis = analyze_primary_lengths(primary_data, args.min_samples)

    secondary_analysis = None
    if not args.primary_only:
        secondary_dir = data_dir / "secondary"
        if secondary_dir.exists():
            print("\nLoading secondary data...")
            secondary_data = load_secondary_data(data_dir, tokenizer, args.verbose)

            print("Analyzing secondary data...")
            secondary_analysis = analyze_secondary_lengths(secondary_data, args.min_samples)

    print_results(primary_analysis, secondary_analysis)

    if args.output_json:
        results = {
            'primary_analysis': {
                'distribution': primary_analysis['distribution'],
                'accuracy_by_bucket': primary_analysis['accuracy_by_bucket'],
                'accuracy_comparison': primary_analysis['accuracy_comparison'],
                'short_prompts': primary_analysis['short_prompts'],
                'long_prompts': primary_analysis['long_prompts'],
            },
        }

        if secondary_analysis:
            results['secondary_analysis'] = {
                'distribution': secondary_analysis['distribution'],
                'completion_by_bucket': secondary_analysis['completion_by_bucket'],
            }

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
