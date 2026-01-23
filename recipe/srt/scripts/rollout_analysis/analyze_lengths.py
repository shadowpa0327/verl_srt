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


def analyze_within_prompt_variance(
    primary_data: Dict[str, Dict],
    min_samples: int = 8
) -> Dict:
    """Analyze variance of output lengths within each prompt.

    For each prompt with sufficient samples, compute:
    - CV (coefficient of variation) = stdev / mean
    - Range = max - min
    - Stdev in tokens

    Returns distribution stats and categorization.
    """
    prompt_variance = []

    for prompt_hash, data in primary_data.items():
        lengths = data['token_lengths']
        if len(lengths) >= min_samples:
            mean_len = statistics.mean(lengths)
            stdev_len = statistics.stdev(lengths) if len(lengths) > 1 else 0
            cv = stdev_len / mean_len if mean_len > 0 else 0
            range_len = max(lengths) - min(lengths)

            prompt_variance.append({
                'hash': prompt_hash,
                'count': len(lengths),
                'mean_tokens': mean_len,
                'stdev_tokens': stdev_len,
                'cv': cv,
                'range_tokens': range_len,
                'min_tokens': min(lengths),
                'max_tokens': max(lengths),
                'acc_rate': sum(data['accs']) / len(data['accs']) if data['accs'] else 0,
            })

    if not prompt_variance:
        return {'error': 'No prompts with sufficient samples'}

    # CV distribution stats
    all_cvs = [p['cv'] for p in prompt_variance]
    all_ranges = [p['range_tokens'] for p in prompt_variance]
    all_stdevs = [p['stdev_tokens'] for p in prompt_variance]

    cv_distribution = {
        'min': min(all_cvs),
        'p25': sorted(all_cvs)[len(all_cvs) // 4],
        'median': statistics.median(all_cvs),
        'mean': statistics.mean(all_cvs),
        'p75': sorted(all_cvs)[3 * len(all_cvs) // 4],
        'max': max(all_cvs),
    }

    range_distribution = {
        'min': min(all_ranges),
        'p25': sorted(all_ranges)[len(all_ranges) // 4],
        'median': statistics.median(all_ranges),
        'mean': statistics.mean(all_ranges),
        'p75': sorted(all_ranges)[3 * len(all_ranges) // 4],
        'max': max(all_ranges),
    }

    stdev_distribution = {
        'min': min(all_stdevs),
        'median': statistics.median(all_stdevs),
        'mean': statistics.mean(all_stdevs),
        'max': max(all_stdevs),
    }

    # Variance categories
    categories = {
        'very_low': [p for p in prompt_variance if p['cv'] < 0.25],
        'low': [p for p in prompt_variance if 0.25 <= p['cv'] < 0.5],
        'medium': [p for p in prompt_variance if 0.5 <= p['cv'] < 1.0],
        'high': [p for p in prompt_variance if p['cv'] >= 1.0],
    }

    variance_categories = {
        cat: {
            'count': len(prompts),
            'pct': len(prompts) / len(prompt_variance) * 100,
            'mean_cv': statistics.mean([p['cv'] for p in prompts]) if prompts else 0,
            'mean_range': statistics.mean([p['range_tokens'] for p in prompts]) if prompts else 0,
        }
        for cat, prompts in categories.items()
    }

    # Variance vs accuracy correlation
    # Group by variance category and compute accuracy
    variance_accuracy = {}
    for cat, prompts in categories.items():
        if prompts:
            variance_accuracy[cat] = {
                'mean_acc': statistics.mean([p['acc_rate'] for p in prompts]),
                'count': len(prompts),
            }

    # Variance vs mean length correlation
    pairs = [(p['cv'], p['mean_tokens']) for p in prompt_variance]
    cv_length_corr = _calculate_pearson(pairs)

    # Correct vs incorrect length comparison
    # For prompts with mixed accuracy (some correct, some incorrect)
    correct_longer = 0
    incorrect_longer = 0
    equal_length = 0

    for prompt_hash, data in primary_data.items():
        lengths = data['token_lengths']
        accs = data['accs']
        if len(lengths) < min_samples:
            continue

        correct_lens = [l for l, a in zip(lengths, accs) if a]
        incorrect_lens = [l for l, a in zip(lengths, accs) if not a]

        if correct_lens and incorrect_lens:
            mean_correct = statistics.mean(correct_lens)
            mean_incorrect = statistics.mean(incorrect_lens)

            if mean_correct > mean_incorrect * 1.1:  # 10% threshold
                correct_longer += 1
            elif mean_incorrect > mean_correct * 1.1:
                incorrect_longer += 1
            else:
                equal_length += 1

    total_mixed = correct_longer + incorrect_longer + equal_length
    correct_vs_incorrect = {
        'correct_longer': correct_longer,
        'incorrect_longer': incorrect_longer,
        'equal_length': equal_length,
        'total_mixed_prompts': total_mixed,
        'correct_longer_pct': correct_longer / total_mixed * 100 if total_mixed > 0 else 0,
        'incorrect_longer_pct': incorrect_longer / total_mixed * 100 if total_mixed > 0 else 0,
    }

    return {
        'total_prompts': len(prompt_variance),
        'cv_distribution': cv_distribution,
        'range_distribution': range_distribution,
        'stdev_distribution': stdev_distribution,
        'variance_categories': variance_categories,
        'variance_accuracy': variance_accuracy,
        'cv_length_correlation': cv_length_corr,
        'correct_vs_incorrect': correct_vs_incorrect,
        'prompt_variance': prompt_variance,
    }


def _calculate_pearson(pairs: List[tuple]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(pairs) < 3:
        return 0.0

    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]

    x_mean = statistics.mean(x_vals)
    y_mean = statistics.mean(y_vals)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    x_std = (sum((x - x_mean) ** 2 for x in x_vals)) ** 0.5
    y_std = (sum((y - y_mean) ** 2 for y in y_vals)) ** 0.5

    return numerator / (x_std * y_std) if x_std * y_std > 0 else 0


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


def print_variance_results(variance_analysis: Dict):
    """Print within-prompt variance analysis results."""
    if 'error' in variance_analysis:
        print(f"\nVariance analysis error: {variance_analysis['error']}")
        return

    print("\n" + "=" * 80)
    print("WITHIN-PROMPT VARIANCE ANALYSIS")
    print("=" * 80)

    cv_dist = variance_analysis['cv_distribution']
    print(f"\nCoefficient of Variation (CV = stdev/mean) distribution ({variance_analysis['total_prompts']} prompts):")
    print(f"  Min:    {cv_dist['min']:.3f}")
    print(f"  25th:   {cv_dist['p25']:.3f}")
    print(f"  Median: {cv_dist['median']:.3f}")
    print(f"  Mean:   {cv_dist['mean']:.3f}")
    print(f"  75th:   {cv_dist['p75']:.3f}")
    print(f"  Max:    {cv_dist['max']:.3f}")

    range_dist = variance_analysis['range_distribution']
    print(f"\nWithin-prompt range (max - min tokens):")
    print(f"  Min:    {range_dist['min']:,.0f} tokens")
    print(f"  Median: {range_dist['median']:,.0f} tokens")
    print(f"  Mean:   {range_dist['mean']:,.0f} tokens")
    print(f"  Max:    {range_dist['max']:,.0f} tokens")

    print("\nVariance categories:")
    print(f"  {'Category':<20} {'Count':<10} {'Pct':<10} {'Mean CV':<10} {'Mean Range'}")
    print("  " + "-" * 65)
    for cat, data in variance_analysis['variance_categories'].items():
        label = {
            'very_low': 'Very low (CV < 0.25)',
            'low': 'Low (CV 0.25-0.50)',
            'medium': 'Medium (CV 0.50-1.0)',
            'high': 'High (CV >= 1.0)',
        }.get(cat, cat)
        print(f"  {label:<20} {data['count']:<10} {data['pct']:>5.1f}%    {data['mean_cv']:<10.3f} {data['mean_range']:>8,.0f}")

    if variance_analysis['variance_accuracy']:
        print("\nAccuracy by variance category:")
        for cat, data in variance_analysis['variance_accuracy'].items():
            label = {
                'very_low': 'Very low (CV < 0.25)',
                'low': 'Low (CV 0.25-0.50)',
                'medium': 'Medium (CV 0.50-1.0)',
                'high': 'High (CV >= 1.0)',
            }.get(cat, cat)
            print(f"  {label:<20} accuracy={data['mean_acc']*100:.1f}% (n={data['count']})")

    print(f"\nCV vs mean length correlation: {variance_analysis['cv_length_correlation']:.4f}")

    cvi = variance_analysis['correct_vs_incorrect']
    if cvi['total_mixed_prompts'] > 0:
        print(f"\nCorrect vs Incorrect answer lengths ({cvi['total_mixed_prompts']} prompts with mixed accuracy):")
        print(f"  Correct answers longer:   {cvi['correct_longer_pct']:>5.1f}% ({cvi['correct_longer']} prompts)")
        print(f"  Incorrect answers longer: {cvi['incorrect_longer_pct']:>5.1f}% ({cvi['incorrect_longer']} prompts)")
        print(f"  Similar length:           {100 - cvi['correct_longer_pct'] - cvi['incorrect_longer_pct']:>5.1f}% ({cvi['equal_length']} prompts)")


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
    parser.add_argument(
        "--variance", action="store_true",
        help="Enable within-prompt variance analysis"
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

    # Variance analysis
    variance_analysis = None
    if args.variance:
        print("\nAnalyzing within-prompt variance...")
        variance_analysis = analyze_within_prompt_variance(primary_data, args.min_samples)
        print_variance_results(variance_analysis)

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

        if variance_analysis and 'error' not in variance_analysis:
            results['variance_analysis'] = {
                'cv_distribution': variance_analysis['cv_distribution'],
                'range_distribution': variance_analysis['range_distribution'],
                'stdev_distribution': variance_analysis['stdev_distribution'],
                'variance_categories': variance_analysis['variance_categories'],
                'variance_accuracy': variance_analysis['variance_accuracy'],
                'cv_length_correlation': variance_analysis['cv_length_correlation'],
                'correct_vs_incorrect': variance_analysis['correct_vs_incorrect'],
            }

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
