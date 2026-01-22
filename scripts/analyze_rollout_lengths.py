#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Analyze rollout output lengths and correlation between secondary (runahead)
and primary rollout data.

This script analyzes:
1. Per-prompt output token length distribution
2. Correlation between output length and accuracy (primary only)
3. Correlation between secondary and primary output lengths
4. Prediction accuracy: Can secondary predict which prompts have long outputs?

Usage:
    python scripts/analyze_rollout_lengths.py \\
        --data_dir /path/to/rollout_data \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct \\
        --output_json /path/to/results.json
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


def load_primary_data(
    data_dir: Path,
    tokenizer,
    verbose: bool = False
) -> Dict[str, Dict]:
    """Load primary rollout data and tokenize outputs."""
    rollout_dir = data_dir / "rollout"
    prompt_data = defaultdict(lambda: {
        'token_lengths': [],
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

                # Tokenize output
                output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
                prompt_hash = compute_prompt_hash(prompt_text)

                prompt_data[prompt_hash]['token_lengths'].append(len(output_tokens))
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

                # Skip rejected entries
                if item.get('status') == 'rejected' or 'prompt' not in item:
                    skipped += 1
                    continue

                prompt_text = item['prompt']
                response_text = item['response']

                # Tokenize response
                response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
                prompt_hash = compute_prompt_hash(prompt_text)

                prompt_data[prompt_hash]['token_lengths'].append(len(response_tokens))
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

    # Calculate per-prompt stats
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
            })

    prompt_stats.sort(key=lambda x: x['mean_tokens'])
    all_means = [p['mean_tokens'] for p in prompt_stats]

    # Distribution stats
    distribution = {
        'total_prompts': len(prompt_stats),
        'min': min(all_means),
        'p25': sorted(all_means)[len(all_means)//4],
        'median': statistics.median(all_means),
        'p75': sorted(all_means)[3*len(all_means)//4],
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

    return {
        'distribution': distribution,
        'accuracy_by_bucket': accuracy_by_bucket,
        'accuracy_comparison': accuracy_comparison,
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
        'p25': sorted(all_means)[len(all_means)//4],
        'median': statistics.median(all_means),
        'p75': sorted(all_means)[3*len(all_means)//4],
        'max': max(all_means),
    }

    return {
        'distribution': distribution,
        'prompt_stats': prompt_stats,
    }


def analyze_correlation(
    primary_data: Dict[str, Dict],
    secondary_data: Dict[str, Dict],
    min_samples: int = 8
) -> Dict:
    """Analyze correlation between secondary and primary output lengths."""

    # Find common prompts
    common_prompts = set(primary_data.keys()) & set(secondary_data.keys())

    # Build paired data
    paired_data = []
    for prompt_hash in common_prompts:
        p_lengths = primary_data[prompt_hash]['token_lengths']
        s_lengths = secondary_data[prompt_hash]['token_lengths']

        if len(p_lengths) >= min_samples and len(s_lengths) >= min_samples:
            paired_data.append({
                'hash': prompt_hash,
                'primary_mean': statistics.mean(p_lengths),
                'secondary_mean': statistics.mean(s_lengths),
            })

    if not paired_data:
        return {'error': 'No common prompts with sufficient samples'}

    # Calculate Pearson correlation
    p_means = [d['primary_mean'] for d in paired_data]
    s_means = [d['secondary_mean'] for d in paired_data]

    p_mean = statistics.mean(p_means)
    s_mean = statistics.mean(s_means)

    numerator = sum((p - p_mean) * (s - s_mean) for p, s in zip(p_means, s_means))
    p_stdev = (sum((p - p_mean)**2 for p in p_means)) ** 0.5
    s_stdev = (sum((s - s_mean)**2 for s in s_means)) ** 0.5

    pearson = numerator / (p_stdev * s_stdev) if p_stdev * s_stdev > 0 else 0

    # Calculate Spearman rank correlation
    paired_sorted_p = sorted(enumerate(paired_data), key=lambda x: x[1]['primary_mean'])
    paired_sorted_s = sorted(enumerate(paired_data), key=lambda x: x[1]['secondary_mean'])

    p_ranks = {idx: rank for rank, (idx, _) in enumerate(paired_sorted_p)}
    s_ranks = {idx: rank for rank, (idx, _) in enumerate(paired_sorted_s)}

    n = len(paired_data)
    d_squared_sum = sum((p_ranks[i] - s_ranks[i])**2 for i in range(n))
    spearman = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))

    # Quartile prediction analysis
    paired_data.sort(key=lambda x: x['secondary_mean'])
    q_size = len(paired_data) // 4

    # Assign primary quartiles
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
        same_q = sum(1 for p in q_prompts if primary_quartile[p['hash']] == q_name)
        quartile_accuracy[q_name] = {
            'count': len(q_prompts),
            'match_rate': same_q / len(q_prompts) if q_prompts else 0,
            'secondary_mean': statistics.mean([p['secondary_mean'] for p in q_prompts]) if q_prompts else 0,
            'primary_mean': statistics.mean([p['primary_mean'] for p in q_prompts]) if q_prompts else 0,
        }

    # Top 25% and top 10% prediction
    q4_secondary = set(p['hash'] for p in paired_data[3*q_size:])
    q4_primary = set(p['hash'] for p in paired_sorted_primary[3*q_size:])
    q4_overlap = q4_secondary & q4_primary

    top10_size = len(paired_data) // 10
    top10_secondary = set(p['hash'] for p in paired_data[-top10_size:])
    top10_primary = set(p['hash'] for p in paired_sorted_primary[-top10_size:])
    top10_overlap = top10_secondary & top10_primary

    prediction_accuracy = {
        'top25': {
            'precision': len(q4_overlap) / len(q4_secondary) if q4_secondary else 0,
            'recall': len(q4_overlap) / len(q4_primary) if q4_primary else 0,
            'overlap': len(q4_overlap),
            'predicted': len(q4_secondary),
            'actual': len(q4_primary),
        },
        'top10': {
            'precision': len(top10_overlap) / len(top10_secondary) if top10_secondary else 0,
            'recall': len(top10_overlap) / len(top10_primary) if top10_primary else 0,
            'overlap': len(top10_overlap),
            'predicted': len(top10_secondary),
            'actual': len(top10_primary),
        },
    }

    return {
        'common_prompts': len(paired_data),
        'pearson_correlation': pearson,
        'spearman_correlation': spearman,
        'quartile_accuracy': quartile_accuracy,
        'prediction_accuracy': prediction_accuracy,
    }


def print_results(
    primary_analysis: Dict,
    secondary_analysis: Optional[Dict],
    correlation_analysis: Optional[Dict],
):
    """Print analysis results to console."""

    print("=" * 80)
    print("PRIMARY ROLLOUT - OUTPUT TOKEN LENGTH ANALYSIS")
    print("=" * 80)

    dist = primary_analysis['distribution']
    print(f"\nDistribution of per-prompt mean token lengths ({dist['total_prompts']} prompts):")
    print(f"  Min:    {dist['min']:,.0f} tokens")
    print(f"  25th:   {dist['p25']:,.0f} tokens")
    print(f"  Median: {dist['median']:,.0f} tokens")
    print(f"  75th:   {dist['p75']:,.0f} tokens")
    print(f"  Max:    {dist['max']:,.0f} tokens")

    print("\nAccuracy by output token length bucket:")
    print(f"  {'Bucket':<16} {'Count':<8} {'Accuracy':<12} {'Median Tokens'}")
    print("  " + "-" * 55)
    for b in primary_analysis['accuracy_by_bucket']:
        print(f"  {b['bucket']:<16} {b['count']:<8} {b['accuracy']*100:>6.1f}%      {b['median_tokens']:>8,.0f}")

    acc_cmp = primary_analysis['accuracy_comparison']
    print(f"\nHigh accuracy (≥80%): {acc_cmp['high_acc_count']} prompts, median={acc_cmp['high_acc_median_tokens']:.0f} tokens")
    print(f"Zero accuracy (0%):   {acc_cmp['zero_acc_count']} prompts, median={acc_cmp['zero_acc_median_tokens']:.0f} tokens")

    if secondary_analysis:
        print("\n" + "=" * 80)
        print("SECONDARY (RUNAHEAD) - OUTPUT TOKEN LENGTH ANALYSIS")
        print("=" * 80)

        dist = secondary_analysis['distribution']
        print(f"\nDistribution ({dist['total_prompts']} prompts):")
        print(f"  Min:    {dist['min']:,.0f} tokens")
        print(f"  Median: {dist['median']:,.0f} tokens")
        print(f"  Max:    {dist['max']:,.0f} tokens")

    if correlation_analysis:
        print("\n" + "=" * 80)
        print("CORRELATION: SECONDARY vs PRIMARY OUTPUT LENGTHS")
        print("=" * 80)

        print(f"\nCommon prompts analyzed: {correlation_analysis['common_prompts']}")
        print(f"Pearson correlation:  {correlation_analysis['pearson_correlation']:.4f}")
        print(f"Spearman correlation: {correlation_analysis['spearman_correlation']:.4f}")

        print("\nQuartile prediction accuracy:")
        for q_name, q_data in correlation_analysis['quartile_accuracy'].items():
            print(f"  {q_name}: {q_data['match_rate']*100:.1f}% match rate "
                  f"(sec={q_data['secondary_mean']:.0f}, pri={q_data['primary_mean']:.0f})")

        print("\nLong output prediction:")
        for level, data in correlation_analysis['prediction_accuracy'].items():
            print(f"  {level}: precision={data['precision']*100:.1f}%, "
                  f"recall={data['recall']*100:.1f}% "
                  f"({data['overlap']}/{data['predicted']} correct)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze rollout output lengths and secondary/primary correlation."
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to rollout data directory (containing rollout/ and secondary/ subdirs)"
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
        help="Minimum samples per prompt to include in analysis"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress information"
    )
    parser.add_argument(
        "--primary_only", action="store_true",
        help="Only analyze primary data (skip secondary and correlation)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Load primary data
    print("\nLoading primary data...")
    primary_data = load_primary_data(data_dir, tokenizer, args.verbose)

    # Analyze primary
    print("Analyzing primary data...")
    primary_analysis = analyze_primary_lengths(primary_data, args.min_samples)

    secondary_analysis = None
    correlation_analysis = None

    if not args.primary_only:
        # Load secondary data
        secondary_dir = data_dir / "secondary"
        if secondary_dir.exists():
            print("\nLoading secondary data...")
            secondary_data = load_secondary_data(data_dir, tokenizer, args.verbose)

            print("Analyzing secondary data...")
            secondary_analysis = analyze_secondary_lengths(secondary_data, args.min_samples)

            print("Analyzing correlation...")
            correlation_analysis = analyze_correlation(
                primary_data, secondary_data, args.min_samples
            )
        else:
            print("\nNo secondary data found, skipping correlation analysis.")

    # Print results
    print_results(primary_analysis, secondary_analysis, correlation_analysis)

    # Save to JSON if requested
    if args.output_json:
        results = {
            'primary_analysis': {
                'distribution': primary_analysis['distribution'],
                'accuracy_by_bucket': primary_analysis['accuracy_by_bucket'],
                'accuracy_comparison': primary_analysis['accuracy_comparison'],
            },
        }

        if secondary_analysis:
            results['secondary_analysis'] = {
                'distribution': secondary_analysis['distribution'],
            }

        if correlation_analysis:
            results['correlation_analysis'] = correlation_analysis

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
