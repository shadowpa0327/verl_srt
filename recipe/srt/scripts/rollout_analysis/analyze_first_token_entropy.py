#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Analyze first token entropy and its relationship to output length.

This script specifically examines whether the entropy of the very first
generated token (right after prefill) can predict final output length.

Usage:
    python -m recipe.srt.scripts.rollout_analysis.analyze_first_token_entropy \
        --input_json /path/to/entropy_length_analysis_full.json \
        --output_dir /path/to/output
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


def load_results(input_path: Path) -> dict:
    """Load analysis results from JSON."""
    with open(input_path) as f:
        return json.load(f)


def analyze_first_token(results: list) -> dict:
    """Detailed analysis of first token entropy vs output length."""

    first_token_entropies = [r['first_token_entropy'] for r in results]
    lengths = [r['response_length'] for r in results]
    accuracies = [r['acc'] for r in results]

    # Basic correlation
    pearson_r, pearson_p = stats.pearsonr(first_token_entropies, lengths)
    spearman_r, spearman_p = stats.spearmanr(first_token_entropies, lengths)

    analysis = {
        'num_samples': len(results),
        'first_token_entropy_stats': {
            'min': min(first_token_entropies),
            'max': max(first_token_entropies),
            'mean': statistics.mean(first_token_entropies),
            'median': statistics.median(first_token_entropies),
            'std': statistics.stdev(first_token_entropies),
        },
        'correlation': {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        }
    }

    # First token entropy by length bucket
    buckets = [(0, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 4000), (4000, float('inf'))]
    analysis['first_token_by_length_bucket'] = []

    for low, high in buckets:
        bucket_results = [r for r in results if low <= r['response_length'] < high]
        if bucket_results:
            bucket_entropies = [r['first_token_entropy'] for r in bucket_results]
            analysis['first_token_by_length_bucket'].append({
                'bucket': f"{low}-{high}" if high != float('inf') else f"{low}+",
                'count': len(bucket_results),
                'mean_entropy': statistics.mean(bucket_entropies),
                'median_entropy': statistics.median(bucket_entropies),
                'std_entropy': statistics.stdev(bucket_entropies) if len(bucket_entropies) > 1 else 0,
            })

    # First token entropy by accuracy
    correct = [r for r in results if r['acc']]
    incorrect = [r for r in results if not r['acc']]

    analysis['first_token_by_accuracy'] = {
        'correct': {
            'count': len(correct),
            'mean_entropy': statistics.mean([r['first_token_entropy'] for r in correct]) if correct else 0,
            'median_entropy': statistics.median([r['first_token_entropy'] for r in correct]) if correct else 0,
        },
        'incorrect': {
            'count': len(incorrect),
            'mean_entropy': statistics.mean([r['first_token_entropy'] for r in incorrect]) if incorrect else 0,
            'median_entropy': statistics.median([r['first_token_entropy'] for r in incorrect]) if incorrect else 0,
        }
    }

    # Quartile analysis: Can first token entropy predict length quartile?
    length_quartiles = np.percentile(lengths, [25, 50, 75])

    def get_length_quartile(length):
        if length < length_quartiles[0]:
            return 'Q1 (short)'
        elif length < length_quartiles[1]:
            return 'Q2'
        elif length < length_quartiles[2]:
            return 'Q3'
        else:
            return 'Q4 (long)'

    entropy_quartiles = np.percentile(first_token_entropies, [25, 50, 75])

    def get_entropy_quartile(entropy):
        if entropy < entropy_quartiles[0]:
            return 'Q1 (low)'
        elif entropy < entropy_quartiles[1]:
            return 'Q2'
        elif entropy < entropy_quartiles[2]:
            return 'Q3'
        else:
            return 'Q4 (high)'

    # Cross-tabulation: entropy quartile vs length quartile
    cross_tab = defaultdict(lambda: defaultdict(int))
    for r in results:
        ent_q = get_entropy_quartile(r['first_token_entropy'])
        len_q = get_length_quartile(r['response_length'])
        cross_tab[ent_q][len_q] += 1

    analysis['quartile_cross_tab'] = {k: dict(v) for k, v in cross_tab.items()}
    analysis['length_quartiles'] = length_quartiles.tolist()
    analysis['entropy_quartiles'] = entropy_quartiles.tolist()

    # Prediction accuracy: if first token entropy is in Q4 (high), how often is length in Q1 (short)?
    high_entropy_samples = [r for r in results if r['first_token_entropy'] >= entropy_quartiles[2]]
    low_entropy_samples = [r for r in results if r['first_token_entropy'] <= entropy_quartiles[0]]

    high_ent_short_output = sum(1 for r in high_entropy_samples if r['response_length'] < length_quartiles[0])
    low_ent_long_output = sum(1 for r in low_entropy_samples if r['response_length'] >= length_quartiles[2])

    analysis['prediction_insights'] = {
        'high_entropy_predicts_short': {
            'total_high_entropy': len(high_entropy_samples),
            'actually_short': high_ent_short_output,
            'rate': high_ent_short_output / len(high_entropy_samples) if high_entropy_samples else 0,
        },
        'low_entropy_predicts_long': {
            'total_low_entropy': len(low_entropy_samples),
            'actually_long': low_ent_long_output,
            'rate': low_ent_long_output / len(low_entropy_samples) if low_entropy_samples else 0,
        }
    }

    # Compare with early window (e.g., first 10 tokens) - how much does more context help?
    early_10_entropies = [r['early_10_mean'] for r in results]
    early_50_entropies = [r['early_50_mean'] for r in results]

    r_first, _ = stats.spearmanr(first_token_entropies, lengths)
    r_10, _ = stats.spearmanr(early_10_entropies, lengths)
    r_50, _ = stats.spearmanr(early_50_entropies, lengths)

    analysis['context_comparison'] = {
        'first_token_spearman': r_first,
        'first_10_spearman': r_10,
        'first_50_spearman': r_50,
        'improvement_10_over_1': abs(r_10) - abs(r_first),
        'improvement_50_over_1': abs(r_50) - abs(r_first),
    }

    return analysis


def plot_first_token_analysis(results: list, analysis: dict, output_dir: Path):
    """Create visualizations for first token entropy analysis."""

    first_token_entropies = [r['first_token_entropy'] for r in results]
    lengths = [r['response_length'] for r in results]
    accuracies = [r['acc'] for r in results]

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))

    # 1. First token entropy vs length scatter
    ax1 = fig.add_subplot(2, 3, 1)
    colors = ['#2ecc71' if acc else '#e74c3c' for acc in accuracies]

    # Sample for visualization
    sample_idx = np.random.choice(len(results), min(8000, len(results)), replace=False)
    ax1.scatter([first_token_entropies[i] for i in sample_idx],
                [lengths[i] for i in sample_idx],
                c=[colors[i] for i in sample_idx], alpha=0.3, s=10)

    corr = analysis['correlation']
    ax1.set_xlabel('First Token Entropy', fontsize=11)
    ax1.set_ylabel('Output Length (tokens)', fontsize=11)
    ax1.set_title(f'First Token Entropy vs Output Length\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax1.set_ylim(0, 6000)
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(first_token_entropies, lengths, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(first_token_entropies), max(first_token_entropies), 100)
    ax1.plot(x_line, p(x_line), 'b-', linewidth=2, label=f'Trend')
    ax1.legend()

    # 2. First token entropy distribution
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(first_token_entropies, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(analysis['first_token_entropy_stats']['median'], color='red', linestyle='--',
                label=f"Median={analysis['first_token_entropy_stats']['median']:.3f}")
    ax2.set_xlabel('First Token Entropy', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Distribution of First Token Entropy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. First token entropy by length bucket
    ax3 = fig.add_subplot(2, 3, 3)
    buckets = analysis['first_token_by_length_bucket']
    labels = [b['bucket'] for b in buckets]
    means = [b['mean_entropy'] for b in buckets]
    stds = [b['std_entropy'] for b in buckets]

    x = np.arange(len(labels))
    bars = ax3.bar(x, means, yerr=stds, capsize=4, color='#9b59b6', alpha=0.7)
    ax3.set_xlabel('Output Length Bucket', fontsize=11)
    ax3.set_ylabel('Mean First Token Entropy', fontsize=11)
    ax3.set_title('First Token Entropy by Output Length', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Comparison: First token vs early windows
    ax4 = fig.add_subplot(2, 3, 4)
    ctx = analysis['context_comparison']
    windows = ['First 1', 'First 10', 'First 50']
    correlations = [abs(ctx['first_token_spearman']), abs(ctx['first_10_spearman']), abs(ctx['first_50_spearman'])]

    bars = ax4.bar(windows, correlations, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8)
    ax4.set_ylabel('|Spearman ρ| with Length', fontsize=11)
    ax4.set_title('Prediction Power: First Token vs More Context', fontsize=12)
    ax4.set_ylim(0, max(correlations) * 1.2)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, corr in zip(bars, correlations):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')

    # 5. First token entropy by accuracy
    ax5 = fig.add_subplot(2, 3, 5)
    acc_data = analysis['first_token_by_accuracy']
    categories = ['Correct', 'Incorrect']
    entropies = [acc_data['correct']['mean_entropy'], acc_data['incorrect']['mean_entropy']]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax5.bar(categories, entropies, color=colors, alpha=0.8)
    ax5.set_ylabel('Mean First Token Entropy', fontsize=11)
    ax5.set_title('First Token Entropy by Correctness', fontsize=12)

    for bar, ent in zip(bars, entropies):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ent:.3f}', ha='center', va='bottom', fontweight='bold')
    ax5.set_ylim(0, max(entropies) * 1.3)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Prediction insights
    ax6 = fig.add_subplot(2, 3, 6)
    pred = analysis['prediction_insights']

    labels = ['High Entropy\n→ Short Output', 'Low Entropy\n→ Long Output']
    rates = [pred['high_entropy_predicts_short']['rate'] * 100,
             pred['low_entropy_predicts_long']['rate'] * 100]
    baseline = 25  # Random guess would be 25% for quartile prediction

    x = np.arange(len(labels))
    bars = ax6.bar(x, rates, color=['#e74c3c', '#2ecc71'], alpha=0.8)
    ax6.axhline(baseline, color='gray', linestyle='--', label=f'Random baseline ({baseline}%)')
    ax6.set_ylabel('Prediction Rate (%)', fontsize=11)
    ax6.set_title('First Token Entropy as Length Predictor', fontsize=12)
    ax6.set_xticks(x)
    ax6.set_xticklabels(labels)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')

    for bar, rate in zip(bars, rates):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('First Token Entropy Analysis (Right After Prefill)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'first_token_entropy_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: first_token_entropy_analysis.png")


def plot_quartile_heatmap(analysis: dict, output_dir: Path):
    """Create heatmap of entropy quartile vs length quartile."""

    cross_tab = analysis['quartile_cross_tab']

    # Define order
    entropy_order = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
    length_order = ['Q1 (short)', 'Q2', 'Q3', 'Q4 (long)']

    # Build matrix
    matrix = np.zeros((4, 4))
    for i, ent_q in enumerate(entropy_order):
        for j, len_q in enumerate(length_order):
            matrix[i, j] = cross_tab.get(ent_q, {}).get(len_q, 0)

    # Normalize by row (entropy quartile)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = matrix / row_sums * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    ax1 = axes[0]
    im1 = ax1.imshow(matrix, cmap='YlOrRd')
    ax1.set_xticks(np.arange(4))
    ax1.set_yticks(np.arange(4))
    ax1.set_xticklabels(length_order)
    ax1.set_yticklabels(entropy_order)
    ax1.set_xlabel('Output Length Quartile', fontsize=11)
    ax1.set_ylabel('First Token Entropy Quartile', fontsize=11)
    ax1.set_title('Cross-tabulation: Entropy vs Length Quartiles\n(Raw Counts)', fontsize=12)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f'{int(matrix[i, j])}', ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max()/2 else 'black')

    plt.colorbar(im1, ax=ax1, label='Count')

    # Percentages (row-normalized)
    ax2 = axes[1]
    im2 = ax2.imshow(matrix_pct, cmap='RdYlGn_r')
    ax2.set_xticks(np.arange(4))
    ax2.set_yticks(np.arange(4))
    ax2.set_xticklabels(length_order)
    ax2.set_yticklabels(entropy_order)
    ax2.set_xlabel('Output Length Quartile', fontsize=11)
    ax2.set_ylabel('First Token Entropy Quartile', fontsize=11)
    ax2.set_title('Row-Normalized (%)\nGiven entropy quartile, what % fall into each length quartile?', fontsize=12)

    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{matrix_pct[i, j]:.1f}%', ha='center', va='center',
                    color='white' if matrix_pct[i, j] > 35 else 'black', fontweight='bold')

    plt.colorbar(im2, ax=ax2, label='Percentage')

    plt.tight_layout()
    plt.savefig(output_dir / 'first_token_quartile_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved: first_token_quartile_heatmap.png")


def print_analysis(analysis: dict):
    """Print analysis results."""

    print("\n" + "=" * 80)
    print("FIRST TOKEN ENTROPY ANALYSIS (Right After Prefill)")
    print("=" * 80)

    stats = analysis['first_token_entropy_stats']
    print(f"\nFirst Token Entropy Statistics:")
    print(f"  Mean:   {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Std:    {stats['std']:.4f}")
    print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")

    corr = analysis['correlation']
    print(f"\nCorrelation with Output Length:")
    print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman ρ: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    print("\n" + "-" * 40)
    print("First Token Entropy by Length Bucket:")
    print("-" * 40)
    for b in analysis['first_token_by_length_bucket']:
        print(f"  {b['bucket']:<12} n={b['count']:>5}  mean={b['mean_entropy']:.4f} (+/- {b['std_entropy']:.4f})")

    print("\n" + "-" * 40)
    print("Comparison: How much does more context help?")
    print("-" * 40)
    ctx = analysis['context_comparison']
    print(f"  First 1 token:  |ρ| = {abs(ctx['first_token_spearman']):.4f}")
    print(f"  First 10 tokens: |ρ| = {abs(ctx['first_10_spearman']):.4f} (improvement: +{ctx['improvement_10_over_1']:.4f})")
    print(f"  First 50 tokens: |ρ| = {abs(ctx['first_50_spearman']):.4f} (improvement: +{ctx['improvement_50_over_1']:.4f})")

    print("\n" + "-" * 40)
    print("Prediction Insights:")
    print("-" * 40)
    pred = analysis['prediction_insights']
    print(f"  High entropy (top 25%) → Short output (bottom 25%):")
    print(f"    Rate: {pred['high_entropy_predicts_short']['rate']*100:.1f}% (baseline: 25%)")
    print(f"  Low entropy (bottom 25%) → Long output (top 25%):")
    print(f"    Rate: {pred['low_entropy_predicts_long']['rate']*100:.1f}% (baseline: 25%)")

    print("\n" + "-" * 40)
    print("First Token Entropy by Accuracy:")
    print("-" * 40)
    acc = analysis['first_token_by_accuracy']
    print(f"  Correct:   mean={acc['correct']['mean_entropy']:.4f} (n={acc['correct']['count']})")
    print(f"  Incorrect: mean={acc['incorrect']['mean_entropy']:.4f} (n={acc['incorrect']['count']})")


def main():
    parser = argparse.ArgumentParser(description="Analyze first token entropy vs output length.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to entropy analysis JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_path}...")
    data = load_results(input_path)
    results = data['per_sample_results']
    print(f"Loaded {len(results)} samples")

    print("\nAnalyzing first token entropy...")
    analysis = analyze_first_token(results)

    print_analysis(analysis)

    print("\nCreating visualizations...")
    plot_first_token_analysis(results, analysis, output_dir)
    plot_quartile_heatmap(analysis, output_dir)

    # Save analysis
    output_json = output_dir / 'first_token_entropy_analysis.json'
    with open(output_json, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to: {output_json}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
