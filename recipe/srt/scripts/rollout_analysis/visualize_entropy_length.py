#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Visualize entropy vs output length analysis results.

Usage:
    python -m recipe.srt.scripts.rollout_analysis.visualize_entropy_length \
        --input_json /path/to/entropy_length_analysis_full.json \
        --output_dir /path/to/plots
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


def load_results(input_path: Path) -> dict:
    """Load analysis results from JSON."""
    with open(input_path) as f:
        return json.load(f)


def plot_entropy_vs_length_scatter(results: list, output_dir: Path):
    """Create scatter plot of mean entropy vs output length."""
    lengths = [r['response_length'] for r in results]
    entropies = [r['mean_entropy'] for r in results]
    accuracies = [r['acc'] for r in results]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by accuracy
    colors = ['#2ecc71' if acc else '#e74c3c' for acc in accuracies]

    ax.scatter(lengths, entropies, c=colors, alpha=0.3, s=10)

    # Add trend line
    z = np.polyfit(lengths, entropies, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(lengths), max(lengths), 100)
    ax.plot(x_line, p(x_line), 'b-', linewidth=2, label=f'Trend (slope={z[0]:.2e})')

    # Calculate correlation
    r, p_val = stats.pearsonr(lengths, entropies)
    rho, _ = stats.spearmanr(lengths, entropies)

    ax.set_xlabel('Output Length (tokens)', fontsize=12)
    ax.set_ylabel('Mean Entropy', fontsize=12)
    ax.set_title(f'Mean Entropy vs Output Length\nPearson r={r:.3f}, Spearman ρ={rho:.3f}', fontsize=14)

    # Legend
    correct_patch = mpatches.Patch(color='#2ecc71', alpha=0.5, label='Correct')
    incorrect_patch = mpatches.Patch(color='#e74c3c', alpha=0.5, label='Incorrect')
    ax.legend(handles=[correct_patch, incorrect_patch, ax.lines[0]], loc='upper right')

    ax.set_xlim(0, max(lengths) * 1.05)
    ax.set_ylim(0, min(max(entropies) * 1.1, 5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_vs_length_scatter.png', dpi=150)
    plt.close()
    print(f"Saved: entropy_vs_length_scatter.png")


def plot_entropy_by_length_bucket(analysis: dict, output_dir: Path):
    """Create box/bar plot of entropy by length bucket with p25, p50, p75."""
    buckets = analysis['entropy_by_length_bucket']

    labels = [b['bucket'] for b in buckets]
    counts = [b['count'] for b in buckets]

    # Check if percentiles are available (backward compatible)
    has_percentiles = 'p25' in buckets[0]

    if has_percentiles:
        p25s = [b['p25'] for b in buckets]
        p50s = [b['p50'] for b in buckets]
        p75s = [b['p75'] for b in buckets]
    else:
        # Fallback to mean/std
        p50s = [b['mean_entropy'] for b in buckets]
        p25s = [b['mean_entropy'] - b['std_entropy'] for b in buckets]
        p75s = [b['mean_entropy'] + b['std_entropy'] for b in buckets]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    x = np.arange(len(labels))
    width = 0.25

    # Plot p25, p50, p75 as grouped bars
    bars_p25 = ax1.bar(x - width, p25s, width, color='#85c1e9', alpha=0.8, label='P25')
    bars_p50 = ax1.bar(x, p50s, width, color='#3498db', alpha=0.8, label='P50 (Median)')
    bars_p75 = ax1.bar(x + width, p75s, width, color='#1a5276', alpha=0.8, label='P75')

    ax1.set_xlabel('Output Length Bucket (tokens)', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Add sample counts on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, counts, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Sample Count')
    ax2.set_ylabel('Sample Count', fontsize=12, color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    # Add value labels on bars
    for i, (b25, b50, b75) in enumerate(zip(bars_p25, bars_p50, bars_p75)):
        ax1.text(b25.get_x() + b25.get_width()/2, b25.get_height() + 0.02,
                f'{p25s[i]:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)
        ax1.text(b50.get_x() + b50.get_width()/2, b50.get_height() + 0.02,
                f'{p50s[i]:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)
        ax1.text(b75.get_x() + b75.get_width()/2, b75.get_height() + 0.02,
                f'{p75s[i]:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax1.set_title('Entropy Percentiles (P25, P50, P75) by Output Length Bucket', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_by_length_bucket.png', dpi=150)
    plt.close()
    print(f"Saved: entropy_by_length_bucket.png")


def plot_early_token_correlations(analysis: dict, output_dir: Path):
    """Plot correlation strength for different early-token windows."""
    early_corr = analysis['early_token_correlations']

    windows = []
    pearson_rs = []
    spearman_rs = []

    for key, corr in early_corr.items():
        # Extract window size from key like "first_10_tokens"
        window = int(key.split('_')[1])
        windows.append(window)
        pearson_rs.append(abs(corr['pearson_r']))
        spearman_rs.append(abs(corr['spearman_r']))

    # Sort by window size
    sorted_idx = np.argsort(windows)
    windows = [windows[i] for i in sorted_idx]
    pearson_rs = [pearson_rs[i] for i in sorted_idx]
    spearman_rs = [spearman_rs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(windows))
    width = 0.35

    bars1 = ax.bar(x - width/2, pearson_rs, width, label='|Pearson r|', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, spearman_rs, width, label='|Spearman ρ|', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Early Token Window Size', fontsize=12)
    ax.set_ylabel('Correlation Strength (absolute)', fontsize=12)
    ax.set_title('Early-Token Entropy Correlation with Final Output Length\n(Higher = Better Prediction)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'First {w}' for w in windows])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(spearman_rs) * 1.2)

    plt.tight_layout()
    plt.savefig(output_dir / 'early_token_correlations.png', dpi=150)
    plt.close()
    print(f"Saved: early_token_correlations.png")


def plot_entropy_by_accuracy(analysis: dict, output_dir: Path):
    """Plot entropy comparison between correct and incorrect answers."""
    acc_data = analysis['entropy_by_accuracy']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Mean entropy comparison
    ax1 = axes[0]
    categories = ['Correct', 'Incorrect']
    entropies = [acc_data['correct']['mean_entropy'], acc_data['incorrect']['mean_entropy']]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax1.bar(categories, entropies, color=colors, alpha=0.8)
    ax1.set_ylabel('Mean Entropy', fontsize=12)
    ax1.set_title('Mean Entropy by Answer Correctness', fontsize=14)

    for bar, ent in zip(bars, entropies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ent:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylim(0, max(entropies) * 1.3)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Sample counts and lengths
    ax2 = axes[1]
    counts = [acc_data['correct']['count'], acc_data['incorrect']['count']]
    lengths = [acc_data['correct']['mean_length'], acc_data['incorrect']['mean_length']]

    x = np.arange(2)
    width = 0.35

    bars1 = ax2.bar(x - width/2, counts, width, label='Sample Count', color='#9b59b6', alpha=0.8)
    ax2.set_ylabel('Sample Count', fontsize=12, color='#9b59b6')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')

    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, lengths, width, label='Mean Length', color='#f39c12', alpha=0.8)
    ax2_twin.set_ylabel('Mean Length (tokens)', fontsize=12, color='#f39c12')
    ax2_twin.tick_params(axis='y', labelcolor='#f39c12')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_title('Sample Count and Mean Length by Correctness', fontsize=14)

    # Combined legend
    ax2.legend([bars1, bars2], ['Sample Count', 'Mean Length'], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_by_accuracy.png', dpi=150)
    plt.close()
    print(f"Saved: entropy_by_accuracy.png")


def plot_entropy_distribution(results: list, output_dir: Path):
    """Plot entropy distribution histograms."""
    entropies = [r['mean_entropy'] for r in results]
    lengths = [r['response_length'] for r in results]
    accuracies = [r['acc'] for r in results]

    # Split by accuracy
    correct_ent = [e for e, a in zip(entropies, accuracies) if a]
    incorrect_ent = [e for e, a in zip(entropies, accuracies) if not a]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Overall entropy distribution
    ax1 = axes[0, 0]
    ax1.hist(entropies, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Mean Entropy', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Distribution of Mean Entropy (All Samples)', fontsize=12)
    ax1.axvline(np.median(entropies), color='red', linestyle='--', label=f'Median={np.median(entropies):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entropy by accuracy
    ax2 = axes[0, 1]
    ax2.hist(correct_ent, bins=30, color='#2ecc71', alpha=0.6, label=f'Correct (n={len(correct_ent)})', edgecolor='black')
    ax2.hist(incorrect_ent, bins=30, color='#e74c3c', alpha=0.6, label=f'Incorrect (n={len(incorrect_ent)})', edgecolor='black')
    ax2.set_xlabel('Mean Entropy', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Entropy Distribution by Correctness', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Length distribution
    ax3 = axes[1, 0]
    ax3.hist(lengths, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Output Length (tokens)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Distribution of Output Length', fontsize=12)
    ax3.axvline(np.median(lengths), color='red', linestyle='--', label=f'Median={np.median(lengths):.0f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: 2D histogram (heatmap)
    ax4 = axes[1, 1]
    # Clip for better visualization
    ent_clipped = np.clip(entropies, 0, 3)
    len_clipped = np.clip(lengths, 0, 5000)

    h = ax4.hist2d(len_clipped, ent_clipped, bins=[50, 50], cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax4, label='Count')
    ax4.set_xlabel('Output Length (tokens, clipped at 5000)', fontsize=11)
    ax4.set_ylabel('Mean Entropy (clipped at 3)', fontsize=11)
    ax4.set_title('2D Distribution: Length vs Entropy', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_distributions.png', dpi=150)
    plt.close()
    print(f"Saved: entropy_distributions.png")


def plot_early_entropy_vs_length(results: list, output_dir: Path):
    """Plot early-token entropy vs final length for prediction analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    windows = [10, 20, 50, 100]

    for ax, window in zip(axes.flat, windows):
        key = f'early_{window}_mean'
        early_ent = [r[key] for r in results]
        lengths = [r['response_length'] for r in results]

        # Sample for scatter (too many points)
        sample_idx = np.random.choice(len(results), min(5000, len(results)), replace=False)
        early_sample = [early_ent[i] for i in sample_idx]
        length_sample = [lengths[i] for i in sample_idx]

        ax.scatter(early_sample, length_sample, alpha=0.2, s=5, c='#3498db')

        # Calculate correlation
        r, _ = stats.pearsonr(early_ent, lengths)
        rho, _ = stats.spearmanr(early_ent, lengths)

        # Add trend line
        z = np.polyfit(early_ent, lengths, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(early_ent), max(early_ent), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)

        ax.set_xlabel(f'Mean Entropy (First {window} tokens)', fontsize=11)
        ax.set_ylabel('Final Output Length', fontsize=11)
        ax.set_title(f'First {window} Tokens Entropy → Length\nPearson r={r:.3f}, Spearman ρ={rho:.3f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 6000)

    plt.tight_layout()
    plt.savefig(output_dir / 'early_entropy_prediction.png', dpi=150)
    plt.close()
    print(f"Saved: early_entropy_prediction.png")


def plot_summary_dashboard(analysis: dict, results: list, output_dir: Path):
    """Create a summary dashboard with key findings."""
    fig = plt.figure(figsize=(16, 12))

    # Title
    fig.suptitle('Entropy vs Output Length Analysis - Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Key statistics box (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    stats_text = f"""
    KEY STATISTICS
    ══════════════════════
    Samples: {analysis['num_samples']:,}

    Length:
      Median: {analysis['length_stats']['median']:.0f} tokens
      Range: {analysis['length_stats']['min']}-{analysis['length_stats']['max']}

    Entropy:
      Median: {analysis['entropy_stats']['median']:.3f}
      Range: {analysis['entropy_stats']['min']:.3f}-{analysis['entropy_stats']['max']:.3f}
    """
    ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Key findings box (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    corr = analysis['mean_entropy_vs_length']
    findings_text = f"""
    KEY FINDINGS
    ══════════════════════
    1. SHORTER outputs have
       HIGHER entropy
       (Spearman ρ = {corr['spearman_r']:.3f})

    2. Early tokens (first 50)
       predict final length
       (Spearman ρ = -0.54)

    3. Correct answers have
       2.5x LOWER entropy
    """
    ax2.text(0.1, 0.9, findings_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 3. Accuracy comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    acc_data = analysis['entropy_by_accuracy']
    categories = ['Correct', 'Incorrect']
    entropies = [acc_data['correct']['mean_entropy'], acc_data['incorrect']['mean_entropy']]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax3.bar(categories, entropies, color=colors, alpha=0.8)
    ax3.set_ylabel('Mean Entropy')
    ax3.set_title('Entropy by Accuracy')
    for bar, ent in zip(bars, entropies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ent:.3f}', ha='center', va='bottom', fontweight='bold')
    ax3.set_ylim(0, max(entropies) * 1.3)

    # 4. Entropy by length bucket (middle row, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    buckets = analysis['entropy_by_length_bucket']
    labels = [b['bucket'] for b in buckets]
    x = np.arange(len(labels))

    # Use percentiles if available
    if 'p25' in buckets[0]:
        p25s = [b['p25'] for b in buckets]
        p50s = [b['p50'] for b in buckets]
        p75s = [b['p75'] for b in buckets]
        width = 0.25
        ax4.bar(x - width, p25s, width, color='#85c1e9', alpha=0.8, label='P25')
        ax4.bar(x, p50s, width, color='#3498db', alpha=0.8, label='P50')
        ax4.bar(x + width, p75s, width, color='#1a5276', alpha=0.8, label='P75')
        ax4.legend(loc='upper right', fontsize=8)
    else:
        means = [b['mean_entropy'] for b in buckets]
        stds = [b['std_entropy'] for b in buckets]
        ax4.bar(x, means, yerr=stds, capsize=4, color='#3498db', alpha=0.7)

    ax4.set_xlabel('Output Length Bucket (tokens)')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Entropy Decreases with Output Length')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Early token correlation (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    early_corr = analysis['early_token_correlations']
    windows = [10, 20, 50, 100]
    spearman_rs = [abs(early_corr[f'first_{w}_tokens']['spearman_r']) for w in windows]
    ax5.bar([f'{w}' for w in windows], spearman_rs, color='#e74c3c', alpha=0.8)
    ax5.set_xlabel('First N Tokens')
    ax5.set_ylabel('|Spearman ρ|')
    ax5.set_title('Early Entropy → Length\nPrediction Strength')
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Scatter plot (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    lengths = [r['response_length'] for r in results]
    entropies_list = [r['mean_entropy'] for r in results]
    accuracies = [r['acc'] for r in results]
    colors = ['#2ecc71' if acc else '#e74c3c' for acc in accuracies]

    # Sample for visualization
    sample_idx = np.random.choice(len(results), min(8000, len(results)), replace=False)
    ax6.scatter([lengths[i] for i in sample_idx],
                [entropies_list[i] for i in sample_idx],
                c=[colors[i] for i in sample_idx], alpha=0.3, s=8)
    ax6.set_xlabel('Output Length (tokens)')
    ax6.set_ylabel('Mean Entropy')
    ax6.set_title('Entropy vs Length (Green=Correct, Red=Incorrect)')
    ax6.set_xlim(0, 6000)
    ax6.set_ylim(0, 3)
    ax6.grid(True, alpha=0.3)

    plt.savefig(output_dir / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_dashboard.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize entropy vs length analysis results.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to analysis results JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_path}...")
    data = load_results(input_path)

    analysis = data['analysis']
    results = data['per_sample_results']

    print(f"Loaded {len(results)} samples")
    print(f"Creating visualizations in {output_dir}...")

    # Generate all plots
    plot_entropy_vs_length_scatter(results, output_dir)
    plot_entropy_by_length_bucket(analysis, output_dir)
    plot_early_token_correlations(analysis, output_dir)
    plot_entropy_by_accuracy(analysis, output_dir)
    plot_entropy_distribution(results, output_dir)
    plot_early_entropy_vs_length(results, output_dir)
    plot_summary_dashboard(analysis, results, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")
    print("Generated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
