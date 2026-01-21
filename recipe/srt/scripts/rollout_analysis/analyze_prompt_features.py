#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Analyze prompt features and their relationship to output length.

This script examines whether prompt characteristics can predict output length
BEFORE generation starts (at prefill time).

Features analyzed:
1. Prompt length (tokens and characters)
2. Prompt complexity indicators

Usage:
    python -m recipe.srt.scripts.rollout_analysis.analyze_prompt_features \
        --data_dir /path/to/rollout_data \
        --model Qwen/Qwen3-8B-Base \
        --output_dir /path/to/output
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import statistics

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (first 16 hex chars)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate per-token entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_prompt_features(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> dict:
    """Compute features from prompt (before any generation).

    Returns:
        Dict with prompt_length, prompt_char_length, prompt_entropy (mean entropy of prompt tokens),
        last_prompt_token_entropy (entropy at the position that will generate first output token)
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    prompt_length = prompt_ids.shape[1]

    if prompt_length == 0:
        return {
            'prompt_length': 0,
            'prompt_char_length': len(prompt),
            'prompt_entropy_mean': 0,
            'prompt_entropy_last': 0,
            'prompt_entropy_last5_mean': 0,
        }

    # Forward pass on prompt only
    input_ids = prompt_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, prompt_len, vocab_size)

    # Compute entropy for each position
    # logits[i] predicts token[i+1], so:
    # - logits[:-1] predict the prompt tokens (positions 1 to end)
    # - logits[-1] predicts the FIRST output token (this is what we want!)

    prompt_logits = logits[0, :-1, :]  # (prompt_len-1, vocab_size) - predicts prompt tokens
    last_logit = logits[0, -1:, :]  # (1, vocab_size) - predicts first output token

    prompt_entropy = entropy_from_logits(prompt_logits.float())  # (prompt_len-1,)
    last_entropy = entropy_from_logits(last_logit.float())  # (1,)

    # Also get entropy of last few prompt positions
    if len(prompt_entropy) >= 5:
        last5_entropy = prompt_entropy[-5:].mean().item()
    else:
        last5_entropy = prompt_entropy.mean().item() if len(prompt_entropy) > 0 else 0

    return {
        'prompt_length': prompt_length,
        'prompt_char_length': len(prompt),
        'prompt_entropy_mean': prompt_entropy.mean().item() if len(prompt_entropy) > 0 else 0,
        'prompt_entropy_last': last_entropy[0].item(),  # This predicts the FIRST output token
        'prompt_entropy_last5_mean': last5_entropy,
    }


def load_rollout_samples(
    data_dir: Path,
    tokenizer,
    max_samples: int = 100000,
    verbose: bool = False,
) -> list:
    """Load rollout samples from data directory.

    Note: Uses tokenized output length (number of tokens) instead of character length.
    """
    rollout_dir = data_dir / "rollout"
    samples = []

    # Group by prompt to avoid duplicates
    prompt_data = defaultdict(lambda: {'responses': [], 'accs': [], 'prompt': None})

    for step_file in sorted(rollout_dir.glob("*.jsonl"), key=lambda x: int(x.stem)):
        step = int(step_file.stem)
        with open(step_file) as f:
            for line in f:
                item = json.loads(line)
                prompt_hash = compute_prompt_hash(item['input'])
                prompt_data[prompt_hash]['prompt'] = item['input']
                prompt_data[prompt_hash]['responses'].append({
                    'output': item['output'],
                    'acc': item.get('acc', False),
                    'step': step,
                })
                prompt_data[prompt_hash]['accs'].append(item.get('acc', False))

    # Create one sample per unique prompt with aggregated stats
    for prompt_hash, data in prompt_data.items():
        if len(samples) >= max_samples:
            break

        # Calculate mean output length (tokenized) for this prompt
        output_token_lengths = [
            len(tokenizer.encode(r['output'], add_special_tokens=False))
            for r in data['responses']
        ]

        samples.append({
            'prompt': data['prompt'],
            'prompt_hash': prompt_hash,
            'num_responses': len(data['responses']),
            'mean_output_length': statistics.mean(output_token_lengths),  # tokenized
            'acc_rate': sum(data['accs']) / len(data['accs']) if data['accs'] else 0,
        })

    if verbose:
        print(f"Loaded {len(samples)} unique prompts")

    return samples


def analyze_prompt_features(
    model,
    tokenizer,
    samples: list,
    device: str = "cuda",
    verbose: bool = False,
) -> list:
    """Compute prompt features for all samples."""

    results = []
    iterator = tqdm(samples, desc="Computing prompt features") if verbose else samples

    for sample in iterator:
        try:
            features = compute_prompt_features(model, tokenizer, sample['prompt'], device)

            # Tokenize a sample output to get output token length
            # (Use first response as representative)

            result = {
                **features,
                'prompt_hash': sample['prompt_hash'],
                'mean_output_length': sample['mean_output_length'],
                'acc_rate': sample['acc_rate'],
                'num_responses': sample['num_responses'],
            }
            results.append(result)

        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue

    return results


def compute_correlations(results: list) -> dict:
    """Compute correlations between prompt features and output length."""

    prompt_lengths = [r['prompt_length'] for r in results]
    prompt_char_lengths = [r['prompt_char_length'] for r in results]
    prompt_entropy_means = [r['prompt_entropy_mean'] for r in results]
    prompt_entropy_lasts = [r['prompt_entropy_last'] for r in results]  # Predicts first output token
    prompt_entropy_last5s = [r['prompt_entropy_last5_mean'] for r in results]
    output_lengths = [r['mean_output_length'] for r in results]
    acc_rates = [r['acc_rate'] for r in results]

    analysis = {
        'num_prompts': len(results),
        'prompt_length_stats': {
            'min': min(prompt_lengths),
            'max': max(prompt_lengths),
            'mean': statistics.mean(prompt_lengths),
            'median': statistics.median(prompt_lengths),
        },
        'output_length_stats': {
            'min': min(output_lengths),
            'max': max(output_lengths),
            'mean': statistics.mean(output_lengths),
            'median': statistics.median(output_lengths),
        },
    }

    # Correlations with output length
    correlations = {}

    # Prompt length vs output length
    r, p = stats.pearsonr(prompt_lengths, output_lengths)
    rho, rho_p = stats.spearmanr(prompt_lengths, output_lengths)
    correlations['prompt_length'] = {'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p}

    # Prompt char length vs output length
    r, p = stats.pearsonr(prompt_char_lengths, output_lengths)
    rho, rho_p = stats.spearmanr(prompt_char_lengths, output_lengths)
    correlations['prompt_char_length'] = {'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p}

    # Prompt mean entropy vs output length
    r, p = stats.pearsonr(prompt_entropy_means, output_lengths)
    rho, rho_p = stats.spearmanr(prompt_entropy_means, output_lengths)
    correlations['prompt_entropy_mean'] = {'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p}

    # Last position entropy (predicts first output token) vs output length
    r, p = stats.pearsonr(prompt_entropy_lasts, output_lengths)
    rho, rho_p = stats.spearmanr(prompt_entropy_lasts, output_lengths)
    correlations['prompt_entropy_last'] = {'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p}

    # Last 5 positions entropy vs output length
    r, p = stats.pearsonr(prompt_entropy_last5s, output_lengths)
    rho, rho_p = stats.spearmanr(prompt_entropy_last5s, output_lengths)
    correlations['prompt_entropy_last5'] = {'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p}

    # Accuracy correlations
    r, p = stats.pearsonr(prompt_entropy_lasts, acc_rates)
    rho, rho_p = stats.spearmanr(prompt_entropy_lasts, acc_rates)
    correlations['prompt_entropy_last_vs_acc'] = {'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p}

    analysis['correlations'] = correlations

    # Entropy stats
    analysis['prompt_entropy_last_stats'] = {
        'min': min(prompt_entropy_lasts),
        'max': max(prompt_entropy_lasts),
        'mean': statistics.mean(prompt_entropy_lasts),
        'median': statistics.median(prompt_entropy_lasts),
        'std': statistics.stdev(prompt_entropy_lasts),
    }

    return analysis


def plot_prompt_analysis(results: list, analysis: dict, output_dir: Path):
    """Create visualizations for prompt feature analysis."""

    prompt_lengths = [r['prompt_length'] for r in results]
    prompt_entropy_lasts = [r['prompt_entropy_last'] for r in results]
    prompt_entropy_means = [r['prompt_entropy_mean'] for r in results]
    output_lengths = [r['mean_output_length'] for r in results]
    acc_rates = [r['acc_rate'] for r in results]

    fig = plt.figure(figsize=(18, 12))

    # 1. Prompt length vs output length
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(prompt_lengths, output_lengths, alpha=0.3, s=10, c='#3498db')
    corr = analysis['correlations']['prompt_length']
    ax1.set_xlabel('Prompt Length (tokens)', fontsize=11)
    ax1.set_ylabel('Mean Output Length (tokens)', fontsize=11)
    ax1.set_title(f'Prompt Length vs Output Length\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(prompt_lengths, output_lengths, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(prompt_lengths), max(prompt_lengths), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2)

    # 2. Last position entropy vs output length (KEY: This is at prefill time!)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(prompt_entropy_lasts, output_lengths, alpha=0.3, s=10, c='#e74c3c')
    corr = analysis['correlations']['prompt_entropy_last']
    ax2.set_xlabel('Entropy at Last Prompt Position\n(Predicts First Output Token)', fontsize=11)
    ax2.set_ylabel('Mean Output Length (tokens)', fontsize=11)
    ax2.set_title(f'Prefill Entropy vs Output Length\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax2.grid(True, alpha=0.3)

    z = np.polyfit(prompt_entropy_lasts, output_lengths, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(prompt_entropy_lasts), max(prompt_entropy_lasts), 100)
    ax2.plot(x_line, p(x_line), 'b-', linewidth=2)

    # 3. Prompt mean entropy vs output length
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(prompt_entropy_means, output_lengths, alpha=0.3, s=10, c='#2ecc71')
    corr = analysis['correlations']['prompt_entropy_mean']
    ax3.set_xlabel('Mean Prompt Entropy', fontsize=11)
    ax3.set_ylabel('Mean Output Length (tokens)', fontsize=11)
    ax3.set_title(f'Prompt Entropy vs Output Length\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # 4. Correlation comparison bar chart
    ax4 = fig.add_subplot(2, 3, 4)
    features = ['Prompt\nLength', 'Prompt\nEntropy\n(mean)', 'Prefill\nEntropy\n(last pos)', 'Prefill\nEntropy\n(last 5)']
    corr_keys = ['prompt_length', 'prompt_entropy_mean', 'prompt_entropy_last', 'prompt_entropy_last5']
    spearman_rs = [abs(analysis['correlations'][k]['spearman_r']) for k in corr_keys]

    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax4.bar(features, spearman_rs, color=colors, alpha=0.8)
    ax4.set_ylabel('|Spearman ρ| with Output Length', fontsize=11)
    ax4.set_title('Prompt Features → Output Length\nPrediction Power Comparison', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, r in zip(bars, spearman_rs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{r:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 5. Last position entropy distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(prompt_entropy_lasts, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    stats_info = analysis['prompt_entropy_last_stats']
    ax5.axvline(stats_info['median'], color='blue', linestyle='--',
                label=f"Median={stats_info['median']:.3f}")
    ax5.set_xlabel('Entropy at Last Prompt Position', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Distribution of Prefill Entropy\n(Entropy for First Output Token Prediction)', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Prefill entropy vs accuracy
    ax6 = fig.add_subplot(2, 3, 6)

    # Bin by accuracy
    high_acc = [r['prompt_entropy_last'] for r in results if r['acc_rate'] >= 0.5]
    low_acc = [r['prompt_entropy_last'] for r in results if r['acc_rate'] < 0.5]

    ax6.hist(high_acc, bins=30, alpha=0.6, label=f'High acc (≥50%, n={len(high_acc)})', color='#2ecc71')
    ax6.hist(low_acc, bins=30, alpha=0.6, label=f'Low acc (<50%, n={len(low_acc)})', color='#e74c3c')
    ax6.set_xlabel('Entropy at Last Prompt Position', fontsize=11)
    ax6.set_ylabel('Count', fontsize=11)
    ax6.set_title('Prefill Entropy by Accuracy', fontsize=12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Prompt Feature Analysis: Can We Predict Output Length at Prefill Time?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'prompt_feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: prompt_feature_analysis.png")


def print_analysis(analysis: dict):
    """Print analysis results."""

    print("\n" + "=" * 80)
    print("PROMPT FEATURE ANALYSIS")
    print("Can we predict output length at PREFILL time (before any generation)?")
    print("=" * 80)

    print(f"\nAnalyzed {analysis['num_prompts']} unique prompts")

    print(f"\nPrompt Length (tokens):")
    ps = analysis['prompt_length_stats']
    print(f"  Range: [{ps['min']}, {ps['max']}], Mean: {ps['mean']:.1f}, Median: {ps['median']:.1f}")

    print(f"\nOutput Length (tokens):")
    os = analysis['output_length_stats']
    print(f"  Range: [{os['min']:.0f}, {os['max']:.0f}], Mean: {os['mean']:.1f}, Median: {os['median']:.1f}")

    print("\n" + "-" * 60)
    print("CORRELATIONS WITH OUTPUT LENGTH")
    print("-" * 60)

    for name, corr in analysis['correlations'].items():
        if 'vs_acc' not in name:
            print(f"\n{name}:")
            print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
            print(f"  Spearman ρ: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    print("\n" + "-" * 60)
    print("KEY FINDING: Prefill Entropy (last position)")
    print("-" * 60)
    es = analysis['prompt_entropy_last_stats']
    print(f"\nThis is the entropy at the last prompt position,")
    print(f"which predicts the FIRST output token.")
    print(f"\n  Mean:   {es['mean']:.4f}")
    print(f"  Median: {es['median']:.4f}")
    print(f"  Std:    {es['std']:.4f}")

    corr = analysis['correlations']['prompt_entropy_last']
    print(f"\nCorrelation with output length:")
    print(f"  Spearman ρ = {corr['spearman_r']:.4f}")

    if abs(corr['spearman_r']) < 0.1:
        print("\n  → WEAK: Prefill entropy has little predictive power for output length")
    elif abs(corr['spearman_r']) < 0.3:
        print("\n  → MODERATE: Some predictive signal exists at prefill time")
    else:
        print("\n  → STRONG: Prefill entropy can help predict output length!")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt features vs output length.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to rollout data directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-Base", help="Model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--max_prompts", type=int, default=2000, help="Maximum unique prompts to analyze")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    print(f"\nLoading rollout samples from {data_dir}...")
    samples = load_rollout_samples(data_dir, tokenizer, args.max_prompts, args.verbose)
    print(f"Loaded {len(samples)} unique prompts")

    print("\nComputing prompt features...")
    results = analyze_prompt_features(model, tokenizer, samples, args.device, args.verbose)
    print(f"Processed {len(results)} prompts")

    print("\nComputing correlations...")
    analysis = compute_correlations(results)

    print_analysis(analysis)

    print("\nCreating visualizations...")
    plot_prompt_analysis(results, analysis, output_dir)

    # Save results
    output_json = output_dir / 'prompt_feature_analysis.json'
    with open(output_json, 'w') as f:
        json.dump({
            'analysis': analysis,
            'per_prompt_results': results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
