#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Analyze prompt perplexity (NLL) and its relationship to output length.

Perplexity = exp(average NLL) = exp(-1/N * Σ log(p(token_i)))

This measures how "surprising" or "difficult" the prompt is to the model.

Usage:
    python -m recipe.srt.scripts.rollout_analysis.analyze_prompt_perplexity \
        --data_dir /path/to/rollout_data \
        --model Qwen/Qwen3-8B-Base \
        --output_dir /path/to/output
"""

import argparse
import json
import hashlib
import math
from pathlib import Path
from collections import defaultdict
import statistics

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats
import matplotlib.pyplot as plt


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (first 16 hex chars)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def compute_prompt_perplexity(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> dict:
    """Compute perplexity and NLL statistics for a prompt.

    Returns:
        Dict with:
        - prompt_length: number of tokens
        - total_nll: sum of negative log likelihoods
        - mean_nll: average NLL per token
        - perplexity: exp(mean_nll)
        - nll_first_half: NLL for first half of prompt
        - nll_second_half: NLL for second half of prompt
    """
    # Tokenize prompt
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    prompt_length = prompt_ids.shape[1]

    if prompt_length < 2:
        return {
            'prompt_length': prompt_length,
            'prompt_char_length': len(prompt),
            'total_nll': 0,
            'mean_nll': 0,
            'perplexity': 1.0,
            'nll_first_half': 0,
            'nll_second_half': 0,
        }

    input_ids = prompt_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, prompt_len, vocab_size)

    # Compute NLL for each token (except the first, which has no context)
    # logits[i] predicts token[i+1]
    # So we use logits[0:-1] to predict tokens[1:]

    shift_logits = logits[0, :-1, :]  # (prompt_len-1, vocab_size)
    shift_labels = input_ids[0, 1:]    # (prompt_len-1,)

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Get the log prob of the actual tokens
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)  # (prompt_len-1,)

    # NLL is negative log probability
    token_nlls = -token_log_probs  # (prompt_len-1,)

    total_nll = token_nlls.sum().item()
    mean_nll = token_nlls.mean().item()
    perplexity = math.exp(mean_nll)

    # Split into first and second half
    mid = len(token_nlls) // 2
    nll_first_half = token_nlls[:mid].mean().item() if mid > 0 else 0
    nll_second_half = token_nlls[mid:].mean().item() if mid > 0 else mean_nll

    # Also compute entropy at last position (for comparison)
    last_logits = logits[0, -1, :]  # (vocab_size,)
    pd = torch.nn.functional.softmax(last_logits, dim=-1)
    last_entropy = (torch.logsumexp(last_logits, dim=-1) - torch.sum(pd * last_logits, dim=-1)).item()

    return {
        'prompt_length': prompt_length,
        'prompt_char_length': len(prompt),
        'total_nll': total_nll,
        'mean_nll': mean_nll,
        'perplexity': perplexity,
        'nll_first_half': nll_first_half,
        'nll_second_half': nll_second_half,
        'last_entropy': last_entropy,
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

    # Group by prompt to avoid duplicates
    prompt_data = defaultdict(lambda: {'responses': [], 'accs': [], 'prompt': None})

    for step_file in sorted(rollout_dir.glob("*.jsonl"), key=lambda x: int(x.stem)):
        with open(step_file) as f:
            for line in f:
                item = json.loads(line)
                prompt_hash = compute_prompt_hash(item['input'])
                prompt_data[prompt_hash]['prompt'] = item['input']
                prompt_data[prompt_hash]['responses'].append(item['output'])
                prompt_data[prompt_hash]['accs'].append(item.get('acc', False))

    # Create one sample per unique prompt
    samples = []
    for prompt_hash, data in prompt_data.items():
        if len(samples) >= max_samples:
            break

        # Use tokenized length instead of character length
        output_token_lengths = [
            len(tokenizer.encode(r, add_special_tokens=False))
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


def analyze_prompts(
    model,
    tokenizer,
    samples: list,
    device: str = "cuda",
    verbose: bool = False,
) -> list:
    """Compute perplexity for all prompts."""

    results = []
    iterator = tqdm(samples, desc="Computing perplexity") if verbose else samples

    for sample in iterator:
        try:
            features = compute_prompt_perplexity(model, tokenizer, sample['prompt'], device)
            result = {
                **features,
                'prompt_hash': sample['prompt_hash'],
                'mean_output_length': sample['mean_output_length'],  # tokenized
                'acc_rate': sample['acc_rate'],
            }
            results.append(result)
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            continue

    return results


def compute_correlations(results: list) -> dict:
    """Compute correlations between perplexity and output length (tokenized)."""

    perplexities = [r['perplexity'] for r in results]
    mean_nlls = [r['mean_nll'] for r in results]
    output_lengths = [r['mean_output_length'] for r in results]  # tokenized
    acc_rates = [r['acc_rate'] for r in results]
    prompt_lengths = [r['prompt_length'] for r in results]
    nll_first_halves = [r['nll_first_half'] for r in results]
    nll_second_halves = [r['nll_second_half'] for r in results]
    last_entropies = [r['last_entropy'] for r in results]

    analysis = {
        'num_prompts': len(results),
        'perplexity_stats': {
            'min': min(perplexities),
            'max': max(perplexities),
            'mean': statistics.mean(perplexities),
            'median': statistics.median(perplexities),
            'std': statistics.stdev(perplexities),
        },
        'mean_nll_stats': {
            'min': min(mean_nlls),
            'max': max(mean_nlls),
            'mean': statistics.mean(mean_nlls),
            'median': statistics.median(mean_nlls),
        },
    }

    # Correlations with output length
    correlations = {}

    # Perplexity vs output length
    r, p = stats.pearsonr(perplexities, output_lengths)
    rho, rho_p = stats.spearmanr(perplexities, output_lengths)
    correlations['perplexity_vs_output_length'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    # Mean NLL vs output length
    r, p = stats.pearsonr(mean_nlls, output_lengths)
    rho, rho_p = stats.spearmanr(mean_nlls, output_lengths)
    correlations['mean_nll_vs_output_length'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    # Prompt length vs output length (for comparison)
    r, p = stats.pearsonr(prompt_lengths, output_lengths)
    rho, rho_p = stats.spearmanr(prompt_lengths, output_lengths)
    correlations['prompt_length_vs_output_length'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    # NLL first half vs output length
    r, p = stats.pearsonr(nll_first_halves, output_lengths)
    rho, rho_p = stats.spearmanr(nll_first_halves, output_lengths)
    correlations['nll_first_half_vs_output_length'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    # NLL second half vs output length
    r, p = stats.pearsonr(nll_second_halves, output_lengths)
    rho, rho_p = stats.spearmanr(nll_second_halves, output_lengths)
    correlations['nll_second_half_vs_output_length'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    # Last position entropy vs output length (for comparison with perplexity)
    r, p = stats.pearsonr(last_entropies, output_lengths)
    rho, rho_p = stats.spearmanr(last_entropies, output_lengths)
    correlations['last_entropy_vs_output_length'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    # Perplexity vs accuracy
    r, p = stats.pearsonr(perplexities, acc_rates)
    rho, rho_p = stats.spearmanr(perplexities, acc_rates)
    correlations['perplexity_vs_accuracy'] = {
        'pearson_r': r, 'pearson_p': p, 'spearman_r': rho, 'spearman_p': rho_p
    }

    analysis['correlations'] = correlations

    return analysis


def plot_analysis(results: list, analysis: dict, output_dir: Path):
    """Create visualizations."""

    perplexities = [r['perplexity'] for r in results]
    mean_nlls = [r['mean_nll'] for r in results]
    output_lengths = [r['mean_output_length'] for r in results]  # tokenized
    acc_rates = [r['acc_rate'] for r in results]
    prompt_lengths = [r['prompt_length'] for r in results]
    last_entropies = [r['last_entropy'] for r in results]

    fig = plt.figure(figsize=(18, 12))

    # 1. Perplexity vs output length
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(perplexities, output_lengths, alpha=0.3, s=10, c='#e74c3c')
    corr = analysis['correlations']['perplexity_vs_output_length']
    ax1.set_xlabel('Prompt Perplexity', fontsize=11)
    ax1.set_ylabel('Mean Output Length (tokens)', fontsize=11)
    ax1.set_title(f'Prompt Perplexity vs Output Length\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    # Clip for visualization
    ax1.set_xlim(0, min(50, max(perplexities)))

    # 2. Mean NLL vs output length
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(mean_nlls, output_lengths, alpha=0.3, s=10, c='#3498db')
    corr = analysis['correlations']['mean_nll_vs_output_length']
    ax2.set_xlabel('Mean NLL (per token)', fontsize=11)
    ax2.set_ylabel('Mean Output Length (tokens)', fontsize=11)
    ax2.set_title(f'Mean NLL vs Output Length\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Perplexity distribution
    ax3 = fig.add_subplot(2, 3, 3)
    # Clip extreme values for visualization
    ppl_clipped = [min(p, 50) for p in perplexities]
    ax3.hist(ppl_clipped, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax3.axvline(analysis['perplexity_stats']['median'], color='blue', linestyle='--',
                label=f"Median={analysis['perplexity_stats']['median']:.2f}")
    ax3.set_xlabel('Prompt Perplexity (clipped at 50)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Distribution of Prompt Perplexity', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Correlation comparison
    ax4 = fig.add_subplot(2, 3, 4)
    features = ['Prompt\nLength', 'Prompt\nPerplexity', 'Mean\nNLL', 'Last Pos\nEntropy']
    corr_keys = ['prompt_length_vs_output_length', 'perplexity_vs_output_length',
                 'mean_nll_vs_output_length', 'last_entropy_vs_output_length']
    spearman_rs = [analysis['correlations'][k]['spearman_r'] for k in corr_keys]

    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    bars = ax4.bar(features, spearman_rs, color=colors, alpha=0.8)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_ylabel('Spearman ρ with Output Length', fontsize=11)
    ax4.set_title('Comparison: What Predicts Output Length?', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, r in zip(bars, spearman_rs):
        ax4.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.01 if r >= 0 else -0.03),
                f'{r:.4f}', ha='center', va='bottom' if r >= 0 else 'top',
                fontweight='bold', fontsize=9)

    # 5. Perplexity vs Accuracy
    ax5 = fig.add_subplot(2, 3, 5)
    colors = ['#2ecc71' if acc >= 0.5 else '#e74c3c' for acc in acc_rates]
    ax5.scatter(perplexities, acc_rates, alpha=0.4, s=15, c=colors)
    corr = analysis['correlations']['perplexity_vs_accuracy']
    ax5.set_xlabel('Prompt Perplexity', fontsize=11)
    ax5.set_ylabel('Accuracy Rate', fontsize=11)
    ax5.set_title(f'Perplexity vs Accuracy\nSpearman ρ={corr["spearman_r"]:.4f}', fontsize=12)
    ax5.set_xlim(0, min(50, max(perplexities)))
    ax5.grid(True, alpha=0.3)

    # 6. NLL vs Entropy comparison
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.scatter(mean_nlls, last_entropies, alpha=0.3, s=10, c='#f39c12')
    ax6.set_xlabel('Mean NLL (prompt)', fontsize=11)
    ax6.set_ylabel('Last Position Entropy', fontsize=11)
    ax6.set_title('NLL vs Entropy Comparison\n(Both at prefill time)', fontsize=12)
    ax6.grid(True, alpha=0.3)

    # Add correlation
    r, _ = stats.spearmanr(mean_nlls, last_entropies)
    ax6.text(0.05, 0.95, f'Spearman ρ={r:.3f}', transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Prompt Perplexity (NLL) Analysis: Can Prompt "Difficulty" Predict Output Length?',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'prompt_perplexity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: prompt_perplexity_analysis.png")


def print_analysis(analysis: dict):
    """Print analysis results."""

    print("\n" + "=" * 80)
    print("PROMPT PERPLEXITY (NLL) ANALYSIS")
    print("=" * 80)

    print(f"\nAnalyzed {analysis['num_prompts']} unique prompts")

    ps = analysis['perplexity_stats']
    print(f"\nPrompt Perplexity Statistics:")
    print(f"  Min:    {ps['min']:.2f}")
    print(f"  Max:    {ps['max']:.2f}")
    print(f"  Mean:   {ps['mean']:.2f}")
    print(f"  Median: {ps['median']:.2f}")
    print(f"  Std:    {ps['std']:.2f}")

    ns = analysis['mean_nll_stats']
    print(f"\nMean NLL Statistics:")
    print(f"  Min:    {ns['min']:.4f}")
    print(f"  Max:    {ns['max']:.4f}")
    print(f"  Mean:   {ns['mean']:.4f}")
    print(f"  Median: {ns['median']:.4f}")

    print("\n" + "-" * 60)
    print("CORRELATIONS WITH OUTPUT LENGTH")
    print("-" * 60)

    for name, corr in analysis['correlations'].items():
        if 'output_length' in name:
            feature = name.replace('_vs_output_length', '').replace('_', ' ').title()
            print(f"\n{feature}:")
            print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
            print(f"  Spearman ρ: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    print("\n" + "-" * 60)
    print("PERPLEXITY vs ACCURACY")
    print("-" * 60)
    corr = analysis['correlations']['perplexity_vs_accuracy']
    print(f"  Spearman ρ: {corr['spearman_r']:.4f}")

    if corr['spearman_r'] < -0.1:
        print("  → Higher perplexity prompts tend to have LOWER accuracy")
    elif corr['spearman_r'] > 0.1:
        print("  → Higher perplexity prompts tend to have HIGHER accuracy")
    else:
        print("  → No strong relationship between perplexity and accuracy")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt perplexity vs output length.")
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

    print("\nComputing prompt perplexity...")
    results = analyze_prompts(model, tokenizer, samples, args.device, args.verbose)
    print(f"Processed {len(results)} prompts")

    print("\nComputing correlations...")
    analysis = compute_correlations(results)

    print_analysis(analysis)

    print("\nCreating visualizations...")
    plot_analysis(results, analysis, output_dir)

    # Save results
    output_json = output_dir / 'prompt_perplexity_analysis.json'
    with open(output_json, 'w') as f:
        json.dump({
            'analysis': analysis,
            'per_prompt_results': results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_json}")


if __name__ == "__main__":
    main()
