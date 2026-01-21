#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Analyze relationship between token entropy and output length.

Hypotheses:
1. Longer outputs have higher mean entropy (more uncertainty = more tokens needed)
2. Early-token entropy can predict final output length

Usage:
    python -m recipe.srt.scripts.rollout_analysis.analyze_entropy_length \
        --data_dir /path/to/rollout_data \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output_json results.json \
        --max_samples 500
"""

import argparse
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import statistics

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy import stats


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (first 16 hex chars)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate per-token entropy from logits.

    Args:
        logits: (seq_len, vocab_size)

    Returns:
        entropy: (seq_len,)
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def compute_entropy_for_response(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda",
) -> Tuple[torch.Tensor, int]:
    """Compute per-token entropy for a response given a prompt.

    Returns:
        Tuple of (entropy_per_token, response_length)
    """
    # Tokenize prompt and response separately
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    response_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors="pt")

    # Concatenate
    input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)
    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]

    if response_len == 0:
        return torch.tensor([]), 0

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Get logits for response tokens (shifted by 1 for autoregressive prediction)
    # logits[t] predicts token[t+1], so for response we need logits[prompt_len-1 : prompt_len+response_len-1]
    response_logits = logits[0, prompt_len-1 : prompt_len+response_len-1, :]  # (response_len, vocab_size)

    # Compute entropy
    entropy = entropy_from_logits(response_logits.float())

    return entropy.cpu(), response_len


def load_rollout_samples(
    data_dir: Path,
    max_samples: int = 500,
    verbose: bool = False,
) -> List[Dict]:
    """Load rollout samples from data directory."""
    rollout_dir = data_dir / "rollout"
    samples = []

    for step_file in sorted(rollout_dir.glob("*.jsonl"), key=lambda x: int(x.stem)):
        step = int(step_file.stem)
        with open(step_file) as f:
            for line in f:
                item = json.loads(line)
                samples.append({
                    'prompt': item['input'],
                    'response': item['output'],
                    'acc': item.get('acc', False),
                    'score': item.get('score', 0),
                    'step': step,
                })

                if len(samples) >= max_samples:
                    break

        if len(samples) >= max_samples:
            break

    if verbose:
        print(f"Loaded {len(samples)} samples")

    return samples


def analyze_entropy_length_correlation(
    model,
    tokenizer,
    samples: List[Dict],
    device: str = "cuda",
    early_token_windows: List[int] = [10, 20, 50, 100],
    verbose: bool = False,
) -> Dict:
    """Analyze correlation between entropy and output length.

    Args:
        model: The language model
        tokenizer: The tokenizer
        samples: List of samples with 'prompt' and 'response' keys
        device: Device to run inference on
        early_token_windows: Window sizes for early-token analysis
        verbose: Print progress

    Returns:
        Dictionary with analysis results
    """
    results = []

    iterator = tqdm(samples, desc="Computing entropy") if verbose else samples

    for sample in iterator:
        prompt = sample['prompt']
        response = sample['response']

        try:
            entropy, response_len = compute_entropy_for_response(
                model, tokenizer, prompt, response, device
            )

            if response_len == 0:
                continue

            entropy_np = entropy.numpy()

            result = {
                'response_length': response_len,
                'mean_entropy': float(np.mean(entropy_np)),
                'median_entropy': float(np.median(entropy_np)),
                'std_entropy': float(np.std(entropy_np)),
                'max_entropy': float(np.max(entropy_np)),
                'min_entropy': float(np.min(entropy_np)),
                'acc': sample.get('acc', False),
                'step': sample.get('step', 0),
            }

            # Early-token entropy for different windows
            for window in early_token_windows:
                if response_len >= window:
                    window_entropy = entropy_np[:window]
                else:
                    window_entropy = entropy_np
                result[f'early_{window}_mean'] = float(np.mean(window_entropy))
                result[f'early_{window}_median'] = float(np.median(window_entropy))
                result[f'early_{window}_p25'] = float(np.percentile(window_entropy, 25))
                result[f'early_{window}_p75'] = float(np.percentile(window_entropy, 75))

            # First token entropy (very early signal)
            result['first_token_entropy'] = float(entropy_np[0]) if len(entropy_np) > 0 else 0

            # Entropy trend (slope of entropy over position)
            if response_len > 10:
                positions = np.arange(response_len)
                slope, _, _, _, _ = stats.linregress(positions, entropy_np)
                result['entropy_slope'] = float(slope)
            else:
                result['entropy_slope'] = 0.0

            results.append(result)

        except Exception as e:
            if verbose:
                print(f"Error processing sample: {e}")
            continue

    return results


def compute_correlations(results: List[Dict], early_token_windows: List[int]) -> Dict:
    """Compute correlation statistics between entropy metrics and output length."""

    if len(results) < 10:
        return {'error': 'Not enough samples for correlation analysis'}

    lengths = [r['response_length'] for r in results]
    mean_entropies = [r['mean_entropy'] for r in results]

    analysis = {
        'num_samples': len(results),
        'length_stats': {
            'min': min(lengths),
            'max': max(lengths),
            'mean': statistics.mean(lengths),
            'median': statistics.median(lengths),
        },
        'entropy_stats': {
            'min': min(mean_entropies),
            'max': max(mean_entropies),
            'mean': statistics.mean(mean_entropies),
            'median': statistics.median(mean_entropies),
        },
    }

    # Correlation: mean entropy vs length
    pearson_r, pearson_p = stats.pearsonr(lengths, mean_entropies)
    spearman_r, spearman_p = stats.spearmanr(lengths, mean_entropies)

    analysis['mean_entropy_vs_length'] = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    }

    # Early-token entropy correlations
    analysis['early_token_correlations'] = {}
    for window in early_token_windows:
        key = f'early_{window}_mean'
        early_entropies = [r[key] for r in results]

        pearson_r, pearson_p = stats.pearsonr(lengths, early_entropies)
        spearman_r, spearman_p = stats.spearmanr(lengths, early_entropies)

        analysis['early_token_correlations'][f'first_{window}_tokens'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        }

    # First token entropy correlation
    first_token_entropies = [r['first_token_entropy'] for r in results]
    pearson_r, pearson_p = stats.pearsonr(lengths, first_token_entropies)
    spearman_r, spearman_p = stats.spearmanr(lengths, first_token_entropies)

    analysis['first_token_entropy_vs_length'] = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    }

    # Entropy slope correlation
    slopes = [r['entropy_slope'] for r in results]
    pearson_r, pearson_p = stats.pearsonr(lengths, slopes)
    spearman_r, spearman_p = stats.spearmanr(lengths, slopes)

    analysis['entropy_slope_vs_length'] = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
    }

    # Entropy by length bucket
    buckets = [(0, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, float('inf'))]
    analysis['entropy_by_length_bucket'] = []

    for low, high in buckets:
        bucket_results = [r for r in results if low <= r['response_length'] < high]
        if bucket_results:
            bucket_entropies = [r['mean_entropy'] for r in bucket_results]
            bucket_entropies_arr = np.array(bucket_entropies)
            analysis['entropy_by_length_bucket'].append({
                'bucket': f"{low}-{high}" if high != float('inf') else f"{low}+",
                'count': len(bucket_results),
                'mean_entropy': statistics.mean(bucket_entropies),
                'std_entropy': statistics.stdev(bucket_entropies) if len(bucket_entropies) > 1 else 0,
                'p25': float(np.percentile(bucket_entropies_arr, 25)),
                'p50': float(np.percentile(bucket_entropies_arr, 50)),
                'p75': float(np.percentile(bucket_entropies_arr, 75)),
            })

    # Accuracy vs entropy
    correct = [r for r in results if r['acc']]
    incorrect = [r for r in results if not r['acc']]

    if correct and incorrect:
        analysis['entropy_by_accuracy'] = {
            'correct': {
                'count': len(correct),
                'mean_entropy': statistics.mean([r['mean_entropy'] for r in correct]),
                'mean_length': statistics.mean([r['response_length'] for r in correct]),
            },
            'incorrect': {
                'count': len(incorrect),
                'mean_entropy': statistics.mean([r['mean_entropy'] for r in incorrect]),
                'mean_length': statistics.mean([r['response_length'] for r in incorrect]),
            },
        }

    return analysis


def print_results(analysis: Dict):
    """Print analysis results in a readable format."""

    print("\n" + "=" * 80)
    print("ENTROPY vs OUTPUT LENGTH ANALYSIS")
    print("=" * 80)

    print(f"\nSamples analyzed: {analysis['num_samples']}")

    print(f"\nLength distribution:")
    ls = analysis['length_stats']
    print(f"  Min: {ls['min']}, Max: {ls['max']}, Mean: {ls['mean']:.1f}, Median: {ls['median']:.1f}")

    print(f"\nEntropy distribution:")
    es = analysis['entropy_stats']
    print(f"  Min: {es['min']:.3f}, Max: {es['max']:.3f}, Mean: {es['mean']:.3f}, Median: {es['median']:.3f}")

    print("\n" + "-" * 40)
    print("HYPOTHESIS 1: Longer outputs have higher entropy")
    print("-" * 40)

    corr = analysis['mean_entropy_vs_length']
    print(f"\nMean entropy vs output length:")
    print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman r: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    if corr['pearson_r'] > 0.1 and corr['pearson_p'] < 0.05:
        print("  -> SUPPORTED: Positive correlation between entropy and length")
    elif corr['pearson_r'] < -0.1 and corr['pearson_p'] < 0.05:
        print("  -> OPPOSITE: Negative correlation (shorter outputs have higher entropy)")
    else:
        print("  -> WEAK/NO correlation found")

    print("\nEntropy by length bucket:")
    for b in analysis['entropy_by_length_bucket']:
        print(f"  {b['bucket']:<12} n={b['count']:>4}  mean_entropy={b['mean_entropy']:.3f} (+/- {b['std_entropy']:.3f})")

    print("\n" + "-" * 40)
    print("HYPOTHESIS 2: Early-token entropy predicts length")
    print("-" * 40)

    print("\nFirst token entropy vs length:")
    corr = analysis['first_token_entropy_vs_length']
    print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman r: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    print("\nEarly-token window correlations:")
    for window, corr in analysis['early_token_correlations'].items():
        print(f"  {window}:")
        print(f"    Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
        print(f"    Spearman r: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    print("\nEntropy slope (trend over position) vs length:")
    corr = analysis['entropy_slope_vs_length']
    print(f"  Pearson r:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.2e})")
    print(f"  Spearman r: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.2e})")

    if 'entropy_by_accuracy' in analysis:
        print("\n" + "-" * 40)
        print("ENTROPY vs ACCURACY")
        print("-" * 40)
        acc = analysis['entropy_by_accuracy']
        print(f"\nCorrect answers (n={acc['correct']['count']}):")
        print(f"  Mean entropy: {acc['correct']['mean_entropy']:.3f}")
        print(f"  Mean length:  {acc['correct']['mean_length']:.1f}")
        print(f"\nIncorrect answers (n={acc['incorrect']['count']}):")
        print(f"  Mean entropy: {acc['incorrect']['mean_entropy']:.3f}")
        print(f"  Mean length:  {acc['incorrect']['mean_length']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze relationship between token entropy and output length."
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to rollout data directory"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path to save results as JSON"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress information"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

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
    samples = load_rollout_samples(data_dir, args.max_samples, args.verbose)
    print(f"Loaded {len(samples)} samples")

    print("\nComputing entropy for each response...")
    early_windows = [10, 20, 50, 100]
    results = analyze_entropy_length_correlation(
        model, tokenizer, samples, args.device, early_windows, args.verbose
    )
    print(f"Successfully processed {len(results)} samples")

    print("\nComputing correlations...")
    analysis = compute_correlations(results, early_windows)

    print_results(analysis)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'analysis': analysis,
            'per_sample_results': results,
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
