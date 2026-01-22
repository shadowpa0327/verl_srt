#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Suffix Tree Scaling Experiment Script.

This script:
1. Loads FSDP checkpoint from a specific training step and merges weights
2. Parses rollout data and identifies prompts with long average output lengths
3. Performs k additional rollouts on selected prompts with configurable temperature
4. Outputs results in secondary rollout format for suffix tree scaling analysis

Usage:
    python -m recipe.srt.scripts.suffix_tree_scaling_experiment \\
        --checkpoint-path /path/to/global_step_N/actor \\
        --rollout-data /path/to/rollout/N.jsonl \\
        --output-dir /path/to/output \\
        --k-rollouts 16 \\
        --temperature 1.0 \\
        --top-percentile 0.2 \\
        --max-prompts 50 \\
        --num-gpus 4

Alternative with pre-merged model:
    python -m recipe.srt.scripts.suffix_tree_scaling_experiment \\
        --model-path /path/to/merged_hf_model \\
        --rollout-data /path/to/rollout/N.jsonl \\
        --output-dir /path/to/output \\
        --k-rollouts 16
"""

# IMPORTANT: Set multiprocessing start method before any CUDA imports
# This prevents "Cannot re-initialize CUDA in forked subprocess" errors
import multiprocessing
import os

# Set environment variable for vLLM multiprocessing (must be before vLLM import)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ScalingExperimentConfig:
    """Configuration for suffix tree scaling experiment."""

    # Checkpoint (FSDP) - mutually exclusive with model_path
    checkpoint_path: Optional[str] = None

    # Pre-merged model path (alternative to checkpoint_path)
    model_path: Optional[str] = None

    # Directory for merged model (if empty, create in output_dir)
    merged_model_dir: str = ""

    # Input
    rollout_data_path: str = ""

    # Output
    output_dir: str = ""

    # Selection criteria
    top_percentile: float = 0.1  # Select top 10% longest prompts
    min_samples_per_prompt: int = 4  # Require at least 4 samples
    max_prompts: int = 100  # Cap at N prompts

    # Generation
    k_rollouts: int = 8  # Samples per prompt
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 12 * 1024

    # VeRL/vLLM settings (multi-GPU support)
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    seed: int = 42


def merge_fsdp_checkpoint(checkpoint_path: str, target_dir: str) -> str:
    """Merge FSDP shards to HuggingFace format.

    Args:
        checkpoint_path: Path to FSDP checkpoint (e.g., global_step_N/actor)
        target_dir: Directory to save merged HF model

    Returns:
        Path to merged model directory
    """
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    # Ensure the checkpoint path has the huggingface config
    hf_config_path = os.path.join(checkpoint_path, "huggingface")
    if not os.path.exists(hf_config_path):
        raise FileNotFoundError(
            f"HuggingFace config not found at {hf_config_path}. "
            "Ensure checkpoint_path points to the actor directory containing 'huggingface' folder."
        )

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=checkpoint_path,
        target_dir=target_dir,
        hf_model_config_path=hf_config_path,
        trust_remote_code=True,
    )

    print(f"Merging FSDP checkpoint from {checkpoint_path} to {target_dir}...")
    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()
    print(f"Merge complete. Model saved to {target_dir}")

    return target_dir


def load_and_group_rollout_data(
    rollout_data_path: str,
    tokenizer,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Load rollout JSONL and group by unique prompt text.

    Args:
        rollout_data_path: Path to rollout JSONL file
        tokenizer: HuggingFace tokenizer for computing token lengths
        verbose: Print progress

    Returns:
        Dictionary mapping prompt_text to stats:
        {
            'prompt_text': original prompt,
            'count': number of samples,
            'avg_length': average response token length,
            'max_length': maximum response token length,
            'lengths': list of all response token lengths,
            'responses': list of all responses
        }
    """
    prompt_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'prompt_text': None,
        'lengths': [],
        'responses': [],
    })

    count = 0
    with open(rollout_data_path) as f:
        for line in f:
            item = json.loads(line)
            prompt_text = item['input']
            output_text = item['output']

            if prompt_data[prompt_text]['prompt_text'] is None:
                prompt_data[prompt_text]['prompt_text'] = prompt_text

            # Tokenize output to get token length
            output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
            prompt_data[prompt_text]['lengths'].append(len(output_tokens))
            prompt_data[prompt_text]['responses'].append(output_text)
            count += 1

            if verbose and count % 1000 == 0:
                print(f"  Loaded {count} samples...")

    # Compute statistics
    result = {}
    for prompt_text, data in prompt_data.items():
        lengths = data['lengths']
        result[prompt_text] = {
            'prompt_text': prompt_text,
            'count': len(lengths),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'lengths': lengths,
            'responses': data['responses'],
        }

    if verbose:
        print(f"  Total: {count} samples, {len(result)} unique prompts")

    return result


def select_long_output_prompts(
    prompt_data: Dict[str, Dict[str, Any]],
    config: ScalingExperimentConfig,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Select prompts with longest average output lengths.

    Args:
        prompt_data: Dictionary from load_and_group_rollout_data
        config: Experiment configuration
        verbose: Print progress

    Returns:
        List of selected prompt stats (sorted by avg_length descending)
    """
    # Filter by minimum samples
    filtered = [
        stats for stats in prompt_data.values()
        if stats['count'] >= config.min_samples_per_prompt
    ]

    if verbose:
        print(f"  After filtering (min_samples={config.min_samples_per_prompt}): {len(filtered)} prompts")

    # Sort by average length (descending)
    sorted_prompts = sorted(filtered, key=lambda x: x['max_length'], reverse=True)

    # Take top percentile
    top_n = max(1, int(len(sorted_prompts) * config.top_percentile))
    selected = sorted_prompts[:top_n]

    if verbose:
        print(f"  After top {config.top_percentile*100:.0f}%: {len(selected)} prompts")

    # Cap at max_prompts
    if len(selected) > config.max_prompts:
        selected = selected[:config.max_prompts]
        if verbose:
            print(f"  After capping at {config.max_prompts}: {len(selected)} prompts")

    return selected


def compute_prompt_hashes(
    prompts: List[str],
    tokenizer,
) -> List[int]:
    """Compute XXH64 hashes for prompts (matching suffix tree format).

    Args:
        prompts: List of prompt texts
        tokenizer: HuggingFace tokenizer

    Returns:
        List of XXH64 hash integers
    """
    try:
        from recipe.srt.srt_plugin.suffix_cache.hash_utils import compute_prompt_hash_xxh64
    except ImportError:
        # Fallback implementation
        import struct
        import xxhash

        def compute_prompt_hash_xxh64(tokens: List[int]) -> int:
            token_bytes = struct.pack(f'{len(tokens)}i', *tokens)
            return xxhash.xxh64(token_bytes).intdigest()

    hashes = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        hashes.append(compute_prompt_hash_xxh64(token_ids))

    return hashes


def parse_raw_prompt_to_messages(raw_prompt: str) -> List[Dict[str, str]]:
    """Parse raw 'user\\n...\\nassistant\\n' format into chat messages.

    Args:
        raw_prompt: Raw prompt string in format "user\\n<content>\\nassistant\\n"

    Returns:
        List of message dicts: [{"role": "user", "content": "..."}]
    """
    # Handle the format: "user\n<content>\nassistant\n"
    # Strip the "user\n" prefix and "assistant\n" suffix to get user content
    content = raw_prompt

    if content.startswith("user\n"):
        content = content[5:]  # Remove "user\n"

    if content.endswith("assistant\n"):
        content = content[:-10]  # Remove "assistant\n"
    elif content.endswith("assistant"):
        content = content[:-9]  # Remove "assistant"

    content = content.strip()

    return [{"role": "user", "content": content}]


def apply_chat_template(
    prompts: List[str],
    tokenizer,
    verbose: bool = True
) -> List[str]:
    """Apply tokenizer's chat template to prompts.

    Args:
        prompts: List of raw prompt texts
        tokenizer: HuggingFace tokenizer with chat template
        verbose: Print progress

    Returns:
        List of templated prompt strings
    """
    templated_prompts = []

    for i, raw_prompt in enumerate(prompts):
        messages = parse_raw_prompt_to_messages(raw_prompt)

        # Apply chat template - add_generation_prompt=True adds the assistant prefix
        templated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        templated_prompts.append(templated)

        if verbose and i == 0:
            print(f"  Example chat template transformation:")
            print(f"    Raw: {raw_prompt[:100]}...")
            print(f"    Templated: {templated[:150]}...")

    return templated_prompts


def run_generation(
    model_path: str,
    prompts: List[str],
    tokenizer,
    config: ScalingExperimentConfig,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run vLLM generation on prompts.

    Args:
        model_path: Path to merged HF model
        prompts: List of raw prompt texts (will be templated)
        tokenizer: HuggingFace tokenizer for chat template
        config: Experiment configuration
        verbose: Print progress

    Returns:
        List of generation results with prompt, response, etc.
    """
    from vllm import LLM, SamplingParams

    # Apply chat template to prompts
    if verbose:
        print("Applying chat template to prompts...")
    templated_prompts = apply_chat_template(prompts, tokenizer, verbose)

    if verbose:
        print(f"Initializing vLLM with model {model_path}...")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        seed=config.seed,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        n=config.k_rollouts,  # Generate k samples per prompt
    )

    if verbose:
        print(f"Generating {config.k_rollouts} samples for each of {len(prompts)} prompts...")

    # Generate using templated prompts
    outputs = llm.generate(templated_prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        # Store original raw prompt for compatibility, not the templated one
        raw_prompt = prompts[i]
        for j, completion in enumerate(output.outputs):
            results.append({
                'prompt_idx': i,
                'sample_idx': j,
                'prompt': raw_prompt,  # Keep original format for output
                'response': completion.text,
                'stop_reason': completion.stop_reason or "unknown",
                'prompt_length': len(output.prompt_token_ids),
                'response_length': len(completion.token_ids),
            })

    if verbose:
        print(f"Generated {len(results)} total samples")

    return results


def write_secondary_format(
    results: List[Dict[str, Any]],
    prompt_stats: Dict[str, Dict[str, Any]],
    prompt_hashes: List[int],
    config: ScalingExperimentConfig,
    output_path: str,
    verbose: bool = True
):
    """Write results in secondary rollout format.

    Args:
        results: Generation results from run_generation
        prompt_stats: Original prompt statistics
        prompt_hashes: XXH64 hashes for prompts
        config: Experiment configuration
        output_path: Path to output JSONL file
        verbose: Print progress
    """
    # Build prompt to hash mapping
    unique_prompts = list(set(r['prompt'] for r in results))
    prompt_to_hash = {p: prompt_hashes[i] for i, p in enumerate(unique_prompts)}
    prompt_to_stats = {s['prompt_text']: s for s in prompt_stats.values()}

    with open(output_path, 'w') as f:
        for i, result in enumerate(results):
            prompt_text = result['prompt']
            source_stats = prompt_to_stats.get(prompt_text, {})

            record = {
                'sample_id': f"scaling_prompt_{result['prompt_idx']}_sample_{result['sample_idx']}",
                'status': 'completed',
                'prompt': prompt_text,
                'response': result['response'],
                'prompt_hash': prompt_to_hash.get(prompt_text, 0),
                'prompt_length': result['prompt_length'],
                'response_length': result['response_length'],
                'stop_reason': result['stop_reason'],
                'is_partial': False,
                'temperature': config.temperature,
                'source_avg_length': source_stats.get('avg_length', 0),
            }
            f.write(json.dumps(record) + '\n')

    if verbose:
        print(f"Wrote {len(results)} samples to {output_path}")


def save_experiment_metadata(
    config: ScalingExperimentConfig,
    prompt_stats: List[Dict[str, Any]],
    all_prompt_stats: Dict[str, Dict[str, Any]],
    output_dir: str
):
    """Save experiment configuration and prompt statistics.

    Args:
        config: Experiment configuration
        prompt_stats: Selected prompt statistics
        all_prompt_stats: All prompt statistics (before selection)
        output_dir: Output directory
    """
    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Save all prompt stats (without responses to save space)
    all_stats_clean = {}
    for prompt_text, stats in all_prompt_stats.items():
        all_stats_clean[prompt_text[:100] + "..."] = {
            'count': stats['count'],
            'avg_length': stats['avg_length'],
            'max_length': stats['max_length'],
            'min_length': stats['min_length'],
        }
    with open(os.path.join(output_dir, 'prompt_stats.json'), 'w') as f:
        json.dump(all_stats_clean, f, indent=2)

    # Save selected prompts with stats
    selected_clean = []
    for stats in prompt_stats:
        selected_clean.append({
            'prompt_text': stats['prompt_text'],
            'count': stats['count'],
            'avg_length': stats['avg_length'],
            'max_length': stats['max_length'],
            'min_length': stats['min_length'],
        })
    with open(os.path.join(output_dir, 'selected_prompts.json'), 'w') as f:
        json.dump(selected_clean, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Suffix Tree Scaling Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to FSDP checkpoint (e.g., global_step_N/actor)'
    )
    model_group.add_argument(
        '--model-path',
        type=str,
        help='Path to pre-merged HuggingFace model'
    )

    # Input/Output
    parser.add_argument(
        '--rollout-data',
        type=str,
        required=True,
        help='Path to rollout JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory for output files'
    )

    # Selection criteria
    parser.add_argument(
        '--top-percentile',
        type=float,
        default=0.1,
        help='Select top N%% longest prompts'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=4,
        help='Minimum samples per prompt to be considered'
    )
    parser.add_argument(
        '--max-prompts',
        type=int,
        default=100,
        help='Maximum number of prompts to select'
    )

    # Generation
    parser.add_argument(
        '--k-rollouts',
        type=int,
        default=8,
        help='Number of samples to generate per prompt'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=12 * 1024,
        help='Maximum tokens to generate'
    )

    # Hardware
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Total number of GPUs to use'
    )
    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=1,
        help='Tensor parallel size'
    )
    parser.add_argument(
        '--gpu-memory-utilization',
        type=float,
        default=0.85,
        help='GPU memory utilization fraction'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    # Misc
    parser.add_argument(
        '--skip-merge',
        action='store_true',
        help='Skip merging if merged model already exists'
    )

    args = parser.parse_args()

    # Build config
    config = ScalingExperimentConfig(
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        rollout_data_path=args.rollout_data,
        output_dir=args.output_dir,
        top_percentile=args.top_percentile,
        min_samples_per_prompt=args.min_samples,
        max_prompts=args.max_prompts,
        k_rollouts=args.k_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        num_gpus=args.num_gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 1: Determine model path (merge if needed)
    if config.model_path:
        model_path = config.model_path
        print(f"Using pre-merged model: {model_path}")
    else:
        merged_dir = os.path.join(config.output_dir, 'merged_model')

        if args.skip_merge and os.path.exists(merged_dir):
            print(f"Skipping merge, using existing: {merged_dir}")
            model_path = merged_dir
        else:
            model_path = merge_fsdp_checkpoint(config.checkpoint_path, merged_dir)

    # Step 2: Load tokenizer (needed for token length computation and hash)
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Step 3: Load rollout data with tokenization
    print(f"\nLoading rollout data from {config.rollout_data_path}...")
    print("  (tokenizing outputs for accurate token length computation)")
    all_prompt_stats = load_and_group_rollout_data(config.rollout_data_path, tokenizer)

    # Step 4: Select prompts
    print("\nSelecting prompts with long outputs (by token count)...")
    selected_prompts = select_long_output_prompts(all_prompt_stats, config)

    if not selected_prompts:
        print("No prompts selected! Check your selection criteria.")
        sys.exit(1)

    print(f"\nSelected {len(selected_prompts)} prompts:")
    for i, stats in enumerate(selected_prompts[:5]):
        print(f"  {i+1}. avg_tokens={stats['avg_length']:.0f}, count={stats['count']}, "
              f"prompt={stats['prompt_text'][:50]}...")
    if len(selected_prompts) > 5:
        print(f"  ... and {len(selected_prompts) - 5} more")

    # Step 5: Compute prompt hashes
    print("\nComputing prompt hashes...")
    prompt_texts = [s['prompt_text'] for s in selected_prompts]
    prompt_hashes = compute_prompt_hashes(prompt_texts, tokenizer)

    # Step 6: Run generation (with chat template)
    print(f"\nRunning generation with k={config.k_rollouts}, temp={config.temperature}...")
    results = run_generation(model_path, prompt_texts, tokenizer, config)

    # Step 7: Write output
    output_path = os.path.join(config.output_dir, 'scaling_rollouts.jsonl')
    print(f"\nWriting results to {output_path}...")
    write_secondary_format(
        results,
        {s['prompt_text']: s for s in selected_prompts},
        prompt_hashes,
        config,
        output_path
    )

    # Step 8: Save metadata
    print("\nSaving experiment metadata...")
    save_experiment_metadata(config, selected_prompts, all_prompt_stats, config.output_dir)

    print("\n" + "="*60)
    print("Experiment complete!")
    print(f"Output directory: {config.output_dir}")
    print(f"  - scaling_rollouts.jsonl: {len(results)} samples")
    print(f"  - config.json: experiment configuration")
    print(f"  - prompt_stats.json: all prompt statistics")
    print(f"  - selected_prompts.json: selected prompts with stats")
    if not config.model_path:
        print(f"  - merged_model/: merged HuggingFace model")
    print("="*60)


if __name__ == '__main__':
    main()