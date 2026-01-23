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
Scaling Replay Simulator for SRT Speculative Decoding.

This simulator measures how speculative decoding performance scales with
the number of samples used to populate suffix trees.

Usage:
    python recipe/srt/scaling_replay_simulator.py \
        --model_path /path/to/Qwen2.5-7B \
        --scaling_data scaling_test/scaling_rollouts.jsonl \
        --simulation_data rollout_datas/rollout/10.jsonl \
        --k_values 1,2,4,8,16,32 \
        --output scaling_results.json

The simulator:
1. Loads scaling data to populate suffix trees (varying k samples per prompt)
2. Loads simulation data to measure speculation performance
3. Runs simulation for each k value and reports scaling metrics
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


@dataclass
class ScalingSimulatorConfig:
    """Configuration for scaling replay simulator."""
    # Model/Tokenizer
    model_path: str = ""

    # Data paths
    scaling_data_path: str = ""      # scaling_rollouts.jsonl
    simulation_data_path: str = ""   # rollout/10.jsonl

    # Scaling parameters
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    # Output
    output_path: str = "scaling_results.json"

    # ParallelCache parameters
    max_tree_depth: int = 64
    num_threads: int = -1  # -1 = auto-detect
    parallel_threshold: int = 4
    hash_token_count: int = 128

    # Speculation parameters
    spec_max_len: int = 16
    spec_prefix_len: int = 7  # Number of recent tokens for pattern
    min_token_prob: float = 0.1
    use_tree_spec: bool = False

    # Simulation options
    max_samples: int = 0  # 0 = all samples
    verbose: bool = False
    online_update: bool = False  # Enable on-the-fly cache updates during simulation
    min_response_tokens: int = 0  # Filter to only samples with at least this many response tokens


def load_scaling_data(path: str) -> Dict[str, List[Dict]]:
    """
    Load scaling data from jsonl file.

    The scaling data has format:
    - prompt: the prompt text
    - response: the response text
    - sample_id: unique identifier with format "scaling_prompt_X_sample_Y"

    Returns:
        Dict mapping prompt text to list of samples (sorted by sample_id).
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Scaling data not found: {data_path}")

    prompt_to_samples: Dict[str, List[Dict]] = defaultdict(list)

    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                prompt_text = item.get('prompt', '')
                if prompt_text:
                    prompt_to_samples[prompt_text].append(item)

    # Sort samples by sample_id to ensure deterministic selection
    for prompt in prompt_to_samples:
        prompt_to_samples[prompt].sort(key=lambda x: x.get('sample_id', ''))

    return dict(prompt_to_samples)


def load_simulation_data(
    path: str,
    filter_prompts: Optional[set] = None,
) -> List[Dict]:
    """
    Load simulation data from jsonl file, optionally filtering to specific prompts.

    The simulation data has format:
    - input: the prompt text
    - output: the response text

    Args:
        path: Path to simulation data file.
        filter_prompts: If provided, only include samples whose input matches.

    Returns:
        List of simulation samples.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Simulation data not found: {data_path}")

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                prompt_text = item.get('input', '')

                if filter_prompts is None or prompt_text in filter_prompts:
                    data.append(item)

    return data


def simulate_single_request(
    cache,
    config: ScalingSimulatorConfig,
    tokenizer,
    prompt_tokens: np.ndarray,
    ground_truth_tokens: np.ndarray,
    request_id: str,
) -> Dict[str, Any]:
    """
    Simulate speculative decoding for a single request.

    Args:
        cache: ParallelSuffixDecodingCache instance.
        config: Simulation configuration.
        tokenizer: Tokenizer for encoding.
        prompt_tokens: Tokenized prompt.
        ground_truth_tokens: Tokenized ground truth response.
        request_id: Unique request identifier.

    Returns:
        Dict with simulation metrics for this request.
    """
    # Start request (will lookup existing tree via hash)
    cache.start_request(request_id, prompt_tokens)

    steps = []
    response = []

    while len(response) < len(ground_truth_tokens):
        # Build context from recent tokens
        if response:
            sequence = np.concatenate([prompt_tokens, np.array(response, dtype=np.int32)])
        else:
            sequence = prompt_tokens

        pattern_size = min(len(sequence), config.spec_prefix_len)
        context = sequence[-pattern_size:]

        # Speculate
        start_time = time.perf_counter()
        try:
            draft = cache.speculate(
                request_id,
                context,
                max_spec_tokens=config.spec_max_len,
                min_token_prob=config.min_token_prob,
                use_tree_spec=config.use_tree_spec,
            )
            draft_tokens = draft.token_ids if draft.token_ids is not None else []
        except Exception as e:
            if config.verbose:
                print(f"  Speculation failed: {e}")
            draft_tokens = []
        spec_time = (time.perf_counter() - start_time) * 1000

        # Verify against ground truth
        accepted_tokens = []
        remaining_gt = ground_truth_tokens[len(response):]

        for j, draft_tok in enumerate(draft_tokens):
            if j < len(remaining_gt) and draft_tok == remaining_gt[j]:
                accepted_tokens.append(draft_tok)
            else:
                break

        new_tokens = accepted_tokens.copy()
        response.extend(accepted_tokens)

        # Add bonus token (simulating model verification step)
        if len(response) < len(ground_truth_tokens):
            bonus_token = int(ground_truth_tokens[len(response)])
            new_tokens.append(bonus_token)
            response.append(bonus_token)

        # Online update: add newly generated tokens to the cache
        if config.online_update and new_tokens:
            cache.add_tokens(request_id, np.array(new_tokens, dtype=np.int32))

        steps.append({
            "step": len(steps),
            "num_spec_toks": len(draft_tokens),
            "num_accept_toks": len(accepted_tokens),
            "num_out_toks": len(new_tokens),
            "spec_ms": spec_time,
        })

    cache.stop_request(request_id)

    total_accept = sum(s["num_accept_toks"] for s in steps)
    total_spec = sum(s["num_spec_toks"] for s in steps)
    total_out = sum(s["num_out_toks"] for s in steps)

    # Hit rate: fraction of steps where cache returned draft tokens
    steps_with_drafts = sum(1 for s in steps if s["num_spec_toks"] > 0)
    hit_rate = steps_with_drafts / len(steps) if steps else 0.0

    # Draft contribution: fraction of output tokens that came from accepted drafts
    draft_contribution = total_accept / total_out if total_out > 0 else 0.0

    # Tokens per hit step (only counting steps where speculation happened)
    hit_steps_out = sum(s["num_out_toks"] for s in steps if s["num_spec_toks"] > 0)
    tokens_per_hit_step = hit_steps_out / steps_with_drafts if steps_with_drafts > 0 else 0.0

    return {
        "request_id": request_id,
        "prompt_len": len(prompt_tokens),
        "response_len": len(ground_truth_tokens),
        "total_steps": len(steps),
        "steps_with_drafts": steps_with_drafts,
        "total_accept_toks": total_accept,
        "total_spec_toks": total_spec,
        "total_out_toks": total_out,
        "hit_rate": hit_rate,
        "acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
        "tokens_per_step": total_out / len(steps) if steps else 0.0,
        "tokens_per_hit_step": tokens_per_hit_step,
        "draft_contribution": draft_contribution,
    }


def run_scaling_simulation(
    config: ScalingSimulatorConfig,
    tokenizer,
    scaling_data: Dict[str, List[Dict]],
    sim_data: List[Dict],
    k: int,
) -> Dict[str, Any]:
    """
    Run simulation for a single k value.

    Args:
        config: Simulation configuration.
        tokenizer: Tokenizer for encoding.
        scaling_data: Dict mapping prompt to list of samples.
        sim_data: List of simulation samples.
        k: Number of samples per prompt to use for populating cache.

    Returns:
        Dict with simulation results for this k value.
    """
    from srt_plugin.suffix_cache import ParallelSuffixDecodingCache

    def tokenize(text: str) -> np.ndarray:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return np.array(tokens, dtype=np.int32)

    # Create fresh cache for this k value
    cache = ParallelSuffixDecodingCache(
        max_tree_depth=config.max_tree_depth,
        num_threads=config.num_threads,
        parallel_threshold=config.parallel_threshold,
        hash_token_count=config.hash_token_count,
    )

    # Step 1: Populate cache with first k samples per prompt
    populate_req_ids = []
    total_added = 0

    for prompt_idx, (prompt_text, samples) in enumerate(scaling_data.items()):
        # Take first k samples (already sorted by sample_id)
        selected_samples = samples[:k]

        if not selected_samples:
            continue

        prompt_tokens = tokenize(prompt_text)
        temp_req_id = f"cache_populate_{prompt_idx}"
        populate_req_ids.append(temp_req_id)

        # Start request (creates/reuses tree based on hash)
        cache.start_request(temp_req_id, prompt_tokens)

        # Add all selected responses to the tree
        for sample in selected_samples:
            response_text = sample.get('response', '')
            if not response_text:
                continue
            response_tokens = tokenize(response_text)
            if len(response_tokens) == 0:
                continue

            # Add prompt + response to tree
            full_sequence = np.concatenate([prompt_tokens, response_tokens])
            cache.add_tokens(temp_req_id, full_sequence)
            total_added += 1

    if config.verbose:
        stats = cache.get_stats()
        print(f"    Cache populated with {total_added} responses, "
              f"{stats['num_trees_in_forest']} trees")

    # Step 2: Run simulation on sim_data
    results = []

    for i, item in enumerate(sim_data):
        request_id = f"sim_{i}"
        prompt_tokens = tokenize(item['input'])
        ground_truth_tokens = tokenize(item['output'])

        if len(ground_truth_tokens) == 0:
            if config.verbose:
                print(f"    Skipping request {i}: empty response")
            continue

        result = simulate_single_request(
            cache=cache,
            config=config,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            ground_truth_tokens=ground_truth_tokens,
            request_id=request_id,
        )
        results.append(result)

        if config.verbose and (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(sim_data)}] "
                  f"hit_rate={result['hit_rate']:.3f}, "
                  f"accept_rate={result['acceptance_rate']:.3f}, "
                  f"toks/step={result['tokens_per_step']:.2f}")

    # Cleanup populate requests
    for req_id in populate_req_ids:
        try:
            cache.stop_request(req_id)
        except ValueError:
            pass

    # Compute aggregates
    if not results:
        return {
            "k": k,
            "num_requests": 0,
            "samples_per_prompt": k,
            "total_samples_in_cache": total_added,
        }

    total_steps = sum(r["total_steps"] for r in results)
    total_steps_with_drafts = sum(r["steps_with_drafts"] for r in results)
    total_accept = sum(r["total_accept_toks"] for r in results)
    total_spec = sum(r["total_spec_toks"] for r in results)
    total_out = sum(r["total_out_toks"] for r in results)

    # Weighted average for tokens per hit step
    mean_tokens_per_hit_step = (
        sum(r["tokens_per_hit_step"] * r["steps_with_drafts"] for r in results) / total_steps_with_drafts
        if total_steps_with_drafts > 0 else 0.0
    )

    return {
        "k": k,
        "num_requests": len(results),
        "samples_per_prompt": k,
        "total_samples_in_cache": total_added,
        "total_steps": total_steps,
        "total_steps_with_drafts": total_steps_with_drafts,
        "total_accept_toks": total_accept,
        "total_spec_toks": total_spec,
        "total_out_toks": total_out,
        "mean_hit_rate": total_steps_with_drafts / total_steps if total_steps > 0 else 0.0,
        "mean_acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
        "mean_tokens_per_step": total_out / total_steps if total_steps > 0 else 0.0,
        "mean_tokens_per_hit_step": mean_tokens_per_hit_step,
        "mean_draft_contribution": total_accept / total_out if total_out > 0 else 0.0,
        "per_request_results": results if config.verbose else [],
    }


def run_scaling_experiment(config: ScalingSimulatorConfig) -> Dict[str, Any]:
    """
    Main entry point: run scaling experiment across all k values.

    Args:
        config: Simulation configuration.

    Returns:
        Dict with full experiment results.
    """
    from transformers import AutoTokenizer

    # Load tokenizer
    print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )

    # Load scaling data
    print(f"\nLoading scaling data from {config.scaling_data_path}...")
    scaling_data = load_scaling_data(config.scaling_data_path)
    print(f"  Loaded {sum(len(v) for v in scaling_data.values())} samples from {len(scaling_data)} unique prompts")

    # Get prompt set for filtering
    scaling_prompts = set(scaling_data.keys())

    # Load simulation data, filtering to overlapping prompts
    print(f"\nLoading simulation data from {config.simulation_data_path}...")
    sim_data = load_simulation_data(config.simulation_data_path, filter_prompts=scaling_prompts)
    print(f"  Loaded {len(sim_data)} samples (filtered to prompts in scaling data)")

    # Filter by minimum response length if specified
    if config.min_response_tokens > 0:
        original_count = len(sim_data)
        filtered_data = []
        for item in sim_data:
            output_text = item.get('output', '')
            output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
            if len(output_tokens) >= config.min_response_tokens:
                filtered_data.append(item)
        sim_data = filtered_data
        print(f"  Filtered to {len(sim_data)} samples with >= {config.min_response_tokens} response tokens (removed {original_count - len(sim_data)})")

    if config.max_samples > 0:
        sim_data = sim_data[:config.max_samples]
        print(f"  Limited to {len(sim_data)} samples")

    # Run simulation for each k value
    online_str = " (online_update=ON)" if config.online_update else ""
    print(f"\nRunning scaling experiment with k values: {config.k_values}{online_str}")
    scaling_results = []

    for k in config.k_values:
        print(f"\n[k={k}] Running simulation...")
        start_time = time.time()

        result = run_scaling_simulation(
            config=config,
            tokenizer=tokenizer,
            scaling_data=scaling_data,
            sim_data=sim_data,
            k=k,
        )

        elapsed = time.time() - start_time
        scaling_results.append(result)

        print(f"[k={k}] Completed in {elapsed:.1f}s")
        print(f"  - Requests: {result['num_requests']}")
        print(f"  - Samples in cache: {result['total_samples_in_cache']}")
        print(f"  - Hit rate: {result.get('mean_hit_rate', 0):.4f} ({result.get('mean_hit_rate', 0)*100:.2f}%)")
        print(f"  - Acceptance rate: {result.get('mean_acceptance_rate', 0):.4f} ({result.get('mean_acceptance_rate', 0)*100:.2f}%)")
        print(f"  - Tokens/step: {result.get('mean_tokens_per_step', 0):.3f}")
        print(f"  - Draft contribution: {result.get('mean_draft_contribution', 0):.4f} ({result.get('mean_draft_contribution', 0)*100:.2f}%)")

    # Create summary table
    summary_table = []
    for r in scaling_results:
        summary_table.append({
            "k": r["k"],
            "samples_in_cache": r["total_samples_in_cache"],
            "hit_rate": round(r.get("mean_hit_rate", 0), 4),
            "acceptance_rate": round(r.get("mean_acceptance_rate", 0), 4),
            "tokens_per_step": round(r.get("mean_tokens_per_step", 0), 3),
            "draft_contribution": round(r.get("mean_draft_contribution", 0), 4),
        })

    # Build final output
    output = {
        "config": {
            "model_path": config.model_path,
            "scaling_data_path": config.scaling_data_path,
            "simulation_data_path": config.simulation_data_path,
            "k_values": config.k_values,
            "max_tree_depth": config.max_tree_depth,
            "spec_max_len": config.spec_max_len,
            "spec_prefix_len": config.spec_prefix_len,
            "min_token_prob": config.min_token_prob,
            "hash_token_count": config.hash_token_count,
            "online_update": config.online_update,
        },
        "data_info": {
            "num_scaling_prompts": len(scaling_data),
            "total_scaling_samples": sum(len(v) for v in scaling_data.values()),
            "num_simulation_samples": len(sim_data),
        },
        "scaling_results": scaling_results,
        "summary_table": summary_table,
    }

    # Save output
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return output


def print_summary_table(results: Dict[str, Any]):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("Scaling Experiment Summary")
    print("=" * 80)

    summary = results.get("summary_table", [])
    if not summary:
        print("No results to display")
        return

    # Print header
    print(f"{'k':>6} | {'Samples':>8} | {'Hit Rate':>10} | {'Accept Rate':>12} | {'Toks/Step':>10} | {'Draft Contrib':>13}")
    print("-" * 80)

    # Print rows
    for row in summary:
        print(f"{row['k']:>6} | {row['samples_in_cache']:>8} | {row['hit_rate']:>10.4f} | {row['acceptance_rate']:>12.4f} | {row['tokens_per_step']:>10.3f} | {row['draft_contribution']:>13.4f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="SRT Scaling Replay Simulator")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model for tokenizer")
    parser.add_argument("--scaling_data", type=str, required=True,
                        help="Path to scaling data JSONL file")
    parser.add_argument("--simulation_data", type=str, required=True,
                        help="Path to simulation data JSONL file")

    # Scaling parameters
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated k values (samples per prompt)")
    parser.add_argument("--output", type=str, default="scaling_results.json",
                        help="Output JSON file path")

    # Cache parameters
    parser.add_argument("--max_tree_depth", type=int, default=64,
                        help="Maximum tree depth (default: 64)")
    parser.add_argument("--num_threads", type=int, default=-1,
                        help="Number of threads (-1=auto, default: -1)")
    parser.add_argument("--parallel_threshold", type=int, default=4,
                        help="Min batch size to trigger parallelization (default: 4)")
    parser.add_argument("--hash_token_count", type=int, default=128,
                        help="Trailing tokens to hash for tree sharing (default: 128)")

    # Speculation parameters
    parser.add_argument("--spec_max_len", type=int, default=16,
                        help="Maximum speculation length (default: 16)")
    parser.add_argument("--spec_prefix_len", type=int, default=7,
                        help="Pattern length for speculation (default: 7)")
    parser.add_argument("--min_token_prob", type=float, default=0.1,
                        help="Minimum token probability (default: 0.1)")

    # Simulation options
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum samples to simulate (0=all)")
    parser.add_argument("--min_response_tokens", type=int, default=0,
                        help="Filter to samples with at least this many response tokens (0=no filter)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--online_update", action="store_true",
                        help="Enable on-the-fly cache updates during simulation")

    args = parser.parse_args()

    # Parse k_values
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    config = ScalingSimulatorConfig(
        model_path=args.model_path,
        scaling_data_path=args.scaling_data,
        simulation_data_path=args.simulation_data,
        k_values=k_values,
        output_path=args.output,
        max_tree_depth=args.max_tree_depth,
        num_threads=args.num_threads,
        parallel_threshold=args.parallel_threshold,
        hash_token_count=args.hash_token_count,
        spec_max_len=args.spec_max_len,
        spec_prefix_len=args.spec_prefix_len,
        min_token_prob=args.min_token_prob,
        max_samples=args.max_samples,
        min_response_tokens=args.min_response_tokens,
        verbose=args.verbose,
        online_update=args.online_update,
    )

    results = run_scaling_experiment(config)
    print_summary_table(results)


if __name__ == "__main__":
    main()
