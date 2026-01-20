#!/usr/bin/env python3
"""
Top-K Longest Response Analysis for SRT Speculative Decoding.

This script analyzes the trade-off between cache population (number of responses
per prompt, prioritizing longest) and acceptance rate.

Usage:
    python recipe/srt/topk_analysis.py \
        --model_path /path/to/model \
        --data_dir /path/to/rollout_datas/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \
        --cache_tick 1 \
        --sim_tick 2
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np


@dataclass
class AnalysisConfig:
    """Configuration for top-k analysis."""
    model_path: str = ""
    data_dir: str = ""
    cache_tick: int = 1
    cache_source: str = "secondary"
    sim_tick: int = 2

    # Server settings
    port: int = 16399
    shm_name: str = "TOPK_ANALYSIS_CACHE"

    # Speculation parameters
    spec_start_len: int = 2
    spec_max_len: int = 16
    spec_prefix_len: int = 7
    min_token_prob: float = 0.1
    use_tree_spec: bool = False

    # Analysis parameters
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    max_samples: int = 0
    verbose: bool = False


def _server_subprocess_main(port: int, shm_name: str, ready_event):
    """Cache server subprocess entry point."""
    try:
        from srt_plugin.shm_cache.suffix_cache import RolloutCacheServer
        server = RolloutCacheServer(f"[::]:{port}", 0, shm_name)
        if not server.initialize():
            print(f"[Server] Failed to initialize cache server")
            sys.exit(1)
        if not server.start():
            print(f"[Server] Failed to start cache server")
            sys.exit(1)
        ready_event.set()
        server.wait()
    except Exception as e:
        print(f"[Server] Cache server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _populate_cache_worker_topk(config_dict: dict, top_k: int, result_queue):
    """
    Worker subprocess that populates the cache with top-k longest responses per prompt.
    """
    try:
        config = AnalysisConfig(**config_dict)

        from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )

        def tokenize(text: str) -> List[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        def load_data(tick: int, source: str) -> List[Dict]:
            data_path = Path(config.data_dir) / source / f"{tick}.jsonl"
            if not data_path.exists():
                raise FileNotFoundError(f"Data not found: {data_path}")
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data

        server_address = f"127.0.0.1:{config.port}"
        updater = SuffixCacheUpdater(server_addresses=[server_address])

        cache_data = load_data(config.cache_tick, config.cache_source)

        # Group by prompt and tokenize
        is_secondary = config.cache_source == "secondary"
        prompt_to_responses = defaultdict(list)

        for item in cache_data:
            if is_secondary:
                prompt_text = item['prompt']
                response_text = item['response']
                response_len = item.get('response_length', len(response_text))
            else:
                prompt_text = item['input']
                response_text = item['output']
                response_len = len(response_text)

            prompt_tokens = tuple(tokenize(prompt_text))
            response_tokens = tokenize(response_text)

            if len(response_tokens) == 0:
                continue

            # Store with length for sorting
            prompt_to_responses[prompt_tokens].append({
                'tokens': response_tokens,
                'length': len(response_tokens),
            })

        # Sort by length (descending) and take top-k
        total_responses = 0
        for prompt_tokens in prompt_to_responses:
            responses = prompt_to_responses[prompt_tokens]
            responses.sort(key=lambda x: x['length'], reverse=True)
            prompt_to_responses[prompt_tokens] = responses[:top_k]
            total_responses += len(prompt_to_responses[prompt_tokens])

        print(f"[Updater k={top_k}] Using {total_responses} responses from {len(prompt_to_responses)} prompts")

        # Send to cache
        for prompt_tokens, resp_list in prompt_to_responses.items():
            batch_prompts = [list(prompt_tokens)] * len(resp_list)
            batch_responses = [r['tokens'] for r in resp_list]
            batch_prompt_lens = [float(len(prompt_tokens))] * len(resp_list)
            batch_response_lens = [float(r['length']) for r in resp_list]

            updater.update_response_cache(
                prompts=batch_prompts,
                responses=batch_responses,
                prompt_lengths=batch_prompt_lens,
                response_lengths=batch_response_lens,
                responses_per_prompt=len(resp_list),
            )

        result_queue.put(("success", total_responses))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def _simulate_worker(config_dict: dict, top_k: int, result_queue):
    """
    Worker subprocess that runs the simulation.
    """
    try:
        config = AnalysisConfig(**config_dict)

        from srt_plugin.shm_cache.suffix_cache import SuffixCache
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )

        def tokenize(text: str) -> np.ndarray:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return np.array(tokens, dtype=np.int32)

        def load_rollout_data(tick: int, source: str = "rollout") -> List[Dict]:
            data_path = Path(config.data_dir) / source / f"{tick}.jsonl"
            if not data_path.exists():
                raise FileNotFoundError(f"Rollout data not found: {data_path}")
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data

        cache = SuffixCache(
            shared_memory_name=config.shm_name,
            spec_start_len=config.spec_start_len,
            spec_max_len=config.spec_max_len,
        )

        sim_data = load_rollout_data(config.sim_tick)
        if config.max_samples > 0:
            sim_data = sim_data[:config.max_samples]

        # Verification
        cache_source_data = load_rollout_data(config.cache_tick, config.cache_source)
        is_secondary = config.cache_source == "secondary"
        verify_prompts = []
        seen_prompts = set()
        for item in cache_source_data:
            prompt_text = item['prompt'] if is_secondary else item['input']
            if prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                verify_prompts.append(tokenize(prompt_text))
                if len(verify_prompts) >= 3:
                    break

        max_retries = 20
        retry_delay = 0.5
        verified = False

        for retry in range(max_retries):
            trees_found = 0
            for i, prompt in enumerate(verify_prompts):
                req_id = f"verify_{i}"
                cache.fetch_responses_by_prompts_batch([req_id], [prompt.tolist()])
                pattern = prompt[-config.spec_prefix_len:].tolist() if len(prompt) >= config.spec_prefix_len else prompt.tolist()
                drafts = cache.speculate([req_id], [pattern], min_token_prob=config.min_token_prob)
                cache.evict_responses(req_id)
                if drafts and drafts[0]:
                    trees_found += 1

            if trees_found >= len(verify_prompts):
                verified = True
                break
            time.sleep(retry_delay)

        if not verified:
            print(f"[Simulator k={top_k}] WARNING: Cache verification failed")

        # Run simulation
        results = []
        total_accept_by_position = defaultdict(int)
        total_spec_by_position = defaultdict(int)

        for i, item in enumerate(sim_data):
            request_id = f"sim_{i}"
            prompt = tokenize(item['input'])
            ground_truth = tokenize(item['output'])

            if len(ground_truth) == 0:
                continue

            cache.fetch_responses_by_prompts_batch([request_id], [prompt.tolist()])

            steps = []
            response = []

            while len(response) < len(ground_truth):
                if response:
                    sequence = np.concatenate([prompt, np.array(response, dtype=np.int32)])
                else:
                    sequence = prompt

                pattern_size = min(len(sequence), config.spec_prefix_len)
                pattern = sequence[-pattern_size:].tolist()

                try:
                    drafts = cache.speculate(
                        [request_id],
                        [pattern],
                        min_token_prob=config.min_token_prob,
                        use_tree_spec=config.use_tree_spec,
                    )
                    draft_tokens = drafts[0] if drafts and drafts[0] else []
                except Exception:
                    draft_tokens = []

                # Verify against ground truth
                accepted_tokens = []
                remaining_gt = ground_truth[len(response):]

                for j, draft_tok in enumerate(draft_tokens):
                    if j < len(remaining_gt) and draft_tok == remaining_gt[j]:
                        accepted_tokens.append(draft_tok)
                        total_accept_by_position[j] += 1
                    else:
                        break
                    total_spec_by_position[j] += 1

                # Count speculated but not accepted positions
                for j in range(len(accepted_tokens), len(draft_tokens)):
                    if j < len(remaining_gt):
                        total_spec_by_position[j] += 1

                new_tokens = accepted_tokens.copy()
                response.extend(accepted_tokens)

                if len(response) < len(ground_truth):
                    bonus_token = int(ground_truth[len(response)])
                    new_tokens.append(bonus_token)
                    response.append(bonus_token)

                # Update cache's internal spec_len (this is the key call!)
                # len(new_tokens) passed to update_spec_len determines if we grow or shrink
                cache.update_spec_len(request_id, len(new_tokens))

                steps.append({
                    "num_spec_toks": len(draft_tokens),
                    "num_accept_toks": len(accepted_tokens),
                    "num_out_toks": len(new_tokens),
                })

            cache.evict_responses(request_id)

            total_accept = sum(s["num_accept_toks"] for s in steps)
            total_spec = sum(s["num_spec_toks"] for s in steps)
            total_out = sum(s["num_out_toks"] for s in steps)

            result = {
                "request_id": request_id,
                "prompt_len": len(prompt),
                "response_len": len(ground_truth),
                "total_steps": len(steps),
                "total_accept_toks": total_accept,
                "total_spec_toks": total_spec,
                "acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
                "tokens_per_step": total_out / len(steps) if steps else 0.0,
            }
            results.append(result)

        # Compute per-position acceptance rates
        position_acceptance = {}
        for pos in sorted(total_spec_by_position.keys()):
            if total_spec_by_position[pos] > 0:
                position_acceptance[pos] = total_accept_by_position[pos] / total_spec_by_position[pos]

        total_steps = sum(r["total_steps"] for r in results)
        total_accept = sum(r["total_accept_toks"] for r in results)
        total_spec = sum(r["total_spec_toks"] for r in results)
        total_out = sum(r["total_steps"] * r["tokens_per_step"] for r in results)

        summary = {
            "top_k": top_k,
            "num_requests": len(results),
            "total_steps": total_steps,
            "total_accept_toks": total_accept,
            "total_spec_toks": total_spec,
            "mean_acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
            "mean_tokens_per_step": total_out / total_steps if total_steps > 0 else 0.0,
            "position_acceptance": position_acceptance,
            "requests": results,
        }

        result_queue.put(("success", summary))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def run_single_k(config: AnalysisConfig, top_k: int) -> Dict[str, Any]:
    """Run simulation for a single k value."""
    # Use unique shm_name per k to avoid conflicts
    config_dict = config.__dict__.copy()
    config_dict['shm_name'] = f"{config.shm_name}_{top_k}"
    config_dict['k_values'] = list(config.k_values)  # Convert to list for serialization

    # Start cache server
    ready_event = multiprocessing.Event()
    server_proc = multiprocessing.Process(
        target=_server_subprocess_main,
        args=(config.port, config_dict['shm_name'], ready_event),
        daemon=True
    )
    server_proc.start()

    if not ready_event.wait(timeout=10):
        if not server_proc.is_alive():
            raise RuntimeError("Cache server failed to start")

    time.sleep(1)

    try:
        # Populate cache
        updater_queue = multiprocessing.Queue()
        updater_proc = multiprocessing.Process(
            target=_populate_cache_worker_topk,
            args=(config_dict, top_k, updater_queue),
            daemon=True
        )
        updater_proc.start()
        updater_proc.join(timeout=300)

        if updater_proc.is_alive():
            updater_proc.terminate()
            raise RuntimeError("Cache population timed out")

        if updater_queue.empty():
            raise RuntimeError("Cache population produced no results")

        status, result = updater_queue.get()
        if status == "error":
            raise RuntimeError(f"Cache population failed: {result}")

        time.sleep(0.5)

        # Run simulation
        sim_queue = multiprocessing.Queue()
        sim_proc = multiprocessing.Process(
            target=_simulate_worker,
            args=(config_dict, top_k, sim_queue),
            daemon=True
        )
        sim_proc.start()
        sim_proc.join(timeout=600)

        if sim_proc.is_alive():
            sim_proc.terminate()
            raise RuntimeError("Simulation timed out")

        if sim_queue.empty():
            raise RuntimeError("Simulation produced no results")

        status, result = sim_queue.get()
        if status == "error":
            raise RuntimeError(f"Simulation failed: {result}")

        return result

    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.kill()


def run_analysis(config: AnalysisConfig) -> List[Dict[str, Any]]:
    """Run analysis for all k values."""
    results = []

    for k in config.k_values:
        print(f"\n{'='*60}")
        print(f"Running analysis for top_k = {k}")
        print('='*60)

        result = run_single_k(config, k)
        results.append(result)

        print(f"  Acceptance rate: {result['mean_acceptance_rate']:.4f}")
        print(f"  Tokens per step: {result['mean_tokens_per_step']:.3f}")

    return results


def print_results(results: List[Dict[str, Any]]):
    """Print analysis results as a comparison table."""
    print("\n" + "="*80)
    print("Top-K Longest Response Analysis Results")
    print("="*80)

    print(f"\n{'k':>4} | {'Responses':>10} | {'Accept Rate':>12} | {'Toks/Step':>10} | {'Total Steps':>12}")
    print("-"*60)

    for r in results:
        # Calculate total responses from requests
        print(f"{r['top_k']:>4} | {r['total_spec_toks']//r['total_steps'] if r['total_steps'] > 0 else 0:>10} | "
              f"{r['mean_acceptance_rate']:>11.4f} | {r['mean_tokens_per_step']:>10.3f} | "
              f"{r['total_steps']:>12}")

    # Position-wise acceptance rates
    print("\n" + "-"*60)
    print("Position-wise Acceptance Rates:")
    print(f"{'Position':>8}", end="")
    for r in results:
        print(f" | k={r['top_k']:>2}", end="")
    print()
    print("-"*60)

    max_pos = max(max(r['position_acceptance'].keys()) for r in results if r['position_acceptance'])
    for pos in range(min(max_pos + 1, 16)):  # Show up to 16 positions
        print(f"{pos:>8}", end="")
        for r in results:
            rate = r['position_acceptance'].get(pos, 0.0)
            print(f" | {rate:>.3f}", end="")
        print()

    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    baseline = results[-1]  # k=16 as baseline
    for r in results:
        speedup = r['mean_tokens_per_step'] / baseline['mean_tokens_per_step'] if baseline['mean_tokens_per_step'] > 0 else 0
        print(f"  k={r['top_k']:>2}: {r['mean_acceptance_rate']*100:.2f}% acceptance, "
              f"{r['mean_tokens_per_step']:.2f} toks/step "
              f"({speedup:.2f}x vs k={baseline['top_k']})")


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    # Remove per-request details for compact output
    compact_results = []
    for r in results:
        compact = {k: v for k, v in r.items() if k != 'requests'}
        compact_results.append(compact)

    with open(output_path, 'w') as f:
        json.dump(compact_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Top-K Longest Response Analysis")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_tick", type=int, default=1)
    parser.add_argument("--cache_source", type=str, default="secondary")
    parser.add_argument("--sim_tick", type=int, default=2)
    parser.add_argument("--port", type=int, default=16399)
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16",
                        help="Comma-separated k values to test")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="topk_analysis_results.json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    config = AnalysisConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        cache_tick=args.cache_tick,
        cache_source=args.cache_source,
        sim_tick=args.sim_tick,
        port=args.port,
        k_values=k_values,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    results = run_analysis(config)
    print_results(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
