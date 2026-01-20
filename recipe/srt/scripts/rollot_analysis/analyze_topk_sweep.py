#!/usr/bin/env python3
"""
Multi-Step Top-K Analysis Sweep for SRT Speculative Decoding.

Runs top-k analysis across multiple training steps to study how the
cache population trade-off evolves during training.

Usage:
    python recipe/srt/topk_sweep_analysis.py \
        --model_path Qwen/Qwen3-8B-Base \
        --data_dir /path/to/rollout_data \
        --step_interval 5
"""

import argparse
import json
import multiprocessing
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

import numpy as np


@dataclass
class SweepConfig:
    """Configuration for multi-step sweep."""
    model_path: str = ""
    data_dir: str = ""
    cache_source: str = "secondary"

    # Step selection
    step_interval: int = 5  # Run every N steps
    start_step: int = 1
    end_step: int = -1  # -1 = auto-detect

    # Server settings
    port: int = 16399
    shm_name: str = "SWEEP_CACHE"

    # Speculation parameters
    spec_start_len: int = 2
    spec_max_len: int = 16
    spec_prefix_len: int = 7
    min_token_prob: float = 0.05

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
            sys.exit(1)
        if not server.start():
            sys.exit(1)
        ready_event.set()
        server.wait()
    except Exception as e:
        print(f"[Server] Error: {e}")
        sys.exit(1)


def _populate_cache_worker_topk(config_dict: dict, cache_tick: int, top_k: int, result_queue):
    """Populate cache with top-k longest responses."""
    try:
        from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config_dict['model_path'],
            trust_remote_code=True
        )

        def tokenize(text: str) -> List[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        data_path = Path(config_dict['data_dir']) / config_dict['cache_source'] / f"{cache_tick}.jsonl"
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        updater = SuffixCacheUpdater(server_addresses=[f"127.0.0.1:{config_dict['port']}"])

        is_secondary = config_dict['cache_source'] == "secondary"
        prompt_to_responses = defaultdict(list)

        for item in data:
            if is_secondary:
                prompt_text = item['prompt']
                response_text = item['response']
            else:
                prompt_text = item['input']
                response_text = item['output']

            prompt_tokens = tuple(tokenize(prompt_text))
            response_tokens = tokenize(response_text)

            if len(response_tokens) == 0:
                continue

            prompt_to_responses[prompt_tokens].append({
                'tokens': response_tokens,
                'length': len(response_tokens),
            })

        # Sort and take top-k
        total_responses = 0
        for prompt_tokens in prompt_to_responses:
            responses = prompt_to_responses[prompt_tokens]
            responses.sort(key=lambda x: x['length'], reverse=True)
            prompt_to_responses[prompt_tokens] = responses[:top_k]
            total_responses += len(prompt_to_responses[prompt_tokens])

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


def _simulate_worker(config_dict: dict, sim_tick: int, top_k: int, result_queue):
    """Run simulation for a specific tick."""
    try:
        from srt_plugin.shm_cache.suffix_cache import SuffixCache
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config_dict['model_path'],
            trust_remote_code=True
        )

        def tokenize(text: str) -> np.ndarray:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return np.array(tokens, dtype=np.int32)

        data_path = Path(config_dict['data_dir']) / "rollout" / f"{sim_tick}.jsonl"
        sim_data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    sim_data.append(json.loads(line))

        if config_dict['max_samples'] > 0:
            sim_data = sim_data[:config_dict['max_samples']]

        cache = SuffixCache(
            shared_memory_name=config_dict['shm_name'],
            spec_start_len=config_dict['spec_start_len'],
            spec_max_len=config_dict['spec_max_len'],
        )

        # Brief wait for cache visibility
        time.sleep(1.0)

        results = []
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

                pattern_size = min(len(sequence), config_dict['spec_prefix_len'])
                pattern = sequence[-pattern_size:].tolist()

                try:
                    drafts = cache.speculate(
                        [request_id],
                        [pattern],
                        min_token_prob=config_dict['min_token_prob'],
                    )
                    draft_tokens = drafts[0] if drafts and drafts[0] else []
                except Exception:
                    draft_tokens = []

                accepted_tokens = []
                remaining_gt = ground_truth[len(response):]

                for j, draft_tok in enumerate(draft_tokens):
                    if j < len(remaining_gt) and draft_tok == remaining_gt[j]:
                        accepted_tokens.append(draft_tok)
                    else:
                        break

                new_tokens = accepted_tokens.copy()
                response.extend(accepted_tokens)

                if len(response) < len(ground_truth):
                    bonus_token = int(ground_truth[len(response)])
                    new_tokens.append(bonus_token)
                    response.append(bonus_token)

                # Update cache's internal spec_len (this is the key call!)
                # num_accepted passed to update_spec_len determines if we grow or shrink
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

            results.append({
                "total_steps": len(steps),
                "total_accept_toks": total_accept,
                "total_spec_toks": total_spec,
                "tokens_per_step": total_out / len(steps) if steps else 0.0,
            })

        total_steps = sum(r["total_steps"] for r in results)
        total_accept = sum(r["total_accept_toks"] for r in results)
        total_spec = sum(r["total_spec_toks"] for r in results)
        total_out = sum(r["total_steps"] * r["tokens_per_step"] for r in results)

        summary = {
            "num_requests": len(results),
            "total_steps": total_steps,
            "total_accept_toks": total_accept,
            "total_spec_toks": total_spec,
            "mean_acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
            "mean_tokens_per_step": total_out / total_steps if total_steps > 0 else 0.0,
        }

        result_queue.put(("success", summary))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def run_single_analysis(config_dict: dict, cache_tick: int, sim_tick: int, top_k: int) -> Dict[str, Any]:
    """Run analysis for a single (cache_tick, sim_tick, k) combination."""
    shm_name = f"{config_dict['shm_name']}_{cache_tick}_{top_k}"
    local_config = config_dict.copy()
    local_config['shm_name'] = shm_name

    ready_event = multiprocessing.Event()
    server_proc = multiprocessing.Process(
        target=_server_subprocess_main,
        args=(config_dict['port'], shm_name, ready_event),
        daemon=True
    )
    server_proc.start()

    if not ready_event.wait(timeout=10):
        if not server_proc.is_alive():
            raise RuntimeError("Cache server failed to start")

    time.sleep(0.5)

    try:
        # Populate cache
        updater_queue = multiprocessing.Queue()
        updater_proc = multiprocessing.Process(
            target=_populate_cache_worker_topk,
            args=(local_config, cache_tick, top_k, updater_queue),
            daemon=True
        )
        updater_proc.start()
        updater_proc.join(timeout=300)

        if updater_proc.is_alive():
            updater_proc.terminate()
            raise RuntimeError("Cache population timed out")

        if updater_queue.empty():
            raise RuntimeError("No results from updater")

        status, result = updater_queue.get()
        if status == "error":
            raise RuntimeError(f"Updater failed: {result}")

        time.sleep(0.5)

        # Run simulation
        sim_queue = multiprocessing.Queue()
        sim_proc = multiprocessing.Process(
            target=_simulate_worker,
            args=(local_config, sim_tick, top_k, sim_queue),
            daemon=True
        )
        sim_proc.start()
        sim_proc.join(timeout=600)

        if sim_proc.is_alive():
            sim_proc.terminate()
            raise RuntimeError("Simulation timed out")

        if sim_queue.empty():
            raise RuntimeError("No results from simulator")

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


def run_sweep(config: SweepConfig) -> Dict[str, Any]:
    """Run the full sweep analysis."""
    data_dir = Path(config.data_dir)

    # Find available ticks
    secondary_ticks = sorted([int(f.stem) for f in (data_dir / config.cache_source).glob('*.jsonl')])
    rollout_ticks = sorted([int(f.stem) for f in (data_dir / 'rollout').glob('*.jsonl')])

    # Build valid pairs
    end_step = config.end_step if config.end_step > 0 else max(secondary_ticks)
    valid_pairs = []
    for cache_tick in range(config.start_step, end_step + 1, config.step_interval):
        if cache_tick in secondary_ticks:
            sim_tick = cache_tick + 1
            if sim_tick in rollout_ticks:
                valid_pairs.append((cache_tick, sim_tick))

    print(f"Running sweep over {len(valid_pairs)} tick pairs: {valid_pairs}")
    print(f"K values: {config.k_values}")
    print()

    config_dict = {
        'model_path': config.model_path,
        'data_dir': config.data_dir,
        'cache_source': config.cache_source,
        'port': config.port,
        'shm_name': config.shm_name,
        'spec_start_len': config.spec_start_len,
        'spec_max_len': config.spec_max_len,
        'spec_prefix_len': config.spec_prefix_len,
        'min_token_prob': config.min_token_prob,
        'max_samples': config.max_samples,
    }

    all_results = {}

    for cache_tick, sim_tick in valid_pairs:
        print(f"{'='*60}")
        print(f"Step {cache_tick} -> {sim_tick}")
        print('='*60)

        step_results = {}
        for k in config.k_values:
            print(f"  k={k}...", end=" ", flush=True)
            try:
                result = run_single_analysis(config_dict, cache_tick, sim_tick, k)
                step_results[k] = result
                print(f"accept={result['mean_acceptance_rate']:.3f}, toks/step={result['mean_tokens_per_step']:.2f}")
            except Exception as e:
                print(f"FAILED: {e}")
                step_results[k] = None

        all_results[cache_tick] = step_results

    return {
        'tick_pairs': valid_pairs,
        'k_values': config.k_values,
        'results': all_results,
    }


def print_sweep_results(sweep_results: Dict[str, Any]):
    """Print sweep results as tables."""
    k_values = sweep_results['k_values']
    results = sweep_results['results']

    print("\n" + "="*80)
    print("SWEEP RESULTS: Acceptance Rate by Step and K")
    print("="*80)

    # Header
    print(f"{'Step':>6}", end="")
    for k in k_values:
        print(f" | k={k:>2}", end="")
    print(" | k=16 rel")
    print("-"*70)

    # Data rows
    for step in sorted(results.keys()):
        step_data = results[step]
        print(f"{step:>6}", end="")

        baseline = step_data.get(16, {})
        baseline_rate = baseline.get('mean_acceptance_rate', 0) if baseline else 0

        for k in k_values:
            data = step_data.get(k)
            if data:
                print(f" | {data['mean_acceptance_rate']:.3f}", end="")
            else:
                print(f" |   N/A", end="")

        # Relative to k=16
        if baseline and baseline_rate > 0:
            for k in k_values[:-1]:
                data = step_data.get(k)
                if data:
                    rel = data['mean_acceptance_rate'] / baseline_rate
                    print(f" {rel:.2f}", end="")
        print()

    # Tokens per step table
    print("\n" + "="*80)
    print("SWEEP RESULTS: Tokens per Step by Step and K")
    print("="*80)

    print(f"{'Step':>6}", end="")
    for k in k_values:
        print(f" | k={k:>2}", end="")
    print()
    print("-"*60)

    for step in sorted(results.keys()):
        step_data = results[step]
        print(f"{step:>6}", end="")
        for k in k_values:
            data = step_data.get(k)
            if data:
                print(f" | {data['mean_tokens_per_step']:.2f}", end="")
            else:
                print(f" |  N/A", end="")
        print()

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Average across all steps")
    print("="*80)

    avg_by_k = {k: {'accept': [], 'tps': []} for k in k_values}
    for step, step_data in results.items():
        for k in k_values:
            data = step_data.get(k)
            if data:
                avg_by_k[k]['accept'].append(data['mean_acceptance_rate'])
                avg_by_k[k]['tps'].append(data['mean_tokens_per_step'])

    print(f"{'k':>4} | {'Avg Accept':>12} | {'Avg Toks/Step':>14} | {'vs k=16':>8}")
    print("-"*50)

    baseline_accept = np.mean(avg_by_k[16]['accept']) if avg_by_k[16]['accept'] else 0
    baseline_tps = np.mean(avg_by_k[16]['tps']) if avg_by_k[16]['tps'] else 0

    for k in k_values:
        if avg_by_k[k]['accept']:
            avg_accept = np.mean(avg_by_k[k]['accept'])
            avg_tps = np.mean(avg_by_k[k]['tps'])
            rel = avg_tps / baseline_tps if baseline_tps > 0 else 0
            print(f"{k:>4} | {avg_accept:>11.4f} | {avg_tps:>14.3f} | {rel:>7.2f}x")


def save_sweep_results(sweep_results: Dict[str, Any], output_path: str):
    """Save sweep results to JSON."""
    # Convert int keys to strings for JSON
    json_results = {
        'tick_pairs': sweep_results['tick_pairs'],
        'k_values': sweep_results['k_values'],
        'results': {str(k): v for k, v in sweep_results['results'].items()}
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Step Top-K Sweep Analysis")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_source", type=str, default="secondary")
    parser.add_argument("--step_interval", type=int, default=5)
    parser.add_argument("--start_step", type=int, default=1)
    parser.add_argument("--end_step", type=int, default=-1)
    parser.add_argument("--port", type=int, default=16399)
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="topk_sweep_results.json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    config = SweepConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        cache_source=args.cache_source,
        step_interval=args.step_interval,
        start_step=args.start_step,
        end_step=args.end_step,
        port=args.port,
        k_values=k_values,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    sweep_results = run_sweep(config)
    print_sweep_results(sweep_results)
    save_sweep_results(sweep_results, args.output)


if __name__ == "__main__":
    main()
