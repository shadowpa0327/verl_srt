#!/usr/bin/env python3
"""
Detailed Diagnostic Analysis for SRT Top-K Trade-off.

Provides granular metrics including:
- Per-position acceptance rates
- Spec length evolution tracking
- Distribution of accepted tokens per step
- Detailed per-request breakdowns
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
class DiagnosticConfig:
    """Configuration for diagnostic analysis."""
    model_path: str = ""
    data_dir: str = ""
    cache_source: str = "secondary"
    cache_tick: int = 1
    sim_tick: int = 2

    port: int = 16399
    shm_name: str = "DIAG_CACHE"

    spec_start_len: int = 2
    spec_max_len: int = 16
    spec_prefix_len: int = 7
    min_token_prob: float = 0.1

    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    max_samples: int = 0
    verbose: bool = False


def _server_subprocess_main(port: int, shm_name: str, ready_event):
    """Cache server subprocess."""
    try:
        from srt_plugin.shm_cache.suffix_cache import RolloutCacheServer
        server = RolloutCacheServer(f"[::]:{port}", 0, shm_name)
        if not server.initialize() or not server.start():
            sys.exit(1)
        ready_event.set()
        server.wait()
    except Exception as e:
        print(f"[Server] Error: {e}")
        sys.exit(1)


def _populate_cache_worker(config_dict: dict, top_k: int, result_queue):
    """Populate cache with top-k longest responses."""
    try:
        from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config_dict['model_path'], trust_remote_code=True
        )

        def tokenize(text: str) -> List[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        data_path = Path(config_dict['data_dir']) / config_dict['cache_source'] / f"{config_dict['cache_tick']}.jsonl"
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        updater = SuffixCacheUpdater(server_addresses=[f"127.0.0.1:{config_dict['port']}"])

        is_secondary = config_dict['cache_source'] == "secondary"
        prompt_to_responses = defaultdict(list)

        for item in data:
            prompt_text = item['prompt'] if is_secondary else item['input']
            response_text = item['response'] if is_secondary else item['output']

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


def _diagnostic_worker(config_dict: dict, top_k: int, result_queue):
    """Run detailed diagnostic simulation."""
    try:
        from srt_plugin.shm_cache.suffix_cache import SuffixCache
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config_dict['model_path'], trust_remote_code=True
        )

        def tokenize(text: str) -> np.ndarray:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return np.array(tokens, dtype=np.int32)

        data_path = Path(config_dict['data_dir']) / "rollout" / f"{config_dict['sim_tick']}.jsonl"
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

        time.sleep(1.0)

        # Detailed tracking
        all_steps = []  # All step-level data
        per_position_accept = defaultdict(int)
        per_position_total = defaultdict(int)
        spec_len_history = []  # Track spec_len evolution
        accepted_per_step_dist = defaultdict(int)  # Distribution of accepted tokens per step

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
            request_spec_len_history = []

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

                # Track spec_len (inferred from draft length)
                current_spec_len = len(draft_tokens) if draft_tokens else config_dict['spec_start_len']
                request_spec_len_history.append(current_spec_len)

                accepted_tokens = []
                remaining_gt = ground_truth[len(response):]

                for j, draft_tok in enumerate(draft_tokens):
                    per_position_total[j] += 1
                    if j < len(remaining_gt) and draft_tok == remaining_gt[j]:
                        accepted_tokens.append(draft_tok)
                        per_position_accept[j] += 1
                    else:
                        # Count remaining speculated positions as misses
                        for jj in range(j + 1, len(draft_tokens)):
                            if jj < len(remaining_gt):
                                per_position_total[jj] += 1
                        break

                new_tokens = accepted_tokens.copy()
                response.extend(accepted_tokens)

                if len(response) < len(ground_truth):
                    bonus_token = int(ground_truth[len(response)])
                    new_tokens.append(bonus_token)
                    response.append(bonus_token)

                # Update cache's internal spec_len
                cache.update_spec_len(request_id, len(new_tokens))

                # Track distribution
                accepted_per_step_dist[len(accepted_tokens)] += 1

                steps.append({
                    "step_idx": len(steps),
                    "num_spec_toks": len(draft_tokens),
                    "num_accept_toks": len(accepted_tokens),
                    "num_out_toks": len(new_tokens),
                    "spec_len_before": current_spec_len,
                })
                all_steps.append(steps[-1])

            cache.evict_responses(request_id)
            spec_len_history.extend(request_spec_len_history)

            total_accept = sum(s["num_accept_toks"] for s in steps)
            total_spec = sum(s["num_spec_toks"] for s in steps)
            total_out = sum(s["num_out_toks"] for s in steps)

            results.append({
                "request_id": request_id,
                "prompt_len": len(prompt),
                "response_len": len(ground_truth),
                "total_steps": len(steps),
                "total_accept_toks": total_accept,
                "total_spec_toks": total_spec,
                "total_out_toks": total_out,
                "acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
                "tokens_per_step": total_out / len(steps) if steps else 0.0,
            })

        # Compute aggregates
        total_steps = sum(r["total_steps"] for r in results)
        total_accept = sum(r["total_accept_toks"] for r in results)
        total_spec = sum(r["total_spec_toks"] for r in results)
        total_out = sum(r["total_out_toks"] for r in results)

        # Per-position acceptance rates
        position_acceptance = {}
        for pos in sorted(per_position_total.keys()):
            if per_position_total[pos] > 0:
                position_acceptance[pos] = {
                    "accepted": per_position_accept[pos],
                    "total": per_position_total[pos],
                    "rate": per_position_accept[pos] / per_position_total[pos],
                }

        # Spec len distribution
        spec_len_dist = defaultdict(int)
        for sl in spec_len_history:
            spec_len_dist[sl] += 1

        summary = {
            "top_k": top_k,
            "num_requests": len(results),
            "total_steps": total_steps,
            "total_accept_toks": total_accept,
            "total_spec_toks": total_spec,
            "total_out_toks": total_out,
            "mean_acceptance_rate": total_accept / total_spec if total_spec > 0 else 0.0,
            "mean_tokens_per_step": total_out / total_steps if total_steps > 0 else 0.0,
            "position_acceptance": position_acceptance,
            "spec_len_distribution": dict(spec_len_dist),
            "accepted_per_step_distribution": dict(accepted_per_step_dist),
            "per_request": results,
        }

        result_queue.put(("success", summary))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def run_diagnostic(config: DiagnosticConfig, top_k: int) -> Dict[str, Any]:
    """Run diagnostic for a single k value."""
    config_dict = {
        'model_path': config.model_path,
        'data_dir': config.data_dir,
        'cache_source': config.cache_source,
        'cache_tick': config.cache_tick,
        'sim_tick': config.sim_tick,
        'port': config.port,
        'shm_name': f"{config.shm_name}_{top_k}",
        'spec_start_len': config.spec_start_len,
        'spec_max_len': config.spec_max_len,
        'spec_prefix_len': config.spec_prefix_len,
        'min_token_prob': config.min_token_prob,
        'max_samples': config.max_samples,
    }

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

    time.sleep(0.5)

    try:
        # Populate cache
        updater_queue = multiprocessing.Queue()
        updater_proc = multiprocessing.Process(
            target=_populate_cache_worker,
            args=(config_dict, top_k, updater_queue),
            daemon=True
        )
        updater_proc.start()
        updater_proc.join(timeout=300)

        if updater_proc.is_alive():
            updater_proc.terminate()
            raise RuntimeError("Cache population timed out")

        status, result = updater_queue.get()
        if status == "error":
            raise RuntimeError(f"Updater failed: {result}")

        time.sleep(0.5)

        # Run diagnostic
        diag_queue = multiprocessing.Queue()
        diag_proc = multiprocessing.Process(
            target=_diagnostic_worker,
            args=(config_dict, top_k, diag_queue),
            daemon=True
        )
        diag_proc.start()
        diag_proc.join(timeout=600)

        if diag_proc.is_alive():
            diag_proc.terminate()
            raise RuntimeError("Diagnostic timed out")

        status, result = diag_queue.get()
        if status == "error":
            raise RuntimeError(f"Diagnostic failed: {result}")

        return result

    finally:
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)
            if server_proc.is_alive():
                server_proc.kill()


def print_diagnostic(results: List[Dict[str, Any]]):
    """Print detailed diagnostic report."""
    print("\n" + "=" * 80)
    print("DETAILED DIAGNOSTIC REPORT")
    print("=" * 80)

    # Summary table
    print("\n## Summary")
    print(f"{'k':>4} | {'Requests':>8} | {'Steps':>8} | {'Accept':>8} | {'Spec':>8} | {'Rate':>8} | {'Toks/Step':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['top_k']:>4} | {r['num_requests']:>8} | {r['total_steps']:>8} | "
              f"{r['total_accept_toks']:>8} | {r['total_spec_toks']:>8} | "
              f"{r['mean_acceptance_rate']:>7.2%} | {r['mean_tokens_per_step']:>10.3f}")

    # Per-position acceptance rates
    print("\n## Per-Position Acceptance Rates")
    max_pos = max(max(int(p) for p in r['position_acceptance'].keys()) for r in results if r['position_acceptance'])

    print(f"{'Pos':>4}", end="")
    for r in results:
        print(f" | k={r['top_k']:>2}", end="")
    print(" | k=16 count")
    print("-" * 70)

    for pos in range(min(max_pos + 1, 16)):
        print(f"{pos:>4}", end="")
        for r in results:
            pos_data = r['position_acceptance'].get(str(pos)) or r['position_acceptance'].get(pos)
            if pos_data:
                print(f" | {pos_data['rate']:>.2%}", end="")
            else:
                print(f" |    - ", end="")
        # Show count for k=16
        k16_data = results[-1]['position_acceptance'].get(str(pos)) or results[-1]['position_acceptance'].get(pos)
        if k16_data:
            print(f" | {k16_data['total']:>6}")
        else:
            print(f" |      -")

    # Spec length distribution
    print("\n## Speculation Length Distribution (k=16)")
    k16_result = results[-1]
    spec_dist = k16_result['spec_len_distribution']
    total_specs = sum(spec_dist.values())
    print(f"{'Spec Len':>8} | {'Count':>8} | {'Pct':>8} | {'Cumulative':>10}")
    print("-" * 45)
    cumulative = 0
    for sl in sorted(int(k) for k in spec_dist.keys()):
        count = spec_dist.get(str(sl)) or spec_dist.get(sl, 0)
        pct = count / total_specs if total_specs > 0 else 0
        cumulative += pct
        print(f"{sl:>8} | {count:>8} | {pct:>7.1%} | {cumulative:>9.1%}")

    # Accepted tokens per step distribution
    print("\n## Accepted Tokens Per Step Distribution (k=16)")
    accept_dist = k16_result['accepted_per_step_distribution']
    total_steps = sum(accept_dist.values())
    print(f"{'Accepted':>8} | {'Count':>8} | {'Pct':>8} | {'Cumulative':>10}")
    print("-" * 45)
    cumulative = 0
    for acc in sorted(int(k) for k in accept_dist.keys()):
        count = accept_dist.get(str(acc)) or accept_dist.get(acc, 0)
        pct = count / total_steps if total_steps > 0 else 0
        cumulative += pct
        print(f"{acc:>8} | {count:>8} | {pct:>7.1%} | {cumulative:>9.1%}")

    # Trade-off summary
    print("\n## Trade-off Summary (vs k=16)")
    baseline = results[-1]
    print(f"{'k':>4} | {'Accept Rate':>12} | {'Toks/Step':>10} | {'Relative':>10} | {'Storage':>10}")
    print("-" * 60)
    for r in results:
        rel_tps = r['mean_tokens_per_step'] / baseline['mean_tokens_per_step'] if baseline['mean_tokens_per_step'] > 0 else 0
        storage_pct = r['top_k'] / baseline['top_k']
        print(f"{r['top_k']:>4} | {r['mean_acceptance_rate']:>11.2%} | {r['mean_tokens_per_step']:>10.3f} | "
              f"{rel_tps:>9.1%} | {storage_pct:>9.1%}")


def main():
    parser = argparse.ArgumentParser(description="Detailed Top-K Diagnostic Analysis")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_source", type=str, default="secondary")
    parser.add_argument("--cache_tick", type=int, default=1)
    parser.add_argument("--sim_tick", type=int, default=2)
    parser.add_argument("--port", type=int, default=16399)
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="topk_diagnostic.json")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    config = DiagnosticConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        cache_source=args.cache_source,
        cache_tick=args.cache_tick,
        sim_tick=args.sim_tick,
        port=args.port,
        k_values=k_values,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    results = []
    for k in k_values:
        print(f"Running diagnostic for k={k}...")
        result = run_diagnostic(config, k)
        results.append(result)
        print(f"  Accept rate: {result['mean_acceptance_rate']:.2%}, Toks/step: {result['mean_tokens_per_step']:.3f}")

    print_diagnostic(results)

    # Save results (without per-request details for compact output)
    compact_results = []
    for r in results:
        compact = {k: v for k, v in r.items() if k != 'per_request'}
        compact_results.append(compact)

    with open(args.output, 'w') as f:
        json.dump(compact_results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
