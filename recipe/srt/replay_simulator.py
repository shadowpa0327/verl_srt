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
Replay Simulator for SRT Speculative Decoding.

This simulator reads collected rollout data and replays the SRT speculation
process to measure acceptance rates offline.

Usage:
    python recipe/srt/replay_simulator.py \
        --model_path /path/to/model \
        --data_dir /path/to/rollout_datas/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \
        --cache_tick 1 \
        --sim_tick 2

The simulator:
1. Starts a local cache server
2. Populates the cache with tick N's rollout data (separate process)
3. Simulates speculation for tick N+1's prompts (separate process)
4. Reports acceptance rate statistics

Note: Due to protobuf conflicts, SuffixCacheUpdater and SuffixCache cannot be
imported in the same process. This simulator uses separate subprocesses for
cache population and simulation.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np


@dataclass
class SimulatorConfig:
    """Configuration for replay simulator."""
    # Model/Tokenizer
    model_path: str = ""

    # Data paths
    data_dir: str = ""
    cache_tick: int = 1  # Tick to populate cache from
    cache_source: str = "secondary"  # "rollout" or "secondary" - which data to use for cache
    sim_tick: int = 2    # Tick to simulate speculation for

    # Mode selection
    mode: str = "shm"  # "shm" (shared memory) or "parallel" (in-process ParallelCache)

    # Cache server settings (for shm mode)
    port: int = 16399
    shm_name: str = "REPLAY_SIM_CACHE"

    # Speculation parameters
    spec_start_len: int = 2
    spec_max_len: int = 16
    spec_prefix_len: int = 7  # Number of recent tokens for pattern
    min_token_prob: float = 0.1
    use_tree_spec: bool = False

    # ParallelCache parameters (for parallel mode)
    max_tree_depth: int = 64
    num_threads: int = -1  # -1 = auto-detect
    parallel_threshold: int = 4
    hash_token_count: int = 128

    # Simulation options
    max_samples: int = 0  # 0 = all samples
    verbose: bool = False
    online_update: bool = False  # Enable on-the-fly cache updates (parallel mode only)


def _server_subprocess_main(port: int, shm_name: str, ready_event):
    """Cache server subprocess entry point."""
    try:
        from srt_plugin.shm_cache.suffix_cache import RolloutCacheServer
        server = RolloutCacheServer(f"[::]:{port}", 0, shm_name)
        if not server.initialize():
            print(f"Failed to initialize cache server")
            sys.exit(1)
        if not server.start():
            print(f"Failed to start cache server")
            sys.exit(1)
        ready_event.set()
        server.wait()
    except Exception as e:
        print(f"Cache server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _populate_cache_worker(config_dict: dict, result_queue):
    """
    Worker subprocess that populates the cache.

    Uses SuffixCacheUpdater (cannot be in same process as SuffixCache).
    Supports both "rollout" format (input/output keys) and "secondary" format (prompt/response keys).
    """
    try:
        config = SimulatorConfig(**config_dict)

        # Import only cache_updater module here
        from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater
        from transformers import AutoTokenizer

        # Load tokenizer
        print(f"[Updater] Loading tokenizer from {config.model_path}...")
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

        # Create updater
        server_address = f"127.0.0.1:{config.port}"
        updater = SuffixCacheUpdater(server_addresses=[server_address])

        # Load cache data (from rollout or secondary)
        cache_data = load_data(config.cache_tick, config.cache_source)
        print(f"[Updater] Loaded {len(cache_data)} samples from {config.cache_source}/{config.cache_tick}.jsonl")

        # Group responses by prompt to send more efficiently
        from collections import defaultdict
        prompt_to_responses = defaultdict(list)

        # Handle different data formats
        is_secondary = config.cache_source == "secondary"

        for item in cache_data:
            if is_secondary:
                # Secondary format: prompt/response are already text
                prompt_text = item['prompt']
                response_text = item['response']
            else:
                # Rollout format: input/output
                prompt_text = item['input']
                response_text = item['output']

            prompt_tokens = tuple(tokenize(prompt_text))  # tuple for dict key
            response_tokens = tokenize(response_text)

            if len(response_tokens) == 0:
                continue  # Skip empty responses

            prompt_to_responses[prompt_tokens].append(response_tokens)

        # Flatten grouped data for the updater API
        prompts = []
        responses = []
        prompt_lengths = []
        response_lengths = []
        responses_per_prompt_list = []

        for prompt_tokens, resp_list in prompt_to_responses.items():
            for resp_tokens in resp_list:
                prompts.append(list(prompt_tokens))
                responses.append(resp_tokens)
                prompt_lengths.append(float(len(prompt_tokens)))
                response_lengths.append(float(len(resp_tokens)))
            responses_per_prompt_list.append(len(resp_list))

        print(f"[Updater] Grouped into {len(prompt_to_responses)} unique prompts")

        # Send grouped by prompt - use the count from first prompt (all should be same in typical use)
        # For varying counts, we need to send multiple batches
        unique_prompts = list(prompt_to_responses.keys())
        for i, prompt_tokens in enumerate(unique_prompts):
            resp_list = prompt_to_responses[prompt_tokens]
            batch_prompts = [list(prompt_tokens)] * len(resp_list)
            batch_responses = resp_list
            batch_prompt_lens = [float(len(prompt_tokens))] * len(resp_list)
            batch_response_lens = [float(len(r)) for r in resp_list]

            updater.update_response_cache(
                prompts=batch_prompts,
                responses=batch_responses,
                prompt_lengths=batch_prompt_lens,
                response_lengths=batch_response_lens,
                responses_per_prompt=len(resp_list),
            )

        print(f"[Updater] Cache populated with {len(prompts)} responses")
        result_queue.put(("success", len(prompts)))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def _simulate_worker(config_dict: dict, result_queue):
    """
    Worker subprocess that runs the simulation.

    Uses SuffixCache (cannot be in same process as SuffixCacheUpdater).
    """
    try:
        config = SimulatorConfig(**config_dict)

        # Import only suffix_cache module here
        from srt_plugin.shm_cache.suffix_cache import SuffixCache
        from transformers import AutoTokenizer

        # Load tokenizer
        print(f"[Simulator] Loading tokenizer from {config.model_path}...")
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

        # Create cache client
        cache = SuffixCache(
            shared_memory_name=config.shm_name,
            spec_start_len=config.spec_start_len,
            spec_max_len=config.spec_max_len,
        )

        # Load simulation data
        sim_data = load_rollout_data(config.sim_tick)
        if config.max_samples > 0:
            sim_data = sim_data[:config.max_samples]

        # Also load cache source data for verification
        cache_source_data = load_rollout_data(config.cache_tick, config.cache_source)

        # Verification step: Wait until trees are visible in shared memory
        # This fixes the race condition between updater completing and data being visible
        print(f"[Simulator] Verifying cache is populated...")
        max_retries = 20
        retry_delay = 0.5  # seconds
        verified = False

        # Get a sample of prompts to verify (first 3 unique prompts from cache source)
        is_secondary = config.cache_source == "secondary"
        verify_prompts = []
        seen_prompts = set()
        for item in cache_source_data:
            if is_secondary:
                prompt_text = item['prompt']
            else:
                prompt_text = item['input']
            if prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                verify_prompts.append(tokenize(prompt_text))
                if len(verify_prompts) >= 3:
                    break

        for retry in range(max_retries):
            trees_found = 0
            for i, prompt in enumerate(verify_prompts):
                req_id = f"verify_{i}"
                cache.fetch_responses_by_prompts_batch([req_id], [prompt.tolist()])
                # Try speculate with the prompt's last few tokens as pattern
                pattern = prompt[-config.spec_prefix_len:].tolist() if len(prompt) >= config.spec_prefix_len else prompt.tolist()
                drafts = cache.speculate([req_id], [pattern], min_token_prob=config.min_token_prob)
                if retry == 0 and config.verbose:
                    print(f"[Simulator] Verify prompt {i}: len={len(prompt)}, pattern={pattern[-5:]}, drafts={drafts[0][:5] if drafts and drafts[0] else []}")
                cache.evict_responses(req_id)
                if drafts and drafts[0]:
                    trees_found += 1

            if trees_found >= len(verify_prompts):
                print(f"[Simulator] Cache verified after {retry + 1} attempts ({trees_found}/{len(verify_prompts)} trees found)")
                verified = True
                break
            elif trees_found > 0:
                print(f"[Simulator] Partial cache ({trees_found}/{len(verify_prompts)} trees), retry {retry + 1}/{max_retries}...")
            else:
                if retry == 0:
                    print(f"[Simulator] Waiting for cache to be visible (retry {retry + 1}/{max_retries})...")

            time.sleep(retry_delay)

        if not verified:
            print(f"[Simulator] WARNING: Cache verification failed after {max_retries} retries. Proceeding anyway...")

        print(f"[Simulator] Running simulation on {len(sim_data)} requests...")

        results = []
        for i, item in enumerate(sim_data):
            request_id = f"sim_{i}"
            prompt = tokenize(item['input'])
            ground_truth = tokenize(item['output'])

            if len(ground_truth) == 0:
                if config.verbose:
                    print(f"  Skipping request {i}: empty response")
                continue

            # Fetch tree for this prompt
            cache.fetch_responses_by_prompts_batch([request_id], [prompt.tolist()])

            steps = []
            response = []

            while len(response) < len(ground_truth):
                # Build pattern from recent tokens
                if response:
                    sequence = np.concatenate([prompt, np.array(response, dtype=np.int32)])
                else:
                    sequence = prompt

                pattern_size = min(len(sequence), config.spec_prefix_len)
                pattern = sequence[-pattern_size:].tolist()

                # Speculate
                start_time = time.perf_counter()
                try:
                    drafts = cache.speculate(
                        [request_id],
                        [pattern],
                        min_token_prob=config.min_token_prob,
                        use_tree_spec=config.use_tree_spec,
                    )
                    draft_tokens = drafts[0] if drafts and drafts[0] else []
                except Exception as e:
                    if config.verbose:
                        print(f"  Speculation failed: {e}")
                    draft_tokens = []
                spec_time = (time.perf_counter() - start_time) * 1000

                # Verify against ground truth
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

                steps.append({
                    "step": len(steps),
                    "num_spec_toks": len(draft_tokens),
                    "num_accept_toks": len(accepted_tokens),
                    "num_out_toks": len(new_tokens),
                    "spec_ms": spec_time,
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

            if config.verbose or (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(sim_data)}] "
                      f"steps={result['total_steps']}, "
                      f"accept_rate={result['acceptance_rate']:.3f}, "
                      f"toks/step={result['tokens_per_step']:.2f}")

        # Compute aggregates
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
            "requests": results,
        }

        result_queue.put(("success", summary))

    except Exception as e:
        import traceback
        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def _run_parallel_simulation(config: SimulatorConfig) -> Dict[str, Any]:
    """
    Run simulation using in-process ParallelSuffixDecodingCache.

    This mode is simpler as it doesn't require separate processes for
    server/updater - everything runs in a single process using the
    ParallelSuffixDecodingCache which manages suffix trees directly.
    """
    from srt_plugin.suffix_cache import ParallelSuffixDecodingCache
    from transformers import AutoTokenizer
    from collections import defaultdict

    # Load tokenizer
    print(f"Loading tokenizer from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True
    )

    def tokenize(text: str) -> np.ndarray:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return np.array(tokens, dtype=np.int32)

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

    # Create ParallelSuffixDecodingCache
    online_str = ", online_update=ON" if config.online_update else ""
    print(f"Creating ParallelSuffixDecodingCache (max_depth={config.max_tree_depth}, "
          f"threads={config.num_threads}, hash_token_count={config.hash_token_count}{online_str})...")
    cache = ParallelSuffixDecodingCache(
        max_tree_depth=config.max_tree_depth,
        num_threads=config.num_threads,
        parallel_threshold=config.parallel_threshold,
        hash_token_count=config.hash_token_count,
    )

    # Step 1: Load and populate cache with tick N's data
    print(f"\n[Step 1] Populating cache from {config.cache_source}/{config.cache_tick}.jsonl...")
    cache_data = load_data(config.cache_tick, config.cache_source)
    print(f"  Loaded {len(cache_data)} samples")

    is_secondary = config.cache_source == "secondary"

    # Group responses by prompt
    prompt_to_responses = defaultdict(list)
    for item in cache_data:
        if is_secondary:
            prompt_text = item['prompt']
            response_text = item['response']
        else:
            prompt_text = item['input']
            response_text = item['output']

        prompt_tokens = tuple(tokenize(prompt_text).tolist())
        response_tokens = tokenize(response_text)

        if len(response_tokens) == 0:
            continue

        prompt_to_responses[prompt_tokens].append(response_tokens)

    # Populate cache: start a request for each prompt, add all responses
    # Note: We keep populate requests active so trees remain accessible via hash lookup.
    # Trees would be removed if we called stop_request (removing hash mapping too).
    total_added = 0
    populate_req_ids = []
    for prompt_idx, (prompt_tokens, resp_list) in enumerate(prompt_to_responses.items()):
        temp_req_id = f"cache_populate_{prompt_idx}"
        prompt_arr = np.array(prompt_tokens, dtype=np.int32)
        populate_req_ids.append(temp_req_id)

        # Start request (creates/reuses tree based on hash)
        cache.start_request(temp_req_id, prompt_arr)

        # Add all responses to the tree
        for resp_tokens in resp_list:
            # Add prompt + response to tree to build the suffix patterns
            full_sequence = np.concatenate([prompt_arr, resp_tokens])
            cache.add_tokens(temp_req_id, full_sequence)
            total_added += 1

    print(f"  Cache populated with {total_added} responses from {len(prompt_to_responses)} unique prompts")
    stats = cache.get_stats()
    print(f"  Forest stats: {stats['num_trees_in_forest']} trees, {stats['num_active_requests']} active requests")

    # Step 2: Run simulation for tick N+1
    print(f"\n[Step 2] Running simulation on rollout/{config.sim_tick}.jsonl...")
    sim_data = load_data(config.sim_tick, "rollout")
    if config.max_samples > 0:
        sim_data = sim_data[:config.max_samples]

    print(f"  Simulating {len(sim_data)} requests...")

    results = []
    for i, item in enumerate(sim_data):
        request_id = f"sim_{i}"
        prompt = tokenize(item['input'])
        ground_truth = tokenize(item['output'])

        if len(ground_truth) == 0:
            if config.verbose:
                print(f"  Skipping request {i}: empty response")
            continue

        # Start request (will lookup existing tree via hash)
        cache.start_request(request_id, prompt)

        steps = []
        response = []

        while len(response) < len(ground_truth):
            # Build context from recent tokens
            if response:
                sequence = np.concatenate([prompt, np.array(response, dtype=np.int32)])
            else:
                sequence = prompt

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
            remaining_gt = ground_truth[len(response):]

            for j, draft_tok in enumerate(draft_tokens):
                if j < len(remaining_gt) and draft_tok == remaining_gt[j]:
                    accepted_tokens.append(draft_tok)
                else:
                    break

            new_tokens = accepted_tokens.copy()
            response.extend(accepted_tokens)

            # Add bonus token (simulating model verification step)
            if len(response) < len(ground_truth):
                bonus_token = int(ground_truth[len(response)])
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

        if config.verbose or (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(sim_data)}] "
                  f"steps={result['total_steps']}, "
                  f"accept_rate={result['acceptance_rate']:.3f}, "
                  f"toks/step={result['tokens_per_step']:.2f}")

    # Compute aggregates
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
        "requests": results,
    }

    # Cleanup populate requests (optional, process will exit anyway)
    for req_id in populate_req_ids:
        try:
            cache.stop_request(req_id)
        except ValueError:
            pass  # Already stopped

    return summary


def _run_shm_simulation(config: SimulatorConfig) -> Dict[str, Any]:
    """Run the simulation and return results."""
    # Start cache server in subprocess
    ready_event = multiprocessing.Event()
    server_proc = multiprocessing.Process(
        target=_server_subprocess_main,
        args=(config.port, config.shm_name, ready_event),
        daemon=True
    )
    server_proc.start()

    # Wait for server to be ready
    if not ready_event.wait(timeout=10):
        if not server_proc.is_alive():
            raise RuntimeError("Cache server failed to start")
        print("Warning: Server ready event not received, proceeding anyway...")

    time.sleep(1)  # Extra time for gRPC to be fully ready
    print(f"Cache server started (PID: {server_proc.pid})")

    try:
        # Step 1: Populate cache in separate subprocess
        # (SuffixCacheUpdater cannot be imported with SuffixCache due to protobuf conflict)
        print("\n[Step 1] Populating cache...")
        updater_queue = multiprocessing.Queue()
        updater_proc = multiprocessing.Process(
            target=_populate_cache_worker,
            args=(config.__dict__, updater_queue),
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

        print(f"[Step 1] Cache populated with {result} samples")

        # Brief pause before starting simulator - actual synchronization is done
        # via verification step in the simulator subprocess
        time.sleep(0.5)

        # Step 2: Run simulation in separate subprocess
        # (SuffixCache for reading from shared memory)
        print("\n[Step 2] Running simulation...")
        sim_queue = multiprocessing.Queue()
        sim_proc = multiprocessing.Process(
            target=_simulate_worker,
            args=(config.__dict__, sim_queue),
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
        print("\nCache server stopped")


def run_simulation(config: SimulatorConfig) -> Dict[str, Any]:
    """
    Run the simulation based on the configured mode.

    Args:
        config: SimulatorConfig with mode set to "shm" or "parallel".

    Returns:
        Dictionary with simulation results.
    """
    if config.mode == "parallel":
        print(f"Running in PARALLEL mode (in-process ParallelSuffixDecodingCache)")
        return _run_parallel_simulation(config)
    elif config.mode == "shm":
        print(f"Running in SHM mode (shared memory with separate processes)")
        return _run_shm_simulation(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}. Must be 'shm' or 'parallel'.")


def print_results(result: Dict[str, Any]):
    """Print simulation results."""
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)
    print(f"  Requests simulated:    {result['num_requests']}")
    print(f"  Total decoding steps:  {result['total_steps']}")
    print(f"  Total speculated toks: {result['total_spec_toks']}")
    print(f"  Total accepted toks:   {result['total_accept_toks']}")
    print("-" * 60)
    print(f"  Mean acceptance rate:  {result['mean_acceptance_rate']:.4f} ({result['mean_acceptance_rate']*100:.2f}%)")
    print(f"  Mean tokens/step:      {result['mean_tokens_per_step']:.3f}")
    print("=" * 60)

    # Per-request statistics
    if result.get('requests'):
        accept_rates = [r['acceptance_rate'] for r in result['requests']]
        toks_per_step = [r['tokens_per_step'] for r in result['requests']]

        print("\nPer-Request Statistics:")
        print(f"  Acceptance rate: min={min(accept_rates):.3f}, max={max(accept_rates):.3f}, "
              f"median={np.median(accept_rates):.3f}")
        print(f"  Tokens/step:     min={min(toks_per_step):.2f}, max={max(toks_per_step):.2f}, "
              f"median={np.median(toks_per_step):.2f}")


def main():
    parser = argparse.ArgumentParser(description="SRT Replay Simulator")

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model for tokenizer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to rollout data directory")

    # Mode selection
    parser.add_argument("--mode", type=str, default="shm",
                        choices=["shm", "parallel"],
                        help="Simulation mode: 'shm' (shared memory, default) or 'parallel' (in-process ParallelCache)")

    # Tick configuration
    parser.add_argument("--cache_tick", type=int, default=1,
                        help="Tick number to populate cache from (default: 1)")
    parser.add_argument("--cache_source", type=str, default="secondary",
                        choices=["rollout", "secondary"],
                        help="Data source for cache: 'secondary' (default, for SRT workflow) or 'rollout'")
    parser.add_argument("--sim_tick", type=int, default=2,
                        help="Tick number to simulate (default: 2)")

    # Cache server settings (for shm mode)
    parser.add_argument("--port", type=int, default=16399,
                        help="Cache server port for shm mode (default: 16399)")
    parser.add_argument("--shm_name", type=str, default="REPLAY_SIM_CACHE",
                        help="Shared memory segment name for shm mode (default: REPLAY_SIM_CACHE)")

    # Speculation parameters
    parser.add_argument("--spec_start_len", type=int, default=2,
                        help="Initial speculation length (default: 2)")
    parser.add_argument("--spec_max_len", type=int, default=16,
                        help="Maximum speculation length (default: 16)")
    parser.add_argument("--spec_prefix_len", type=int, default=7,
                        help="Pattern length for speculation (default: 7)")
    parser.add_argument("--min_token_prob", type=float, default=0.1,
                        help="Minimum token probability (default: 0.1)")

    # ParallelCache parameters (for parallel mode)
    parser.add_argument("--max_tree_depth", type=int, default=64,
                        help="Maximum tree depth for parallel mode (default: 64)")
    parser.add_argument("--num_threads", type=int, default=-1,
                        help="Number of threads for parallel mode (-1=auto, default: -1)")
    parser.add_argument("--parallel_threshold", type=int, default=4,
                        help="Min batch size to trigger parallelization (default: 4)")
    parser.add_argument("--hash_token_count", type=int, default=128,
                        help="Trailing tokens to hash for tree sharing (default: 128)")

    # Simulation options
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum samples to simulate (0=all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--online_update", action="store_true",
                        help="Enable on-the-fly cache updates during simulation (parallel mode only)")

    args = parser.parse_args()

    config = SimulatorConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        mode=args.mode,
        cache_tick=args.cache_tick,
        cache_source=args.cache_source,
        sim_tick=args.sim_tick,
        port=args.port,
        shm_name=args.shm_name,
        spec_start_len=args.spec_start_len,
        spec_max_len=args.spec_max_len,
        spec_prefix_len=args.spec_prefix_len,
        min_token_prob=args.min_token_prob,
        max_tree_depth=args.max_tree_depth,
        num_threads=args.num_threads,
        parallel_threshold=args.parallel_threshold,
        hash_token_count=args.hash_token_count,
        max_samples=args.max_samples,
        verbose=args.verbose,
        online_update=args.online_update,
    )

    result = run_simulation(config)
    print_results(result)


if __name__ == "__main__":
    main()
