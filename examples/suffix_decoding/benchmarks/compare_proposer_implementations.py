#!/usr/bin/env python3
"""
Comparison benchmark: Sequential vs Parallel Suffix Decoding Proposer.

This script compares the latency of:
- SuffixDecodingCache (sequential, dual-tree architecture)
- ParallelSuffixDecodingCache (parallel, forest architecture with OpenMP)

Usage:
    python compare_proposer_implementations.py
    python compare_proposer_implementations.py --num-trees 200 --iterations 100
"""

import argparse
import sys
import time

import numpy as np

# =============================================================================
# Import Note: Why we import from arctic_inference instead of vLLM
# =============================================================================
# The vLLM proposers (SuffixDecodingProposer, ParallelSuffixDecodingProposer)
# are located at:
#   - vllm/v1/spec_decode/suffix_decoding.py (sequential)
#   - vllm/v1/spec_decode/suffix_decoding_parallel.py (parallel)
#
# However, importing from vLLM triggers heavy dependencies (CUDA, torch.distributed,
# model configs) that cause circular imports when running standalone benchmarks.
# Even vLLM itself uses lazy imports for this reason (see suffix_decoding.py:23).
#
# We import directly from arctic_inference, which provides the underlying cache
# implementations without vLLM's infrastructure:
#   - SuffixDecodingCache: arctic_inference/suffix_decoding/cache.py
#   - ParallelSuffixDecodingCache: arctic_inference/suffix_decoding/parallel_cache.py
# =============================================================================
try:
    from arctic_inference.suffix_decoding import SuffixDecodingCache
    from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache
except ImportError:
    print("ERROR: Could not import suffix decoding caches from arctic_inference.")
    print("Make sure ArcticInference is installed: pip install arctic-inference")
    sys.exit(1)


class MockInputBatch:
    """Mock InputBatch for benchmarking."""

    def __init__(
        self,
        batch_size: int,
        max_model_len: int = 4096,
        prompt_len: int = 512,
        num_generated: int = 100,
    ):
        self.batch_size = batch_size
        self.max_model_len = max_model_len
        self.prompt_len = prompt_len
        self.num_generated = num_generated

        self.req_ids = [f"req_{i}" for i in range(batch_size)]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}

        total_tokens = prompt_len + num_generated
        self.token_ids_cpu = np.zeros((batch_size, max_model_len), dtype=np.int32)

        for i in range(batch_size):
            self.token_ids_cpu[i, :prompt_len] = np.arange(
                i * 1000, i * 1000 + prompt_len, dtype=np.int32
            )
            self.token_ids_cpu[i, prompt_len:total_tokens] = np.arange(
                i * 1000 + prompt_len, i * 1000 + total_tokens, dtype=np.int32
            )

        self.num_tokens_no_spec = np.full(batch_size, total_tokens, dtype=np.int32)
        self.num_prompt_tokens = np.full(batch_size, prompt_len, dtype=np.int32)
        self.spec_decode_unsupported_reqs = set()
        self.prompt_hashes = {}

    def add_generated_tokens(self, num_new_tokens: int = 1):
        for i in range(self.batch_size):
            current_len = self.num_tokens_no_spec[i]
            new_len = min(current_len + num_new_tokens, self.max_model_len)
            for j in range(current_len, new_len):
                self.token_ids_cpu[i, j] = i * 1000 + j
            self.num_tokens_no_spec[i] = new_len


class SequentialProposer:
    """
    Proposer using SuffixDecodingCache (non-parallel).
    Uses dual-tree architecture: local tree per request + global shared tree.
    """

    def __init__(
        self,
        max_tree_depth: int = 64,
        num_speculative_tokens: int = 5,
        max_model_len: int = 4096,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.max_model_len = max_model_len

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=-1,  # No limit
        )

    def propose(
        self,
        input_batch: MockInputBatch,
        sampled_token_ids: list,
    ) -> list:
        """Sequential propose: process each request one by one."""
        draft_token_ids = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]

            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]

            # Start request if needed
            if req_id not in self.suffix_cache.active_requests:
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Add tokens (sequential)
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Get context for speculation
            start = max(0, num_tokens - self.max_tree_depth)
            context = input_batch.token_ids_cpu[i, start:num_tokens]

            # Speculate (sequential)
            max_spec = min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
            draft = self.suffix_cache.speculate(
                req_id=req_id,
                context=context,
                max_spec_tokens=max_spec,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            draft_token_ids.append(draft.token_ids)

        # Cleanup inactive requests
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())
        for req_id in (active_req_ids - input_req_ids):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def get_stats(self) -> dict:
        return {
            "type": "sequential",
            "num_active_requests": len(list(self.suffix_cache.active_requests)),
            "max_tree_depth": self.max_tree_depth,
        }


class ParallelProposer:
    """
    Proposer using ParallelSuffixDecodingCache (parallel).
    Uses forest architecture with OpenMP parallelization.
    """

    def __init__(
        self,
        max_tree_depth: int = 64,
        num_speculative_tokens: int = 5,
        max_model_len: int = 4096,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
        num_threads: int = 8,
        parallel_threshold: int = 8,
    ):
        self.num_speculative_tokens = num_speculative_tokens
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.max_model_len = max_model_len

        self.suffix_cache = ParallelSuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            num_threads=num_threads,
            parallel_threshold=parallel_threshold,
        )

    def propose(
        self,
        input_batch: MockInputBatch,
        sampled_token_ids: list,
    ) -> list:
        """Parallel propose: batch all operations."""
        req_ids_to_add_tokens = []
        tokens_to_add = []
        req_ids_to_speculate = []
        contexts_to_speculate = []
        max_spec_tokens_list = []
        input_indices_with_drafts = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]

            if num_tokens >= self.max_model_len:
                continue

            index = input_batch.req_id_to_index[req_id]

            # Start request if needed
            if req_id not in self.suffix_cache.active_requests:
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            req_ids_to_add_tokens.append(req_id)
            tokens_to_add.append(sampled_ids)

            start = max(0, num_tokens - self.max_tree_depth)
            context = input_batch.token_ids_cpu[i, start:num_tokens]

            req_ids_to_speculate.append(req_id)
            contexts_to_speculate.append(context)
            max_spec_tokens_list.append(
                min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
            )
            input_indices_with_drafts.append(i)

        # Batch add tokens (parallelized)
        if req_ids_to_add_tokens:
            self.suffix_cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)

        # Batch speculate (parallelized)
        drafts = []
        if req_ids_to_speculate:
            min_max_spec = min(max_spec_tokens_list) if max_spec_tokens_list else self.num_speculative_tokens
            drafts = self.suffix_cache.batch_speculate(
                req_ids=req_ids_to_speculate,
                contexts=contexts_to_speculate,
                max_spec_tokens=min_max_spec,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

        # Build results
        draft_token_ids = []
        draft_idx = 0
        for i in range(len(sampled_token_ids)):
            if i in input_indices_with_drafts:
                draft_token_ids.append(drafts[draft_idx].token_ids)
                draft_idx += 1
            else:
                draft_token_ids.append([])

        # Cleanup inactive requests
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())
        for req_id in (active_req_ids - input_req_ids):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def get_stats(self) -> dict:
        return self.suffix_cache.get_stats()


def pre_init_cache(proposer, num_trees: int, prompt_len: int = 512, response_len: int = 500):
    """Pre-initialize cache with trees."""
    for i in range(num_trees):
        req_id = f"preinit_req_{i}"
        prompt = np.arange(i * 10000, i * 10000 + prompt_len, dtype=np.int32)

        if isinstance(proposer, SequentialProposer):
            proposer.suffix_cache.start_request(req_id, prompt)
            # Add response tokens
            chunk_size = 50
            for start in range(0, response_len, chunk_size):
                end = min(start + chunk_size, response_len)
                tokens = np.arange(
                    i * 10000 + prompt_len + start,
                    i * 10000 + prompt_len + end,
                    dtype=np.int32
                )
                proposer.suffix_cache.add_active_response(req_id, tokens)
        else:
            proposer.suffix_cache.start_request(req_id, prompt)
            # Add response tokens
            chunk_size = 50
            for start in range(0, response_len, chunk_size):
                end = min(start + chunk_size, response_len)
                tokens = np.arange(
                    i * 10000 + prompt_len + start,
                    i * 10000 + prompt_len + end,
                    dtype=np.int32
                )
                proposer.suffix_cache.add_tokens(req_id, tokens)


def benchmark_proposer(
    proposer,
    batch_size: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> dict:
    """Benchmark a proposer implementation."""
    input_batch = MockInputBatch(
        batch_size=batch_size,
        prompt_len=512,
        num_generated=100,
    )

    def get_sampled_tokens():
        return [[np.random.randint(1, 10000)] for _ in range(batch_size)]

    # Warmup
    for _ in range(num_warmup):
        sampled_token_ids = get_sampled_tokens()
        _ = proposer.propose(input_batch, sampled_token_ids)
        input_batch.add_generated_tokens(1)

    # Reset batch
    input_batch = MockInputBatch(
        batch_size=batch_size,
        prompt_len=512,
        num_generated=100,
    )

    # Benchmark
    latencies = []
    for _ in range(num_iterations):
        sampled_token_ids = get_sampled_tokens()
        start = time.perf_counter()
        _ = proposer.propose(input_batch, sampled_token_ids)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        input_batch.add_generated_tokens(1)

    # Cleanup
    for req_id in list(proposer.suffix_cache.active_requests):
        if req_id.startswith("req_"):
            proposer.suffix_cache.stop_request(req_id)

    latencies = np.array(latencies)
    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p90_ms": np.percentile(latencies, 90),
        "p99_ms": np.percentile(latencies, 99),
    }


def run_comparison(
    batch_sizes: list = None,
    num_trees: int = 100,
    prompt_len: int = 512,
    response_len: int = 500,
    num_threads: int = 8,
    parallel_threshold: int = 8,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Run comparison between sequential and parallel implementations."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 80)
    print("Sequential vs Parallel Suffix Decoding Comparison")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  num_trees: {num_trees}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  response_len: {response_len}")
    print(f"  num_threads (parallel): {num_threads}")
    print(f"  parallel_threshold: {parallel_threshold}")
    print(f"  warmup: {num_warmup}, iterations: {num_iterations}")

    # Create both proposers
    print("\nInitializing Sequential proposer...")
    seq_proposer = SequentialProposer()
    pre_init_cache(seq_proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    print("Initializing Parallel proposer...")
    par_proposer = ParallelProposer(num_threads=num_threads, parallel_threshold=parallel_threshold)
    pre_init_cache(par_proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    # Run benchmarks
    results = []
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}...")

        seq_result = benchmark_proposer(seq_proposer, batch_size, num_warmup, num_iterations)
        par_result = benchmark_proposer(par_proposer, batch_size, num_warmup, num_iterations)

        speedup = seq_result["mean_ms"] / par_result["mean_ms"]

        results.append({
            "batch_size": batch_size,
            "seq_mean_ms": seq_result["mean_ms"],
            "seq_p50_ms": seq_result["p50_ms"],
            "par_mean_ms": par_result["mean_ms"],
            "par_p50_ms": par_result["p50_ms"],
            "speedup": speedup,
        })

    # Print results
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Batch':>8} | {'Sequential (ms)':>15} | {'Parallel (ms)':>14} | {'Speedup':>8}")
    print("-" * 55)

    crossover_batch = None
    for r in results:
        marker = ""
        if r["speedup"] >= 1.0 and crossover_batch is None:
            crossover_batch = r["batch_size"]
            marker = " *"

        print(
            f"{r['batch_size']:>8} | "
            f"{r['seq_mean_ms']:>15.3f} | "
            f"{r['par_mean_ms']:>14.3f} | "
            f"{r['speedup']:>7.2f}x{marker}"
        )

    print("-" * 55)
    if crossover_batch:
        print(f"* Crossover point: batch_size >= {crossover_batch} (parallel becomes faster)")
    else:
        print("* Sequential is faster for all tested batch sizes")

    # Per-request latency comparison
    print("\n" + "=" * 80)
    print("Per-Request Latency")
    print("=" * 80)
    print(f"{'Batch':>8} | {'Seq/req (ms)':>12} | {'Par/req (ms)':>12}")
    print("-" * 40)

    for r in results:
        seq_per_req = r["seq_mean_ms"] / r["batch_size"]
        par_per_req = r["par_mean_ms"] / r["batch_size"]
        print(
            f"{r['batch_size']:>8} | "
            f"{seq_per_req:>12.4f} | "
            f"{par_per_req:>12.4f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sequential vs Parallel Suffix Decoding Proposer"
    )
    parser.add_argument("--num-trees", type=int, default=100,
                        help="Number of trees to pre-initialize (default: 100)")
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Prompt length for pre-initialized trees (default: 512)")
    parser.add_argument("--response-len", type=int, default=500,
                        help="Response length for pre-initialized trees (default: 500)")
    parser.add_argument("--num-threads", type=int, default=8,
                        help="Number of threads for parallel version (default: 8)")
    parser.add_argument("--parallel-threshold", type=int, default=8,
                        help="Parallel threshold (default: 8)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations (default: 10)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of benchmark iterations (default: 100)")

    args = parser.parse_args()

    run_comparison(
        num_trees=args.num_trees,
        prompt_len=args.prompt_len,
        response_len=args.response_len,
        num_threads=args.num_threads,
        parallel_threshold=args.parallel_threshold,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
