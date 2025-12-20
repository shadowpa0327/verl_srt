#!/usr/bin/env python3
"""
Comparison benchmark: Sequential vs Parallel Suffix Decoding Proposer.

This script compares the latency of three implementations:
- SequentialProposer: SuffixDecodingCache (sequential, dual-tree architecture)
- ParallelProposer (Python loop): ParallelSuffixDecodingCache with batch_speculate
- ParallelProposer (Zero-copy): ParallelSuffixDecodingCache with propose_from_batch

Usage:
    python compare_proposer_implementations.py
    python compare_proposer_implementations.py --num-trees 200 --iterations 100
"""

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

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


@dataclass
class LatencyBreakdown:
    """Stores latency breakdown for a single propose() call."""
    total_ms: float = 0.0
    setup_ms: float = 0.0
    batch_add_tokens_ms: float = 0.0
    batch_speculate_ms: float = 0.0
    result_building_ms: float = 0.0


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    result = {"elapsed_ms": 0.0}
    yield result
    result["elapsed_ms"] = (time.perf_counter() - start) * 1000


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
    ) -> tuple[list, LatencyBreakdown]:
        """Sequential propose: process each request one by one."""
        breakdown = LatencyBreakdown()
        draft_token_ids = []

        with timer() as total_timer:
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

        breakdown.total_ms = total_timer["elapsed_ms"]

        # Cleanup inactive requests
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())
        for req_id in (active_req_ids - input_req_ids):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids, breakdown

    def get_stats(self) -> dict:
        return {
            "type": "sequential",
            "num_active_requests": len(list(self.suffix_cache.active_requests)),
            "max_tree_depth": self.max_tree_depth,
        }


class ParallelProposerPythonLoop:
    """
    Proposer using ParallelSuffixDecodingCache with Python loop + batch_speculate.
    Uses forest architecture with OpenMP parallelization, but Python loop for setup.
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
    ) -> tuple[list, LatencyBreakdown]:
        """Parallel propose with Python loop setup: batch all operations."""
        breakdown = LatencyBreakdown()

        with timer() as total_timer:
            # Phase 1: Setup (THE BOTTLENECK - Python loop)
            with timer() as setup_timer:
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

                    # COPY 1: numpy slice for context
                    start = max(0, num_tokens - self.max_tree_depth)
                    context = input_batch.token_ids_cpu[i, start:num_tokens].copy()

                    # COPY 2: list append
                    req_ids_to_speculate.append(req_id)
                    contexts_to_speculate.append(context)
                    max_spec_tokens_list.append(
                        min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
                    )
                    input_indices_with_drafts.append(i)

            breakdown.setup_ms = setup_timer["elapsed_ms"]

            # Phase 2: Batch add tokens (parallelized in C++)
            with timer() as add_timer:
                if req_ids_to_add_tokens:
                    self.suffix_cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)
            breakdown.batch_add_tokens_ms = add_timer["elapsed_ms"]

            # Phase 3: Batch speculate (parallelized in C++) - COPY 3 in binding
            with timer() as spec_timer:
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
            breakdown.batch_speculate_ms = spec_timer["elapsed_ms"]

            # Phase 4: Build results
            with timer() as result_timer:
                draft_token_ids = []
                draft_idx = 0
                for i in range(len(sampled_token_ids)):
                    if i in input_indices_with_drafts:
                        draft_token_ids.append(drafts[draft_idx].token_ids)
                        draft_idx += 1
                    else:
                        draft_token_ids.append([])
            breakdown.result_building_ms = result_timer["elapsed_ms"]

        breakdown.total_ms = total_timer["elapsed_ms"]

        # Cleanup inactive requests
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())
        for req_id in (active_req_ids - input_req_ids):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids, breakdown

    def get_stats(self) -> dict:
        return self.suffix_cache.get_stats()


class ParallelProposerZeroCopy:
    """
    Proposer using ParallelSuffixDecodingCache with zero-copy propose_from_batch.
    Eliminates Python loop overhead by passing arrays directly to C++.
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

        # Tree index mapping for zero-copy API
        self._tree_indices = np.zeros(0, dtype=np.int32)

    def propose(
        self,
        input_batch: MockInputBatch,
        sampled_token_ids: list,
    ) -> tuple[list, LatencyBreakdown]:
        """Zero-copy propose: use propose_from_batch for maximum performance."""
        breakdown = LatencyBreakdown()

        with timer() as total_timer:
            # Phase 1: Minimal setup (just prepare indices)
            with timer() as setup_timer:
                req_ids_to_add_tokens = []
                tokens_to_add = []
                batch_indices = []

                # Ensure tree_indices array is large enough
                if len(self._tree_indices) < input_batch.batch_size:
                    self._tree_indices = np.zeros(input_batch.batch_size, dtype=np.int32)

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
                    batch_indices.append(i)

                    # Get tree index for zero-copy API
                    self._tree_indices[i] = self.suffix_cache.get_tree_idx_for_request(req_id)

            breakdown.setup_ms = setup_timer["elapsed_ms"]

            # Phase 2: Batch add tokens (parallelized in C++)
            with timer() as add_timer:
                if req_ids_to_add_tokens:
                    self.suffix_cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)
            breakdown.batch_add_tokens_ms = add_timer["elapsed_ms"]

            # Phase 3: Zero-copy speculation via propose_from_batch
            with timer() as spec_timer:
                drafts = []
                if batch_indices:
                    batch_indices_arr = np.array(batch_indices, dtype=np.int32)
                    drafts = self.suffix_cache.propose_from_batch(
                        token_ids_cpu=input_batch.token_ids_cpu[batch_indices_arr],
                        num_tokens=input_batch.num_tokens_no_spec[batch_indices_arr],
                        tree_indices=self._tree_indices[batch_indices_arr],
                        max_spec_tokens=self.num_speculative_tokens,
                        max_spec_factor=self.max_spec_factor,
                        min_token_prob=self.min_token_prob,
                    )
            breakdown.batch_speculate_ms = spec_timer["elapsed_ms"]

            # Phase 4: Build results
            with timer() as result_timer:
                draft_token_ids = []
                draft_idx = 0
                for i in range(len(sampled_token_ids)):
                    if i in batch_indices:
                        draft_token_ids.append(drafts[draft_idx].token_ids)
                        draft_idx += 1
                    else:
                        draft_token_ids.append([])
            breakdown.result_building_ms = result_timer["elapsed_ms"]

        breakdown.total_ms = total_timer["elapsed_ms"]

        # Cleanup inactive requests
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())
        for req_id in (active_req_ids - input_req_ids):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids, breakdown

    def get_stats(self) -> dict:
        return self.suffix_cache.get_stats()


def pre_init_cache_sequential(proposer: SequentialProposer, num_trees: int, prompt_len: int = 512, response_len: int = 500):
    """Pre-initialize cache with trees for SequentialProposer."""
    for i in range(num_trees):
        req_id = f"preinit_req_{i}"
        prompt = np.arange(i * 10000, i * 10000 + prompt_len, dtype=np.int32)
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


def pre_init_cache_parallel(proposer, num_trees: int, prompt_len: int = 512, response_len: int = 500):
    """Pre-initialize cache with trees for ParallelProposer variants."""
    for i in range(num_trees):
        req_id = f"preinit_req_{i}"
        prompt = np.arange(i * 10000, i * 10000 + prompt_len, dtype=np.int32)
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
    breakdowns = []
    for _ in range(num_iterations):
        sampled_token_ids = get_sampled_tokens()
        _, breakdown = proposer.propose(input_batch, sampled_token_ids)
        latencies.append(breakdown.total_ms)
        breakdowns.append(breakdown)
        input_batch.add_generated_tokens(1)

    # Cleanup
    for req_id in list(proposer.suffix_cache.active_requests):
        if req_id.startswith("req_"):
            proposer.suffix_cache.stop_request(req_id)

    latencies = np.array(latencies)

    # Calculate average breakdown
    avg_breakdown = LatencyBreakdown(
        total_ms=np.mean([b.total_ms for b in breakdowns]),
        setup_ms=np.mean([b.setup_ms for b in breakdowns]),
        batch_add_tokens_ms=np.mean([b.batch_add_tokens_ms for b in breakdowns]),
        batch_speculate_ms=np.mean([b.batch_speculate_ms for b in breakdowns]),
        result_building_ms=np.mean([b.result_building_ms for b in breakdowns]),
    )

    return {
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p90_ms": np.percentile(latencies, 90),
        "p99_ms": np.percentile(latencies, 99),
        "breakdown": avg_breakdown,
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
    """Run comparison between all three implementations."""
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print("=" * 100)
    print("Sequential vs Parallel (Python Loop) vs Parallel (Zero-Copy) Comparison")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  num_trees: {num_trees}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  response_len: {response_len}")
    print(f"  num_threads (parallel): {num_threads}")
    print(f"  parallel_threshold: {parallel_threshold}")
    print(f"  warmup: {num_warmup}, iterations: {num_iterations}")

    # Create all three proposers
    print("\nInitializing Sequential proposer...")
    seq_proposer = SequentialProposer()
    pre_init_cache_sequential(seq_proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    print("Initializing Parallel (Python loop) proposer...")
    par_loop_proposer = ParallelProposerPythonLoop(num_threads=num_threads, parallel_threshold=parallel_threshold)
    pre_init_cache_parallel(par_loop_proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    print("Initializing Parallel (Zero-copy) proposer...")
    par_zerocopy_proposer = ParallelProposerZeroCopy(num_threads=num_threads, parallel_threshold=parallel_threshold)
    pre_init_cache_parallel(par_zerocopy_proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    # Run benchmarks
    results = []
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}...")

        seq_result = benchmark_proposer(seq_proposer, batch_size, num_warmup, num_iterations)
        par_loop_result = benchmark_proposer(par_loop_proposer, batch_size, num_warmup, num_iterations)
        par_zerocopy_result = benchmark_proposer(par_zerocopy_proposer, batch_size, num_warmup, num_iterations)

        # Calculate speedups
        speedup_par_loop_vs_seq = seq_result["mean_ms"] / par_loop_result["mean_ms"]
        speedup_zerocopy_vs_seq = seq_result["mean_ms"] / par_zerocopy_result["mean_ms"]
        speedup_zerocopy_vs_loop = par_loop_result["mean_ms"] / par_zerocopy_result["mean_ms"]

        results.append({
            "batch_size": batch_size,
            "seq_mean_ms": seq_result["mean_ms"],
            "seq_p50_ms": seq_result["p50_ms"],
            "par_loop_mean_ms": par_loop_result["mean_ms"],
            "par_loop_p50_ms": par_loop_result["p50_ms"],
            "par_zerocopy_mean_ms": par_zerocopy_result["mean_ms"],
            "par_zerocopy_p50_ms": par_zerocopy_result["p50_ms"],
            "speedup_par_loop_vs_seq": speedup_par_loop_vs_seq,
            "speedup_zerocopy_vs_seq": speedup_zerocopy_vs_seq,
            "speedup_zerocopy_vs_loop": speedup_zerocopy_vs_loop,
            "seq_breakdown": seq_result["breakdown"],
            "par_loop_breakdown": par_loop_result["breakdown"],
            "par_zerocopy_breakdown": par_zerocopy_result["breakdown"],
        })

    # Print results summary
    print("\n" + "=" * 100)
    print("Results Summary")
    print("=" * 100)
    print(f"{'Batch':>8} | {'Sequential':>12} | {'Par(Loop)':>12} | {'Par(Zero)':>12} | {'Loop/Seq':>10} | {'Zero/Seq':>10} | {'Zero/Loop':>10}")
    print("-" * 100)

    for r in results:
        print(
            f"{r['batch_size']:>8} | "
            f"{r['seq_mean_ms']:>10.3f}ms | "
            f"{r['par_loop_mean_ms']:>10.3f}ms | "
            f"{r['par_zerocopy_mean_ms']:>10.3f}ms | "
            f"{r['speedup_par_loop_vs_seq']:>9.2f}x | "
            f"{r['speedup_zerocopy_vs_seq']:>9.2f}x | "
            f"{r['speedup_zerocopy_vs_loop']:>9.2f}x"
        )

    # Detailed breakdown for parallel implementations
    print("\n" + "=" * 100)
    print("Phase Breakdown: Parallel (Python Loop) vs Parallel (Zero-Copy)")
    print("=" * 100)
    print(f"{'Batch':>6} | {'Phase':>18} | {'Loop (ms)':>12} | {'ZeroCopy (ms)':>14} | {'Speedup':>10}")
    print("-" * 100)

    for r in results:
        loop_b = r["par_loop_breakdown"]
        zero_b = r["par_zerocopy_breakdown"]

        # Setup
        setup_speedup = loop_b.setup_ms / zero_b.setup_ms if zero_b.setup_ms > 0 else float('inf')
        print(f"{r['batch_size']:>6} | {'Setup':>18} | {loop_b.setup_ms:>10.4f} | {zero_b.setup_ms:>12.4f} | {setup_speedup:>9.1f}x")

        # Add tokens
        add_speedup = loop_b.batch_add_tokens_ms / zero_b.batch_add_tokens_ms if zero_b.batch_add_tokens_ms > 0 else 1.0
        print(f"{'':>6} | {'Add Tokens':>18} | {loop_b.batch_add_tokens_ms:>10.4f} | {zero_b.batch_add_tokens_ms:>12.4f} | {add_speedup:>9.1f}x")

        # Speculate
        spec_speedup = loop_b.batch_speculate_ms / zero_b.batch_speculate_ms if zero_b.batch_speculate_ms > 0 else 1.0
        print(f"{'':>6} | {'Speculate':>18} | {loop_b.batch_speculate_ms:>10.4f} | {zero_b.batch_speculate_ms:>12.4f} | {spec_speedup:>9.1f}x")

        # Total
        total_speedup = loop_b.total_ms / zero_b.total_ms if zero_b.total_ms > 0 else 1.0
        print(f"{'':>6} | {'TOTAL':>18} | {loop_b.total_ms:>10.4f} | {zero_b.total_ms:>12.4f} | {total_speedup:>9.1f}x")
        print("-" * 100)

    # Per-request latency comparison
    print("\n" + "=" * 100)
    print("Per-Request Latency")
    print("=" * 100)
    print(f"{'Batch':>8} | {'Seq/req':>12} | {'Loop/req':>12} | {'Zero/req':>12}")
    print("-" * 60)

    for r in results:
        seq_per_req = r["seq_mean_ms"] / r["batch_size"]
        loop_per_req = r["par_loop_mean_ms"] / r["batch_size"]
        zero_per_req = r["par_zerocopy_mean_ms"] / r["batch_size"]
        print(
            f"{r['batch_size']:>8} | "
            f"{seq_per_req:>10.4f}ms | "
            f"{loop_per_req:>10.4f}ms | "
            f"{zero_per_req:>10.4f}ms"
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
    parser.add_argument("--iterations", type=int, default=5000,
                        help="Number of benchmark iterations (default: 100)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64",
                        help="Comma-separated batch sizes (default: 1,2,4,8,16,32,64)")

    args = parser.parse_args()

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    run_comparison(
        batch_sizes=batch_sizes,
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
