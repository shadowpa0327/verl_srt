#!/usr/bin/env python3
"""
Detailed profiling script for ParallelSuffixDecodingProposer.propose().

This script provides fine-grained latency breakdown of the propose() method,
measuring individual operations:
- Request start time
- batch_add_tokens latency
- batch_speculate latency
- Request cleanup time

It also generates Chrome traces via torch.profiler for visualization.

Usage:
    # Basic profiling with latency breakdown
    python profile_suffix_decoding_proposer.py

    # With Chrome trace export
    python profile_suffix_decoding_proposer.py --chrome-trace

    # Detailed profiling with all metrics
    python profile_suffix_decoding_proposer.py --detailed --chrome-trace --batch-size 64

    # Run batch size comparison
    python profile_suffix_decoding_proposer.py --compare
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

# =============================================================================
# Import Note: Why we import from arctic_inference instead of vLLM
# =============================================================================
# The vLLM proposer (ParallelSuffixDecodingProposer) is located at:
#   - vllm/v1/spec_decode/suffix_decoding_parallel.py
#
# However, importing from vLLM triggers heavy dependencies (CUDA, torch.distributed,
# model configs) that cause circular imports when running standalone benchmarks.
# Even vLLM itself uses lazy imports for this reason (see suffix_decoding.py:23).
#
# We import directly from arctic_inference, which provides the underlying cache:
#   - ParallelSuffixDecodingCache: arctic_inference/suffix_decoding/parallel_cache.py
# =============================================================================
try:
    from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache
except ImportError:
    print("ERROR: Could not import ParallelSuffixDecodingCache from arctic_inference.")
    print("Make sure ArcticInference is installed: pip install arctic-inference")
    sys.exit(1)


@dataclass
class LatencyBreakdown:
    """Stores latency breakdown for a single propose() call."""
    total_ms: float = 0.0
    request_setup_ms: float = 0.0
    batch_add_tokens_ms: float = 0.0
    batch_speculate_ms: float = 0.0
    result_building_ms: float = 0.0
    cleanup_ms: float = 0.0
    num_new_requests: int = 0
    num_active_requests: int = 0
    avg_draft_length: float = 0.0


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""
    samples: list = field(default_factory=list)

    def add(self, breakdown: LatencyBreakdown):
        self.samples.append(breakdown)

    def summary(self) -> dict:
        if not self.samples:
            return {}

        def _stats(values):
            arr = np.array(values)
            return {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "min": np.min(arr),
                "max": np.max(arr),
                "p50": np.percentile(arr, 50),
                "p90": np.percentile(arr, 90),
                "p99": np.percentile(arr, 99),
            }

        return {
            "total": _stats([s.total_ms for s in self.samples]),
            "request_setup": _stats([s.request_setup_ms for s in self.samples]),
            "batch_add_tokens": _stats([s.batch_add_tokens_ms for s in self.samples]),
            "batch_speculate": _stats([s.batch_speculate_ms for s in self.samples]),
            "result_building": _stats([s.result_building_ms for s in self.samples]),
            "cleanup": _stats([s.cleanup_ms for s in self.samples]),
            "num_samples": len(self.samples),
            "avg_draft_length": np.mean([s.avg_draft_length for s in self.samples]),
        }


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    result = {"elapsed_ms": 0.0}
    yield result
    result["elapsed_ms"] = (time.perf_counter() - start) * 1000


class InstrumentedProposer:
    """
    Standalone proposer that mimics ParallelSuffixDecodingProposer behavior
    using ParallelSuffixDecodingCache directly for detailed latency measurement.
    """

    def __init__(
        self,
        num_threads: int = 8,
        parallel_threshold: int = 8,
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

        # Initialize the suffix cache directly
        self.suffix_cache = ParallelSuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            num_threads=num_threads,
            parallel_threshold=parallel_threshold,
        )

        print(f"Initialized InstrumentedProposer with num_threads={num_threads}, "
              f"parallel_threshold={parallel_threshold}")

    def instrumented_propose(
        self,
        input_batch: "MockInputBatch",
        sampled_token_ids: list,
    ) -> tuple:
        """
        Instrumented version of propose() that returns latency breakdown.

        Returns:
            tuple: (draft_token_ids, latency_breakdown)
        """
        breakdown = LatencyBreakdown()

        with timer() as total_timer:
            # Phase 1: Collect data and setup new requests
            with timer() as setup_timer:
                req_ids_to_add_tokens = []
                tokens_to_add = []
                req_ids_to_speculate = []
                contexts_to_speculate = []
                max_spec_tokens_list = []
                input_indices_with_drafts = []
                new_requests = 0

                for i, sampled_ids in enumerate(sampled_token_ids):
                    if not sampled_ids:
                        continue

                    req_id = input_batch.req_ids[i]
                    if req_id in input_batch.spec_decode_unsupported_reqs:
                        continue

                    num_tokens = input_batch.num_tokens_no_spec[i]
                    if num_tokens >= self.max_model_len:
                        continue

                    index = input_batch.req_id_to_index[req_id]

                    # Start new requests if needed
                    if req_id not in self.suffix_cache.active_requests:
                        num_prompt_tokens = input_batch.num_prompt_tokens[index]
                        prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                        pre_computed_hash = input_batch.prompt_hashes.get(req_id)
                        self.suffix_cache.start_request(
                            req_id, prompt_token_ids, pre_computed_hash=pre_computed_hash
                        )
                        new_requests += 1

                    req_ids_to_add_tokens.append(req_id)
                    tokens_to_add.append(sampled_ids)

                    start = max(0, num_tokens - self.max_tree_depth)
                    pattern = input_batch.token_ids_cpu[i, start:num_tokens]

                    req_ids_to_speculate.append(req_id)
                    contexts_to_speculate.append(pattern)
                    max_spec_tokens_list.append(
                        min(self.num_speculative_tokens, self.max_model_len - num_tokens - 1)
                    )
                    input_indices_with_drafts.append(i)

            breakdown.request_setup_ms = setup_timer["elapsed_ms"]
            breakdown.num_new_requests = new_requests
            breakdown.num_active_requests = len(req_ids_to_speculate)

            # Phase 2: Batch add tokens
            with timer() as add_timer:
                if req_ids_to_add_tokens:
                    self.suffix_cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)
            breakdown.batch_add_tokens_ms = add_timer["elapsed_ms"]

            # Phase 3: Batch speculate
            with timer() as spec_timer:
                drafts = []
                if req_ids_to_speculate:
                    min_max_spec_tokens = min(max_spec_tokens_list) if max_spec_tokens_list else self.num_speculative_tokens
                    drafts = self.suffix_cache.batch_speculate(
                        req_ids=req_ids_to_speculate,
                        contexts=contexts_to_speculate,
                        max_spec_tokens=min_max_spec_tokens,
                        max_spec_factor=self.max_spec_factor,
                        min_token_prob=self.min_token_prob,
                    )
            breakdown.batch_speculate_ms = spec_timer["elapsed_ms"]

            # Phase 4: Build results
            with timer() as result_timer:
                draft_token_ids = []
                draft_idx = 0
                draft_lengths = []

                for i in range(len(sampled_token_ids)):
                    if i in input_indices_with_drafts:
                        tokens = drafts[draft_idx].token_ids
                        draft_token_ids.append(tokens)
                        draft_lengths.append(len(tokens))
                        draft_idx += 1
                    else:
                        draft_token_ids.append([])
            breakdown.result_building_ms = result_timer["elapsed_ms"]
            breakdown.avg_draft_length = np.mean(draft_lengths) if draft_lengths else 0.0

            # Phase 5: Cleanup inactive requests
            with timer() as cleanup_timer:
                active_req_ids = set(self.suffix_cache.active_requests)
                input_req_ids = set(input_batch.req_id_to_index.keys())
                for req_id in (active_req_ids - input_req_ids):
                    self.suffix_cache.stop_request(req_id)
            breakdown.cleanup_ms = cleanup_timer["elapsed_ms"]

        breakdown.total_ms = total_timer["elapsed_ms"]

        return draft_token_ids, breakdown

    def get_stats(self) -> dict:
        return self.suffix_cache.get_stats()


class MockInputBatch:
    """Mock InputBatch for profiling."""

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


def pre_init_cache(proposer: InstrumentedProposer, num_trees: int, prompt_len: int = 512, response_len: int = 500):
    """Pre-initialize suffix cache with trees."""
    print(f"Pre-initializing {num_trees} suffix trees...")

    for i in range(num_trees):
        req_id = f"preinit_req_{i}"
        prompt = np.arange(i * 10000, i * 10000 + prompt_len, dtype=np.int32)
        proposer.suffix_cache.start_request(req_id, prompt)

        chunk_size = 50
        for start in range(0, response_len, chunk_size):
            end = min(start + chunk_size, response_len)
            tokens = np.arange(
                i * 10000 + prompt_len + start,
                i * 10000 + prompt_len + end,
                dtype=np.int32
            )
            proposer.suffix_cache.add_tokens(req_id, tokens)

    print(f"Pre-initialized {num_trees} trees")
    stats = proposer.get_stats()
    print(f"Cache stats: {stats}")


def run_instrumented_benchmark(
    proposer: InstrumentedProposer,
    input_batch: MockInputBatch,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> LatencyStats:
    """Run benchmark with detailed latency instrumentation."""
    print(f"\nRunning instrumented benchmark...")
    print(f"  Batch size: {input_batch.batch_size}")
    print(f"  Warmup: {num_warmup}, Iterations: {num_iterations}")

    def get_sampled_tokens():
        return [[np.random.randint(1, 10000)] for _ in range(input_batch.batch_size)]

    # Warmup
    for _ in range(num_warmup):
        sampled_token_ids = get_sampled_tokens()
        _, _ = proposer.instrumented_propose(input_batch, sampled_token_ids)
        input_batch.add_generated_tokens(1)

    # Reset batch
    input_batch = MockInputBatch(
        batch_size=input_batch.batch_size,
        max_model_len=input_batch.max_model_len,
        prompt_len=input_batch.prompt_len,
        num_generated=input_batch.num_generated,
    )

    # Benchmark
    stats = LatencyStats()
    for _ in range(num_iterations):
        sampled_token_ids = get_sampled_tokens()
        _, breakdown = proposer.instrumented_propose(input_batch, sampled_token_ids)
        stats.add(breakdown)
        input_batch.add_generated_tokens(1)

    return stats


def run_chrome_trace_profiling(
    proposer: InstrumentedProposer,
    input_batch: MockInputBatch,
    num_warmup: int = 5,
    num_iterations: int = 20,
    output_dir: str = "./profiler_output",
) -> str:
    """Run profiling with torch.profiler and export Chrome trace."""
    print(f"\nGenerating Chrome trace...")
    os.makedirs(output_dir, exist_ok=True)

    def get_sampled_tokens():
        return [[np.random.randint(1, 10000)] for _ in range(input_batch.batch_size)]

    # Warmup outside profiler
    for _ in range(num_warmup):
        sampled_token_ids = get_sampled_tokens()
        _, _ = proposer.instrumented_propose(input_batch, sampled_token_ids)
        input_batch.add_generated_tokens(1)

    # Reset batch
    input_batch = MockInputBatch(
        batch_size=input_batch.batch_size,
        max_model_len=input_batch.max_model_len,
        prompt_len=input_batch.prompt_len,
        num_generated=input_batch.num_generated,
    )

    # Profile - don't use on_trace_ready handler so we can export manually
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(num_iterations):
            sampled_token_ids = get_sampled_tokens()

            # Use markers for different phases
            with torch.profiler.record_function("propose_total"):
                _, _ = proposer.instrumented_propose(input_batch, sampled_token_ids)

            input_batch.add_generated_tokens(1)

    # Export Chrome trace
    chrome_trace_path = os.path.join(output_dir, "chrome_trace.json")
    prof.export_chrome_trace(chrome_trace_path)

    print(f"  Chrome trace exported to: {chrome_trace_path}")
    print("  Open in Chrome: chrome://tracing/")

    # Print summary
    print("\n" + "=" * 80)
    print("Profiler Summary")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    return chrome_trace_path


def print_latency_breakdown(stats: LatencyStats, title: str = "Latency Breakdown"):
    """Print detailed latency breakdown."""
    summary = stats.summary()
    if not summary:
        print("No data to summarize")
        return

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # Header
    print(f"{'Phase':<25} | {'Mean (ms)':>10} | {'Std':>8} | {'P50':>8} | {'P90':>8} | {'P99':>8} | {'%':>6}")
    print("-" * 90)

    total_mean = summary["total"]["mean"]

    for phase in ["request_setup", "batch_add_tokens", "batch_speculate", "result_building", "cleanup"]:
        s = summary[phase]
        pct = (s["mean"] / total_mean * 100) if total_mean > 0 else 0
        print(
            f"{phase:<25} | "
            f"{s['mean']:>10.3f} | "
            f"{s['std']:>8.3f} | "
            f"{s['p50']:>8.3f} | "
            f"{s['p90']:>8.3f} | "
            f"{s['p99']:>8.3f} | "
            f"{pct:>5.1f}%"
        )

    print("-" * 90)
    s = summary["total"]
    print(
        f"{'TOTAL':<25} | "
        f"{s['mean']:>10.3f} | "
        f"{s['std']:>8.3f} | "
        f"{s['p50']:>8.3f} | "
        f"{s['p90']:>8.3f} | "
        f"{s['p99']:>8.3f} | "
        f"{'100.0%':>6}"
    )

    print(f"\nSamples: {summary['num_samples']}, Avg draft length: {summary['avg_draft_length']:.2f}")


def run_batch_size_comparison(
    num_threads: int = 8,
    parallel_threshold: int = 8,
    num_trees: int = 100,
    prompt_len: int = 512,
    response_len: int = 500,
    batch_sizes: list = None,
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Run profiling comparison across batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]

    print("=" * 80)
    print("Batch Size Comparison - Latency Breakdown")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  num_trees: {num_trees}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  response_len: {response_len}")

    proposer = InstrumentedProposer(
        num_threads=num_threads,
        parallel_threshold=parallel_threshold,
    )
    pre_init_cache(proposer, num_trees=num_trees, prompt_len=prompt_len, response_len=response_len)

    all_summaries = {}

    for batch_size in batch_sizes:
        input_batch = MockInputBatch(
            batch_size=batch_size,
            prompt_len=512,
            num_generated=100,
        )

        stats = run_instrumented_benchmark(
            proposer=proposer,
            input_batch=input_batch,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
        )

        all_summaries[batch_size] = stats.summary()
        print_latency_breakdown(stats, f"Batch Size: {batch_size}")

        # Clean up
        for req_id in list(proposer.suffix_cache.active_requests):
            if req_id.startswith("req_"):
                proposer.suffix_cache.stop_request(req_id)

    # Print comparison table
    print("\n" + "=" * 80)
    print("Comparison Summary (Mean latencies in ms)")
    print("=" * 80)
    print(f"{'Batch':>8} | {'Total':>10} | {'Setup':>10} | {'Add Tok':>10} | {'Speculate':>10} | {'Build':>10} | {'Cleanup':>10}")
    print("-" * 80)

    for batch_size in batch_sizes:
        s = all_summaries[batch_size]
        print(
            f"{batch_size:>8} | "
            f"{s['total']['mean']:>10.3f} | "
            f"{s['request_setup']['mean']:>10.3f} | "
            f"{s['batch_add_tokens']['mean']:>10.3f} | "
            f"{s['batch_speculate']['mean']:>10.3f} | "
            f"{s['result_building']['mean']:>10.3f} | "
            f"{s['cleanup']['mean']:>10.3f}"
        )

    # Per-request latency
    print("\n" + "=" * 80)
    print("Per-Request Latency (ms)")
    print("=" * 80)
    print(f"{'Batch':>8} | {'Total':>10} | {'Speculate':>10}")
    print("-" * 35)

    for batch_size in batch_sizes:
        s = all_summaries[batch_size]
        print(
            f"{batch_size:>8} | "
            f"{s['total']['mean']/batch_size:>10.4f} | "
            f"{s['batch_speculate']['mean']/batch_size:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Profile ParallelSuffixDecodingProposer.propose() with detailed latency breakdown"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-trees", type=int, default=100)
    parser.add_argument("--prompt-len", type=int, default=512,
                        help="Prompt length for pre-initialized trees (default: 512)")
    parser.add_argument("--response-len", type=int, default=500,
                        help="Response length for pre-initialized trees (default: 500)")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--parallel-threshold", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--chrome-trace", action="store_true", help="Generate Chrome trace")
    parser.add_argument("--output-dir", type=str, default="./profiler_output")
    parser.add_argument("--detailed", action="store_true", help="Run detailed latency breakdown")
    parser.add_argument("--compare", action="store_true", help="Run batch size comparison")

    args = parser.parse_args()

    if args.compare:
        run_batch_size_comparison(
            num_threads=args.num_threads,
            parallel_threshold=args.parallel_threshold,
            num_trees=args.num_trees,
            prompt_len=args.prompt_len,
            response_len=args.response_len,
            num_warmup=args.warmup,
            num_iterations=args.iterations,
        )
    else:
        print(f"Configuration:")
        print(f"  num_trees: {args.num_trees}")
        print(f"  prompt_len: {args.prompt_len}")
        print(f"  response_len: {args.response_len}")

        proposer = InstrumentedProposer(
            num_threads=args.num_threads,
            parallel_threshold=args.parallel_threshold,
        )
        pre_init_cache(proposer, num_trees=args.num_trees, prompt_len=args.prompt_len, response_len=args.response_len)

        input_batch = MockInputBatch(
            batch_size=args.batch_size,
            prompt_len=512,
            num_generated=100,
        )

        if args.detailed or not args.chrome_trace:
            stats = run_instrumented_benchmark(
                proposer=proposer,
                input_batch=input_batch,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
            )
            print_latency_breakdown(stats)

        if args.chrome_trace:
            # Reset batch
            input_batch = MockInputBatch(
                batch_size=args.batch_size,
                prompt_len=512,
                num_generated=100,
            )
            run_chrome_trace_profiling(
                proposer=proposer,
                input_batch=input_batch,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
