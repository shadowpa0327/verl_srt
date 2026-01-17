#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Benchmark script to compare drafting latency between:
1. Parallel suffix decoding cache (snapshot-based, OpenMP parallelization)
2. Shared memory suffix cache (SpecRL's SuffixCache)

This script measures the speculation/drafting latency under various conditions:
- Different cached sequence lengths (how much data in trees)
- Different batch sizes
- Same spec_prefix_len for fair comparison

Usage:
    python benchmark_suffix_drafting.py [--batch-sizes 1,8,32,64,128] \
                                        [--cached-lengths 2048,4096,8192] \
                                        [--iterations 100] \
                                        [--spec-prefix-len 7] \
                                        [--max-spec-tokens 8] \
                                        [--num-trees 64] \
                                        [--output results.json]

Note: Due to a protobuf conflict bug in specrl (both suffix_cache and cache_updater
register the same proto file), the SHM benchmark runs in a subprocess.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

# Parallel cache imports
PARALLEL_AVAILABLE = False
ParallelSuffixDecodingCache = None
try:
    from srt_plugin.suffix_cache.parallel_cache import ParallelSuffixDecodingCache
    PARALLEL_AVAILABLE = True
except ImportError:
    print("Warning: ParallelSuffixDecodingCache not available")

# SHM cache availability check (don't import both modules to avoid protobuf crash)
SHM_AVAILABLE = False
try:
    # Just check if the module exists without importing cache_updater
    import srt_plugin.shm_cache.suffix_cache
    SHM_AVAILABLE = True
except ImportError:
    print("Warning: SRT SHM SuffixCache not available (shared memory mode)")


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64, 128])
    cached_sequence_lengths: List[int] = field(default_factory=lambda: [2048, 4096, 8192])
    spec_prefix_len: int = 7  # Context length for pattern matching (same for both)
    max_spec_tokens: int = 2  # Maximum draft tokens to return (default 2 for fair comparison)
    num_trees: int = 64  # Number of distinct trees to populate
    iterations: int = 100  # Number of iterations per configuration
    warmup_iterations: int = 10  # Warmup iterations before timing
    seed: int = 42  # Random seed for reproducibility
    vocab_size: int = 32000  # Vocabulary size for random tokens


@dataclass
class BenchmarkResult:
    """Result of a single benchmark configuration."""
    impl_type: str  # "parallel" or "shm"
    batch_size: int
    cached_seq_len: int
    spec_prefix_len: int
    max_spec_tokens: int
    num_trees: int
    iterations: int

    # Timing results (in milliseconds)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    requests_per_sec: float = 0.0

    # Draft statistics
    avg_draft_tokens: float = 0.0
    total_draft_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "impl_type": self.impl_type,
            "batch_size": self.batch_size,
            "cached_seq_len": self.cached_seq_len,
            "spec_prefix_len": self.spec_prefix_len,
            "max_spec_tokens": self.max_spec_tokens,
            "num_trees": self.num_trees,
            "iterations": self.iterations,
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "requests_per_sec": self.requests_per_sec,
            "avg_draft_tokens": self.avg_draft_tokens,
            "total_draft_tokens": self.total_draft_tokens,
        }


class ParallelCacheBenchmark:
    """Benchmark for ParallelSuffixDecodingCache."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.cache: Optional[ParallelSuffixDecodingCache] = None

    def setup(self, cached_seq_len: int, num_trees: int):
        """Initialize cache and populate trees with random sequences."""
        # Create cache with reasonable defaults
        self.cache = ParallelSuffixDecodingCache(
            max_tree_depth=64,
            num_threads=-1,  # Auto-detect
            parallel_threshold=4,
            hash_token_count=128,
        )

        # Create trees and populate with random sequences
        self._tree_indices = []
        self._tree_sequences = []  # Store sequences for pattern matching

        for tree_id in range(num_trees):
            req_id = f"tree_{tree_id}"
            # Generate random prompt (short)
            prompt = self.rng.integers(0, self.config.vocab_size, size=32, dtype=np.int32)

            # Start request (creates tree)
            self.cache.start_request(req_id, prompt)
            tree_idx = self.cache.get_tree_idx_for_request(req_id)
            self._tree_indices.append(tree_idx)

            # Generate random sequence to add to tree
            sequence = self.rng.integers(0, self.config.vocab_size, size=cached_seq_len, dtype=np.int32)
            self.cache.add_tokens(req_id, sequence)

            # Store sequence for later pattern extraction
            self._tree_sequences.append(np.concatenate([prompt, sequence]))

    def teardown(self):
        """Clean up cache."""
        self.cache = None
        self._tree_indices = []
        self._tree_sequences = []

    def run_speculation(self, batch_size: int) -> tuple:
        """
        Run a single speculation batch.

        Returns:
            (latency_seconds, total_draft_tokens)
        """
        # Select random trees for this batch
        selected_indices = self.rng.choice(len(self._tree_indices), size=batch_size, replace=True)

        # Build request IDs and contexts
        req_ids = [f"tree_{idx}" for idx in selected_indices]

        # Build contexts from the last spec_prefix_len tokens of each sequence
        contexts = []
        for idx in selected_indices:
            seq = self._tree_sequences[idx]
            # Random position within the sequence for pattern matching
            max_start = max(0, len(seq) - self.config.spec_prefix_len - 100)
            start_pos = self.rng.integers(0, max(1, max_start))
            context = seq[start_pos:start_pos + self.config.spec_prefix_len]
            contexts.append(context)

        # Time the speculation
        start_time = time.perf_counter()
        drafts = self.cache.batch_speculate(
            req_ids=req_ids,
            contexts=contexts,
            max_spec_tokens=self.config.max_spec_tokens,
            min_token_prob=0.1,
            use_tree_spec=False,
        )
        end_time = time.perf_counter()

        latency = end_time - start_time
        # Truncate to max_spec_tokens for consistent counting
        total_draft_tokens = sum(min(len(d.token_ids), self.config.max_spec_tokens) for d in drafts)

        return latency, total_draft_tokens

    def benchmark(self, batch_size: int, cached_seq_len: int) -> BenchmarkResult:
        """Run full benchmark for a configuration."""
        self.setup(cached_seq_len, self.config.num_trees)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            self.run_speculation(batch_size)

        # Timed runs
        latencies = []
        total_draft_tokens = 0

        for _ in range(self.config.iterations):
            latency, draft_tokens = self.run_speculation(batch_size)
            latencies.append(latency)
            total_draft_tokens += draft_tokens

        latencies_ms = np.array(latencies) * 1000  # Convert to ms

        result = BenchmarkResult(
            impl_type="parallel",
            batch_size=batch_size,
            cached_seq_len=cached_seq_len,
            spec_prefix_len=self.config.spec_prefix_len,
            max_spec_tokens=self.config.max_spec_tokens,
            num_trees=self.config.num_trees,
            iterations=self.config.iterations,
            mean_latency_ms=float(np.mean(latencies_ms)),
            std_latency_ms=float(np.std(latencies_ms)),
            min_latency_ms=float(np.min(latencies_ms)),
            max_latency_ms=float(np.max(latencies_ms)),
            p50_latency_ms=float(np.percentile(latencies_ms, 50)),
            p95_latency_ms=float(np.percentile(latencies_ms, 95)),
            p99_latency_ms=float(np.percentile(latencies_ms, 99)),
            requests_per_sec=batch_size * self.config.iterations / sum(latencies),
            avg_draft_tokens=total_draft_tokens / (self.config.iterations * batch_size),
            total_draft_tokens=total_draft_tokens,
        )

        self.teardown()
        return result


def run_shm_benchmark_multiprocess(
    batch_size: int,
    cached_seq_len: int,
    spec_prefix_len: int,
    max_spec_tokens: int,
    num_trees: int,
    iterations: int,
    warmup_iterations: int,
    seed: int,
    vocab_size: int,
) -> Optional[Dict[str, Any]]:
    """
    Run SHM benchmark using multiple processes to avoid protobuf conflicts.

    The specrl package has a bug where both suffix_cache and cache_updater
    register the same proto file. We work around this by:
    1. Starting the server in a separate process
    2. Populating data via cache_updater in another process
    3. Running the benchmark via suffix_cache in the main subprocess

    Returns the benchmark result as a dict, or None on failure.
    """
    import tempfile
    import os

    # Generate test data and save to temp file
    rng = np.random.default_rng(seed)
    prompts = []
    responses = []
    patterns = []

    for tree_id in range(num_trees):
        prompt = rng.integers(0, vocab_size, size=64, dtype=np.int32).tolist()
        prompts.append(prompt)

        response = rng.integers(0, vocab_size, size=cached_seq_len, dtype=np.int32).tolist()
        responses.append(response)

        # Pattern from middle of response
        max_start = max(0, cached_seq_len - spec_prefix_len - 50)
        start_pos = rng.integers(0, max(1, max_start))
        pattern = response[start_pos:start_pos + spec_prefix_len]
        patterns.append(pattern)

    # Save test data to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "prompts": prompts,
            "responses": responses,
            "patterns": patterns,
        }, f)
        data_file = f.name

    try:
        # Step 1: Start server process
        server_code = '''
import time
from srt_plugin.shm_cache.suffix_cache import RolloutCacheServer

server = RolloutCacheServer('[::]:16379', shared_memory_size_gb=1)
if not server.initialize():
    exit(1)
if not server.start():
    exit(1)
print("SERVER_READY", flush=True)
# Wait for signal to shutdown (read from stdin)
input()
server.shutdown()
'''
        server_proc = subprocess.Popen(
            [sys.executable, "-c", server_code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for server to be ready (may output multiple lines)
        server_ready = False
        for _ in range(10):  # Max 10 lines
            line = server_proc.stdout.readline()
            if "SERVER_READY" in line:
                server_ready = True
                break
            if server_proc.poll() is not None:
                break

        if not server_ready:
            server_proc.terminate()
            print(f"  [SHM] Server failed to start: {server_proc.stderr.read()}")
            return None

        time.sleep(0.3)

        # Step 2: Populate cache via separate process
        populate_code = f'''
import json
from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater

with open("{data_file}") as f:
    data = json.load(f)

updater = SuffixCacheUpdater(['[::1]:16379'])
prompt_lengths = [float(len(p)) for p in data["prompts"]]
response_lengths = [float(len(r)) for r in data["responses"]]
updater.update_response_cache(
    prompts=data["prompts"],
    responses=data["responses"],
    prompt_lengths=prompt_lengths,
    response_lengths=response_lengths,
    responses_per_prompt=1,
)
print("POPULATED")
'''
        populate_result = subprocess.run(
            [sys.executable, "-c", populate_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if populate_result.returncode != 0 or "POPULATED" not in populate_result.stdout:
            server_proc.stdin.write("quit\\n")
            server_proc.stdin.flush()
            server_proc.terminate()
            print(f"  [SHM] Failed to populate cache (rc={populate_result.returncode})")
            print(f"  [SHM] stderr: {populate_result.stderr[:500] if populate_result.stderr else 'empty'}")
            print(f"  [SHM] stdout: {populate_result.stdout[:500] if populate_result.stdout else 'empty'}")
            return None

        time.sleep(0.2)

        # Step 3: Run benchmark via separate process
        benchmark_code = f'''
import json
import time
import numpy as np
from srt_plugin.shm_cache.suffix_cache import SuffixCache

with open("{data_file}") as f:
    data = json.load(f)

prompts = data["prompts"]
patterns = data["patterns"]
num_trees = len(prompts)
batch_size = {batch_size}
iterations = {iterations}
warmup_iterations = {warmup_iterations}

req_ids = [f"shm_tree_{{i}}" for i in range(num_trees)]

cache = SuffixCache()
cache.fetch_responses_by_prompts_batch(req_ids, prompts)

rng = np.random.default_rng({seed})

# Warmup
for _ in range(warmup_iterations):
    indices = rng.choice(num_trees, size=batch_size, replace=True)
    batch_req_ids = [req_ids[i] for i in indices]
    batch_patterns = [patterns[i] for i in indices]
    cache.speculate(batch_req_ids, batch_patterns, min_token_prob=0.1)

# Benchmark
latencies = []
total_draft_tokens = 0
for _ in range(iterations):
    indices = rng.choice(num_trees, size=batch_size, replace=True)
    batch_req_ids = [req_ids[i] for i in indices]
    batch_patterns = [patterns[i] for i in indices]

    start = time.perf_counter()
    drafts = cache.speculate(batch_req_ids, batch_patterns, min_token_prob=0.1)
    end = time.perf_counter()

    latencies.append(end - start)
    # Truncate to max_spec_tokens for fair comparison (SHM API doesn't have this param)
    # Truncate to max_spec_tokens for fair comparison (SHM API doesn't have this param)
    total_draft_tokens += sum(min(len(d) if d else 0, {max_spec_tokens}) for d in drafts)

# Cleanup
for req_id in req_ids:
    try:
        cache.evict_responses(req_id)
    except:
        pass

latencies_ms = np.array(latencies) * 1000
result = {{
    "impl_type": "shm",
    "batch_size": {batch_size},
    "cached_seq_len": {cached_seq_len},
    "spec_prefix_len": {spec_prefix_len},
    "max_spec_tokens": {max_spec_tokens},
    "num_trees": {num_trees},
    "iterations": {iterations},
    "mean_latency_ms": float(np.mean(latencies_ms)),
    "std_latency_ms": float(np.std(latencies_ms)),
    "min_latency_ms": float(np.min(latencies_ms)),
    "max_latency_ms": float(np.max(latencies_ms)),
    "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
    "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
    "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
    "requests_per_sec": {batch_size} * {iterations} / sum(latencies),
    "avg_draft_tokens": total_draft_tokens / ({iterations} * {batch_size}),
    "total_draft_tokens": total_draft_tokens,
}}
print(json.dumps(result))
'''
        bench_result = subprocess.run(
            [sys.executable, "-c", benchmark_code],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Shutdown server
        server_proc.stdin.write("quit\\n")
        server_proc.stdin.flush()
        server_proc.terminate()

        if bench_result.returncode != 0:
            print(f"  [SHM] Benchmark failed (rc={bench_result.returncode})")
            print(f"  [SHM] stderr: {bench_result.stderr[:500] if bench_result.stderr else 'empty'}")
            print(f"  [SHM] stdout: {bench_result.stdout[:500] if bench_result.stdout else 'empty'}")
            return None

        output = bench_result.stdout.strip()
        if not output:
            print(f"  [SHM] No output from benchmark")
            print(f"  [SHM] stderr: {bench_result.stderr[:500] if bench_result.stderr else 'empty'}")
            return None

        # Find the JSON line (starts with {)
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('{'):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

        print(f"  [SHM] No valid JSON found in output")
        print(f"  [SHM] Raw output: {output[:500]}")
        return None

    except subprocess.TimeoutExpired as e:
        print(f"  [SHM] Timeout: {e}")
        try:
            server_proc.terminate()
        except:
            pass
        return None
    except json.JSONDecodeError as e:
        print(f"  [SHM] JSON decode error: {e}")
        try:
            server_proc.terminate()
        except:
            pass
        return None
    except Exception as e:
        import traceback
        print(f"  [SHM] Error: {e}")
        print(f"  [SHM] Traceback: {traceback.format_exc()}")
        try:
            server_proc.terminate()
        except:
            pass
        return None
    finally:
        # Cleanup temp file
        try:
            os.unlink(data_file)
        except:
            pass


class SHMCacheBenchmark:
    """
    Benchmark for SpecRL's SuffixCache (shared memory mode).

    Due to a protobuf conflict bug in specrl (both suffix_cache and cache_updater
    register the same proto file), this runs benchmarks using multiple processes:
    1. Server process (suffix_cache.RolloutCacheServer)
    2. Populate process (cache_updater.SuffixCacheUpdater)
    3. Benchmark process (suffix_cache.SuffixCache)
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def benchmark(self, batch_size: int, cached_seq_len: int) -> Optional[BenchmarkResult]:
        """Run benchmark using multi-process approach."""
        result_dict = run_shm_benchmark_multiprocess(
            batch_size=batch_size,
            cached_seq_len=cached_seq_len,
            spec_prefix_len=self.config.spec_prefix_len,
            max_spec_tokens=self.config.max_spec_tokens,
            num_trees=self.config.num_trees,
            iterations=self.config.iterations,
            warmup_iterations=self.config.warmup_iterations,
            seed=self.config.seed,
            vocab_size=self.config.vocab_size,
        )

        if result_dict is None:
            return None

        return BenchmarkResult(
            impl_type=result_dict["impl_type"],
            batch_size=result_dict["batch_size"],
            cached_seq_len=result_dict["cached_seq_len"],
            spec_prefix_len=result_dict["spec_prefix_len"],
            max_spec_tokens=result_dict["max_spec_tokens"],
            num_trees=result_dict["num_trees"],
            iterations=result_dict["iterations"],
            mean_latency_ms=result_dict["mean_latency_ms"],
            std_latency_ms=result_dict["std_latency_ms"],
            min_latency_ms=result_dict["min_latency_ms"],
            max_latency_ms=result_dict["max_latency_ms"],
            p50_latency_ms=result_dict["p50_latency_ms"],
            p95_latency_ms=result_dict["p95_latency_ms"],
            p99_latency_ms=result_dict["p99_latency_ms"],
            requests_per_sec=result_dict["requests_per_sec"],
            avg_draft_tokens=result_dict["avg_draft_tokens"],
            total_draft_tokens=result_dict["total_draft_tokens"],
        )


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Group by (batch_size, cached_seq_len)
    grouped = {}
    for r in results:
        key = (r.batch_size, r.cached_seq_len)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.impl_type] = r

    # Print header
    print(f"{'Batch':<8} {'CachedLen':<12} {'Impl':<10} {'Mean(ms)':<12} {'Std(ms)':<10} "
          f"{'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10} {'Req/s':<12} {'AvgDraft':<10}")
    print("-" * 100)

    for (batch_size, cached_seq_len) in sorted(grouped.keys()):
        impls = grouped[(batch_size, cached_seq_len)]
        for impl_type in ["parallel", "shm"]:
            if impl_type in impls:
                r = impls[impl_type]
                print(f"{r.batch_size:<8} {r.cached_seq_len:<12} {r.impl_type:<10} "
                      f"{r.mean_latency_ms:<12.3f} {r.std_latency_ms:<10.3f} "
                      f"{r.p50_latency_ms:<10.3f} {r.p95_latency_ms:<10.3f} "
                      f"{r.p99_latency_ms:<10.3f} {r.requests_per_sec:<12.1f} "
                      f"{r.avg_draft_tokens:<10.2f}")
        print("-" * 100)


def run_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    results = []

    print(f"\nBenchmark Configuration:")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Cached sequence lengths: {config.cached_sequence_lengths}")
    print(f"  Spec prefix length: {config.spec_prefix_len}")
    print(f"  Max spec tokens: {config.max_spec_tokens}")
    print(f"  Number of trees: {config.num_trees}")
    print(f"  Iterations: {config.iterations}")
    print(f"  Warmup iterations: {config.warmup_iterations}")
    print()

    # Run parallel cache benchmark
    if PARALLEL_AVAILABLE:
        print("Running ParallelSuffixDecodingCache benchmark...")
        parallel_bench = ParallelCacheBenchmark(config)

        for cached_seq_len in config.cached_sequence_lengths:
            for batch_size in config.batch_sizes:
                print(f"  [Parallel] batch_size={batch_size}, cached_seq_len={cached_seq_len}")
                result = parallel_bench.benchmark(batch_size, cached_seq_len)
                results.append(result)
                print(f"    Mean latency: {result.mean_latency_ms:.3f} ms, "
                      f"Avg draft tokens: {result.avg_draft_tokens:.2f}")
    else:
        print("Skipping ParallelSuffixDecodingCache benchmark (not available)")

    # Run SHM cache benchmark
    if SHM_AVAILABLE:
        print("\nRunning SHM SuffixCache benchmark...")
        shm_bench = SHMCacheBenchmark(config)

        for cached_seq_len in config.cached_sequence_lengths:
            for batch_size in config.batch_sizes:
                print(f"  [SHM] batch_size={batch_size}, cached_seq_len={cached_seq_len}")
                result = shm_bench.benchmark(batch_size, cached_seq_len)
                if result:
                    results.append(result)
                    print(f"    Mean latency: {result.mean_latency_ms:.3f} ms, "
                          f"Avg draft tokens: {result.avg_draft_tokens:.2f}")
    else:
        print("\nSkipping SHM SuffixCache benchmark (not available)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suffix tree drafting latency"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,64,128",
        help="Comma-separated list of batch sizes to test"
    )
    parser.add_argument(
        "--cached-lengths",
        type=str,
        default="2048,4096,8192",
        help="Comma-separated list of cached sequence lengths"
    )
    parser.add_argument(
        "--spec-prefix-len",
        type=int,
        default=7,
        help="Context length for pattern matching (same for both implementations)"
    )
    parser.add_argument(
        "--max-spec-tokens",
        type=int,
        default=8,
        help="Maximum speculative tokens to return"
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        default=64,
        help="Number of distinct trees to populate"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per configuration"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before timing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Parse comma-separated values
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    cached_lengths = [int(x.strip()) for x in args.cached_lengths.split(",")]

    config = BenchmarkConfig(
        batch_sizes=batch_sizes,
        cached_sequence_lengths=cached_lengths,
        spec_prefix_len=args.spec_prefix_len,
        max_spec_tokens=args.max_spec_tokens,
        num_trees=args.num_trees,
        iterations=args.iterations,
        warmup_iterations=args.warmup,
        seed=args.seed,
    )

    results = run_benchmark(config)

    # Print comparison table
    print_comparison_table(results)

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "config": {
                "batch_sizes": config.batch_sizes,
                "cached_sequence_lengths": config.cached_sequence_lengths,
                "spec_prefix_len": config.spec_prefix_len,
                "max_spec_tokens": config.max_spec_tokens,
                "num_trees": config.num_trees,
                "iterations": config.iterations,
                "warmup_iterations": config.warmup_iterations,
                "seed": config.seed,
            },
            "results": [r.to_dict() for r in results],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
