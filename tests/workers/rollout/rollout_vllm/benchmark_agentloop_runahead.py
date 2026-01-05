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
AgentLoopManager Runahead Tradeoff Benchmark

Measures the trade-off between:
1. Primary overhead: Extra time caused by runahead competing for resources
2. Runahead benefit: Tokens generated speculatively (free work)

This benchmark uses the AgentLoopManager with Ray-native busy loop (ray.wait + drip-feed)
and DataProto input/output format.

Usage:
    # Run full experiment matrix
    NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/benchmark_agentloop_runahead.py --rounds 3

    # Run single config
    PRIMARY_SIZE=16 LONG_TAIL_RATIO=0.20 \
        python tests/workers/rollout/rollout_vllm/benchmark_agentloop_runahead.py --single

Environment Variables:
    MODEL_PATH: Model to use (default: Qwen/Qwen3-8B)
    SHORT_MAX_TOKENS: Short request max tokens (default: 2048)
    LONG_MAX_TOKENS: Long request max tokens (default: 16384)
    NUM_GPUS: Number of GPUs / DP size (default: 2)
    PRIMARY_SIZE: Primary batch size (default: 32)
    LONG_TAIL_RATIO: Fraction of long requests (default: 0.20)
    LOAD_THRESHOLD: Runahead admission threshold (default: 16)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.runahead import RunaheadConfig
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


# =============================================================================
# Hard Math Problem for Long-Tail Workload
# =============================================================================

HARD_MATH_PROBLEM = """In triangle ABC, sin(angle A) = 4/5 and angle A < 90 degrees. Let D be a point outside triangle ABC such that angle BAD = angle DAC and angle BDC = 90 degrees. Suppose that AD = 1 and that BD/CD = 3/2. If AB + AC can be expressed in the form (a*sqrt(b))/c where a, b, c are pairwise relatively prime integers, find a + b + c. Show your complete step-by-step solution with all mathematical reasoning."""


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    primary_size: int = 32
    long_tail_ratio: float = 0.20
    load_threshold: int = 16
    max_secondary_concurrent: int = 8
    short_max_tokens: int = 2048
    long_max_tokens: int = 16384
    num_gpus: int = 2
    tp_size: int = 1
    num_workers: int = 2
    model_path: str = "Qwen/Qwen3-8B"

    @property
    def dp_size(self) -> int:
        return self.num_gpus // self.tp_size


# =============================================================================
# Metrics and Results
# =============================================================================


@dataclass
class RunMetrics:
    """Metrics from a single run (baseline or runahead)."""

    primary_time: float = 0.0
    primary_tokens: int = 0
    primary_completed: int = 0
    runahead_tokens_total: int = 0
    runahead_tokens_completed: int = 0
    runahead_tokens_aborted: int = 0
    runahead_completed_count: int = 0
    runahead_aborted_count: int = 0
    runahead_rejected_count: int = 0


@dataclass
class ExperimentResult:
    """Result of a single experiment (baseline + runahead)."""

    experiment_id: str
    config: dict
    baseline: RunMetrics
    runahead: RunMetrics
    primary_overhead_pct: float = 0.0
    effective_throughput_gain_pct: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.baseline.primary_time > 0:
            self.primary_overhead_pct = (
                (self.runahead.primary_time - self.baseline.primary_time)
                / self.baseline.primary_time
                * 100
            )
            # Effective gain: compare total tokens / time
            baseline_throughput = self.baseline.primary_tokens / self.baseline.primary_time
            runahead_total_tokens = self.runahead.primary_tokens + self.runahead.runahead_tokens_total
            runahead_throughput = runahead_total_tokens / self.runahead.primary_time
            self.effective_throughput_gain_pct = (runahead_throughput / baseline_throughput - 1) * 100


# =============================================================================
# Config Composition
# =============================================================================


def compose_config(model_path: str, num_gpus: int, tp_size: int, num_workers: int) -> DictConfig:
    """Compose Hydra config for AgentLoopManager."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/verl/trainer/config")
    if not os.path.exists(config_dir):
        config_dir = os.path.abspath("verl/trainer/config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = num_gpus
    config.trainer.nnodes = 1

    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
    config.actor_rollout_ref.rollout.data_parallel_size = 1
    config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1

    # Token length bounds
    config.actor_rollout_ref.rollout.prompt_length = 512
    config.actor_rollout_ref.rollout.response_length = 16384

    config.actor_rollout_ref.rollout.agent.num_workers = num_workers

    # For polling
    config.actor_rollout_ref.rollout.disable_log_stats = False
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9

    # Disable reward for benchmark
    if hasattr(config, "reward_model"):
        config.reward_model.enable = False
        config.reward_model.use_reward_loop = False
        config.reward_model.enable_resource_pool = False

    return config


# =============================================================================
# DataProto Builders
# =============================================================================


def build_primary_dataproto(
    size: int,
    long_tail_ratio: float,
    short_max_tokens: int,
    long_max_tokens: int,
) -> tuple[DataProto, list[int]]:
    """Build DataProto with non_tensor_batch for primary (AgentLoop format).

    Returns:
        Tuple of (DataProto, list of max_tokens per sample for verification).
    """
    num_long = max(1, int(size * long_tail_ratio))
    num_short = size - num_long

    raw_prompts = []
    max_tokens_list = []

    # Short prompts
    for i in range(num_short):
        raw_prompts.append([{"role": "user", "content": HARD_MATH_PROBLEM}])
        max_tokens_list.append(short_max_tokens)

    # Long prompts at the end
    for i in range(num_long):
        raw_prompts.append([{"role": "user", "content": HARD_MATH_PROBLEM}])
        max_tokens_list.append(long_max_tokens)

    # Shuffle to distribute long tasks randomly
    combined = list(zip(raw_prompts, max_tokens_list))
    #random.shuffle(combined)
    raw_prompts, max_tokens_list = zip(*combined)
    raw_prompts = list(raw_prompts)
    max_tokens_list = list(max_tokens_list)

    dp = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "agent_name": np.array(["single_turn_agent"] * size, dtype=object),
            "data_source": np.array(["benchmark"] * size, dtype=object),
            "reward_model": np.array([{}] * size, dtype=object),
            "max_tokens": np.array(max_tokens_list, dtype=object),  # Per-sample max_tokens
        },
    )

    return dp, max_tokens_list


def build_secondary_dataproto(
    tokenizer,
    size: int,
    long_tail_ratio: float,
    short_max_tokens: int,
    long_max_tokens: int,
) -> tuple[DataProto, list[int]]:
    """Build DataProto with batch (input_ids, attention_mask) for secondary.

    Returns:
        Tuple of (DataProto, list of max_tokens per sample).
    """
    num_long = max(1, int(size * long_tail_ratio))
    num_short = size - num_long

    raw_prompts = []
    max_tokens_list = []

    # Short prompts
    for i in range(num_short):
        raw_prompts.append([{"role": "user", "content": HARD_MATH_PROBLEM}])
        max_tokens_list.append(short_max_tokens)

    # Long prompts
    for i in range(num_long):
        raw_prompts.append([{"role": "user", "content": HARD_MATH_PROBLEM}])
        max_tokens_list.append(long_max_tokens)

    # Shuffle
    combined = list(zip(raw_prompts, max_tokens_list))
    random.shuffle(combined)
    raw_prompts, max_tokens_list = zip(*combined)
    raw_prompts = list(raw_prompts)
    max_tokens_list = list(max_tokens_list)

    # Tokenize
    prompt_ids_list = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        for messages in raw_prompts
    ]
    padded = tokenizer.pad(
        {"input_ids": prompt_ids_list},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    batch = TensorDict(
        {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        },
        batch_size=(size,),
    )
    dp = DataProto(
        batch=batch,
        non_tensor_batch={
            "max_tokens": np.array(max_tokens_list, dtype=object),
        },
    )

    return dp, max_tokens_list


# =============================================================================
# Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """Runs benchmark experiments using AgentLoopManager."""

    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.manager: Optional[AgentLoopManager] = None
        self.tokenizer = None

    def setup(self):
        """Initialize Ray and AgentLoopManager."""
        print("=" * 80)
        print("AGENTLOOP RUNAHEAD TRADEOFF BENCHMARK")
        print("=" * 80)
        print(f"Model: {self.exp_config.model_path}")
        print(f"GPUs: {self.exp_config.num_gpus} | TP: {self.exp_config.tp_size} | DP: {self.exp_config.dp_size}")
        print(f"Workers: {self.exp_config.num_workers}")
        print("=" * 80)

        print("\n[1] Initializing Ray...")
        if ray.is_initialized():
            ray.shutdown()

        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": os.getenv("VLLM_LOGGING_LEVEL", "WARNING"),
                    "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
                }
            },
            ignore_reinit_error=True,
        )

        print("\n[2] Creating AgentLoopManager (this may take a while)...")
        config = compose_config(
            model_path=self.exp_config.model_path,
            num_gpus=self.exp_config.num_gpus,
            tp_size=self.exp_config.tp_size,
            num_workers=self.exp_config.num_workers,
        )
        self.manager = AgentLoopManager(config)

        print("\n[3] Loading tokenizer...")
        self.tokenizer = hf_tokenizer(self.exp_config.model_path, trust_remote_code=True)

        print("\n[4] Setup complete")

    def teardown(self):
        """Shutdown Ray."""
        print("\nShutting down Ray...")
        ray.shutdown()

    def run_baseline(
        self,
        primary_dp: DataProto,
    ) -> RunMetrics:
        """Run baseline (primary only, no runahead)."""
        print("\n   Running BASELINE (no runahead)...")

        t0 = time.perf_counter()
        result = self.manager.generate_sequences(primary_dp)
        dt = time.perf_counter() - t0

        # Count tokens from output
        primary_tokens = 0
        if "responses" in result.batch.keys():
            resp = result.batch["responses"]
            resp_mask = result.batch.get("response_mask")
            if resp_mask is not None:
                for i in range(len(result)):
                    primary_tokens += resp_mask[i].sum().item()
            else:
                primary_tokens = resp.numel()

        return RunMetrics(
            primary_time=dt,
            primary_tokens=int(primary_tokens),
            primary_completed=len(result),
        )

    def run_with_runahead(
        self,
        primary_dp: DataProto,
        secondary_dp: DataProto,
        runahead_cfg: RunaheadConfig,
    ) -> RunMetrics:
        """Run with runahead enabled."""
        print("\n   Running WITH RUNAHEAD...")

        t0 = time.perf_counter()
        result = self.manager.generate_sequences_with_runahead(
            primary_dp, secondary_dp, runahead_cfg
        )
        dt = time.perf_counter() - t0

        # Count primary tokens
        primary_tokens = 0
        primary_out = result.primary_outputs
        if primary_out is not None and "responses" in primary_out.batch.keys():
            resp = primary_out.batch["responses"]
            resp_mask = primary_out.batch.get("response_mask")
            if resp_mask is not None:
                for i in range(len(primary_out)):
                    primary_tokens += resp_mask[i].sum().item()
            else:
                primary_tokens = resp.numel()

        # Count secondary tokens
        runahead_tokens_completed = 0
        runahead_tokens_aborted = 0
        runahead_completed_count = 0
        runahead_aborted_count = 0
        runahead_rejected_count = 0

        for sec_out in result.secondary_outputs:
            if sec_out.status == "completed":
                runahead_completed_count += 1
                runahead_tokens_completed += sec_out.tokens_generated
            elif sec_out.status == "aborted":
                runahead_aborted_count += 1
                runahead_tokens_aborted += sec_out.tokens_generated
            elif sec_out.status == "rejected":
                runahead_rejected_count += 1

        # Also use metrics from RunaheadResult
        metrics = result.metrics
        print(f"      RunaheadMetrics: started={metrics.secondary_started}, "
              f"completed={metrics.secondary_completed}, aborted={metrics.secondary_aborted}, "
              f"rejected={metrics.secondary_rejected}")

        return RunMetrics(
            primary_time=dt,
            primary_tokens=int(primary_tokens),
            primary_completed=len(primary_out) if primary_out else 0,
            runahead_tokens_total=runahead_tokens_completed + runahead_tokens_aborted,
            runahead_tokens_completed=runahead_tokens_completed,
            runahead_tokens_aborted=runahead_tokens_aborted,
            runahead_completed_count=runahead_completed_count,
            runahead_aborted_count=runahead_aborted_count,
            runahead_rejected_count=runahead_rejected_count,
        )

    def run_single_experiment(
        self,
        primary_size: int,
        long_tail_ratio: float,
    ) -> ExperimentResult:
        """Run a single experiment (baseline + runahead)."""
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT: PRIMARY_SIZE={primary_size}, LONG_TAIL_RATIO={long_tail_ratio:.0%}")
        print(f"{'=' * 80}")

        runahead_cfg = RunaheadConfig(
            enabled=True,
            load_threshold=self.exp_config.load_threshold,
            poll_interval_s=5,
            max_retries=999999,  # Allow retries for rejected requests
            max_secondary_concurrent=self.exp_config.max_secondary_concurrent,
        )

        # Build workloads
        primary_dp, primary_max_tokens = build_primary_dataproto(
            size=primary_size,
            long_tail_ratio=long_tail_ratio,
            short_max_tokens=self.exp_config.short_max_tokens,
            long_max_tokens=self.exp_config.long_max_tokens,
        )
        secondary_dp, secondary_max_tokens = build_secondary_dataproto(
            tokenizer=self.tokenizer,
            size=primary_size,  # Same size as primary
            long_tail_ratio=long_tail_ratio,
            short_max_tokens=self.exp_config.short_max_tokens,
            long_max_tokens=self.exp_config.long_max_tokens,
        )

        num_long = sum(1 for mt in primary_max_tokens if mt == self.exp_config.long_max_tokens)
        num_short = primary_size - num_long
        print(f"   Workload: {num_short} short ({self.exp_config.short_max_tokens} tok), "
              f"{num_long} long ({self.exp_config.long_max_tokens} tok)")

        # Run baseline
        baseline_metrics = self.run_baseline(primary_dp)
        print(f"   Baseline: {baseline_metrics.primary_time:.2f}s, "
              f"{baseline_metrics.primary_tokens} tokens")

        # Run runahead (rebuild primary_dp since generate_sequences may modify it)
        primary_dp, _ = build_primary_dataproto(
            size=primary_size,
            long_tail_ratio=long_tail_ratio,
            short_max_tokens=self.exp_config.short_max_tokens,
            long_max_tokens=self.exp_config.long_max_tokens,
        )
        runahead_metrics = self.run_with_runahead(primary_dp, secondary_dp, runahead_cfg)
        print(f"   Runahead: {runahead_metrics.primary_time:.2f}s, "
              f"primary={runahead_metrics.primary_tokens} tokens, "
              f"runahead={runahead_metrics.runahead_tokens_total} tokens")

        # Calculate overhead
        result = ExperimentResult(
            experiment_id=f"exp_{primary_size}_{int(long_tail_ratio * 100)}pct_{uuid4().hex[:8]}",
            config={
                "primary_size": primary_size,
                "long_tail_ratio": long_tail_ratio,
                "load_threshold": self.exp_config.load_threshold,
                "short_max_tokens": self.exp_config.short_max_tokens,
                "long_max_tokens": self.exp_config.long_max_tokens,
                "num_gpus": self.exp_config.num_gpus,
            },
            baseline=baseline_metrics,
            runahead=runahead_metrics,
        )

        print(f"   Overhead: {result.primary_overhead_pct:+.2f}%")
        print(f"   Effective throughput gain: {result.effective_throughput_gain_pct:+.2f}%")
        print(f"   Runahead completed: {runahead_metrics.runahead_completed_count}, "
              f"aborted: {runahead_metrics.runahead_aborted_count}, "
              f"rejected: {runahead_metrics.runahead_rejected_count}")

        return result


# =============================================================================
# Experiment Matrix
# =============================================================================


def run_experiment_matrix(runner: BenchmarkRunner, num_rounds: int = 1) -> list:
    """Run full experiment matrix with multiple rounds."""
    primary_sizes = [16, 32, 64]
    long_tail_ratios = [0.20, 0.40, 0.60]

    results = []
    total = len(primary_sizes) * len(long_tail_ratios) * num_rounds
    current = 0

    for round_idx in range(num_rounds):
        for primary_size in primary_sizes:
            for long_tail_ratio in long_tail_ratios:
                current += 1
                print(f"\n\n{'#' * 80}")
                print(f"# EXPERIMENT {current}/{total} (Round {round_idx + 1}/{num_rounds})")
                print(f"{'#' * 80}")

                result = runner.run_single_experiment(primary_size, long_tail_ratio)
                result.config["round"] = round_idx + 1
                results.append(result)

    return results


# =============================================================================
# Output Functions
# =============================================================================


def print_summary_table(results: list):
    """Print summary table of results."""
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    print(f"\n{'PRIMARY':<8} | {'RATIO':<6} | {'BASELINE(s)':<11} | "
          f"{'RUNAHEAD(s)':<11} | {'OVERHEAD%':<10} | {'RA_TOKENS':<10} | "
          f"{'RA_COMPLETE':<11} | {'GAIN%':<8}")
    print("-" * 120)

    for r in results:
        print(f"{r.config['primary_size']:<8} | "
              f"{r.config['long_tail_ratio'] * 100:5.0f}% | "
              f"{r.baseline.primary_time:<11.2f} | "
              f"{r.runahead.primary_time:<11.2f} | "
              f"{r.primary_overhead_pct:<+10.2f} | "
              f"{r.runahead.runahead_tokens_total:<10} | "
              f"{r.runahead.runahead_completed_count:<11} | "
              f"{r.effective_throughput_gain_pct:<+8.1f}")

    print("=" * 120)


def print_averaged_summary(results: list, num_rounds: int):
    """Print averaged summary across rounds."""
    if num_rounds <= 1:
        return

    print("\n" + "=" * 140)
    print(f"AVERAGED SUMMARY ({num_rounds} rounds)")
    print("=" * 140)

    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        key = (r.config["primary_size"], r.config["long_tail_ratio"])
        grouped[key].append(r)

    print(f"\n{'PRIMARY':<8} | {'RATIO':<6} | {'BASELINE(s)':<11} | "
          f"{'RUNAHEAD(s)':<11} | {'OVERHEAD%':<10} | {'RA_TOKENS':<10} | "
          f"{'RA_COMPLETED':<12} | {'RA_ABORTED':<10} | {'GAIN%':<8}")
    print("-" * 140)

    for key in sorted(grouped.keys()):
        runs = grouped[key]
        n = len(runs)

        avg_baseline = sum(r.baseline.primary_time for r in runs) / n
        avg_runahead = sum(r.runahead.primary_time for r in runs) / n
        avg_overhead = sum(r.primary_overhead_pct for r in runs) / n
        avg_ra_total = sum(r.runahead.runahead_tokens_total for r in runs) / n
        avg_ra_completed = sum(r.runahead.runahead_tokens_completed for r in runs) / n
        avg_ra_aborted = sum(r.runahead.runahead_tokens_aborted for r in runs) / n
        avg_gain = sum(r.effective_throughput_gain_pct for r in runs) / n

        print(f"{key[0]:<8} | {key[1] * 100:5.0f}% | {avg_baseline:11.2f} | "
              f"{avg_runahead:11.2f} | {avg_overhead:9.2f}% | {avg_ra_total:10.0f} | "
              f"{avg_ra_completed:12.0f} | {avg_ra_aborted:10.0f} | {avg_gain:+8.1f}")

    print("=" * 140)


def save_results(results: list, output_dir: str = "results"):
    """Save results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/agentloop_runahead_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "benchmark_type": "agentloop_runahead",
        "num_experiments": len(results),
        "results": [],
    }

    for r in results:
        data["results"].append({
            "experiment_id": r.experiment_id,
            "config": r.config,
            "baseline": asdict(r.baseline),
            "runahead": asdict(r.runahead),
            "primary_overhead_pct": r.primary_overhead_pct,
            "effective_throughput_gain_pct": r.effective_throughput_gain_pct,
            "timestamp": r.timestamp,
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="AgentLoopManager Runahead Tradeoff Benchmark")
    parser.add_argument("--single", action="store_true", help="Run single config from env vars")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to run (default: 1)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    args = parser.parse_args()

    # Build config from environment
    exp_config = ExperimentConfig(
        primary_size=int(os.environ.get("PRIMARY_SIZE", "128")),
        long_tail_ratio=float(os.environ.get("LONG_TAIL_RATIO", "0.20")),
        load_threshold=int(os.environ.get("LOAD_THRESHOLD", "16")),
        max_secondary_concurrent=int(os.environ.get("MAX_SECONDARY_CONCURRENT", "64")),
        short_max_tokens=int(os.environ.get("SHORT_MAX_TOKENS", "2048")),
        long_max_tokens=int(os.environ.get("LONG_MAX_TOKENS", "16384")),
        num_gpus=int(os.environ.get("NUM_GPUS", "2")),
        tp_size=int(os.environ.get("TP_SIZE", "1")),
        num_workers=int(os.environ.get("NUM_WORKERS", "2")),
        model_path=os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"),
    )

    runner = BenchmarkRunner(exp_config)

    try:
        runner.setup()

        if args.single:
            result = runner.run_single_experiment(
                primary_size=exp_config.primary_size,
                long_tail_ratio=exp_config.long_tail_ratio,
            )
            results = [result]
        else:
            results = run_experiment_matrix(runner, num_rounds=args.rounds)

        print_summary_table(results)
        print_averaged_summary(results, args.rounds)
        save_results(results, args.output_dir)

    finally:
        runner.teardown()


if __name__ == "__main__":
    main()
