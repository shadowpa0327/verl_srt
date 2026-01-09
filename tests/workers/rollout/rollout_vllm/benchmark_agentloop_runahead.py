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
AgentLoopManager Runahead Benchmark

Runs either baseline (no runahead) or runahead mode to measure performance.
Results can be aggregated later to compute overhead and throughput gains.

Usage:
    # Run baseline mode
    python benchmark_agentloop_runahead.py --mode baseline

    # Run runahead mode
    python benchmark_agentloop_runahead.py --mode runahead

    # Run with custom config and save output
    python benchmark_agentloop_runahead.py --mode runahead \\
        --primary-size 64 --long-tail-ratio 0.30 -o results.json

    # Show all options
    python benchmark_agentloop_runahead.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import ray
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.runahead import RunaheadConfig
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


# =============================================================================
# DAPO Dataset
# =============================================================================

DAPO_DATASET_NAME = "haizhongzheng/DAPO-Math-17K-cleaned"


def load_dapo_dataset(cache_dir: str | None = None) -> datasets.Dataset:
    """Load the DAPO math dataset from HuggingFace."""
    return datasets.load_dataset(DAPO_DATASET_NAME, cache_dir=cache_dir)["train"]


def sample_prompts(
    dataset: datasets.Dataset, n: int, seed: int | None = None
) -> list[str]:
    """Sample n prompts from the dataset."""
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return [dataset[idx]["prompt"] for idx in indices]


def filter_prompts_by_length(
    prompts: list[str],
    tokenizer,
    max_length: int,
    min_required: int,
) -> list[str]:
    """Filter prompts to only include those within max_length tokens.

    Args:
        prompts: List of prompt strings
        tokenizer: Tokenizer to use for length calculation
        max_length: Maximum allowed token length
        min_required: Minimum number of prompts required

    Returns:
        List of prompts that fit within max_length

    Raises:
        ValueError: If fewer than min_required prompts pass the filter
    """
    filtered = []
    for prompt in prompts:
        # Tokenize with chat template to get accurate length
        messages = [{"role": "user", "content": prompt}]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if len(tokens) <= max_length:
            filtered.append(prompt)

    if len(filtered) < min_required:
        raise ValueError(
            f"Only {len(filtered)} prompts fit within {max_length} tokens, "
            f"but {min_required} are required. Consider increasing --max-prompt-length."
        )

    return filtered


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AgentLoop Runahead Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Benchmark mode (required)
    parser.add_argument(
        "--mode",
        choices=["baseline", "runahead"],
        required=True,
        help="Run mode: 'baseline' (no runahead) or 'runahead' (with runahead)",
    )

    # Model settings
    parser.add_argument(
        "--model-path", default="Qwen/Qwen3-8B", help="Model to use"
    )

    # Hardware settings
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of agent workers"
    )

    # Workload settings
    parser.add_argument(
        "--primary-size", type=int, default=512, help="Primary batch size"
    )
    parser.add_argument(
        "--long-tail-ratio", type=float, default=0.1, help="Fraction of long requests"
    )
    parser.add_argument(
        "--short-max-tokens", type=int, default=2048, help="Short request max tokens"
    )
    parser.add_argument(
        "--long-max-tokens", type=int, default=16384, help="Long request max tokens"
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length in tokens (prompts exceeding this are filtered out)",
    )

    # Runahead settings
    parser.add_argument(
        "--load-threshold", type=int, default=16, help="Runahead admission threshold"
    )
    parser.add_argument(
        "--max-secondary-concurrent",
        type=int,
        default=64,
        help="Max concurrent secondary requests",
    )
    parser.add_argument(
        "--admit-loop-poll-s",
        type=float,
        default=0.05,
        help="Admission loop poll interval",
    )

    # Dataset settings
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Cache directory for HuggingFace dataset",
    )

    # Output
    parser.add_argument(
        "--output-file", "-o", default=None, help="JSON output file path"
    )

    # Metrics collection
    parser.add_argument(
        "--collect-metrics",
        action="store_true",
        help="Collect time-series vLLM metrics (kv_cache_usage, requests_running, etc.)",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="metrics.csv",
        help="Output CSV file for collected metrics",
    )

    # Multi-round settings
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of benchmark rounds to run for statistical significance",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=0,
        help="Number of warmup rounds (not included in statistics)",
    )

    # Determinism settings
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling (shuffle + vLLM generation). Use -1 to disable.",
    )

    return parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""

    # Model settings
    model_path: str = "Qwen/Qwen3-8B"

    # Hardware settings
    num_gpus: int = 2
    tp_size: int = 1
    num_workers: int = 2

    # Workload settings
    primary_size: int = 128
    long_tail_ratio: float = 0.20
    short_max_tokens: int = 2048
    long_max_tokens: int = 16384
    max_prompt_length: int = 512

    # Runahead settings
    load_threshold: int = 16
    max_secondary_concurrent: int = 64
    admit_loop_poll_s: float = 0.05

    # Output
    output_file: Optional[str] = None

    # Dataset settings
    dataset_seed: Optional[int] = None
    dataset_cache_dir: Optional[str] = None

    # Multi-round settings
    num_rounds: int = 1
    warmup_rounds: int = 0

    # Determinism settings
    sampling_seed: int = 42  # Seed for reproducible sampling, -1 to disable

    @property
    def dp_size(self) -> int:
        return self.num_gpus // self.tp_size

    @classmethod
    def from_args(cls, args) -> "BenchmarkConfig":
        """Create config from parsed arguments."""
        return cls(
            model_path=args.model_path,
            num_gpus=args.num_gpus,
            tp_size=args.tp_size,
            num_workers=args.num_workers,
            primary_size=args.primary_size,
            long_tail_ratio=args.long_tail_ratio,
            short_max_tokens=args.short_max_tokens,
            long_max_tokens=args.long_max_tokens,
            max_prompt_length=args.max_prompt_length,
            load_threshold=args.load_threshold,
            max_secondary_concurrent=args.max_secondary_concurrent,
            admit_loop_poll_s=args.admit_loop_poll_s,
            output_file=args.output_file,
            dataset_seed=args.dataset_seed,
            dataset_cache_dir=args.dataset_cache_dir,
            num_rounds=args.num_rounds,
            warmup_rounds=args.warmup_rounds,
            sampling_seed=args.sampling_seed,
        )


# =============================================================================
# Results
# =============================================================================


@dataclass
class RunMetrics:
    """Metrics from a single run (baseline or runahead)."""

    time_seconds: float = 0.0
    primary_tokens: int = 0
    primary_completed: int = 0
    runahead_tokens_total: int = 0
    runahead_tokens_completed: int = 0
    runahead_tokens_aborted: int = 0
    runahead_completed_count: int = 0
    runahead_aborted_count: int = 0
    runahead_rejected_count: int = 0


@dataclass
class MetricsStatistics:
    """Statistics computed across multiple rounds."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0

    @classmethod
    def from_values(cls, values: list[float]) -> "MetricsStatistics":
        """Compute statistics from a list of values."""
        if not values:
            return cls()
        arr = np.array(values)
        return cls(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )


@dataclass
class MultiRoundMetrics:
    """Aggregated metrics from multiple rounds."""

    num_rounds: int = 0
    warmup_rounds: int = 0
    time_stats: MetricsStatistics = None
    primary_tokens_stats: MetricsStatistics = None
    throughput_stats: MetricsStatistics = None  # tokens/sec
    per_round_metrics: list = None  # List of RunMetrics dicts

    def __post_init__(self):
        if self.time_stats is None:
            self.time_stats = MetricsStatistics()
        if self.primary_tokens_stats is None:
            self.primary_tokens_stats = MetricsStatistics()
        if self.throughput_stats is None:
            self.throughput_stats = MetricsStatistics()
        if self.per_round_metrics is None:
            self.per_round_metrics = []

    @classmethod
    def from_rounds(cls, metrics_list: list[RunMetrics], warmup_rounds: int = 0) -> "MultiRoundMetrics":
        """Compute aggregated metrics from a list of per-round metrics."""
        if not metrics_list:
            return cls()

        times = [m.time_seconds for m in metrics_list]
        tokens = [m.primary_tokens for m in metrics_list]
        throughputs = [m.primary_tokens / m.time_seconds if m.time_seconds > 0 else 0 for m in metrics_list]

        return cls(
            num_rounds=len(metrics_list),
            warmup_rounds=warmup_rounds,
            time_stats=MetricsStatistics.from_values(times),
            primary_tokens_stats=MetricsStatistics.from_values(tokens),
            throughput_stats=MetricsStatistics.from_values(throughputs),
            per_round_metrics=[asdict(m) for m in metrics_list],
        )


@dataclass
class BenchmarkResult:
    """Result of the benchmark (single mode)."""

    mode: str  # "baseline" or "runahead"
    config: dict
    metrics: RunMetrics  # Last round metrics (for backward compatibility)
    multi_round: MultiRoundMetrics = None  # Multi-round statistics
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.multi_round is None:
            self.multi_round = MultiRoundMetrics()


# =============================================================================
# Hydra Config Composition
# =============================================================================


def compose_hydra_config(config: BenchmarkConfig) -> DictConfig:
    """Compose Hydra config for AgentLoopManager."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/verl/trainer/config")
    if not os.path.exists(config_dir):
        config_dir = os.path.abspath("verl/trainer/config")

    if not os.path.exists(config_dir):
        raise FileNotFoundError(
            f"Config directory not found. Tried: verl/verl/trainer/config, verl/trainer/config"
        )

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        hydra_config = compose(config_name="ppo_trainer")

    # Trainer settings
    hydra_config.trainer.n_gpus_per_node = config.num_gpus
    hydra_config.trainer.nnodes = 1

    hydra_config.actor_rollout_ref.model.path = config.model_path
    hydra_config.actor_rollout_ref.rollout.name = "vllm"
    hydra_config.actor_rollout_ref.rollout.mode = "async"
    hydra_config.actor_rollout_ref.rollout.tensor_model_parallel_size = config.tp_size
    hydra_config.actor_rollout_ref.rollout.data_parallel_size = 1
    hydra_config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
    hydra_config.actor_rollout_ref.rollout.enable_prefix_caching = False    

    # Token length bounds
    hydra_config.actor_rollout_ref.rollout.prompt_length = config.max_prompt_length
    hydra_config.actor_rollout_ref.rollout.response_length = config.long_max_tokens

    # Agent workers
    hydra_config.actor_rollout_ref.rollout.agent.num_workers = config.num_workers

    # vLLM settings
    hydra_config.actor_rollout_ref.rollout.disable_log_stats = False
    hydra_config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9

    # Disable reward model for benchmark
    if hasattr(hydra_config, "reward_model"):
        hydra_config.reward_model.enable = False
        hydra_config.reward_model.use_reward_loop = False
        hydra_config.reward_model.enable_resource_pool = False

    return hydra_config


# =============================================================================
# DataProto Builders
# =============================================================================


def build_primary_dataproto(
    config: BenchmarkConfig,
    prompts: list[str],
    shuffle_seed: Optional[int] = None,
    sampling_seed_base: Optional[int] = None,
) -> DataProto:
    """Build DataProto with non_tensor_batch for primary (AgentLoop format).

    Args:
        config: Benchmark configuration
        prompts: List of prompt strings
        shuffle_seed: Seed for reproducible shuffle order (None = random)
        sampling_seed_base: Base seed for per-sample vLLM sampling (None = random)
    """
    num_long = max(1, int(config.primary_size * config.long_tail_ratio))
    num_short = config.primary_size - num_long

    raw_prompts = []
    max_tokens_list = []

    # Short prompts (cycle through prompts if we need more than available)
    for i in range(num_short):
        prompt = prompts[i % len(prompts)]
        raw_prompts.append([{"role": "user", "content": prompt}])
        max_tokens_list.append(config.short_max_tokens)

    # Long prompts
    for i in range(num_long):
        prompt = prompts[(num_short + i) % len(prompts)]
        raw_prompts.append([{"role": "user", "content": prompt}])
        max_tokens_list.append(config.long_max_tokens)

    # Shuffle to distribute long tasks (seeded for reproducibility)
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
    combined = list(zip(raw_prompts, max_tokens_list))
    random.shuffle(combined)
    raw_prompts, max_tokens_list = zip(*combined)
    raw_prompts = list(raw_prompts)
    max_tokens_list = list(max_tokens_list)

    # Build non_tensor_batch
    non_tensor_batch = {
        "raw_prompt": np.array(raw_prompts, dtype=object),
        "agent_name": np.array(["single_turn_agent"] * config.primary_size, dtype=object),
        "data_source": np.array(["benchmark"] * config.primary_size, dtype=object),
        "reward_model": np.array([{}] * config.primary_size, dtype=object),
        "max_tokens": np.array(max_tokens_list, dtype=object),
    }

    # Add per-sample seeds for deterministic vLLM sampling
    if sampling_seed_base is not None:
        sampling_seeds = [sampling_seed_base + i for i in range(config.primary_size)]
        non_tensor_batch["sampling_seed"] = np.array(sampling_seeds, dtype=object)

    return DataProto(non_tensor_batch=non_tensor_batch)


def build_secondary_dataproto(
    config: BenchmarkConfig,
    prompts: list[str],
    tokenizer,
    shuffle_seed: Optional[int] = None,
    sampling_seed_base: Optional[int] = None,
) -> DataProto:
    """Build DataProto with batch (input_ids, attention_mask) for secondary.

    Args:
        config: Benchmark configuration
        prompts: List of prompt strings
        tokenizer: Tokenizer for encoding prompts
        shuffle_seed: Seed for reproducible shuffle order (None = random)
        sampling_seed_base: Base seed for per-sample vLLM sampling (None = random)
    """
    num_long = max(1, int(config.primary_size * config.long_tail_ratio))
    num_short = config.primary_size - num_long

    raw_prompts = []
    max_tokens_list = []

    # Short prompts (cycle through prompts if we need more than available)
    for i in range(num_short):
        prompt = prompts[i % len(prompts)]
        raw_prompts.append([{"role": "user", "content": prompt}])
        max_tokens_list.append(config.short_max_tokens)

    # Long prompts
    for i in range(num_long):
        prompt = prompts[(num_short + i) % len(prompts)]
        raw_prompts.append([{"role": "user", "content": prompt}])
        max_tokens_list.append(config.long_max_tokens)

    # Shuffle (seeded for reproducibility)
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
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
        batch_size=(config.primary_size,),
    )

    # Build non_tensor_batch
    non_tensor_batch = {
        "max_tokens": np.array(max_tokens_list, dtype=object),
    }

    # Add per-sample seeds for deterministic vLLM sampling
    if sampling_seed_base is not None:
        sampling_seeds = [sampling_seed_base + i for i in range(config.primary_size)]
        non_tensor_batch["sampling_seed"] = np.array(sampling_seeds, dtype=object)

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


# =============================================================================
# Benchmark Runner
# =============================================================================


def run_baseline(manager: AgentLoopManager, primary_dp: DataProto) -> RunMetrics:
    """Run baseline (primary only, no runahead)."""
    print("\n  Running BASELINE (no runahead)...")

    t0 = time.perf_counter()
    result = manager.generate_sequences(primary_dp)
    elapsed = time.perf_counter() - t0

    # Count tokens from output
    primary_tokens = 0
    if "responses" in result.batch.keys():
        resp_mask = result.batch.get("response_mask")
        if resp_mask is not None:
            for i in range(len(result)):
                primary_tokens += resp_mask[i].sum().item()
        else:
            primary_tokens = result.batch["responses"].numel()

    return RunMetrics(
        time_seconds=elapsed,
        primary_tokens=int(primary_tokens),
        primary_completed=len(result),
    )


def run_with_runahead(
    manager: AgentLoopManager,
    primary_dp: DataProto,
    secondary_dp: DataProto,
    runahead_cfg: RunaheadConfig,
) -> RunMetrics:
    """Run with runahead enabled."""
    print("\n  Running WITH RUNAHEAD...")

    t0 = time.perf_counter()
    result = manager.generate_sequences_with_runahead(primary_dp, secondary_dp, runahead_cfg)
    elapsed = time.perf_counter() - t0

    # Count primary tokens
    primary_tokens = 0
    primary_out = result.primary_outputs
    if primary_out is not None and "responses" in primary_out.batch.keys():
        resp_mask = primary_out.batch.get("response_mask")
        if resp_mask is not None:
            for i in range(len(primary_out)):
                primary_tokens += resp_mask[i].sum().item()
        else:
            primary_tokens = primary_out.batch["responses"].numel()

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

    metrics = result.metrics
    print(
        f"    RunaheadMetrics: started={metrics.secondary_started}, "
        f"completed={metrics.secondary_completed}, aborted={metrics.secondary_aborted}, "
        f"rejected={metrics.secondary_rejected}"
    )

    return RunMetrics(
        time_seconds=elapsed,
        primary_tokens=int(primary_tokens),
        primary_completed=len(primary_out) if primary_out else 0,
        runahead_tokens_total=runahead_tokens_completed + runahead_tokens_aborted,
        runahead_tokens_completed=runahead_tokens_completed,
        runahead_tokens_aborted=runahead_tokens_aborted,
        runahead_completed_count=runahead_completed_count,
        runahead_aborted_count=runahead_aborted_count,
        runahead_rejected_count=runahead_rejected_count,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    config = BenchmarkConfig.from_args(args)

    total_rounds = config.warmup_rounds + config.num_rounds

    print("=" * 80)
    print(f"AGENTLOOP RUNAHEAD BENCHMARK - {args.mode.upper()} MODE")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"GPUs: {config.num_gpus} | TP: {config.tp_size} | DP: {config.dp_size}")
    print(f"Workers: {config.num_workers}")
    print(f"Primary size: {config.primary_size} | Long-tail ratio: {config.long_tail_ratio:.0%}")
    print(f"Short max tokens: {config.short_max_tokens} | Long max tokens: {config.long_max_tokens}")
    if args.mode == "runahead":
        print(f"Load threshold: {config.load_threshold} | Max secondary concurrent: {config.max_secondary_concurrent}")
    print(f"Rounds: {config.num_rounds} measurement + {config.warmup_rounds} warmup = {total_rounds} total")
    print("=" * 80)

    # Initialize Ray
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

    try:
        # Create AgentLoopManager
        print("\n[2] Creating AgentLoopManager...")
        hydra_config = compose_hydra_config(config)
        manager = AgentLoopManager(hydra_config)

        # Load tokenizer first (needed for filtering prompts by length)
        print("\n[3] Loading tokenizer...")
        tokenizer = hf_tokenizer(config.model_path, trust_remote_code=True)

        # Load DAPO dataset
        print("\n[4] Loading DAPO dataset...")
        dapo_dataset = load_dapo_dataset(cache_dir=config.dataset_cache_dir)
        print(f"    Dataset loaded: {len(dapo_dataset)} samples available")

        # Sample more prompts than needed (to allow for filtering)
        # We sample 3x the required amount to have enough after filtering
        sample_multiplier = 3
        print(f"\n[5] Sampling and filtering prompts (max {config.max_prompt_length} tokens)...")

        primary_prompts_raw = sample_prompts(
            dapo_dataset, config.primary_size * sample_multiplier, seed=config.dataset_seed
        )
        primary_prompts = filter_prompts_by_length(
            primary_prompts_raw, tokenizer, config.max_prompt_length, config.primary_size
        )[:config.primary_size]  # Take only what we need

        secondary_prompts = None
        if args.mode == "runahead":
            # Use different seed for secondary (seed+1 if seed provided, else None)
            secondary_seed = (
                config.dataset_seed + 1 if config.dataset_seed is not None else None
            )
            secondary_prompts_raw = sample_prompts(
                dapo_dataset, config.primary_size * sample_multiplier, seed=secondary_seed
            )
            secondary_prompts = filter_prompts_by_length(
                secondary_prompts_raw, tokenizer, config.max_prompt_length, config.primary_size
            )[:config.primary_size]
            print(
                f"    Filtered to {len(primary_prompts)} primary + "
                f"{len(secondary_prompts)} secondary prompts"
            )
        else:
            print(f"    Filtered to {len(primary_prompts)} primary prompts")

        # Build workloads
        print("\n[6] Building workloads...")
        num_long = max(1, int(config.primary_size * config.long_tail_ratio))
        num_short = config.primary_size - num_long
        print(f"    {num_short} short ({config.short_max_tokens} tokens), "
              f"{num_long} long ({config.long_max_tokens} tokens)")
        if config.sampling_seed >= 0:
            print(f"    Deterministic mode: sampling_seed={config.sampling_seed}")
        else:
            print("    Random mode: sampling_seed disabled")

        # Prepare runahead config if runahead mode
        runahead_cfg = None
        if args.mode == "runahead":
            runahead_cfg = RunaheadConfig(
                enabled=True,
                load_threshold=config.load_threshold,
                admit_loop_poll_s=config.admit_loop_poll_s,
                max_secondary_concurrent=config.max_secondary_concurrent,
                max_queue_size=config.primary_size,  # Allow queueing all secondary requests
            )

        # Run multiple rounds
        print(f"\n[7] Running {args.mode} ({total_rounds} rounds)...")
        all_metrics: list[RunMetrics] = []
        measurement_metrics: list[RunMetrics] = []

        # Start metrics collection if requested
        if args.collect_metrics:
            print("\n  Starting metrics collection...")
            manager.start_metrics_collection()

        for round_idx in range(total_rounds):
            is_warmup = round_idx < config.warmup_rounds
            round_label = f"Warmup {round_idx + 1}/{config.warmup_rounds}" if is_warmup else f"Round {round_idx - config.warmup_rounds + 1}/{config.num_rounds}"

            print(f"\n  [{round_label}]")

            # Derive deterministic seeds for this round (-1 means disabled)
            shuffle_seed = None
            sampling_seed_base = None
            if config.sampling_seed >= 0:
                shuffle_seed = config.sampling_seed
                sampling_seed_base = config.sampling_seed

            # Rebuild dataproto for each round with deterministic seeds
            primary_dp = build_primary_dataproto(
                config, primary_prompts,
                shuffle_seed=shuffle_seed,
                sampling_seed_base=sampling_seed_base,
            )

            if args.mode == "baseline":
                metrics = run_baseline(manager, primary_dp)
                print(f"    Time: {metrics.time_seconds:.2f}s, "
                      f"{metrics.primary_tokens} tokens")
            else:  # runahead
                secondary_dp = build_secondary_dataproto(
                    config, secondary_prompts, tokenizer,
                    shuffle_seed=shuffle_seed + 5000 if shuffle_seed is not None else None,
                    sampling_seed_base=sampling_seed_base + 5000 if sampling_seed_base is not None else None,
                )
                metrics = run_with_runahead(manager, primary_dp, secondary_dp, runahead_cfg)
                print(f"    Time: {metrics.time_seconds:.2f}s, "
                      f"primary={metrics.primary_tokens} tokens, "
                      f"runahead={metrics.runahead_tokens_total} tokens")

            all_metrics.append(metrics)
            if not is_warmup:
                measurement_metrics.append(metrics)

        # Stop metrics collection and export if requested
        if args.collect_metrics:
            print("\n  Stopping metrics collection...")
            metrics_result = manager.stop_metrics_collection()
            print(f"  Collected {metrics_result.get('num_samples', 0)} samples over {metrics_result.get('duration_s', 0):.2f}s")

            # Export to CSV
            export_result = manager.export_metrics_csv(args.metrics_output)
            if export_result.get("status") == "success":
                print(f"  Exported metrics to {args.metrics_output} ({export_result.get('num_rows', 0)} rows)")
            else:
                print(f"  Warning: Failed to export metrics: {export_result}")

        # Compute multi-round statistics
        multi_round = MultiRoundMetrics.from_rounds(measurement_metrics, warmup_rounds=config.warmup_rounds)

        # Use last measurement round for backward compatibility
        final_metrics = measurement_metrics[-1] if measurement_metrics else all_metrics[-1]

        # Build result
        result = BenchmarkResult(
            mode=args.mode,
            config=asdict(config),
            metrics=final_metrics,
            multi_round=multi_round,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Mode:                 {result.mode}")
        print(f"Rounds:               {config.num_rounds} (+ {config.warmup_rounds} warmup)")

        if config.num_rounds > 1:
            print("\n--- Multi-Round Statistics ---")
            print(f"Time (s):             mean={multi_round.time_stats.mean:.2f}, "
                  f"std={multi_round.time_stats.std:.2f}, "
                  f"min={multi_round.time_stats.min:.2f}, max={multi_round.time_stats.max:.2f}")
            print(f"Throughput (tok/s):   mean={multi_round.throughput_stats.mean:.1f}, "
                  f"std={multi_round.throughput_stats.std:.1f}, "
                  f"min={multi_round.throughput_stats.min:.1f}, max={multi_round.throughput_stats.max:.1f}")
            print(f"Primary tokens:       mean={multi_round.primary_tokens_stats.mean:.0f}, "
                  f"std={multi_round.primary_tokens_stats.std:.0f}")

        print("\n--- Last Round Details ---")
        print(f"Time:                 {final_metrics.time_seconds:.2f}s")
        print(f"Primary tokens:       {final_metrics.primary_tokens}")
        print(f"Primary completed:    {final_metrics.primary_completed}")
        if args.mode == "runahead":
            print(f"Runahead tokens:      {final_metrics.runahead_tokens_total}")
            print(f"  - Completed:        {final_metrics.runahead_completed_count} "
                  f"({final_metrics.runahead_tokens_completed} tokens)")
            print(f"  - Aborted:          {final_metrics.runahead_aborted_count} "
                  f"({final_metrics.runahead_tokens_aborted} tokens)")
            print(f"  - Rejected:         {final_metrics.runahead_rejected_count}")
        print("=" * 80)

        # Save results if output file specified
        if config.output_file:
            output_path = Path(config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "mode": result.mode,
                "timestamp": result.timestamp,
                "config": result.config,
                "metrics": asdict(result.metrics),
                "multi_round": {
                    "num_rounds": multi_round.num_rounds,
                    "warmup_rounds": multi_round.warmup_rounds,
                    "time_stats": asdict(multi_round.time_stats),
                    "primary_tokens_stats": asdict(multi_round.primary_tokens_stats),
                    "throughput_stats": asdict(multi_round.throughput_stats),
                    "per_round_metrics": multi_round.per_round_metrics,
                },
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\nResults saved to: {output_path}")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
