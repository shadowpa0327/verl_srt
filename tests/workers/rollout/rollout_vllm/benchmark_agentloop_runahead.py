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
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of agent workers"
    )

    # Workload settings
    parser.add_argument(
        "--primary-size", type=int, default=128, help="Primary batch size"
    )
    parser.add_argument(
        "--long-tail-ratio", type=float, default=0.20, help="Fraction of long requests"
    )
    parser.add_argument(
        "--short-max-tokens", type=int, default=2048, help="Short request max tokens"
    )
    parser.add_argument(
        "--long-max-tokens", type=int, default=16384, help="Long request max tokens"
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

    # Runahead settings
    load_threshold: int = 16
    max_secondary_concurrent: int = 64
    admit_loop_poll_s: float = 0.05

    # Output
    output_file: Optional[str] = None

    # Dataset settings
    dataset_seed: Optional[int] = None
    dataset_cache_dir: Optional[str] = None

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
            load_threshold=args.load_threshold,
            max_secondary_concurrent=args.max_secondary_concurrent,
            admit_loop_poll_s=args.admit_loop_poll_s,
            output_file=args.output_file,
            dataset_seed=args.dataset_seed,
            dataset_cache_dir=args.dataset_cache_dir,
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
class BenchmarkResult:
    """Result of the benchmark (single mode)."""

    mode: str  # "baseline" or "runahead"
    config: dict
    metrics: RunMetrics
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


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

    # Model and rollout settings
    hydra_config.actor_rollout_ref.model.path = config.model_path
    hydra_config.actor_rollout_ref.rollout.name = "vllm"
    hydra_config.actor_rollout_ref.rollout.mode = "async"
    hydra_config.actor_rollout_ref.rollout.tensor_model_parallel_size = config.tp_size
    hydra_config.actor_rollout_ref.rollout.data_parallel_size = 1
    hydra_config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1

    # Token length bounds
    hydra_config.actor_rollout_ref.rollout.prompt_length = 512
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


def build_primary_dataproto(config: BenchmarkConfig, prompts: list[str]) -> DataProto:
    """Build DataProto with non_tensor_batch for primary (AgentLoop format)."""
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

    # Shuffle to distribute long tasks randomly
    combined = list(zip(raw_prompts, max_tokens_list))
    random.shuffle(combined)
    raw_prompts, max_tokens_list = zip(*combined)
    raw_prompts = list(raw_prompts)
    max_tokens_list = list(max_tokens_list)

    return DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "agent_name": np.array(["single_turn_agent"] * config.primary_size, dtype=object),
            "data_source": np.array(["benchmark"] * config.primary_size, dtype=object),
            "reward_model": np.array([{}] * config.primary_size, dtype=object),
            "max_tokens": np.array(max_tokens_list, dtype=object),
        },
    )


def build_secondary_dataproto(
    config: BenchmarkConfig, prompts: list[str], tokenizer
) -> DataProto:
    """Build DataProto with batch (input_ids, attention_mask) for secondary."""
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
        batch_size=(config.primary_size,),
    )

    return DataProto(
        batch=batch,
        non_tensor_batch={
            "max_tokens": np.array(max_tokens_list, dtype=object),
        },
    )


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

        # Load DAPO dataset
        print("\n[3] Loading DAPO dataset...")
        dapo_dataset = load_dapo_dataset(cache_dir=config.dataset_cache_dir)
        print(f"    Dataset loaded: {len(dapo_dataset)} samples available")

        # Sample prompts
        primary_prompts = sample_prompts(
            dapo_dataset, config.primary_size, seed=config.dataset_seed
        )
        secondary_prompts = None
        if args.mode == "runahead":
            # Use different seed for secondary (seed+1 if seed provided, else None)
            secondary_seed = (
                config.dataset_seed + 1 if config.dataset_seed is not None else None
            )
            secondary_prompts = sample_prompts(
                dapo_dataset, config.primary_size, seed=secondary_seed
            )
            print(
                f"    Sampled {len(primary_prompts)} primary + "
                f"{len(secondary_prompts)} secondary prompts"
            )
        else:
            print(f"    Sampled {len(primary_prompts)} primary prompts")

        # Load tokenizer
        print("\n[4] Loading tokenizer...")
        tokenizer = hf_tokenizer(config.model_path, trust_remote_code=True)

        # Build workloads
        print("\n[5] Building workloads...")
        num_long = max(1, int(config.primary_size * config.long_tail_ratio))
        num_short = config.primary_size - num_long
        print(f"    {num_short} short ({config.short_max_tokens} tokens), "
              f"{num_long} long ({config.long_max_tokens} tokens)")

        primary_dp = build_primary_dataproto(config, primary_prompts)

        # Run based on mode
        print(f"\n[6] Running {args.mode}...")
        if args.mode == "baseline":
            metrics = run_baseline(manager, primary_dp)
            print(f"    Time: {metrics.time_seconds:.2f}s, "
                  f"{metrics.primary_tokens} tokens")
        else:  # runahead
            secondary_dp = build_secondary_dataproto(config, secondary_prompts, tokenizer)
            runahead_cfg = RunaheadConfig(
                enabled=True,
                load_threshold=config.load_threshold,
                admit_loop_poll_s=config.admit_loop_poll_s,
                max_secondary_concurrent=config.max_secondary_concurrent,
            )
            metrics = run_with_runahead(manager, primary_dp, secondary_dp, runahead_cfg)
            print(f"    Time: {metrics.time_seconds:.2f}s, "
                  f"primary={metrics.primary_tokens} tokens, "
                  f"runahead={metrics.runahead_tokens_total} tokens")

        # Build result
        result = BenchmarkResult(
            mode=args.mode,
            config=asdict(config),
            metrics=metrics,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Mode:                 {result.mode}")
        print(f"Time:                 {metrics.time_seconds:.2f}s")
        print(f"Primary tokens:       {metrics.primary_tokens}")
        print(f"Primary completed:    {metrics.primary_completed}")
        if args.mode == "runahead":
            print(f"Runahead tokens:      {metrics.runahead_tokens_total}")
            print(f"  - Completed:        {metrics.runahead_completed_count} "
                  f"({metrics.runahead_tokens_completed} tokens)")
            print(f"  - Aborted:          {metrics.runahead_aborted_count} "
                  f"({metrics.runahead_tokens_aborted} tokens)")
            print(f"  - Rejected:         {metrics.runahead_rejected_count}")
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
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\nResults saved to: {output_path}")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
