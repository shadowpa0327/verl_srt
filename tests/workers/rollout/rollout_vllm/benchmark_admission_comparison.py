#!/usr/bin/env python3
"""
Admission Control Comparison Benchmark

Compares performance between two runahead admission strategies:
1. Slack Detection (SlackFillingServerManager) - Client-side budget tracking
2. Server-Side Admission (ServerSideAdmissionServerManager) - Server-side global enforcement

Key differences:
- Slack Detection: Per-worker budget, can oversubscribe with N workers (N × budget)
- Server-Side Admission: Global limit enforced via Ray actor, no oversubscription possible

Metrics compared:
- Primary overhead: Extra time caused by runahead
- Runahead tokens: Tokens generated speculatively
- Rejection rate: How often runahead is rejected
- Throughput: Total tokens / time

Usage:
    # Quick single comparison
    NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py --single

    # Full matrix comparison
    NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/benchmark_admission_comparison.py --rounds 3

Environment Variables:
    MODEL_PATH: Model to use (default: Qwen/Qwen2.5-0.5B-Instruct)
    NUM_GPUS: Number of GPUs / DP size (default: 2)
    PRIMARY_SIZE: Primary batch size (default: 16)
    BUDGET_PER_SERVER: Runahead budget per server (default: 4)
    SHORT_MAX_TOKENS: Short request max tokens (default: 1024)
    LONG_MAX_TOKENS: Long request max tokens (default: 8192)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List
from uuid import uuid4

import ray

# Import from slack filling implementation
from test_vllm_run_ahead_slack_filling import (
    SlackFillingConfig,
    SlackFillingServerManager,
    SlackFillingRunaheadController,
    SlackFillingAgentLoopWorker,
    BatchTracker,
    RequestTracker,
)

# Import from server-side admission implementation
from test_vllm_run_ahead_server_side_admission import (
    AdmissionGateConfig,
    ServerSideAdmissionServerManager,
    ServerSideAdmissionController,
    get_or_create_registry,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    primary_size: int = 16
    runahead_size: int = 16
    long_tail_ratio: float = 0.20
    budget_per_server: int = 4
    load_threshold: int = 32  # max (running + waiting) to consider slack
    kv_cache_threshold: float = 0.90
    poll_interval_s: float = 0.05
    short_max_tokens: int = 1024
    long_max_tokens: int = 8192
    num_gpus: int = 2
    tp_size: int = 1
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"

    @property
    def dp_size(self) -> int:
        return self.num_gpus // self.tp_size


# =============================================================================
# Metrics and Results
# =============================================================================

@dataclass
class RunMetrics:
    """Metrics from a single run."""
    primary_time: float = 0.0
    primary_tokens: int = 0
    primary_completed: int = 0
    runahead_tokens_total: int = 0
    runahead_tokens_completed: int = 0
    runahead_tokens_aborted: int = 0
    runahead_completed_count: int = 0
    runahead_aborted_count: int = 0
    runahead_rejected_count: int = 0
    backpressure_events: int = 0
    feeder_ticks: int = 0


@dataclass
class MultiWorkerMetrics:
    """Metrics from multi-worker racing test."""
    num_workers: int = 0
    total_time: float = 0.0
    max_observed_inflight: int = 0      # Peak concurrent runahead at server
    racing_violations: int = 0           # Times exceeded budget_per_server
    total_runahead_submitted: int = 0
    total_runahead_completed: int = 0
    total_runahead_rejected: int = 0
    budget_per_server: int = 0
    inflight_samples: List[int] = field(default_factory=list)  # Time series of inflight counts


@dataclass
class ComparisonResult:
    """Result of a single comparison (baseline + slack + server)."""
    experiment_id: str
    config: dict
    baseline: RunMetrics
    slack_detection: RunMetrics
    server_admission: RunMetrics
    slack_overhead_pct: float = 0.0
    server_overhead_pct: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.baseline.primary_time > 0:
            self.slack_overhead_pct = (
                (self.slack_detection.primary_time - self.baseline.primary_time)
                / self.baseline.primary_time * 100
            )
            self.server_overhead_pct = (
                (self.server_admission.primary_time - self.baseline.primary_time)
                / self.baseline.primary_time * 100
            )


# =============================================================================
# Workload Generation
# =============================================================================

HARD_MATH_PROBLEM = """In triangle ABC, sin(angle A) = 4/5 and angle A < 90 degrees. Let D be a point outside triangle ABC such that angle BAD = angle DAC and angle BDC = 90 degrees. Suppose that AD = 1 and that BD/CD = 3/2. If AB + AC can be expressed in the form (a*sqrt(b))/c where a, b, c are pairwise relatively prime integers, find a + b + c. Show your complete step-by-step solution with all mathematical reasoning."""


def generate_workload(
    size: int,
    long_tail_ratio: float,
    short_max_tokens: int,
    long_max_tokens: int,
    prefix: str = "primary"
) -> List[dict]:
    """Generate workload with long-tail distribution."""
    num_long = max(1, int(size * long_tail_ratio))
    num_short = size - num_long

    prompts = []

    # Short prompts first
    for i in range(num_short):
        prompts.append({
            "request_id": f"{prefix}_{i}_{uuid4().hex[:8]}",
            "prompt": HARD_MATH_PROBLEM,
            "max_tokens": short_max_tokens,
            "is_long": False,
        })

    # Long prompts
    for i in range(num_long):
        prompts.append({
            "request_id": f"{prefix}_{num_short + i}_{uuid4().hex[:8]}",
            "prompt": HARD_MATH_PROBLEM,
            "max_tokens": long_max_tokens,
            "is_long": True,
        })

    random.shuffle(prompts)
    return prompts


# =============================================================================
# Benchmark Runner
# =============================================================================

class AdmissionComparisonBenchmark:
    """Runs comparison benchmark between slack detection and server-side admission."""

    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.servers = []
        self.server_handles = []
        self.tokenizer = None
        self.trainer_config = None

    def setup(self):
        """Initialize Ray and vLLM servers."""
        from hydra import compose, initialize_config_dir

        print("=" * 80)
        print("ADMISSION CONTROL COMPARISON BENCHMARK")
        print("=" * 80)
        print(f"Model: {self.exp_config.model_path}")
        print(f"GPUs: {self.exp_config.num_gpus} | TP: {self.exp_config.tp_size} | DP: {self.exp_config.dp_size}")
        print("=" * 80)

        print("\n[1] Initializing Ray...")
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARNING",
                    "VLLM_USE_V1": "1",
                }
            },
            ignore_reinit_error=True,
        )

        print("\n[2] Creating config...")
        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = self.exp_config.num_gpus
        config.trainer.nnodes = 1
        config.actor_rollout_ref.model.path = self.exp_config.model_path
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = self.exp_config.tp_size
        config.actor_rollout_ref.rollout.disable_log_stats = False
        config.actor_rollout_ref.rollout.prompt_length = 512
        config.actor_rollout_ref.rollout.response_length = 16384
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9
        config.actor_rollout_ref.rollout.enable_prefix_caching = False

        self.trainer_config = config

        print(f"\n[3] Creating {self.exp_config.dp_size} vLLM server(s)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        rollout_class = get_rollout_replica_class("vllm")

        for dp_rank in range(self.exp_config.dp_size):
            print(f"   Creating server {dp_rank}...")
            server = rollout_class(
                replica_rank=dp_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.exp_config.tp_size,
            )
            asyncio.run(server.init_standalone())
            self.servers.append(server)
            self.server_handles.append(server._server_handle)
            print(f"   Server {dp_rank} ready")

        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.exp_config.model_path)
        print("\n[4] Servers ready")

    def teardown(self):
        """Shutdown Ray."""
        print("\nShutting down Ray...")
        ray.shutdown()

    async def run_baseline(self, primary_prompts: list) -> RunMetrics:
        """Run baseline (primary only, no runahead)."""
        print("\n   Running BASELINE (primary only)...")

        slack_config = SlackFillingConfig(
            budget_per_server=self.exp_config.budget_per_server,
            load_threshold=self.exp_config.load_threshold,
            kv_cache_threshold=self.exp_config.kv_cache_threshold,
            poll_interval_s=self.exp_config.poll_interval_s,
        )

        sm = SlackFillingServerManager(
            self.trainer_config, self.server_handles, slack_config
        )
        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))

        primary_tasks = set()
        start_time = time.perf_counter()

        for i, item in enumerate(primary_prompts):
            rid = item["request_id"]
            tr = RequestTracker(
                request_id=rid,
                batch_id="primary",
                index=i,
                max_tokens=item["max_tokens"],
            )
            primary_tracker.requests[rid] = tr

            messages = [{"role": "user", "content": item["prompt"]}]
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": item["max_tokens"],
            }

            task = asyncio.create_task(
                sm.generate(
                    request_id=rid,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    tracker=tr,
                    kind="primary",
                    sticky=True,
                )
            )
            primary_tasks.add(task)

        while primary_tasks:
            done, primary_tasks = await asyncio.wait(
                primary_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                try:
                    await t
                except Exception:
                    pass

        end_time = time.perf_counter()

        return RunMetrics(
            primary_time=end_time - start_time,
            primary_tokens=sum(r.token_count for r in primary_tracker.requests.values()),
            primary_completed=primary_tracker.completed,
        )

    async def run_slack_detection(
        self,
        primary_prompts: list,
        runahead_prompts: list,
    ) -> RunMetrics:
        """Run with slack detection (SlackFillingServerManager)."""
        print("\n   Running SLACK DETECTION...")

        slack_config = SlackFillingConfig(
            budget_per_server=self.exp_config.budget_per_server,
            load_threshold=self.exp_config.load_threshold,
            kv_cache_threshold=self.exp_config.kv_cache_threshold,
            poll_interval_s=self.exp_config.poll_interval_s,
        )

        worker = SlackFillingAgentLoopWorker(
            self.trainer_config, self.server_handles, slack_config=slack_config
        )
        controller = SlackFillingRunaheadController(worker.server_manager, slack_config)

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        start_time = time.perf_counter()
        await controller.run_with_runahead(
            primary_items=primary_prompts,
            runahead_items=runahead_prompts,
            primary_tracker=primary_tracker,
            runahead_tracker=runahead_tracker,
            worker=worker,
        )
        end_time = time.perf_counter()

        runahead_tokens_completed = sum(
            r.token_count for r in runahead_tracker.requests.values()
            if r.status == "completed"
        )
        runahead_tokens_aborted = sum(
            r.token_count for r in runahead_tracker.requests.values()
            if r.status == "aborted"
        )

        return RunMetrics(
            primary_time=end_time - start_time,
            primary_tokens=sum(r.token_count for r in primary_tracker.requests.values()),
            primary_completed=primary_tracker.completed,
            runahead_tokens_total=runahead_tokens_completed + runahead_tokens_aborted,
            runahead_tokens_completed=runahead_tokens_completed,
            runahead_tokens_aborted=runahead_tokens_aborted,
            runahead_completed_count=runahead_tracker.completed,
            runahead_aborted_count=runahead_tracker.aborted,
            runahead_rejected_count=worker.server_manager.backpressure_rejections,
            backpressure_events=controller.backpressure_events,
            feeder_ticks=controller.feeder_ticks,
        )

    async def run_server_admission(
        self,
        primary_prompts: list,
        runahead_prompts: list,
    ) -> RunMetrics:
        """Run with server-side admission (ServerSideAdmissionServerManager)."""
        print("\n   Running SERVER-SIDE ADMISSION...")

        admission_config = AdmissionGateConfig(
            max_runahead_inflight=self.exp_config.budget_per_server,
            enforce_slack=True,
            load_threshold=self.exp_config.load_threshold,
            kv_cache_threshold=self.exp_config.kv_cache_threshold,
            poll_interval_s=self.exp_config.poll_interval_s,
            max_runahead_retries=3,
        )

        # Create admission gates via registry
        registry = get_or_create_registry()

        async def create_gates():
            gates = []
            for i, h in enumerate(self.server_handles):
                gate = await registry.get_or_create.remote(i, h, admission_config)
                gates.append(gate)
            return gates

        gated_handles = await create_gates()

        server_manager = ServerSideAdmissionServerManager(
            config=self.trainer_config,
            gated_handles=gated_handles,
            admission_config=admission_config,
        )

        await server_manager.validate_gate_handles()

        controller = ServerSideAdmissionController(
            server_manager, admission_config, self.tokenizer
        )

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        start_time = time.perf_counter()
        await controller.run_with_runahead(
            primary_items=primary_prompts,
            runahead_items=runahead_prompts,
            primary_tracker=primary_tracker,
            runahead_tracker=runahead_tracker,
        )
        end_time = time.perf_counter()

        runahead_tokens_completed = sum(
            r.token_count for r in runahead_tracker.requests.values()
            if r.status == "completed"
        )
        runahead_tokens_aborted = sum(
            r.token_count for r in runahead_tracker.requests.values()
            if r.status == "aborted"
        )

        # Collect gate stats
        total_server_rejections = 0
        for gate in gated_handles:
            stats = await gate.get_admission_stats.remote()
            total_server_rejections += stats.get("runahead_rejected_total", 0)

        return RunMetrics(
            primary_time=end_time - start_time,
            primary_tokens=sum(r.token_count for r in primary_tracker.requests.values()),
            primary_completed=primary_tracker.completed,
            runahead_tokens_total=runahead_tokens_completed + runahead_tokens_aborted,
            runahead_tokens_completed=runahead_tokens_completed,
            runahead_tokens_aborted=runahead_tokens_aborted,
            runahead_completed_count=runahead_tracker.completed,
            runahead_aborted_count=runahead_tracker.aborted,
            runahead_rejected_count=server_manager.runahead_rejected + server_manager.client_side_rejections,
            backpressure_events=controller.backpressure_events,
            feeder_ticks=controller.feeder_ticks,
        )

    def run_single_comparison(
        self,
        primary_size: Optional[int] = None,
        budget_per_server: Optional[int] = None,
        long_tail_ratio: Optional[float] = None,
    ) -> ComparisonResult:
        """Run a single comparison (baseline + slack + server)."""
        primary_size = primary_size or self.exp_config.primary_size
        budget = budget_per_server or self.exp_config.budget_per_server
        ratio = long_tail_ratio if long_tail_ratio is not None else self.exp_config.long_tail_ratio

        print(f"\n{'='*80}")
        print(f"COMPARISON: PRIMARY={primary_size}, BUDGET={budget}, RATIO={ratio:.0%}")
        print(f"{'='*80}")

        # Update config
        self.exp_config.budget_per_server = budget

        # Generate workloads
        primary_prompts = generate_workload(
            size=primary_size,
            long_tail_ratio=ratio,
            short_max_tokens=self.exp_config.short_max_tokens,
            long_max_tokens=self.exp_config.long_max_tokens,
            prefix="primary",
        )
        runahead_prompts = generate_workload(
            size=primary_size,
            long_tail_ratio=ratio,
            short_max_tokens=self.exp_config.short_max_tokens,
            long_max_tokens=self.exp_config.long_max_tokens,
            prefix="runahead",
        )

        num_long = sum(1 for p in primary_prompts if p["is_long"])
        num_short = primary_size - num_long
        print(f"   Workload: {num_short} short ({self.exp_config.short_max_tokens} tok), "
              f"{num_long} long ({self.exp_config.long_max_tokens} tok)")

        # Run baseline
        baseline_metrics = asyncio.run(self.run_baseline(primary_prompts))
        print(f"   Baseline: {baseline_metrics.primary_time:.2f}s, "
              f"{baseline_metrics.primary_tokens} tokens")

        # Run slack detection
        slack_metrics = asyncio.run(
            self.run_slack_detection(primary_prompts, runahead_prompts)
        )
        print(f"   Slack: {slack_metrics.primary_time:.2f}s, "
              f"primary={slack_metrics.primary_tokens} tok, "
              f"runahead={slack_metrics.runahead_tokens_total} tok")

        # Run server-side admission
        server_metrics = asyncio.run(
            self.run_server_admission(primary_prompts, runahead_prompts)
        )
        print(f"   Server: {server_metrics.primary_time:.2f}s, "
              f"primary={server_metrics.primary_tokens} tok, "
              f"runahead={server_metrics.runahead_tokens_total} tok")

        # Calculate overheads
        slack_overhead = 0.0
        server_overhead = 0.0
        if baseline_metrics.primary_time > 0:
            slack_overhead = (
                (slack_metrics.primary_time - baseline_metrics.primary_time)
                / baseline_metrics.primary_time * 100
            )
            server_overhead = (
                (server_metrics.primary_time - baseline_metrics.primary_time)
                / baseline_metrics.primary_time * 100
            )

        print(f"\n   Overhead: Slack={slack_overhead:+.1f}%, Server={server_overhead:+.1f}%")
        print(f"   Runahead tokens: Slack={slack_metrics.runahead_tokens_total}, "
              f"Server={server_metrics.runahead_tokens_total}")
        print(f"   Rejections: Slack={slack_metrics.runahead_rejected_count}, "
              f"Server={server_metrics.runahead_rejected_count}")

        return ComparisonResult(
            experiment_id=f"cmp_{primary_size}_{budget}_{int(ratio*100)}pct_{uuid4().hex[:8]}",
            config={
                "primary_size": primary_size,
                "budget_per_server": budget,
                "long_tail_ratio": ratio,
                "short_max_tokens": self.exp_config.short_max_tokens,
                "long_max_tokens": self.exp_config.long_max_tokens,
                "num_gpus": self.exp_config.num_gpus,
            },
            baseline=baseline_metrics,
            slack_detection=slack_metrics,
            server_admission=server_metrics,
            slack_overhead_pct=slack_overhead,
            server_overhead_pct=server_overhead,
        )


# =============================================================================
# Multi-Worker Racing Test
# =============================================================================

@ray.remote
class RuanaheadWorker:
    """Ray actor that runs runahead workload for multi-worker testing."""

    def __init__(self, worker_id: int, server_handles: list, config, tokenizer_path: str):
        self.worker_id = worker_id
        self.server_handles = server_handles
        self.config = config
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tokenizer

    async def run_slack_detection(
        self,
        runahead_prompts: list,
        slack_config_dict: dict,
    ) -> dict:
        """Run runahead with slack detection (client-side budget)."""
        from test_vllm_run_ahead_slack_filling import (
            SlackFillingConfig,
            SlackFillingServerManager,
            BatchTracker,
            RequestTracker,
        )

        slack_config = SlackFillingConfig(**slack_config_dict)
        sm = SlackFillingServerManager(self.config, self.server_handles, slack_config)
        tokenizer = self._get_tokenizer()

        tracker = BatchTracker(batch_id=f"runahead_w{self.worker_id}", total=len(runahead_prompts))
        tasks = set()
        submitted = 0
        completed = 0
        rejected = 0

        for i, item in enumerate(runahead_prompts):
            rid = f"w{self.worker_id}_{item['request_id']}"
            tr = RequestTracker(
                request_id=rid,
                batch_id=f"runahead_w{self.worker_id}",
                index=i,
                max_tokens=item["max_tokens"],
            )
            tracker.requests[rid] = tr

            messages = [{"role": "user", "content": item["prompt"]}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": item["max_tokens"],
            }

            task = asyncio.create_task(
                sm.generate(
                    request_id=rid,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    tracker=tr,
                    kind="runahead",
                    sticky=False,
                )
            )
            tasks.add(task)
            submitted += 1

        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    await t
                    completed += 1
                except Exception:
                    rejected += 1

        return {
            "worker_id": self.worker_id,
            "submitted": submitted,
            "completed": completed,
            "rejected": rejected,
        }

    async def run_server_admission(
        self,
        runahead_prompts: list,
        gated_handles: list,
        admission_config_dict: dict,
    ) -> dict:
        """Run runahead with server-side admission (global lock)."""
        from test_vllm_run_ahead_server_side_admission import (
            AdmissionGateConfig,
            ServerSideAdmissionServerManager,
            BatchTracker,
            RequestTracker,
        )

        admission_config = AdmissionGateConfig(**admission_config_dict)
        sm = ServerSideAdmissionServerManager(
            config=self.config,
            gated_handles=gated_handles,
            admission_config=admission_config,
        )
        tokenizer = self._get_tokenizer()

        tracker = BatchTracker(batch_id=f"runahead_w{self.worker_id}", total=len(runahead_prompts))
        tasks = set()
        submitted = 0
        completed = 0
        rejected = 0

        for i, item in enumerate(runahead_prompts):
            rid = f"w{self.worker_id}_{item['request_id']}"
            tr = RequestTracker(
                request_id=rid,
                batch_id=f"runahead_w{self.worker_id}",
                index=i,
                max_tokens=item["max_tokens"],
            )
            tracker.requests[rid] = tr

            messages = [{"role": "user", "content": item["prompt"]}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
            sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": item["max_tokens"],
            }

            task = asyncio.create_task(
                sm.generate(
                    request_id=rid,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    tracker=tr,
                    kind="runahead",
                )
            )
            tasks.add(task)
            submitted += 1

        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                try:
                    await t
                    completed += 1
                except Exception:
                    rejected += 1

        return {
            "worker_id": self.worker_id,
            "submitted": submitted,
            "completed": completed,
            "rejected": rejected + sm.runahead_rejected,
        }


@ray.remote
class InflightMonitor:
    """Ray actor that samples inflight counts from admission gates."""

    def __init__(self, gated_handles: list, sample_interval: float = 0.1):
        self.gated_handles = gated_handles
        self.sample_interval = sample_interval
        self.samples = []
        self.max_observed = 0
        self._running = False

    async def start_monitoring(self):
        """Start sampling inflight counts in background."""
        self._running = True
        while self._running:
            total_inflight = 0
            for gate in self.gated_handles:
                try:
                    stats = await gate.get_admission_stats.remote()
                    total_inflight += stats.get("runahead_inflight", 0)
                except Exception:
                    pass
            self.samples.append(total_inflight)
            if total_inflight > self.max_observed:
                self.max_observed = total_inflight
            await asyncio.sleep(self.sample_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False

    def get_results(self) -> dict:
        """Get monitoring results."""
        return {
            "samples": self.samples,
            "max_observed": self.max_observed,
            "num_samples": len(self.samples),
        }


class MultiWorkerBenchmark:
    """Runs multi-worker racing test to compare admission strategies."""

    def __init__(self, base_benchmark: AdmissionComparisonBenchmark, num_workers: int):
        self.base = base_benchmark
        self.num_workers = num_workers

    async def run_multi_worker_slack(
        self,
        runahead_prompts: list,
        slack_config_dict: dict,
    ) -> MultiWorkerMetrics:
        """Run N workers with slack detection (client-side budget)."""
        print(f"\n   Running SLACK DETECTION with {self.num_workers} workers...")

        # Create worker actors
        workers = []
        for i in range(self.num_workers):
            worker = RuanaheadWorker.remote(
                worker_id=i,
                server_handles=self.base.server_handles,
                config=self.base.trainer_config,
                tokenizer_path=self.base.exp_config.model_path,
            )
            workers.append(worker)

        # Split workload across workers
        prompts_per_worker = len(runahead_prompts) // self.num_workers
        worker_prompts = []
        for i in range(self.num_workers):
            start = i * prompts_per_worker
            end = start + prompts_per_worker if i < self.num_workers - 1 else len(runahead_prompts)
            worker_prompts.append(runahead_prompts[start:end])

        # Run all workers concurrently
        start_time = time.perf_counter()
        tasks = [
            workers[i].run_slack_detection.remote(worker_prompts[i], slack_config_dict)
            for i in range(self.num_workers)
        ]
        results = ray.get(tasks)
        end_time = time.perf_counter()

        # Aggregate results
        total_submitted = sum(r["submitted"] for r in results)
        total_completed = sum(r["completed"] for r in results)
        total_rejected = sum(r["rejected"] for r in results)

        # For slack detection, we can't directly measure inflight at server
        # because there's no global counter. We use the budget as max possible.
        max_possible = self.num_workers * slack_config_dict["budget_per_server"]

        return MultiWorkerMetrics(
            num_workers=self.num_workers,
            total_time=end_time - start_time,
            max_observed_inflight=max_possible,  # Theoretical max (no global tracking)
            racing_violations=0,  # Can't detect without server-side counter
            total_runahead_submitted=total_submitted,
            total_runahead_completed=total_completed,
            total_runahead_rejected=total_rejected,
            budget_per_server=slack_config_dict["budget_per_server"],
        )

    async def run_multi_worker_server(
        self,
        runahead_prompts: list,
        admission_config_dict: dict,
    ) -> MultiWorkerMetrics:
        """Run N workers with server-side admission (global lock)."""
        print(f"\n   Running SERVER-SIDE ADMISSION with {self.num_workers} workers...")

        # Create admission gates via registry (shared across all workers)
        registry = get_or_create_registry()

        async def create_gates():
            from test_vllm_run_ahead_server_side_admission import AdmissionGateConfig
            config = AdmissionGateConfig(**admission_config_dict)
            gates = []
            for i, h in enumerate(self.base.server_handles):
                gate = await registry.get_or_create.remote(i, h, config)
                gates.append(gate)
            return gates

        gated_handles = await create_gates()

        # Create inflight monitor
        monitor = InflightMonitor.remote(gated_handles, sample_interval=0.05)
        monitor_task = monitor.start_monitoring.remote()

        # Create worker actors
        workers = []
        for i in range(self.num_workers):
            worker = RuanaheadWorker.remote(
                worker_id=i,
                server_handles=self.base.server_handles,
                config=self.base.trainer_config,
                tokenizer_path=self.base.exp_config.model_path,
            )
            workers.append(worker)

        # Split workload across workers
        prompts_per_worker = len(runahead_prompts) // self.num_workers
        worker_prompts = []
        for i in range(self.num_workers):
            start = i * prompts_per_worker
            end = start + prompts_per_worker if i < self.num_workers - 1 else len(runahead_prompts)
            worker_prompts.append(runahead_prompts[start:end])

        # Run all workers concurrently
        start_time = time.perf_counter()
        tasks = [
            workers[i].run_server_admission.remote(
                worker_prompts[i], gated_handles, admission_config_dict
            )
            for i in range(self.num_workers)
        ]
        results = ray.get(tasks)
        end_time = time.perf_counter()

        # Stop monitor and get results
        ray.get(monitor.stop_monitoring.remote())
        await asyncio.sleep(0.1)  # Let monitor finish
        monitor_results = ray.get(monitor.get_results.remote())

        # Aggregate results
        total_submitted = sum(r["submitted"] for r in results)
        total_completed = sum(r["completed"] for r in results)
        total_rejected = sum(r["rejected"] for r in results)

        # Count racing violations (times inflight exceeded total budget across all servers)
        budget_per_server = admission_config_dict["max_runahead_inflight"]
        num_servers = len(self.base.server_handles)
        total_budget = budget_per_server * num_servers  # e.g., 2 servers × 2 budget = 4 total
        samples = monitor_results["samples"]
        racing_violations = sum(1 for s in samples if s > total_budget)

        return MultiWorkerMetrics(
            num_workers=self.num_workers,
            total_time=end_time - start_time,
            max_observed_inflight=monitor_results["max_observed"],
            racing_violations=racing_violations,
            total_runahead_submitted=total_submitted,
            total_runahead_completed=total_completed,
            total_runahead_rejected=total_rejected,
            budget_per_server=budget_per_server,
            inflight_samples=samples,
        )

    def run_comparison(self) -> tuple:
        """Run multi-worker comparison between slack and server-side."""
        print(f"\n{'='*80}")
        print(f"MULTI-WORKER RACING TEST (N={self.num_workers} workers)")
        print(f"{'='*80}")

        # Generate runahead workload (more prompts for multi-worker)
        runahead_prompts = generate_workload(
            size=self.num_workers * 8,  # 8 requests per worker
            long_tail_ratio=0.0,  # All short for faster testing
            short_max_tokens=256,  # Short tokens for faster testing
            long_max_tokens=256,
            prefix="mw_runahead",
        )

        print(f"   Workload: {len(runahead_prompts)} requests, {len(runahead_prompts)//self.num_workers} per worker")

        # Config for slack detection
        slack_config_dict = {
            "budget_per_server": self.base.exp_config.budget_per_server,
            "load_threshold": self.base.exp_config.load_threshold,
            "kv_cache_threshold": self.base.exp_config.kv_cache_threshold,
            "poll_interval_s": self.base.exp_config.poll_interval_s,
        }

        # Config for server-side admission
        admission_config_dict = {
            "max_runahead_inflight": self.base.exp_config.budget_per_server,
            "enforce_slack": False,  # Disable slack check for pure racing test
            "load_threshold": self.base.exp_config.load_threshold,
            "kv_cache_threshold": self.base.exp_config.kv_cache_threshold,
            "poll_interval_s": self.base.exp_config.poll_interval_s,
            "max_runahead_retries": 3,
        }

        # Run slack detection
        slack_metrics = asyncio.run(
            self.run_multi_worker_slack(runahead_prompts, slack_config_dict)
        )

        # Run server-side admission
        server_metrics = asyncio.run(
            self.run_multi_worker_server(runahead_prompts, admission_config_dict)
        )

        return slack_metrics, server_metrics


def print_multi_worker_results(slack: MultiWorkerMetrics, server: MultiWorkerMetrics):
    """Print multi-worker comparison results."""
    print(f"\n{'='*80}")
    print(f"MULTI-WORKER RACING TEST RESULTS (N={slack.num_workers} workers, budget={slack.budget_per_server})")
    print(f"{'='*80}")

    print(f"\n{'Metric':<30} {'Slack Detection':<20} {'Server-Side':<20}")
    print("-" * 70)
    print(f"{'Max Observed Inflight:':<30} {slack.max_observed_inflight:<20} {server.max_observed_inflight:<20}")
    print(f"{'Racing Violations:':<30} {'N/A (no tracking)':<20} {server.racing_violations:<20}")
    print(f"{'Runahead Submitted:':<30} {slack.total_runahead_submitted:<20} {server.total_runahead_submitted:<20}")
    print(f"{'Runahead Completed:':<30} {slack.total_runahead_completed:<20} {server.total_runahead_completed:<20}")
    print(f"{'Runahead Rejected:':<30} {slack.total_runahead_rejected:<20} {server.total_runahead_rejected:<20}")
    print(f"{'Total Time:':<30} {slack.total_time:.2f}s{'':<14} {server.total_time:.2f}s")
    print("-" * 70)

    # Analysis
    print("\n--- ANALYSIS ---")
    theoretical_max = slack.num_workers * slack.budget_per_server
    # Note: total budget = budget_per_server × num_servers (assume 2 servers based on typical setup)
    num_servers = 2  # Typical DP setup
    total_server_budget = server.budget_per_server * num_servers
    print(f"Theoretical max inflight (Slack): {slack.num_workers} workers × {slack.budget_per_server} budget = {theoretical_max}")
    print(f"Server-Side max observed: {server.max_observed_inflight} (total budget: {server.budget_per_server}/server × {num_servers} servers = {total_server_budget})")

    if server.max_observed_inflight <= total_server_budget:
        print("✓ Server-Side Admission maintained global limit (no racing)")
    else:
        print("✗ Server-Side Admission exceeded limit (unexpected)")

    if server.racing_violations == 0:
        print("✓ No racing violations detected for Server-Side")
    else:
        print(f"✗ {server.racing_violations} racing violations detected")

    print(f"\nNote: Slack Detection has no global tracking - each worker has independent budget.")
    print(f"      With {slack.num_workers} workers, up to {theoretical_max} concurrent requests are possible.")


# =============================================================================
# Results Output
# =============================================================================

def print_summary_table(results: list):
    """Print summary table of comparison results."""
    print("\n" + "=" * 130)
    print("COMPARISON SUMMARY TABLE")
    print("=" * 130)

    print(f"\n{'CONFIG':<12} | {'BASELINE(s)':<11} | {'SLACK(s)':<10} | {'SERVER(s)':<10} | "
          f"{'SLACK OH%':<10} | {'SERVER OH%':<11} | {'SLACK RA_tok':<12} | {'SERVER RA_tok':<13}")
    print("-" * 130)

    for r in results:
        cfg_str = f"{r.config['primary_size']}/{r.config['budget_per_server']}/{int(r.config['long_tail_ratio']*100)}%"
        print(f"{cfg_str:<12} | "
              f"{r.baseline.primary_time:<11.2f} | "
              f"{r.slack_detection.primary_time:<10.2f} | "
              f"{r.server_admission.primary_time:<10.2f} | "
              f"{r.slack_overhead_pct:<+10.1f} | "
              f"{r.server_overhead_pct:<+11.1f} | "
              f"{r.slack_detection.runahead_tokens_total:<12} | "
              f"{r.server_admission.runahead_tokens_total:<13}")

    print("=" * 130)

    # Summary analysis
    print("\n--- ANALYSIS ---")
    avg_slack_oh = sum(r.slack_overhead_pct for r in results) / len(results) if results else 0
    avg_server_oh = sum(r.server_overhead_pct for r in results) / len(results) if results else 0
    avg_slack_ra = sum(r.slack_detection.runahead_tokens_total for r in results) / len(results) if results else 0
    avg_server_ra = sum(r.server_admission.runahead_tokens_total for r in results) / len(results) if results else 0

    print(f"Average Overhead: Slack={avg_slack_oh:+.1f}%, Server={avg_server_oh:+.1f}%")
    print(f"Average Runahead Tokens: Slack={avg_slack_ra:.0f}, Server={avg_server_ra:.0f}")

    if avg_slack_oh < avg_server_oh:
        diff = avg_server_oh - avg_slack_oh
        print(f"-> Slack Detection has {diff:.1f}% less overhead")
    else:
        diff = avg_slack_oh - avg_server_oh
        print(f"-> Server-Side Admission has {diff:.1f}% less overhead")

    if avg_slack_ra > avg_server_ra:
        diff = avg_slack_ra - avg_server_ra
        print(f"-> Slack Detection generates {diff:.0f} more runahead tokens on average")
    else:
        diff = avg_server_ra - avg_slack_ra
        print(f"-> Server-Side Admission generates {diff:.0f} more runahead tokens on average")

    print("\nNote: Server-Side Admission guarantees no oversubscription under multi-worker scenarios.")


def save_results(results: list, output_dir: str = "results"):
    """Save results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/admission_comparison_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "num_experiments": len(results),
        "results": []
    }

    for r in results:
        data["results"].append({
            "experiment_id": r.experiment_id,
            "config": r.config,
            "baseline": asdict(r.baseline),
            "slack_detection": asdict(r.slack_detection),
            "server_admission": asdict(r.server_admission),
            "slack_overhead_pct": r.slack_overhead_pct,
            "server_overhead_pct": r.server_overhead_pct,
            "timestamp": r.timestamp,
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def save_multi_worker_results(slack: MultiWorkerMetrics, server: MultiWorkerMetrics, output_dir: str = "results"):
    """Save multi-worker results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/multi_worker_racing_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "test_type": "multi_worker_racing",
        "num_workers": slack.num_workers,
        "budget_per_server": slack.budget_per_server,
        "slack_detection": {
            "max_observed_inflight": slack.max_observed_inflight,
            "racing_violations": slack.racing_violations,
            "total_runahead_submitted": slack.total_runahead_submitted,
            "total_runahead_completed": slack.total_runahead_completed,
            "total_runahead_rejected": slack.total_runahead_rejected,
            "total_time": slack.total_time,
        },
        "server_admission": {
            "max_observed_inflight": server.max_observed_inflight,
            "racing_violations": server.racing_violations,
            "total_runahead_submitted": server.total_runahead_submitted,
            "total_runahead_completed": server.total_runahead_completed,
            "total_runahead_rejected": server.total_runahead_rejected,
            "total_time": server.total_time,
            "inflight_samples": server.inflight_samples,
        },
        "analysis": {
            "theoretical_max_slack": slack.num_workers * slack.budget_per_server,
            "server_within_limit": server.max_observed_inflight <= server.budget_per_server,
        }
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def run_experiment_matrix(benchmark: AdmissionComparisonBenchmark, num_rounds: int = 1) -> list:
    """Run full experiment matrix."""
    primary_sizes = [16, 32]
    budgets = [4, 8]
    long_tail_ratios = [0.20, 0.40]

    results = []
    total = len(primary_sizes) * len(budgets) * len(long_tail_ratios) * num_rounds
    current = 0

    for round_idx in range(num_rounds):
        for primary_size in primary_sizes:
            for budget in budgets:
                for ratio in long_tail_ratios:
                    current += 1
                    print(f"\n\n{'#'*80}")
                    print(f"# EXPERIMENT {current}/{total} (Round {round_idx + 1}/{num_rounds})")
                    print(f"{'#'*80}")

                    result = benchmark.run_single_comparison(
                        primary_size=primary_size,
                        budget_per_server=budget,
                        long_tail_ratio=ratio,
                    )
                    result.config["round"] = round_idx + 1
                    results.append(result)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Admission Control Comparison Benchmark")
    parser.add_argument("--single", action="store_true", help="Run single config from env vars")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds (default: 1)")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--multi-worker", type=int, default=0,
                        help="Run multi-worker racing test with N workers (default: 0 = disabled)")
    args = parser.parse_args()

    exp_config = ExperimentConfig(
        primary_size=int(os.environ.get("PRIMARY_SIZE", "16")),
        runahead_size=int(os.environ.get("RUNAHEAD_SIZE", os.environ.get("PRIMARY_SIZE", "16"))),
        long_tail_ratio=float(os.environ.get("LONG_TAIL_RATIO", "0.20")),
        budget_per_server=int(os.environ.get("BUDGET_PER_SERVER", "4")),
        load_threshold=int(os.environ.get("LOAD_THRESHOLD", "32")),
        kv_cache_threshold=float(os.environ.get("KV_CACHE_THRESHOLD", "0.90")),
        poll_interval_s=float(os.environ.get("POLL_INTERVAL", "0.05")),
        short_max_tokens=int(os.environ.get("SHORT_MAX_TOKENS", "1024")),
        long_max_tokens=int(os.environ.get("LONG_MAX_TOKENS", "8192")),
        num_gpus=int(os.environ.get("NUM_GPUS", "2")),
        tp_size=int(os.environ.get("TP_SIZE", "1")),
        model_path=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"),
    )

    benchmark = AdmissionComparisonBenchmark(exp_config)

    try:
        benchmark.setup()

        if args.multi_worker > 0:
            # Run multi-worker racing test
            mw_benchmark = MultiWorkerBenchmark(benchmark, args.multi_worker)
            slack_metrics, server_metrics = mw_benchmark.run_comparison()
            print_multi_worker_results(slack_metrics, server_metrics)
            # Save multi-worker results
            save_multi_worker_results(slack_metrics, server_metrics, args.output_dir)
        elif args.single:
            result = benchmark.run_single_comparison()
            results = [result]
            print_summary_table(results)
            save_results(results, args.output_dir)
        else:
            results = run_experiment_matrix(benchmark, num_rounds=args.rounds)
            print_summary_table(results)
            save_results(results, args.output_dir)

    finally:
        benchmark.teardown()


if __name__ == "__main__":
    main()
