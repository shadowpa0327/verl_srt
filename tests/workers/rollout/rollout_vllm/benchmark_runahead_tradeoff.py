#!/usr/bin/env python3
"""
Runahead Tradeoff Benchmark

Measures the trade-off between:
1. Primary overhead: Extra time caused by runahead competing for resources
2. Runahead benefit: Tokens generated speculatively (free work)

See claude_docs/EXPERIMENT_PROGRESS.md for:
- Full experiment history (v1-v7)
- Results summary tables
- How to resume experiments

Current Configuration (v7 - Qwen3-8B):
- PRIMARY_SIZE: 16, 32, 64
- LOAD_THRESHOLD: 8, 16
- LONG_TAIL_RATIOS: 0.20, 0.40, 0.50
- Tokens: SHORT=2048, LONG=16384

Usage:
    # Run full experiment matrix
    NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/benchmark_runahead_tradeoff.py --rounds 3

    # Run single config
    PRIMARY_SIZE=16 LOAD_THRESHOLD=8 \
        python tests/workers/rollout/rollout_vllm/benchmark_runahead_tradeoff.py --single

Environment Variables:
    MODEL_PATH: Model to use (default: Qwen/Qwen3-8B)
    SHORT_MAX_TOKENS: Short request max tokens (default: 2048)
    LONG_MAX_TOKENS: Long request max tokens (default: 16384)
    NUM_GPUS: Number of GPUs / DP size (default: 2)
"""

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

from cachetools import LRUCache


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    primary_size: int = 32
    runahead_size: int = 32  # Match primary_size
    long_tail_ratio: float = 0.20
    load_threshold: int = 8
    budget_per_server: int = 16
    kv_cache_threshold: float = 0.90
    poll_interval_s: float = 0.05
    short_max_tokens: int = 1024
    long_max_tokens: int = 16384
    num_gpus: int = 2
    tp_size: int = 1
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"

    @property
    def dp_size(self) -> int:
        return self.num_gpus // self.tp_size


@dataclass
class SlackFillingConfig:
    """Configuration for slack-filling runahead."""
    load_threshold: int = 8  # Allow runahead if (running + waiting) <= threshold
    kv_cache_threshold: float = 0.85
    budget_per_server: int = 1
    poll_interval_s: float = 0.1
    poll_jitter_s: float = 0.03
    workload_cache_ttl_s: float = 0.3
    primary_priority: int = 0
    runahead_priority: int = 10


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
    backpressure_events: int = 0
    feeder_ticks: int = 0


@dataclass
class ExperimentResult:
    """Result of a single experiment (baseline + runahead)."""
    experiment_id: str
    config: dict
    baseline: RunMetrics
    runahead: RunMetrics
    primary_overhead_pct: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.baseline.primary_time > 0:
            self.primary_overhead_pct = (
                (self.runahead.primary_time - self.baseline.primary_time)
                / self.baseline.primary_time * 100
            )


# =============================================================================
# Workload Monitoring
# =============================================================================

@dataclass
class WorkloadSnapshot:
    """Snapshot of server workload at a point in time."""
    timestamp: float
    server_idx: int = -1
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    kv_cache_usage: float = 0.0
    error: Optional[str] = None

    @property
    def total_load(self) -> int:
        return self.num_requests_running + self.num_requests_waiting

    def has_slack(self, config: SlackFillingConfig) -> bool:
        if self.error:
            return False
        return (
            self.total_load <= config.load_threshold
            and self.kv_cache_usage <= config.kv_cache_threshold
        )


class WorkloadMonitor:
    """Monitor vLLM server workload via Prometheus metrics."""

    def __init__(self, server_handle, server_idx: int):
        self.server_handle = server_handle
        self.server_idx = server_idx
        self._warned_missing_metrics = False

    async def get_workload(self) -> WorkloadSnapshot:
        try:
            result = await self.server_handle.get_workload.remote()
            warning = result.get("warning")
            if warning and not self._warned_missing_metrics:
                self._warned_missing_metrics = True

            error_msg = None
            if result.get("error"):
                error_msg = str(result.get("error"))
            elif warning:
                error_msg = f"metrics_unavailable: {warning}"

            return WorkloadSnapshot(
                timestamp=time.perf_counter(),
                server_idx=self.server_idx,
                num_requests_running=result.get("num_requests_running", 0),
                num_requests_waiting=result.get("num_requests_waiting", 0),
                kv_cache_usage=result.get("kv_cache_usage", 0.0),
                error=error_msg,
            )
        except Exception as e:
            return WorkloadSnapshot(
                timestamp=time.perf_counter(),
                server_idx=self.server_idx,
                error=str(e)
            )


# =============================================================================
# Request Tracking
# =============================================================================

@dataclass
class RequestTracker:
    """Track individual request state and metrics."""
    request_id: str
    server_request_id: str = ""
    batch_id: str = ""
    index: int = 0
    max_tokens: int = 0
    server_idx: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    token_count: int = 0
    status: str = "pending"
    stop_reason: Optional[str] = None
    token_ids: list = field(default_factory=list)
    priority: int = 0

    @property
    def duration(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "aborted", "error", "rejected")


@dataclass
class BatchTracker:
    """Track batch-level metrics."""
    batch_id: str
    total: int
    requests: dict = field(default_factory=dict)
    start_time: float = 0.0

    @property
    def completed(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "completed")

    @property
    def aborted(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "aborted")

    @property
    def total_tokens(self) -> int:
        return sum(r.token_count for r in self.requests.values())

    def get_running_server_request_ids(self) -> list:
        return [
            r.server_request_id
            for r in self.requests.values()
            if r.status == "running" and r.server_request_id
        ]


# =============================================================================
# Server Manager
# =============================================================================

class BenchmarkServerManager:
    """Server manager for benchmark with slack-filling support."""

    def __init__(self, config, server_handles: list, slack_config: SlackFillingConfig):
        self.config = config
        self.server_handles = server_handles
        self.slack_config = slack_config
        self.num_servers = len(server_handles)
        self._server_to_idx = {s: i for i, s in enumerate(server_handles)}

        self._request_to_server: dict = {}
        self._sticky_cache: LRUCache = LRUCache(maxsize=10000)
        self._rr_idx = 0

        self.submitted_per_server = [0] * self.num_servers
        self.runahead_inflight_per_server = [0] * self.num_servers

        self.monitors = [WorkloadMonitor(h, i) for i, h in enumerate(server_handles)]
        self._cached_workloads: list = [None] * self.num_servers
        self._workload_cache_time: float = 0.0

        # Metrics
        self.total_requests = 0
        self.runahead_submitted = 0
        self.runahead_completed = 0
        self.runahead_aborted = 0
        self.backpressure_rejections = 0

    async def get_all_workloads(self, force_refresh: bool = False) -> list:
        now = time.perf_counter()
        jitter = random.uniform(0, self.slack_config.poll_jitter_s)
        cache_ttl = self.slack_config.workload_cache_ttl_s + jitter

        if not force_refresh and (now - self._workload_cache_time < cache_ttl):
            result = []
            for i, w in enumerate(self._cached_workloads):
                if w is not None:
                    result.append(w)
                else:
                    result.append(WorkloadSnapshot(
                        timestamp=now, server_idx=i, error="Missing cached workload"
                    ))
            return result

        tasks = [m.get_workload() for m in self.monitors]
        workloads = await asyncio.gather(*tasks)
        self._cached_workloads = list(workloads)
        self._workload_cache_time = now
        return workloads

    def _choose_server_primary(self, request_id: str, sticky: bool = True):
        if sticky and request_id in self._sticky_cache:
            return self._sticky_cache[request_id]

        idx = self._rr_idx % self.num_servers
        self._rr_idx += 1
        server = self.server_handles[idx]
        result = (server, idx)
        if sticky:
            self._sticky_cache[request_id] = result
        return result

    def can_submit_runahead(self, server_idx: int) -> bool:
        return self.runahead_inflight_per_server[server_idx] < self.slack_config.budget_per_server

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list,
        sampling_params: dict,
        image_data=None,
        tracker: Optional[RequestTracker] = None,
        sticky: bool = True,
        kind: str = "primary",
        priority: Optional[int] = None,
        preferred_server_idx: Optional[int] = None,
    ):
        cfg = self.slack_config

        if priority is None:
            priority = cfg.primary_priority if kind == "primary" else cfg.runahead_priority

        if preferred_server_idx is not None:
            server = self.server_handles[preferred_server_idx]
            server_idx = preferred_server_idx
            if kind == "runahead" and not self.can_submit_runahead(server_idx):
                self.backpressure_rejections += 1
                if tracker:
                    tracker.status = "rejected"
                return None
        elif kind == "primary":
            server, server_idx = self._choose_server_primary(request_id, sticky=sticky)
        else:
            workloads = await self.get_all_workloads()
            slack_servers = [
                w for w in workloads
                if w.has_slack(cfg) and self.can_submit_runahead(w.server_idx)
            ]
            if not slack_servers:
                self.backpressure_rejections += 1
                if tracker:
                    tracker.status = "rejected"
                return None
            best = min(
                slack_servers,
                key=lambda w: (w.total_load + self.runahead_inflight_per_server[w.server_idx], w.server_idx)
            )
            server_idx = best.server_idx
            server = self.server_handles[server_idx]

        server_request_id = uuid4().hex
        self._request_to_server[server_request_id] = server_idx

        self.submitted_per_server[server_idx] += 1
        self.total_requests += 1
        if kind == "runahead":
            self.runahead_inflight_per_server[server_idx] += 1
            self.runahead_submitted += 1

        if tracker:
            tracker.server_request_id = server_request_id
            tracker.server_idx = server_idx
            tracker.start_time = time.perf_counter()
            tracker.status = "running"
            tracker.priority = priority

        try:
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )

            stop_reason = getattr(output, "stop_reason", None)
            if tracker:
                tracker.end_time = time.perf_counter()
                tracker.status = "aborted" if stop_reason == "aborted" else "completed"
                tracker.stop_reason = stop_reason
                tracker.token_count = len(getattr(output, "token_ids", []))
                tracker.token_ids = list(getattr(output, "token_ids", []))

            if kind == "runahead":
                if stop_reason == "aborted":
                    self.runahead_aborted += 1
                else:
                    self.runahead_completed += 1

            return output

        except asyncio.CancelledError:
            if server_request_id in self._request_to_server:
                try:
                    await asyncio.shield(
                        server.abort_request.remote(server_request_id, reset_prefix_cache=False)
                    )
                except BaseException:
                    try:
                        await asyncio.shield(server.abort_request.remote(server_request_id))
                    except BaseException:
                        pass
                if kind == "runahead":
                    self.runahead_aborted += 1
            if tracker:
                tracker.status = "aborted"
                tracker.end_time = time.perf_counter()
            raise

        except Exception:
            if tracker:
                tracker.status = "error"
                tracker.end_time = time.perf_counter()
            raise

        finally:
            self.submitted_per_server[server_idx] -= 1
            if kind == "runahead":
                self.runahead_inflight_per_server[server_idx] -= 1
            self._request_to_server.pop(server_request_id, None)


# =============================================================================
# Runahead Controller
# =============================================================================

class RunaheadController:
    """Controller for runahead with continuous slack-filling."""

    def __init__(self, server_manager: BenchmarkServerManager, config: SlackFillingConfig):
        self.sm = server_manager
        self.config = config
        self.primary_start_time: Optional[float] = None
        self.primary_done_time: Optional[float] = None
        self.feeder_ticks = 0
        self.runahead_submissions = 0
        self.backpressure_events = 0
        self.slack_checks = 0

    @property
    def primary_duration(self) -> Optional[float]:
        if self.primary_start_time is None or self.primary_done_time is None:
            return None
        return self.primary_done_time - self.primary_start_time

    async def run_with_runahead(
        self,
        *,
        primary_items: list,
        runahead_items: list,
        primary_tracker: BatchTracker,
        runahead_tracker: BatchTracker,
        tokenizer,
    ) -> tuple:
        from collections import deque
        cfg = self.config
        primary_tasks = set()
        primary_results = []
        runahead_results = []

        self.primary_start_time = time.perf_counter()
        primary_tracker.start_time = self.primary_start_time

        # Launch all primary immediately
        for i, item in enumerate(primary_items):
            rid = item["request_id"]
            tr = RequestTracker(
                request_id=rid,
                batch_id="primary",
                index=i,
                max_tokens=item["max_tokens"],
            )
            primary_tracker.requests[rid] = tr

            task = asyncio.create_task(
                self._generate_prompt(
                    item["prompt"],
                    request_id=rid,
                    max_tokens=item["max_tokens"],
                    tracker=tr,
                    kind="primary",
                    sticky=True,
                    tokenizer=tokenizer,
                )
            )
            primary_tasks.add(task)

        # Prepare runahead queue
        runahead_queue = deque((i, item) for i, item in enumerate(runahead_items))
        runahead_tasks: set = set()

        async def maybe_submit_runahead():
            nonlocal runahead_queue
            if not runahead_queue:
                return

            self.feeder_ticks += 1
            workloads = await self.sm.get_all_workloads()
            self.slack_checks += 1

            for w in workloads:
                if not runahead_queue:
                    break
                if not w.has_slack(cfg):
                    continue
                if not self.sm.can_submit_runahead(w.server_idx):
                    continue

                idx, item = runahead_queue.popleft()
                rid = item.get("request_id") or f"runahead_{uuid4().hex[:8]}"

                tr = RequestTracker(
                    request_id=rid,
                    batch_id="runahead",
                    index=idx,
                    max_tokens=item["max_tokens"],
                )
                runahead_tracker.requests[rid] = tr

                task = asyncio.create_task(
                    self._generate_prompt(
                        item["prompt"],
                        request_id=rid,
                        max_tokens=item["max_tokens"],
                        tracker=tr,
                        kind="runahead",
                        sticky=False,
                        preferred_server_idx=w.server_idx,
                        tokenizer=tokenizer,
                    )
                )
                runahead_tasks.add(task)
                self.runahead_submissions += 1

            if runahead_queue and not any(
                w.has_slack(cfg) and self.sm.can_submit_runahead(w.server_idx)
                for w in workloads
            ):
                self.backpressure_events += 1

        def collect_done_runahead():
            nonlocal runahead_tasks
            done = {t for t in runahead_tasks if t.done()}
            for t in done:
                runahead_tasks.remove(t)
                try:
                    result = t.result()
                    runahead_results.append(result)
                except Exception as e:
                    runahead_results.append({"error": str(e)})

        # Main loop
        while primary_tasks:
            jitter = random.uniform(0, cfg.poll_jitter_s)
            timeout = cfg.poll_interval_s + jitter

            done, primary_tasks = await asyncio.wait(
                primary_tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for t in done:
                try:
                    result = await t
                    primary_results.append(result)
                except Exception as e:
                    primary_results.append({"error": str(e)})

            collect_done_runahead()
            await maybe_submit_runahead()

        self.primary_done_time = time.perf_counter()

        # Abort remaining runahead requests (don't cancel - let them return partial tokens)
        if runahead_tasks:
            # First, tell vLLM to abort all running runahead requests
            # This will cause them to return quickly with partial output
            running_requests = [
                req for req in runahead_tracker.requests.values()
                if req.status == "running" and req.server_request_id
            ]

            abort_tasks = []
            for req in running_requests:
                server = self.sm.server_handles[req.server_idx]
                abort_tasks.append(
                    server.abort_request.remote(req.server_request_id, reset_prefix_cache=False)
                )

            # Wait for aborts to be processed
            if abort_tasks:
                await asyncio.gather(*abort_tasks, return_exceptions=True)

            # Now wait for all runahead tasks to complete (they should return quickly with partial output)
            gathered_results = await asyncio.gather(*runahead_tasks, return_exceptions=True)
            for result in gathered_results:
                if isinstance(result, Exception):
                    runahead_results.append({"error": str(result)})
                else:
                    runahead_results.append(result)

        return primary_results, runahead_results

    async def _generate_prompt(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        tracker: Optional[RequestTracker] = None,
        kind: str = "primary",
        sticky: bool = True,
        preferred_server_idx: Optional[int] = None,
        tokenizer,
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        return await self.sm.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            tracker=tracker,
            sticky=sticky,
            kind=kind,
            preferred_server_idx=preferred_server_idx,
        )


# =============================================================================
# Workload Generation
# =============================================================================

def generate_workload(
    size: int,
    long_tail_ratio: float,
    short_max_tokens: int,
    long_max_tokens: int,
    prefix: str = "primary"
) -> List[dict]:
    """Generate workload with long-tail distribution."""
    HARD_MATH_PROBLEM = """In triangle ABC, sin(angle A) = 4/5 and angle A < 90 degrees. Let D be a point outside triangle ABC such that angle BAD = angle DAC and angle BDC = 90 degrees. Suppose that AD = 1 and that BD/CD = 3/2. If AB + AC can be expressed in the form (a*sqrt(b))/c where a, b, c are pairwise relatively prime integers, find a + b + c. Show your complete step-by-step solution with all mathematical reasoning."""

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

    # Long prompts at the end (to simulate realistic scenarios)
    for i in range(num_long):
        prompts.append({
            "request_id": f"{prefix}_{num_short + i}_{uuid4().hex[:8]}",
            "prompt": HARD_MATH_PROBLEM,
            "max_tokens": long_max_tokens,
            "is_long": True,
        })

    # Shuffle to distribute long tasks randomly
    random.shuffle(prompts)

    return prompts


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs benchmark experiments."""

    def __init__(self, exp_config: ExperimentConfig):
        self.exp_config = exp_config
        self.servers = []
        self.server_handles = []
        self.tokenizer = None

    def setup(self):
        """Initialize Ray and vLLM servers."""
        import ray
        from hydra import compose, initialize_config_dir

        print("=" * 80)
        print("RUNAHEAD TRADEOFF BENCHMARK")
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
        config.actor_rollout_ref.rollout.enable_prefix_caching = False  # Disable for fair baseline vs runahead comparison

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
        import ray
        print("\nShutting down Ray...")
        ray.shutdown()

    async def run_baseline(
        self,
        primary_prompts: list,
        slack_config: SlackFillingConfig,
    ) -> RunMetrics:
        """Run baseline (primary only, no runahead)."""
        print("\n   Running BASELINE (no runahead)...")

        sm = BenchmarkServerManager(
            self.exp_config, self.server_handles, slack_config
        )
        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))

        primary_tasks = set()
        primary_results = []
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

        # Wait for all primary to complete
        while primary_tasks:
            done, primary_tasks = await asyncio.wait(
                primary_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                try:
                    result = await t
                    primary_results.append(result)
                except Exception as e:
                    primary_results.append({"error": str(e)})

        end_time = time.perf_counter()

        return RunMetrics(
            primary_time=end_time - start_time,
            primary_tokens=primary_tracker.total_tokens,
            primary_completed=primary_tracker.completed,
        )

    async def run_runahead(
        self,
        primary_prompts: list,
        runahead_prompts: list,
        slack_config: SlackFillingConfig,
    ) -> RunMetrics:
        """Run with runahead enabled."""
        print("\n   Running WITH RUNAHEAD...")

        sm = BenchmarkServerManager(
            self.exp_config, self.server_handles, slack_config
        )
        controller = RunaheadController(sm, slack_config)

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        start_time = time.perf_counter()
        primary_results, runahead_results = await controller.run_with_runahead(
            primary_items=primary_prompts,
            runahead_items=runahead_prompts,
            primary_tracker=primary_tracker,
            runahead_tracker=runahead_tracker,
            tokenizer=self.tokenizer,
        )
        end_time = time.perf_counter()

        # Calculate runahead tokens
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
            primary_tokens=primary_tracker.total_tokens,
            primary_completed=primary_tracker.completed,
            runahead_tokens_total=runahead_tokens_completed + runahead_tokens_aborted,
            runahead_tokens_completed=runahead_tokens_completed,
            runahead_tokens_aborted=runahead_tokens_aborted,
            runahead_completed_count=runahead_tracker.completed,
            runahead_aborted_count=runahead_tracker.aborted,
            backpressure_events=controller.backpressure_events,
            feeder_ticks=controller.feeder_ticks,
        )

    def run_single_experiment(
        self,
        primary_size: int,
        load_threshold: int,
        long_tail_ratio: Optional[float] = None,
    ) -> ExperimentResult:
        """Run a single experiment (baseline + runahead)."""
        # Use provided ratio or fall back to config default
        ratio = long_tail_ratio if long_tail_ratio is not None else self.exp_config.long_tail_ratio

        print(f"\n{'='*80}")
        print(f"EXPERIMENT: PRIMARY_SIZE={primary_size}, LOAD_THRESHOLD={load_threshold}, LONG_TAIL_RATIO={ratio:.0%}")
        print(f"{'='*80}")

        slack_config = SlackFillingConfig(
            load_threshold=load_threshold,
            kv_cache_threshold=self.exp_config.kv_cache_threshold,
            budget_per_server=self.exp_config.budget_per_server,
            poll_interval_s=self.exp_config.poll_interval_s,
        )

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
        baseline_metrics = asyncio.run(
            self.run_baseline(primary_prompts, slack_config)
        )
        print(f"   Baseline: {baseline_metrics.primary_time:.2f}s, "
              f"{baseline_metrics.primary_tokens} tokens")

        # Run runahead
        runahead_metrics = asyncio.run(
            self.run_runahead(primary_prompts, runahead_prompts, slack_config)
        )
        print(f"   Runahead: {runahead_metrics.primary_time:.2f}s, "
              f"primary={runahead_metrics.primary_tokens} tokens, "
              f"runahead={runahead_metrics.runahead_tokens_total} tokens")

        # Calculate overhead
        overhead_pct = 0.0
        if baseline_metrics.primary_time > 0:
            overhead_pct = (
                (runahead_metrics.primary_time - baseline_metrics.primary_time)
                / baseline_metrics.primary_time * 100
            )
        print(f"   Overhead: {overhead_pct:+.2f}%")
        print(f"   Runahead completed: {runahead_metrics.runahead_completed_count}, "
              f"aborted: {runahead_metrics.runahead_aborted_count}")

        return ExperimentResult(
            experiment_id=f"exp_{primary_size}_{load_threshold}_{int(ratio*100)}pct_{uuid4().hex[:8]}",
            config={
                "primary_size": primary_size,
                "load_threshold": load_threshold,
                "budget_per_server": self.exp_config.budget_per_server,
                "long_tail_ratio": ratio,
                "short_max_tokens": self.exp_config.short_max_tokens,
                "long_max_tokens": self.exp_config.long_max_tokens,
                "num_gpus": self.exp_config.num_gpus,
            },
            baseline=baseline_metrics,
            runahead=runahead_metrics,
            primary_overhead_pct=overhead_pct,
        )


# =============================================================================
# Main
# =============================================================================

def run_experiment_matrix(runner: BenchmarkRunner, num_rounds: int = 1) -> list:
    """Run full experiment matrix with multiple rounds."""
    # v7: Qwen3-8B model - smaller batches due to larger model
    primary_sizes = [16, 32, 64]
    load_thresholds = [8, 16]
    long_tail_ratios = [0.20, 0.40, 0.50]  # focused sweep

    results = []
    total = len(primary_sizes) * len(load_thresholds) * len(long_tail_ratios) * num_rounds
    current = 0

    for round_idx in range(num_rounds):
        for primary_size in primary_sizes:
            for load_threshold in load_thresholds:
                for long_tail_ratio in long_tail_ratios:
                    current += 1
                    print(f"\n\n{'#'*80}")
                    print(f"# EXPERIMENT {current}/{total} (Round {round_idx + 1}/{num_rounds})")
                    print(f"{'#'*80}")

                    result = runner.run_single_experiment(
                        primary_size, load_threshold, long_tail_ratio=long_tail_ratio
                    )
                    # Add round info to the result
                    result.config["round"] = round_idx + 1
                    results.append(result)

    return results


def print_summary_table(results: list):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)

    print(f"\n{'PRIMARY_SIZE':<12} | {'LOAD_THRESH':<11} | {'BASELINE(s)':<11} | "
          f"{'RUNAHEAD(s)':<11} | {'OVERHEAD%':<10} | {'RA_TOKENS':<10} | {'RA_COMPLETE':<11}")
    print("-" * 100)

    for r in results:
        print(f"{r.config['primary_size']:<12} | "
              f"{r.config['load_threshold']:<11} | "
              f"{r.baseline.primary_time:<11.2f} | "
              f"{r.runahead.primary_time:<11.2f} | "
              f"{r.primary_overhead_pct:<+10.2f} | "
              f"{r.runahead.runahead_tokens_total:<10} | "
              f"{r.runahead.runahead_completed_count:<11}")

    print("=" * 100)


def save_results(results: list, output_dir: str = "results"):
    """Save results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/runahead_experiment_{timestamp}.json"

    # Convert to serializable format
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
            "runahead": asdict(r.runahead),
            "primary_overhead_pct": r.primary_overhead_pct,
            "timestamp": r.timestamp,
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def print_averaged_summary(results: list, num_rounds: int):
    """Print averaged summary across rounds."""
    if num_rounds <= 1:
        return

    print("\n" + "=" * 150)
    print(f"AVERAGED SUMMARY ({num_rounds} rounds)")
    print("=" * 150)

    # Group results by (primary_size, load_threshold, long_tail_ratio)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r.config["primary_size"], r.config["load_threshold"], r.config["long_tail_ratio"])
        grouped[key].append(r)

    print(f"\n{'PRIMARY':<8} | {'THRESH':<6} | {'RATIO':<6} | {'BASELINE(s)':<11} | "
          f"{'RUNAHEAD(s)':<11} | {'OVERHEAD%':<10} | {'RA_TOKENS':<10} | {'RA_COMPLETED':<12} | {'RA_ABORTED':<10}")
    print("-" * 150)

    for key in sorted(grouped.keys()):
        runs = grouped[key]
        n = len(runs)

        avg_baseline = sum(r.baseline.primary_time for r in runs) / n
        avg_runahead = sum(r.runahead.primary_time for r in runs) / n
        avg_overhead = sum(r.primary_overhead_pct for r in runs) / n
        avg_ra_total = sum(r.runahead.runahead_tokens_total for r in runs) / n
        avg_ra_completed = sum(r.runahead.runahead_tokens_completed for r in runs) / n
        avg_ra_aborted = sum(r.runahead.runahead_tokens_aborted for r in runs) / n

        print(f"{key[0]:<8} | {key[1]:<6} | {key[2]*100:5.0f}% | {avg_baseline:11.2f} | "
              f"{avg_runahead:11.2f} | {avg_overhead:9.2f}% | {avg_ra_total:10.0f} | "
              f"{avg_ra_completed:12.0f} | {avg_ra_aborted:10.0f}")

    print("=" * 150)


def main():
    parser = argparse.ArgumentParser(description="Runahead Tradeoff Benchmark")
    parser.add_argument("--single", action="store_true", help="Run single config from env vars")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to run (default: 1)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    args = parser.parse_args()

    # Build config from environment
    exp_config = ExperimentConfig(
        primary_size=int(os.environ.get("PRIMARY_SIZE", "32")),
        runahead_size=int(os.environ.get("RUNAHEAD_SIZE", os.environ.get("PRIMARY_SIZE", "32"))),
        long_tail_ratio=float(os.environ.get("LONG_TAIL_RATIO", "0.20")),
        load_threshold=int(os.environ.get("LOAD_THRESHOLD", "8")),
        budget_per_server=int(os.environ.get("BUDGET_PER_SERVER", "16")),
        kv_cache_threshold=float(os.environ.get("KV_CACHE_THRESHOLD", "0.90")),
        poll_interval_s=float(os.environ.get("POLL_INTERVAL", "0.05")),
        short_max_tokens=int(os.environ.get("SHORT_MAX_TOKENS", "2048")),
        long_max_tokens=int(os.environ.get("LONG_MAX_TOKENS", "16384")),
        num_gpus=int(os.environ.get("NUM_GPUS", "2")),
        tp_size=int(os.environ.get("TP_SIZE", "1")),
        model_path=os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B"),
    )

    runner = BenchmarkRunner(exp_config)

    try:
        runner.setup()

        if args.single:
            # Run single experiment from env vars
            result = runner.run_single_experiment(
                primary_size=exp_config.primary_size,
                load_threshold=exp_config.load_threshold,
            )
            results = [result]
        else:
            # Run full matrix with specified rounds
            results = run_experiment_matrix(runner, num_rounds=args.rounds)

        print_summary_table(results)
        print_averaged_summary(results, args.rounds)
        save_results(results, args.output_dir)

    finally:
        runner.teardown()


if __name__ == "__main__":
    main()
