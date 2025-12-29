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
Slack-Filling Runahead Implementation

This implements "continuous slack-filling with backpressure" as an improvement
over the one-shot trigger approach in test_vllm_runahead_agentloop_standalone.py.

Key improvements over the original design:
1. Continuous slack-filling: drip-feed runahead instead of batch trigger
2. Backpressure: only submit when num_requests_waiting <= W and kv_cache <= K
3. Per-server budget: cap runahead in-flight per server (not global)
4. Priority scheduling: config-only placeholder (TODO: wire into server)
5. Safe cancellation: don't lose abort mapping on asyncio.CancelledError

Mental model:
- Primary = "must-run" traffic (submitted immediately)
- Runahead = "nice-to-have" traffic (only when servers have slack)
- Slack = unused capacity (detected via num_requests_waiting, kv_cache)
- Backpressure = stop feeding when server is busy

Usage:
    python tests/workers/rollout/rollout_vllm/test_vllm_runahead_slack_filling.py

    NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/test_vllm_runahead_slack_filling.py
"""

import asyncio
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from cachetools import LRUCache

from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase, AsyncLLMServerManager


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SlackFillingConfig:
    """Configuration for slack-filling runahead.

    Backpressure rule: only submit runahead if:
        num_requests_waiting <= waiting_threshold (W)
        AND kv_cache_usage <= kv_cache_threshold (K)

    Budget rule: max budget_per_server runahead requests in-flight per server.

    Note: This is per-worker budget, not global. With N workers, you can have
    up to N * budget_per_server runahead per server. For strict global limits,
    you need server-side admission control or a central coordinator.
    """

    # Backpressure thresholds
    waiting_threshold: int = 0  # W: only submit if num_requests_waiting <= W
    kv_cache_threshold: float = 0.85  # K: only submit if kv_cache_usage <= K

    # Budget limits (per-worker, not global)
    budget_per_server: int = 1  # Max runahead in-flight per server (start conservative)

    # Polling settings
    poll_interval_s: float = 0.1  # How often to check slack (100ms)
    poll_jitter_s: float = 0.03  # Jitter to reduce herding across workers

    # Workload cache
    workload_cache_ttl_s: float = 0.3  # Cache workload queries (300ms)

    # Priority (TODO: requires server support)
    primary_priority: int = 0  # Lower = higher priority
    runahead_priority: int = 10  # Higher = lower priority


# =============================================================================
# Workload Monitoring (same as original, with minor improvements)
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

    def __str__(self) -> str:
        if self.error:
            return f"WorkloadSnapshot(server={self.server_idx}, error={self.error})"
        return (
            f"WorkloadSnapshot(server={self.server_idx}, running={self.num_requests_running}, "
            f"waiting={self.num_requests_waiting}, kv_cache={self.kv_cache_usage:.2%})"
        )

    @property
    def total_load(self) -> int:
        """Total load = running + waiting requests."""
        return self.num_requests_running + self.num_requests_waiting

    def has_slack(self, config: SlackFillingConfig) -> bool:
        """Check if this server has slack for runahead."""
        if self.error:
            return False
        return (
            self.num_requests_waiting <= config.waiting_threshold
            and self.kv_cache_usage <= config.kv_cache_threshold
        )


class WorkloadMonitor:
    """Monitor vLLM server workload via Prometheus metrics."""

    def __init__(self, server_handle, server_idx: int):
        self.server_handle = server_handle
        self.server_idx = server_idx
        self._warned_missing_metrics = False

    async def get_workload(self) -> WorkloadSnapshot:
        """Fetch current workload from server."""
        try:
            result = await self.server_handle.get_workload.remote()

            warning = result.get("warning")
            if warning and not self._warned_missing_metrics:
                self._warned_missing_metrics = True
                print(f"\n   WARNING: Server {self.server_idx}: {warning}")

            # If metrics are missing, be conservative: treat as "no slack" by setting error.
            missing = []
            for key in ("num_requests_running", "num_requests_waiting", "kv_cache_usage"):
                if key not in result:
                    missing.append(key)

            error = result.get("error")
            if error:
                error_msg = str(error)
            elif warning:
                error_msg = f"metrics_unavailable: {warning}"
            elif missing:
                error_msg = f"metrics_incomplete: missing {','.join(missing)}"
            else:
                error_msg = None

            return WorkloadSnapshot(
                timestamp=time.perf_counter(),
                server_idx=self.server_idx,
                num_requests_running=result.get("num_requests_running", 0),
                num_requests_waiting=result.get("num_requests_waiting", 0),
                kv_cache_usage=result.get("kv_cache_usage", 0.0),
                error=error_msg,
            )
        except Exception as e:
            return WorkloadSnapshot(timestamp=time.perf_counter(), server_idx=self.server_idx, error=str(e))


# =============================================================================
# Request Tracking (same as original)
# =============================================================================


@dataclass
class RequestTracker:
    """Track individual request state, timing, and server_request_id.

    Important distinction:
    - `status`: Logical state for our tracking (completed/aborted/error/rejected/running/pending)
    - `stop_reason`: Backend-reported reason (stop/length/eos/etc.) - informational only
    """

    request_id: str
    server_request_id: str = ""
    batch_id: str = ""
    index: int = 0  # Must be set by caller when creating tracker
    max_tokens: int = 0
    server_idx: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    token_count: int = 0
    status: str = "pending"  # completed | running | aborted | error | rejected | pending
    stop_reason: Optional[str] = None  # Backend-reported: stop | length | eos | etc.
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
    requests: dict[str, RequestTracker] = field(default_factory=dict)
    start_time: float = 0.0

    @property
    def completed(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "completed")

    @property
    def aborted(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "aborted")

    @property
    def running(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "running")

    @property
    def completion_ratio(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0

    def get_running_server_request_ids(self) -> list[str]:
        return [r.server_request_id for r in self.requests.values() if r.status == "running" and r.server_request_id]


# =============================================================================
# Slack-Filling Server Manager
# =============================================================================


class SlackFillingServerManager(AsyncLLMServerManager):
    """AsyncLLMServerManager with slack-filling, backpressure, and priority support.

    Key differences from RunaheadAsyncLLMServerManager:
    1. Per-server budget tracking and enforcement
    2. Priority field in generate() (TODO: needs server support)
    3. Safe cancellation handling with asyncio.shield

    Routing modes:
    - Primary: LRU-based sticky sessions (inherited)
    - Runahead with preferred_server_idx: Direct to specified server (budget checked)
    - Runahead without preferred_server_idx: Pick lowest-load slack server

    Note: The SlackFillingRunaheadController always specifies preferred_server_idx
    based on its own slack assessment.
    """

    def __init__(
        self,
        config,
        server_handles: list,
        slack_config: SlackFillingConfig,
        max_cache_size: int = 10000,
    ):
        super().__init__(config, server_handles, max_cache_size=max_cache_size)
        self.slack_config = slack_config
        self.num_servers = len(self.server_handles)
        self._server_to_idx = {server: idx for idx, server in enumerate(self.server_handles)}

        # Request tracking for targeted abort
        # IMPORTANT: We keep this mapping until server confirms completion/abort
        self._request_to_server: dict[str, int] = {}
        self._sticky_cache: LRUCache = LRUCache(maxsize=max_cache_size)

        # Per-server tracking
        self.submitted_per_server = [0] * self.num_servers
        self.runahead_inflight_per_server = [0] * self.num_servers  # NEW: track runahead separately

        # Workload monitoring with caching
        self.monitors = [WorkloadMonitor(h, idx) for idx, h in enumerate(server_handles)]
        self._cached_workloads: list[Optional[WorkloadSnapshot]] = [None] * self.num_servers
        self._workload_cache_time: float = 0.0

        # Metrics
        self.total_requests = 0
        self.runahead_submitted = 0
        self.runahead_completed = 0
        self.runahead_aborted = 0
        self.backpressure_rejections = 0

    async def get_all_workloads(self, force_refresh: bool = False) -> list[WorkloadSnapshot]:
        """Fetch workload from all servers (with caching + jitter).

        Returns a list of WorkloadSnapshot aligned by server_idx (index = server_idx).
        Never returns None entries - if a server has an error, the snapshot will have
        error field set but will still be present at the correct index.
        """
        now = time.perf_counter()

        # Add jitter to cache TTL to reduce herding
        jitter = random.uniform(0, self.slack_config.poll_jitter_s)
        cache_ttl = self.slack_config.workload_cache_ttl_s + jitter

        if not force_refresh and (now - self._workload_cache_time < cache_ttl):
            # FIXED: Return full list aligned by server_idx, not filtering out None
            # If any entry is None (shouldn't happen), create an error snapshot
            result = []
            for i, w in enumerate(self._cached_workloads):
                if w is not None:
                    result.append(w)
                else:
                    result.append(WorkloadSnapshot(
                        timestamp=now, server_idx=i, error="Missing cached workload"
                    ))
            return result

        # Fetch fresh workloads in parallel
        tasks = [m.get_workload() for m in self.monitors]
        workloads = await asyncio.gather(*tasks)
        self._cached_workloads = list(workloads)
        self._workload_cache_time = now
        return workloads

    def _choose_server_primary(self, request_id: str, sticky: bool = True):
        """Use inherited LRU-based sticky session for primary requests."""
        if sticky and request_id in self._sticky_cache:
            return self._sticky_cache[request_id]

        server = super()._choose_server(request_id)
        server_idx = self._server_to_idx[server]
        result = (server, server_idx)
        if sticky:
            self._sticky_cache[request_id] = result
        return result

    def can_submit_runahead(self, server_idx: int) -> bool:
        """Check if we can submit another runahead to this server (budget check)."""
        return self.runahead_inflight_per_server[server_idx] < self.slack_config.budget_per_server

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        tracker: Optional[RequestTracker] = None,
        sticky: bool = True,
        kind: str = "primary",
        priority: Optional[int] = None,
        preferred_server_idx: Optional[int] = None,  # NEW: for directed runahead
    ):
        """Generate with kind-specific routing and priority support."""
        cfg = self.slack_config

        # Determine priority
        if priority is None:
            priority = cfg.primary_priority if kind == "primary" else cfg.runahead_priority

        # Choose server
        if preferred_server_idx is not None:
            server = self.server_handles[preferred_server_idx]
            server_idx = preferred_server_idx
            # FIXED: Enforce budget even with preferred_server_idx
            # This ensures budget is an invariant regardless of caller behavior
            if kind == "runahead" and not self.can_submit_runahead(server_idx):
                self.backpressure_rejections += 1
                if tracker:
                    tracker.status = "rejected"
                return None  # Budget exceeded
        elif kind == "primary":
            server, server_idx = self._choose_server_primary(request_id, sticky=sticky)
        else:
            # For runahead without preferred server, pick the lowest-load slack server.
            workloads = await self.get_all_workloads()
            slack_servers = [
                w for w in workloads if w.has_slack(cfg) and self.can_submit_runahead(w.server_idx)
            ]
            if not slack_servers:
                self.backpressure_rejections += 1
                if tracker:
                    tracker.status = "rejected"
                return None  # No slack available
            best = min(
                slack_servers,
                key=lambda w: (w.total_load + self.runahead_inflight_per_server[w.server_idx], w.server_idx),
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
            # TODO: Priority plumbing - priority should be passed as separate argument
            # to server.generate.remote(), not stuffed into sampling_params.
            # Current approach may fail if server doesn't expect priority in sampling_params.
            # Proper fix: update vLLMHttpServer.generate() to accept priority parameter
            # and route it to engine.add_request(..., priority=priority)
            #
            # For now, we pass priority as a separate key that server should pop out
            # before constructing SamplingParams, or ignore if not supported.
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                # priority=priority,  # TODO: Enable when server supports it
            )

            stop_reason = getattr(output, "stop_reason", None)
            if tracker:
                tracker.end_time = time.perf_counter()
                # FIXED: Separate status (our logical state) from stop_reason (backend info)
                tracker.status = "aborted" if stop_reason == "aborted" else "completed"
                tracker.stop_reason = stop_reason  # Backend detail
                tracker.token_count = len(getattr(output, "token_ids", []))
                tracker.token_ids = list(getattr(output, "token_ids", []))

            if kind == "runahead":
                if stop_reason == "aborted":
                    self.runahead_aborted += 1
                else:
                    self.runahead_completed += 1

            return output

        except asyncio.CancelledError:
            # IMPORTANT: Safe cancellation - abort the server request before re-raising
            # Use asyncio.shield to prevent nested cancellation during abort
            # Catch BaseException (not Exception) because CancelledError is BaseException in Python 3.11+
            if server_request_id in self._request_to_server:
                try:
                    # Avoid resetting global prefix cache when aborting runahead.
                    await asyncio.shield(server.abort_request.remote(server_request_id, reset_prefix_cache=False))
                except BaseException:
                    try:
                        await asyncio.shield(server.abort_request.remote(server_request_id))
                    except BaseException:
                        pass  # Best effort - don't let abort failure prevent re-raise
                if kind == "runahead":
                    self.runahead_aborted += 1
            if tracker:
                tracker.status = "aborted"
                tracker.end_time = time.perf_counter()
            raise  # Re-raise CancelledError

        except Exception:
            if tracker:
                tracker.status = "error"
                tracker.end_time = time.perf_counter()
            raise

        finally:
            self.submitted_per_server[server_idx] -= 1
            if kind == "runahead":
                self.runahead_inflight_per_server[server_idx] -= 1
            # Only remove mapping after we know server is done
            self._request_to_server.pop(server_request_id, None)

    async def abort_requests(self, server_request_ids: list[str]) -> dict[str, Any]:
        """Targeted abort by server_request_id (multi-server safe)."""
        if not server_request_ids:
            return {"aborted_count": 0, "request_ids": []}

        by_server: dict[int, list[str]] = {}
        for rid in server_request_ids:
            server_idx = self._request_to_server.get(rid)
            if server_idx is not None:
                by_server.setdefault(server_idx, []).append(rid)

        async def abort_on_server(server_idx: int, ids: list[str]) -> dict:
            try:
                server = self.server_handles[server_idx]
                results = []
                for rid in ids:
                    try:
                        result = await server.abort_request.remote(rid, reset_prefix_cache=False)
                    except Exception as e:
                        msg = str(e)
                        if "reset_prefix_cache" in msg and "unexpected keyword argument" in msg:
                            result = await server.abort_request.remote(rid)
                        else:
                            raise
                    results.append(result)
                # FIXED: Only count actual aborts (when server confirms it aborted the request)
                aborted = sum(1 for r in results if r.get("aborted", False))
                return {"server_idx": server_idx, "aborted_count": aborted, "request_ids": ids}
            except Exception as e:
                return {"server_idx": server_idx, "error": str(e), "aborted_count": 0}

        tasks = [abort_on_server(idx, ids) for idx, ids in by_server.items()]
        results = await asyncio.gather(*tasks) if tasks else []

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_ids = [rid for r in results for rid in r.get("request_ids", [])]

        return {
            "aborted_count": total_aborted,
            "request_ids": all_ids,
            "per_server_results": results,
        }


# =============================================================================
# Slack-Filling Runahead Controller (Budgeted Feeder)
# =============================================================================


class SlackFillingRunaheadController:
    """Runahead controller using continuous slack-filling with backpressure.

    Instead of a one-shot trigger, this controller:
    1. Maintains a queue of runahead prompts
    2. Periodically checks server slack
    3. Submits runahead only when servers have slack AND budget allows
    4. Stops feeding automatically when servers get busy (backpressure)

    This is the "budgeted feeder" pattern.
    """

    def __init__(
        self,
        server_manager: SlackFillingServerManager,
        config: SlackFillingConfig,
    ):
        self.sm = server_manager
        self.config = config

        # State
        self.primary_start_time: Optional[float] = None
        self.primary_done_time: Optional[float] = None

        # Metrics
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
        primary_items: list[dict],  # {"request_id", "prompt", "max_tokens"}
        runahead_items: list[dict],  # same shape
        primary_tracker: BatchTracker,
        runahead_tracker: BatchTracker,
        worker,  # RunaheadAgentLoopWorker for tokenization
    ) -> tuple[list, list]:
        """
        Run primary batch with continuous slack-filling runahead.

        Unlike the one-shot trigger approach:
        - Runahead is fed continuously whenever servers have slack
        - Backpressure prevents overloading servers
        - Per-server budgets limit runahead in-flight
        """
        cfg = self.config
        primary_tasks = set()
        primary_results = []
        runahead_results = []

        self.primary_start_time = time.perf_counter()
        primary_tracker.start_time = self.primary_start_time

        # --- Launch all primary immediately ---
        for i, item in enumerate(primary_items):
            rid = item["request_id"]
            tr = RequestTracker(
                request_id=rid,
                batch_id="primary",
                index=i,  # FIXED: Set index for proper ordering in output
                max_tokens=item["max_tokens"],
            )
            primary_tracker.requests[rid] = tr

            task = asyncio.create_task(
                worker.generate_prompt(
                    item["prompt"],
                    request_id=rid,
                    max_tokens=item["max_tokens"],
                    tracker=tr,
                    kind="primary",
                    sticky=True,
                )
            )
            primary_tasks.add(task)

        # --- Prepare runahead queue with index tracking ---
        # Store (index, item) tuples so we can set index when creating trackers
        runahead_queue = deque((i, item) for i, item in enumerate(runahead_items))
        runahead_tasks: set[asyncio.Task] = set()

        async def maybe_submit_runahead():
            """One feeder tick: check slack and submit runahead if possible."""
            nonlocal runahead_queue

            if not runahead_queue:
                return

            self.feeder_ticks += 1

            # Get fresh workloads
            workloads = await self.sm.get_all_workloads()
            self.slack_checks += 1

            # Find servers with slack AND budget
            for w in workloads:
                if not runahead_queue:
                    break

                if not w.has_slack(cfg):
                    continue

                if not self.sm.can_submit_runahead(w.server_idx):
                    continue

                # Submit one runahead to this server
                idx, item = runahead_queue.popleft()
                rid = item.get("request_id") or f"runahead_{uuid4().hex[:8]}"

                tr = RequestTracker(
                    request_id=rid,
                    batch_id="runahead",
                    index=idx,  # FIXED: Set index for proper ordering in output
                    max_tokens=item["max_tokens"],
                )
                runahead_tracker.requests[rid] = tr

                task = asyncio.create_task(
                    worker.generate_prompt(
                        item["prompt"],
                        request_id=rid,
                        max_tokens=item["max_tokens"],
                        tracker=tr,
                        kind="runahead",
                        sticky=False,
                        preferred_server_idx=w.server_idx,
                    )
                )
                runahead_tasks.add(task)
                self.runahead_submissions += 1

            # Check if we couldn't submit anything due to backpressure
            if runahead_queue and not any(
                w.has_slack(cfg) and self.sm.can_submit_runahead(w.server_idx) for w in workloads
            ):
                self.backpressure_events += 1

        def collect_done_runahead():
            """Collect completed runahead tasks (non-blocking)."""
            nonlocal runahead_tasks
            done = {t for t in runahead_tasks if t.done()}
            for t in done:
                runahead_tasks.remove(t)
                try:
                    result = t.result()
                    runahead_results.append(result)
                except Exception as e:
                    runahead_results.append({"error": str(e)})

        # --- Main loop: primary + continuous slack-filling ---
        while primary_tasks:
            # Wait for primary progress with timeout for feeder ticks
            jitter = random.uniform(0, cfg.poll_jitter_s)
            timeout = cfg.poll_interval_s + jitter

            done, primary_tasks = await asyncio.wait(
                primary_tasks,
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Collect completed primary
            for t in done:
                try:
                    result = await t
                    primary_results.append(result)
                except Exception as e:
                    primary_results.append({"error": str(e)})

            # Collect any done runahead
            collect_done_runahead()

            # Feeder tick: submit runahead if slack available
            await maybe_submit_runahead()

        self.primary_done_time = time.perf_counter()

        # --- Primary finished: cancel and abort remaining runahead ---
        # FIXED: Cancel asyncio tasks first (triggers safe cancellation path in generate())
        # This is better than just calling abort_requests because:
        # 1. Local work stops immediately
        # 2. CancelledError path in generate() calls abort with asyncio.shield
        # 3. We use gather with return_exceptions=True to avoid CancelledError propagation

        if runahead_tasks:
            print(f"\n   >>> Cancelling {len(runahead_tasks)} runahead tasks...")

            # Cancel all runahead tasks
            for t in runahead_tasks:
                t.cancel()

            # Gather with return_exceptions=True to collect all results/errors
            # This won't raise even if tasks were cancelled
            gathered_results = await asyncio.gather(*runahead_tasks, return_exceptions=True)

            for result in gathered_results:
                if isinstance(result, asyncio.CancelledError):
                    # Task was cancelled - already handled by generate()'s CancelledError path
                    pass
                elif isinstance(result, Exception):
                    runahead_results.append({"error": str(result)})
                else:
                    runahead_results.append(result)

            # Update tracker for any remaining running requests
            for req in runahead_tracker.requests.values():
                if req.status == "running":
                    req.status = "aborted"
                    req.end_time = time.perf_counter()

            aborted_count = sum(1 for r in runahead_tracker.requests.values() if r.status == "aborted")
            print(f"   >>> Cancelled/aborted: {aborted_count}")

        return primary_results, runahead_results


# =============================================================================
# Worker with Slack-Filling Support
# =============================================================================


class SlackFillingAgentLoopWorker(AgentLoopWorkerBase):
    """AgentLoopWorkerBase with slack-filling server manager."""

    def __init__(
        self,
        config,
        server_handles,
        slack_config: SlackFillingConfig,
        reward_router_address: str = None,
    ):
        self.slack_config = slack_config
        self.server_manager = SlackFillingServerManager(
            config, server_handles, slack_config=slack_config
        )
        super().__init__(config, server_handles, reward_router_address)

    async def generate_prompt(
        self,
        prompt: str,
        *,
        request_id: str,
        max_tokens: int,
        tracker: Optional[RequestTracker] = None,
        kind: str = "primary",
        sticky: bool = True,
        preferred_server_idx: Optional[int] = None,
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        sampling_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
        return await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            tracker=tracker,
            sticky=sticky,
            kind=kind,
            preferred_server_idx=preferred_server_idx,
        )


# =============================================================================
# Test Function
# =============================================================================


def test_runahead_slack_filling():
    """
    Test slack-filling runahead with backpressure.

    Demonstrates:
    1. Continuous slack-filling (not one-shot trigger)
    2. Backpressure based on num_requests_waiting and kv_cache
    3. Per-server budget limiting
    4. Priority config placeholder (TODO: wire into server)
    5. Safe cancellation for runahead tasks
    """
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    DP_SIZE = int(os.environ.get("DP_SIZE", str(NUM_GPUS // TP_SIZE)))

    PRIMARY_SIZE = int(os.environ.get("PRIMARY_SIZE", "8"))
    RUNAHEAD_SIZE = int(os.environ.get("RUNAHEAD_SIZE", "4"))

    # Slack-filling configuration
    BUDGET_PER_SERVER = int(os.environ.get("BUDGET_PER_SERVER", "1"))
    WAITING_THRESHOLD = int(os.environ.get("WAITING_THRESHOLD", "0"))
    KV_CACHE_THRESHOLD = float(os.environ.get("KV_CACHE_THRESHOLD", "0.85"))
    POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "0.1"))

    slack_config = SlackFillingConfig(
        budget_per_server=BUDGET_PER_SERVER,
        waiting_threshold=WAITING_THRESHOLD,
        kv_cache_threshold=KV_CACHE_THRESHOLD,
        poll_interval_s=POLL_INTERVAL,
    )

    print("=" * 80)
    print("Slack-Filling Runahead (Continuous Backpressure)")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print(f"Primary batch: {PRIMARY_SIZE} | Runahead batch: {RUNAHEAD_SIZE}")
    print("-" * 80)
    print("Slack-Filling Configuration:")
    print(f"  - budget_per_server: {BUDGET_PER_SERVER}")
    print(f"  - waiting_threshold (W): {WAITING_THRESHOLD}")
    print(f"  - kv_cache_threshold (K): {KV_CACHE_THRESHOLD}")
    print(f"  - poll_interval: {POLL_INTERVAL}s")
    print(f"  - primary_priority: {slack_config.primary_priority}")
    print(f"  - runahead_priority: {slack_config.runahead_priority}")
    print("=" * 80)

    print("\n[1] Initializing Ray...")
    import ray

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

    try:
        print("\n[2] Creating config...")
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = NUM_GPUS
        config.trainer.nnodes = 1
        config.actor_rollout_ref.model.path = MODEL_PATH
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE
        config.actor_rollout_ref.rollout.disable_log_stats = False
        if hasattr(config, "reward_model"):
            config.reward_model.use_reward_loop = False

        print(f"\n[3] Creating {DP_SIZE} vLLM server(s)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        rollout_class = get_rollout_replica_class("vllm")
        servers = []
        server_handles = []
        for dp_rank in range(DP_SIZE):
            print(f"   Creating server {dp_rank}...")
            server = rollout_class(
                replica_rank=dp_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=TP_SIZE,
            )
            asyncio.run(server.init_standalone())
            servers.append(server)
            server_handles.append(server._server_handle)
            print(f"   Server {dp_rank} ready")

        print("\n[4] Creating slack-filling worker...")
        worker = SlackFillingAgentLoopWorker(config, server_handles, slack_config=slack_config)

        # Test prompts with varying lengths
        primary_prompts = [
            {"prompt": "What is 2+2?", "max_tokens": 16},
            {"prompt": "Say hi.", "max_tokens": 16},
            {"prompt": "Name a color.", "max_tokens": 16},
            {"prompt": "What is 1+1?", "max_tokens": 16},
            {"prompt": "Write a detailed essay about AI history.", "max_tokens": 200},
            {"prompt": "Explain quantum computing step by step.", "max_tokens": 200},
            {"prompt": "Describe machine learning training.", "max_tokens": 200},
            {"prompt": "Write a story about robots.", "max_tokens": 200},
        ][:PRIMARY_SIZE]

        runahead_prompts = [
            {"prompt": "What is the capital of France?", "max_tokens": 32},
            {"prompt": "Speed of light?", "max_tokens": 32},
            {"prompt": "Write about math history.", "max_tokens": 300},
            {"prompt": "Explain climate change.", "max_tokens": 300},
        ][:RUNAHEAD_SIZE]

        # Add request IDs
        for i, item in enumerate(primary_prompts):
            item["request_id"] = f"primary_{i}_{uuid4().hex[:8]}"
        for i, item in enumerate(runahead_prompts):
            item["request_id"] = f"runahead_{i}_{uuid4().hex[:8]}"

        print(f"\n[5] Primary: {len(primary_prompts)} | Runahead: {len(runahead_prompts)}")
        print("   Primary prompts:")
        for item in primary_prompts:
            label = "short" if item["max_tokens"] <= 32 else "LONG"
            print(f"      {item['request_id']} ({label})")

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        async def run_simulation():
            controller = SlackFillingRunaheadController(
                server_manager=worker.server_manager,
                config=slack_config,
            )

            print("\n[6] Running with slack-filling runahead...")
            primary_results, runahead_results = await controller.run_with_runahead(
                primary_items=primary_prompts,
                runahead_items=runahead_prompts,
                primary_tracker=primary_tracker,
                runahead_tracker=runahead_tracker,
                worker=worker,
            )

            return primary_results, runahead_results, controller

        print("\n" + "=" * 80)
        start_time = time.perf_counter()
        primary_results, runahead_results, controller = asyncio.run(run_simulation())
        total_time = time.perf_counter() - start_time

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print("\n--- Primary Batch ---")
        for _, req in sorted(primary_tracker.requests.items(), key=lambda x: x[1].index):
            stop_info = f" ({req.stop_reason})" if req.stop_reason else ""
            print(
                f"   [{req.index}] {req.status:10s}{stop_info:8s} | {req.token_count:3d} tok | "
                f"{req.duration:.2f}s | server {req.server_idx}"
            )

        print("\n--- Runahead Batch ---")
        runahead_completed = 0
        runahead_aborted = 0

        if runahead_tracker.requests:
            for _, req in sorted(runahead_tracker.requests.items(), key=lambda x: x[1].index):
                stop_info = f" ({req.stop_reason})" if req.stop_reason else ""
                print(
                    f"   [{req.index}] {req.status:10s}{stop_info:8s} | {req.token_count:3d} tok | "
                    f"{req.duration:.2f}s | server {req.server_idx}"
                )
                if req.status == "completed":
                    runahead_completed += 1
                elif req.status == "aborted":
                    runahead_aborted += 1
        else:
            print("   (no runahead submitted - servers always busy)")

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Primary duration: {controller.primary_duration:.2f}s")
        print(f"Primary: {primary_tracker.completed}/{primary_tracker.total} completed")
        print(f"Runahead: {runahead_completed} completed, {runahead_aborted} aborted")

        print("\n--- Slack-Filling Metrics ---")
        print(f"Feeder ticks: {controller.feeder_ticks}")
        print(f"Slack checks: {controller.slack_checks}")
        print(f"Runahead submissions: {controller.runahead_submissions}")
        print(f"Backpressure events: {controller.backpressure_events}")

        print("\n--- Server Manager Metrics ---")
        sm = worker.server_manager
        print(f"Total requests: {sm.total_requests}")
        print(f"Runahead submitted: {sm.runahead_submitted}")
        print(f"Runahead completed: {sm.runahead_completed}")
        print(f"Runahead aborted: {sm.runahead_aborted}")
        print(f"Backpressure rejections: {sm.backpressure_rejections}")
        print(f"Runahead inflight per server: {sm.runahead_inflight_per_server}")

        print("\n--- Key Improvements Over One-Shot Trigger ---")
        print("1. Continuous feeding: runahead starts as soon as ANY slack appears")
        print("2. Backpressure: stops feeding automatically when servers busy")
        print(f"3. Per-server budget: max {slack_config.budget_per_server} runahead per server")
        print(f"4. Priority: primary={slack_config.primary_priority}, runahead={slack_config.runahead_priority}")
        print("5. Safe cancellation: abort before re-raise (prevents orphaned requests)")

        print("\n--- Safety Check ---")
        print("Used targeted abort by server_request_id: YES")
        print("Safe cancellation (abort before re-raise): YES")
        print("Priority scheduling: NO (TODO: pass priority to server)")

        print("\n" + "=" * 80)

        assert primary_tracker.completed == primary_tracker.total, "All primary should complete"
        print("\nTest PASSED!")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_runahead_slack_filling()
