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
Central Router for Agent Loop Architecture.

This module provides a centralized routing layer between AgentLoopWorkers and vLLM servers.
Instead of each worker having its own AsyncLLMServerManager with independent load tracking,
all workers route through a single CentralRouter Ray Actor that has a global view of all loads.

Architecture:
    AgentLoopWorker-0 ─┐
    AgentLoopWorker-1 ─┼→ CentralRouter (Ray Actor) → vLLMHttpServer[0..N]
    AgentLoopWorker-N ─┘
"""
import asyncio
import heapq
import logging
import os
import time
from collections import deque
from typing import Any, Optional
from uuid import uuid4

import ray
from cachetools import LRUCache

from verl.experimental.agent_loop.runahead.types import (
    RunaheadBatchResult,
    RunaheadMetrics,
    SecondaryOutput,
    SecondaryWorkItem,
)
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Ray async actor concurrency.
#
# Why this exists:
# - Ray async actors have a default concurrency limit (commonly 1000).
# - For large `primary_size` (e.g. 2048) the router becomes the bottleneck and caps
#   the number of in-flight generate() calls, even if vLLM servers have capacity.
#
# Keep it configurable so benchmarks/training can tune without code changes.
_ROUTER_MAX_CONCURRENCY = int(os.getenv("VERL_AGENT_LOOP_ROUTER_MAX_CONCURRENCY", "4096"))


@ray.remote(max_concurrency=_ROUTER_MAX_CONCURRENCY)
class CentralRouter:
    """
    Central router for all AgentLoopWorkers to vLLM servers.

    This Ray Actor provides:
    - Global load balancing: least-requests routing with global view across all workers
    - Sticky sessions: LRU cache maps request_id → server for prefix caching benefits
    - Concurrent request handling: async generate() yields during await

    Usage:
        router = CentralRouter.remote(server_handles)
        output = await router.generate.remote(request_id, prompt_ids=..., sampling_params=...)
    """

    def __init__(self, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the CentralRouter.

        Args:
            server_handles: List of Ray actor handles to vLLM servers.
            max_cache_size: Max size for request_id to server LRU cache (for sticky sessions).
        """
        self.server_handles = server_handles
        self.num_servers = len(server_handles)

        # Least-requests load balancing (same pattern as AsyncLLMServerManager):
        # A min-heap of [num_sessions, server_idx, server_handle].
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache for sticky sessions (same request_id → same server for prefix caching)
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # Track in-flight requests per server
        self.server_load = {idx: 0 for idx in range(self.num_servers)}

        # Minimal metrics
        self.total_requests = 0

        logger.info(f"CentralRouter initialized with {self.num_servers} servers")

    def _choose_server(self, request_id: str) -> tuple[int, ray.actor.ActorHandle]:
        """Choose server using least-requests load balancing with sticky sessions.

        Args:
            request_id: Request ID for sticky session lookup.

        Returns:
            Tuple of (server_index, server_handle).
        """
        # Check sticky session cache first (for multi-turn conversations)
        if request_id in self.request_id_to_server:
            server_idx = self.request_id_to_server[request_id]
            return server_idx, self.server_handles[server_idx]

        # Find least loaded server from heap
        _, server_idx, server = self.weighted_serveres[0]

        # Increment session counter and reheapify (same pattern as AsyncLLMServerManager).
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        # Cache for sticky sessions (store only the index; server handle is in server_handles)
        self.request_id_to_server[request_id] = server_idx

        return server_idx, server

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Route generate request to appropriate server.

        Args:
            request_id: Request ID for sticky session routing.
            prompt_ids: List of prompt token IDs.
            sampling_params: Sampling parameters for generation.
            image_data: Optional multi-modal image data.

        Returns:
            TokenOutput from the vLLM server.
        """
        self.total_requests += 1
        server_idx, server = self._choose_server(request_id)
        self.server_load[server_idx] += 1

        try:
            # Use new request_id for each turn (vLLM requirement)
            output = await server.generate.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )
            return output
        finally:
            self.server_load[server_idx] -= 1
            if self.server_load[server_idx] < 0:
                logger.warning(
                    "CentralRouter server_load went negative for server_idx=%s; resetting to 0.", server_idx
                )
                self.server_load[server_idx] = 0

    def get_server_loads(self) -> dict[int, int]:
        """Get current load per server (for monitoring/debugging).

        Returns:
            Dict mapping server_idx to current in-flight request count.
        """
        return dict(self.server_load)

    def get_total_requests(self) -> int:
        """Get total number of requests processed.

        Returns:
            Total request count.
        """
        return self.total_requests


@ray.remote(max_concurrency=_ROUTER_MAX_CONCURRENCY)
class RunaheadCentralRouter:
    """
    Central router with run-ahead (secondary) request support and router-owned queue.

    This router provides all CentralRouter functionality plus:
    - Router-owned queue: secondary work items queued internally, admitted when slack appears
    - Admission control: only admit secondary when server_load < load_threshold
    - Internal admit loop: background task polls for slack and admits pending items
    - No retry needed: queue model eliminates capacity-based rejection
    - Atomic cleanup: stop_runahead_batch() drops pending + aborts in-flight

    Note: This is a standalone class (not inheriting from CentralRouter) because
    Ray does not support inheritance from actor classes.

    Usage:
        router = RunaheadCentralRouter.remote(server_handles, load_threshold=32)

        # Primary requests (same as CentralRouter):
        output = await router.generate.remote(request_id, prompt_ids=..., ...)

        # Secondary requests via batch API:
        await router.start_runahead_batch.remote(work_items)
        # ... run primary batch ...
        result = await router.stop_runahead_batch.remote(abort_grace_s=1.0)
        # result.outputs contains completed/aborted/rejected SecondaryOutput items
    """

    def __init__(
        self,
        server_handles: list[ray.actor.ActorHandle],
        load_threshold: int = 32,
        max_cache_size: int = 10000,
    ):
        """Initialize the RunaheadCentralRouter.

        Args:
            server_handles: List of Ray actor handles to vLLM servers.
            load_threshold: Admit secondary when server_load < threshold.
            max_cache_size: Max size for request_id to server LRU cache.
        """
        # Initialize parent (but we can't call super().__init__ for Ray remote classes)
        self.server_handles = server_handles
        self.num_servers = len(server_handles)
        self.load_threshold = load_threshold

        # Least-requests load balancing
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache for sticky sessions
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # Track in-flight requests per server
        self.server_load = {idx: 0 for idx in range(self.num_servers)}

        # Track secondary (runahead) load per server (subset of server_load)
        self._secondary_load: dict[int, int] = {idx: 0 for idx in range(self.num_servers)}
        # Optimistic reservation for secondary load to prevent burst admission
        self._secondary_reserved_load: dict[int, int] = {idx: 0 for idx in range(self.num_servers)}
        # sample_id -> server_idx mapping for reservations
        self._secondary_reservation_by_sample_id: dict[str, int] = {}

        # Primary reservation to prevent startup race where secondaries are admitted
        # before primaries have registered load at the router. Uses global tracking
        # (not per-server) to avoid mismatch when primaries don't distribute evenly.
        self._primary_reserved_total: int = 0

        # Track server_request_id → server_idx for targeted abort
        self._request_to_server: dict[str, int] = {}

        # Optional workload-aware admission (kv-cache as a coarse safety valve).
        # When enabled, we poll vLLM /metrics via server.get_workload() and gate
        # secondary admission when kv_cache_usage is above threshold.
        self.use_kv_cache_admission: bool = False
        self.require_fresh_workload: bool = False
        self.kv_cache_threshold: float = 0.85
        self.workload_poll_interval_s: float = 0.5
        self.workload_staleness_threshold_s: float = 2.0

        self._workload_polling_active: bool = False
        self._workload_poll_task: Optional[asyncio.Task] = None

        self._workload_kv_cache_usage: list[Optional[float]] = [None] * self.num_servers
        self._workload_num_requests_running: list[Optional[int]] = [None] * self.num_servers
        self._workload_num_requests_waiting: list[Optional[int]] = [None] * self.num_servers
        self._workload_itl_sum: list[Optional[float]] = [None] * self.num_servers
        self._workload_itl_count: list[Optional[int]] = [None] * self.num_servers
        # Previous values for computing per-interval ITL average
        self._prev_itl_sum: list[Optional[float]] = [None] * self.num_servers
        self._prev_itl_count: list[Optional[int]] = [None] * self.num_servers
        self._workload_last_poll_s: list[float] = [0.0] * self.num_servers
        self._workload_last_error: list[Optional[str]] = [None] * self.num_servers
        self._workload_last_warning: list[Optional[str]] = [None] * self.num_servers

        # Metrics
        self.total_requests = 0
        self.total_secondary_requests = 0
        self.secondary_rejected = 0
        self.secondary_aborted = 0

        # Runahead batch state (router-owned queue model)
        self._batch_active: bool = False
        self._batch_id: Optional[str] = None
        self._pending_queue: deque[SecondaryWorkItem] = deque()
        # sample_id -> (asyncio.Task, server_idx, server_request_id)
        self._in_flight_batch: dict[str, tuple[asyncio.Task, int, str]] = {}
        self._batch_results: list[SecondaryOutput] = []
        self._batch_metrics: RunaheadMetrics = RunaheadMetrics()

        # Admit loop control
        self._admit_loop_task: Optional[asyncio.Task] = None
        self._admit_loop_stop: asyncio.Event = asyncio.Event()
        self._admit_loop_config: dict[str, Any] = {}

        # Primary request priority (can be set per-batch via set_primary_priority)
        self._primary_priority: int = 0

        # Round-robin starting point for fair server selection
        self._round_robin_start: int = 0

        # Time-series metrics collection
        self._metrics_collection_enabled: bool = False
        self._metrics_samples: list[dict[str, Any]] = []
        self._metrics_start_time: float = 0.0

        logger.info(
            f"RunaheadCentralRouter initialized with {self.num_servers} servers, "
            f"load_threshold={load_threshold}"
        )

    def set_load_threshold(self, load_threshold: int) -> int:
        """Update admission load threshold for secondary requests.

        Args:
            load_threshold: New threshold. Secondary admitted when server_load < threshold.

        Returns:
            The updated threshold.
        """
        self.load_threshold = int(load_threshold)
        return self.load_threshold

    def set_primary_priority(self, priority: int) -> int:
        """Set the priority for primary requests.

        Args:
            priority: Priority value (lower = higher priority). Default is 0.

        Returns:
            The updated priority.
        """
        self._primary_priority = int(priority)
        return self._primary_priority

    def reserve_primary_load(self, total_primaries: int) -> dict[str, Any]:
        """Reserve load for expected primary requests to prevent startup race.

        When secondaries start before primaries, server_load is still low and
        admission is overly optimistic. This reserves capacity globally (not
        per-server) to avoid mismatch when primaries don't distribute evenly
        due to load balancing or sticky sessions. Each primary that arrives
        at generate() releases one reservation from the global pool.

        Args:
            total_primaries: Total number of primary requests expected.

        Returns:
            Dict with reservation total and estimated per-server load.
        """
        self._primary_reserved_total = total_primaries
        per_server_estimate = total_primaries // self.num_servers

        logger.info(
            f"RunaheadCentralRouter: Reserved primary load for {total_primaries} requests "
            f"(~{per_server_estimate} per server)"
        )

        return {
            "total_primaries": total_primaries,
            "per_server_estimate": per_server_estimate,
        }

    async def configure_workload_polling(
        self,
        *,
        enabled: bool,
        kv_cache_threshold: float = 0.85,
        poll_interval_s: float = 0.5,
        staleness_threshold_s: float = 2.0,
        require_fresh_workload: bool = False,
        prime_cache: bool = False,
    ) -> dict[str, Any]:
        """Configure optional workload polling for kv-cache-aware admission.

        This is intentionally conservative and best-effort:
        - If metrics are unavailable or stale, we can fall back to server_load admission.
        - When require_fresh_workload=True, we reject secondaries until we have fresh metrics.

        Args:
            enabled: Enable kv-cache-aware admission.
            kv_cache_threshold: Reject secondary when kv_cache_usage >= threshold.
            poll_interval_s: Background polling interval (seconds).
            staleness_threshold_s: Consider metrics stale after this many seconds.
            require_fresh_workload: If True, require fresh metrics for admission.
            prime_cache: If True, perform one polling round before returning.
        """
        # If metrics collection is enabled, keep polling active regardless of `enabled` param
        if self._metrics_collection_enabled:
            enabled = True

        self.use_kv_cache_admission = bool(enabled)
        self.require_fresh_workload = bool(require_fresh_workload)
        self.kv_cache_threshold = float(kv_cache_threshold)
        self.workload_poll_interval_s = float(poll_interval_s)
        self.workload_staleness_threshold_s = float(staleness_threshold_s)

        if self.use_kv_cache_admission or self._metrics_collection_enabled:
            await self._ensure_workload_polling_task()
            self._workload_polling_active = True
            if prime_cache:
                await self._poll_all_servers()
        else:
            self._workload_polling_active = False

        return {
            "enabled": self.use_kv_cache_admission,
            "kv_cache_threshold": self.kv_cache_threshold,
            "poll_interval_s": self.workload_poll_interval_s,
            "staleness_threshold_s": self.workload_staleness_threshold_s,
            "require_fresh_workload": self.require_fresh_workload,
        }

    async def refresh_workload_cache(self) -> dict[int, dict[str, Any]]:
        """Poll workload once and return a snapshot of cached metrics (for debugging/tests)."""
        await self._poll_all_servers()
        now = time.monotonic()
        snapshot: dict[int, dict[str, Any]] = {}
        for idx in range(self.num_servers):
            kv = self._workload_kv_cache_usage[idx]
            last_poll_s = self._workload_last_poll_s[idx]
            snapshot[idx] = {
                "num_requests_running": self._workload_num_requests_running[idx],
                "num_requests_waiting": self._workload_num_requests_waiting[idx],
                "kv_cache_usage": kv,
                "age_s": (now - last_poll_s) if last_poll_s else None,
                "workload_error": self._workload_last_error[idx],
                "workload_warning": self._workload_last_warning[idx],
            }
        return snapshot

    async def get_server_state(self, *, poll_workload: bool = False) -> dict[int, dict[str, Any]]:
        """Return a per-server snapshot of router load and cached workload metrics.

        Args:
            poll_workload: If True, poll all servers once before returning.
        """
        if poll_workload:
            await self._poll_all_servers()

        now = time.monotonic()
        snapshot: dict[int, dict[str, Any]] = {}
        for idx in range(self.num_servers):
            last_poll_s = self._workload_last_poll_s[idx]
            snapshot[idx] = {
                "server_load": int(self.server_load[idx]),
                "secondary_reserved_load": self._secondary_reserved_load.get(idx, 0),
                "num_requests_running": self._workload_num_requests_running[idx],
                "num_requests_waiting": self._workload_num_requests_waiting[idx],
                "kv_cache_usage": self._workload_kv_cache_usage[idx],
                "age_s": (now - last_poll_s) if last_poll_s else None,
                "workload_error": self._workload_last_error[idx],
                "workload_warning": self._workload_last_warning[idx],
            }
        return snapshot

    # =========================================================================
    # Time-Series Metrics Collection
    # =========================================================================

    async def start_metrics_collection(self) -> dict[str, Any]:
        """Start recording time-series metrics samples.

        Samples are collected each time _poll_all_servers() is called.
        The workload polling must be active for samples to be collected.

        Returns:
            Dict with status and start_time.
        """
        self._metrics_samples.clear()
        self._metrics_start_time = time.monotonic()
        self._metrics_collection_enabled = True

        # Ensure workload polling is active so we get samples
        if not self._workload_polling_active:
            await self.configure_workload_polling(enabled=True, prime_cache=True)

        logger.info("RunaheadCentralRouter: Started metrics collection")
        return {"status": "started", "start_time": self._metrics_start_time}

    async def stop_metrics_collection(self) -> dict[str, Any]:
        """Stop recording and return all collected samples.

        Returns:
            Dict with status, duration, num_samples, and samples list.
        """
        self._metrics_collection_enabled = False
        duration = time.monotonic() - self._metrics_start_time

        result = {
            "status": "stopped",
            "duration_s": duration,
            "num_samples": len(self._metrics_samples),
            "samples": list(self._metrics_samples),
        }

        logger.info(
            f"RunaheadCentralRouter: Stopped metrics collection. "
            f"duration={duration:.2f}s, samples={len(self._metrics_samples)}"
        )
        return result

    def get_metrics_samples(self) -> list[dict[str, Any]]:
        """Get collected samples (may be partial if still running).

        Returns:
            List of sample dicts with timestamp, wall_time, and per-server metrics.
        """
        return list(self._metrics_samples)

    def export_metrics_csv(self, filepath: str) -> dict[str, Any]:
        """Export collected metrics to CSV file.

        Args:
            filepath: Path to output CSV file.

        Returns:
            Dict with status and num_rows written.
        """
        import csv

        if not self._metrics_samples:
            logger.warning("No metrics samples to export")
            return {"status": "empty", "num_rows": 0}

        # Flatten samples to rows
        rows = []
        for sample in self._metrics_samples:
            timestamp = sample["timestamp"]
            wall_time = sample["wall_time"]
            for server_idx, server_metrics in sample["servers"].items():
                rows.append({
                    "timestamp": timestamp,
                    "wall_time": wall_time,
                    "server_idx": server_idx,
                    "num_requests_running": server_metrics.get("num_requests_running"),
                    "num_requests_waiting": server_metrics.get("num_requests_waiting"),
                    "kv_cache_usage": server_metrics.get("kv_cache_usage"),
                    "server_load": server_metrics.get("server_load"),
                    "secondary_load": server_metrics.get("secondary_load"),
                    "itl_avg_ms": server_metrics.get("itl_avg_ms"),
                })

        fieldnames = [
            "timestamp", "wall_time", "server_idx",
            "num_requests_running", "num_requests_waiting",
            "kv_cache_usage", "server_load", "secondary_load",
            "itl_avg_ms",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Exported {len(rows)} metrics rows to {filepath}")
        return {"status": "success", "num_rows": len(rows), "filepath": filepath}

    async def _ensure_workload_polling_task(self) -> None:
        if self._workload_poll_task is not None and not self._workload_poll_task.done():
            return
        self._workload_poll_task = asyncio.create_task(self._workload_poll_loop())

    async def _workload_poll_loop(self) -> None:
        try:
            while True:
                if self._workload_polling_active:
                    try:
                        await self._poll_all_servers()
                    except Exception:
                        logger.exception("RunaheadCentralRouter workload polling iteration failed")
                    await asyncio.sleep(self.workload_poll_interval_s)
                else:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            return

    async def _poll_all_servers(self) -> None:
        # Query all servers in parallel via Ray; each server is responsible for how it collects metrics.
        tasks = [server.get_workload.remote() for server in self.server_handles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        now = time.monotonic()

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                self._workload_last_error[idx] = repr(result)
                self._workload_last_warning[idx] = None
                continue
            if not isinstance(result, dict):
                self._workload_last_error[idx] = f"Invalid workload result type: {type(result)!r}"
                self._workload_last_warning[idx] = None
                continue
            if result.get("error"):
                self._workload_last_error[idx] = str(result.get("error"))
                self._workload_last_warning[idx] = None
                continue
            self._workload_last_error[idx] = None
            warning = result.get("warning")
            if warning:
                available = result.get("available_vllm_metrics")
                if isinstance(available, list) and available:
                    self._workload_last_warning[idx] = f"{warning}; available_vllm_metrics={available}"
                else:
                    self._workload_last_warning[idx] = str(warning)
            else:
                self._workload_last_warning[idx] = None

            running = result.get("num_requests_running")
            waiting = result.get("num_requests_waiting")
            kv_usage = result.get("kv_cache_usage")
            itl_sum = result.get("inter_token_latency_sum")
            itl_count = result.get("inter_token_latency_count")

            if isinstance(running, (int, float)):
                self._workload_num_requests_running[idx] = int(running)
            if isinstance(waiting, (int, float)):
                self._workload_num_requests_waiting[idx] = int(waiting)
            if isinstance(itl_sum, (int, float)):
                self._workload_itl_sum[idx] = float(itl_sum)
            if isinstance(itl_count, (int, float)):
                self._workload_itl_count[idx] = int(itl_count)
            self._workload_last_poll_s[idx] = now

            if isinstance(kv_usage, (int, float)):
                kv = float(kv_usage)
                # vLLM metric variants sometimes report percent (0-100) instead of fraction (0-1).
                if kv > 1.0:
                    kv /= 100.0
                kv = max(0.0, min(1.0, kv))
                self._workload_kv_cache_usage[idx] = kv

        # Store time-series sample if metrics collection is enabled
        if self._metrics_collection_enabled:
            sample: dict[str, Any] = {
                "timestamp": now - self._metrics_start_time,
                "wall_time": time.time(),
                "servers": {},
            }
            for idx in range(self.num_servers):
                # Compute per-interval average ITL (in milliseconds)
                itl_avg_ms = None
                curr_sum = self._workload_itl_sum[idx]
                curr_count = self._workload_itl_count[idx]
                prev_sum = self._prev_itl_sum[idx]
                prev_count = self._prev_itl_count[idx]

                if (curr_sum is not None and curr_count is not None and
                    prev_sum is not None and prev_count is not None):
                    delta_sum = curr_sum - prev_sum
                    delta_count = curr_count - prev_count
                    if delta_count > 0:
                        itl_avg_ms = (delta_sum / delta_count) * 1000  # Convert to ms

                # Update previous values for next iteration
                self._prev_itl_sum[idx] = curr_sum
                self._prev_itl_count[idx] = curr_count

                sample["servers"][idx] = {
                    "num_requests_running": self._workload_num_requests_running[idx],
                    "num_requests_waiting": self._workload_num_requests_waiting[idx],
                    "kv_cache_usage": self._workload_kv_cache_usage[idx],
                    "server_load": self.server_load.get(idx, 0),
                    "secondary_load": self._secondary_load.get(idx, 0),
                    "itl_avg_ms": itl_avg_ms,
                }
            self._metrics_samples.append(sample)

    def _choose_server(self, request_id: str) -> tuple[int, ray.actor.ActorHandle]:
        """Choose server using least-requests load balancing with sticky sessions."""
        # Check sticky session cache first
        if request_id in self.request_id_to_server:
            server_idx = self.request_id_to_server[request_id]
            return server_idx, self.server_handles[server_idx]

        # Find least loaded server from heap
        _, server_idx, server = self.weighted_serveres[0]

        # Increment session counter and reheapify
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])

        # Cache for sticky sessions
        self.request_id_to_server[request_id] = server_idx

        return server_idx, server

    def pick_slack_server(self) -> Optional[int]:
        """Find server with lowest load under threshold using round-robin start.

        Returns:
            Server index with slack, or None if all servers at capacity.

        The round-robin starting point ensures fair distribution across servers
        when multiple servers have equal load (avoids always picking server 0).

        Note: Use priority queue if scaling beyond ~100 servers.

        TODO: kv_cache_usage is polled periodically and may be stale. When we admit
        a request, the actual kv_cache on the server increases but our cached value
        doesn't reflect that until the next poll. Consider adding optimistic
        reservation: increment cached kv_cache_usage by an estimate when admitting,
        decrement when request completes. This prevents over-admission between polls.
        """
        # Block ALL secondaries until primaries fully registered at the router.
        # This prevents the startup race where secondaries are admitted before
        # primaries have converted their reservations to actual server_load.
        if self._primary_reserved_total > 0:
            return None

        best_idx = None
        best_load = self.load_threshold  # Start at threshold
        now = time.monotonic()

        # Round-robin: start from different server each time for fairness
        start = self._round_robin_start
        for i in range(self.num_servers):
            idx = (start + i) % self.num_servers
            # Effective load includes secondary reservations only
            # (primary reservations block entirely via check above)
            load = (self.server_load[idx] +
                    self._secondary_reserved_load.get(idx, 0))
            if load >= best_load:
                continue

            if self.use_kv_cache_admission:
                kv = self._workload_kv_cache_usage[idx]
                last_poll_s = self._workload_last_poll_s[idx]
                fresh = kv is not None and (now - last_poll_s) <= self.workload_staleness_threshold_s

                if fresh:
                    if kv >= self.kv_cache_threshold:
                        continue
                elif self.require_fresh_workload:
                    continue

            best_idx = idx
            best_load = load

        # Advance round-robin for next call
        if best_idx is not None:
            self._round_robin_start = (best_idx + 1) % self.num_servers

        return best_idx

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        priority: Optional[int] = None,
    ) -> TokenOutput:
        """Route primary generate request to appropriate server.

        Args:
            request_id: Request ID for sticky session routing.
            prompt_ids: List of prompt token IDs.
            sampling_params: Sampling parameters for generation.
            image_data: Optional multi-modal image data.
            priority: Request priority (lower = higher priority). If None, uses
                the router's configured primary priority (default 0).

        Returns:
            TokenOutput from the vLLM server.
        """
        # Use configured primary priority if not explicitly provided
        if priority is None:
            priority = self._primary_priority

        self.total_requests += 1
        server_idx, server = self._choose_server(request_id)

        # Release one primary reservation if active (converts reservation to actual load)
        if self._primary_reserved_total > 0:
            self._primary_reserved_total -= 1

        self.server_load[server_idx] += 1

        # Generate unique server request ID
        server_request_id = uuid4().hex

        try:
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                priority=priority,
            )
            return output
        finally:
            self.server_load[server_idx] -= 1
            if self.server_load[server_idx] < 0:
                logger.warning(
                    "RunaheadCentralRouter server_load went negative for server_idx=%s; resetting to 0.",
                    server_idx,
                )
                self.server_load[server_idx] = 0

    async def abort_requests(self, server_request_ids: list[str]) -> dict[str, int]:
        """Abort specific requests by server_request_id.

        Args:
            server_request_ids: List of server request IDs to abort.

        Returns:
            Dict with abort statistics: {"aborted": count, "not_found": count}
        """
        aborted = 0
        not_found = 0

        # Group by server for efficient batching
        by_server: dict[int, list[str]] = {}
        for rid in server_request_ids:
            server_idx = self._request_to_server.get(rid)
            if server_idx is not None:
                by_server.setdefault(server_idx, []).append(rid)
            else:
                not_found += 1

        # Issue abort calls in parallel
        abort_tasks = []
        for server_idx, rids in by_server.items():
            server = self.server_handles[server_idx]
            for rid in rids:
                # vLLM abort_request takes single request_id
                abort_tasks.append(server.abort_request.remote(rid))
                # Clean up tracking
                self._request_to_server.pop(rid, None)
                aborted += 1

        # Wait for all aborts to complete
        if abort_tasks:
            await asyncio.gather(*abort_tasks, return_exceptions=True)

        self.secondary_aborted += aborted
        return {"aborted": aborted, "not_found": not_found}

    def get_server_loads(self) -> dict[int, int]:
        """Get current load per server."""
        return dict(self.server_load)

    def get_total_requests(self) -> int:
        """Get total number of primary requests processed."""
        return self.total_requests

    async def wait_for_total_requests(
        self,
        *,
        min_total_requests: int,
        poll_interval_s: float = 0.05,
        timeout_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """Wait until total_requests reaches min_total_requests.

        This is used by the Manager to avoid a startup race where secondaries get
        admitted before any primary request has reached the router (so server_load
        is still zero and admission is overly optimistic).

        Args:
            min_total_requests: Target total primary request count.
            poll_interval_s: Sleep interval between checks (seconds).
            timeout_s: Optional timeout; if None, wait indefinitely.

        Returns:
            Dict with {"ready": bool, "total_requests": int}.
        """
        start_s = time.monotonic()
        while self.total_requests < min_total_requests:
            if timeout_s is not None and (time.monotonic() - start_s) >= timeout_s:
                return {"ready": False, "total_requests": self.total_requests}
            await asyncio.sleep(poll_interval_s)
        return {"ready": True, "total_requests": self.total_requests}

    def get_runahead_stats(self) -> dict[str, int]:
        """Get runahead-specific statistics.

        Returns:
            Dict with secondary_requests, secondary_rejected, secondary_aborted.
        """
        return {
            "secondary_requests": self.total_secondary_requests,
            "secondary_rejected": self.secondary_rejected,
            "secondary_aborted": self.secondary_aborted,
            "in_flight_tracked": len(self._request_to_server),
        }

    # =========================================================================
    # Router-Owned Queue Model: Batch API
    # =========================================================================

    async def start_runahead_batch(
        self,
        items: list[SecondaryWorkItem],
        *,
        poll_interval_s: float = 0.05,
        max_queue_size: int = 256,
    ) -> dict[str, Any]:
        """Start a batch of secondary (runahead) work items.

        The router queues all items internally and runs an admit loop that
        polls for slack and admits items when capacity is available.

        With the router-owned queue model, capacity-based rejection is eliminated:
        items wait in the queue until slack exists. No retry logic is needed.

        Admission is gated by load_threshold on a per-server basis. This naturally
        limits total secondaries in flight without needing a global cap.

        Args:
            items: List of SecondaryWorkItem to process.
            poll_interval_s: Polling interval for admit loop (seconds).
            max_queue_size: Maximum pending items in queue (oldest dropped on overflow).

        Returns:
            Dict with {"batch_id": str, "queued": int, "status": "started"}.

        Raises:
            RuntimeError: If a batch is already active.
        """
        if self._batch_active:
            raise RuntimeError("A runahead batch is already active. Call stop_runahead_batch() first.")

        # Reset batch state
        self._batch_id = uuid4().hex
        self._batch_active = True
        self._pending_queue.clear()
        self._in_flight_batch.clear()
        self._batch_results.clear()
        self._batch_metrics = RunaheadMetrics()
        self._admit_loop_stop.clear()

        # Store config for admit loop
        self._admit_loop_config = {
            "poll_interval_s": poll_interval_s,
            "max_queue_size": max_queue_size,
        }

        # Queue items (enforce max_queue_size)
        for item in items:
            if len(self._pending_queue) >= max_queue_size:
                # Drop oldest item
                dropped = self._pending_queue.popleft()
                dropped_hash = self._extract_prompt_hash(dropped)
                self._batch_results.append(SecondaryOutput(
                    sample_id=dropped.sample_id,
                    output=None,
                    status="rejected",
                    tokens_generated=0,
                    prompt_hash=dropped_hash,
                ))
                self._batch_metrics.secondary_rejected += 1
            self._pending_queue.append(item)

        # Start admit loop
        self._admit_loop_task = asyncio.create_task(self._admit_loop())

        logger.info(
            f"RunaheadCentralRouter started batch {self._batch_id} with {len(self._pending_queue)} items"
        )

        return {
            "batch_id": self._batch_id,
            "queued": len(self._pending_queue),
            "status": "started",
        }

    async def stop_runahead_batch(
        self,
        *,
        abort_grace_s: float = 1.0,
    ) -> RunaheadBatchResult:
        """Stop the current runahead batch and return results.

        This atomically:
        1. Stops the admit loop
        2. Drops all pending items (marks as rejected)
        3. Aborts all in-flight requests
        4. Waits up to abort_grace_s for in-flight tasks to complete
        5. Force-cancels any stragglers

        Args:
            abort_grace_s: Grace period for in-flight tasks to complete (seconds).

        Returns:
            RunaheadBatchResult with completed, aborted, and rejected outputs.

        Raises:
            RuntimeError: If no batch is active.
        """
        if not self._batch_active:
            raise RuntimeError("No runahead batch is active.")

        batch_id = self._batch_id or ""

        # 1. Signal admit loop to stop
        self._admit_loop_stop.set()

        # 2. Wait for admit loop to stop (with timeout)
        if self._admit_loop_task and not self._admit_loop_task.done():
            try:
                await asyncio.wait_for(self._admit_loop_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._admit_loop_task.cancel()
                try:
                    await self._admit_loop_task
                except asyncio.CancelledError:
                    pass

        # 3. Drop all pending items
        while self._pending_queue:
            item = self._pending_queue.popleft()
            item_hash = self._extract_prompt_hash(item)
            self._batch_results.append(SecondaryOutput(
                sample_id=item.sample_id,
                output=None,
                status="rejected",
                tokens_generated=0,
                prompt_hash=item_hash,
            ))
            self._batch_metrics.secondary_rejected += 1

        # 4. Abort all in-flight requests
        in_flight_ids = list(self._in_flight_batch.keys())
        if in_flight_ids:
            # Issue abort to vLLM servers
            server_request_ids = [info[2] for info in self._in_flight_batch.values()]
            await self.abort_requests(server_request_ids)

            # Wait for tasks with grace period
            tasks = [info[0] for info in self._in_flight_batch.values()]
            if tasks:
                done, pending = await asyncio.wait(tasks, timeout=abort_grace_s)

                # Collect results from done tasks
                for task in done:
                    try:
                        # Task completed (result already collected in _execute_secondary)
                        pass
                    except Exception:
                        pass

                # Cancel remaining pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Cleanup any remaining reservations for tasks that never started or were cancelled before start
        for sample_id, server_idx in list(self._secondary_reservation_by_sample_id.items()):
            self._secondary_reserved_load[server_idx] -= 1
            del self._secondary_reservation_by_sample_id[sample_id]

        # Cleanup unredeemed primary reservations (defensive - should already be released)
        if self._primary_reserved_total > 0:
            logger.info(
                f"RunaheadCentralRouter: Clearing {self._primary_reserved_total} "
                "unredeemed primary reservations"
            )
            self._primary_reserved_total = 0

        # 5. Any remaining in-flight items that weren't already collected, mark as aborted.
        # Note: _execute_secondary may have already added results during the grace period,
        # so we check for existing sample_ids to avoid double-counting.
        # Aborted items won't be used for cache updates, so prompt_hash=0 is acceptable.
        existing_sample_ids = {o.sample_id for o in self._batch_results}
        for sample_id, (task, server_idx, server_request_id) in list(self._in_flight_batch.items()):
            if sample_id not in existing_sample_ids:
                self._batch_results.append(SecondaryOutput(
                    sample_id=sample_id,
                    output=None,
                    status="aborted",
                    tokens_generated=0,
                    prompt_hash=0,  # Aborted items don't need hash for cache updates
                ))
                self._batch_metrics.secondary_aborted += 1
        self._in_flight_batch.clear()

        # 6. Build result
        result = RunaheadBatchResult(
            batch_id=batch_id,
            outputs=list(self._batch_results),
            metrics=self._batch_metrics,
        )

        # 7. Reset state
        self._batch_active = False
        self._batch_id = None
        self._pending_queue.clear()
        self._batch_results.clear()
        self._admit_loop_task = None

        logger.info(
            f"RunaheadCentralRouter stopped batch {batch_id}: "
            f"completed={result.metrics.secondary_completed}, "
            f"aborted={result.metrics.secondary_aborted}, "
            f"rejected={result.metrics.secondary_rejected}"
        )

        return result

    def get_runahead_batch_status(self) -> dict[str, Any]:
        """Get current status of the active runahead batch.

        Returns:
            Dict with batch_active, pending_count, in_flight_count, completed_count, etc.
        """
        completed = sum(1 for o in self._batch_results if o.status == "completed")
        aborted = sum(1 for o in self._batch_results if o.status == "aborted")
        rejected = sum(1 for o in self._batch_results if o.status == "rejected")

        return {
            "batch_active": self._batch_active,
            "batch_id": self._batch_id,
            "pending_count": len(self._pending_queue),
            "in_flight_count": len(self._in_flight_batch),
            "completed_count": completed,
            "aborted_count": aborted,
            "rejected_count": rejected,
            "total_results": len(self._batch_results),
        }

    # =========================================================================
    # Admit Loop (internal)
    # =========================================================================

    async def _admit_loop(self) -> None:
        """Background task that admits pending items when slack is available.

        Loop invariant:
        - Runs while _batch_active and not _admit_loop_stop.is_set()
        - Admits items when per-server slack exists (server_load < load_threshold)
        - Items wait in queue until slack exists (no retry needed)
        - Collects results from completed tasks
        """
        poll_interval_s = self._admit_loop_config.get("poll_interval_s", 0.05)

        try:
            while self._batch_active and not self._admit_loop_stop.is_set():
                # 1. Collect completed in-flight tasks (non-blocking)
                await self._collect_completed_tasks()

                admitted_counts = {}

                # 2. Admit new items if per-server slack available
                while (self._pending_queue and
                       not self._admit_loop_stop.is_set()):

                    server_idx = self.pick_slack_server()
                    if server_idx is None:
                        break  # No slack, wait for next poll

                    # Optimistic reservation
                    self._secondary_reserved_load[server_idx] += 1

                    work_item = self._pending_queue.popleft()
                    self._secondary_reservation_by_sample_id[work_item.sample_id] = server_idx
                    admitted_counts[server_idx] = admitted_counts.get(server_idx, 0) + 1

                    # Use a marker prefix so the proposer can identify secondary requests
                    # and skip speculative decoding for them (to isolate primary metrics)
                    server_request_id = f"runahead_{uuid4().hex}"

                    # Start task
                    task = asyncio.create_task(
                        self._execute_secondary(work_item, server_idx, server_request_id)
                    )
                    self._in_flight_batch[work_item.sample_id] = (task, server_idx, server_request_id)
                    self._request_to_server[server_request_id] = server_idx

                    self._batch_metrics.secondary_started += 1
                    self.total_secondary_requests += 1

                if admitted_counts:
                    logger.info(f"Admitted secondaries distribution: {admitted_counts}")
                    # Yield to let tasks start and convert reservations to actual load
                    await asyncio.sleep(0)

                # 3. Sleep before next poll (interruptible)
                try:
                    await asyncio.wait_for(
                        self._admit_loop_stop.wait(),
                        timeout=poll_interval_s
                    )
                    break  # Stop requested
                except asyncio.TimeoutError:
                    pass  # Continue polling

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("RunaheadCentralRouter admit loop failed")

    def _extract_prompt_hash(self, work_item: SecondaryWorkItem) -> int:
        """Extract pre-computed prompt_hash from work item's sampling_params."""
        extra_args = work_item.sampling_params.get("extra_args", {})
        if extra_args:
            return extra_args.get("prompt_hash", 0)
        return 0

    async def _execute_secondary(
        self,
        work_item: SecondaryWorkItem,
        server_idx: int,
        server_request_id: str,
    ) -> None:
        """Execute a single secondary request and handle result/retry.

        Race Safety:
            This method may complete concurrently with stop_runahead_batch().
            The design handles this gracefully:
            1. Results are appended to _batch_results BEFORE removing from _in_flight_batch
            2. stop_runahead_batch() checks existing_sample_ids before adding aborted entries
            3. Even if timing is tight, we get exactly one result per sample_id

        TODO: Add optimistic kv_cache reservation here to prevent over-admission.
            Before generate: self._workload_kv_cache_usage[server_idx] += estimated_kv
            In finally block: self._workload_kv_cache_usage[server_idx] -= estimated_kv
            See pick_slack_server() docstring for details.
        """
        # Convert reservation to actual load
        if self._secondary_reservation_by_sample_id.pop(work_item.sample_id, None) is not None:
            self._secondary_reserved_load[server_idx] -= 1

        self.server_load[server_idx] += 1
        self._secondary_load[server_idx] += 1

        # Extract pre-computed hash for cache updates
        prompt_hash = self._extract_prompt_hash(work_item)

        try:
            server = self.server_handles[server_idx]
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=work_item.prompt_ids,
                sampling_params=work_item.sampling_params,
                image_data=work_item.image_data,
                priority=work_item.priority,
            )

            if output is not None:
                tokens_generated = len(output.token_ids) if output.token_ids else 0
                # Check if the request was aborted (vLLM returns output with stop_reason="aborted")
                stop_reason = getattr(output, "stop_reason", None)
                if stop_reason == "aborted":
                    self._batch_results.append(SecondaryOutput(
                        sample_id=work_item.sample_id,
                        output=output,
                        status="aborted",
                        tokens_generated=tokens_generated,
                        prompt_ids=work_item.prompt_ids,
                        prompt_hash=prompt_hash,
                    ))
                    self._batch_metrics.secondary_aborted += 1
                    self.secondary_aborted += 1
                else:
                    self._batch_results.append(SecondaryOutput(
                        sample_id=work_item.sample_id,
                        output=output,
                        status="completed",
                        tokens_generated=tokens_generated,
                        prompt_ids=work_item.prompt_ids,
                        prompt_hash=prompt_hash,
                    ))
                    self._batch_metrics.secondary_completed += 1
            else:
                # Should not happen with router-owned queue, but handle gracefully
                self._handle_failure(work_item, prompt_hash)

        except asyncio.CancelledError:
            # Aborted by stop_runahead_batch - don't add to results here,
            # stop_runahead_batch will handle it
            raise

        except Exception as e:
            logger.warning(f"Secondary {work_item.sample_id} failed: {e}")
            self._handle_failure(work_item, prompt_hash)

        finally:
            self.server_load[server_idx] -= 1
            self._secondary_load[server_idx] -= 1
            if self.server_load[server_idx] < 0:
                self.server_load[server_idx] = 0
            if self._secondary_load[server_idx] < 0:
                self._secondary_load[server_idx] = 0
            self._request_to_server.pop(server_request_id, None)
            # Remove from in_flight AFTER adding to _batch_results (see Race Safety above)
            self._in_flight_batch.pop(work_item.sample_id, None)

    async def _collect_completed_tasks(self) -> None:
        """Collect results from completed in-flight tasks (non-blocking)."""
        completed_sample_ids = []

        for sample_id, (task, server_idx, server_request_id) in list(self._in_flight_batch.items()):
            if task.done():
                completed_sample_ids.append(sample_id)
                # Results already added in _execute_secondary
                try:
                    task.result()  # Raise any exception
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass  # Already handled in _execute_secondary

        # Clean up (already done in _execute_secondary.finally, but be safe)
        for sample_id in completed_sample_ids:
            self._in_flight_batch.pop(sample_id, None)

    def _handle_failure(self, work_item: SecondaryWorkItem, prompt_hash: int = 0) -> None:
        """Handle a failed item (no retries - queue model prevents capacity rejection)."""
        self._batch_results.append(SecondaryOutput(
            sample_id=work_item.sample_id,
            output=None,
            status="rejected",
            tokens_generated=0,
            prompt_hash=prompt_hash,
        ))
        self._batch_metrics.secondary_rejected += 1
        self.secondary_rejected += 1
