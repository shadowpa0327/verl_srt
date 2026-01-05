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
from typing import Any, Optional
from uuid import uuid4

import ray
from cachetools import LRUCache

from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
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


@ray.remote
class RunaheadCentralRouter:
    """
    Central router with run-ahead (secondary) request support.

    This router provides all CentralRouter functionality plus:
    - Admission control: only admit secondary when server_load < load_threshold
    - Secondary routing: pick_slack_server() finds server with lowest load
    - Request tracking: maps server_request_id → server_idx for targeted abort
    - Targeted abort: abort_requests() cancels specific requests

    Note: This is a standalone class (not inheriting from CentralRouter) because
    Ray does not support inheritance from actor classes.

    Usage:
        router = RunaheadCentralRouter.remote(server_handles, load_threshold=32)

        # Primary requests (same as CentralRouter):
        output = await router.generate.remote(request_id, prompt_ids=..., ...)

        # Secondary requests (caller provides server_request_id for abort tracking):
        server_request_id = uuid4().hex  # Generate upfront
        output = await router.generate_secondary.remote(server_request_id, prompt_ids=..., ...)
        # Returns None if rejected (server at capacity)

        # Abort remaining secondary:
        await router.abort_requests.remote([server_request_id_1, server_request_id_2, ...])
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
        self._workload_last_poll_s: list[float] = [0.0] * self.num_servers
        self._workload_last_error: list[Optional[str]] = [None] * self.num_servers
        self._workload_last_warning: list[Optional[str]] = [None] * self.num_servers

        # Metrics
        self.total_requests = 0
        self.total_secondary_requests = 0
        self.secondary_rejected = 0
        self.secondary_aborted = 0

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
        self.use_kv_cache_admission = bool(enabled)
        self.require_fresh_workload = bool(require_fresh_workload)
        self.kv_cache_threshold = float(kv_cache_threshold)
        self.workload_poll_interval_s = float(poll_interval_s)
        self.workload_staleness_threshold_s = float(staleness_threshold_s)

        if self.use_kv_cache_admission:
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
                "num_requests_running": self._workload_num_requests_running[idx],
                "num_requests_waiting": self._workload_num_requests_waiting[idx],
                "kv_cache_usage": self._workload_kv_cache_usage[idx],
                "age_s": (now - last_poll_s) if last_poll_s else None,
                "workload_error": self._workload_last_error[idx],
                "workload_warning": self._workload_last_warning[idx],
            }
        return snapshot

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

            if isinstance(running, (int, float)):
                self._workload_num_requests_running[idx] = int(running)
            if isinstance(waiting, (int, float)):
                self._workload_num_requests_waiting[idx] = int(waiting)
            self._workload_last_poll_s[idx] = now

            if isinstance(kv_usage, (int, float)):
                kv = float(kv_usage)
                # vLLM metric variants sometimes report percent (0-100) instead of fraction (0-1).
                if kv > 1.0:
                    kv /= 100.0
                kv = max(0.0, min(1.0, kv))
                self._workload_kv_cache_usage[idx] = kv

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
        """Find server with lowest load under threshold.

        Returns:
            Server index with slack, or None if all servers at capacity.

        Note: Use priority queue if scaling beyond ~100 servers.
        """
        best_idx = None
        best_load = self.load_threshold  # Start at threshold
        now = time.monotonic()

        # Debug: print current loads
        print(f"[pick_slack_server] threshold={self.load_threshold}, server_loads={self.server_load}")

        for idx in range(self.num_servers):
            load = self.server_load[idx]
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

        if best_idx is None:
            print(f"[pick_slack_server] REJECTED: all servers at capacity (loads >= {self.load_threshold})")
        else:
            print(f"[pick_slack_server] ADMITTED: server_idx={best_idx}, load={best_load}")
        return best_idx

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Route primary generate request to appropriate server."""
        self.total_requests += 1
        server_idx, server = self._choose_server(request_id)
        self.server_load[server_idx] += 1

        # Generate unique server request ID
        server_request_id = uuid4().hex

        try:
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
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

    async def generate_secondary(
        self,
        server_request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> Optional[TokenOutput]:
        """Route secondary (runahead) request with admission control.

        The caller provides the server_request_id upfront, enabling reliable abort
        tracking before the request completes.

        Args:
            server_request_id: Caller-provided ID for abort tracking.
            prompt_ids: List of prompt token IDs.
            sampling_params: Sampling parameters for generation.
            image_data: Optional multi-modal image data.

        Returns:
            TokenOutput if admitted and completed, None if rejected due to capacity.
        """
        self.total_secondary_requests += 1

        # Admission control: find server with slack
        server_idx = self.pick_slack_server()
        if server_idx is None:
            self.secondary_rejected += 1
            return None

        # Track for abort BEFORE calling vLLM (critical for reliable abort)
        self._request_to_server[server_request_id] = server_idx
        self.server_load[server_idx] += 1
        print(f"Admitted secondary request {server_request_id} to server_idx={server_idx}")
        print(f"Current server loads: {self.server_load}")
        try:
            server = self.server_handles[server_idx]
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )
            return output
        finally:
            self.server_load[server_idx] -= 1
            if self.server_load[server_idx] < 0:
                self.server_load[server_idx] = 0
            # Clean up tracking after completion (abort already handled if needed)
            self._request_to_server.pop(server_request_id, None)

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
