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
Server-Side Admission Prototype (Standalone)

Goal
----
Prototype a "server-side admission" mechanism for runahead WITHOUT modifying
Verl library code. This is implemented as a per-server Ray actor wrapper that:

1) Distinguishes primary vs runahead via a special sampling_params key:
      sampling_params["_verl_request_kind"] in {"primary", "runahead"}
   The wrapper pops this key before forwarding to the real vLLM server to avoid
   breaking vLLM SamplingParams construction.

2) Enforces a global (cross-worker) runahead inflight limit per server.

3) Optionally gates runahead admission by server workload metrics (cached with
   a short TTL to avoid hammering /metrics).

This prototype also integrates the slack-filling pattern from
test_vllm_runahead_slack_filling.py, demonstrating:
- Two-layer admission: client-side slack check + server-side global limit
- Continuous drip-feeding of runahead during primary execution
- Graceful handling of server-side rejections

Usage
-----
  python tests/workers/rollout/rollout_vllm/test_vllm_runahead_server_side_admission_prototype.py

Key env vars
------------
  MODEL_PATH: HF model path (default: Qwen/Qwen2.5-0.5B-Instruct)
  NUM_GPUS / TP_SIZE / DP_SIZE: vLLM server topology (same as other scripts)

  PRIMARY_SIZE: number of primary requests (default: 4)
  RUNAHEAD_SIZE: number of runahead requests (default: 8)
  ADMISSION_MAX_INFLIGHT: max concurrent runahead allowed per server (default: 1)

  ADMISSION_ENFORCE_SLACK: 1 to require slack based on workload metrics (default: 1)
  ADMISSION_WAITING_THRESHOLD: W in (waiting <= W) (default: 0)
  ADMISSION_KV_CACHE_THRESHOLD: K in (kv_cache <= K) (default: 0.85)
  ADMISSION_WORKLOAD_CACHE_TTL_S: cache TTL for get_workload() (default: 0.2)

Notes
-----
- This script focuses on correctness of admission (no oversubscription).
- Tests mixed primary+runahead workloads with continuous slack-filling.
- Server-side admission ensures global limits even with multiple workers.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput


@dataclass
class AdmissionGateConfig:
    """Configuration for server-side admission control.

    Server-side enforcement:
    - max_runahead_inflight: Global limit per server (across all workers)
    - enforce_slack: Whether to check workload before admitting runahead

    Slack thresholds (when enforce_slack=True):
    - load_threshold: Max (running + waiting) to consider server has slack
    - kv_cache_threshold: Max kv_cache_usage to consider server has slack

    Polling (for continuous slack-filling):
    - poll_interval_s: How often client checks for slack
    - poll_jitter_s: Jitter to reduce herding across workers
    - workload_cache_ttl_s: Cache TTL for workload queries

    Retry:
    - max_runahead_retries: Max times to retry a rejected runahead (0 = no retry)

    Testing:
    - skip_client_budget_check: Skip client-side budget check to test server-side rejection/retry
    """

    max_runahead_inflight: int = 1
    enforce_slack: bool = True
    load_threshold: int = 32  # max (running + waiting) to consider slack
    kv_cache_threshold: float = 0.85
    poll_interval_s: float = 0.1
    poll_jitter_s: float = 0.03
    workload_cache_ttl_s: float = 0.2
    max_runahead_retries: int = 3  # Retry rejected runahead up to 3 times
    skip_client_budget_check: bool = False  # For testing: skip client budget to force server rejection


# =============================================================================
# Request Tracking
# =============================================================================


@dataclass
class RequestTracker:
    """Track individual request state and timing."""

    request_id: str
    server_request_id: str = ""
    batch_id: str = ""
    index: int = 0
    max_tokens: int = 0
    server_idx: int = -1
    start_time: float = 0.0
    end_time: float = 0.0
    token_count: int = 0
    status: str = "pending"  # pending | running | completed | aborted | rejected | error
    stop_reason: Optional[str] = None
    token_ids: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "aborted", "rejected", "error")


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
    def rejected(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "rejected")

    @property
    def running(self) -> int:
        return sum(1 for r in self.requests.values() if r.status == "running")

    def get_running_server_request_ids(self) -> list[str]:
        return [r.server_request_id for r in self.requests.values() if r.status == "running" and r.server_request_id]


@ray.remote
class AdmissionControlledServer:
    """A wrapper around a vLLM server actor that enforces runahead admission."""

    def __init__(self, server_handle, config: AdmissionGateConfig):
        self._server = server_handle
        self._cfg = config
        self._lock = asyncio.Lock()
        self._workload_lock = asyncio.Lock()

        self._runahead_inflight = 0
        self._runahead_max_observed = 0
        self._runahead_rejected_total = 0

        self._cached_workload: Optional[dict[str, Any]] = None
        self._cached_workload_time_s: float = 0.0

    async def _get_cached_workload(self) -> dict[str, Any]:
        now = time.perf_counter()
        if self._cached_workload is not None and (now - self._cached_workload_time_s) < self._cfg.workload_cache_ttl_s:
            return self._cached_workload

        # Prevent a thundering herd when many concurrent callers refresh metrics.
        async with self._workload_lock:
            now = time.perf_counter()
            if self._cached_workload is not None and (now - self._cached_workload_time_s) < self._cfg.workload_cache_ttl_s:
                return self._cached_workload

            workload = await self._server.get_workload.remote()
            self._cached_workload = dict(workload) if isinstance(workload, dict) else {"error": "non-dict workload"}
            self._cached_workload_time_s = now
            return self._cached_workload

    def _has_slack(self, workload: dict[str, Any]) -> bool:
        if workload.get("error") is not None:
            return False
        if workload.get("warning") is not None:
            return False
        if "kv_cache_usage" not in workload:
            return False
        try:
            running = int(workload.get("num_requests_running", 0))
            waiting = int(workload.get("num_requests_waiting", 0))
            total_load = running + waiting
            kv_cache = float(workload["kv_cache_usage"])
        except Exception:
            return False
        return total_load <= self._cfg.load_threshold and kv_cache <= self._cfg.kv_cache_threshold

    async def get_admission_stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "max_runahead_inflight": self._cfg.max_runahead_inflight,
                "runahead_inflight": self._runahead_inflight,
                "runahead_max_observed": self._runahead_max_observed,
                "runahead_rejected_total": self._runahead_rejected_total,
                "enforce_slack": self._cfg.enforce_slack,
            }

    async def get_workload(self) -> dict[str, Any]:
        return await self._get_cached_workload()

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        try:
            return await self._server.abort_request.remote(request_id, reset_prefix_cache=reset_prefix_cache)
        except Exception as e:
            msg = str(e)
            if "reset_prefix_cache" in msg and "unexpected keyword argument" in msg:
                return await self._server.abort_request.remote(request_id)
            raise

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        # Never mutate caller's dict in-place.
        params = dict(sampling_params)
        kind = params.pop("_verl_request_kind", "primary")

        acquired_slot = False
        if kind == "runahead":
            # Fast reject if we're already at the inflight limit (avoid metrics calls).
            async with self._lock:
                if self._runahead_inflight >= self._cfg.max_runahead_inflight:
                    self._runahead_rejected_total += 1
                    return TokenOutput(token_ids=[], log_probs=None, routed_experts=None, stop_reason="rejected")

            if self._cfg.enforce_slack:
                workload = await self._get_cached_workload()
                if not self._has_slack(workload):
                    async with self._lock:
                        self._runahead_rejected_total += 1
                    return TokenOutput(token_ids=[], log_probs=None, routed_experts=None, stop_reason="rejected")

            async with self._lock:
                if self._runahead_inflight >= self._cfg.max_runahead_inflight:
                    self._runahead_rejected_total += 1
                    return TokenOutput(token_ids=[], log_probs=None, routed_experts=None, stop_reason="rejected")
                self._runahead_inflight += 1
                acquired_slot = True
                if self._runahead_inflight > self._runahead_max_observed:
                    self._runahead_max_observed = self._runahead_inflight

        try:
            return await self._server.generate.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=params,
                image_data=image_data,
            )
        finally:
            if acquired_slot:
                async with self._lock:
                    self._runahead_inflight -= 1


# =============================================================================
# Admission Gate Registry (Prevents N Independent Counters)
# =============================================================================


@ray.remote
class AdmissionGateRegistry:
    """Singleton registry for admission gates. Ensures one gate per server.

    This prevents the "N independent counters" problem where multiple workers
    accidentally create their own gate instances, each with an independent
    _runahead_inflight counter, leading to oversubscription of the vLLM server.

    Usage:
    - Driver: Use get_or_create_registry() to create/get the registry
    - Driver: Use registry.get_or_create(server_idx, server_handle, config)
    - Workers: Use get_admission_registry() to get existing registry (fails if missing)
    - Workers: Use registry.get(server_idx) to get existing gates (never create)
    """

    def __init__(self):
        self._gates: dict[int, Any] = {}  # server_idx → gate handle
        self._configs: dict[int, AdmissionGateConfig] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        server_idx: int,
        server_handle,
        config: AdmissionGateConfig,
    ):
        """Get existing gate or create new one. Thread-safe."""
        async with self._lock:
            if server_idx in self._gates:
                # Validate config matches
                existing_cfg = self._configs[server_idx]
                if existing_cfg != config:
                    raise ValueError(
                        f"Config mismatch for server {server_idx}: "
                        f"existing max_inflight={existing_cfg.max_runahead_inflight}, "
                        f"requested max_inflight={config.max_runahead_inflight}"
                    )
                return self._gates[server_idx]

            # Create new gate with deterministic name
            gate = AdmissionControlledServer.options(
                name=f"admission_gate_{server_idx}",
                max_concurrency=64,
            ).remote(server_handle, config)

            self._gates[server_idx] = gate
            self._configs[server_idx] = config
            return gate

    async def get(self, server_idx: int):
        """Get existing gate. Raises KeyError if not found."""
        if server_idx not in self._gates:
            raise KeyError(f"No gate registered for server {server_idx}")
        return self._gates[server_idx]

    async def get_all(self) -> list:
        """Get all registered gates."""
        return list(self._gates.values())

    async def validate_handle(self, handle) -> bool:
        """Validate a handle is a proper admission gate."""
        try:
            stats = await handle.get_admission_stats.remote()
            return "max_runahead_inflight" in stats
        except Exception:
            return False

    async def get_stats(self) -> dict[str, Any]:
        """Get registry stats."""
        return {
            "num_gates": len(self._gates),
            "server_indices": list(self._gates.keys()),
        }


def get_admission_registry(namespace: str = "default"):
    """Get the admission registry (for workers). Fails if not found.

    Workers should never create the registry - only the driver should.
    This ensures workers fail fast if the registry wasn't properly initialized.
    """
    try:
        return ray.get_actor("admission_registry", namespace=namespace)
    except ValueError:
        raise RuntimeError(
            "AdmissionGateRegistry not found. "
            "Ensure the driver creates it before starting workers."
        )


def get_or_create_registry(namespace: str = "default"):
    """Get or create the admission registry (for driver only).

    Uses lifetime='detached' so the registry persists across driver restarts
    and can be accessed by workers in different processes.
    """
    try:
        return ray.get_actor("admission_registry", namespace=namespace)
    except ValueError:
        return AdmissionGateRegistry.options(
            name="admission_registry",
            namespace=namespace,
            lifetime="detached",
            max_concurrency=100,
        ).remote()


# =============================================================================
# Server Manager with Server-Side Admission
# =============================================================================


class ServerSideAdmissionServerManager(AsyncLLMServerManager):
    """Server manager that works with AdmissionControlledServer wrappers.

    Inherits from AsyncLLMServerManager to reuse:
    - Least-requests load balancing (heap-based)
    - Sticky sessions via LRU cache (request_id → server mapping)

    Adds two-layer admission control for runahead:
    1. Client-side: Checks cached workload + local budget (avoid unnecessary remote calls)
    2. Server-side: AdmissionControlledServer enforces global limits

    Key features:
    - Plumbs _verl_request_kind through sampling_params for server-side routing
    - Handles server-side rejections gracefully (stop_reason="rejected")
    - Tracks both client-side and server-side rejections separately
    """

    def __init__(
        self,
        config,  # DictConfig for parent class
        gated_handles: list,  # List of AdmissionControlledServer actor handles
        admission_config: AdmissionGateConfig,
        max_cache_size: int = 10000,
    ):
        # Initialize parent with admission-controlled server handles
        super().__init__(config, gated_handles, max_cache_size=max_cache_size)

        self.admission_config = admission_config
        self.num_servers = len(gated_handles)

        # Build server_idx mapping (parent shuffles server_handles, so we need this)
        self._server_to_idx = {server: idx for idx, server in enumerate(self.server_handles)}

        # Request tracking for targeted abort
        self._request_to_server: dict[str, int] = {}

        # Client-side workload cache (first filter before hitting server)
        self._cached_workloads: list[Optional[dict[str, Any]]] = [None] * self.num_servers
        self._workload_cache_time: float = 0.0

        # Client-side runahead budget tracking (optimistic filtering)
        self.runahead_inflight_per_server = [0] * self.num_servers

        # Metrics
        self.total_requests = 0
        self.primary_submitted = 0
        self.runahead_submitted = 0
        self.runahead_completed = 0
        self.runahead_aborted = 0
        self.runahead_rejected = 0  # Server-side rejections
        self.client_side_rejections = 0  # Client-side slack/budget rejections

        # Validation flag
        self._handles_validated = False

    async def validate_gate_handles(self) -> None:
        """Validate all handles are proper admission gates with matching config.

        This is a guardrail to catch misconfiguration early:
        - Ensures all handles respond to get_admission_stats()
        - Ensures all handles have matching max_runahead_inflight config

        Should be called after construction, before processing requests.
        """
        if self._handles_validated:
            return

        for i, h in enumerate(self.server_handles):
            try:
                stats = await h.get_admission_stats.remote()
            except Exception as e:
                raise ValueError(
                    f"Handle {i} is not a valid admission gate (get_admission_stats failed): {e}"
                )

            if "max_runahead_inflight" not in stats:
                raise ValueError(
                    f"Handle {i} missing 'max_runahead_inflight' in admission stats"
                )

            if stats["max_runahead_inflight"] != self.admission_config.max_runahead_inflight:
                raise ValueError(
                    f"Handle {i} config mismatch: "
                    f"expected max_inflight={self.admission_config.max_runahead_inflight}, "
                    f"got {stats['max_runahead_inflight']}"
                )

        self._handles_validated = True

    async def get_all_workloads(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch workload from all servers (with caching + jitter)."""
        now = time.perf_counter()
        jitter = random.uniform(0, self.admission_config.poll_jitter_s)
        cache_ttl = self.admission_config.workload_cache_ttl_s + jitter

        if not force_refresh and (now - self._workload_cache_time < cache_ttl):
            result = []
            for i, w in enumerate(self._cached_workloads):
                if w is not None:
                    result.append(w)
                else:
                    result.append({"error": "missing", "server_idx": i})
            return result

        # Fetch fresh workloads in parallel
        tasks = [h.get_workload.remote() for h in self.server_handles]
        workloads = await asyncio.gather(*tasks, return_exceptions=True)

        result = []
        for i, w in enumerate(workloads):
            if isinstance(w, Exception):
                result.append({"error": str(w), "server_idx": i})
            elif isinstance(w, dict):
                w = dict(w)
                w["server_idx"] = i
                result.append(w)
            else:
                result.append({"error": "non-dict workload", "server_idx": i})

        self._cached_workloads = result
        self._workload_cache_time = now
        return result

    def _has_slack(self, workload: dict[str, Any]) -> bool:
        """Client-side slack check (first filter)."""
        if workload.get("error") is not None:
            return False
        if workload.get("warning") is not None:
            return False
        try:
            running = int(workload.get("num_requests_running", 0))
            waiting = int(workload.get("num_requests_waiting", 0))
            total_load = running + waiting
            kv_cache = float(workload.get("kv_cache_usage", 1.0))
        except (ValueError, TypeError):
            return False
        return (
            total_load <= self.admission_config.load_threshold
            and kv_cache <= self.admission_config.kv_cache_threshold
        )

    def _can_submit_runahead(self, server_idx: int) -> bool:
        """Client-side budget check (optimistic - server has final say)."""
        return self.runahead_inflight_per_server[server_idx] < self.admission_config.max_runahead_inflight

    def _find_slack_server(self, workloads: list[dict[str, Any]]) -> Optional[int]:
        """Find a server with slack AND budget (client-side check)."""
        slack_servers = [
            (w["server_idx"], w)
            for w in workloads
            if self._has_slack(w) and self._can_submit_runahead(w["server_idx"])
        ]
        if not slack_servers:
            return None
        # Pick lowest load + inflight combination
        best_idx, _ = min(
            slack_servers,
            key=lambda x: (
                x[1].get("num_requests_running", 0)
                + x[1].get("num_requests_waiting", 0)
                + self.runahead_inflight_per_server[x[0]],
                x[0],  # tie-breaker: server index
            ),
        )
        return best_idx

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        tracker: Optional[RequestTracker] = None,
        kind: str = "primary",
        preferred_server_idx: Optional[int] = None,
    ) -> Optional[TokenOutput]:
        """Generate with two-layer admission control.

        For primary:
        - Uses parent's _choose_server for sticky sessions

        For runahead:
        1. Client-side: Check cached workload + local budget (avoid network if no slack)
        2. Server-side: AdmissionControlledServer enforces global limit

        Returns None only for client-side rejections (no slack / budget exceeded).
        Server-side rejections return a TokenOutput with stop_reason="rejected".
        """
        # Determine server
        if preferred_server_idx is not None:
            server_idx = preferred_server_idx
            server = self.server_handles[server_idx]
            # Still do client-side budget check for runahead
            if kind == "runahead" and not self._can_submit_runahead(server_idx):
                self.client_side_rejections += 1
                if tracker:
                    tracker.status = "rejected"
                    tracker.stop_reason = "client_budget_exceeded"
                return None
        elif kind == "primary":
            # Use parent's _choose_server for sticky sessions
            server = self._choose_server(request_id)
            server_idx = self._server_to_idx[server]
        else:
            # For runahead without preferred server, do client-side slack + budget check
            workloads = await self.get_all_workloads()
            server_idx = self._find_slack_server(workloads)
            if server_idx is None:
                self.client_side_rejections += 1
                if tracker:
                    tracker.status = "rejected"
                    tracker.stop_reason = "client_no_slack"
                return None
            server = self.server_handles[server_idx]

        server_request_id = uuid4().hex
        self._request_to_server[server_request_id] = server_idx

        # Plumb request kind through sampling_params
        params = dict(sampling_params)
        params["_verl_request_kind"] = kind

        self.total_requests += 1
        if kind == "primary":
            self.primary_submitted += 1
        else:
            self.runahead_submitted += 1
            self.runahead_inflight_per_server[server_idx] += 1

        if tracker:
            tracker.server_request_id = server_request_id
            tracker.server_idx = server_idx
            tracker.start_time = time.perf_counter()
            tracker.status = "running"

        try:
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=params,
                image_data=image_data,
            )

            stop_reason = getattr(output, "stop_reason", None)

            if tracker:
                tracker.end_time = time.perf_counter()
                tracker.stop_reason = stop_reason
                tracker.token_count = len(getattr(output, "token_ids", []))
                tracker.token_ids = list(getattr(output, "token_ids", []))

                if stop_reason == "rejected":
                    tracker.status = "rejected"
                elif stop_reason == "aborted":
                    tracker.status = "aborted"
                else:
                    tracker.status = "completed"

            # Update metrics
            if kind == "runahead":
                if stop_reason == "rejected":
                    self.runahead_rejected += 1
                elif stop_reason == "aborted":
                    self.runahead_aborted += 1
                else:
                    self.runahead_completed += 1

            return output

        except asyncio.CancelledError:
            # Safe cancellation: abort server request before re-raising
            if server_request_id in self._request_to_server:
                try:
                    await asyncio.shield(server.abort_request.remote(server_request_id, reset_prefix_cache=False))
                except BaseException:
                    try:
                        await asyncio.shield(server.abort_request.remote(server_request_id))
                    except BaseException:
                        pass  # Best effort
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
            if kind == "runahead":
                self.runahead_inflight_per_server[server_idx] -= 1
            self._request_to_server.pop(server_request_id, None)

    async def abort_requests(self, server_request_ids: list[str]) -> dict[str, Any]:
        """Abort requests by server_request_id."""
        if not server_request_ids:
            return {"aborted_count": 0, "request_ids": []}

        by_server: dict[int, list[str]] = {}
        for rid in server_request_ids:
            server_idx = self._request_to_server.get(rid)
            if server_idx is not None:
                by_server.setdefault(server_idx, []).append(rid)

        async def abort_on_server(server_idx: int, ids: list[str]) -> dict:
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
                        result = {"aborted": False, "error": msg}
                results.append(result)
            aborted = sum(1 for r in results if r.get("aborted", False))
            return {"server_idx": server_idx, "aborted_count": aborted, "request_ids": ids}

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
# Continuous Slack-Filling Controller
# =============================================================================


@dataclass
class RunaheadQueueItem:
    """Item in the runahead queue with retry tracking."""

    index: int
    item: dict  # {"request_id", "prompt", "max_tokens"}
    retry_count: int = 0

    @property
    def request_id(self) -> str:
        return self.item.get("request_id") or f"runahead_{self.index}"


class ServerSideAdmissionController:
    """Controller for continuous slack-filling with server-side admission.

    Pattern:
    1. Launch all primary requests immediately
    2. Continuously drip-feed runahead when servers have slack
    3. Server-side admission enforces global limits (handles herding)
    4. Retry rejected runahead by requeuing to back of queue
    5. Cancel remaining runahead when primary completes
    """

    def __init__(
        self,
        server_manager: ServerSideAdmissionServerManager,
        config: AdmissionGateConfig,
        tokenizer,
    ):
        self.sm = server_manager
        self.config = config
        self.tokenizer = tokenizer

        # Timing
        self.primary_start_time: Optional[float] = None
        self.primary_done_time: Optional[float] = None

        # Metrics
        self.feeder_ticks = 0
        self.runahead_submissions = 0
        self.backpressure_events = 0
        self.runahead_requeues = 0  # Times a rejected item was requeued
        self.runahead_dropped = 0  # Times a rejected item exceeded max retries

    @property
    def primary_duration(self) -> Optional[float]:
        if self.primary_start_time is None or self.primary_done_time is None:
            return None
        return self.primary_done_time - self.primary_start_time

    def _tokenize(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    async def run_with_runahead(
        self,
        *,
        primary_items: list[dict],  # {"request_id", "prompt", "max_tokens"}
        runahead_items: list[dict],
        primary_tracker: BatchTracker,
        runahead_tracker: BatchTracker,
    ) -> tuple[list, list]:
        """Run primary batch with continuous slack-filling runahead."""
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
                index=i,
                max_tokens=item["max_tokens"],
            )
            primary_tracker.requests[rid] = tr

            prompt_ids = self._tokenize(item["prompt"])
            sampling_params = {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": item["max_tokens"],
            }

            async def run_primary(rid=rid, prompt_ids=prompt_ids, sampling_params=sampling_params, tracker=tr):
                return await self.sm.generate(
                    request_id=rid,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    tracker=tracker,
                    kind="primary",
                )

            task = asyncio.create_task(run_primary())
            primary_tasks.add(task)

        # --- Prepare runahead queue with retry tracking ---
        runahead_queue: deque[RunaheadQueueItem] = deque(
            RunaheadQueueItem(index=i, item=item) for i, item in enumerate(runahead_items)
        )
        runahead_tasks: set[asyncio.Task] = set()
        # Map task -> queue_item for requeue on rejection
        task_to_queue_item: dict[asyncio.Task, RunaheadQueueItem] = {}

        async def maybe_submit_runahead():
            """One feeder tick: check slack and submit runahead if possible."""
            if not runahead_queue:
                return

            self.feeder_ticks += 1

            # Get fresh workloads for client-side check
            workloads = await self.sm.get_all_workloads()

            # Find servers with slack AND budget (unless skip_client_budget_check for testing)
            for w in workloads:
                if not runahead_queue:
                    break

                server_idx = w["server_idx"]
                if not self.sm._has_slack(w):
                    continue
                if not cfg.skip_client_budget_check and not self.sm._can_submit_runahead(server_idx):
                    continue

                # Submit one runahead to this server
                queue_item = runahead_queue.popleft()
                rid = queue_item.item.get("request_id") or f"runahead_{queue_item.index}_{uuid4().hex[:8]}"

                # Create or update tracker
                if rid in runahead_tracker.requests:
                    # Reusing existing tracker for retry
                    tr = runahead_tracker.requests[rid]
                    tr.status = "pending"  # Reset status for retry
                else:
                    tr = RequestTracker(
                        request_id=rid,
                        batch_id="runahead",
                        index=queue_item.index,
                        max_tokens=queue_item.item["max_tokens"],
                    )
                    runahead_tracker.requests[rid] = tr

                prompt_ids = self._tokenize(queue_item.item["prompt"])
                sampling_params = {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": queue_item.item["max_tokens"],
                }

                async def run_runahead(rid=rid, prompt_ids=prompt_ids, sampling_params=sampling_params, tracker=tr, server_idx=w["server_idx"]):
                    return await self.sm.generate(
                        request_id=rid,
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        tracker=tracker,
                        kind="runahead",
                        preferred_server_idx=server_idx,
                    )

                task = asyncio.create_task(run_runahead())
                runahead_tasks.add(task)
                task_to_queue_item[task] = queue_item
                self.runahead_submissions += 1

            # Track backpressure
            if runahead_queue and not any(self.sm._has_slack(w) for w in workloads):
                self.backpressure_events += 1

        def collect_done_runahead():
            """Collect completed runahead tasks, requeue rejected ones."""
            nonlocal runahead_tasks
            done = {t for t in runahead_tasks if t.done()}
            for t in done:
                runahead_tasks.remove(t)
                queue_item = task_to_queue_item.pop(t, None)

                try:
                    result = t.result()

                    # Check if rejected - may need to requeue
                    # Rejection can be:
                    # 1. result is None (client-side rejection in generate())
                    # 2. result.stop_reason == "rejected" (server-side rejection)
                    is_rejected = False
                    if result is None:
                        is_rejected = True
                    elif hasattr(result, "stop_reason") and result.stop_reason == "rejected":
                        is_rejected = True

                    if is_rejected and queue_item is not None:
                        # Check if we can retry
                        if queue_item.retry_count < cfg.max_runahead_retries:
                            # Requeue to back with incremented retry count
                            queue_item.retry_count += 1
                            runahead_queue.append(queue_item)
                            self.runahead_requeues += 1
                            # Don't add to results yet - will retry
                            continue
                        else:
                            # Max retries exceeded, drop it
                            self.runahead_dropped += 1

                    runahead_results.append(result)
                except Exception as e:
                    runahead_results.append({"error": str(e)})

        # --- Main loop: primary + continuous slack-filling ---
        while primary_tasks:
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

        # --- Primary finished: cancel remaining runahead ---
        if runahead_tasks:
            print(f"\n   >>> Cancelling {len(runahead_tasks)} runahead tasks...")

            for t in runahead_tasks:
                t.cancel()

            gathered_results = await asyncio.gather(*runahead_tasks, return_exceptions=True)

            for result in gathered_results:
                if isinstance(result, asyncio.CancelledError):
                    pass  # Handled by generate()'s CancelledError path
                elif isinstance(result, Exception):
                    runahead_results.append({"error": str(result)})
                else:
                    runahead_results.append(result)

            # Update tracker for remaining running requests
            for req in runahead_tracker.requests.values():
                if req.status == "running":
                    req.status = "aborted"
                    req.end_time = time.perf_counter()

        return primary_results, runahead_results


def test_server_side_admission_prototype():
    """Test server-side admission with mixed primary + runahead workload.

    This test demonstrates:
    1. Two-layer admission: client-side slack check + server-side global limit
    2. Continuous drip-feeding of runahead during primary execution
    3. Server-side rejection handling (stop_reason="rejected")
    4. No oversubscription even under herding scenarios
    """
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    DP_SIZE = int(os.environ.get("DP_SIZE", str(NUM_GPUS // TP_SIZE)))

    PRIMARY_SIZE = int(os.environ.get("PRIMARY_SIZE", "4"))
    RUNAHEAD_SIZE = int(os.environ.get("RUNAHEAD_SIZE", "8"))

    cfg = AdmissionGateConfig(
        max_runahead_inflight=int(os.environ.get("ADMISSION_MAX_INFLIGHT", "1")),
        enforce_slack=os.environ.get("ADMISSION_ENFORCE_SLACK", "1") == "1",
        waiting_threshold=int(os.environ.get("ADMISSION_WAITING_THRESHOLD", "0")),
        kv_cache_threshold=float(os.environ.get("ADMISSION_KV_CACHE_THRESHOLD", "0.85")),
        poll_interval_s=float(os.environ.get("POLL_INTERVAL", "0.1")),
        workload_cache_ttl_s=float(os.environ.get("ADMISSION_WORKLOAD_CACHE_TTL_S", "0.2")),
        max_runahead_retries=int(os.environ.get("MAX_RUNAHEAD_RETRIES", "3")),
        # Set SKIP_CLIENT_BUDGET=1 to test server-side rejection and retry logic
        skip_client_budget_check=os.environ.get("SKIP_CLIENT_BUDGET", "0") == "1",
    )

    print("=" * 80)
    print("Server-Side Admission with Continuous Slack-Filling")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print(f"Primary batch: {PRIMARY_SIZE} | Runahead batch: {RUNAHEAD_SIZE}")
    print("-" * 80)
    print("Admission config (server-side enforced):")
    print(f"  - max_runahead_inflight: {cfg.max_runahead_inflight} (global per server)")
    print(f"  - enforce_slack: {cfg.enforce_slack}")
    print(f"  - waiting_threshold: {cfg.waiting_threshold}")
    print(f"  - kv_cache_threshold: {cfg.kv_cache_threshold}")
    print(f"  - poll_interval_s: {cfg.poll_interval_s}")
    print(f"  - workload_cache_ttl_s: {cfg.workload_cache_ttl_s}")
    print(f"  - max_runahead_retries: {cfg.max_runahead_retries}")
    print(f"  - skip_client_budget_check: {cfg.skip_client_budget_check}")
    if cfg.skip_client_budget_check:
        print("    ^ Testing mode: client budget check skipped to force server-side rejections")
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

    try:
        print("\n[2] Creating config...")
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            trainer_config = compose(config_name="ppo_trainer")

        trainer_config.trainer.n_gpus_per_node = NUM_GPUS
        trainer_config.trainer.nnodes = 1
        trainer_config.actor_rollout_ref.model.path = MODEL_PATH
        trainer_config.actor_rollout_ref.rollout.name = "vllm"
        trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE
        trainer_config.actor_rollout_ref.rollout.disable_log_stats = False
        if hasattr(trainer_config, "reward_model"):
            trainer_config.reward_model.use_reward_loop = False

        print(f"\n[3] Creating {DP_SIZE} vLLM server(s)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = trainer_config.actor_rollout_ref.rollout
        model_config = trainer_config.actor_rollout_ref.model
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

        print("\n[4] Creating admission registry and gates...")
        registry = get_or_create_registry()

        async def create_gates():
            gates = []
            for i, h in enumerate(server_handles):
                gate = await registry.get_or_create.remote(i, h, cfg)
                gates.append(gate)
            return gates

        gated_handles = asyncio.run(create_gates())
        print(f"   Created {len(gated_handles)} gates via registry")

        print("\n[5] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        # Create server manager and controller
        # Pass trainer_config as the DictConfig for AsyncLLMServerManager parent
        server_manager = ServerSideAdmissionServerManager(
            config=trainer_config,
            gated_handles=gated_handles,
            admission_config=cfg,
        )
        controller = ServerSideAdmissionController(server_manager, cfg, tokenizer)

        # Test prompts
        primary_prompts = [
            {"prompt": "What is 2+2?", "max_tokens": 16},
            {"prompt": "Say hi.", "max_tokens": 16},
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
            {"prompt": "Describe the solar system.", "max_tokens": 300},
            {"prompt": "Write about ancient Rome.", "max_tokens": 300},
            {"prompt": "Explain gravity.", "max_tokens": 300},
            {"prompt": "Write about the ocean.", "max_tokens": 300},
        ][:RUNAHEAD_SIZE]

        # Add request IDs
        for i, item in enumerate(primary_prompts):
            item["request_id"] = f"primary_{i}_{uuid4().hex[:8]}"
        for i, item in enumerate(runahead_prompts):
            item["request_id"] = f"runahead_{i}_{uuid4().hex[:8]}"

        print(f"\n[6] Primary: {len(primary_prompts)} | Runahead: {len(runahead_prompts)}")
        print("   Primary prompts:")
        for item in primary_prompts:
            label = "short" if item["max_tokens"] <= 32 else "LONG"
            print(f"      {item['request_id']} ({label})")

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        async def run_simulation():
            # Validate gate handles before running
            print("\n[7] Validating gate handles...")
            await server_manager.validate_gate_handles()
            print("   Gate handles validated successfully")

            print("\n[8] Running with server-side admission + continuous slack-filling...")
            primary_results, runahead_results = await controller.run_with_runahead(
                primary_items=primary_prompts,
                runahead_items=runahead_prompts,
                primary_tracker=primary_tracker,
                runahead_tracker=runahead_tracker,
            )

            # Collect admission stats from all servers
            stats_list = await asyncio.gather(*[h.get_admission_stats.remote() for h in gated_handles])
            return primary_results, runahead_results, stats_list

        print("\n" + "=" * 80)
        start_time = time.perf_counter()
        primary_results, runahead_results, all_admission_stats = asyncio.run(run_simulation())
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
        if runahead_tracker.requests:
            for _, req in sorted(runahead_tracker.requests.items(), key=lambda x: x[1].index):
                stop_info = f" ({req.stop_reason})" if req.stop_reason else ""
                print(
                    f"   [{req.index}] {req.status:10s}{stop_info:12s} | {req.token_count:3d} tok | "
                    f"{req.duration:.2f}s | server {req.server_idx}"
                )
        else:
            print("   (no runahead submitted - servers always busy)")

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Primary duration: {controller.primary_duration:.2f}s")
        print(f"Primary: {primary_tracker.completed}/{primary_tracker.total} completed")
        print(
            f"Runahead: {runahead_tracker.completed} completed, "
            f"{runahead_tracker.aborted} aborted, {runahead_tracker.rejected} rejected"
        )

        print("\n--- Controller Metrics ---")
        print(f"Feeder ticks: {controller.feeder_ticks}")
        print(f"Runahead submissions: {controller.runahead_submissions}")
        print(f"Backpressure events: {controller.backpressure_events}")
        print(f"Runahead requeues (retries): {controller.runahead_requeues}")
        print(f"Runahead dropped (max retries exceeded): {controller.runahead_dropped}")
        if controller.runahead_requeues == 0:
            print("   (Note: Retries trigger in multi-worker scenarios when server rejects due to global limit)")

        print("\n--- Server Manager Metrics ---")
        sm = server_manager
        print(f"Total requests: {sm.total_requests}")
        print(f"Primary submitted: {sm.primary_submitted}")
        print(f"Runahead submitted: {sm.runahead_submitted}")
        print(f"Runahead completed: {sm.runahead_completed}")
        print(f"Runahead aborted: {sm.runahead_aborted}")
        print(f"Runahead rejected (server-side): {sm.runahead_rejected}")
        print(f"Client-side rejections: {sm.client_side_rejections}")
        print(f"Runahead inflight per server (final): {sm.runahead_inflight_per_server}")

        print("\n--- Server-Side Admission Stats (per server) ---")
        for i, stats in enumerate(all_admission_stats):
            print(f"   Server {i}:")
            print(f"      max_observed_inflight: {stats['runahead_max_observed']} / {stats['max_runahead_inflight']}")
            print(f"      rejected_total: {stats['runahead_rejected_total']}")

        print("\n--- Key Features Demonstrated ---")
        print("1. Two-layer admission: client-side slack check + server-side global limit")
        print("2. Continuous drip-feeding: runahead submitted during primary execution")
        print("3. Server-side rejection: stop_reason='rejected' handled gracefully")
        print(f"4. Global limit enforced: max {cfg.max_runahead_inflight} runahead per server (across all workers)")
        print(f"5. Retry on rejection: requeue to back of queue (max {cfg.max_runahead_retries} retries)")

        print("\n--- Safety Check ---")
        oversubscribed = False
        for i, stats in enumerate(all_admission_stats):
            if stats["runahead_max_observed"] > cfg.max_runahead_inflight:
                print(f"   ERROR: Server {i} oversubscribed: {stats['runahead_max_observed']} > {cfg.max_runahead_inflight}")
                oversubscribed = True
        if not oversubscribed:
            print("   No oversubscription detected (global limit enforced correctly)")

        print("\n" + "=" * 80)

        assert primary_tracker.completed == primary_tracker.total, "All primary should complete"
        assert not oversubscribed, "No server should be oversubscribed"
        print("\nTest PASSED!")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_server_side_admission_prototype()
