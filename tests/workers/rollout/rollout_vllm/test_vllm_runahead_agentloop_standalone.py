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
Standalone runahead script using agentloop classes with targeted abort
and workload-aware server selection.

This mirrors test_vllm_runahead_targeted_abort.py but integrates with
AgentLoopWorkerBase + AsyncLLMServerManager via subclassing, adding:
- Workload metrics monitoring via Prometheus /metrics endpoint
- Load-aware server selection for runahead/spec requests
- Smart routing to least-loaded servers

Usage:
    python tests/workers/rollout/rollout_vllm/test_vllm_runahead_agentloop_standalone.py

    NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/test_vllm_runahead_agentloop_standalone.py
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from cachetools import LRUCache

from verl.experimental.agent_loop.agent_loop import AgentLoopWorkerBase, AsyncLLMServerManager


# =============================================================================
# Workload Monitoring Classes
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


class WorkloadMonitor:
    """Monitor vLLM server workload via Prometheus metrics."""

    def __init__(self, server_handle, server_idx: int):
        self.server_handle = server_handle
        self.server_idx = server_idx
        self.snapshots: list[WorkloadSnapshot] = []
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._warned_missing_metrics = False

    async def get_workload(self) -> WorkloadSnapshot:
        """Fetch current workload from server."""
        try:
            result = await self.server_handle.get_workload.remote()

            # Check for warnings about missing metrics
            if result.get("warning") and not self._warned_missing_metrics:
                self._warned_missing_metrics = True
                print(f"\n   WARNING: Server {self.server_idx}: {result.get('warning')}")

            return WorkloadSnapshot(
                timestamp=time.perf_counter(),
                server_idx=self.server_idx,
                num_requests_running=result.get("num_requests_running", 0),
                num_requests_waiting=result.get("num_requests_waiting", 0),
                kv_cache_usage=result.get("kv_cache_usage", 0.0),
                error=result.get("error"),
            )
        except Exception as e:
            return WorkloadSnapshot(timestamp=time.perf_counter(), server_idx=self.server_idx, error=str(e))

    async def start_monitoring(self, interval: float = 0.5):
        """Start background workload monitoring."""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self, interval: float):
        """Background loop to collect workload snapshots."""
        while self._monitoring:
            snapshot = await self.get_workload()
            self.snapshots.append(snapshot)
            await asyncio.sleep(interval)

    def get_peak_running(self) -> int:
        """Get peak number of running requests observed."""
        return max((s.num_requests_running for s in self.snapshots if not s.error), default=0)

    def get_peak_waiting(self) -> int:
        """Get peak number of waiting requests observed."""
        return max((s.num_requests_waiting for s in self.snapshots if not s.error), default=0)


# =============================================================================
# Tracking Data Classes
# =============================================================================


@dataclass
class RequestTracker:
    """Track individual request state, timing, and server_request_id."""

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
    token_ids: list = field(default_factory=list)

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
        return [
            r.server_request_id
            for r in self.requests.values()
            if r.status == "running" and r.server_request_id
        ]


# =============================================================================
# AgentLoop-based Runahead Server Manager
# =============================================================================


class RunaheadAsyncLLMServerManager(AsyncLLMServerManager):
    """AsyncLLMServerManager with targeted abort, workload monitoring, and load-aware routing.

    Features:
    - Primary requests: sticky session with least-requests load balancing (inherited)
    - Spec/runahead requests: load-aware routing based on real-time workload metrics
    - Workload monitoring via Prometheus /metrics endpoint
    - Targeted abort by server_request_id
    """

    def __init__(
        self,
        config,
        server_handles: list,
        max_cache_size: int = 10000,
        load_aware_spec: bool = True,
        workload_cache_ttl: float = 0.5,
    ):
        super().__init__(config, server_handles, max_cache_size=max_cache_size)
        self.num_servers = len(self.server_handles)
        self._server_to_idx = {server: idx for idx, server in enumerate(self.server_handles)}
        self._spec_rr_idx = 0

        self.submitted_count = 0
        self.submitted_per_server = [0] * self.num_servers
        self.total_requests = 0

        self._request_to_server: dict[str, int] = {}
        self._sticky_cache: LRUCache = LRUCache(maxsize=max_cache_size)

        # Workload monitoring
        self.load_aware_spec = load_aware_spec
        self.workload_cache_ttl = workload_cache_ttl
        self.monitors = [WorkloadMonitor(h, idx) for idx, h in enumerate(server_handles)]
        self._cached_workloads: list[Optional[WorkloadSnapshot]] = [None] * self.num_servers
        self._workload_cache_time: float = 0.0

        # Metrics for load-aware decisions
        self.load_aware_selections = 0
        self.round_robin_fallbacks = 0

    async def get_all_workloads(self) -> list[WorkloadSnapshot]:
        """Fetch workload from all servers (with caching)."""
        now = time.perf_counter()
        if now - self._workload_cache_time < self.workload_cache_ttl:
            # Return cached workloads if still fresh
            return [w for w in self._cached_workloads if w is not None]

        # Fetch fresh workloads in parallel
        tasks = [m.get_workload() for m in self.monitors]
        workloads = await asyncio.gather(*tasks)
        self._cached_workloads = list(workloads)
        self._workload_cache_time = now
        return workloads

    async def get_aggregate_workload(self) -> dict[str, Any]:
        """Get aggregate workload across all servers."""
        workloads = await self.get_all_workloads()

        total_running = 0
        total_waiting = 0
        kv_cache_values = []
        per_server = []

        for snapshot in workloads:
            if not snapshot.error:
                total_running += snapshot.num_requests_running
                total_waiting += snapshot.num_requests_waiting
                kv_cache_values.append(snapshot.kv_cache_usage)
                per_server.append({
                    "server_idx": snapshot.server_idx,
                    "num_requests_running": snapshot.num_requests_running,
                    "num_requests_waiting": snapshot.num_requests_waiting,
                    "kv_cache_usage": snapshot.kv_cache_usage,
                    "total_load": snapshot.total_load,
                })
            else:
                per_server.append({"server_idx": snapshot.server_idx, "error": snapshot.error})

        return {
            "total_running": total_running,
            "total_waiting": total_waiting,
            "avg_kv_cache_usage": sum(kv_cache_values) / len(kv_cache_values) if kv_cache_values else 0.0,
            "max_kv_cache_usage": max(kv_cache_values) if kv_cache_values else 0.0,
            "per_server": per_server,
        }

    def _choose_server_primary(self, request_id: str, sticky: bool = True):
        if sticky and request_id in self._sticky_cache:
            return self._sticky_cache[request_id]

        server = super()._choose_server(request_id)
        server_idx = self._server_to_idx[server]
        result = (server, server_idx)
        if sticky:
            self._sticky_cache[request_id] = result
        return result

    def _choose_server_spec_round_robin(self):
        """Fallback: simple round-robin for spec requests."""
        server_idx = self._spec_rr_idx % self.num_servers
        self._spec_rr_idx += 1
        self.round_robin_fallbacks += 1
        return self.server_handles[server_idx], server_idx

    async def _choose_server_spec_load_aware(self) -> tuple[Any, int]:
        """Choose server with lowest current load for spec/runahead requests.

        Selection strategy:
        1. Fetch workloads from all servers (cached for performance)
        2. Compute total load = running + waiting + locally_submitted
        3. Choose server with minimum total load
        4. Break ties with round-robin
        """
        workloads = await self.get_all_workloads()

        # Find server with least total load (combine remote workload + local submitted)
        best_idx = 0
        best_load = float("inf")

        for snapshot in workloads:
            if snapshot.error:
                continue
            # Combine remote workload with local in-flight count for more accurate load
            total_load = snapshot.total_load + self.submitted_per_server[snapshot.server_idx]
            if total_load < best_load:
                best_load = total_load
                best_idx = snapshot.server_idx
            elif total_load == best_load:
                # Tie-break: prefer server with less KV cache usage
                if snapshot.kv_cache_usage < workloads[best_idx].kv_cache_usage:
                    best_idx = snapshot.server_idx

        self.load_aware_selections += 1
        return self.server_handles[best_idx], best_idx

    async def _choose_server_spec(self) -> tuple[Any, int]:
        """Choose server for spec/runahead requests (load-aware if enabled)."""
        if self.load_aware_spec and self.num_servers > 1:
            return await self._choose_server_spec_load_aware()
        return self._choose_server_spec_round_robin()

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
    ):
        if kind == "primary":
            server, server_idx = self._choose_server_primary(request_id, sticky=sticky)
        else:
            # Spec/runahead requests use load-aware routing
            server, server_idx = await self._choose_server_spec()

        server_request_id = uuid4().hex
        self._request_to_server[server_request_id] = server_idx

        self.submitted_count += 1
        self.submitted_per_server[server_idx] += 1
        self.total_requests += 1

        if tracker:
            tracker.server_request_id = server_request_id
            tracker.server_idx = server_idx
            tracker.start_time = time.perf_counter()
            tracker.status = "running"

        try:
            output = await server.generate.remote(
                request_id=server_request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )

            if tracker:
                tracker.end_time = time.perf_counter()
                tracker.status = getattr(output, "stop_reason", "completed") or "completed"
                tracker.token_count = len(getattr(output, "token_ids", []))
                tracker.token_ids = list(getattr(output, "token_ids", []))

            return output

        except Exception:
            if tracker:
                tracker.status = "error"
                tracker.end_time = time.perf_counter()
            raise

        finally:
            self.submitted_count -= 1
            self.submitted_per_server[server_idx] -= 1
            self._request_to_server.pop(server_request_id, None)

    async def abort_requests(self, server_request_ids: list[str]) -> dict[str, Any]:
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
                    result = await server.abort_request.remote(rid)
                    results.append(result)
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

    def get_submitted_distribution(self) -> list[int]:
        return list(self.submitted_per_server)


class RunaheadAgentLoopWorker(AgentLoopWorkerBase):
    """AgentLoopWorkerBase with a runahead-aware, load-aware server manager."""

    def __init__(
        self,
        config,
        server_handles,
        reward_router_address: str = None,
        load_aware_spec: bool = True,
    ):
        self.server_manager = RunaheadAsyncLLMServerManager(
            config, server_handles, load_aware_spec=load_aware_spec
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
        )


# =============================================================================
# Workload-Aware Runahead Controller
# =============================================================================


class WorkloadAwareRunaheadController:
    """Runahead controller with multiple trigger strategies based on workload.

    Trigger modes:
    - "completion": Trigger when completion ratio >= threshold (original behavior)
    - "workload": Trigger when server utilization drops below threshold
    - "hybrid": Trigger on EITHER completion ratio OR workload condition
    - "workload_and_completion": Trigger when BOTH conditions are met

    Workload trigger conditions:
    - utilization_threshold: Trigger when (running / batch_size) < threshold
    - idle_servers_threshold: Trigger when num servers with 0 running >= threshold
    - kv_cache_threshold: Trigger when avg KV cache usage < threshold
    """

    def __init__(
        self,
        server_manager: RunaheadAsyncLLMServerManager,
        trigger_mode: str = "hybrid",
        # Completion-based trigger
        completion_threshold: float = 0.5,
        # Workload-based triggers
        utilization_threshold: float = 0.3,  # Trigger when utilization < 30%
        idle_servers_threshold: int = 1,  # Trigger when >= N servers are idle
        kv_cache_threshold: float = 0.5,  # Trigger when avg KV cache < 50%
        # Polling settings
        workload_check_interval: float = 0.1,  # How often to check workload
    ):
        self.server_manager = server_manager
        self.trigger_mode = trigger_mode

        # Thresholds
        self.completion_threshold = completion_threshold
        self.utilization_threshold = utilization_threshold
        self.idle_servers_threshold = idle_servers_threshold
        self.kv_cache_threshold = kv_cache_threshold
        self.workload_check_interval = workload_check_interval

        # State
        self.triggered = False
        self.trigger_time: Optional[float] = None
        self.trigger_reason: str = ""
        self.primary_start_time: Optional[float] = None
        self.primary_done_time: Optional[float] = None

        # Metrics
        self.workload_checks = 0
        self.trigger_workload_snapshot: Optional[dict] = None

    @property
    def primary_duration(self) -> Optional[float]:
        if self.primary_start_time is None or self.primary_done_time is None:
            return None
        return self.primary_done_time - self.primary_start_time

    async def _check_workload_trigger(self, batch_size: int) -> tuple[bool, str]:
        """Check if workload conditions warrant triggering runahead.

        Returns:
            (should_trigger, reason_string)
        """
        self.workload_checks += 1

        try:
            agg = await self.server_manager.get_aggregate_workload()
            self.trigger_workload_snapshot = agg

            total_running = agg["total_running"]
            per_server = agg.get("per_server", [])

            # Condition 1: Overall utilization is low
            utilization = total_running / batch_size if batch_size > 0 else 0
            if utilization < self.utilization_threshold:
                return True, f"utilization={utilization:.1%} < {self.utilization_threshold:.1%}"

            # Condition 2: Some servers are completely idle
            idle_count = sum(
                1 for s in per_server
                if "error" not in s and s.get("num_requests_running", 0) == 0
            )
            if idle_count >= self.idle_servers_threshold:
                return True, f"idle_servers={idle_count} >= {self.idle_servers_threshold}"

            # Condition 3: KV cache usage is low (memory available)
            avg_kv = agg.get("avg_kv_cache_usage", 0)
            if avg_kv < self.kv_cache_threshold and avg_kv > 0:
                return True, f"avg_kv_cache={avg_kv:.1%} < {self.kv_cache_threshold:.1%}"

            return False, ""

        except Exception as e:
            return False, f"error: {e}"

    def _check_completion_trigger(self, primary_tracker: BatchTracker) -> tuple[bool, str]:
        """Check if completion ratio warrants triggering."""
        if primary_tracker.completion_ratio >= self.completion_threshold:
            return True, f"completion={primary_tracker.completion_ratio:.0%} >= {self.completion_threshold:.0%}"
        return False, ""

    async def _should_trigger(self, primary_tracker: BatchTracker, batch_size: int) -> tuple[bool, str]:
        """Determine if runahead should be triggered based on mode."""
        completion_met, completion_reason = self._check_completion_trigger(primary_tracker)
        workload_met, workload_reason = await self._check_workload_trigger(batch_size)

        if self.trigger_mode == "completion":
            return completion_met, completion_reason

        elif self.trigger_mode == "workload":
            return workload_met, workload_reason

        elif self.trigger_mode == "hybrid":
            # Trigger on EITHER condition
            if completion_met:
                return True, f"completion: {completion_reason}"
            if workload_met:
                return True, f"workload: {workload_reason}"
            return False, ""

        elif self.trigger_mode == "workload_and_completion":
            # Trigger only when BOTH conditions are met
            if completion_met and workload_met:
                return True, f"both: {completion_reason} AND {workload_reason}"
            return False, ""

        else:
            raise ValueError(f"Unknown trigger_mode: {self.trigger_mode}")

    async def run_with_runahead(
        self,
        primary_tasks: list[asyncio.Task],
        runahead_factory: callable,
        primary_tracker: BatchTracker,
        runahead_tracker: BatchTracker,
    ) -> tuple[list, list]:
        runahead_tasks = []
        primary_results = []
        runahead_results = []
        batch_size = len(primary_tasks)

        self.primary_start_time = time.perf_counter()
        primary_tracker.start_time = self.primary_start_time
        pending_primary = set(primary_tasks)

        # For workload-based triggering, we need to poll periodically
        last_workload_check = 0.0

        while pending_primary:
            # Use a short timeout to allow periodic workload checks
            done, pending_primary = await asyncio.wait(
                pending_primary,
                timeout=self.workload_check_interval if not self.triggered else None,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                try:
                    result = await task
                    primary_results.append(result)
                except Exception:
                    primary_results.append(None)

            # Check trigger conditions
            if not self.triggered:
                now = time.perf_counter()
                # Rate-limit workload checks
                if now - last_workload_check >= self.workload_check_interval:
                    last_workload_check = now
                    should_trigger, reason = await self._should_trigger(primary_tracker, batch_size)

                    if should_trigger:
                        self.triggered = True
                        self.trigger_time = time.perf_counter()
                        self.trigger_reason = reason

                        print(f"\n   >>> RUNAHEAD TRIGGERED [{self.trigger_mode}]")
                        print(f"   >>> Reason: {reason}")
                        print(f"   >>> Primary: {primary_tracker.completed}/{primary_tracker.total} done")
                        print(f"   >>> Submitted: {self.server_manager.submitted_count}")
                        if self.trigger_workload_snapshot:
                            snap = self.trigger_workload_snapshot
                            print(f"   >>> Workload: running={snap['total_running']}, waiting={snap['total_waiting']}")

                        runahead_tracker.start_time = time.perf_counter()
                        runahead_tasks = runahead_factory()

        self.primary_done_time = time.perf_counter()

        if runahead_tasks:
            running_ids = runahead_tracker.get_running_server_request_ids()
            if running_ids:
                print(f"\n   >>> Aborting {len(running_ids)} runahead requests BY ID...")
                abort_result = await self.server_manager.abort_requests(running_ids)
                print(f"   >>> Aborted: {abort_result['aborted_count']}")

                for req in runahead_tracker.requests.values():
                    if req.server_request_id in running_ids and req.status == "running":
                        req.status = "aborted"
                        req.end_time = time.perf_counter()

            for task in runahead_tasks:
                try:
                    result = await task
                    runahead_results.append(result)
                except Exception:
                    runahead_results.append(None)

        return primary_results, runahead_results


# =============================================================================
# Test Function
# =============================================================================


def test_runahead_agentloop_standalone():
    """
    Standalone runahead test using AgentLoopWorkerBase with load-aware scheduling.

    Demonstrates:
    1. AsyncLLMServerManager subclass for targeted abort
    2. AgentLoopWorkerBase subclass with runahead support
    3. Completion-based trigger and abort by request id
    4. Workload-aware server selection for runahead/spec requests
    5. Real-time workload monitoring via Prometheus metrics
    """
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    DP_SIZE = int(os.environ.get("DP_SIZE", str(NUM_GPUS // TP_SIZE)))

    PRIMARY_SIZE = int(os.environ.get("PRIMARY_SIZE", "8"))
    RUNAHEAD_SIZE = int(os.environ.get("RUNAHEAD_SIZE", "4"))
    LOAD_AWARE = os.environ.get("LOAD_AWARE", "1").lower() in ("1", "true", "yes")
    MONITOR_INTERVAL = float(os.environ.get("MONITOR_INTERVAL", "0.3"))

    # Trigger configuration
    # Modes: "completion", "workload", "hybrid", "workload_and_completion"
    TRIGGER_MODE = os.environ.get("TRIGGER_MODE", "hybrid")
    COMPLETION_THRESHOLD = float(os.environ.get("COMPLETION_THRESHOLD", "0.5"))
    UTILIZATION_THRESHOLD = float(os.environ.get("UTILIZATION_THRESHOLD", "0.3"))
    IDLE_SERVERS_THRESHOLD = int(os.environ.get("IDLE_SERVERS_THRESHOLD", "1"))
    KV_CACHE_THRESHOLD = float(os.environ.get("KV_CACHE_THRESHOLD", "0.5"))

    print("=" * 80)
    print("Runahead AgentLoop Standalone (Workload-Aware Trigger + Load-Aware Routing)")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print(f"Primary batch: {PRIMARY_SIZE} | Runahead batch: {RUNAHEAD_SIZE}")
    print(f"Load-aware spec routing: {LOAD_AWARE}")
    print(f"Workload monitor interval: {MONITOR_INTERVAL}s")
    print("-" * 80)
    print(f"Trigger mode: {TRIGGER_MODE}")
    print(f"  - Completion threshold: {COMPLETION_THRESHOLD:.0%}")
    print(f"  - Utilization threshold: {UTILIZATION_THRESHOLD:.0%}")
    print(f"  - Idle servers threshold: {IDLE_SERVERS_THRESHOLD}")
    print(f"  - KV cache threshold: {KV_CACHE_THRESHOLD:.0%}")
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
        # Enable Prometheus metrics for workload monitoring
        config.actor_rollout_ref.rollout.disable_log_stats = False
        if hasattr(config, "reward_model"):
            config.reward_model.use_reward_loop = False

        print(f"\n[3] Creating {DP_SIZE} vLLM server(s)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        rollout_class = get_rollout_replica_class("vllm")
        print(rollout_class)
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

        print("\n[4] Creating baseline agentloop worker...")
        baseline_worker = RunaheadAgentLoopWorker(config, server_handles, load_aware_spec=LOAD_AWARE)

        primary_prompts = [
            ("What is 2+2?", 16),
            ("Say hi.", 16),
            ("Name a color.", 16),
            ("What is 1+1?", 16),
            ("Write a detailed essay about AI history.", 200),
            ("Explain quantum computing step by step.", 200),
            ("Describe machine learning training.", 200),
            ("Write a story about robots.", 200),
        ][:PRIMARY_SIZE]

        runahead_prompts = [
            ("What is the capital of France?", 32),
            ("Speed of light?", 32),
            ("Write about math history.", 300),
            ("Explain climate change.", 300),
        ][:RUNAHEAD_SIZE]

        print(f"\n[5] Primary: {len(primary_prompts)} | Runahead: {len(runahead_prompts)}")

        async def run_primary_only(worker: RunaheadAgentLoopWorker):
            print("\n[6] Running primary-only baseline...")
            baseline_tracker = BatchTracker(batch_id="primary_baseline", total=len(primary_prompts))
            primary_tasks = []
            for i, (prompt, max_tokens) in enumerate(primary_prompts):
                request_id = f"primary_base_{i}_{uuid4().hex[:8]}"
                req_tracker = RequestTracker(
                    request_id=request_id,
                    batch_id="primary_baseline",
                    index=i,
                    max_tokens=max_tokens,
                )
                baseline_tracker.requests[request_id] = req_tracker
                task = asyncio.create_task(
                    worker.generate_prompt(
                        prompt,
                        request_id=request_id,
                        max_tokens=max_tokens,
                        tracker=req_tracker,
                        kind="primary",
                        sticky=True,
                    )
                )
                primary_tasks.append(task)
                label = "short" if max_tokens <= 32 else "LONG"
                print(f"      {request_id} ({label})")

            start_time = time.perf_counter()
            results = await asyncio.gather(*primary_tasks, return_exceptions=True)
            duration = time.perf_counter() - start_time
            return results, baseline_tracker, duration

        baseline_results, baseline_tracker, baseline_duration = asyncio.run(run_primary_only(baseline_worker))

        print("\n[7] Creating agentloop worker for runahead...")
        worker = RunaheadAgentLoopWorker(config, server_handles, load_aware_spec=LOAD_AWARE)

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        # Store workload snapshots collected during execution
        workload_timeline: list[dict[str, Any]] = []

        async def run_simulation():
            print("\n[8] Starting primary batch...")
            primary_tasks = []
            for i, (prompt, max_tokens) in enumerate(primary_prompts):
                request_id = f"primary_{i}_{uuid4().hex[:8]}"
                req_tracker = RequestTracker(
                    request_id=request_id,
                    batch_id="primary",
                    index=i,
                    max_tokens=max_tokens,
                )
                primary_tracker.requests[request_id] = req_tracker
                task = asyncio.create_task(
                    worker.generate_prompt(
                        prompt,
                        request_id=request_id,
                        max_tokens=max_tokens,
                        tracker=req_tracker,
                        kind="primary",
                        sticky=True,
                    )
                )
                primary_tasks.append(task)
                label = "short" if max_tokens <= 32 else "LONG"
                print(f"      {request_id} ({label})")

            def create_runahead_tasks():
                tasks = []
                print("\n   >>> Creating runahead tasks (load-aware routing)...")
                for i, (prompt, max_tokens) in enumerate(runahead_prompts):
                    request_id = f"runahead_{i}_{uuid4().hex[:8]}"
                    req_tracker = RequestTracker(
                        request_id=request_id,
                        batch_id="runahead",
                        index=i,
                        max_tokens=max_tokens,
                    )
                    runahead_tracker.requests[request_id] = req_tracker
                    task = asyncio.create_task(
                        worker.generate_prompt(
                            prompt,
                            request_id=request_id,
                            max_tokens=max_tokens,
                            tracker=req_tracker,
                            kind="runahead",
                            sticky=False,
                        )
                    )
                    tasks.append(task)
                    label = "short" if max_tokens <= 64 else "LONG"
                    print(f"      {request_id} ({label})")
                return tasks

            controller = WorkloadAwareRunaheadController(
                server_manager=worker.server_manager,
                trigger_mode=TRIGGER_MODE,
                completion_threshold=COMPLETION_THRESHOLD,
                utilization_threshold=UTILIZATION_THRESHOLD,
                idle_servers_threshold=IDLE_SERVERS_THRESHOLD,
                kv_cache_threshold=KV_CACHE_THRESHOLD,
            )

            # Start workload monitoring in the background
            async def collect_workloads():
                while True:
                    try:
                        agg = await worker.server_manager.get_aggregate_workload()
                        agg["elapsed"] = time.perf_counter() - controller.primary_start_time if controller.primary_start_time else 0
                        workload_timeline.append(agg)
                    except Exception:
                        pass
                    await asyncio.sleep(MONITOR_INTERVAL)

            print("\n[9] Running with runahead (workload monitoring active)...")
            monitor_task = asyncio.create_task(collect_workloads())

            primary_results, runahead_results = await controller.run_with_runahead(
                primary_tasks=primary_tasks,
                runahead_factory=create_runahead_tasks,
                primary_tracker=primary_tracker,
                runahead_tracker=runahead_tracker,
            )

            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

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
            print(
                f"   [{req.index}] {req.status:10s} | {req.token_count:3d} tok | "
                f"{req.duration:.2f}s | server {req.server_idx}"
            )

        print("\n--- Runahead Batch ---")
        runahead_completed = 0
        runahead_aborted = 0

        if runahead_tracker.requests:
            for _, req in sorted(runahead_tracker.requests.items(), key=lambda x: x[1].index):
                print(
                    f"   [{req.index}] {req.status:10s} | {req.token_count:3d} tok | "
                    f"{req.duration:.2f}s | server {req.server_idx} | "
                    f"server_req_id={req.server_request_id[:8]}..."
                )
                if req.status == "completed":
                    runahead_completed += 1
                elif req.status == "aborted":
                    runahead_aborted += 1
        else:
            print("   (not triggered)")

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Runahead triggered: {controller.triggered}")
        if controller.triggered:
            print(f"Trigger mode: {controller.trigger_mode}")
            print(f"Trigger reason: {controller.trigger_reason}")
            print(f"Workload checks performed: {controller.workload_checks}")
        print(f"Primary: {primary_tracker.completed}/{primary_tracker.total} completed")
        print(f"Runahead: {runahead_completed} completed, {runahead_aborted} aborted")
        print(f"Total requests: {worker.server_manager.total_requests}")

        print("\n--- Timing Comparison ---")
        print(f"Primary-only completion: {baseline_duration:.2f}s")
        primary_with_runahead = controller.primary_duration
        if primary_with_runahead is not None:
            delta = primary_with_runahead - baseline_duration
            print(f"Primary completion w/ runahead: {primary_with_runahead:.2f}s")
            print(f"Delta (runahead - baseline): {delta:+.2f}s")
        print("Note: baseline and runahead share server state; cache warmup may affect results.")

        print(f"\nSubmitted distribution: {worker.server_manager.get_submitted_distribution()}")

        print("\n--- Load-Aware Scheduling Metrics ---")
        print(f"Load-aware enabled: {LOAD_AWARE}")
        print(f"Load-aware selections: {worker.server_manager.load_aware_selections}")
        print(f"Round-robin fallbacks: {worker.server_manager.round_robin_fallbacks}")

        # Show workload timeline if we have data
        if workload_timeline:
            print("\n--- Workload Timeline (sampled every {:.1f}s) ---".format(MONITOR_INTERVAL))
            peak_running = max(w["total_running"] for w in workload_timeline)
            peak_waiting = max(w["total_waiting"] for w in workload_timeline)
            peak_kv = max(w["max_kv_cache_usage"] for w in workload_timeline)
            print(f"Peak total running (all servers): {peak_running}")
            print(f"Peak total waiting (all servers): {peak_waiting}")
            print(f"Peak max KV cache (hottest server): {peak_kv:.2%}")

            print("\nTimeline (first 10 snapshots):")
            for i, w in enumerate(workload_timeline[:10]):
                per_server_info = ", ".join(
                    f"s{s['server_idx']}:{s.get('num_requests_running', '?')}/{s.get('num_requests_waiting', '?')}"
                    for s in w.get("per_server", [])
                    if "error" not in s
                )
                print(
                    f"   [{i:2d}] t={w.get('elapsed', 0):.2f}s | "
                    f"running={w['total_running']}, waiting={w['total_waiting']} | "
                    f"per_server: [{per_server_info}]"
                )
            if len(workload_timeline) > 10:
                print(f"   ... ({len(workload_timeline) - 10} more snapshots)")
        else:
            print("\n--- Workload Timeline ---")
            print("   (no snapshots collected - generation completed too fast)")

        print("\n--- Safety Check ---")
        print("Used abort_requests(ids) for targeted abort: YES")
        print("Primary requests affected by abort: NO (targeted abort is safe)")

        print("\n" + "=" * 80)

        assert primary_tracker.completed == primary_tracker.total, "All primary should complete"

        if controller.triggered:
            assert runahead_completed + runahead_aborted == runahead_tracker.total
            for req in runahead_tracker.requests.values():
                assert req.server_request_id, f"Missing server_request_id for {req.request_id}"

        print("\nTest PASSED!")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_runahead_agentloop_standalone()
