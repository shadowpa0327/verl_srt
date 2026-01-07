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
CPU-only tests for the router-owned queue model runahead implementation.

These tests use mock Ray actors (no GPU, no vLLM dependency) to validate:
1) RunaheadCentralRouter batch API (start_runahead_batch, stop_runahead_batch)
2) Router-internal admit loop and queue management
3) AgentLoopManager.generate_sequences_with_runahead() with simplified API

Usage:
    python tests/workers/rollout/rollout_vllm/test_runahead_manager_busy_loop.py

Environment:
    VERL_TEST_VERBOSE: Set to 0 to reduce script output (default: 1 when run as __main__).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.router import RunaheadCentralRouter
from verl.experimental.agent_loop.runahead import RunaheadConfig, SecondaryWorkItem
from verl.protocol import DataProto


_VERBOSE = False


def _log(msg: str) -> None:
    if _VERBOSE:
        print(msg, flush=True)


@dataclass
class MockTokenOutput:
    token_ids: list[int] = field(default_factory=list)
    stop_reason: str = "completed"


@ray.remote
class MockVLLMServer:
    """Mock vLLM-like server actor for router/manager tests."""

    def __init__(self, server_id: int, delay_s: float = 1.0, kv_cache_usage: float = 0.0):
        self.server_id = server_id
        self.delay_s = delay_s
        self.kv_cache_usage = kv_cache_usage
        self.requests_received: list[str] = []
        self.requests_aborted: list[str] = []
        self.request_start_s: dict[str, float] = {}
        self.request_end_s: dict[str, float] = {}
        self.request_sampling_params: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()
        self._abort_events: dict[str, asyncio.Event] = {}

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> MockTokenOutput:
        self.requests_received.append(request_id)
        self.request_sampling_params[request_id] = dict(sampling_params)
        self._active.add(request_id)
        self.request_start_s[request_id] = time.perf_counter()
        abort_event = asyncio.Event()
        self._abort_events[request_id] = abort_event

        sleep_task = asyncio.create_task(asyncio.sleep(self.delay_s))
        abort_task = asyncio.create_task(abort_event.wait())
        done, pending = await asyncio.wait({sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        aborted = abort_task in done and sleep_task not in done
        self.request_end_s[request_id] = time.perf_counter()
        self._active.discard(request_id)
        self._abort_events.pop(request_id, None)

        max_tokens = sampling_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_new_tokens")
        try:
            max_tokens_int = int(max_tokens) if max_tokens is not None else 5
        except (TypeError, ValueError):
            max_tokens_int = 5
        max_tokens_int = max(1, max_tokens_int)

        if aborted:
            out_len = min(max_tokens_int, 8)
            return MockTokenOutput(token_ids=list(range(1, out_len + 1)), stop_reason="aborted")

        out_len = min(max_tokens_int, 16)
        return MockTokenOutput(token_ids=list(range(1, out_len + 1)), stop_reason="completed")

    async def abort_request(self, request_id: str) -> None:
        self.requests_aborted.append(request_id)
        abort_event = self._abort_events.get(request_id)
        if abort_event is not None:
            abort_event.set()
        self._active.discard(request_id)

    async def get_workload(self) -> dict[str, Any]:
        # Mirror vLLMHttpServerBase.get_workload() contract (best-effort).
        return {
            "num_requests_running": len(self._active),
            "num_requests_waiting": 0,
            "kv_cache_usage": float(self.kv_cache_usage),
        }

    def set_kv_cache_usage(self, kv_cache_usage: float) -> None:
        self.kv_cache_usage = float(kv_cache_usage)

    def get_stats(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "requests_received": len(self.requests_received),
            "requests_aborted": len(self.requests_aborted),
            "active_requests": len(self._active),
            "request_start_s": dict(self.request_start_s),
            "request_end_s": dict(self.request_end_s),
            "request_sampling_params": dict(self.request_sampling_params),
        }


@ray.remote
class MockAgentLoopWorker:
    """Mock AgentLoopWorker actor that returns the chunk after a delay."""

    def __init__(self, worker_id: int, delay_s: float):
        self.worker_id = worker_id
        self.delay_s = delay_s
        self.start_s: Optional[float] = None
        self.end_s: Optional[float] = None

    async def generate_sequences(self, chunk: DataProto) -> DataProto:
        self.start_s = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        self.end_s = time.perf_counter()

        out = DataProto(
            batch=chunk.batch,
            non_tensor_batch=dict(chunk.non_tensor_batch),
            meta_info=dict(chunk.meta_info),
        )
        out.non_tensor_batch["worker_id"] = np.full((len(out),), self.worker_id, dtype=np.int64)
        return out

    def get_stats(self) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "start_s": self.start_s, "end_s": self.end_s}


def _make_primary_dataproto(num_items: int) -> DataProto:
    return DataProto(non_tensor_batch={"idx": np.arange(num_items, dtype=np.int64)})


def _make_secondary_dataproto(
    num_items: int,
    seq_len: int = 8,
    sampling_params: Optional[list[dict[str, Any]]] = None,
) -> DataProto:
    input_ids = torch.zeros((num_items, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((num_items, seq_len), dtype=torch.long)
    for i in range(num_items):
        # Variable-length prompts to exercise attention_mask-based unpadding.
        prompt_len = 2 + (i % (seq_len - 2))
        input_ids[i, :prompt_len] = torch.arange(1, prompt_len + 1, dtype=torch.long)
        attention_mask[i, :prompt_len] = 1

    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=(num_items,),
    )
    non_tensor_batch: dict[str, Any] = {}
    if sampling_params is not None:
        if len(sampling_params) != num_items:
            raise ValueError(f"sampling_params length ({len(sampling_params)}) must match num_items ({num_items})")
        non_tensor_batch["sampling_params"] = np.array(sampling_params, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def test_router_batch_api_basic() -> None:
    _log("\n[TEST] Router batch API basic functionality")
    server = MockVLLMServer.remote(0, delay_s=0.1)
    router = RunaheadCentralRouter.remote([server], load_threshold=10)

    # Create work items
    work_items = [
        SecondaryWorkItem(sample_id="s0", prompt_ids=[1, 2, 3], sampling_params={}),
        SecondaryWorkItem(sample_id="s1", prompt_ids=[4, 5, 6], sampling_params={}),
    ]

    # Start batch
    _log("  - Start runahead batch")
    start_result = ray.get(router.start_runahead_batch.remote(
        work_items,
        max_concurrent=2,
        poll_interval_s=0.05,
    ))
    assert start_result["status"] == "started"
    assert start_result["queued"] == 2
    _log(f"  - Batch started: {start_result}")

    # Check status
    status = ray.get(router.get_runahead_batch_status.remote())
    assert status["batch_active"] is True
    _log(f"  - Batch status: {status}")

    # Wait a bit for work to complete
    time.sleep(0.5)

    # Stop batch and get results
    _log("  - Stop runahead batch")
    result = ray.get(router.stop_runahead_batch.remote(abort_grace_s=1.0))
    _log(f"  - Batch result: completed={result.metrics.secondary_completed}, aborted={result.metrics.secondary_aborted}")

    # Both should complete (short delay)
    assert result.metrics.secondary_completed == 2
    assert len([o for o in result.outputs if o.status == "completed"]) == 2
    _log("  - OK")


def test_router_respects_max_concurrent() -> None:
    _log("\n[TEST] Router respects max_concurrent limit")
    server = MockVLLMServer.remote(0, delay_s=0.5)
    router = RunaheadCentralRouter.remote([server], load_threshold=10)

    # Create many work items but limit concurrency
    work_items = [
        SecondaryWorkItem(sample_id=f"s{i}", prompt_ids=[i], sampling_params={})
        for i in range(10)
    ]

    ray.get(router.start_runahead_batch.remote(
        work_items,
        max_concurrent=2,  # Only 2 at a time
        poll_interval_s=0.05,
    ))

    # Check that in_flight never exceeds max_concurrent
    time.sleep(0.1)
    status = ray.get(router.get_runahead_batch_status.remote())
    assert status["in_flight_count"] <= 2
    _log(f"  - Status after 0.1s: {status}")

    # Stop and verify
    result = ray.get(router.stop_runahead_batch.remote(abort_grace_s=0.5))
    _log(f"  - Result: completed={result.metrics.secondary_completed}, "
         f"aborted={result.metrics.secondary_aborted}, rejected={result.metrics.secondary_rejected}")
    _log("  - OK")


def test_router_kv_cache_admission() -> None:
    _log("\n[TEST] Router respects kv_cache_usage in admission (via batch API)")
    server = MockVLLMServer.remote(0, delay_s=0.1, kv_cache_usage=0.95)
    router = RunaheadCentralRouter.remote([server], load_threshold=10)

    _log("  - Enable workload polling (prime cache, require fresh metrics)")
    ray.get(
        router.configure_workload_polling.remote(
            enabled=True,
            kv_cache_threshold=0.8,
            poll_interval_s=0.05,
            staleness_threshold_s=10.0,
            require_fresh_workload=True,
            prime_cache=True,
        )
    )

    # With high kv cache, admission should be blocked
    work_items = [SecondaryWorkItem(sample_id="s0", prompt_ids=[1, 2, 3], sampling_params={})]
    ray.get(router.start_runahead_batch.remote(work_items, max_concurrent=1, poll_interval_s=0.05))

    # Items should stay pending (not admitted due to high kv cache)
    time.sleep(0.2)
    status = ray.get(router.get_runahead_batch_status.remote())
    _log(f"  - Status with high kv_cache: {status}")

    # Lower kv cache and let it admit
    _log("  - Set kv_cache_usage=0.10 and refresh cache")
    ray.get(server.set_kv_cache_usage.remote(0.1))
    ray.get(router.refresh_workload_cache.remote())
    time.sleep(0.3)

    result = ray.get(router.stop_runahead_batch.remote(abort_grace_s=1.0))
    _log(f"  - Result: completed={result.metrics.secondary_completed}")

    ray.get(router.configure_workload_polling.remote(enabled=False))
    _log("  - OK")


def test_manager_busy_loop_e2e() -> None:
    _log("\n[TEST] Manager E2E with router-owned queue (mock servers/workers)")
    # One server; long secondary delay ensures secondary is still running when primary completes.
    server = MockVLLMServer.remote(0, delay_s=2.0)
    router = RunaheadCentralRouter.remote([server], load_threshold=10)

    # Worker 1 finishes before worker 0 (to catch completion-order concat bugs).
    workers = [
        MockAgentLoopWorker.remote(worker_id=0, delay_s=0.6),
        MockAgentLoopWorker.remote(worker_id=1, delay_s=0.1),
    ]

    manager = AgentLoopManager.__new__(AgentLoopManager)
    manager.router = router
    manager.agent_loop_workers = workers
    manager.reward_model_manager = None
    manager.wake_up = lambda: None
    manager.sleep = lambda: None

    primary_prompts = _make_primary_dataproto(num_items=4)
    secondary_prompts = _make_secondary_dataproto(num_items=16, seq_len=8)
    cfg = RunaheadConfig(
        enabled=True,
        load_threshold=10,
        max_secondary_concurrent=4,
        admit_loop_poll_s=0.05,
        wait_for_primary_start=False,
    )
    _log(
        "  - Config:"
        f" load_threshold={cfg.load_threshold},"
        f" admit_loop_poll_s={cfg.admit_loop_poll_s},"
        f" max_secondary_concurrent={cfg.max_secondary_concurrent}"
    )

    result = manager.generate_sequences_with_runahead(primary_prompts, secondary_prompts, cfg)
    assert result.primary_outputs is not None

    # Primary order must match original AgentLoopManager.generate_sequences() semantics (chunk order).
    assert np.array_equal(result.primary_outputs.non_tensor_batch["idx"], np.arange(4, dtype=np.int64))

    # Secondary should have been submitted while primary still running.
    worker_stats = ray.get([w.get_stats.remote() for w in workers])
    primary_end_s = max(s["end_s"] for s in worker_stats if s["end_s"] is not None)
    server_stats = ray.get(server.get_stats.remote())
    _log(f"  - Worker timing: {worker_stats}")
    _log(f"  - Server stats: {server_stats}")
    assert server_stats["requests_received"] > 0
    secondary_first_start_s = min(server_stats["request_start_s"].values())
    assert secondary_first_start_s < primary_end_s, (secondary_first_start_s, primary_end_s)

    # With long server delay, secondaries should be aborted when primary completes.
    # Note: With router-owned queue, items may be aborted or rejected depending on timing
    assert result.metrics.secondary_started <= cfg.max_secondary_concurrent
    # All started should be aborted (server returns stop_reason="aborted")
    assert result.metrics.secondary_aborted == result.metrics.secondary_started
    _log(f"  - Runahead metrics: {result.metrics}")

    # Verify abort was called
    assert server_stats["requests_aborted"] >= 1
    _log("  - OK")


def test_manager_mixed_secondary_max_tokens() -> None:
    _log("\n[TEST] Manager supports per-sample secondary max_tokens")
    server = MockVLLMServer.remote(0, delay_s=0.05)
    router = RunaheadCentralRouter.remote([server], load_threshold=10)

    # Keep primary running long enough to complete multiple secondaries.
    workers = [
        MockAgentLoopWorker.remote(worker_id=0, delay_s=0.8),
        MockAgentLoopWorker.remote(worker_id=1, delay_s=0.8),
    ]

    manager = AgentLoopManager.__new__(AgentLoopManager)
    manager.router = router
    manager.agent_loop_workers = workers
    manager.reward_model_manager = None
    manager.wake_up = lambda: None
    manager.sleep = lambda: None

    primary_prompts = _make_primary_dataproto(num_items=4)
    secondary_sampling_params = [{"max_tokens": 32} if i < 2 else {"max_tokens": 4} for i in range(8)]
    secondary_prompts = _make_secondary_dataproto(
        num_items=8,
        seq_len=8,
        sampling_params=secondary_sampling_params,
    )

    cfg = RunaheadConfig(
        enabled=True,
        load_threshold=10,
        max_secondary_concurrent=4,
        admit_loop_poll_s=0.05,
        wait_for_primary_start=False,
    )

    result = manager.generate_sequences_with_runahead(primary_prompts, secondary_prompts, cfg)
    completed = [s for s in result.secondary_outputs if s.status == "completed"]
    _log(f"  - Completed: {len(completed)}, metrics: {result.metrics}")

    # Should have some completed
    assert completed, f"No completed secondaries, metrics={result.metrics}"

    # Check that max_tokens was passed through
    server_stats = ray.get(server.get_stats.remote())
    seen = list(server_stats["request_sampling_params"].values())
    _log(f"  - Server saw sampling_params: {seen}")
    assert any(p.get("max_tokens") == 32 for p in seen), f"Expected max_tokens=32, got {seen}"
    _log("  - OK")


def main() -> None:
    global _VERBOSE
    _VERBOSE = os.environ.get("VERL_TEST_VERBOSE", "1").strip() not in ("0", "false", "False")

    if _VERBOSE:
        print("=" * 80)
        print("Runahead router-owned queue model tests (CPU-only)")
        print("=" * 80)

    if ray.is_initialized():
        ray.shutdown()
    _log("[TEST] ray.init()")
    ray.init(ignore_reinit_error=True)
    try:
        test_router_batch_api_basic()
        test_router_respects_max_concurrent()
        test_router_kv_cache_admission()
        test_manager_busy_loop_e2e()
        test_manager_mixed_secondary_max_tokens()
        print("\n[OK] runahead router-owned queue model tests passed")
    finally:
        _log("[TEST] ray.shutdown()")
        ray.shutdown()


if __name__ == "__main__":
    main()
