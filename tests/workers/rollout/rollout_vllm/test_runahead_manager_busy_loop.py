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
CPU-only tests for the manager-level Ray-native runahead busy loop.

These tests use mock Ray actors (no GPU, no vLLM dependency) to validate:
1) RunaheadCentralRouter admission control + targeted abort plumbing
2) AgentLoopManager.generate_sequences_with_runahead() primary output ordering
3) Secondary interleaving and abort when primary completes

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
from verl.experimental.agent_loop.runahead import RunaheadConfig
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


def test_router_rejects_when_at_capacity() -> None:
    _log("\n[TEST] Router rejects when at capacity")
    server = MockVLLMServer.remote(0, delay_s=0.5)
    router = RunaheadCentralRouter.remote([server], load_threshold=1)

    # Submit first request and wait until the server has actually started it.
    # Polling the router actor itself can starve the async task on some Ray configs.
    _log("  - Submit secondary #1 (expect admit)")
    ref1 = router.generate_secondary.remote(uuid4().hex, prompt_ids=[1, 2, 3], sampling_params={})
    deadline = time.time() + 10.0
    while True:
        stats = ray.get(server.get_stats.remote())
        if stats["active_requests"] >= 1:
            _log(f"  - Server active_requests={stats['active_requests']}")
            break
        if time.time() > deadline:
            raise AssertionError(f"Timed out waiting for server to start request, stats={stats}")
        time.sleep(0.05)

    # With threshold=1 and one server, a second request should be rejected.
    _log("  - Submit secondary #2 (expect reject)")
    out2 = ray.get(router.generate_secondary.remote(uuid4().hex, prompt_ids=[4, 5], sampling_params={}))
    assert out2 is None

    out1 = ray.get(ref1)
    assert out1 is not None
    _log("  - OK")


def test_router_rejects_when_kv_cache_high() -> None:
    _log("\n[TEST] Router rejects when kv_cache_usage is high")
    server = MockVLLMServer.remote(0, delay_s=0.01, kv_cache_usage=0.95)
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

    # High kv cache usage should reject secondary even with load slack.
    _log("  - Submit secondary with kv_cache_usage=0.95 (expect reject)")
    out = ray.get(router.generate_secondary.remote(uuid4().hex, prompt_ids=[1, 2, 3], sampling_params={}))
    assert out is None

    # Lower kv cache usage and refresh workload; admission should succeed.
    _log("  - Set kv_cache_usage=0.10 and refresh cache (expect admit)")
    ray.get(server.set_kv_cache_usage.remote(0.1))
    ray.get(router.refresh_workload_cache.remote())
    out2 = ray.get(router.generate_secondary.remote(uuid4().hex, prompt_ids=[4, 5], sampling_params={}))
    assert out2 is not None

    ray.get(router.configure_workload_polling.remote(enabled=False))
    _log("  - OK")


def test_manager_busy_loop_e2e() -> None:
    _log("\n[TEST] Manager busy-loop E2E (mock servers/workers)")
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
        poll_interval_s=0.01,
        max_retries=0,
        max_secondary_concurrent=4,
    )
    _log(
        "  - Config:"
        f" load_threshold={cfg.load_threshold},"
        f" poll_interval_s={cfg.poll_interval_s},"
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

    # With long server delay, no secondary should complete before primary, and all started should be aborted.
    assert result.metrics.secondary_started == cfg.max_secondary_concurrent
    assert result.metrics.secondary_completed == 0
    assert result.metrics.secondary_aborted == cfg.max_secondary_concurrent
    assert server_stats["requests_aborted"] >= 1
    aborted_outputs = [o for o in result.secondary_outputs if o.status == "aborted"]
    assert len(aborted_outputs) == cfg.max_secondary_concurrent
    assert all(o.output is not None for o in aborted_outputs)
    assert all(getattr(o.output, "stop_reason", None) == "aborted" for o in aborted_outputs)
    assert all(o.tokens_generated > 0 for o in aborted_outputs)
    _log(f"  - Runahead metrics: {result.metrics}")
    _log("  - OK")


def test_manager_mixed_secondary_max_tokens() -> None:
    _log("\n[TEST] Manager supports per-sample secondary max_tokens")
    server = MockVLLMServer.remote(0, delay_s=0.01)
    router = RunaheadCentralRouter.remote([server], load_threshold=10)

    # Keep primary running long enough to complete multiple secondaries.
    workers = [
        MockAgentLoopWorker.remote(worker_id=0, delay_s=0.4),
        MockAgentLoopWorker.remote(worker_id=1, delay_s=0.4),
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
        poll_interval_s=0.01,
        max_retries=0,
        max_secondary_concurrent=2,
    )

    result = manager.generate_sequences_with_runahead(primary_prompts, secondary_prompts, cfg)
    completed = [s for s in result.secondary_outputs if s.status == "completed"]
    assert completed, result.metrics

    # Long max_tokens should yield longer token_ids for completed samples.
    by_sample_id = {s.sample_id: s.tokens_generated for s in completed}
    assert by_sample_id.get("secondary_0") == 16
    assert by_sample_id.get("secondary_1") == 16
    assert by_sample_id.get("secondary_2") == 4

    server_stats = ray.get(server.get_stats.remote())
    seen = list(server_stats["request_sampling_params"].values())
    assert any(p.get("max_tokens") == 32 for p in seen)
    assert any(p.get("max_tokens") == 4 for p in seen)
    _log("  - OK")


def main() -> None:
    global _VERBOSE
    _VERBOSE = os.environ.get("VERL_TEST_VERBOSE", "1").strip() not in ("0", "false", "False")

    if _VERBOSE:
        print("=" * 80)
        print("Runahead manager busy-loop tests (CPU-only)")
        print("=" * 80)

    if ray.is_initialized():
        ray.shutdown()
    _log("[TEST] ray.init()")
    ray.init(ignore_reinit_error=True)
    try:
        test_router_rejects_when_at_capacity()
        test_router_rejects_when_kv_cache_high()
        test_manager_busy_loop_e2e()
        test_manager_mixed_secondary_max_tokens()
        print("\n[OK] runahead manager busy-loop tests passed")
    finally:
        _log("[TEST] ray.shutdown()")
        ray.shutdown()


if __name__ == "__main__":
    main()
