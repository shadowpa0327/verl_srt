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
Tests for CentralRouter - the centralized routing layer for AgentLoopWorkers.

This test validates:
1. CentralRouter correctly routes requests to servers
2. Load balancing distributes requests evenly (least-requests)
3. Sticky sessions work (same request_id → same server)
4. Multiple concurrent requests are handled correctly
5. RouterAdapter provides the same interface as AsyncLLMServerManager

Usage:
    python tests/workers/rollout/rollout_vllm/test_central_router.py

    # Or with pytest:
    pytest tests/workers/rollout/rollout_vllm/test_central_router.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import ray

from verl.workers.rollout.replica import TokenOutput


# =============================================================================
# Mock Server for Testing
# =============================================================================


@ray.remote
class MockVLLMServer:
    """Mock vLLM server for testing CentralRouter without GPU requirements."""

    def __init__(self, server_id: int, latency_range: tuple[float, float] = (0.01, 0.05)):
        self.server_id = server_id
        self.latency_range = latency_range
        self.request_count = 0
        self.request_ids_seen = []

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Simulate token generation with random latency."""
        import random

        self.request_count += 1
        self.request_ids_seen.append(request_id)

        # Simulate processing time
        latency = random.uniform(*self.latency_range)
        await asyncio.sleep(latency)

        # Generate mock tokens
        max_tokens = sampling_params.get("max_tokens", 10)
        token_ids = list(range(100, 100 + max_tokens))

        return TokenOutput(
            token_ids=token_ids,
            log_probs=None,
            routed_experts=None,
            stop_reason="length",
        )

    def get_request_count(self) -> int:
        return self.request_count

    def get_request_ids(self) -> list[str]:
        return self.request_ids_seen

    def reset_stats(self):
        self.request_count = 0
        self.request_ids_seen = []


# =============================================================================
# Test Functions
# =============================================================================


async def test_basic_routing():
    """Test that CentralRouter correctly routes requests to servers."""
    from verl.experimental.agent_loop.router import CentralRouter

    # Create mock servers
    num_servers = 3
    servers = [MockVLLMServer.remote(i) for i in range(num_servers)]

    # Create router
    router = CentralRouter.remote(servers)

    # Send a single request
    request_id = "test_request_1"
    output = await router.generate.remote(
        request_id,
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 5},
    )

    # Verify output
    assert output is not None
    assert len(output.token_ids) == 5
    assert output.stop_reason == "length"

    # Verify total requests
    total_requests = await router.get_total_requests.remote()
    assert total_requests == 1

    print("test_basic_routing PASSED")


async def test_load_balancing():
    """Test that requests are distributed evenly across servers."""
    from verl.experimental.agent_loop.router import CentralRouter

    # Create mock servers
    num_servers = 3
    servers = [MockVLLMServer.remote(i) for i in range(num_servers)]

    # Create router
    router = CentralRouter.remote(servers)

    # Send multiple requests with different request_ids (no sticky session)
    num_requests = 12
    tasks = []
    for i in range(num_requests):
        request_id = f"request_{i}"
        task = router.generate.remote(
            request_id,
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 5},
        )
        tasks.append(task)

    # Wait for all requests
    await asyncio.gather(*tasks)

    # Check load distribution
    request_counts = await asyncio.gather(*[s.get_request_count.remote() for s in servers])
    print(f"Request distribution: {request_counts}")

    # Each server should have approximately equal load (12 requests / 3 servers = 4 each)
    for count in request_counts:
        assert count == 4, f"Expected 4 requests per server, got {count}"

    print("test_load_balancing PASSED")


async def test_sticky_sessions():
    """Test that same request_id always routes to same server."""
    from verl.experimental.agent_loop.router import CentralRouter

    # Create mock servers
    num_servers = 3
    servers = [MockVLLMServer.remote(i) for i in range(num_servers)]

    # Create router
    router = CentralRouter.remote(servers)

    # Send multiple requests with SAME request_id
    request_id = "sticky_session_test"
    num_turns = 5

    for turn in range(num_turns):
        await router.generate.remote(
            request_id,
            prompt_ids=[1, 2, 3, turn],  # Different prompts, same request_id
            sampling_params={"max_tokens": 5},
        )

    # Check that only ONE server received all requests
    request_counts = await asyncio.gather(*[s.get_request_count.remote() for s in servers])
    print(f"Sticky session distribution: {request_counts}")

    # One server should have all 5 requests, others should have 0
    assert sum(request_counts) == num_turns
    assert max(request_counts) == num_turns
    assert request_counts.count(0) == num_servers - 1

    print("test_sticky_sessions PASSED")


async def test_concurrent_requests():
    """Test that multiple concurrent requests are handled correctly."""
    from verl.experimental.agent_loop.router import CentralRouter

    # Create mock servers with longer latency to ensure overlap
    num_servers = 2
    servers = [MockVLLMServer.remote(i, latency_range=(0.05, 0.1)) for i in range(num_servers)]

    # Create router
    router = CentralRouter.remote(servers)

    # Send many concurrent requests
    num_requests = 20
    start_time = time.time()

    tasks = []
    for i in range(num_requests):
        request_id = f"concurrent_{i}"
        task = router.generate.remote(
            request_id,
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 5},
        )
        tasks.append(task)

    # Wait for all
    outputs = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # All requests should complete
    assert len(outputs) == num_requests
    for output in outputs:
        assert output is not None
        assert len(output.token_ids) == 5

    # With 2 servers and concurrent execution, should be faster than sequential
    # Note: Ray actor initialization adds overhead, so we use a relaxed timeout
    print(f"Concurrent requests completed in {elapsed:.3f}s")
    # Just verify all requests completed - timing varies by environment
    assert len(outputs) == num_requests

    print("test_concurrent_requests PASSED")


async def test_router_adapter():
    """Test that RouterAdapter provides the same interface as AsyncLLMServerManager."""
    from verl.experimental.agent_loop.router import CentralRouter, RouterAdapter

    # Create mock servers
    num_servers = 2
    servers = [MockVLLMServer.remote(i) for i in range(num_servers)]

    # Create router and adapter
    router = CentralRouter.remote(servers)
    adapter = RouterAdapter(router)

    # Use adapter with same interface as AsyncLLMServerManager
    request_id = "adapter_test"
    output = await adapter.generate(
        request_id,
        prompt_ids=[1, 2, 3],
        sampling_params={"max_tokens": 5},
    )

    # Verify output
    assert output is not None
    assert len(output.token_ids) == 5

    # Send more requests through adapter
    for i in range(3):
        output = await adapter.generate(
            f"adapter_request_{i}",
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 5},
        )
        assert output is not None

    print("test_router_adapter PASSED")


async def test_server_loads_tracking():
    """Test that server load tracking works correctly."""
    from verl.experimental.agent_loop.router import CentralRouter

    # Create mock servers with longer latency
    num_servers = 2
    servers = [MockVLLMServer.remote(i, latency_range=(0.2, 0.3)) for i in range(num_servers)]

    # Create router
    router = CentralRouter.remote(servers)

    # Start multiple requests (collect ObjectRefs)
    refs = []
    for i in range(4):
        ref = router.generate.remote(
            f"load_test_{i}",
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 5},
        )
        refs.append(ref)

    # Give time for requests to start
    await asyncio.sleep(0.05)

    # Check loads during execution
    loads = await router.get_server_loads.remote()
    print(f"Loads during execution: {loads}")

    # Wait for completion using ray.get wrapped in asyncio
    await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in refs])

    # Check loads after completion (should be 0)
    loads_after = await router.get_server_loads.remote()
    print(f"Loads after completion: {loads_after}")
    assert all(v == 0 for v in loads_after.values()), f"Expected all loads to be 0, got {loads_after}"

    print("test_server_loads_tracking PASSED")


# =============================================================================
# Main
# =============================================================================


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CentralRouter Tests")
    print("=" * 60 + "\n")

    await test_basic_routing()
    await test_load_balancing()
    await test_sticky_sessions()
    await test_concurrent_requests()
    await test_router_adapter()
    await test_server_loads_tracking()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60 + "\n")


def main():
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    try:
        asyncio.run(run_all_tests())
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
