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
import heapq
import logging
import os
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
