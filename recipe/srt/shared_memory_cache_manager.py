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
Shared Memory Cache Manager for SRT.

Adapts SpecRL's CacheManager infrastructure for SRT recipe, providing an
alternative to snapshot-based cache loading with zero-copy shared memory access.

Architecture comparison:
- Snapshot-based (SRT default): Trainer serializes trees -> push via Ray -> workers deserialize
- Shared memory (this module): Cache servers own shared memory -> trainer sends gRPC updates ->
  workers read directly from shared memory (zero-copy)

Key differences from snapshot mode:
- Cache updates happen AFTER rollout (not before)
- Workers access cache directly via SuffixCache (no load_snapshot calls)
- Cache servers deployed per GPU node via RolloutCacheServer
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import ray

from verl import DataProto

logger = logging.getLogger(__name__)


class SharedMemoryCacheManager:
    """
    Manages distributed suffix cache infrastructure using shared memory.

    Unlike SuffixTreeManager (snapshot-based), this manager:
    - Deploys RolloutCacheServer per GPU node (owns shared memory)
    - Sends async gRPC updates after rollouts (not before)
    - Workers read directly from shared memory (zero-copy)

    This implementation adapts SpecRL's CacheManager for use within the SRT recipe,
    enabling a toggle between snapshot and shared memory modes.

    Example:
        >>> config = {"srt_shared_memory": {"port": 6378, "memory_size_gb": 100}}
        >>> manager = SharedMemoryCacheManager(
        ...     config=config,
        ...     role_worker_mapping=role_worker_mapping,
        ...     resource_pool_manager=resource_pool_manager,
        ...     tokenizer=tokenizer,
        ... )
        >>> manager.initialize()  # Deploy cache servers
        >>> # After rollout
        >>> stats = manager.update_from_rollout(batch, responses_per_prompt=4)
    """

    def __init__(
        self,
        config: dict,
        role_worker_mapping: dict,
        resource_pool_manager,
        tokenizer,
        port: int = 6378,
    ):
        """
        Initialize SharedMemoryCacheManager.

        Args:
            config: SRT configuration dict containing shared_memory settings.
            role_worker_mapping: Mapping from roles to worker types.
            resource_pool_manager: Ray resource pool manager for placement group access.
            tokenizer: Tokenizer for encoding prompts/responses (may be needed for future features).
            port: gRPC server port for cache servers (default: 6378).
        """
        self._config = config
        self._role_worker_mapping = role_worker_mapping
        self._resource_pool_manager = resource_pool_manager
        self._tokenizer = tokenizer
        self._port = port

        # Internal state
        self._cache_servers: List[dict] = []
        self._cache_updater = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._update_futures: List = []
        self._max_futures = 5
        self._enabled = False

        # Statistics
        self._total_updates = 0
        self._total_tokens_sent = 0

    def initialize(self):
        """
        Deploy cache servers to GPU nodes and create updater.

        This method:
        1. Gets resource pool for actor/rollout workers
        2. Deploys CacheWorker (RolloutCacheServer) per unique node
        3. Collects server addresses for gRPC client
        4. Creates SuffixCacheUpdater for async updates

        Should be called after workers are initialized (in trainer.init_workers()).
        """
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from recipe.specRL.cache_manager import CacheWorker
        from verl.trainer.ppo.utils import Role

        # Get resource pool for actor/rollout workers
        actor_role = (
            Role.ActorRolloutRef
            if Role.ActorRolloutRef in self._role_worker_mapping
            else Role.ActorRollout
        )
        resource_pool = self._resource_pool_manager.get_resource_pool(actor_role)

        # Get unique node IDs from placement groups
        pgs = resource_pool.get_placement_groups()
        node_ids = set()
        for pg in pgs:
            specs = ray._private.state.state.placement_group_table(pg.id)
            # All bundles in a placement group should be on the same node
            node_id = specs["bundles_to_node_id"][0]
            node_ids.add(node_id)

        # Deploy CacheWorker (RolloutCacheServer) per node
        for node_id in node_ids:
            strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            server = CacheWorker.options(
                scheduling_strategy=strategy,
                name=f"srt_shm_cache_server_{node_id[:8]}",  # Truncate node_id for readability
            ).remote(port=self._port)

            # Get node's IPv6 address
            ip = ray.get(server.get_node_ip.remote())

            self._cache_servers.append({
                "server": server,
                "ip": ip,
                "port": self._port,
                "node_id": node_id,
            })

        # Create updater with server addresses
        server_addresses = self._get_server_addresses()

        try:
            from specrl.cache_updater import SuffixCacheUpdater
            self._cache_updater = SuffixCacheUpdater(server_addresses=server_addresses)
        except ImportError as e:
            logger.error(f"Failed to import SuffixCacheUpdater from specrl: {e}")
            logger.error("Shared memory mode requires the specrl package to be installed.")
            raise

        # Thread pool for async cache updates
        self._executor = ThreadPoolExecutor(max_workers=self._max_futures)
        self._enabled = True

        logger.info(
            f"SharedMemoryCacheManager: Deployed {len(self._cache_servers)} cache servers on port {self._port}"
        )
        logger.info(f"Server addresses: {server_addresses}")

    def _get_server_addresses(self) -> List[str]:
        """Get formatted gRPC addresses for all cache servers."""
        return [f"[{s['ip']}]:{s['port']}" for s in self._cache_servers]

    @property
    def enabled(self) -> bool:
        """Check if shared memory cache is initialized and active."""
        return self._enabled

    def update_from_rollout(
        self,
        batch: DataProto,
        responses_per_prompt: int = 1,
    ) -> Dict[str, Any]:
        """
        Send async gRPC update with rollout results.

        Unlike snapshot-based approach, this sends updates AFTER rollout
        to populate cache for NEXT batch (not current batch).

        Args:
            batch: DataProto containing prompts, responses, and attention masks.
            responses_per_prompt: Number of responses generated per prompt.

        Returns:
            Dict with update statistics for logging.
        """
        if not self._enabled:
            return {}

        # Extract response length from the batch
        response_length = batch.batch["responses"].shape[-1]

        # Split attention mask into prompt and response parts
        prompt_mask = batch.batch["attention_mask"][:, :-response_length]
        response_mask = batch.batch["attention_mask"][:, -response_length:]

        # Calculate actual lengths (excluding padding)
        prompt_lengths = prompt_mask.sum(-1).float().tolist()
        response_lengths = response_mask.sum(-1).float().tolist()

        # Get prompts and responses as lists
        prompts = batch.batch["prompts"].tolist()
        responses = batch.batch["responses"].tolist()

        # Limit concurrent futures to prevent memory overflow
        if len(self._update_futures) >= self._max_futures:
            oldest_future = self._update_futures.pop(0)
            try:
                oldest_future.result()
            except Exception as e:
                logger.warning(f"Cache update future failed: {e}")

        # Submit async cache update
        future = self._executor.submit(
            self._cache_updater.update_response_cache,
            prompts=prompts,
            responses=responses,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            responses_per_prompt=responses_per_prompt,
        )
        self._update_futures.append(future)

        # Update statistics
        self._total_updates += 1
        batch_size = len(prompts)
        total_response_tokens = sum(response_lengths)
        self._total_tokens_sent += total_response_tokens

        return {
            "shm_cache/update_submitted": 1,
            "shm_cache/batch_size": batch_size,
            "shm_cache/response_tokens": total_response_tokens,
        }

    def update_from_secondary(
        self,
        secondary_outputs: list,
        responses_per_prompt: int = 1,
    ) -> Dict[str, Any]:
        """
        Update cache with secondary (runahead) outputs.

        Similar to update_from_rollout but handles SecondaryOutput format
        from runahead generation.

        Args:
            secondary_outputs: List of SecondaryOutput from runahead.
            responses_per_prompt: Number of responses generated per prompt.

        Returns:
            Dict with update statistics.
        """
        if not self._enabled:
            return {}

        # Filter to outputs with actual tokens
        usable_outputs = [
            out for out in secondary_outputs
            if out.status in ("completed", "aborted")
            and out.output is not None
            and len(out.output.token_ids) > 0
            and len(out.prompt_ids) > 0
        ]

        if not usable_outputs:
            return {"shm_cache/secondary_skipped": 1}

        # Build lists for cache update
        prompts = []
        responses = []
        prompt_lengths = []
        response_lengths = []

        for out in usable_outputs:
            prompt_tokens = list(out.prompt_ids)
            response_tokens = list(out.output.token_ids)

            prompts.append(prompt_tokens)
            responses.append(response_tokens)
            prompt_lengths.append(float(len(prompt_tokens)))
            response_lengths.append(float(len(response_tokens)))

        # Limit concurrent futures
        if len(self._update_futures) >= self._max_futures:
            oldest_future = self._update_futures.pop(0)
            try:
                oldest_future.result()
            except Exception as e:
                logger.warning(f"Secondary cache update future failed: {e}")

        # Submit async cache update
        future = self._executor.submit(
            self._cache_updater.update_response_cache,
            prompts=prompts,
            responses=responses,
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            responses_per_prompt=responses_per_prompt,
        )
        self._update_futures.append(future)

        total_tokens = sum(response_lengths)
        self._total_tokens_sent += total_tokens

        return {
            "shm_cache/secondary_outputs_processed": len(usable_outputs),
            "shm_cache/secondary_tokens": total_tokens,
        }

    def get_server_addresses(self) -> List[str]:
        """
        Get gRPC addresses for workers to use.

        This can be passed to worker config if workers need to know
        server addresses (though in shared memory mode, workers connect
        via SuffixCache which auto-discovers the shared memory segment).

        Returns:
            List of formatted gRPC addresses.
        """
        return self._get_server_addresses()

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for logging."""
        if not self._enabled:
            return {}

        return {
            "shm_cache/num_servers": len(self._cache_servers),
            "shm_cache/total_updates": self._total_updates,
            "shm_cache/total_tokens_sent": self._total_tokens_sent,
            "shm_cache/pending_futures": len(self._update_futures),
        }

    def wait_for_pending_updates(self, timeout: float = 30.0):
        """
        Wait for all pending cache updates to complete.

        Args:
            timeout: Maximum time to wait for each future in seconds.
        """
        for future in self._update_futures:
            if not future.done():
                try:
                    future.result(timeout=timeout)
                except Exception as e:
                    logger.warning(f"Cache update future failed during wait: {e}")

        self._update_futures.clear()

    def shutdown(self):
        """Cleanup servers and executor."""
        if not self._enabled:
            return

        # Wait for pending updates
        self.wait_for_pending_updates(timeout=5.0)

        # Shutdown executor
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Shutdown cache servers
        if self._cache_servers:
            shutdown_futures = []
            for server_info in self._cache_servers:
                try:
                    future = server_info["server"].shutdown.remote()
                    shutdown_futures.append(future)
                except Exception as e:
                    logger.warning(f"Failed to initiate cache server shutdown: {e}")

            if shutdown_futures:
                try:
                    ray.get(shutdown_futures, timeout=10)
                except Exception as e:
                    logger.warning(f"Error waiting for cache server shutdowns: {e}")

            self._cache_servers.clear()

        self._enabled = False
        logger.info("SharedMemoryCacheManager shutdown complete")

    def __del__(self):
        """Ensure cleanup on destruction."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during destruction
