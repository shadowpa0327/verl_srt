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
SRT vLLM server and replica classes.

These classes extend the base vLLM server to apply SRT config patches
before engine creation, enabling suffix decoding support.

Supports two cache modes:
- "snapshot" (default): Uses worker_extension_cls and speculative_config
  with ParallelSuffixDecodingProposer, snapshots pushed via collective_rpc
- "shared_memory": Uses SpecRL's GPUModelRunnerPatch for zero-copy shared
  memory access, cache populated via gRPC from trainer
"""

import logging
import os
from typing import Any

import ray
from ray.actor import ActorHandle

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServerBase,
    vLLMReplica,
)

logger = logging.getLogger(__name__)


# Track if SRT patches have been applied in this process
_srt_patches_applied = False
_shm_patches_applied = False


def _is_shared_memory_mode() -> bool:
    """Check if SRT shared memory mode is enabled.

    Checks for environment variable SRT_CACHE_MODE=shared_memory which is
    set by SRTRayPPOTrainer._inject_srt_engine_kwargs() when cache_mode is
    configured as "shared_memory".

    Returns:
        True if shared memory mode should be used.
    """
    return os.environ.get("SRT_CACHE_MODE", "").lower() == "shared_memory"


def _apply_shared_memory_patches():
    """Apply SpecRL's shared memory patches for zero-copy cache access.

    This applies WorkerBasePatch which in turn applies GPUModelRunnerPatch
    and RejectionSamplerPatch when workers are initialized.
    """
    global _shm_patches_applied
    if _shm_patches_applied:
        return

    try:
        from recipe.srt.srt_plugin.patches.shm_patches import apply_shared_memory_patches
        apply_shared_memory_patches()
        logger.info("Applied shared memory patches in vLLMHttpServer")
        _shm_patches_applied = True
    except ImportError as e:
        logger.warning(f"Failed to import shared memory patches: {e}")
        logger.warning(
            "Shared memory mode requires the specrl package. "
            "Falling back to snapshot mode."
        )


def _apply_srt_patches():
    """Apply SRT patches for suffix decoding support.

    Applies both config_patches (to add 'suffix' to method Literal) and
    arg_utils_patches (to extract srt_* params and register SRTSuffixConfig).

    These patches are needed for snapshot mode but not for shared_memory mode
    (which uses SpecRL's GPUModelRunnerPatch instead).
    """
    global _srt_patches_applied
    if _srt_patches_applied:
        return

    try:
        # Config patches: Add 'suffix' to SpeculativeConfig.method Literal
        from recipe.srt.srt_plugin.patches import config_patches
        config_patches.apply_patches()

        # Arg utils patches: Extract srt_* params and register SRTSuffixConfig
        from recipe.srt.srt_plugin.patches import arg_utils_patches
        arg_utils_patches.apply_patches()

        logger.info("Applied SRT config and arg_utils patches in vLLMHttpServer")
        _srt_patches_applied = True
    except ImportError as e:
        logger.warning(f"Failed to import SRT patches: {e}")


@ray.remote(num_cpus=1)
class SRTvLLMHttpServer(vLLMHttpServerBase):
    """vLLM HTTP server with SRT suffix decoding support.

    This server applies SRT config patches before engine creation to enable
    'suffix' and 'suffix_remote' speculative decoding methods.

    Supports two modes based on SRT_CACHE_MODE environment variable:
    - snapshot (default): Uses ParallelSuffixDecodingProposer with worker_extension_cls
    - shared_memory: Uses SpecRL's GPUModelRunnerPatch for zero-copy shared memory

    Note: We extend vLLMHttpServerBase (not vLLMHttpServer) because Ray
    doesn't allow subclassing actor classes.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
    ):
        # Determine cache mode and apply appropriate patches
        self._use_shared_memory = _is_shared_memory_mode()

        if self._use_shared_memory:
            # Shared memory mode: Apply SpecRL's GPUModelRunnerPatch
            # This patches WorkerBase.__init__ to apply GPUModelRunner patches
            logger.info("SRTvLLMHttpServer: Using shared memory mode")
            _apply_shared_memory_patches()
        else:
            # Snapshot mode: Apply SRT config patches
            logger.info("SRTvLLMHttpServer: Using snapshot mode")
            _apply_srt_patches()

        super().__init__(
            config, model_config, rollout_mode, workers,
            replica_rank, node_rank, gpus_per_node, nnodes
        )


    async def load_suffix_snapshot(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ) -> dict[str, Any]:
        """Load suffix tree snapshots via collective_rpc to workers.

        This method broadcasts snapshots to all workers in the vLLM engine
        subprocess via collective_rpc. The workers must have the
        SuffixTreeWorkerExtension injected via worker_extension_cls.

        Note: In shared_memory mode, this is a no-op because workers access
        the cache directly via shared memory (populated by trainer via gRPC).

        Note: This is an async method because collective_rpc uses ZMQ sockets
        bound to the engine's event loop. Ray handles async actor methods natively.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples
            hash_mapping: Dict mapping prompt_hash -> tree_idx

        Returns:
            Dict with status and worker results
        """
        # Shared memory mode: no snapshot loading needed
        if self._use_shared_memory:
            return {
                "status": "skipped",
                "reason": "shared_memory_mode",
                "message": "Cache accessed via shared memory, no snapshot loading needed",
            }

        if not snapshots:
            return {"status": "skipped", "reason": "empty_snapshots"}

        if getattr(self, "engine", None) is None:
            logger.error("Cannot load suffix snapshots - engine not ready")
            return {"status": "error", "reason": "engine_not_ready"}

        try:
            # Direct await - Ray handles async actor methods natively
            results = await self.engine.collective_rpc(
                method="load_suffix_snapshot",
                args=(snapshots, hash_mapping),
            )

            total_bytes = sum(len(s[1]) for s in snapshots)
            logger.info(
                f"collective_rpc load_suffix_snapshot: loaded {len(snapshots)} trees "
                f"({total_bytes} bytes) with {len(hash_mapping)} hash mappings, "
                f"worker results: {results}"
            )
            return {"status": "success", "worker_results": results}
        except Exception as e:
            logger.error(f"Failed to load suffix snapshot via collective_rpc: {e}")
            return {"status": "error", "reason": str(e)}


class SRTvLLMReplica(vLLMReplica):
    """vLLM Replica with SRT suffix decoding support.

    Uses SRTvLLMHttpServer which applies config patches before engine creation.

    Supports two cache modes (determined by SRT_CACHE_MODE env var):
    - snapshot: Uses worker_extension_cls with collective_rpc snapshot loading
    - shared_memory: Cache accessed via shared memory (no snapshot loading)
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        # Call grandparent's __init__ to avoid setting server_class to vLLMHttpServer
        RolloutReplica.__init__(
            self, replica_rank, config, model_config, gpus_per_node, is_reward_model
        )
        # Set our SRT server class
        self.server_class = SRTvLLMHttpServer
        # Track cache mode
        self._use_shared_memory = _is_shared_memory_mode()

    def load_suffix_snapshot(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ) -> list[dict[str, Any]]:
        """Load suffix tree snapshots across all servers in this replica.

        Note: In shared_memory mode, this is a no-op because workers access
        the cache directly via shared memory.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples
            hash_mapping: Dict mapping prompt_hash -> tree_idx

        Returns:
            List of result dicts from each server
        """
        # Shared memory mode: no snapshot loading needed
        if self._use_shared_memory:
            return [{
                "status": "skipped",
                "reason": "shared_memory_mode",
                "message": "Cache accessed via shared memory",
            }]

        if not snapshots:
            return [{"status": "skipped", "reason": "empty_snapshots"}]

        refs = [
            server.load_suffix_snapshot.remote(snapshots, hash_mapping)
            for server in self.servers
        ]
        results = ray.get(refs)
        logger.info(f"Loaded suffix snapshots across {len(self.servers)} servers: {results}")
        return results


def register_srt_vllm_replica():
    """Override the 'vllm' replica to use SRT version.

    This overrides the default 'vllm' entry in RolloutReplicaRegistry
    so that AgentLoopManager uses SRTvLLMReplica instead of vLLMReplica.

    We override 'vllm' instead of registering a new 'srt_vllm' type because
    rollout.name is used by multiple registries:
    - RolloutReplicaRegistry (for replica/server classes)
    - _ROLLOUT_REGISTRY in base.py (for rollout worker classes)

    By overriding 'vllm', we keep rollout.name="vllm" which works with both.
    """
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    def _load_srt_vllm():
        return SRTvLLMReplica

    RolloutReplicaRegistry.register("vllm", _load_srt_vllm)
    logger.info("Overriding 'vllm' rollout replica with SRT version")
