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
Worker extension for suffix tree snapshot loading.

This extension is injected into vLLM workers at runtime via worker_extension_cls,
enabling load_suffix_snapshot() to be called via collective_rpc from the server.

Usage in vLLM config:
    worker_extension_cls="recipe.srt.vllm_plugin.worker_extension.SuffixTreeWorkerExtension"
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SuffixTreeWorkerExtension:
    """
    Worker extension that adds suffix tree snapshot loading capability.

    This class is dynamically inherited by vLLM Worker class at runtime.
    Methods defined here become accessible via collective_rpc().

    The injection happens in WorkerWrapperBase.init_worker() when
    worker_extension_cls is specified in ParallelConfig.
    """

    def load_suffix_snapshot(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ) -> dict[str, Any]:
        """
        Load suffix tree snapshots into the drafter's suffix cache.

        This method is called via collective_rpc from SRTvLLMHttpServer.
        It accesses self.model_runner.drafter which is the suffix proposer.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples containing
                serialized suffix trees from the trainer's SuffixTreeManager.
            hash_mapping: Dict mapping prompt_hash -> tree_idx for hash-based
                tree matching during speculation.

        Returns:
            Dict with status and statistics about the loading operation.
        """
        if not snapshots:
            return {"status": "skipped", "reason": "empty_snapshots"}

        # Access drafter through model_runner (self is the Worker instance)
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return {"status": "error", "reason": "no_model_runner"}

        drafter = getattr(model_runner, "drafter", None)
        if drafter is None:
            # Not all workers have a drafter (e.g., non-last PP ranks)
            return {"status": "skipped", "reason": "no_drafter"}

        if not hasattr(drafter, "load_snapshot"):
            return {"status": "error", "reason": "drafter_no_load_snapshot"}

        try:
            drafter.load_snapshot(snapshots, hash_mapping)
            total_bytes = sum(len(s[1]) for s in snapshots)
            logger.info(
                f"Loaded {len(snapshots)} suffix trees ({total_bytes} bytes) "
                f"with {len(hash_mapping)} hash mappings"
            )
            return {
                "status": "success",
                "num_trees": len(snapshots),
                "total_bytes": total_bytes,
                "num_hash_mappings": len(hash_mapping),
            }
        except Exception as e:
            logger.error(f"Failed to load suffix snapshot: {e}")
            return {"status": "error", "reason": str(e)}

    def get_suffix_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics from the suffix cache (for debugging/monitoring).

        Returns:
            Dict with cache statistics or error status.
        """
        model_runner = getattr(self, "model_runner", None)
        if model_runner is None:
            return {"status": "error", "reason": "no_model_runner"}

        drafter = getattr(model_runner, "drafter", None)
        if drafter is None:
            return {"status": "error", "reason": "no_drafter"}

        if hasattr(drafter, "get_stats"):
            return drafter.get_stats()
        elif hasattr(drafter, "suffix_cache"):
            cache = drafter.suffix_cache
            stats = {"status": "success"}
            if hasattr(cache, "active_requests"):
                stats["num_active_requests"] = len(cache.active_requests)
            if hasattr(cache, "num_trees"):
                stats["num_trees"] = cache.num_trees
            return stats

        return {"status": "no_stats_available"}
