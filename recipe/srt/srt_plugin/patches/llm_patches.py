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
LLM patches for suffix tree snapshot loading.

Adds LLM.load_snapshot() method for loading pre-built suffix trees.
"""

import logging

from ..patching import ArcticPatch
from vllm.entrypoints.llm import LLM

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


class LLMSnapshotPatch(ArcticPatch[LLM]):
    """Adds load_snapshot() method to LLM for suffix tree loading."""

    def load_snapshot(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ) -> None:
        """Load suffix tree snapshots for speculative decoding.

        Loads pre-built suffix trees with hash-based prompt matching.
        Requests with identical prompts will automatically reuse the
        same tree for improved speculation.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples from
                ParallelSuffixDecodingCache.create_snapshot()
            hash_mapping: Dict mapping prompt_hash -> tree_idx from
                ParallelSuffixDecodingCache.create_snapshot(include_hash_mapping=True)

        Example:
            snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)
            llm.load_snapshot(snapshots, hash_mapping)
        """
        if not snapshots:
            logger.debug("load_snapshot called with empty snapshots, skipping")
            return

        # Access model runner through the engine hierarchy
        # V1 engine: llm_engine -> model_executor -> driver_worker -> worker -> model_runner
        try:
            model_runner = (
                self.llm_engine
                .model_executor
                .driver_worker
                .worker
                .model_runner
            )
        except AttributeError:
            # Try alternative path for different vLLM versions
            try:
                model_runner = (
                    self.llm_engine
                    .model_executor
                    .driver_worker
                    .model_runner
                )
            except AttributeError as e:
                raise RuntimeError(
                    "load_snapshot() could not find model_runner. "
                    "This may indicate an incompatible vLLM version."
                ) from e

        if hasattr(model_runner, 'drafter') and hasattr(model_runner.drafter, 'load_snapshot'):
            model_runner.drafter.load_snapshot(snapshots, hash_mapping)
            logger.debug(
                "Loaded %d suffix tree snapshots with %d hash mappings",
                len(snapshots), len(hash_mapping)
            )
        else:
            raise RuntimeError(
                "load_snapshot() requires suffix decoding to be enabled. "
                "Set speculative_method='suffix' or 'suffix_remote' and ensure "
                "the drafter supports snapshot loading."
            )


def apply_patches():
    """Apply LLM patches."""
    global _patches_applied

    if _patches_applied:
        logger.debug("LLM patches already applied, skipping")
        return

    # Use ArcticPatch's apply_patch method
    LLMSnapshotPatch.apply_patch()

    _patches_applied = True
    logger.info("Applied LLM load_snapshot patch")
