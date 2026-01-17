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
Suffix Tree Manager for VERL Training.

This module provides the SuffixTreeManager class that manages suffix tree
accumulation during RL training. It wraps srt_plugin's ParallelSuffixDecodingCache
and provides methods to:
- Update trees from rollout batches
- Create snapshots for pushing to vLLM workers
- Save/load state for checkpointing
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SuffixTreeManagerConfig:
    """Configuration for SuffixTreeManager.

    Attributes:
        enable: Whether suffix tree management is enabled.
        max_tree_depth: Maximum depth of suffix trees.
        hash_token_count: Number of trailing tokens to hash for tree sharing.
            Set to 0 to disable hash-based sharing.
        num_threads: Number of threads for parallel operations (-1 = auto).
        parallel_threshold: Minimum batch size to trigger parallelization.
    """

    enable: bool = False
    max_tree_depth: int = 64
    hash_token_count: int = 128
    num_threads: int = -1
    parallel_threshold: int = 4


class SuffixTreeManager:
    """
    Manages suffix tree accumulation and snapshot distribution for VERL.

    This class wraps ParallelSuffixDecodingCache and provides:
    1. Accumulation of Q/A patterns from training rollouts
    2. Snapshot creation for pushing to vLLM workers
    3. State persistence across checkpoints

    The manager uses hash-based tree sharing, so identical prompts
    (same hash) share a single tree, which is important for RL training
    where the same question may be sampled multiple times with different
    rollouts.

    Example:
        >>> manager = SuffixTreeManager(config, tokenizer)
        >>> # After rollout
        >>> stats = manager.update_from_rollout(batch)
        >>> # Before next rollout, push to workers
        >>> snapshots, hash_mapping = manager.get_snapshot()
        >>> worker.load_suffix_snapshot(snapshots, hash_mapping)
    """

    def __init__(
        self,
        config: SuffixTreeManagerConfig,
        tokenizer: Any,
    ):
        """
        Initialize the SuffixTreeManager.

        Args:
            config: Configuration for suffix tree management.
            tokenizer: Tokenizer for encoding prompts/responses.
                Must support encode() with add_special_tokens parameter.
        """
        self.config = config
        self.tokenizer = tokenizer
        self._cache = None
        self._batch_counter = 0
        self._total_responses = 0

        # Accumulated token counters for primary (main rollout) and secondary (runahead)
        self._total_primary_tokens_added = 0
        self._total_secondary_tokens_added = 0
        self._secondary_batch_counter = 0
        self._secondary_outputs_processed = 0

        if config.enable:
            self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize the ParallelSuffixDecodingCache."""
        try:
            from recipe.srt.srt_plugin.suffix_cache import ParallelSuffixDecodingCache

            self._cache = ParallelSuffixDecodingCache(
                max_tree_depth=self.config.max_tree_depth,
                hash_token_count=self.config.hash_token_count,
                num_threads=self.config.num_threads,
                parallel_threshold=self.config.parallel_threshold,
            )
            logger.info(
                f"Initialized SuffixTreeManager with max_tree_depth={self.config.max_tree_depth}, "
                f"hash_token_count={self.config.hash_token_count}"
            )
        except ImportError as e:
            logger.warning(
                f"Failed to import srt_plugin.suffix_cache, suffix tree management disabled: {e}"
            )
            self._cache = None

    @property
    def enabled(self) -> bool:
        """Check if suffix tree management is enabled and cache is available."""
        return self.config.enable and self._cache is not None

    def update_from_rollout(
        self,
        batch: Any,
    ) -> Dict[str, Any]:
        """
        Update suffix trees with Q/A patterns from a rollout batch.

        This method extracts prompts and responses from the batch, tokenizes
        them with BOS tokens (matching vLLM's tokenization), and adds them
        to the suffix tree cache.

        Args:
            batch: DataProto batch containing:
                - non_tensor_batch["raw_prompt_ids"]: Raw prompt token IDs (matches vLLM)
                - batch["responses"]: Response token IDs (batch_size, response_length)
                - batch["prompts"]: Fallback if raw_prompt_ids not available

        Returns:
            Dict with update statistics:
                - suffix_tree/prompts_processed: Number of Q/A pairs processed
                - suffix_tree/tokens_added: Total tokens added to trees
                - suffix_tree/num_trees: Current number of trees in forest
        """
        if not self.enabled:
            return {}

        self._batch_counter += 1
        stats: Dict[str, Any] = {
            "suffix_tree/prompts_processed": 0,
            "suffix_tree/tokens_added": 0,
        }

        # Extract response tokens from batch
        response_ids = batch.batch.get("responses")

        if response_ids is None:
            logger.warning("Batch missing responses, skipping suffix tree update")
            return stats

        # Use vllm_prompt_tokens from non_tensor_batch - these are the EXACT tokens
        # vLLM used for generation, ensuring hash matching for suffix tree lookup
        vllm_prompt_tokens = batch.non_tensor_batch.get("vllm_prompt_tokens")

        if vllm_prompt_tokens is None:
            # Fallback to prompts tensor if vllm_prompt_tokens not available
            # This can happen with non-vLLM rollout backends
            prompt_ids = batch.batch.get("prompts")
            if prompt_ids is None:
                logger.warning("Batch missing vllm_prompt_tokens and prompts, skipping suffix tree update")
                return stats
            if hasattr(prompt_ids, "cpu"):
                prompt_ids = prompt_ids.cpu().numpy()
            use_vllm_prompt_tokens = False
            logger.debug("Using fallback prompts tensor - hash matching may fail with vLLM")
        else:
            use_vllm_prompt_tokens = True

        # Convert response_ids to numpy if needed
        if hasattr(response_ids, "cpu"):
            response_ids = response_ids.cpu().numpy()

        batch_size = len(response_ids)

        # Get pre-computed prompt hashes from rollout worker (if available)
        # These are computed from the exact tokens vLLM used, ensuring hash consistency
        prompt_hashes = batch.non_tensor_batch.get("prompt_hashes")

        for i in range(batch_size):
            # Get prompt tokens - use vllm_prompt_tokens if available (exact match with vLLM)
            if use_vllm_prompt_tokens:
                # vllm_prompt_tokens contains the exact tokens vLLM used
                prompt_tokens = list(vllm_prompt_tokens[i])
            else:
                # Fallback: remove padding from prompts tensor
                prompt_tokens = self._remove_left_padding(prompt_ids[i])

            if len(prompt_tokens) == 0:
                continue

            # Create unique request ID
            req_id = f"train_{self._batch_counter}_{i}"

            # Convert to numpy array for cache
            prompt_array = np.array(prompt_tokens, dtype=np.int32)

            # Use pre-computed hash if available (ensures consistency with vLLM)
            pre_computed_hash = prompt_hashes[i] if prompt_hashes is not None else None

            # Start request (creates or reuses tree based on prompt hash)
            self._cache.start_request(req_id, prompt_array, pre_computed_hash=pre_computed_hash)

            # Remove padding from response (right-padded typically)
            response_tokens = self._remove_right_padding(response_ids[i])

            # Add response tokens to tree
            if len(response_tokens) > 0:
                response_array = np.array(response_tokens, dtype=np.int32)
                self._cache.add_tokens(req_id, response_array)
                stats["suffix_tree/tokens_added"] += len(response_tokens)

            # NOTE: We do NOT call stop_request() here!
            # When processing requests sequentially, stop_request() would delete
            # the tree since no other requests reference it. We want trees to
            # persist for speculation, so we keep requests "active" in the cache.
            # The hash-based sharing will ensure future requests with the same
            # prompt reuse the existing tree.

            stats["suffix_tree/prompts_processed"] += 1
            self._total_responses += 1

        # Add cache statistics
        try:
            cache_stats = self._cache.get_stats()
            stats["suffix_tree/num_trees"] = cache_stats.get("num_trees_in_forest", 0)
            stats["suffix_tree/active_requests"] = cache_stats.get("num_active_requests", 0)
        except Exception:
            # get_stats may not be available in all versions
            pass

        # Add memory estimation metrics
        try:
            memory_stats = self._cache.get_memory_stats()
            total_bytes = memory_stats.get("total_bytes", 0)
            stats["suffix_tree/memory_bytes"] = total_bytes
            stats["suffix_tree/memory_mb"] = total_bytes / (1024 * 1024)
            stats["suffix_tree/avg_bytes_per_tree"] = memory_stats.get("avg_bytes_per_tree", 0)
        except Exception:
            # get_memory_stats may not be available in all versions
            pass

        # Accumulate primary tokens added
        self._total_primary_tokens_added += stats["suffix_tree/tokens_added"]

        logger.debug(
            f"Suffix tree update: processed {stats['suffix_tree/prompts_processed']} prompts, "
            f"added {stats['suffix_tree/tokens_added']} tokens"
        )

        return stats

    def _remove_left_padding(self, token_ids: np.ndarray) -> List[int]:
        """Remove left padding tokens from token array."""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # No padding token defined, return as-is
            return token_ids.tolist()

        # Find first non-pad token
        mask = token_ids != pad_id
        if not mask.any():
            return []

        first_non_pad = mask.argmax()
        return token_ids[first_non_pad:].tolist()

    def _remove_right_padding(self, token_ids: np.ndarray) -> List[int]:
        """Remove right padding tokens from token array."""
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # No padding token defined, return as-is
            return token_ids.tolist()

        # Find last non-pad token
        mask = token_ids != pad_id
        if not mask.any():
            return []

        # Reverse the mask to find from the right
        last_non_pad = len(token_ids) - 1 - mask[::-1].argmax()
        return token_ids[: last_non_pad + 1].tolist()

    def add_sequence(
        self,
        prompt_tokens: List[int],
        response_tokens: List[int],
        req_id: Optional[str] = None,
    ) -> None:
        """Add a single prompt/response sequence to the suffix tree.

        This is a lower-level API for adding individual sequences,
        useful for processing secondary outputs from runahead.

        Args:
            prompt_tokens: Tokenized prompt (must match vLLM tokenization).
            response_tokens: Response tokens to add to the tree.
            req_id: Optional request ID (auto-generated if not provided).
        """
        if not self.enabled:
            return

        if len(prompt_tokens) == 0 or len(response_tokens) == 0:
            return

        # Generate unique request ID if not provided
        if req_id is None:
            self._secondary_counter = getattr(self, "_secondary_counter", 0) + 1
            req_id = f"secondary_{self._secondary_counter}"

        # Convert to numpy arrays for cache
        prompt_array = np.array(prompt_tokens, dtype=np.int32)
        response_array = np.array(response_tokens, dtype=np.int32)

        # Start request (creates or reuses tree based on prompt hash)
        self._cache.start_request(req_id, prompt_array)

        # Add response tokens to tree
        self._cache.add_tokens(req_id, response_array)

        # NOTE: We don't call stop_request() - keep trees active for speculation

    def record_secondary_update(
        self,
        tokens_added: int,
        outputs_processed: int,
    ) -> None:
        """Record statistics from a secondary (runahead) batch update.

        Called by the trainer after processing secondary outputs to track
        accumulated secondary statistics separately from primary rollout stats.

        Args:
            tokens_added: Number of tokens added from this secondary batch.
            outputs_processed: Number of secondary outputs processed.
        """
        self._total_secondary_tokens_added += tokens_added
        self._secondary_outputs_processed += outputs_processed
        self._secondary_batch_counter += 1

    def get_snapshot(self) -> Tuple[List[Tuple[int, bytes]], Dict[str, int]]:
        """
        Create snapshot of current suffix trees for pushing to workers.

        Returns:
            Tuple of:
                - snapshots: List of (tree_idx, snapshot_bytes) tuples
                - hash_mapping: Dict mapping prompt_hash -> tree_idx
        """
        if not self.enabled:
            return [], {}

        try:
            return self._cache.create_snapshot(include_hash_mapping=True)
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return [], {}

    def extract_batch_hashes(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> List[str]:
        """
        Extract prompt hashes from a batch of input_ids.

        This computes hashes for each prompt in the batch, handling padding
        via the attention mask. Used to determine which trees to include
        in selective snapshots.

        Args:
            input_ids: Batch of token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
                           1 for valid tokens, 0 for padding

        Returns:
            List of unique prompt hashes from the batch (deduplicated)
        """
        if not self.enabled:
            return []

        batch_size = len(input_ids)
        seen_hashes = set()
        unique_hashes = []

        for i in range(batch_size):
            # Extract actual tokens using attention mask
            mask = attention_mask[i]
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0:
                continue

            tokens = input_ids[i][valid_indices]

            # Compute hash using the cache's hash function
            prompt_hash = self._cache._hash_prompt(tokens)

            if prompt_hash not in seen_hashes:
                seen_hashes.add(prompt_hash)
                unique_hashes.append(prompt_hash)

        return unique_hashes

    def get_selective_snapshot(
        self,
        hashes: List[str],
    ) -> Tuple[List[Tuple[int, bytes]], Dict[str, int]]:
        """
        Get snapshots for specific prompt hashes only.

        This is more efficient than get_snapshot() when only a subset of trees
        is needed for a particular batch. Workers only receive trees for the
        prompts they will process.

        Args:
            hashes: Prompt hashes needed for this batch (from DataProto.non_tensor_batch["prompt_hashes"])

        Returns:
            Tuple of:
                - snapshots: List of (tree_idx, snapshot_bytes) for requested trees only
                - hash_mapping: Dict mapping prompt_hash -> tree_idx for requested hashes
        """
        if not self.enabled:
            return [], {}

        if not hashes:
            logger.warning("get_selective_snapshot called with empty hashes list")
            return [], {}

        try:
            # Map hashes to tree indices
            cache_hash_to_tree = self._cache._hash_to_tree_idx
            tree_indices = []
            hash_mapping: Dict[str, int] = {}

            for prompt_hash in hashes:
                if prompt_hash in cache_hash_to_tree:
                    tree_idx = cache_hash_to_tree[prompt_hash]
                    if tree_idx not in tree_indices:  # Dedupe
                        tree_indices.append(tree_idx)
                    hash_mapping[prompt_hash] = tree_idx

            if not tree_indices:
                logger.warning(
                    f"None of the {len(hashes)} requested hashes found in cache "
                    f"(cache has {len(cache_hash_to_tree)} hashes)"
                )
                return [], {}

            # Use C++ selective snapshot (efficient)
            snapshots = self._cache.create_selective_snapshot(tree_indices)

            logger.debug(
                f"Selective snapshot: {len(snapshots)} trees for {len(hash_mapping)} hashes "
                f"(out of {self._cache._forest.num_trees()} total trees)"
            )

            return snapshots, hash_mapping

        except Exception as e:
            logger.error(f"Failed to create selective snapshot: {e}")
            return [], {}

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for logging.

        Returns:
            Dict with suffix tree metrics including memory usage.
        """
        if not self.enabled:
            return {}

        metrics = {
            "suffix_tree/total_responses": self._total_responses,
            "suffix_tree/batch_counter": self._batch_counter,
            # Accumulated token counters (primary vs secondary)
            "suffix_tree/total_primary_tokens_added": self._total_primary_tokens_added,
            "suffix_tree/total_secondary_tokens_added": self._total_secondary_tokens_added,
            "suffix_tree/total_tokens_added": self._total_primary_tokens_added + self._total_secondary_tokens_added,
            # Secondary batch statistics
            "suffix_tree/secondary_batch_counter": self._secondary_batch_counter,
            "suffix_tree/secondary_outputs_processed": self._secondary_outputs_processed,
        }

        try:
            cache_stats = self._cache.get_stats()
            metrics["suffix_tree/num_trees"] = cache_stats.get("num_trees_in_forest", 0)
            metrics["suffix_tree/active_requests"] = cache_stats.get("num_active_requests", 0)
        except Exception:
            pass

        # Add memory estimation metrics
        try:
            memory_stats = self._cache.get_memory_stats()
            total_bytes = memory_stats.get("total_bytes", 0)
            metrics["suffix_tree/memory_bytes"] = total_bytes
            metrics["suffix_tree/memory_mb"] = total_bytes / (1024 * 1024)
            metrics["suffix_tree/avg_bytes_per_tree"] = memory_stats.get("avg_bytes_per_tree", 0)
        except Exception:
            pass

        return metrics

    def save(self, path: str) -> None:
        """
        Save suffix tree state to disk for checkpointing.

        Args:
            path: Directory path to save state.
        """
        if not self.enabled:
            return

        os.makedirs(path, exist_ok=True)

        # Get snapshot
        snapshots, hash_mapping = self.get_snapshot()

        # Save state
        state = {
            "config": {
                "enable": self.config.enable,
                "max_tree_depth": self.config.max_tree_depth,
                "hash_token_count": self.config.hash_token_count,
                "num_threads": self.config.num_threads,
                "parallel_threshold": self.config.parallel_threshold,
            },
            "snapshots": snapshots,
            "hash_mapping": hash_mapping,
            "batch_counter": self._batch_counter,
            "total_responses": self._total_responses,
        }

        state_path = os.path.join(path, "suffix_tree_state.pkl")
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

        logger.info(
            f"Saved suffix tree state to {state_path}: "
            f"{len(snapshots)} trees, {len(hash_mapping)} hash mappings"
        )

    def load(self, path: str) -> bool:
        """
        Load suffix tree state from disk.

        Args:
            path: Directory path containing saved state.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        state_path = os.path.join(path, "suffix_tree_state.pkl")
        if not os.path.exists(state_path):
            logger.debug(f"No suffix tree state found at {state_path}")
            return False

        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            # Restore counters
            self._batch_counter = state.get("batch_counter", 0)
            self._total_responses = state.get("total_responses", 0)

            # Restore cache from snapshot
            snapshots = state.get("snapshots", [])
            hash_mapping = state.get("hash_mapping", {})

            if snapshots:
                if self._cache is None:
                    self._initialize_cache()

                if self._cache is not None:
                    self._cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)
                    logger.info(
                        f"Loaded suffix tree state from {state_path}: "
                        f"{len(snapshots)} trees, {len(hash_mapping)} hash mappings"
                    )
                    return True

        except Exception as e:
            logger.error(f"Failed to load suffix tree state: {e}")

        return False

    def reset(self) -> None:
        """Reset the suffix tree cache to empty state."""
        if self.enabled:
            self._initialize_cache()
            self._batch_counter = 0
            self._total_responses = 0
            logger.info("Reset suffix tree manager")
