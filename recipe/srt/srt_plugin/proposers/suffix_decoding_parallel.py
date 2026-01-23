# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Based on vLLM suffix decoding implementation.
# Original: https://github.com/vllm-project/vllm
"""
Parallel suffix decoding proposer implementation.

This proposer uses ParallelSuffixDecodingCache from srt_plugin.suffix_cache for
batch-parallel speculative decoding with hash-based tree mapping.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)


class ParallelSuffixDecodingProposer:
    """
    Parallel speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).

    This implementation uses ParallelSuffixDecodingCache with batch operations for improved
    performance when processing multiple concurrent requests. Key improvements:

    - Uses batch_add_tokens() to add tokens to multiple trees in parallel
    - Uses batch_speculate() to perform speculation across multiple requests in parallel
    - Achieves ~2x speedup for batch sizes >= 32 with 4 threads
    - Supports loading suffix tree snapshots with hash-based tree mapping

    For batch sizes below parallel_threshold, operations run sequentially to avoid overhead.
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        max_tree_depth: int,
        max_spec_factor: float,
        min_token_prob: float,
        max_model_len: int,
        enable_in_flight_updates: bool = True,
        num_threads: int = 8,
        parallel_threshold: int = 8,
        enable_runahead_speculation: bool = False,
    ):
        """
        Initialize the parallel suffix decoding proposer.

        Args:
            num_speculative_tokens: Maximum number of speculative tokens per step
            max_tree_depth: Maximum depth of suffix trees for pattern matching
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability for speculation
            max_model_len: Maximum model sequence length
            enable_in_flight_updates: Whether to add newly sampled tokens to suffix
                trees during speculation. Set to False to rely only on pre-loaded
                snapshots. (default: True)
            num_threads: Number of threads for parallel operations (default: 8)
                -1 = auto-detect from CPU count
                0 = sequential execution
                >0 = use specified number of threads
            parallel_threshold: Minimum batch size to trigger parallelization (default: 8)
                Smaller batches will run sequentially to avoid overhead
            enable_runahead_speculation: Whether to apply speculative decoding to
                runahead/secondary requests (with "runahead_" prefix). (default: False)
                False = runahead requests get no draft tokens (isolates metrics)
                True = runahead requests also get speculative tokens
        """
        self.num_speculative_tokens = num_speculative_tokens
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.max_model_len = max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from recipe.srt.srt_plugin.suffix_cache import ParallelSuffixDecodingCache

        # Initialize parallel cache with batched speculation support
        self.suffix_cache = ParallelSuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            num_threads=num_threads,
            parallel_threshold=parallel_threshold,
        )
        self.enable_in_flight_updates = enable_in_flight_updates
        self.enable_runahead_speculation = enable_runahead_speculation
        logger.info(
            "Initialized ParallelSuffixDecodingProposer with num_threads=%s, "
            "parallel_threshold=%s, enable_in_flight_updates=%s, enable_runahead_speculation=%s",
            num_threads,
            parallel_threshold,
            enable_in_flight_updates,
            enable_runahead_speculation,
        )

    def propose(
        self,
        input_batch: "InputBatch",
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch.

        This method processes all requests in parallel using optimized batch operations:
        1. Identifies active requests and starts new ones if needed
        2. Adds newly sampled tokens to all active requests in parallel (batch_add_tokens)
        3. Performs optimized speculation using propose_from_batch (zero-copy context extraction)

        Args:
            input_batch: Batch of input requests
            sampled_token_ids: List of newly sampled token IDs for each request

        Returns:
            List of draft token IDs for each request
        """
        import numpy as np

        # Collect data for batch operations
        req_ids_to_add_tokens = []
        tokens_to_add = []

        # Track which input batch indices should receive draft tokens
        input_indices_with_drafts = []
        batch_indices_for_speculation = []

        # Process each request and prepare batch data
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                continue

            # Skip runahead/secondary requests unless speculation is explicitly enabled.
            # This isolates spec decode metrics to primary requests only.
            # Runahead requests are identified by "runahead_" prefix in their request ID.
            if not self.enable_runahead_speculation and req_id.startswith("runahead_"):
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                continue

            index = input_batch.req_id_to_index[req_id]

            # Start new requests if needed
            if req_id not in self.suffix_cache.active_requests:
                # Note: ParallelSuffixDecodingCache doesn't have cached_requests
                # (no global tree), so we just start new requests directly
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Use pre-computed hash if available (ensures consistency with trainer)
                pre_computed_hash = None
                if hasattr(input_batch, "prompt_hashes"):
                    pre_computed_hash = input_batch.prompt_hashes.get(req_id)
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(
                    req_id, prompt_token_ids, pre_computed_hash=pre_computed_hash
                )

            # Collect tokens to add (for batch_add_tokens)
            req_ids_to_add_tokens.append(req_id)
            tokens_to_add.append(sampled_ids)

            # Track batch indices for optimized speculation
            batch_indices_for_speculation.append(i)
            input_indices_with_drafts.append(i)

        # BATCH ADD TOKENS: Add all newly sampled tokens in parallel
        # Only add tokens if in-flight updates are enabled
        if self.enable_in_flight_updates and req_ids_to_add_tokens:
            self.suffix_cache.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)

        # OPTIMIZED BATCH SPECULATE: Use propose_from_batch for zero-copy context extraction
        drafts = []
        if batch_indices_for_speculation:
            # Prepare numpy arrays for propose_from_batch
            batch_indices_array = np.array(batch_indices_for_speculation, dtype=np.int32)

            # Extract token_ids_cpu for the batch (2D slice)
            token_ids_batch = input_batch.token_ids_cpu[batch_indices_array]

            # Extract num_tokens for the batch
            num_tokens_batch = input_batch.num_tokens_no_spec[batch_indices_array].astype(
                np.int32
            )

            # Get tree indices for each request
            tree_indices_list = []
            for i in batch_indices_for_speculation:
                req_id = input_batch.req_ids[i]
                tree_idx = self.suffix_cache.get_tree_idx_for_request(req_id)
                if tree_idx is None:
                    raise RuntimeError(
                        f"Request {req_id} should be active but tree_idx is None"
                    )
                tree_indices_list.append(tree_idx)
            tree_indices_batch = np.array(tree_indices_list, dtype=np.int32)

            # Calculate max_spec_tokens (use minimum across batch for simplicity)
            max_spec_tokens_list = [
                min(
                    self.num_speculative_tokens,
                    self.max_model_len - input_batch.num_tokens_no_spec[i] - 1,
                )
                for i in batch_indices_for_speculation
            ]
            min_max_spec_tokens = (
                min(max_spec_tokens_list)
                if max_spec_tokens_list
                else self.num_speculative_tokens
            )

            # Call optimized C++ method (zero-copy, parallelized)
            drafts = self.suffix_cache.propose_from_batch(
                token_ids_cpu=token_ids_batch,
                num_tokens=num_tokens_batch,
                tree_indices=tree_indices_batch,
                max_spec_tokens=min_max_spec_tokens,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

        # Build result list with draft tokens
        draft_token_ids: list[list[int]] = []
        draft_idx = 0

        for i in range(len(sampled_token_ids)):
            if i in input_indices_with_drafts:
                # This request has a draft
                draft_token_ids.append(drafts[draft_idx].token_ids)
                draft_idx += 1
            else:
                # This request was skipped
                draft_token_ids.append([])

        # Stop requests that were not seen in the input batch
        active_req_ids = set(self.suffix_cache.active_requests)
        input_req_ids = set(input_batch.req_id_to_index.keys())

        for req_id in active_req_ids - input_req_ids:
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    def get_stats(self) -> dict:
        """
        Get statistics about the parallel suffix cache.

        Returns:
            Dictionary with cache statistics including:
            - num_active_requests: Number of currently active requests
            - max_tree_depth: Maximum tree depth
            - num_threads: Number of threads configured
            - parallel_threshold: Batch size threshold for parallelization
            - num_trees_in_forest: Total number of trees in the forest
        """
        return self.suffix_cache.get_stats()

    def load_snapshot(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ) -> None:
        """
        Load suffix tree snapshots for hash-based tree matching.

        This method enables distributed pattern sharing by loading pre-built
        suffix trees. New requests with matching prompt hash will automatically
        reuse the corresponding pre-loaded tree.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples from
                ParallelSuffixDecodingCache.create_snapshot()
            hash_mapping: Dict mapping prompt_hash -> tree_idx from
                ParallelSuffixDecodingCache.create_snapshot(include_hash_mapping=True)
        """
        if not snapshots:
            logger.debug("load_snapshot called with empty snapshots, skipping")
            return

        # Load trees into suffix cache with hash mapping for tree reuse
        self.suffix_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)

        total_bytes = sum(len(s[1]) for s in snapshots)
        logger.info(
            "Loaded %d suffix trees (%d bytes total) with %d hash mappings",
            len(snapshots),
            total_bytes,
            len(hash_mapping),
        )

    def create_snapshot(self) -> tuple[list[tuple[int, bytes]], dict[str, int]]:
        """
        Create a snapshot of the current suffix cache state.

        Returns:
            Tuple of (snapshots, hash_mapping) where:
            - snapshots: List of (tree_idx, snapshot_bytes) tuples
            - hash_mapping: Dict mapping prompt_hash -> tree_idx
        """
        result = self.suffix_cache.create_snapshot(include_hash_mapping=True)
        if isinstance(result, tuple):
            snapshots, hash_mapping = result
        else:
            snapshots = result
            hash_mapping = {}

        if not snapshots:
            return [], {}

        return snapshots, hash_mapping
