# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Based on vLLM suffix decoding implementation.
# Original: https://github.com/vllm-project/vllm
"""
Sequential suffix decoding proposer implementation.

This proposer uses SuffixDecodingCache from srt_plugin for
speculative decoding based on suffix tree pattern matching.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).

    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        max_tree_depth: int,
        max_cached_requests: int,
        max_spec_factor: float,
        min_token_prob: float,
        max_model_len: int,
        enable_in_flight_updates: bool = True,
    ):
        """
        Initialize the suffix decoding proposer.

        Args:
            num_speculative_tokens: Maximum number of speculative tokens per step
            max_tree_depth: Maximum depth of suffix trees for pattern matching
            max_cached_requests: Maximum number of cached requests in the suffix cache
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability for speculation
            max_model_len: Maximum model sequence length
            enable_in_flight_updates: Whether to add newly sampled tokens to suffix
                trees during speculation. Set to False to rely only on pre-loaded
                snapshots. (default: True)
        """
        self.num_speculative_tokens = num_speculative_tokens
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.max_model_len = max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from recipe.srt.srt_plugin.suffix_cache import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.enable_in_flight_updates = enable_in_flight_updates

    def propose(
        self,
        input_batch: "InputBatch",
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch.

        Suffix Decoding will speculate a dynamic number of tokens for each request
        every decoding step, so each entry in the returned list may have different lengths.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            # Only add tokens if in-flight updates are enabled
            if self.enable_in_flight_updates:
                self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)

        # Stop requests that were not seen in the input batch.
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    def load_snapshot(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ) -> None:
        """Load suffix tree snapshots from serialized data.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) containing serialized trees.
            hash_mapping: Dict mapping prompt_hash -> tree_idx for tree lookup.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not snapshots:
            logger.debug("load_snapshot called with empty snapshots, skipping")
            return

        self.suffix_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)
        logger.info(
            f"Loaded {len(snapshots)} suffix trees with {len(hash_mapping)} hash mappings"
        )
