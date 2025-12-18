# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Based on vLLM suffix decoding implementation.
# Original: https://github.com/vllm-project/vllm
"""
Remote suffix decoding proposer implementation.

This proposer connects to a suffix decoding gRPC server for
speculative decoding with centralized tree management.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)


class RemoteSuffixDecodingProposer:
    """
    Remote speculative decoding proposer that connects to a suffix decoding
    gRPC server.

    This implementation uses the SuffixDecodingClient from arctic_inference to
    communicate with a pre-launched suffix decoding server. The server manages
    the suffix trees and performs speculation, while this proposer acts as a
    thin client.

    Usage:
        1. Start the server:
           python -m arctic_inference.suffix_decoding.server --port 50051

        2. Configure vLLM:
           --speculative-method suffix_remote
           --suffix-decoding-server-host localhost
           --suffix-decoding-server-port 50051
    """

    def __init__(self, vllm_config: "VllmConfig"):
        """
        Initialize the remote suffix decoding proposer.

        Args:
            vllm_config: vLLM configuration object
        """
        config = vllm_config.speculative_config
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        host = config.suffix_decoding_server_host
        port = config.suffix_decoding_server_port

        # Lazy import to avoid error when not used
        from arctic_inference.suffix_decoding.client import SuffixDecodingClient
        import uuid

        self.client = SuffixDecodingClient(host=host, port=port)

        # Generate a unique prefix to avoid request ID conflicts across workers
        # When multiple workers share the same server, each needs unique request IDs
        self.worker_prefix = str(uuid.uuid4())[:8] + "_"

        # Health check - fail fast if server is unavailable
        try:
            stats = self.client.get_stats()
            logger.info(
                "Connected to suffix decoding server at %s:%d (prefix=%s), "
                "active_requests=%d, max_tree_depth=%d",
                host, port, self.worker_prefix,
                stats['num_active_requests'],
                stats['max_tree_depth']
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to suffix decoding server at {host}:{port}. "
                f"Make sure the server is running. Error: {e}"
            ) from e

        # Track active requests locally since client doesn't expose this
        self.active_requests: set[str] = set()

    def propose(
        self,
        input_batch: "InputBatch",
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch.

        This method communicates with the remote server to:
        1. Start new requests with their prompts
        2. Add newly sampled tokens to active requests
        3. Perform speculation to get draft tokens
        4. Stop completed requests

        Args:
            input_batch: Batch of input requests
            sampled_token_ids: List of newly sampled token IDs for each request

        Returns:
            List of draft token IDs for each request
        """
        # Collect data for batch operations
        req_ids_to_add_tokens = []
        tokens_to_add = []

        req_ids_to_speculate = []
        contexts_to_speculate = []
        max_spec_tokens_list = []

        # Map from req_id to index in speculation batch
        req_id_to_spec_index = {}

        # Track which input batch indices should receive draft tokens
        input_indices_with_drafts = []

        # Process each request and prepare batch data
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills
                continue

            # Skip requests that require unsupported sampling parameters
            req_id = input_batch.req_ids[i]
            if req_id in input_batch.spec_decode_unsupported_reqs:
                continue

            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have reached max model length
                continue

            index = input_batch.req_id_to_index[req_id]

            # Add worker prefix to make request ID unique across workers
            prefixed_req_id = self.worker_prefix + str(req_id)

            # Start new requests if needed
            if req_id not in self.active_requests:
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[
                    index, :num_prompt_tokens]
                # Start a new request on the server with prefixed ID
                self.client.start_request(prefixed_req_id, prompt_token_ids.tolist())
                self.active_requests.add(req_id)

            # Collect tokens to add (use prefixed ID for server)
            req_ids_to_add_tokens.append(prefixed_req_id)
            tokens_to_add.append(sampled_ids)

            # Collect contexts for speculation
            # Suffix decoding uses the most recent tokens up to max_tree_depth
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]

            req_ids_to_speculate.append(prefixed_req_id)
            contexts_to_speculate.append(pattern.tolist())
            max_spec_tokens_list.append(
                min(self.num_speculative_tokens,
                    self.max_model_len - num_tokens - 1)
            )

            # Map req_id to its position in speculation results
            req_id_to_spec_index[req_id] = len(req_ids_to_speculate) - 1
            input_indices_with_drafts.append(i)

        # BATCH ADD TOKENS: Add all newly sampled tokens
        if req_ids_to_add_tokens:
            self.client.batch_add_tokens(req_ids_to_add_tokens, tokens_to_add)

        # BATCH SPECULATE: Perform speculation for all requests
        drafts = []
        if req_ids_to_speculate:
            # Use minimum max_spec_tokens across all requests
            min_max_spec_tokens = (
                min(max_spec_tokens_list) if max_spec_tokens_list
                else self.num_speculative_tokens
            )

            drafts = self.client.batch_speculate(
                req_ids=req_ids_to_speculate,
                contexts=contexts_to_speculate,
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

        # Stop requests that are no longer in the input batch
        input_req_ids = set(input_batch.req_id_to_index.keys())
        completed_req_ids = self.active_requests - input_req_ids

        for req_id in completed_req_ids:
            # Use prefixed ID when communicating with server
            prefixed_req_id = self.worker_prefix + str(req_id)
            self.client.stop_request(prefixed_req_id)
            self.active_requests.discard(req_id)

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        """No model to load - server manages the suffix trees."""
        pass

    def get_stats(self) -> dict:
        """
        Get statistics from the remote suffix decoding server.

        Returns:
            Dictionary with server statistics including:
            - num_active_requests: Number of currently active requests
            - max_tree_depth: Maximum tree depth configured on server
            - num_threads: Number of threads used by server
            - parallel_threshold: Batch size threshold for parallelization
            - num_trees_in_forest: Total number of trees in the forest
        """
        return self.client.get_stats()

    def __del__(self):
        """Clean up client connection on destruction."""
        if hasattr(self, 'client'):
            self.client.close()
