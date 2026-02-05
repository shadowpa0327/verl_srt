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
GPUModelRunner patches for suffix proposer integration.

Patches:
- Drafter initialization in __init__ for suffix methods
- Proposal dispatch logic for suffix methods
- Request lifecycle management for shared memory mode (execute_model hooks)

This module reads configuration from SRTSuffixConfig.get() which is populated
by arg_utils_patches during engine initialization.

Supports three cache modes:
- "snapshot": Uses ParallelSuffixDecodingProposer with snapshot-based loading
- "shared_memory": Uses SharedMemorySuffixDecodingProposer with SpecRL's SuffixCache
- "global_cache": Uses SuffixDecodingProposer with self-maintained cache (no snapshot loading)

Note: Imports are lazy to avoid triggering CUDA initialization at module load time.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


def apply_patches():
    """Apply GPUModelRunner patches.

    Imports are done lazily inside this function to avoid triggering CUDA
    initialization when the module is loaded (e.g., in Ray actors without GPU).
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("GPUModelRunner patches already applied, skipping")
        return

    # Lazy imports to avoid CUDA initialization at module load time
    from ..patching import ArcticPatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Store original methods for wrapping
    _original_init = GPUModelRunner.__init__
    _original_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids
    _original_execute_model = GPUModelRunner.execute_model

    class GPUModelRunnerSuffixPatch(ArcticPatch[GPUModelRunner]):
        """
        Patches GPUModelRunner for suffix decoding support.

        Supports both snapshot mode and shared memory mode:
        - Snapshot mode: Uses ParallelSuffixDecodingProposer with load_snapshot()
        - Shared memory mode: Uses SharedMemorySuffixDecodingProposer with
          execute_model() lifecycle hooks for fetch/evict

        - Extends __init__ to initialize suffix proposers
        - Extends execute_model to handle shared memory lifecycle
        - Extends propose_draft_token_ids to handle suffix methods
        """

        def __init__(self, vllm_config, device):
            """Extended __init__ with suffix proposer initialization.

            Detects suffix method from vllm_config.speculative_config.method.
            Uses SRTSuffixConfig from registry if available, otherwise uses defaults.
            """
            from concurrent.futures import ThreadPoolExecutor

            from vllm.distributed.parallel_state import get_pp_group
            from vllm.v1.sample.rejection_sampler import RejectionSampler

            from recipe.srt.srt_plugin.config import SRTSuffixConfig

            # Check if this is a suffix method directly from vllm_config
            spec_config = vllm_config.speculative_config
            is_suffix_method = (
                spec_config is not None
                and spec_config.method == "suffix"
            )

            # Get SRT config from registry if available, otherwise create from defaults
            srt_config = SRTSuffixConfig.get()
            if is_suffix_method and (srt_config is None or not srt_config.enabled):
                # In subprocess: create config from vllm_config values + defaults
                srt_config = SRTSuffixConfig(
                    enabled=True,
                    method=spec_config.method,
                    num_speculative_tokens=spec_config.num_speculative_tokens,
                    # Use defaults for SRT-specific params not in vllm_config
                )
                logger.debug(
                    "Created SRTSuffixConfig from vllm_config in subprocess: %s",
                    srt_config,
                )

            # Detect cache mode from (in order of priority):
            # 1. Environment variable (for testing/override)
            # 2. SpeculativeConfig.srt_cache_mode (passed from server)
            # 3. SRTSuffixConfig.cache_mode (registered in server process)
            # 4. Default to "snapshot"
            cache_mode = os.environ.get("SRT_CACHE_MODE", "").lower()
            if not cache_mode and spec_config and hasattr(spec_config, "srt_cache_mode"):
                cache_mode = spec_config.srt_cache_mode or ""
            if not cache_mode and srt_config:
                cache_mode = srt_config.cache_mode
            if not cache_mode:
                cache_mode = "snapshot"
            cache_mode = cache_mode.lower()

            # Store cache mode for later use
            self._srt_cache_mode = cache_mode
            self._suffix_cache = None
            self._cache_updater = None
            self._fetch_future = None

            if srt_config and srt_config.enabled:
                # Save the original speculative_config
                saved_spec_config = vllm_config.speculative_config

                # Temporarily hide speculative_config to bypass original validation
                vllm_config.speculative_config = None

                try:
                    # Call original init without speculative_config
                    _original_init(self, vllm_config, device)
                finally:
                    # Restore speculative_config on vllm_config
                    vllm_config.speculative_config = saved_spec_config

                # Now set speculative_config on self (original skipped this)
                self.speculative_config = saved_spec_config

                # Initialize suffix drafter if we're on the last PP rank
                if get_pp_group().is_last_rank:
                    method = srt_config.method

                    if method == "suffix":
                        if cache_mode == "shared_memory":
                            # Shared memory mode: Initialize ThreadPoolExecutor and proposer
                            # SuffixCache is initialized lazily because the cache server
                            # may not be running yet when the worker starts
                            try:
                                self._cache_updater = ThreadPoolExecutor(max_workers=1)
                                self._suffix_cache_initialized = False

                                # Store shared memory config for lazy initialization
                                self._shm_shared_memory_name = srt_config.shared_memory_name
                                self._shm_spec_start_len = srt_config.spec_start_len
                                self._shm_spec_max_len = srt_config.spec_max_len

                                from recipe.srt.srt_plugin.proposers.suffix_decoding_shm import (
                                    SharedMemorySuffixDecodingProposer,
                                )

                                # Pass suffix_cache=None, proposer will lazily init
                                self.drafter = SharedMemorySuffixDecodingProposer(
                                    num_speculative_tokens=srt_config.num_speculative_tokens,
                                    max_model_len=vllm_config.model_config.max_model_len,
                                    spec_prefix_len=7,  # SPECRL_PREFIX_LEN
                                    min_token_prob=srt_config.min_token_prob,
                                    suffix_cache=None,  # Lazy initialization
                                    enable_runahead_speculation=srt_config.enable_runahead_speculation,
                                )
                                logger.info(
                                    f"Using SharedMemorySuffixDecodingProposer (shared memory mode, lazy init, "
                                    f"shm_name={srt_config.shared_memory_name or 'SUFFIX_CACHE'}, "
                                    f"spec_len={srt_config.spec_start_len}-{srt_config.spec_max_len})"
                                )
                            except ImportError as e:
                                logger.error(
                                    f"Failed to import SharedMemorySuffixDecodingProposer: {e}. "
                                    "Falling back to snapshot mode."
                                )
                                cache_mode = "snapshot"
                                self._srt_cache_mode = cache_mode

                        if cache_mode == "snapshot":
                            # Snapshot mode: Use ParallelSuffixDecodingProposer (existing)
                            from recipe.srt.srt_plugin.proposers.suffix_decoding_parallel import (
                                ParallelSuffixDecodingProposer,
                            )

                            logger.info(
                                "Using ParallelSuffixDecodingProposer (batch operations, "
                                "enable_in_flight_updates=%s)",
                                srt_config.enable_in_flight_updates,
                            )
                            self.drafter = ParallelSuffixDecodingProposer(
                                num_speculative_tokens=srt_config.num_speculative_tokens,
                                max_tree_depth=srt_config.max_tree_depth,
                                max_spec_factor=srt_config.max_spec_factor,
                                min_token_prob=srt_config.min_token_prob,
                                max_model_len=vllm_config.model_config.max_model_len,
                                enable_in_flight_updates=srt_config.enable_in_flight_updates,
                                enable_runahead_speculation=srt_config.enable_runahead_speculation,
                            )

                        if cache_mode == "global_cache":
                            # Global cache mode: SuffixDecodingProposer with self-maintained cache
                            # (no snapshot loading, cache builds from in-flight tokens only)
                            from recipe.srt.srt_plugin.proposers.suffix_decoding import (
                                SuffixDecodingProposer,
                            )

                            logger.info(
                                "Using SuffixDecodingProposer (global_cache mode, "
                                "enable_in_flight_updates=%s)",
                                srt_config.enable_in_flight_updates,
                            )
                            self.drafter = SuffixDecodingProposer(
                                num_speculative_tokens=srt_config.num_speculative_tokens,
                                max_tree_depth=srt_config.max_tree_depth,
                                max_cached_requests=srt_config.max_cached_requests,
                                max_spec_factor=srt_config.max_spec_factor,
                                min_token_prob=srt_config.min_token_prob,
                                max_model_len=vllm_config.model_config.max_model_len,
                                enable_in_flight_updates=srt_config.enable_in_flight_updates,
                                enable_runahead_speculation=srt_config.enable_runahead_speculation,
                            )

                    # Set up rejection sampler (same as other spec decode methods)
                    self.rejection_sampler = RejectionSampler()

                # Fix uniform_decode_query_len which depends on speculative_config
                if self.speculative_config:
                    self.uniform_decode_query_len = (
                        1 + self.speculative_config.num_speculative_tokens
                    )
            else:
                # For non-suffix methods, just call original init
                _original_init(self, vllm_config, device)

        def __del__(self):
            """Cleanup resources."""
            if hasattr(self, '_cache_updater') and self._cache_updater is not None:
                self._cache_updater.shutdown(wait=False)

        def _ensure_suffix_cache_initialized(self):
            """Lazily initialize SuffixCache on first use.

            The cache server may not be running when the worker starts,
            so we defer initialization until the first execute_model call.

            Reads configuration from instance variables set during __init__:
            - self._shm_shared_memory_name: Name of shared memory segment
            - self._shm_spec_start_len: Initial/minimum speculation length
            - self._shm_spec_max_len: Maximum speculation length
            """
            if not getattr(self, '_suffix_cache_initialized', True):
                try:
                    from srt_plugin.shm_cache.suffix_cache import SuffixCache

                    # Read configuration from instance variables (set in __init__)
                    shared_memory_name = getattr(self, '_shm_shared_memory_name', "")
                    spec_start_len = getattr(self, '_shm_spec_start_len', 2)
                    spec_max_len = getattr(self, '_shm_spec_max_len', 16)

                    # Create SuffixCache with configurable parameters
                    self._suffix_cache = SuffixCache(
                        shared_memory_name=shared_memory_name,
                        spec_start_len=spec_start_len,
                        spec_max_len=spec_max_len,
                    )
                    self._suffix_cache_initialized = True

                    # Also inject into proposer if it exists
                    if hasattr(self, 'drafter') and hasattr(self.drafter, '_cache'):
                        self.drafter._cache = self._suffix_cache

                    logger.info(
                        f"Lazily initialized SuffixCache for shared memory mode "
                        f"(shm_name={shared_memory_name or 'SUFFIX_CACHE'}, "
                        f"spec_len={spec_start_len}-{spec_max_len})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize SuffixCache: {e}. "
                                   "Speculation will not use shared memory.")
                    self._suffix_cache = None
                    self._suffix_cache_initialized = True

        def execute_model(
            self,
            scheduler_output,
            intermediate_tensors=None,
        ):
            """Extended execute_model with shared memory cache lifecycle hooks.

            For shared memory mode:
            1. Lazily initialize SuffixCache if needed
            2. Evict finished requests from cache
            3. Submit async fetch for new requests
            4. Run original execute_model
            """
            from concurrent.futures import Future

            # Lazy init for shared memory mode
            if self._srt_cache_mode == "shared_memory":
                self._ensure_suffix_cache_initialized()

            if self._srt_cache_mode == "shared_memory" and self._suffix_cache is not None:
                # 1. Cleanup finished requests
                for req_id in scheduler_output.finished_req_ids:
                    try:
                        self._suffix_cache.evict_responses(req_id)
                    except Exception as e:
                        logger.debug(f"evict_responses failed for {req_id}: {e}")

                # 2. Async fetch for new requests
                if scheduler_output.scheduled_new_reqs:
                    def fetch_suffix_responses():
                        req_ids = [r.req_id for r in scheduler_output.scheduled_new_reqs]
                        prompts = [r.prompt_token_ids for r in scheduler_output.scheduled_new_reqs]

                        # Extract pre-computed hashes from sampling_params.extra_args
                        # This ensures hash consistency between secondary cache updates
                        # (which use prompt_ids from tokenization) and primary fetches.
                        pre_computed_hashes = []
                        for r in scheduler_output.scheduled_new_reqs:
                            hash_val = 0
                            if hasattr(r, 'sampling_params') and r.sampling_params:
                                extra_args = getattr(r.sampling_params, 'extra_args', None)
                                if extra_args:
                                    prompt_hash = extra_args.get("prompt_hash")
                                    if prompt_hash is not None:
                                        # Handle both int and hex string formats
                                        if isinstance(prompt_hash, str):
                                            hash_val = int(prompt_hash, 16) if prompt_hash else 0
                                        else:
                                            hash_val = int(prompt_hash)
                            pre_computed_hashes.append(hash_val)

                        self._suffix_cache.fetch_responses_by_prompts_batch(
                            req_ids, prompts, pre_computed_hashes
                        )
                        return 1

                    self._fetch_future = self._cache_updater.submit(fetch_suffix_responses)
                else:
                    self._fetch_future = Future()
                    self._fetch_future.set_result(1)

            # Call original execute_model
            return _original_execute_model(self, scheduler_output, intermediate_tensors)

        def propose_draft_token_ids(
            self,
            scheduler_output,
            sampled_token_ids,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            common_attn_metadata,
        ):
            """Extended propose_draft_token_ids with suffix method handling and fetch sync."""
            # Check suffix method directly from speculative_config
            is_suffix_method = (
                self.speculative_config is not None
                and self.speculative_config.method == "suffix"
            )

            # Check if speculative decoding should be disabled due to high batch size
            if (
                self.speculative_config
                and self.speculative_config.disable_by_batch_size is not None
                and len(self.input_batch.req_ids)
                > self.speculative_config.disable_by_batch_size
            ):
                logger.debug(
                    "Speculative decoding disabled: batch size %d exceeds threshold %d",
                    len(self.input_batch.req_ids),
                    self.speculative_config.disable_by_batch_size,
                )
                if isinstance(sampled_token_ids, list):
                    return [[] for _ in sampled_token_ids]
                else:
                    batch_size = sampled_token_ids.shape[0]
                    return [[] for _ in range(batch_size)]

            # Handle suffix method
            if is_suffix_method:
                assert isinstance(sampled_token_ids, list)

                # For shared memory mode: wait for async fetch and update spec_len
                if self._srt_cache_mode == "shared_memory":
                    # Wait for async fetch to complete before speculation
                    if self._fetch_future is not None:
                        self._fetch_future.result()
                        self._fetch_future = None

                    # Update spec_len for cache coherency
                    if self._suffix_cache is not None:
                        for i, token_ids in enumerate(sampled_token_ids):
                            if token_ids and i < len(self.input_batch.req_ids):
                                req_id = self.input_batch.req_ids[i]
                                self._suffix_cache.update_spec_len(req_id, len(token_ids))

                # Dispatch to drafter (both modes use same interface)
                draft_token_ids = self.drafter.propose(
                    input_batch=self.input_batch, sampled_token_ids=sampled_token_ids
                )
                return draft_token_ids

            # For other methods, use original implementation
            return _original_propose_draft_token_ids(
                self,
                scheduler_output,
                sampled_token_ids,
                sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                common_attn_metadata,
            )

    # Use ArcticPatch's apply_patch method
    GPUModelRunnerSuffixPatch.apply_patch()

    _patches_applied = True
    logger.info("Applied GPUModelRunner suffix decoding patches")
