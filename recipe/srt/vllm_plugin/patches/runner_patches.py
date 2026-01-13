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

Note: Imports are lazy to avoid triggering CUDA initialization at module load time.
"""

import logging

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
    from arctic_inference.patching import ArcticPatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # Store original methods for wrapping
    _original_init = GPUModelRunner.__init__
    _original_propose_draft_token_ids = GPUModelRunner.propose_draft_token_ids

    class GPUModelRunnerSuffixPatch(ArcticPatch[GPUModelRunner]):
        """
        Patches GPUModelRunner for suffix decoding support.

        - Extends __init__ to initialize suffix proposers
        - Extends propose_draft_token_ids to handle suffix methods
        """

        def __init__(self, vllm_config, device):
            """Extended __init__ with suffix proposer initialization.

            For suffix methods, we temporarily hide speculative_config from the
            original init to bypass its unknown method validation, then restore
            it and set up our drafter after.
            """
            from vllm.distributed.parallel_state import get_pp_group
            from vllm.v1.sample.rejection_sampler import RejectionSampler

            # Check if this is a suffix method that needs special handling
            spec_config = vllm_config.speculative_config
            is_suffix_method = (
                spec_config is not None and
                spec_config.method in ("suffix", "suffix_remote")
            )

            if is_suffix_method:
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
                    method = self.speculative_config.method

                    if method == "suffix":
                        use_parallel = getattr(
                            self.speculative_config, 'suffix_decoding_use_parallel', True
                        )

                        if use_parallel:
                            from recipe.srt.vllm_plugin.proposers.suffix_decoding_parallel import (
                                ParallelSuffixDecodingProposer
                            )
                            logger.info("Using ParallelSuffixDecodingProposer (batch operations)")
                            self.drafter = ParallelSuffixDecodingProposer(vllm_config)
                        else:
                            from recipe.srt.vllm_plugin.proposers.suffix_decoding import (
                                SuffixDecodingProposer
                            )
                            logger.info("Using SuffixDecodingProposer (sequential)")
                            self.drafter = SuffixDecodingProposer(vllm_config)

                    elif method == "suffix_remote":
                        from recipe.srt.vllm_plugin.proposers.suffix_decoding_remote import (
                            RemoteSuffixDecodingProposer
                        )
                        logger.info("Using RemoteSuffixDecodingProposer (gRPC client)")
                        self.drafter = RemoteSuffixDecodingProposer(vllm_config)

                    # Set up rejection sampler (same as other spec decode methods)
                    self.rejection_sampler = RejectionSampler()

                # Fix uniform_decode_query_len which depends on speculative_config
                self.uniform_decode_query_len = 1 + self.speculative_config.num_speculative_tokens
            else:
                # For non-suffix methods, just call original init
                _original_init(self, vllm_config, device)

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
            """Extended propose_draft_token_ids with suffix method handling."""
            # Check if speculative decoding should be disabled due to high batch size
            if (self.speculative_config.disable_by_batch_size is not None
                and len(self.input_batch.req_ids) > self.speculative_config.disable_by_batch_size):
                logger.debug(
                    "Speculative decoding disabled: batch size %d exceeds threshold %d",
                    len(self.input_batch.req_ids),
                    self.speculative_config.disable_by_batch_size
                )
                if isinstance(sampled_token_ids, list):
                    return [[] for _ in sampled_token_ids]
                else:
                    batch_size = sampled_token_ids.shape[0]
                    return [[] for _ in range(batch_size)]

            method = self.speculative_config.method

            if method == "suffix":
                assert isinstance(sampled_token_ids, list)
                from recipe.srt.vllm_plugin.proposers.suffix_decoding import (
                    SuffixDecodingProposer
                )
                from recipe.srt.vllm_plugin.proposers.suffix_decoding_parallel import (
                    ParallelSuffixDecodingProposer
                )
                assert isinstance(self.drafter, (SuffixDecodingProposer, ParallelSuffixDecodingProposer))
                draft_token_ids = self.drafter.propose(
                    input_batch=self.input_batch,
                    sampled_token_ids=sampled_token_ids
                )
                return draft_token_ids

            elif method == "suffix_remote":
                assert isinstance(sampled_token_ids, list)
                from recipe.srt.vllm_plugin.proposers.suffix_decoding_remote import (
                    RemoteSuffixDecodingProposer
                )
                assert isinstance(self.drafter, RemoteSuffixDecodingProposer)
                draft_token_ids = self.drafter.propose(
                    input_batch=self.input_batch,
                    sampled_token_ids=sampled_token_ids
                )
                return draft_token_ids

            else:
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
