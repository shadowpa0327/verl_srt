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
AsyncEngineArgs patches for SRT suffix decoding config extraction.

This module intercepts create_speculative_config to:
1. Extract srt_* prefixed params from the speculative_config dict
2. Register them as SRTSuffixConfig for runner_patches to access
3. Pass only vLLM-standard params to SpeculativeConfig validation
"""

import logging

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


def apply_patches():
    """Apply AsyncEngineArgs patches for SRT config extraction.

    Patches create_speculative_config to extract srt_* params and register
    them via SRTSuffixConfig before passing to vLLM.
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("AsyncEngineArgs patches already applied, skipping")
        return

    from vllm.engine.arg_utils import AsyncEngineArgs

    # Store original method
    _original_create_speculative_config = AsyncEngineArgs.create_speculative_config

    def patched_create_speculative_config(
        self,
        target_model_config,
        target_parallel_config,
        enable_chunked_prefill,
        disable_log_stats,
    ):
        """Extended create_speculative_config that extracts SRT params.

        If speculative_config contains srt_* prefixed keys, they are extracted
        and registered via SRTSuffixConfig. Only vLLM-standard keys are passed
        to SpeculativeConfig validation.

        srt_cache_mode is set as an attribute on the returned SpeculativeConfig
        for worker processes to access.
        """
        from recipe.srt.srt_plugin.config import SRTSuffixConfig

        srt_cache_mode = None  # Will be set if SRT is enabled

        if self.speculative_config:
            # Extract SRT params and register
            srt_config, vllm_only = SRTSuffixConfig.extract_from_dict(
                self.speculative_config
            )
            SRTSuffixConfig.register(srt_config)

            if srt_config.enabled:
                # Save cache_mode to set on SpeculativeConfig later
                srt_cache_mode = srt_config.cache_mode
                logger.info(
                    f"Registered SRTSuffixConfig: method={srt_config.method}, "
                    f"num_speculative_tokens={srt_config.num_speculative_tokens}, "
                    f"max_tree_depth={srt_config.max_tree_depth}, "
                    f"cache_mode={srt_config.cache_mode}"
                )

            # Replace with filtered dict for SpeculativeConfig validation
            self.speculative_config = vllm_only

        # Call original method with filtered config
        result = _original_create_speculative_config(
            self,
            target_model_config,
            target_parallel_config,
            enable_chunked_prefill,
            disable_log_stats,
        )

        # Set srt_cache_mode as attribute on result for worker access
        if result is not None and srt_cache_mode is not None:
            result.srt_cache_mode = srt_cache_mode
            logger.debug(f"Set srt_cache_mode={srt_cache_mode} on SpeculativeConfig")

        return result

    # Apply patch
    AsyncEngineArgs.create_speculative_config = patched_create_speculative_config

    _patches_applied = True
    logger.info("Applied AsyncEngineArgs SRT config extraction patch")
