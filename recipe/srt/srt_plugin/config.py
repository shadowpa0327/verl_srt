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
SRT Suffix Decoding Configuration.

This module provides a clean configuration dataclass for SRT suffix decoding,
avoiding the complex SpeculativeConfig patching. Config travels via the
speculative_config dict with srt_* prefixed keys, then gets extracted and
registered in the subprocess for runner_patches to access.
"""

from dataclasses import dataclass
from typing import Optional

# Module-level registry for cross-function access within subprocess
_srt_config: Optional["SRTSuffixConfig"] = None


@dataclass
class SRTSuffixConfig:
    """Configuration for SRT suffix decoding.

    This config is extracted from the speculative_config dict (keys with srt_ prefix)
    and registered for access by runner_patches.py when creating the drafter.
    """

    enabled: bool = False
    method: str = "suffix"
    num_speculative_tokens: int = 24
    max_tree_depth: int = 24
    max_spec_factor: float = 1.0
    min_token_prob: float = 0.1
    enable_in_flight_updates: bool = True
    # Cache mode: "snapshot" for snapshot-based loading, "shared_memory" for
    # zero-copy shared memory access via SpecRL's SuffixCache
    cache_mode: str = "snapshot"

    # Shared memory mode specific params
    shared_memory_name: str = ""  # Empty string means use default "SUFFIX_CACHE"
    spec_start_len: int = 2  # Initial/minimum speculation length
    spec_max_len: int = 16  # Maximum speculation length

    # Prefix for keys in speculative_config dict
    KEY_PREFIX = "srt_"

    @classmethod
    def extract_from_dict(cls, spec_config: dict) -> tuple["SRTSuffixConfig", dict]:
        """Extract SRT params from speculative_config dict.

        SRT params are identified by the 'srt_' prefix. They are extracted,
        the prefix is stripped, and the remaining dict is returned for
        SpeculativeConfig validation.

        Args:
            spec_config: The speculative_config dict from engine_kwargs

        Returns:
            Tuple of (srt_config, vllm_only_dict) where:
            - srt_config: SRTSuffixConfig with extracted params
            - vllm_only_dict: Dict with only vLLM-standard fields
        """
        srt_params = {}
        vllm_params = {}

        for k, v in spec_config.items():
            if k.startswith(cls.KEY_PREFIX):
                # Strip prefix: srt_max_tree_depth → max_tree_depth
                param_name = k[len(cls.KEY_PREFIX) :]
                srt_params[param_name] = v
            else:
                vllm_params[k] = v

        # Check if SRT is enabled (method is suffix or suffix_remote)
        method = vllm_params.get("method")
        if method in ("suffix", "suffix_remote"):
            # Include num_speculative_tokens from vllm_params if present
            if "num_speculative_tokens" in vllm_params:
                srt_params.setdefault(
                    "num_speculative_tokens", vllm_params["num_speculative_tokens"]
                )
            srt_params["method"] = method
            srt_config = cls(enabled=True, **srt_params)

            # Note: srt_cache_mode is handled separately in arg_utils_patches
            # since SpeculativeConfig doesn't have this field as a dataclass field
        else:
            srt_config = cls(enabled=False)

        return srt_config, vllm_params

    @classmethod
    def register(cls, config: "SRTSuffixConfig") -> None:
        """Register config for later access by runner_patches.

        This should be called in arg_utils_patches after extracting the config
        from speculative_config dict.

        Args:
            config: The SRTSuffixConfig to register
        """
        global _srt_config
        _srt_config = config

    @classmethod
    def get(cls) -> Optional["SRTSuffixConfig"]:
        """Get the registered SRT config.

        Returns:
            The registered SRTSuffixConfig, or None if not registered.
        """
        return _srt_config

    @classmethod
    def clear(cls) -> None:
        """Clear the registered config (useful for testing)."""
        global _srt_config
        _srt_config = None
