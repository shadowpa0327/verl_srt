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
Shared memory patches for SRT.

Extends runner_patches.py to support shared memory mode, enabling zero-copy
shared memory cache access instead of snapshot-based loading.

When shared memory mode is enabled:
- Sets SRT_CACHE_MODE=shared_memory environment variable
- Applies runner_patches which then:
  - Initializes SuffixCache from specrl for shared memory access
  - Uses SharedMemorySuffixDecodingProposer instead of ParallelSuffixDecodingProposer
  - Adds execute_model() hooks for request lifecycle (fetch/evict)

Usage:
    # In vLLM server or worker initialization
    if srt_use_shared_memory:
        from recipe.srt.srt_plugin.patches.shm_patches import apply_shared_memory_patches
        apply_shared_memory_patches()
"""

import logging
import os

logger = logging.getLogger(__name__)

# Track if shared memory patches have been applied in this process
_shm_patches_applied = False


def apply_shared_memory_patches():
    """
    Apply shared memory mode patches for SRT.

    This sets the SRT_CACHE_MODE environment variable to "shared_memory"
    and applies runner_patches. The runner_patches module detects the
    cache mode and initializes the appropriate components:

    - SuffixCache from specrl for shared memory access
    - SharedMemorySuffixDecodingProposer for speculation
    - execute_model() hooks for request lifecycle management

    Note: These patches modify global class behavior. Only apply once per process.
    """
    global _shm_patches_applied
    if _shm_patches_applied:
        logger.debug("Shared memory patches already applied, skipping")
        return

    # Set environment variable for cache mode detection
    os.environ["SRT_CACHE_MODE"] = "shared_memory"
    logger.debug("Set SRT_CACHE_MODE=shared_memory")

    try:
        # Apply runner_patches which will detect shared_memory mode
        from recipe.srt.srt_plugin.patches import runner_patches
        runner_patches.apply_patches()

        _shm_patches_applied = True
        logger.info("Applied shared memory patches for SRT (via runner_patches)")

    except ImportError as e:
        logger.error(f"Failed to import runner_patches: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply shared memory patches: {e}")
        raise


def is_shared_memory_mode() -> bool:
    """
    Check if SRT shared memory mode is enabled.

    Checks for:
    1. Environment variable SRT_CACHE_MODE=shared_memory
    2. Configuration flag passed to vLLM (if available)

    Returns:
        True if shared memory mode should be used.
    """
    # Check environment variable
    cache_mode = os.environ.get("SRT_CACHE_MODE", "").lower()
    if cache_mode == "shared_memory":
        return True

    # Check SRTSuffixConfig if available
    try:
        from recipe.srt.srt_plugin.config import SRTSuffixConfig
        srt_config = SRTSuffixConfig.get()
        if srt_config and srt_config.cache_mode == "shared_memory":
            return True
    except ImportError:
        pass

    return False


def maybe_apply_shared_memory_patches():
    """
    Conditionally apply shared memory patches based on configuration.

    This is a convenience function that checks is_shared_memory_mode() and
    applies patches if shared memory mode is enabled.

    Call this in worker initialization code to handle both modes transparently.
    """
    if is_shared_memory_mode():
        logger.info("SRT shared memory mode detected, applying patches")
        apply_shared_memory_patches()
    else:
        logger.debug("SRT shared memory mode not enabled, skipping patches")
