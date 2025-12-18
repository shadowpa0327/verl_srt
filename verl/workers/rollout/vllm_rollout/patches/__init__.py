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
vLLM patches for suffix decoding integration.

This module provides monkey patches for vanilla vLLM to enable suffix
decoding speculative generation without maintaining a vLLM fork.

Usage:
    from verl.workers.rollout.vllm_rollout.patches import apply_all_patches

    # Apply before importing vLLM LLM class
    apply_all_patches()

    from vllm import LLM
    llm = LLM(...)

Patches Applied:
    1. SpeculativeConfig - adds suffix decoding fields and validation
    2. GPUModelRunner - initializes suffix proposers and handles dispatch
    3. InputBatch - adds prompt_hashes field for hash-based tree mapping
    4. InprocClient - fixes get_output() to call post_step
    5. LLM - adds load_snapshot() method for suffix tree loading

Environment Variables:
    VERL_VLLM_AUTO_PATCH: Set to "1" to auto-apply patches on import
"""

import logging
import os

logger = logging.getLogger(__name__)

# Track if all patches have been applied
_all_patches_applied = False


def apply_all_patches(force: bool = False) -> None:
    """Apply all vLLM patches for suffix decoding support.

    This function applies patches in the correct dependency order:
    1. Config patches first (SpeculativeConfig)
    2. InputBatch patches (needed by runner)
    3. Runner patches (GPUModelRunner)
    4. Client patches (InprocClient)
    5. LLM patches (load_snapshot method)

    Args:
        force: If True, re-apply patches even if already applied.
               Default False to avoid duplicate patching.

    Note:
        Patches are idempotent - calling multiple times is safe.
        Each patch module tracks its own application state.
    """
    global _all_patches_applied

    if _all_patches_applied and not force:
        logger.debug("All vLLM patches already applied, skipping")
        return

    logger.info("Applying vLLM suffix decoding patches...")

    # 1. Config patches - must be first as other patches depend on config
    from verl.workers.rollout.vllm_rollout.patches import config_patches
    config_patches.apply_patches()

    # 2. InputBatch patches - needed by runner for prompt_hashes
    from verl.workers.rollout.vllm_rollout.patches import input_batch_patches
    input_batch_patches.apply_patches()

    # 3. Runner patches - depends on config and input_batch
    from verl.workers.rollout.vllm_rollout.patches import runner_patches
    runner_patches.apply_patches()

    # 4. Client patches - independent but applied after core patches
    from verl.workers.rollout.vllm_rollout.patches import client_patches
    client_patches.apply_patches()

    # 5. LLM patches - depends on runner having drafter initialized
    from verl.workers.rollout.vllm_rollout.patches import llm_patches
    llm_patches.apply_patches()

    _all_patches_applied = True
    logger.info("All vLLM suffix decoding patches applied successfully")


def is_patched() -> bool:
    """Check if patches have been applied.

    Returns:
        True if apply_all_patches() has been called successfully.
    """
    return _all_patches_applied


# Auto-apply patches if environment variable is set
if os.environ.get("VERL_VLLM_AUTO_PATCH", "").lower() in ("1", "true", "yes"):
    apply_all_patches()
