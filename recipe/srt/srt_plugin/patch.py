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
SRT vLLM plugin for suffix decoding.

This plugin enables suffix decoding to be used with vLLM. It consists of
patches that are applied at runtime to support the "suffix" speculative
decoding method.

USAGE:
    Set environment variable VERL_SRT_ENABLED=1 to enable suffix decoding.
    Without this, the plugin does nothing (safe for normal vLLM usage).

ARCHITECTURE:
    The plugin is registered as a vLLM general plugin via entry_points.
    When vLLM calls load_general_plugins(), this plugin:

    1. Checks VERL_SRT_ENABLED - if not "1", returns immediately (no-op)
    2. Applies config_patches in main process (SpeculativeConfig)
    3. Applies arg_utils_patches in main process (CLI arguments)
    4. Patches WorkerBase.__init__ to apply remaining patches in subprocesses:
       - config_patches (re-apply for subprocess)
       - runner_patches (GPUModelRunner for suffix proposer)
       - input_batch_patches (prompt_hashes support)

    This ensures all patches are applied in ALL processes (main, EngineCore, Workers).
"""

import logging
import os

logger = logging.getLogger(__name__)

# Environment variable to control SRT plugin activation
SRT_ENABLED_ENV = "VERL_SRT_ENABLED"

# Track if plugin has been applied
_plugin_applied = False


def _apply_main_process_patches():
    """Apply patches needed in the main process.

    These patches are applied before vLLM creates any engines/workers.
    They configure SpeculativeConfig and CLI arguments.
    """
    # Config patches: Add "suffix"/"suffix_remote" methods to SpeculativeConfig
    from recipe.srt.srt_plugin.patches import config_patches
    config_patches.apply_patches()

    # Arg utils patches: Add suffix decoding CLI arguments
    from recipe.srt.srt_plugin.patches import arg_utils_patches
    arg_utils_patches.apply_patches()


def _apply_worker_process_patches():
    """Apply patches needed in worker subprocesses.

    These patches are applied after fork, in each worker subprocess.
    They configure GPUModelRunner and InputBatch for suffix decoding.
    """
    # Config patches: Re-apply in subprocess (doesn't propagate from main)
    from recipe.srt.srt_plugin.patches import config_patches
    config_patches.apply_patches()

    # Runner patches: GPUModelRunner suffix proposer integration
    from recipe.srt.srt_plugin.patches import runner_patches
    runner_patches.apply_patches()

    # Input batch patches: prompt_hashes support
    from recipe.srt.srt_plugin.patches import input_batch_patches
    input_batch_patches.apply_patches()


def srt_plugin():
    """vLLM plugin entry point for SRT suffix decoding.

    This function is called by vLLM's load_general_plugins() in ALL processes:
    - Main process (before engine creation)
    - EngineCore subprocess
    - Worker subprocesses

    ACTIVATION:
        Set VERL_SRT_ENABLED=1 to enable. Without this, the plugin does nothing.

    CONFLICT AVOIDANCE:
        If ARCTIC_INFERENCE_ENABLED=1, this plugin skips to avoid conflicts
        with arctic_inference's own patches.
    """
    global _plugin_applied

    # === GUARD: Check if SRT is enabled ===
    if os.getenv(SRT_ENABLED_ENV) != "1":
        # SRT not enabled - do nothing
        # This makes the plugin safe for normal vLLM usage
        return

    # === GUARD: Already applied ===
    if _plugin_applied:
        logger.debug("SRT plugin already applied, skipping")
        return

    # === GUARD: Conflict with arctic_inference ===
    if os.getenv("ARCTIC_INFERENCE_ENABLED") == "1":
        logger.warning(
            "ARCTIC_INFERENCE_ENABLED=1 detected. Skipping SRT plugin to avoid "
            "conflicts with arctic_inference patches. If you need SRT suffix "
            "decoding, please disable ARCTIC_INFERENCE_ENABLED."
        )
        return

    # === GUARD: vLLM V1 only ===
    if os.getenv("VLLM_USE_V1") == "0":
        logger.warning(
            "SRT suffix decoding only supports vLLM V1, but VLLM_USE_V1=0. "
            "Ignoring plugin! Set VLLM_USE_V1=1 to enable."
        )
        return

    # === GUARD: Version check ===
    import vllm
    if not vllm.__version__.startswith("0.11"):
        logger.warning(
            f"SRT suffix decoding requires vllm==0.11.x but found "
            f"vllm=={vllm.__version__}. Plugin may not work correctly."
        )

    # === APPLY MAIN PROCESS PATCHES ===
    # These are needed in main process and EngineCore subprocess
    _apply_main_process_patches()

    # === PATCH WorkerBase FOR SUBPROCESS PATCHES ===
    # Worker subprocesses need additional patches (GPUModelRunner, InputBatch)
    # We patch WorkerBase.__init__ to apply them after fork
    from vllm.v1.worker.worker_base import WorkerBase

    _original_worker_init = WorkerBase.__init__

    def _patched_worker_init(self, *args, **kwargs):
        """Patched WorkerBase.__init__ that applies suffix decoding patches.

        Called AFTER fork in each worker subprocess.
        """
        _apply_worker_process_patches()
        logger.debug("Applied SRT suffix decoding patches in worker subprocess")
        return _original_worker_init(self, *args, **kwargs)

    WorkerBase.__init__ = _patched_worker_init

    _plugin_applied = True
    logger.info(
        "SRT vLLM plugin enabled (VERL_SRT_ENABLED=1). "
        "Suffix decoding patches applied."
    )
