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

The key challenge is that vLLM spawns EngineCore subprocesses, and patches
applied in the main process don't propagate to subprocesses. Following the
specRL pattern, we patch WorkerBase.__init__ so that patches are applied
AFTER the process fork, ensuring they're available in all worker processes.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Track if plugin has been applied
_plugin_applied = False


def srt_plugin():
    """vLLM plugin for SRT suffix decoding.

    This plugin enables suffix decoding to be used with vLLM. It applies
    patches to vLLM at runtime.

    The plugin must be called before vLLM creates any workers. It patches
    WorkerBase.__init__ to apply config_patches and runner_patches when
    each worker is initialized (after forking).
    """
    global _plugin_applied

    if _plugin_applied:
        logger.debug("SRT plugin already applied, skipping")
        return

    if os.getenv("VLLM_USE_V1") == "0":
        logger.warning(
            "SRT suffix decoding only supports vLLM V1, but detected V0 engine. "
            "Ignoring plugin!\n"
            "Hint: To strictly enforce the V1 vLLM engine, please set "
            "VLLM_USE_V1=1."
        )
        return

    import vllm
    if not vllm.__version__.startswith("0.11"):
        logger.warning(
            f"SRT suffix decoding requires vllm==0.11.x but found "
            f"vllm=={vllm.__version__}. Plugin may not work correctly."
        )

    # Import WorkerBase lazily to avoid CUDA initialization
    from vllm.v1.worker.worker_base import WorkerBase

    # Store original __init__
    _original_worker_init = WorkerBase.__init__

    def _patched_worker_init(self, *args, **kwargs):
        """Patched WorkerBase.__init__ that applies suffix decoding patches.

        This is called AFTER the process fork, in each worker subprocess.
        By applying patches here, we ensure they're available in all workers.
        """
        # Apply config patches first (to register "suffix" method)
        from recipe.srt.vllm_plugin.patches import config_patches
        config_patches.apply_patches()

        # Apply runner patches (for suffix proposer support)
        from recipe.srt.vllm_plugin.patches import runner_patches
        runner_patches.apply_patches()

        # Apply input_batch patches (for prompt_hashes)
        from recipe.srt.vllm_plugin.patches import input_batch_patches
        input_batch_patches.apply_patches()

        logger.debug("Applied SRT suffix decoding patches in worker subprocess")

        # Call original __init__
        return _original_worker_init(self, *args, **kwargs)

    # Patch WorkerBase.__init__
    WorkerBase.__init__ = _patched_worker_init

    _plugin_applied = True
    logger.info("Applied SRT vLLM plugin (WorkerBase patch installed)")
