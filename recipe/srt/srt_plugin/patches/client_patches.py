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
InprocClient patches for proper output handling.

Fixes get_output() to properly call post_step after step_fn.
"""

import logging

from ..patching import ArcticPatch
from vllm.v1.engine.core_client import InprocClient

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


class InprocClientPatch(ArcticPatch[InprocClient]):
    """
    Patches InprocClient to fix get_output() method.

    Original issue: post_step was not being called after step_fn,
    which could cause issues with certain engine states.
    """

    def get_output(self):
        """Fixed get_output with proper post_step call."""
        from vllm.v1.engine import EngineCoreOutputs

        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed)

        return outputs and outputs.get(0) or EngineCoreOutputs()


def apply_patches():
    """Apply InprocClient patches."""
    global _patches_applied

    if _patches_applied:
        logger.debug("InprocClient patches already applied, skipping")
        return

    # Use ArcticPatch's apply_patch method
    InprocClientPatch.apply_patch()

    _patches_applied = True
    logger.info("Applied InprocClient get_output patch")
