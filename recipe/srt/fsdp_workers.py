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
SRT FSDP Workers with vLLM suffix decoding plugin support.

These workers extend the base workers to apply vLLM patches for suffix decoding
support. The patches are applied in worker __init__, which ensures they're
installed before vLLM creates any subprocesses.
"""

from omegaconf import DictConfig

from recipe.srt.srt_plugin.patch import srt_plugin
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


class _SRTPluginMixin:
    """Mixin that applies SRT vLLM plugin when rollout is enabled."""

    def _apply_srt_plugin(self):
        """Apply SRT vLLM patches if this worker handles rollout."""
        if getattr(self, '_is_rollout', False):
            print("SRT: Applying vLLM suffix decoding patches on this node...")
            srt_plugin()
            print("SRT: vLLM patches applied successfully")


class SRTActorRolloutRefWorker(_SRTPluginMixin, ActorRolloutRefWorker):
    """ActorRolloutRefWorker with SRT vLLM suffix decoding patch."""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self._apply_srt_plugin()


class SRTAsyncActorRolloutRefWorker(_SRTPluginMixin, AsyncActorRolloutRefWorker):
    """AsyncActorRolloutRefWorker with SRT vLLM suffix decoding patch."""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        self._apply_srt_plugin()
