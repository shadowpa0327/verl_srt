# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
shm_cache - Shared Memory Cache for Speculative Decoding RL

This package provides shared memory cache management tools for speculative decoding:
- cache_updater: For updating the rollout cache via gRPC
- suffix_cache: For suffix tree based cache management with shared memory support

Usage:
    from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater
    from srt_plugin.shm_cache.suffix_cache import SuffixCache, SuffixSpecResult, RolloutCacheServer
"""

__version__ = "0.1.0"

# Do NOT automatically import submodules to avoid protobuf registration conflicts
# Users should explicitly import what they need:
#   from srt_plugin.shm_cache.cache_updater import SuffixCacheUpdater
#   from srt_plugin.shm_cache.suffix_cache import SuffixCache, SuffixSpecResult, RolloutCacheServer
