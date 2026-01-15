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
SRT plugin for vLLM - suffix decoding integration.

This plugin provides:
- suffix_cache: Suffix tree data structures for speculative decoding
- patches: Runtime patches for vLLM to support suffix methods
- proposers: Suffix decoding proposer implementations
- patching: ArcticPatch framework for clean monkey-patching

Usage:
    # Install the plugin (registers automatically via entry_points)
    pip install -e recipe/srt/srt_plugin

    # Plugin is enabled by default. To disable:
    VERL_SRT_DISABLED=1 python your_script.py
"""

from recipe.srt.srt_plugin.patch import srt_plugin

__all__ = ["srt_plugin"]
