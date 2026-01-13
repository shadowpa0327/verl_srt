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
SRT (Speculative Rollout with Tree-Structured Cache) Recipe.

This recipe accelerates RL rollout by using suffix trees to cache and reuse
token sequences from historical responses as draft sequences for speculative decoding.

Usage:
    Replace `verl.trainer.main_ppo` with `recipe.srt.main_ppo` in training scripts.
"""

from recipe.srt.suffix_tree_manager import SuffixTreeManager, SuffixTreeManagerConfig

__all__ = ["SuffixTreeManager", "SuffixTreeManagerConfig"]
