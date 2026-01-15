# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Originally from ArcticInference (Snowflake), vendored for SRT plugin.
"""
Suffix decoding cache for speculative decoding.

This module provides suffix tree-based caching for speculative decoding,
enabling efficient draft token generation without a separate draft model.
"""

from .cache import SuffixDecodingCache, SuffixDecodingDraft
from .hash_utils import compute_prompt_hash

# Try to import ParallelSuffixDecodingCache - requires SuffixForest C++ extension
try:
    from .parallel_cache import ParallelSuffixDecodingCache
    _has_parallel = True
except ImportError:
    # SuffixForest not available (C++ extension not built)
    _has_parallel = False

# Try to import SuffixTree directly for snapshot utilities
try:
    from ._C import SuffixTree as _SuffixTree

    def save_suffix_tree(tree: "_SuffixTree", path: str) -> None:
        """Save a SuffixTree to a file.

        Args:
            tree: The SuffixTree instance to save.
            path: File path to save the snapshot to.
        """
        snapshot = tree.create_snapshot()
        with open(path, "wb") as f:
            f.write(snapshot)

    def load_suffix_tree(path: str) -> "_SuffixTree":
        """Load a SuffixTree from a file.

        Args:
            path: File path to load the snapshot from.

        Returns:
            A restored SuffixTree instance.
        """
        with open(path, "rb") as f:
            snapshot = f.read()
        return _SuffixTree.restore_snapshot(snapshot)

    _has_snapshot = True
except ImportError:
    _has_snapshot = False

# Build __all__ based on available features
__all__ = ["SuffixDecodingCache", "SuffixDecodingDraft", "compute_prompt_hash"]
if _has_parallel:
    __all__.append("ParallelSuffixDecodingCache")
if _has_snapshot:
    __all__.extend(["save_suffix_tree", "load_suffix_tree"])
