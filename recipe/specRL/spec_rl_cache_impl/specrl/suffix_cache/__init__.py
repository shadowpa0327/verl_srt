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
suffix_cache - Suffix Tree Cache Module

This module provides classes for suffix tree based cache management
and speculative decoding support.

Classes:
    SuffixCache: Main cache class for storing and querying responses
    SuffixSpecResult: Result class for speculative decoding
    RolloutCacheServer: gRPC server for cache operations

Example:
    >>> from suffix_cache import SuffixCache, SuffixSpecResult, RolloutCacheServer
    >>> cache = SuffixCache()
    >>> result = cache.speculate(req_id, pattern)
"""

from ._C import SuffixCache, SuffixSpecResult, RolloutCacheServer

__all__ = ["SuffixCache", "SuffixSpecResult", "RolloutCacheServer"]
__version__ = "0.1.0"
