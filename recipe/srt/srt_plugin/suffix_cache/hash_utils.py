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
Hash utilities for suffix decoding - pure Python, no C++ dependencies.
"""

import hashlib
from typing import Sequence, Union

import numpy as np


def compute_prompt_hash(
    prompt_tokens: Union[np.ndarray, Sequence[int]],
    hash_token_count: int = 128,
) -> str:
    """
    Compute hash for prompt tokens - canonical implementation.

    This is the single source of truth for prompt hash computation. Use this
    function everywhere to ensure hash consistency between:
    - Tree building (trainer/SuffixTreeManager)
    - Tree lookup (vLLM/ArcticInference)
    - Hash passing (rollout worker -> vLLM via SamplingParams)

    Args:
        prompt_tokens: Full prompt token sequence (list or numpy array).
        hash_token_count: Number of trailing tokens to hash (default: 128).
            Using trailing tokens focuses on the question/task rather than
            system prompts that may be shared across requests.

    Returns:
        16-character hex string (64 bits of SHA-256).

    Example:
        >>> from recipe.srt.srt_plugin.suffix_cache import compute_prompt_hash
        >>> tokens = [1, 2, 3, 4, 5]
        >>> hash_value = compute_prompt_hash(tokens)
        >>> print(hash_value)  # e.g., "a1b2c3d4e5f6g7h8"
    """
    # Use last N tokens to focus on the question, not system prompt
    if len(prompt_tokens) > hash_token_count:
        tokens_to_hash = prompt_tokens[-hash_token_count:]
    else:
        tokens_to_hash = prompt_tokens

    # Convert to bytes for hashing
    if isinstance(tokens_to_hash, np.ndarray):
        token_bytes = tokens_to_hash.astype(np.int32).tobytes()
    else:
        token_bytes = np.array(list(tokens_to_hash), dtype=np.int32).tobytes()

    return hashlib.sha256(token_bytes).hexdigest()[:16]
