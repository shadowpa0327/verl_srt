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
InputBatch patches for prompt hash tracking.

Adds prompt_hashes field for hash-based tree mapping in suffix decoding.
"""

import logging

from ..patching import ArcticPatch
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False

# Store original methods for wrapping
_original_init = InputBatch.__init__
_original_add_request = InputBatch.add_request
_original_remove_request = InputBatch.remove_request


class InputBatchHashPatch(ArcticPatch[InputBatch]):
    """
    Patches InputBatch to add prompt_hashes field for suffix decoding.

    - Adds prompt_hashes dict to track pre-computed hashes per request
    - Extends add_request to extract prompt_hash from sampling params
    - Extends remove_request to clean up prompt_hash
    """

    def __init__(self, *args, **kwargs):
        """Extended __init__ with prompt_hashes field."""
        # Call original __init__
        _original_init(self, *args, **kwargs)

        # Add prompt_hashes field
        # Pre-computed prompt hashes for suffix tree lookup (req_id -> hash)
        # Used by speculative decoding to ensure hash consistency between
        # tree building (trainer) and tree lookup (inference).
        self.prompt_hashes: dict[str, str] = {}

    def add_request(self, request, *args, **kwargs):
        """Extended add_request that extracts prompt_hash from sampling params."""
        # Call original add_request
        result = _original_add_request(self, request, *args, **kwargs)

        # Extract prompt hash from sampling params if available
        # Note: CachedRequestState uses 'req_id', not 'request_id'
        req_id = request.req_id
        if hasattr(request, 'sampling_params') and request.sampling_params:
            extra_args = getattr(request.sampling_params, 'extra_args', None)
            if extra_args:
                prompt_hash = extra_args.get("prompt_hash")
                if prompt_hash:
                    self.prompt_hashes[req_id] = prompt_hash

        return result

    def remove_request(self, req_id: str, *args, **kwargs):
        """Extended remove_request that cleans up prompt_hash."""
        # Call original remove_request
        result = _original_remove_request(self, req_id, *args, **kwargs)

        # Clean up prompt_hashes
        self.prompt_hashes.pop(req_id, None)

        return result


def apply_patches():
    """Apply InputBatch patches."""
    global _patches_applied

    if _patches_applied:
        logger.debug("InputBatch patches already applied, skipping")
        return

    # Use ArcticPatch's apply_patch method
    InputBatchHashPatch.apply_patch()

    _patches_applied = True
    logger.info("Applied InputBatch prompt_hashes patch")
