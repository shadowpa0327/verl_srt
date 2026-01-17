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
Shared Memory Suffix Decoding Proposer.

Uses SpecRL's SuffixCache for zero-copy shared memory access instead of
snapshot-based loading.

Unlike ParallelSuffixDecodingProposer:
- No load_snapshot() - cache is always fresh via shared memory
- Uses SuffixCache.speculate() instead of local suffix_cache
- Request lifecycle (fetch/evict) is handled by GPUModelRunner hooks,
  NOT inside propose(). This proposer only calls speculate().
- No serialization overhead

NOTE: Request lifecycle (fetch/evict) is handled by runner_patches.py hooks,
NOT inside propose(). This proposer only builds patterns and calls speculate().

Usage:
    # In runner_patches.py (shared_memory mode)
    from specrl.suffix_cache import SuffixCache

    # Config comes from SRTSuffixConfig (populated from speculative_config dict)
    suffix_cache = SuffixCache(
        shared_memory_name=srt_config.shared_memory_name,
        spec_start_len=srt_config.spec_start_len,
        spec_max_len=srt_config.spec_max_len,
    )

    self.drafter = SharedMemorySuffixDecodingProposer(
        num_speculative_tokens=24,
        max_model_len=8192,
        suffix_cache=suffix_cache,  # Inject cache
    )
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = logging.getLogger(__name__)


# Constants matching SpecRL's defaults
SPEC_PREFIX_LEN = 7
MIN_TOKEN_PROB = 0.1


class SharedMemorySuffixDecodingProposer:
    """
    Proposer using shared memory SuffixCache from SpecRL.

    NOTE: Request lifecycle (fetch/evict) is handled by GPUModelRunner hooks,
    NOT inside propose(). This proposer only calls speculate().

    This proposer provides the same interface as ParallelSuffixDecodingProposer
    but uses shared memory for cache access instead of snapshot loading.

    Key differences from ParallelSuffixDecodingProposer:
    - No load_snapshot() - cache is populated via gRPC by trainer
    - Uses SuffixCache.speculate() - reads from shared memory segment
    - fetch/evict lifecycle is handled by runner_patches execute_model hooks

    Attributes:
        num_speculative_tokens: Maximum number of draft tokens to propose.
        max_model_len: Maximum model context length.
    """

    def __init__(
        self,
        num_speculative_tokens: int = 24,
        max_model_len: int = 8192,
        spec_prefix_len: int = SPEC_PREFIX_LEN,
        min_token_prob: float = MIN_TOKEN_PROB,
        suffix_cache=None,  # Injected from runner_patches
    ):
        """
        Initialize SharedMemorySuffixDecodingProposer.

        Args:
            num_speculative_tokens: Maximum draft tokens to propose per step.
            max_model_len: Maximum model context length.
            spec_prefix_len: Number of recent tokens to use for pattern matching.
            min_token_prob: Minimum token probability for speculation.
            suffix_cache: Injected SuffixCache instance from runner_patches.
                         If None, creates a new instance (fallback).
        """
        self.num_speculative_tokens = num_speculative_tokens
        self.max_model_len = max_model_len
        self._spec_prefix_len = spec_prefix_len
        self._min_token_prob = min_token_prob

        # Cache is injected by runner_patches via lazy initialization.
        # When suffix_cache=None is passed, we defer initialization until
        # runner_patches injects the cache via self._cache = <SuffixCache instance>
        # This avoids the "No such file or directory" error when shared memory
        # hasn't been created yet.
        if suffix_cache is not None:
            self._cache = suffix_cache
            logger.info("SharedMemorySuffixDecodingProposer: Using injected SuffixCache")
        else:
            # Lazy initialization: cache will be injected later by runner_patches
            self._cache = None
            logger.info("SharedMemorySuffixDecodingProposer: Lazy init mode (cache will be injected)")

        # Statistics
        self._total_proposals = 0
        self._total_draft_tokens = 0

    def _initialize_cache(self):
        """Initialize the SuffixCache connection to shared memory (fallback).

        This is a fallback path used only if suffix_cache is not injected by
        runner_patches.py. Uses default parameters.

        NOTE: In normal operation, SuffixCache is created and injected by
        runner_patches.py with config from SRTSuffixConfig. This fallback
        uses defaults for standalone/testing use.
        """
        try:
            from specrl.suffix_cache import SuffixCache

            # Use defaults - in normal operation, cache is injected with proper config
            self._cache = SuffixCache()  # Uses C++ defaults: "", 2, 16
            logger.info(
                "SharedMemorySuffixDecodingProposer: Created new SuffixCache connection (defaults)"
            )
        except ImportError as e:
            logger.error(f"Failed to import SuffixCache from specrl: {e}")
            logger.error("Shared memory proposer requires the specrl package to be installed.")
            self._cache = None
        except Exception as e:
            logger.error(f"Failed to initialize SuffixCache: {e}")
            self._cache = None

    def propose(
        self,
        input_batch: "InputBatch",
        sampled_token_ids: List[List[int]],
    ) -> List[List[int]]:
        """
        Propose draft tokens using shared memory cache.

        NOTE: Fetching is done asynchronously by runner_patches hooks
        BEFORE this method is called. We just build patterns and speculate.

        This method:
        1. Builds patterns from recent tokens
        2. Calls speculate() to get draft tokens from shared memory
        3. Returns draft tokens for verification

        Args:
            input_batch: Current batch with token_ids_cpu, req_ids, etc.
            sampled_token_ids: Just-sampled tokens from the model.

        Returns:
            List of draft token lists, one per request in batch.
        """
        if self._cache is None:
            return [[] for _ in sampled_token_ids]

        req_ids_to_spec = []
        patterns_to_spec = []
        batch_indices = []

        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip requests with no sampled tokens (e.g., partial prefills)
                continue

            req_id = input_batch.req_ids[i]

            # Build pattern from recent tokens
            num_tokens = input_batch.num_tokens_no_spec[i]
            size = min(num_tokens, self._spec_prefix_len)
            pattern = input_batch.token_ids_cpu[i, num_tokens - size : num_tokens].tolist()

            req_ids_to_spec.append(req_id)
            patterns_to_spec.append(pattern)
            batch_indices.append(i)

        if not req_ids_to_spec:
            return [[] for _ in sampled_token_ids]

        # Get draft tokens via speculate()
        try:
            drafts = self._cache.speculate(
                req_ids_to_spec,
                patterns_to_spec,
                min_token_prob=self._min_token_prob,
            )
        except Exception as e:
            logger.warning(f"speculate() failed: {e}")
            return [[] for _ in sampled_token_ids]

        # Map draft tokens back to full batch
        result = [[] for _ in sampled_token_ids]
        for draft_idx, batch_idx in enumerate(batch_indices):
            if draft_idx < len(drafts):
                draft_tokens = drafts[draft_idx]
                # Limit to max speculative tokens
                draft_tokens = draft_tokens[:self.num_speculative_tokens]
                result[batch_idx] = draft_tokens
                self._total_draft_tokens += len(draft_tokens)

        self._total_proposals += 1
        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get proposer statistics.

        Returns:
            Dict with statistics about proposer usage.
        """
        return {
            "status": "success",
            "proposer_type": "shared_memory_suffix",
            "total_proposals": self._total_proposals,
            "total_draft_tokens": self._total_draft_tokens,
            "avg_draft_per_proposal": (
                self._total_draft_tokens / self._total_proposals
                if self._total_proposals > 0
                else 0.0
            ),
        }

    def load_model(self, model=None):
        """
        No-op for shared memory mode.

        vLLM's GPUModelRunner calls drafter.load_model(model) during initialization.
        SharedMemorySuffixDecodingProposer doesn't use a draft model - it uses
        shared memory cache for speculation instead.

        This method exists for interface compatibility.
        """
        logger.debug(
            "load_model() called on SharedMemorySuffixDecodingProposer - "
            "no-op (no draft model in shared memory mode)"
        )

    # For compatibility with snapshot-based interface
    def load_snapshot(
        self,
        snapshots: List[tuple],
        hash_mapping: Optional[Dict[str, int]] = None,
    ):
        """
        No-op for shared memory mode.

        In shared memory mode, the cache is populated via gRPC by the trainer,
        not via snapshot loading. This method exists for interface compatibility.
        """
        logger.debug(
            "load_snapshot() called on SharedMemorySuffixDecodingProposer - "
            "no-op in shared memory mode"
        )
