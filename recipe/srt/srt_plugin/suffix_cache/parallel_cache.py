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

from __future__ import annotations

import hashlib
from typing import Dict, Hashable, KeysView, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._C import SuffixTree, SuffixForest, Draft
from .cache import SuffixDecodingDraft
from .hash_utils import compute_prompt_hash


class ParallelSuffixDecodingCache:
    """
    Parallel implementation of SuffixDecodingCache using SuffixForest for
    batched parallel speculation across multiple requests.

    This class is designed for high-throughput scenarios where multiple requests
    are processed simultaneously. It uses C++ SuffixForest with OpenMP for
    efficient parallel speculation.

    Key differences from SuffixDecodingCache:
    - No global tree (removed as per design plan)
    - Batch speculation with parallel execution
    - Configurable parallelization parameters
    - Simpler API focused on batched operations
    """

    def __init__(self,
                 max_tree_depth: int = 64,
                 num_threads: int = -1,
                 parallel_threshold: int = 4,
                 hash_token_count: int = 128):
        """
        Initialize the ParallelSuffixDecodingCache.

        Args:
            max_tree_depth (int): The maximum depth of the suffix trees.
                Determines how much context history is retained.
            num_threads (int): Number of threads for parallel speculation.
                -1 = auto-detect from CPU count (default)
                0 = sequential execution (no parallelization)
                >0 = use specified number of threads
            parallel_threshold (int): Minimum batch size to trigger parallelization.
                Batches smaller than this will run sequentially to avoid overhead.
                Default: 4.
            hash_token_count (int): Number of trailing prompt tokens to hash for
                tree lookup. Multiple requests with the same prompt hash share a tree.
                Set to 0 to disable hash-based sharing (one tree per request).
                Default: 128.
        """
        self._max_tree_depth = max_tree_depth
        self._num_threads = num_threads
        self._parallel_threshold = parallel_threshold
        self._hash_token_count = hash_token_count

        # C++ SuffixForest for parallel batched speculation
        self._forest = SuffixForest(max_tree_depth, num_threads, parallel_threshold)

        # Maps request ID to tree index in the forest
        self._req_to_tree_idx: Dict[Hashable, int] = {}

        # Hash-based tree lookup: prompt_hash → tree_idx
        self._hash_to_tree_idx: Dict[str, int] = {}

        # Per-request seq_id tracking (prevents token interleaving in shared trees)
        self._req_to_seq_id: Dict[Hashable, int] = {}

        # Next available seq_id per tree
        self._tree_next_seq_id: Dict[int, int] = {}

        # Protected tree indices (loaded from snapshot, not GC'd)
        self._protected_tree_indices: set = set()

    @property
    def max_tree_depth(self) -> int:
        """Maximum depth of suffix trees."""
        return self._max_tree_depth

    @property
    def num_threads(self) -> int:
        """Number of threads configured for parallel execution."""
        return self._forest.get_num_threads()

    @property
    def parallel_threshold(self) -> int:
        """Minimum batch size to trigger parallelization."""
        return self._forest.get_parallel_threshold()

    @property
    def active_requests(self) -> KeysView:
        """
        Returns a view of the currently active request IDs.

        Active requests are those that have been started via `start_request`
        and not yet stopped via `stop_request`.
        """
        return self._req_to_tree_idx.keys()

    @property
    def num_active_requests(self) -> int:
        """Number of currently active requests."""
        return len(self._req_to_tree_idx)

    @property
    def hash_token_count(self) -> int:
        """Number of trailing tokens used for prompt hashing."""
        return self._hash_token_count

    def _hash_prompt(
        self,
        prompt_tokens: Union[np.ndarray, Sequence[int]],
        hash_token_count: Optional[int] = None,
    ) -> str:
        """
        Hash the last N tokens of the prompt.

        Args:
            prompt_tokens: Full prompt token sequence.
            hash_token_count: Number of trailing tokens to hash.
                             If None, uses the default from __init__.

        Returns:
            16-character hex string (64 bits of SHA-256).
        """
        if hash_token_count is None:
            hash_token_count = self._hash_token_count

        # Delegate to the canonical standalone function
        return compute_prompt_hash(prompt_tokens, hash_token_count)

    def _extend_tree(
        self,
        tree_idx: int,
        seq_id: int,
        token_ids: Union[np.ndarray, Sequence[int]],
    ) -> None:
        """
        Thread-safe tree extension using C++ locking.

        Args:
            tree_idx: Index of the tree in the forest.
            seq_id: Sequence ID within the tree.
            token_ids: Token IDs to extend the tree with.
        """
        if isinstance(token_ids, np.ndarray):
            self._validate_ndarray(token_ids)
            self._forest.extend_tree_ndarray(tree_idx, seq_id, token_ids)
        else:
            tokens = list(token_ids) if not isinstance(token_ids, list) else token_ids
            self._forest.extend_tree(tree_idx, seq_id, tokens)

    def start_request(
        self,
        req_id: Hashable,
        prompt_token_ids: np.ndarray | Sequence[int],
        hash_token_count: Optional[int] = None,
        pre_computed_hash: Optional[str] = None,
    ):
        """
        Start processing a new request by creating or reusing a suffix tree
        and inserting the prompt tokens.

        If hash-based sharing is enabled (hash_token_count > 0), requests with
        the same prompt hash will share a suffix tree. Each request gets a unique
        seq_id within the shared tree to prevent token interleaving.

        Args:
            req_id (Hashable): The request identifier. Must be a hashable value
                that uniquely identifies the request.
            prompt_token_ids (np.ndarray | Sequence[int]): A sequence of token
                IDs representing the prompt of the request.
            hash_token_count (int, optional): Number of trailing prompt tokens
                to hash for tree lookup. If None, uses the default from __init__.
                Set to 0 to disable hash-based sharing for this request.
            pre_computed_hash (str, optional): Pre-computed prompt hash. If provided,
                this hash is used directly for tree lookup instead of computing it
                from prompt_token_ids. This ensures hash consistency when the same
                hash is computed elsewhere (e.g., in the rollout worker).

        Raises:
            ValueError: If a request with the same `req_id` is already active.
        """
        if req_id in self._req_to_tree_idx:
            raise ValueError(f"Request '{req_id}' is already active")

        # Determine if hash lookup is enabled
        effective_hash_count = hash_token_count if hash_token_count is not None else self._hash_token_count
        use_hash_lookup = effective_hash_count > 0 or pre_computed_hash is not None

        if use_hash_lookup:
            # Use pre-computed hash if provided, otherwise compute from tokens
            if pre_computed_hash is not None:
                prompt_hash = pre_computed_hash
            else:
                prompt_hash = self._hash_prompt(prompt_token_ids, effective_hash_count)
            if prompt_hash in self._hash_to_tree_idx:
                # Reuse existing tree
                tree_idx = self._hash_to_tree_idx[prompt_hash]
                seq_id = self._tree_next_seq_id.get(tree_idx, 0)
                self._tree_next_seq_id[tree_idx] = seq_id + 1
            else:
                # Create new tree and register hash
                tree_idx = self._forest.create_tree()
                self._hash_to_tree_idx[prompt_hash] = tree_idx
                seq_id = 0
                self._tree_next_seq_id[tree_idx] = 1
                # Initialize tree with prompt tokens
                # self._extend_tree(tree_idx, seq_id, prompt_token_ids)
        else:
            # Original behavior: one tree per request
            tree_idx = self._forest.create_tree()
            seq_id = 0
            self._tree_next_seq_id[tree_idx] = 1
            # Initialize tree with prompt tokens
            self._extend_tree(tree_idx, seq_id, prompt_token_ids)

        self._req_to_tree_idx[req_id] = tree_idx
        self._req_to_seq_id[req_id] = seq_id

    def stop_request(self, req_id: Hashable):
        """
        Stop processing a request and free its suffix tree if no longer used.

        If the tree is shared by other requests (via hash-based lookup), the
        tree is only removed when the last request using it stops.

        Args:
            req_id (Hashable): The request identifier.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._req_to_tree_idx:
            raise ValueError(f"Request '{req_id}' is not active")

        tree_idx = self._req_to_tree_idx.pop(req_id)
        self._req_to_seq_id.pop(req_id, None)
        # Only remove tree if:
        # 1. No other requests reference it, AND
        # 2. It's not a protected tree (loaded from snapshot)
        if tree_idx not in self._req_to_tree_idx.values():
            if tree_idx not in self._protected_tree_indices:
                self._forest.remove_tree(tree_idx)
                # Remove from hash mapping
                for hash_key, idx in list(self._hash_to_tree_idx.items()):
                    if idx == tree_idx:
                        del self._hash_to_tree_idx[hash_key]
                        break
                self._tree_next_seq_id.pop(tree_idx, None)

    def add_tokens(
        self,
        req_id: Hashable,
        token_ids: np.ndarray | Sequence[int],
    ):
        """
        Add generated tokens to a request's suffix tree.

        This updates the tree with newly generated tokens, allowing them to be
        used for future speculations. Uses thread-safe extension via C++ locking.

        Args:
            req_id (Hashable): The unique identifier for the request.
            token_ids (np.ndarray | Sequence[int]): A sequence of token IDs to
                be appended to the tree.

        Raises:
            ValueError: If the request with the given `req_id` is not active.
        """
        if req_id not in self._req_to_tree_idx:
            raise ValueError(f"Request '{req_id}' is not active")

        tree_idx = self._req_to_tree_idx[req_id]
        seq_id = self._req_to_seq_id.get(req_id, 0)  # Use per-request seq_id

        # Use thread-safe extension
        self._extend_tree(tree_idx, seq_id, token_ids)

    def batch_add_tokens(
        self,
        req_ids: List[Hashable],
        token_batches: List[np.ndarray | Sequence[int]],
    ):
        """
        Add generated tokens to multiple requests in parallel.

        This is more efficient than calling add_tokens() in a loop when you
        have multiple requests with new tokens. Perfect for batch inference
        where all requests get new tokens simultaneously.

        Uses thread-safe C++ batch extend with per-tree locking.

        Args:
            req_ids (List[Hashable]): List of request identifiers.
            token_batches (List[np.ndarray | Sequence[int]]): List of token
                sequences, one per request.

        Raises:
            ValueError: If req_ids and token_batches have different lengths,
                or if any request is not active.
            RuntimeError: If adding tokens fails for any tree.
        """
        if len(req_ids) != len(token_batches):
            raise ValueError(
                f"req_ids and token_batches must have same length. "
                f"Got {len(req_ids)} and {len(token_batches)}"
            )

        if len(req_ids) == 0:
            return

        # Validate all requests are active and get tree indices and seq_ids
        tree_indices = []
        seq_ids = []
        for req_id in req_ids:
            if req_id not in self._req_to_tree_idx:
                raise ValueError(f"Request '{req_id}' is not active")
            tree_indices.append(self._req_to_tree_idx[req_id])
            seq_ids.append(self._req_to_seq_id.get(req_id, 0))  # Per-request seq_id

        # Prepare token batches for C++
        tokens_for_cpp = []
        use_ndarray = False

        for tokens in token_batches:
            if isinstance(tokens, np.ndarray):
                self._validate_ndarray(tokens)
                tokens_for_cpp.append(tokens)
                use_ndarray = True
            else:
                # Convert to list for C++
                tokens_for_cpp.append(
                    list(tokens) if not isinstance(tokens, list) else tokens
                )

        # Call C++ batch extend (releases GIL internally, uses per-tree locking)
        if use_ndarray:
            # Use ndarray-optimized version
            tree_indices_np = np.array(tree_indices, dtype=np.int32)
            seq_ids_np = np.array(seq_ids, dtype=np.int32)
            self._forest.batch_extend_ndarray(
                tree_indices_np,
                seq_ids_np,
                tokens_for_cpp
            )
        else:
            # Use standard vector version
            self._forest.batch_extend(
                tree_indices,
                seq_ids,
                tokens_for_cpp
            )

    def batch_speculate(
        self,
        req_ids: List[Hashable],
        contexts: List[np.ndarray | Sequence[int]],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> List[SuffixDecodingDraft]:
        """
        Perform batched parallel speculation across multiple requests.

        This is the main method for speculation. It processes all requests in
        parallel (if batch size ≥ parallel_threshold) using OpenMP.

        Args:
            req_ids (List[Hashable]): List of request identifiers.
            contexts (List[np.ndarray | Sequence[int]]): List of context sequences,
                one per request. Each context is the most recent tokens to match
                against the suffix tree.
            max_spec_tokens (int, optional): Maximum number of tokens to speculate
                per request. If None, uses max_tree_depth.
            max_spec_factor (float): Factor that limits speculation based on
                matched context length. The number of speculated tokens is
                limited by `max_spec_factor * match_length + max_spec_offset`.
            max_spec_offset (float): Offset for speculation limit.
            min_token_prob (float): Minimum estimated probability threshold for
                draft tokens. Tokens with lower probability are not included.
            use_tree_spec (bool): If True, uses tree-based speculation (explores
                multiple paths). If False, uses path-based speculation (greedy).

        Returns:
            List of SuffixDecodingDraft objects, one per request.

        Raises:
            ValueError: If req_ids and contexts have different lengths, or if
                any request is not active.
            RuntimeError: If speculation fails for any tree in the batch.
        """
        if len(req_ids) != len(contexts):
            raise ValueError(
                f"req_ids and contexts must have same length. "
                f"Got {len(req_ids)} and {len(contexts)}"
            )

        if max_spec_tokens is None:
            max_spec_tokens = self._max_tree_depth

        # Validate all requests are active and get tree indices
        tree_indices = []
        for req_id in req_ids:
            if req_id not in self._req_to_tree_idx:
                raise ValueError(f"Request '{req_id}' is not active")
            tree_indices.append(self._req_to_tree_idx[req_id])

        # Truncate contexts to max_tree_depth and prepare for C++
        truncated_contexts = []
        use_ndarray = False

        for ctx in contexts:
            # Truncate if necessary
            if len(ctx) > self._max_tree_depth:
                ctx = ctx[-self._max_tree_depth:]

            # Validate and add to list
            if isinstance(ctx, np.ndarray):
                self._validate_ndarray(ctx)
                truncated_contexts.append(ctx)
                use_ndarray = True
            else:
                # Convert to list for C++
                truncated_contexts.append(
                    list(ctx) if not isinstance(ctx, list) else ctx
                )

        # Call C++ batch speculation (releases GIL internally)
        if use_ndarray:
            # Use ndarray-optimized version
            tree_indices_np = np.array(tree_indices, dtype=np.int32)
            drafts = self._forest.batch_speculate_ndarray(
                tree_indices_np,
                truncated_contexts,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec
            )
        else:
            # Use standard vector version
            drafts = self._forest.batch_speculate(
                tree_indices,
                truncated_contexts,
                max_spec_tokens,
                max_spec_factor,
                max_spec_offset,
                min_token_prob,
                use_tree_spec
            )

        # Convert C++ Draft objects to Python SuffixDecodingDraft objects
        return [SuffixDecodingDraft.from_native(d) for d in drafts]

    def propose_from_batch(
        self,
        token_ids_cpu: np.ndarray,
        num_tokens: np.ndarray,
        tree_indices: np.ndarray,
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> List[SuffixDecodingDraft]:
        """
        Optimized batch speculation that directly consumes InputBatch-like numpy arrays.

        This method eliminates Python loop overhead by delegating context extraction
        and speculation to C++. Expected speedup: ~3-3.5x for batch_size >= 32.

        Args:
            token_ids_cpu (np.ndarray): 2D array [batch_size, max_seq_len] of token IDs.
                Must be int32, C-contiguous.
            num_tokens (np.ndarray): 1D array [batch_size] of valid token counts per request.
                Must be int32.
            tree_indices (np.ndarray): 1D array [batch_size] of tree indices for each request.
                Must be int32.
            max_spec_tokens (int, optional): Maximum tokens to speculate per request.
                If None, uses max_tree_depth.
            max_spec_factor (float): Speculation length factor.
            max_spec_offset (float): Speculation length offset.
            min_token_prob (float): Minimum probability threshold.
            use_tree_spec (bool): Use tree-based (vs path-based) speculation.

        Returns:
            List of SuffixDecodingDraft objects, one per batch item.

        Raises:
            ValueError: If arrays have inconsistent shapes or wrong dtypes.
            RuntimeError: If speculation fails for any tree in the batch.
        """
        # Validate inputs
        if token_ids_cpu.ndim != 2:
            raise ValueError(f"token_ids_cpu must be 2D, got shape {token_ids_cpu.shape}")
        if token_ids_cpu.dtype != np.int32:
            raise ValueError(f"token_ids_cpu must be int32, got {token_ids_cpu.dtype}")
        if not token_ids_cpu.flags['C_CONTIGUOUS']:
            raise ValueError("token_ids_cpu must be C-contiguous")

        batch_size, max_seq_len = token_ids_cpu.shape

        if num_tokens.shape != (batch_size,):
            raise ValueError(
                f"num_tokens shape must be ({batch_size},), got {num_tokens.shape}"
            )
        if num_tokens.dtype != np.int32:
            raise ValueError(f"num_tokens must be int32, got {num_tokens.dtype}")

        if tree_indices.shape != (batch_size,):
            raise ValueError(
                f"tree_indices shape must be ({batch_size},), got {tree_indices.shape}"
            )
        if tree_indices.dtype != np.int32:
            raise ValueError(f"tree_indices must be int32, got {tree_indices.dtype}")

        if max_spec_tokens is None:
            max_spec_tokens = self._max_tree_depth

        # Call C++ method (releases GIL internally)
        drafts = self._forest.propose_from_batch(
            token_ids_cpu,
            num_tokens,
            tree_indices,
            self._max_tree_depth,
            max_spec_tokens,
            max_spec_factor,
            max_spec_offset,
            min_token_prob,
            use_tree_spec
        )

        # Convert C++ Draft objects to Python SuffixDecodingDraft objects
        return [SuffixDecodingDraft.from_native(d) for d in drafts]

    def speculate(
        self,
        req_id: Hashable,
        context: np.ndarray | Sequence[int],
        max_spec_tokens: Optional[int] = None,
        max_spec_factor: float = 1.0,
        max_spec_offset: float = 0.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
    ) -> SuffixDecodingDraft:
        """
        Speculate for a single request.

        This is a convenience method that delegates to batch_speculate with a
        single request. For better performance with multiple requests, use
        batch_speculate directly.

        Args:
            req_id (Hashable): The unique identifier for the request.
            context (np.ndarray | Sequence[int]): Context sequence to match.
            max_spec_tokens (int, optional): Maximum tokens to speculate.
            max_spec_factor (float): Speculation length factor.
            max_spec_offset (float): Speculation length offset.
            min_token_prob (float): Minimum probability threshold.
            use_tree_spec (bool): Use tree-based (vs path-based) speculation.

        Returns:
            SuffixDecodingDraft: The speculation result.

        Raises:
            ValueError: If the request is not active.
        """
        results = self.batch_speculate(
            req_ids=[req_id],
            contexts=[context],
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            max_spec_offset=max_spec_offset,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec
        )
        return results[0]

    def get_stats(self) -> dict:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics including number of active requests,
            configured threads, and parallel threshold.
        """
        return {
            "num_active_requests": self.num_active_requests,
            "max_tree_depth": self._max_tree_depth,
            "num_threads": self.num_threads,
            "parallel_threshold": self.parallel_threshold,
            "num_trees_in_forest": self._forest.num_trees(),
        }

    def get_tree_idx_for_request(self, req_id: Hashable) -> Optional[int]:
        """
        Get the tree index for a request.

        Args:
            req_id: The request identifier.

        Returns:
            Tree index if request is active, None otherwise.
        """
        return self._req_to_tree_idx.get(req_id)

    def get_requests_per_tree(self) -> Dict[int, int]:
        """
        Get count of active requests per tree index.

        Returns:
            Dictionary mapping tree index to count of active requests using that tree.
        """
        counts: Dict[int, int] = {}
        for tree_idx in self._req_to_tree_idx.values():
            counts[tree_idx] = counts.get(tree_idx, 0) + 1
        return counts

    def estimate_memory(self) -> int:
        """
        Estimate total memory usage of the suffix forest in bytes.

        This walks all trees and accumulates their memory usage.
        Use sparingly as it's O(N) in total tree size.

        Returns:
            Total estimated memory in bytes.
        """
        return self._forest.estimate_forest_memory()

    def get_memory_stats(self) -> dict:
        """
        Get memory statistics for the cache.

        Returns:
            Dictionary with memory statistics including:
            - total_bytes: Total memory usage
            - num_trees: Number of trees in forest
            - avg_bytes_per_tree: Average memory per tree
        """
        total_bytes = self._forest.estimate_forest_memory()
        num_trees = self._forest.num_trees()

        return {
            "total_bytes": total_bytes,
            "num_trees": num_trees,
            "avg_bytes_per_tree": total_bytes // num_trees if num_trees > 0 else 0,
        }

    # ========== Snapshot Methods ==========

    def load_snapshot(
        self,
        snapshots: List[Tuple[int, bytes]],
        hash_to_tree: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Load snapshot, replacing current forest state.

        This method clears all existing trees and restores from the provided
        snapshots. Used at batch start to receive patterns from controller.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples from controller.
                       Empty list resets to an empty forest.
            hash_to_tree: Optional mapping from prompt hashes to tree indices.
                         If provided, enables hash-based tree sharing for
                         the loaded trees. If None, hash lookup is empty.
        """
        # Clear all existing state
        self._req_to_tree_idx.clear()
        self._req_to_seq_id.clear()
        self._hash_to_tree_idx.clear()
        self._tree_next_seq_id.clear()
        self._protected_tree_indices.clear()

        # Restore forest from snapshots
        if snapshots:
            self._forest = SuffixForest.from_snapshots(
                self._max_tree_depth, snapshots,
                self._num_threads, self._parallel_threshold
            )
            # Restore hash-to-tree mapping if provided
            if hash_to_tree:
                self._hash_to_tree_idx.update(hash_to_tree)
            # Initialize seq_id counters for loaded trees
            # and mark them as protected (won't be GC'd)
            for tree_idx in self._forest.get_tree_indices():
                # Start seq_id after existing sequences to avoid collision
                tree = self._forest.get_tree(tree_idx)
                self._tree_next_seq_id[tree_idx] = tree.num_seqs()
                self._protected_tree_indices.add(tree_idx)
        else:
            self._forest = SuffixForest(
                self._max_tree_depth, self._num_threads, self._parallel_threshold
            )

    def register_requests_for_loaded_trees(
        self,
        req_id_mapping: Dict[Hashable, int],
    ) -> None:
        """
        Register request IDs to be associated with loaded tree indices.

        This is useful when you've loaded a snapshot and need to associate
        new request IDs with the pre-loaded trees. Each request gets a unique
        seq_id within its tree.

        Args:
            req_id_mapping: Mapping from request IDs to tree indices in the forest.
                           The tree indices must exist in the current forest.

        Raises:
            ValueError: If any tree index doesn't exist in the forest.
        """
        valid_indices = set(self._forest.get_tree_indices())
        for req_id, tree_idx in req_id_mapping.items():
            if tree_idx not in valid_indices:
                raise ValueError(
                    f"Tree index {tree_idx} for request '{req_id}' "
                    f"not found in forest. Valid indices: {valid_indices}"
                )
            self._req_to_tree_idx[req_id] = tree_idx
            # Assign next available seq_id for this tree
            seq_id = self._tree_next_seq_id.get(tree_idx, 0)
            self._req_to_seq_id[req_id] = seq_id
            self._tree_next_seq_id[tree_idx] = seq_id + 1

    def create_snapshot(
        self,
        include_hash_mapping: bool = False,
    ) -> Union[
        List[Tuple[int, bytes]],
        Tuple[List[Tuple[int, bytes]], Dict[str, int]]
    ]:
        """
        Create snapshot of current forest state.

        Args:
            include_hash_mapping: If True, also return the hash-to-tree mapping.
                                 This enables hash-based tree sharing after restore.

        Returns:
            If include_hash_mapping is False:
                List of (tree_idx, snapshot_bytes) tuples representing all trees.
            If include_hash_mapping is True:
                Tuple of (snapshots_list, hash_to_tree_mapping) where hash_to_tree_mapping
                maps prompt hashes to their tree indices.
        """
        snapshots = [(idx, self._forest.get_tree(idx).create_snapshot())
                     for idx in self._forest.get_tree_indices()]

        if include_hash_mapping:
            return snapshots, dict(self._hash_to_tree_idx)
        return snapshots

    def create_selective_snapshot(
        self,
        tree_indices: List[int],
        include_hash_mapping: bool = False,
    ) -> Union[
        List[Tuple[int, bytes]],
        Tuple[List[Tuple[int, bytes]], Dict[str, int]]
    ]:
        """
        Create snapshot of specific trees only (selective serialization).

        This is more efficient than create_snapshot() when only a subset of trees
        is needed for a particular batch. Uses C++ optimized implementation that
        releases the GIL during serialization.

        Args:
            tree_indices: List of tree indices to serialize. Missing indices
                         are silently skipped (partial results returned).
            include_hash_mapping: If True, also return the hash-to-tree mapping
                                 filtered to only include the requested trees.

        Returns:
            If include_hash_mapping is False:
                List of (tree_idx, snapshot_bytes) tuples for found trees.
            If include_hash_mapping is True:
                Tuple of (snapshots_list, filtered_hash_mapping) where
                filtered_hash_mapping only includes hashes for requested trees.
        """
        # Call C++ selective snapshot (releases GIL, sequential serialization)
        snapshots = self._forest.create_selective_snapshot(tree_indices)

        if include_hash_mapping:
            # Filter hash mapping to only include requested trees
            tree_set = set(tree_indices)
            filtered_mapping = {
                h: idx for h, idx in self._hash_to_tree_idx.items()
                if idx in tree_set
            }
            return snapshots, filtered_mapping

        return snapshots

    def _validate_ndarray(self, arr: np.ndarray):
        """Validate that a numpy array has the correct shape and dtype."""
        if arr.ndim != 1:
            raise ValueError(
                f"ndarray input must have ndim=1, got ndim={arr.ndim}"
            )
        if arr.dtype != np.int32:
            raise ValueError(
                f"ndarray input must have dtype=int32, got dtype={arr.dtype.name}"
            )
        if not arr.flags["CONTIGUOUS"]:
            raise ValueError("ndarray input must be contiguous")
