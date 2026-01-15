// Copyright 2025 Snowflake Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <span>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <shared_mutex>

#include "int32_map.h"
#include "suffix_tree.h"

/**
 * SuffixForest manages a collection of independent SuffixTree instances and
 * provides batched parallel speculation operations.
 *
 * This class enables efficient parallel speculation across multiple requests,
 * with each tree being completely independent and thus naturally parallelizable.
 */
class SuffixForest {
public:
    /**
     * Construct a new SuffixForest.
     *
     * @param max_depth Maximum depth for all suffix trees in this forest.
     * @param num_threads Number of threads to use for parallel operations.
     *                    -1 = auto-detect from CPU count (default)
     *                    0 = disable parallelization (sequential)
     *                    >0 = use specified number of threads
     * @param parallel_threshold Minimum batch size to trigger parallelization.
     *                          Batches smaller than this will run sequentially
     *                          to avoid thread overhead. Default: 4.
     */
    SuffixForest(int max_depth, int num_threads = -1, int parallel_threshold = 4);

    /**
     * Construct a SuffixForest from tree snapshots.
     *
     * This constructor restores trees from their serialized snapshots,
     * preserving the original tree indices.
     *
     * @param max_depth Maximum depth for all suffix trees in this forest.
     * @param tree_snapshots Vector of (tree_index, snapshot_bytes) pairs.
     * @param num_threads Number of threads to use for parallel operations.
     * @param parallel_threshold Minimum batch size to trigger parallelization.
     */
    SuffixForest(
        int max_depth,
        const std::vector<std::pair<int, std::span<const uint8_t>>>& tree_snapshots,
        int num_threads = -1,
        int parallel_threshold = 4);

    /**
     * Create a new tree in the forest.
     *
     * @return The tree index (ID) that can be used to reference this tree.
     */
    int create_tree();

    /**
     * Remove a tree from the forest.
     *
     * @param tree_index The index of the tree to remove.
     * @throws std::invalid_argument if tree_index does not exist.
     */
    void remove_tree(int tree_index);

    /**
     * Get a reference to a tree in the forest.
     *
     * @param tree_index The index of the tree to retrieve.
     * @return Reference to the SuffixTree.
     * @throws std::invalid_argument if tree_index does not exist.
     */
    SuffixTree& get_tree(int tree_index);

    /**
     * Get a const reference to a tree in the forest.
     *
     * @param tree_index The index of the tree to retrieve.
     * @return Const reference to the SuffixTree.
     * @throws std::invalid_argument if tree_index does not exist.
     */
    const SuffixTree& get_tree(int tree_index) const;

    /**
     * Check if a tree exists in the forest.
     *
     * @param tree_index The index of the tree to check.
     * @return true if the tree exists, false otherwise.
     */
    bool has_tree(int tree_index) const;

    /**
     * Get the number of trees currently in the forest.
     *
     * @return Number of active trees.
     */
    int num_trees() const {
        return static_cast<int>(_trees.size());
    }

    /**
     * Get all tree indices in the forest.
     *
     * @return Vector of tree indices (sorted).
     */
    std::vector<int> get_tree_indices() const;

    /**
     * Batch extend multiple trees with new tokens in parallel.
     *
     * This allows efficiently adding tokens to multiple trees simultaneously.
     * Each tree is extended independently, enabling parallelization.
     *
     * @param tree_indices Indices of the trees to extend.
     * @param seq_ids Sequence IDs within each tree (typically 0 for single sequence per tree).
     * @param token_batches Token sequences to add, one per tree_index.
     *
     * @throws std::invalid_argument if tree_indices, seq_ids, and token_batches have different sizes,
     *                               or if any tree_index does not exist.
     * @throws std::runtime_error if any extend operation fails.
     */
    void batch_extend(
        std::span<const int> tree_indices,
        std::span<const int> seq_ids,
        const std::vector<std::vector<int32_t>>& token_batches);

    /**
     * Perform batched speculation across multiple trees in parallel.
     *
     * This is the main method for parallel speculation. Each tree is processed
     * independently, allowing for efficient parallelization. If any tree fails
     * during speculation, the entire batch fails (fail-fast behavior).
     *
     * @param tree_indices Indices of the trees to speculate on. Must not contain
     *                     duplicates (undefined behavior if duplicates present).
     * @param contexts Context sequences for each tree (one per tree_index).
     *                 contexts[i] is the context for tree_indices[i].
     * @param max_spec_tokens Maximum number of tokens to speculate.
     * @param max_spec_factor Factor for speculation length calculation.
     * @param max_spec_offset Offset for speculation limit.
     * @param min_token_prob Minimum probability threshold for tokens.
     * @param use_tree_spec Whether to use tree speculation (vs path speculation).
     *
     * @return Vector of Draft objects, one per tree_index.
     * @throws std::invalid_argument if tree_indices and contexts have different sizes,
     *                               or if any tree_index does not exist.
     * @throws std::runtime_error if any speculation fails.
     */
    std::vector<Draft> batch_speculate(
        std::span<const int> tree_indices,
        const std::vector<std::vector<int32_t>>& contexts,
        int max_spec_tokens,
        float max_spec_factor,
        float max_spec_offset,
        float min_token_prob,
        bool use_tree_spec);

    /**
     * Optimized batch speculation that directly consumes InputBatch-like numpy arrays.
     *
     * This method eliminates Python loop overhead by:
     * 1. Directly accessing token_ids_cpu via pointer arithmetic (zero-copy)
     * 2. Extracting contexts in parallel with OpenMP
     * 3. Performing speculation in the same parallel region
     *
     * Expected speedup: ~3-3.5x for batch_size >= 32 compared to Python loop + batch_speculate.
     *
     * @param token_ids_cpu Flattened 2D array [batch_size, max_seq_len] of token IDs.
     *                      Passed as 1D span, indexed as [i * max_seq_len + j].
     * @param max_seq_len Sequence length dimension (stride for 2D indexing).
     * @param num_tokens Number of valid tokens for each request [batch_size].
     * @param tree_indices Tree index for each request [batch_size].
     * @param max_tree_depth Maximum context depth to extract from token_ids_cpu.
     * @param max_spec_tokens Maximum number of tokens to speculate per request.
     * @param max_spec_factor Factor for speculation length calculation.
     * @param max_spec_offset Offset for speculation limit.
     * @param min_token_prob Minimum probability threshold for tokens.
     * @param use_tree_spec Whether to use tree speculation (vs path speculation).
     *
     * @return Vector of Draft objects, one per batch item.
     * @throws std::invalid_argument if arrays have inconsistent sizes or invalid tree indices.
     * @throws std::runtime_error if any speculation fails.
     */
    std::vector<Draft> propose_from_batch(
        std::span<const int32_t> token_ids_cpu,
        int max_seq_len,
        std::span<const int32_t> num_tokens,
        std::span<const int32_t> tree_indices,
        int max_tree_depth,
        int max_spec_tokens,
        float max_spec_factor,
        float max_spec_offset,
        float min_token_prob,
        bool use_tree_spec);

    /**
     * Get the configured number of threads.
     *
     * @return Number of threads (resolved from auto-detect if -1 was specified).
     */
    int get_num_threads() const {
        return _num_threads;
    }

    /**
     * Get the parallel threshold.
     *
     * @return Minimum batch size for parallelization.
     */
    int get_parallel_threshold() const {
        return _parallel_threshold;
    }

    /**
     * Get exclusive (write) lock for a tree. Use for extend/remove operations.
     *
     * This method acquires an exclusive lock that blocks both readers and writers.
     * Use this when modifying a tree (extend, append, remove operations).
     *
     * @param tree_index The index of the tree to lock.
     * @return A unique_lock that holds exclusive access until destroyed.
     * @throws std::invalid_argument if tree_index does not exist.
     */
    std::unique_lock<std::shared_mutex> get_write_lock(int tree_index);

    /**
     * Get shared (read) lock for a tree. Use for speculate operations.
     *
     * This method acquires a shared lock that allows concurrent readers but
     * blocks writers. Use this when reading from a tree (speculate operations).
     *
     * @param tree_index The index of the tree to lock.
     * @return A shared_lock that holds read access until destroyed.
     * @throws std::invalid_argument if tree_index does not exist.
     */
    std::shared_lock<std::shared_mutex> get_read_lock(int tree_index) const;

    /**
     * Estimate total memory usage across all trees in the forest.
     *
     * This walks all trees and accumulates their memory usage.
     * Use sparingly as it's O(N) in total tree size.
     *
     * @return Total estimated memory in bytes.
     */
    size_t estimate_forest_memory() const;

    /**
     * Estimate memory usage for each tree in the forest.
     *
     * @return Vector of (tree_index, memory_bytes) pairs.
     */
    std::vector<std::pair<int, size_t>> estimate_trees_memory() const;

    /**
     * Create snapshots for specific trees only (selective serialization).
     *
     * This is more efficient than serializing all trees when only a subset
     * is needed for a particular batch. Serializes trees sequentially.
     *
     * @param tree_indices Indices of trees to serialize.
     * @return Vector of (tree_index, snapshot_bytes) pairs for found trees.
     *         Missing indices are silently skipped (partial results returned).
     */
    std::vector<std::pair<int, std::vector<uint8_t>>> create_selective_snapshot(
        const std::vector<int>& tree_indices) const;

private:
    // Maximum depth for all trees in this forest
    int _max_depth;

    // Number of threads to use for parallel operations
    int _num_threads;

    // Minimum batch size to trigger parallelization
    int _parallel_threshold;

    // Map from tree index to SuffixTree instance
    Int32Map<std::unique_ptr<SuffixTree>> _trees;

    // Per-tree mutexes for thread-safe access
    // Using shared_mutex to allow concurrent reads (speculate) but exclusive writes (extend)
    mutable Int32Map<std::unique_ptr<std::shared_mutex>> _tree_mutexes;

    // Counter for assigning new tree indices
    int _next_tree_id = 0;

    // Helper method to validate tree index
    void _validate_tree_index(int tree_index) const;

    // Helper method to resolve num_threads from -1 (auto) to actual count
    static int _resolve_num_threads(int num_threads);
};
