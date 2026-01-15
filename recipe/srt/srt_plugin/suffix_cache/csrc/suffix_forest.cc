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

#include "suffix_forest.h"

#include <algorithm>
#include <thread>
#include <stdexcept>
#include <sstream>
#include <iostream>

// OpenMP support (optional, will be enabled via CMake if available)
#ifdef _OPENMP
#include <omp.h>
#endif

SuffixForest::SuffixForest(int max_depth, int num_threads, int parallel_threshold)
    : _max_depth(max_depth),
      _num_threads(_resolve_num_threads(num_threads)),
      _parallel_threshold(parallel_threshold) {

    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    if (parallel_threshold < 0) {
        throw std::invalid_argument("parallel_threshold must be non-negative");
    }
}

SuffixForest::SuffixForest(
    int max_depth,
    const std::vector<std::pair<int, std::span<const uint8_t>>>& tree_snapshots,
    int num_threads,
    int parallel_threshold)
    : _max_depth(max_depth),
      _num_threads(_resolve_num_threads(num_threads)),
      _parallel_threshold(parallel_threshold) {

    if (max_depth <= 0) {
        throw std::invalid_argument("max_depth must be positive");
    }

    if (parallel_threshold < 0) {
        throw std::invalid_argument("parallel_threshold must be non-negative");
    }

    int max_id = 0;
    for (const auto& [tree_idx, snapshot] : tree_snapshots) {
        if (tree_idx < 0) {
            throw std::invalid_argument("tree_index must be non-negative");
        }
        auto tree = SuffixTree::restore_snapshot(snapshot);
        _trees[tree_idx] = std::make_unique<SuffixTree>(std::move(tree));
        _tree_mutexes[tree_idx] = std::make_unique<std::shared_mutex>();
        max_id = std::max(max_id, tree_idx + 1);
    }
    _next_tree_id = max_id;
}

int SuffixForest::_resolve_num_threads(int num_threads) {
    if (num_threads < -1) {
        throw std::invalid_argument("num_threads must be >= -1");
    }

    if (num_threads == -1) {
        // Auto-detect from hardware concurrency
        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) {
            // Fallback if hardware_concurrency fails
            return 4;
        }
        return static_cast<int>(hw_threads);
    }

    return num_threads;
}

int SuffixForest::create_tree() {
    int tree_index = _next_tree_id++;
    _trees[tree_index] = std::make_unique<SuffixTree>(_max_depth);
    _tree_mutexes[tree_index] = std::make_unique<std::shared_mutex>();
    return tree_index;
}

void SuffixForest::remove_tree(int tree_index) {
    _validate_tree_index(tree_index);
    _trees.erase(tree_index);
    _tree_mutexes.erase(tree_index);
}

SuffixTree& SuffixForest::get_tree(int tree_index) {
    _validate_tree_index(tree_index);
    return *_trees[tree_index];
}

const SuffixTree& SuffixForest::get_tree(int tree_index) const {
    _validate_tree_index(tree_index);
    auto it = _trees.find(tree_index);
    return *it->second;
}

bool SuffixForest::has_tree(int tree_index) const {
    return _trees.find(tree_index) != _trees.end();
}

std::vector<int> SuffixForest::get_tree_indices() const {
    std::vector<int> indices;
    indices.reserve(_trees.size());
    for (const auto& [idx, _] : _trees) {
        indices.push_back(idx);
    }
    std::sort(indices.begin(), indices.end());
    return indices;
}

void SuffixForest::_validate_tree_index(int tree_index) const {
    if (!has_tree(tree_index)) {
        std::ostringstream oss;
        oss << "Tree index " << tree_index << " does not exist in forest";
        throw std::invalid_argument(oss.str());
    }
}

std::unique_lock<std::shared_mutex> SuffixForest::get_write_lock(int tree_index) {
    _validate_tree_index(tree_index);
    return std::unique_lock<std::shared_mutex>(*_tree_mutexes[tree_index]);
}

std::shared_lock<std::shared_mutex> SuffixForest::get_read_lock(int tree_index) const {
    _validate_tree_index(tree_index);
    auto it = _tree_mutexes.find(tree_index);
    return std::shared_lock<std::shared_mutex>(*it->second);
}

void SuffixForest::batch_extend(
    std::span<const int> tree_indices,
    std::span<const int> seq_ids,
    const std::vector<std::vector<int32_t>>& token_batches) {

    // Validate inputs
    if (tree_indices.size() != seq_ids.size() || tree_indices.size() != token_batches.size()) {
        std::ostringstream oss;
        oss << "tree_indices, seq_ids, and token_batches must have the same size. Got "
            << tree_indices.size() << ", " << seq_ids.size() << ", and " << token_batches.size();
        throw std::invalid_argument(oss.str());
    }

    const size_t batch_size = tree_indices.size();

    // Early return for empty batch
    if (batch_size == 0) {
        return;
    }

    // Validate all tree indices exist before starting
    for (size_t i = 0; i < batch_size; ++i) {
        _validate_tree_index(tree_indices[i]);
    }

    // Determine whether to use parallel or sequential execution
    const bool use_parallel = (batch_size >= static_cast<size_t>(_parallel_threshold)) &&
                             (_num_threads > 0);

    if (use_parallel) {
#ifdef _OPENMP
        // OpenMP parallel execution
        // Flag to track if any thread encountered an error
        bool error_occurred = false;
        std::string error_message;

        // Use num_threads clause instead of omp_set_num_threads() to avoid overhead
        // Use static scheduling for better performance with uniform work
        #pragma omp parallel for schedule(static) num_threads(_num_threads)
        for (size_t i = 0; i < batch_size; ++i) {
            try {
                int tree_idx = tree_indices[i];
                int seq_id = seq_ids[i];

                // Acquire write lock for this tree to ensure thread-safety
                // when multiple requests share the same tree
                std::unique_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

                SuffixTree& tree = *_trees[tree_idx];
                tree.extend(seq_id, token_batches[i]);
            } catch (const std::exception& e) {
                // Fail-fast: mark error and save message
                #pragma omp critical
                {
                    if (!error_occurred) {
                        error_occurred = true;
                        std::ostringstream oss;
                        oss << "Extend failed for tree " << tree_indices[i]
                            << " at batch index " << i << ": " << e.what();
                        error_message = oss.str();
                    }
                }
            }
        }

        // Check if any errors occurred
        if (error_occurred) {
            throw std::runtime_error(error_message);
        }
#else
        // OpenMP not available, fall back to sequential
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            int seq_id = seq_ids[i];

            // Acquire write lock for this tree
            std::unique_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

            SuffixTree& tree = *_trees[tree_idx];

            try {
                tree.extend(seq_id, token_batches[i]);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Extend failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
#endif
    } else {
        // Sequential execution (below threshold or num_threads == 0)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            int seq_id = seq_ids[i];

            // Acquire write lock for this tree
            std::unique_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

            SuffixTree& tree = *_trees[tree_idx];

            try {
                tree.extend(seq_id, token_batches[i]);
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Extend failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }
}

std::vector<Draft> SuffixForest::batch_speculate(
    std::span<const int> tree_indices,
    const std::vector<std::vector<int32_t>>& contexts,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Validate inputs
    if (tree_indices.size() != contexts.size()) {
        std::ostringstream oss;
        oss << "tree_indices and contexts must have the same size. Got "
            << tree_indices.size() << " and " << contexts.size();
        throw std::invalid_argument(oss.str());
    }

    const size_t batch_size = tree_indices.size();

    // Early return for empty batch
    if (batch_size == 0) {
        return std::vector<Draft>();
    }

    // Validate all tree indices exist before starting speculation
    for (size_t i = 0; i < batch_size; ++i) {
        _validate_tree_index(tree_indices[i]);
    }

    // Allocate results vector
    std::vector<Draft> results(batch_size);

    // Determine whether to use parallel or sequential execution
    const bool use_parallel = (batch_size >= static_cast<size_t>(_parallel_threshold)) &&
                             (_num_threads > 0);

    if (use_parallel) {
#ifdef _OPENMP
        // OpenMP parallel execution
        // Flag to track if any thread encountered an error
        bool error_occurred = false;
        std::string error_message;

        // Use num_threads clause instead of omp_set_num_threads() to avoid overhead
        // Use static scheduling for better performance with uniform work
        #pragma omp parallel for schedule(static) num_threads(_num_threads)
        for (size_t i = 0; i < batch_size; ++i) {
            try {
                int tree_idx = tree_indices[i];

                // Acquire read lock for this tree (allows concurrent reads)
                std::shared_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

                SuffixTree& tree = *_trees[tree_idx];

                results[i] = tree.speculate(
                    contexts[i],
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                // Fail-fast: mark error and save message
                #pragma omp critical
                {
                    if (!error_occurred) {
                        error_occurred = true;
                        std::ostringstream oss;
                        oss << "Speculation failed for tree " << tree_indices[i]
                            << " at batch index " << i << ": " << e.what();
                        error_message = oss.str();
                    }
                }
            }
        }

        // Check if any errors occurred
        if (error_occurred) {
            throw std::runtime_error(error_message);
        }
#else
        // OpenMP not available, fall back to sequential
        // (This branch should rarely be hit if CMake is configured correctly)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];

            // Acquire read lock for this tree
            std::shared_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

            SuffixTree& tree = *_trees[tree_idx];

            try {
                results[i] = tree.speculate(
                    contexts[i],
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Speculation failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
#endif
    } else {
        // Sequential execution (below threshold or num_threads == 0)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];

            // Acquire read lock for this tree
            std::shared_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

            SuffixTree& tree = *_trees[tree_idx];

            try {
                results[i] = tree.speculate(
                    contexts[i],
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Speculation failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }

    return results;
}

std::vector<Draft> SuffixForest::propose_from_batch(
    std::span<const int32_t> token_ids_cpu,
    int max_seq_len,
    std::span<const int32_t> num_tokens,
    std::span<const int32_t> tree_indices,
    int max_tree_depth,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Validate inputs
    const size_t batch_size = num_tokens.size();
    if (tree_indices.size() != batch_size) {
        std::ostringstream oss;
        oss << "tree_indices and num_tokens must have the same size. Got "
            << tree_indices.size() << " and " << batch_size;
        throw std::invalid_argument(oss.str());
    }

    if (token_ids_cpu.size() != static_cast<size_t>(batch_size * max_seq_len)) {
        std::ostringstream oss;
        oss << "token_ids_cpu size must equal batch_size * max_seq_len. Got "
            << token_ids_cpu.size() << " but expected " << (batch_size * max_seq_len);
        throw std::invalid_argument(oss.str());
    }

    // Early return for empty batch
    if (batch_size == 0) {
        return std::vector<Draft>();
    }

    // Validate all tree indices exist before starting
    for (size_t i = 0; i < batch_size; ++i) {
        _validate_tree_index(tree_indices[i]);
    }

    // Allocate results vector
    std::vector<Draft> results(batch_size);

    // Determine whether to use parallel or sequential execution
    const bool use_parallel = (batch_size >= static_cast<size_t>(_parallel_threshold)) &&
                             (_num_threads > 0);

    if (use_parallel) {
#ifdef _OPENMP
        // OpenMP parallel execution
        bool error_occurred = false;
        std::string error_message;

        #pragma omp parallel for schedule(static) num_threads(_num_threads)
        for (size_t i = 0; i < batch_size; ++i) {
            try {
                int tree_idx = tree_indices[i];
                int num_tok = num_tokens[i];

                // Extract context using direct pointer arithmetic (zero-copy)
                int start = std::max(0, num_tok - max_tree_depth);
                int context_len = num_tok - start;

                // Direct pointer to context in token_ids_cpu
                const int32_t* context_ptr = &token_ids_cpu[i * max_seq_len + start];

                // Create span for context (zero-copy)
                std::span<const int32_t> context(context_ptr, context_len);

                // Acquire read lock for this tree (allows concurrent reads)
                std::shared_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

                SuffixTree& tree = *_trees[tree_idx];

                // Perform speculation
                results[i] = tree.speculate(
                    context,
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                // Fail-fast: mark error and save message
                #pragma omp critical
                {
                    if (!error_occurred) {
                        error_occurred = true;
                        std::ostringstream oss;
                        oss << "Speculation failed for tree " << tree_indices[i]
                            << " at batch index " << i << ": " << e.what();
                        error_message = oss.str();
                    }
                }
            }
        }

        // Check if any errors occurred
        if (error_occurred) {
            throw std::runtime_error(error_message);
        }
#else
        // OpenMP not available, fall back to sequential
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            int num_tok = num_tokens[i];

            // Extract context using direct pointer arithmetic (zero-copy)
            int start = std::max(0, num_tok - max_tree_depth);
            int context_len = num_tok - start;

            const int32_t* context_ptr = &token_ids_cpu[i * max_seq_len + start];
            std::span<const int32_t> context(context_ptr, context_len);

            // Acquire read lock for this tree
            std::shared_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

            SuffixTree& tree = *_trees[tree_idx];

            try {
                results[i] = tree.speculate(
                    context,
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Speculation failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
#endif
    } else {
        // Sequential execution (below threshold or num_threads == 0)
        for (size_t i = 0; i < batch_size; ++i) {
            int tree_idx = tree_indices[i];
            int num_tok = num_tokens[i];

            // Extract context using direct pointer arithmetic (zero-copy)
            int start = std::max(0, num_tok - max_tree_depth);
            int context_len = num_tok - start;

            const int32_t* context_ptr = &token_ids_cpu[i * max_seq_len + start];
            std::span<const int32_t> context(context_ptr, context_len);

            // Acquire read lock for this tree
            std::shared_lock<std::shared_mutex> lock(*_tree_mutexes[tree_idx]);

            SuffixTree& tree = *_trees[tree_idx];

            try {
                results[i] = tree.speculate(
                    context,
                    max_spec_tokens,
                    max_spec_factor,
                    max_spec_offset,
                    min_token_prob,
                    use_tree_spec
                );
            } catch (const std::exception& e) {
                std::ostringstream oss;
                oss << "Speculation failed for tree " << tree_idx
                    << " at batch index " << i << ": " << e.what();
                throw std::runtime_error(oss.str());
            }
        }
    }

    return results;
}

size_t SuffixForest::estimate_forest_memory() const {
    size_t total = sizeof(*this);
    total += _trees.memory_usage();
    total += _tree_mutexes.memory_usage();

    for (const auto& [tree_idx, tree_ptr] : _trees) {
        if (tree_ptr) {
            total += tree_ptr->estimate_memory();
        }
    }
    return total;
}

std::vector<std::pair<int, size_t>> SuffixForest::estimate_trees_memory() const {
    std::vector<std::pair<int, size_t>> result;
    result.reserve(_trees.size());

    for (const auto& [tree_idx, tree_ptr] : _trees) {
        if (tree_ptr) {
            result.emplace_back(tree_idx, tree_ptr->estimate_memory());
        }
    }
    return result;
}

std::vector<std::pair<int, std::vector<uint8_t>>> SuffixForest::create_selective_snapshot(
    const std::vector<int>& tree_indices) const {

    std::vector<std::pair<int, std::vector<uint8_t>>> results;
    results.reserve(tree_indices.size());

    // Sequential serialization - one tree at a time
    for (int tree_idx : tree_indices) {
        auto it = _trees.find(tree_idx);
        if (it != _trees.end() && it->second) {
            // Acquire read lock for thread safety during serialization
            auto lock = get_read_lock(tree_idx);
            results.emplace_back(tree_idx, it->second->create_snapshot());
        }
        // Silently skip missing indices - allows partial results
    }

    return results;
}
