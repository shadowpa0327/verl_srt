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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <span>
#include <utility>

#include "suffix_tree.h"
#include "suffix_forest.h"

namespace nb = nanobind;

using Int32Array1D = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>,
                                 nb::device::cpu, nb::any_contig>;


void extend_ndarray(SuffixTree& tree,
                    int seq_id,
                    const Int32Array1D& tokens) {
    tree.extend(
        seq_id,
        std::span<const int32_t>(tokens.data(), tokens.size()));
}


void extend_vector(SuffixTree& tree,
                   int seq_id,
                   const std::vector<int32_t>& tokens) {
    tree.extend(seq_id, std::span<const int32_t>(tokens));
}


Draft speculate_ndarray(SuffixTree& tree,
                        const Int32Array1D& context,
                        int max_spec_tokens,
                        float max_spec_factor,
                        float max_spec_offset,
                        float min_token_prob,
                        bool use_tree_spec) {
    return tree.speculate(
        std::span<const int32_t>(context.data(), context.size()),
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


Draft speculate_vector(SuffixTree& tree,
                       const std::vector<int32_t>& context,
                       int max_spec_tokens,
                       float max_spec_factor,
                       float max_spec_offset,
                       float min_token_prob,
                       bool use_tree_spec) {
    return tree.speculate(
        std::span<const int32_t>(context),
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


// SuffixForest wrapper functions

void batch_extend_forest(
    SuffixForest& forest,
    const std::vector<int>& tree_indices,
    const std::vector<int>& seq_ids,
    const std::vector<std::vector<int32_t>>& token_batches) {

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    forest.batch_extend(
        std::span<const int>(tree_indices),
        std::span<const int>(seq_ids),
        token_batches);
}

void batch_extend_forest_ndarray(
    SuffixForest& forest,
    const Int32Array1D& tree_indices,
    const Int32Array1D& seq_ids,
    nb::list token_batches_list) {

    // Convert tree_indices ndarray to vector
    std::vector<int> tree_indices_vec(tree_indices.size());
    for (size_t i = 0; i < tree_indices.size(); ++i) {
        tree_indices_vec[i] = tree_indices.data()[i];
    }

    // Convert seq_ids ndarray to vector
    std::vector<int> seq_ids_vec(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        seq_ids_vec[i] = seq_ids.data()[i];
    }

    // Convert token_batches list of ndarrays to vector of vectors
    std::vector<std::vector<int32_t>> token_batches_vec;
    token_batches_vec.reserve(token_batches_list.size());

    for (size_t i = 0; i < token_batches_list.size(); ++i) {
        nb::handle tokens_handle = token_batches_list[i];
        Int32Array1D tokens = nb::cast<Int32Array1D>(tokens_handle);
        token_batches_vec.emplace_back(tokens.data(), tokens.data() + tokens.size());
    }

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    forest.batch_extend(
        std::span<const int>(tree_indices_vec),
        std::span<const int>(seq_ids_vec),
        token_batches_vec);
}

std::vector<Draft> batch_speculate_forest(
    SuffixForest& forest,
    const std::vector<int>& tree_indices,
    const std::vector<std::vector<int32_t>>& contexts,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    return forest.batch_speculate(
        std::span<const int>(tree_indices),
        contexts,
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


std::vector<Draft> batch_speculate_forest_ndarray(
    SuffixForest& forest,
    const Int32Array1D& tree_indices,
    nb::list contexts_list,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Convert tree_indices ndarray to vector
    std::vector<int> tree_indices_vec(tree_indices.size());
    for (size_t i = 0; i < tree_indices.size(); ++i) {
        tree_indices_vec[i] = tree_indices.data()[i];
    }

    // Convert contexts list of ndarrays to vector of vectors
    std::vector<std::vector<int32_t>> contexts_vec;
    contexts_vec.reserve(contexts_list.size());

    for (size_t i = 0; i < contexts_list.size(); ++i) {
        nb::handle ctx_handle = contexts_list[i];
        Int32Array1D ctx = nb::cast<Int32Array1D>(ctx_handle);
        contexts_vec.emplace_back(ctx.data(), ctx.data() + ctx.size());
    }

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    return forest.batch_speculate(
        std::span<const int>(tree_indices_vec),
        contexts_vec,
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


std::vector<Draft> propose_from_batch_forest(
    SuffixForest& forest,
    const nb::ndarray<int32_t, nb::numpy, nb::shape<-1, -1>, nb::device::cpu, nb::c_contig>& token_ids_cpu,
    const Int32Array1D& num_tokens,
    const Int32Array1D& tree_indices,
    int max_tree_depth,
    int max_spec_tokens,
    float max_spec_factor,
    float max_spec_offset,
    float min_token_prob,
    bool use_tree_spec) {

    // Get batch size and max_seq_len from the 2D array shape
    size_t batch_size = token_ids_cpu.shape(0);
    size_t max_seq_len = token_ids_cpu.shape(1);

    // Validate array sizes
    if (num_tokens.size() != batch_size) {
        throw std::invalid_argument(
            "num_tokens size must match token_ids_cpu batch dimension");
    }
    if (tree_indices.size() != batch_size) {
        throw std::invalid_argument(
            "tree_indices size must match token_ids_cpu batch dimension");
    }

    // Get flat pointer to 2D array data (C-contiguous layout)
    const int32_t* token_ids_ptr = token_ids_cpu.data();

    // Create spans for the input arrays
    std::span<const int32_t> token_ids_span(token_ids_ptr, batch_size * max_seq_len);
    std::span<const int32_t> num_tokens_span(num_tokens.data(), num_tokens.size());
    std::span<const int32_t> tree_indices_span(tree_indices.data(), tree_indices.size());

    // Release GIL for parallel execution
    nb::gil_scoped_release release;

    // Call the C++ method
    return forest.propose_from_batch(
        token_ids_span,
        static_cast<int>(max_seq_len),
        num_tokens_span,
        tree_indices_span,
        max_tree_depth,
        max_spec_tokens,
        max_spec_factor,
        max_spec_offset,
        min_token_prob,
        use_tree_spec);
}


NB_MODULE(_C, m) {
    nb::set_leak_warnings(false);

    nb::class_<Draft>(m, "Draft")
        .def_rw("token_ids", &Draft::token_ids)
        .def_rw("parents", &Draft::parents)
        .def_rw("probs", &Draft::probs)
        .def_rw("score", &Draft::score)
        .def_rw("match_len", &Draft::match_len);

    nb::class_<SuffixTree>(m, "SuffixTree")
        .def(nb::init<int>())
        .def("num_seqs", &SuffixTree::num_seqs)
        .def("remove", &SuffixTree::remove)
        // Overloads for extend method. Use different names to avoid overload
        // resolution overhead at run-time.
        .def("extend", &extend_vector)
        .def("extend_ndarray", &extend_ndarray)
        // Overloads for speculate method.
        .def("speculate", &speculate_vector)
        .def("speculate_ndarray", &speculate_ndarray)
        // Debugging methods, not meant to be used in critical loop.
        .def("check_integrity", &SuffixTree::check_integrity)
        .def("estimate_memory", &SuffixTree::estimate_memory)
        .def("dump", &SuffixTree::dump, "Return tree structure as a string for visualization")
        // Serialization methods
        .def("create_snapshot", [](const SuffixTree& tree) {
            auto data = tree.create_snapshot();
            return nb::bytes(reinterpret_cast<const char*>(data.data()), data.size());
        }, "Serialize tree to binary format")
        .def_static("restore_snapshot", [](nb::bytes data) {
            return SuffixTree::restore_snapshot(
                std::span<const uint8_t>(
                    reinterpret_cast<const uint8_t*>(data.data()),
                    data.size()
                )
            );
        }, nb::arg("data"),
           "Deserialize tree from binary format");

    nb::class_<SuffixForest>(m, "SuffixForest")
        .def(nb::init<int, int, int>(),
             nb::arg("max_depth"),
             nb::arg("num_threads") = -1,
             nb::arg("parallel_threshold") = 4,
             "Create a SuffixForest with configurable parallelization settings.\n\n"
             "Args:\n"
             "    max_depth: Maximum depth for all trees in the forest\n"
             "    num_threads: Number of threads (-1=auto, 0=sequential, >0=explicit)\n"
             "    parallel_threshold: Minimum batch size to trigger parallelization")
        .def("create_tree", &SuffixForest::create_tree,
             "Create a new tree in the forest and return its index")
        .def("remove_tree", &SuffixForest::remove_tree,
             nb::arg("tree_index"),
             "Remove a tree from the forest")
        .def("get_tree", nb::overload_cast<int>(&SuffixForest::get_tree),
             nb::arg("tree_index"),
             nb::rv_policy::reference_internal,
             "Get a reference to a tree in the forest")
        .def("has_tree", &SuffixForest::has_tree,
             nb::arg("tree_index"),
             "Check if a tree exists in the forest")
        .def("num_trees", &SuffixForest::num_trees,
             "Get the number of trees currently in the forest")
        .def("get_num_threads", &SuffixForest::get_num_threads,
             "Get the configured number of threads")
        .def("get_parallel_threshold", &SuffixForest::get_parallel_threshold,
             "Get the parallel threshold")
        // Batch extend methods
        .def("batch_extend", &batch_extend_forest,
             nb::arg("tree_indices"),
             nb::arg("seq_ids"),
             nb::arg("token_batches"),
             "Batch extend multiple trees with new tokens in parallel")
        .def("batch_extend_ndarray", &batch_extend_forest_ndarray,
             nb::arg("tree_indices"),
             nb::arg("seq_ids"),
             nb::arg("token_batches"),
             "Batch extend multiple trees with new tokens (ndarray version)")
        // Batch speculation methods
        .def("batch_speculate", &batch_speculate_forest,
             nb::arg("tree_indices"),
             nb::arg("contexts"),
             nb::arg("max_spec_tokens"),
             nb::arg("max_spec_factor"),
             nb::arg("max_spec_offset"),
             nb::arg("min_token_prob"),
             nb::arg("use_tree_spec"),
             "Perform batched parallel speculation across multiple trees")
        .def("batch_speculate_ndarray", &batch_speculate_forest_ndarray,
             nb::arg("tree_indices"),
             nb::arg("contexts"),
             nb::arg("max_spec_tokens"),
             nb::arg("max_spec_factor"),
             nb::arg("max_spec_offset"),
             nb::arg("min_token_prob"),
             nb::arg("use_tree_spec"),
             "Perform batched parallel speculation with ndarray inputs")
        .def("propose_from_batch", &propose_from_batch_forest,
             nb::arg("token_ids_cpu"),
             nb::arg("num_tokens"),
             nb::arg("tree_indices"),
             nb::arg("max_tree_depth"),
             nb::arg("max_spec_tokens"),
             nb::arg("max_spec_factor"),
             nb::arg("max_spec_offset"),
             nb::arg("min_token_prob"),
             nb::arg("use_tree_spec"),
             "Optimized batch speculation with direct InputBatch array consumption.\n\n"
             "Eliminates Python loop overhead by:\n"
             "1. Directly accessing token_ids_cpu via pointer arithmetic (zero-copy)\n"
             "2. Extracting contexts in parallel with OpenMP\n"
             "3. Performing speculation in the same parallel region\n\n"
             "Args:\n"
             "    token_ids_cpu: 2D array [batch_size, max_seq_len] of token IDs (int32, C-contiguous)\n"
             "    num_tokens: 1D array [batch_size] of valid token counts per request\n"
             "    tree_indices: 1D array [batch_size] of tree indices for each request\n"
             "    max_tree_depth: Maximum context depth to extract from token_ids_cpu\n"
             "    max_spec_tokens: Maximum number of tokens to speculate per request\n"
             "    max_spec_factor: Factor for speculation length calculation\n"
             "    max_spec_offset: Offset for speculation limit\n"
             "    min_token_prob: Minimum probability threshold for tokens\n"
             "    use_tree_spec: Whether to use tree speculation (vs path speculation)\n\n"
             "Returns:\n"
             "    List of Draft objects, one per batch item")
        .def("get_tree_indices", &SuffixForest::get_tree_indices,
             "Get all tree indices in the forest (sorted)")
        .def("estimate_forest_memory", &SuffixForest::estimate_forest_memory,
             "Estimate total memory usage across all trees in bytes")
        .def("estimate_trees_memory", &SuffixForest::estimate_trees_memory,
             "Get per-tree memory estimates as list of (tree_idx, bytes) pairs")
        // Selective snapshot creation - serialize only specific trees
        .def("create_selective_snapshot", [](const SuffixForest& forest,
                                             const std::vector<int>& tree_indices) {
            // Release GIL during serialization
            nb::gil_scoped_release release;

            auto cpp_results = forest.create_selective_snapshot(tree_indices);

            // Re-acquire GIL for Python object creation
            nb::gil_scoped_acquire acquire;

            // Convert to Python-compatible format
            std::vector<std::pair<int, nb::bytes>> py_results;
            py_results.reserve(cpp_results.size());

            for (const auto& [idx, data] : cpp_results) {
                py_results.emplace_back(
                    idx,
                    nb::bytes(reinterpret_cast<const char*>(data.data()), data.size())
                );
            }
            return py_results;
        },
        nb::arg("tree_indices"),
        "Create snapshots for specific tree indices only.\n\n"
        "Args:\n"
        "    tree_indices: List of tree indices to serialize\n\n"
        "Returns:\n"
        "    List of (tree_idx, snapshot_bytes) tuples for found trees.\n"
        "    Missing indices are silently skipped (partial results).")
        // Thread-safe single-tree extend (acquires write lock internally)
        .def("extend_tree", [](SuffixForest& forest, int tree_idx, int seq_id,
                               const std::vector<int32_t>& tokens) {
            auto lock = forest.get_write_lock(tree_idx);
            forest.get_tree(tree_idx).extend(seq_id, std::span<const int32_t>(tokens));
        },
        nb::arg("tree_index"),
        nb::arg("seq_id"),
        nb::arg("tokens"),
        "Thread-safe extend for a single tree (acquires write lock internally)")
        // Thread-safe single-tree extend with ndarray
        .def("extend_tree_ndarray", [](SuffixForest& forest, int tree_idx, int seq_id,
                                       const Int32Array1D& tokens) {
            auto lock = forest.get_write_lock(tree_idx);
            forest.get_tree(tree_idx).extend(seq_id, std::span<const int32_t>(tokens.data(), tokens.size()));
        },
        nb::arg("tree_index"),
        nb::arg("seq_id"),
        nb::arg("tokens"),
        "Thread-safe extend for a single tree with ndarray (acquires write lock internally)")
        // Thread-safe single-tree speculation (acquires read lock internally)
        .def("speculate_tree", [](SuffixForest& forest, int tree_idx,
                                  const std::vector<int32_t>& context,
                                  int max_spec_tokens,
                                  float max_spec_factor,
                                  float max_spec_offset,
                                  float min_token_prob,
                                  bool use_tree_spec) {
            auto lock = forest.get_read_lock(tree_idx);
            return forest.get_tree(tree_idx).speculate(
                std::span<const int32_t>(context),
                max_spec_tokens, max_spec_factor, max_spec_offset,
                min_token_prob, use_tree_spec);
        },
        nb::arg("tree_index"),
        nb::arg("context"),
        nb::arg("max_spec_tokens"),
        nb::arg("max_spec_factor"),
        nb::arg("max_spec_offset"),
        nb::arg("min_token_prob"),
        nb::arg("use_tree_spec"),
        "Thread-safe speculation for a single tree (acquires read lock internally)")
        // Thread-safe single-tree speculation with ndarray
        .def("speculate_tree_ndarray", [](SuffixForest& forest, int tree_idx,
                                          const Int32Array1D& context,
                                          int max_spec_tokens,
                                          float max_spec_factor,
                                          float max_spec_offset,
                                          float min_token_prob,
                                          bool use_tree_spec) {
            auto lock = forest.get_read_lock(tree_idx);
            return forest.get_tree(tree_idx).speculate(
                std::span<const int32_t>(context.data(), context.size()),
                max_spec_tokens, max_spec_factor, max_spec_offset,
                min_token_prob, use_tree_spec);
        },
        nb::arg("tree_index"),
        nb::arg("context"),
        nb::arg("max_spec_tokens"),
        nb::arg("max_spec_factor"),
        nb::arg("max_spec_offset"),
        nb::arg("min_token_prob"),
        nb::arg("use_tree_spec"),
        "Thread-safe speculation for a single tree with ndarray (acquires read lock internally)")
        .def_static("from_snapshots", [](
            int max_depth,
            const std::vector<std::pair<int, nb::bytes>>& tree_snapshots,
            int num_threads,
            int parallel_threshold) {
            // Convert Python bytes to C++ spans
            // We need to keep the data alive, so we build a vector of pairs with spans
            std::vector<std::pair<int, std::span<const uint8_t>>> cpp_snapshots;
            cpp_snapshots.reserve(tree_snapshots.size());
            for (const auto& [idx, data] : tree_snapshots) {
                cpp_snapshots.emplace_back(
                    idx,
                    std::span<const uint8_t>(
                        reinterpret_cast<const uint8_t*>(data.data()),
                        data.size()
                    )
                );
            }
            return SuffixForest(max_depth, cpp_snapshots, num_threads, parallel_threshold);
        },
        nb::arg("max_depth"),
        nb::arg("tree_snapshots"),
        nb::arg("num_threads") = -1,
        nb::arg("parallel_threshold") = 4,
        "Create a SuffixForest from tree snapshots.\n\n"
        "Args:\n"
        "    max_depth: Maximum depth for all trees in the forest\n"
        "    tree_snapshots: List of (tree_index, snapshot_bytes) tuples\n"
        "    num_threads: Number of threads (-1=auto, 0=sequential, >0=explicit)\n"
        "    parallel_threshold: Minimum batch size to trigger parallelization");
}
