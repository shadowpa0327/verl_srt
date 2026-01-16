// Copyright 2025 Snowflake Ltd. and/or its affiliates
// Copyright 2025 Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cassert>
#include <deque>
#include <memory>
#include <queue>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/deque.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <vector>
#include <unordered_map>
#include <map>

using namespace boost::interprocess;

// Forward declarations
struct Node;
struct LocalNode;
class SuffixTree;
class LocalSuffixTree;

// Void allocator for shared memory
using ShmemAllocator = allocator<void, managed_shared_memory::segment_manager>;

// Pointer to a Node in shared memory
using NodePtr = offset_ptr<Node>;

// Map from token to child Node
using ChildrenMapAllocator = allocator<std::pair<const int, NodePtr>, managed_shared_memory::segment_manager>;
using ChildrenMap = map<int, NodePtr, std::less<int>, ChildrenMapAllocator>;

// Local tree node structure for building in regular memory
struct LocalNode {
    int count;
    LocalNode* parent;  // Using raw pointers for simplicity
    std::unordered_map<int, std::unique_ptr<LocalNode>> children;
    int seq_id;
    int start;
    int length;
    LocalNode* suffix_link;

    LocalNode() : count(0), parent(nullptr), seq_id(-1), start(0), length(0), suffix_link(nullptr) {}
};

// Shared memory Node structure
struct Node {
    // Number of suffixes from the root that end at or pass through this node.
    int count;

    // Parent node.
    NodePtr parent;

    // Children nodes, the key should always be the first token of the child.
    ChildrenMap children;

    // ID of a "reference" sequence that contains the tokens in this node.
    int seq_id;

    // Start index of this node's tokens in the reference sequence.
    int start;

    // Number of tokens in this node (for leaf nodes, -1 means extends to end of sequence).
    int length;

    // Suffix link for Ukkonen's algorithm
    NodePtr suffix_link;

    Node(const ChildrenMap::allocator_type& alloc) : count(0), parent(nullptr), children(alloc), seq_id(-1), start(0), length(0), suffix_link(nullptr) {}
};

struct Candidate {
    // The token ids of the speculation candidate.
    std::vector<int> token_ids;

    // For each token, the index of its parent token (-1 if no parent).
    std::vector<int> parents;

    // For each token, the estimated probability of the token.
    std::vector<float> probs;

    // Floating point score of the candidate (sum of all probs).
    float score = 0.0;

    // Length of the prefix match for the speculated tokens.
    int match_len = 0;
};

// vector<int> for sequences
using IntVectorAllocator = allocator<int, managed_shared_memory::segment_manager>;
using IntVector = vector<int, IntVectorAllocator>;

// map<int, IntVector> for _seqs
using IntVectorMapAllocator = allocator<std::pair<const int, IntVector>, managed_shared_memory::segment_manager>;
using SeqMap = map<int, IntVector, std::less<int>, IntVectorMapAllocator>;

// Ukkonen algorithm state for each sequence (local version)
struct LocalUkkonenState {
    LocalNode* active_node;
    int active_edge;      // -1 if no active edge
    int active_length;
    int remaining_suffixes;

    LocalUkkonenState() : active_node(nullptr), active_edge(-1), active_length(0), remaining_suffixes(0) {}
};

// Ukkonen algorithm state for each sequence (shared memory version)
struct UkkonenState {
    NodePtr active_node;
    int active_edge;      // -1 if no active edge
    int active_length;
    int remaining_suffixes;

    // Performance optimization: cache frequently accessed data
    int last_pos;         // Last processed position
    NodePtr last_node;    // Last created node for suffix link optimization

    UkkonenState() : active_node(nullptr), active_edge(-1), active_length(0),
                     remaining_suffixes(0), last_pos(-1), last_node(nullptr) {}
};

using UkkonenStateAllocator = allocator<std::pair<const int, UkkonenState>, managed_shared_memory::segment_manager>;
using UkkonenStateMap = map<int, UkkonenState, std::less<int>, UkkonenStateAllocator>;

// Local suffix tree class for building in regular memory
class LocalSuffixTree {
public:
    LocalSuffixTree() : _root(nullptr) {}

    void extend(int seq_id, const std::vector<int>& tokens);
    LocalNode* get_root() const { return _root.get(); }
    const std::map<int, std::vector<int>>& get_seqs() const { return _seqs; }

private:
    std::unique_ptr<LocalNode> _root;
    std::map<int, std::vector<int>> _seqs;
    std::map<int, LocalUkkonenState> _states;

    void _extend_tree_local(int seq_id, int pos);
    LocalNode* _create_leaf_node_local(int seq_id, int start_pos, LocalNode* parent);
    LocalNode* _create_internal_node_local(LocalNode* parent, int seq_id, int start, int length);
    LocalNode* _split_edge_local(LocalNode* node, int split_pos);
    int _get_edge_length_local(LocalNode* node, int seq_id, int pos);
    bool _walk_down_local(LocalNode* node, LocalUkkonenState& state, int seq_id, int pos);
    int _update_node_counts_local(LocalNode* node);
};

class SuffixTree {
public:

    SuffixTree(const ShmemAllocator& alloc);
    ~SuffixTree();

    int num_seqs() const {
        return static_cast<int>(_seqs.size());
    }


    // Append multiple new elements to the sequence with id seq_id.
    void extend(int seq_id, const std::vector<int>& tokens);

    // Build tree from local tree (new method)
    void extend_from_local(int seq_id, const std::vector<int>& tokens);

    Candidate speculate(const std::vector<int>& pattern,
                        int max_spec_tokens,
                        float min_token_prob = 0.1f,
                        bool use_tree_spec = false);

private:
    ShmemAllocator _alloc;

    // The root node of the suffix tree.
    NodePtr _root;

    // Mapping from seq id to its sequence (vector of ints).
    SeqMap _seqs;

    // For each sequence, Ukkonen algorithm state
    UkkonenStateMap _ukkonen_states;

    // Bulk memory allocation for one-time extend trees
    struct BulkMemory {
        void* memory_block = nullptr;
        size_t total_size = 0;
        size_t used_size = 0;
        size_t node_count = 0;
        managed_shared_memory::segment_manager* segment_manager = nullptr;
    } _bulk_memory;

    std::pair<NodePtr, int> _match_pattern(const std::vector<int>& pattern,
                                         int start_idx = 0);

    Candidate _speculate_path(NodePtr node, int idx, int max_spec_tokens,
                              float min_token_prob);

    Candidate _speculate_tree(NodePtr node, int idx, int max_spec_tokens,
                              float min_token_prob);

    // Bulk memory management for one-time extend trees
    void _allocate_bulk_memory(size_t estimated_nodes);
    NodePtr _create_node_from_bulk(const ChildrenMap::allocator_type& alloc);
    void _destroy_bulk_memory();

    // Copy local tree to shared memory
    NodePtr _copy_local_tree_to_shared(LocalNode* local_node,
                                       const std::map<int, std::vector<int>>& local_seqs,
                                       std::unordered_map<LocalNode*, NodePtr>& node_map);

    // Ukkonen algorithm helper methods
    int _get_edge_length(NodePtr node, int seq_id, int pos);
    bool _walk_down(NodePtr node, UkkonenState& state, int seq_id, int pos);
    bool _walk_down_optimized(NodePtr node, UkkonenState& state, int edge_length);
    NodePtr _split_edge(NodePtr node, int split_pos, int seq_id, int pos);
    void _extend_tree(int seq_id, int pos);

    // Performance optimization: batch create nodes
    NodePtr _create_leaf_node(int seq_id, int start_pos, NodePtr parent);
    NodePtr _create_internal_node(NodePtr parent, int seq_id, int start, int length);

    // Update node counts after tree construction
    int _update_node_counts(NodePtr node, int seq_id);

    // Performance optimizations: cache frequently accessed data
    inline int _get_edge_length_fast(NodePtr node, int pos) {
        return (node->length == -1) ? pos - node->start + 1 : node->length;
    }

    // Optimized edge character access with bounds checking
    inline int _get_edge_char_safe(NodePtr node, const IntVector& seq, int offset) {
        int pos = node->start + offset;
        if (node->length == -1) {
            return (pos < static_cast<int>(seq.size())) ? seq[pos] : -1;
        } else {
            return (offset < node->length) ? seq[pos] : -1;
        }
    }

    void _destroy_tree_optimized();
};
