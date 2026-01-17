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

#include <cassert>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <memory>
#include "suffix_tree.h"
#include <algorithm>

SuffixTree::SuffixTree(const ShmemAllocator& alloc)
    : _alloc(alloc),
      _seqs(std::less<int>(), alloc),
      _ukkonen_states(std::less<int>(), alloc),
      _root(nullptr) {
    // Root node will be created lazily in extend() to use bulk allocation
}

SuffixTree::~SuffixTree() {
    _destroy_bulk_memory();
}

void SuffixTree::_destroy_bulk_memory() {
    if (_bulk_memory.memory_block && _bulk_memory.segment_manager) {
        // Ultra-fast destruction: since all nodes (including root) are in bulk memory
        // and this is one-time use, just deallocate the entire memory block
        // The shared memory manager handles the cleanup
        _bulk_memory.segment_manager->deallocate(_bulk_memory.memory_block);
        _bulk_memory.memory_block = nullptr;
        _root = nullptr;
    }
}

// Performance optimization: batch extend for multiple tokens
void SuffixTree::extend(int seq_id, const std::vector<int>& tokens) {
    if (tokens.empty()) return;

    // For one-time extend: pre-allocate bulk memory for all nodes
    // Suffix tree node count analysis:
    // - Theoretical maximum: 2n-1 nodes for sequence of length n
    // - n leaf nodes (one per suffix)
    // - up to n-1 internal nodes (including root)
    // - In practice: depends on repetition patterns in the sequence
    //   * High repetition -> fewer internal nodes (shared prefixes)
    //   * Random sequence -> closer to theoretical maximum
    // Using 2n as conservative estimate to handle worst case + some buffer
    size_t n = tokens.size();
    size_t estimated_nodes = 2 * n + 30; // Theoretical max + small buffer
    _allocate_bulk_memory(estimated_nodes);

    // Create root node lazily using bulk memory
    if (!_root) {
        auto s_manager = _alloc.get_segment_manager();
        ChildrenMapAllocator children_map_alloc(s_manager);
        _root = _create_node_from_bulk(children_map_alloc);
        _root->suffix_link = _root;  // Root's suffix link points to itself
    }

    // Pre-allocate space for the sequence to avoid repeated reallocations
    auto s_manager = _alloc.get_segment_manager();
    _seqs.try_emplace(seq_id, IntVector(s_manager));
    _ukkonen_states.try_emplace(seq_id, UkkonenState());

    auto& seq = _seqs.at(seq_id);
    auto& state = _ukkonen_states.at(seq_id);

    // Initialize state for new sequence
    if (seq.empty()) {
        state.active_node = _root;
        state.active_edge = -1;
        state.active_length = 0;
        state.remaining_suffixes = 0;
    }

    // Reserve space to avoid reallocations
    size_t old_size = seq.size();
    seq.reserve(old_size + tokens.size());

    // Batch insert tokens
    for (int token : tokens) {
        seq.push_back(token);
        int pos = static_cast<int>(seq.size()) - 1;
        _extend_tree(seq_id, pos);
    }

    // Update counts for all nodes after construction
    if (!tokens.empty()) {
        _update_node_counts(_root, seq_id);
    }
}

// New method: Build tree from local tree and copy to shared memory
void SuffixTree::extend_from_local(int seq_id, const std::vector<int>& tokens) {
    if (tokens.empty()) return;

    // Step 1: Build tree in local memory
    LocalSuffixTree local_tree;
    local_tree.extend(seq_id, tokens);

    // Step 2: Estimate nodes and allocate shared memory
    size_t n = tokens.size();
    size_t estimated_nodes = 2 * n + 30;
    _allocate_bulk_memory(estimated_nodes);

    // Step 3: Copy local tree to shared memory
    auto s_manager = _alloc.get_segment_manager();
    _seqs.try_emplace(seq_id, IntVector(s_manager));
    auto& seq = _seqs.at(seq_id);
    seq.reserve(tokens.size());
    for (int token : tokens) {
        seq.push_back(token);
    }

    // Step 4: Copy tree structure
    std::unordered_map<LocalNode*, NodePtr> node_map;
    if (local_tree.get_root()) {
        _root = _copy_local_tree_to_shared(local_tree.get_root(), local_tree.get_seqs(), node_map);
        _root->suffix_link = _root;  // Root's suffix link points to itself
    }
}

Candidate SuffixTree::speculate(const std::vector<int>& pattern,
                                int max_spec_tokens,
                                float min_token_prob,
                                bool use_tree_spec) {
    Candidate result;
    // Try all starting points in the pattern
    for (int start_idx = 0; start_idx < static_cast<int>(pattern.size()) - 3; start_idx++) {
        auto[node, idx] = _match_pattern(pattern, start_idx);
        if (node == nullptr) {
            continue;
        }
        // int match_len = static_cast<int>(pattern.size()) - start_idx;
        Candidate candidate;
        if (use_tree_spec) {
            candidate = _speculate_tree(node, idx, max_spec_tokens, min_token_prob);
        } else {
            candidate = _speculate_path(node, idx, max_spec_tokens, min_token_prob);
        }
        if (candidate.score)
            return candidate;  // Early return if we found a valid candidate
    }
    return result;
}

std::pair<NodePtr, int> SuffixTree::_match_pattern(
        const std::vector<int>& pattern, int start_idx) {
    NodePtr current_node = _root;
    int edge_idx = 0;

    for (size_t i = start_idx; i < pattern.size(); i++) {
        int c = pattern[i];

        // If we're at the root or at the end of an edge, move to a child
        while (true) {
            // Handle root node case
            if (current_node == _root) {
                auto it = current_node->children.find(c);
                if (it == current_node->children.end()) {
                    return {nullptr, -1};
                }
                current_node = it->second;
                edge_idx = 0;
                break;
            }

            // Get edge length for current node
            int edge_length;
            if (current_node->length == -1) {
                // Leaf node - edge extends to sequence end
                edge_length = static_cast<int>(_seqs.at(current_node->seq_id).size()) - current_node->start;
            } else {
                // Internal node - use specified length
                edge_length = current_node->length;
            }

            // If we're at the end of current edge, move to child
            if (edge_idx >= edge_length) {
                auto it = current_node->children.find(c);
                if (it == current_node->children.end()) {
                    return {nullptr, -1};
                }
                current_node = it->second;
                edge_idx = 0;
                continue; // Check again with new node
            }

            // We're in the middle of an edge
            break;
        }

        // Check character match at current position on edge
        auto& seq = _seqs.at(current_node->seq_id);
        int edge_pos = current_node->start + edge_idx;

        // Bounds checking
        if (current_node->length == -1) {
            // Leaf node
            if (edge_pos >= static_cast<int>(seq.size())) {
                return {nullptr, -1};
            }
        } else {
            // Internal node
            if (edge_idx >= current_node->length) {
                return {nullptr, -1};
            }
        }

        int edge_char = seq[edge_pos];
        if (edge_char != c) {
            return {nullptr, -1};
        }

        edge_idx++;
    }

    return {current_node, edge_idx};
}

Candidate SuffixTree::_speculate_path(NodePtr node, int idx,
                                      int max_spec_tokens,
                                      float min_token_prob) {
    Candidate ret;
    float prob = 1.0f;
    while (ret.token_ids.size() < static_cast<size_t>(max_spec_tokens) && prob >= min_token_prob) {
        // Get actual edge length (handle leaf nodes with length = -1)
        int edge_length = (node->length == -1) ?
            static_cast<int>(_seqs.at(node->seq_id).size()) - node->start : node->length;

        if (idx < edge_length) {
            auto& seq = _seqs.at(node->seq_id);
            int token = seq[node->start + idx];
            if(token == -1) {
                break;
            }
            // Use previous token index as parent; if none, mark as -1.
            ret.parents.push_back(static_cast<int>(ret.token_ids.size()) - 1);
            ret.token_ids.push_back(token);
            ret.probs.push_back(prob);
            ret.score += prob;
            idx++;
        } else {
            NodePtr child = nullptr;
            int count = 0;
            // Choose the child with the maximum count.
            for (auto& kv : node->children) {
                NodePtr ch = kv.second;
                if (ch->count > count) {
                    child = ch;
                    count = ch->count;
                }
            }
            if (child == nullptr) {
                break;
            }
            prob *= static_cast<float>(count) / node->count;
            node = child;
            idx = 0;
        }
    }
    return ret;
}

struct HeapItem {
    float prob;
    NodePtr node;
    int idx;
    int parent;   // index in the candidate token list; -1 if none.

    HeapItem(float p, NodePtr n, int i, int par)
        : prob(p), node(n), idx(i), parent(par) {}
};

struct HeapItemCompare {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        // In C++ priority_queue by default returns the largest element.
        // Thus, we compare probabilities so that the highest prob is returned.
        return a.prob < b.prob;
    }
};

// Get a candidate token tree using a priority queue.
Candidate SuffixTree::_speculate_tree(NodePtr node, int idx,
                                      int max_spec_tokens,
                                      float min_token_prob) {
    Candidate ret;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapItemCompare> queue;
    queue.emplace(1.0, node, idx, -1);
    while (ret.token_ids.size() < static_cast<size_t>(max_spec_tokens) && !queue.empty()) {
        HeapItem item = queue.top();
        queue.pop();

        // Get actual edge length (handle leaf nodes with length = -1)
        int edge_length = (item.node->length == -1) ?
            static_cast<int>(_seqs.at(item.node->seq_id).size()) - item.node->start : item.node->length;

        if (item.idx < edge_length) {
            auto& seq = _seqs.at(item.node->seq_id);
            int token = seq[item.node->start + item.idx];
            ret.token_ids.push_back(token);
            ret.parents.push_back(item.parent);
            ret.probs.push_back(item.prob);
            ret.score += item.prob;
            queue.emplace(item.prob, item.node, item.idx + 1,
                          static_cast<int>(ret.token_ids.size()) - 1);
        } else {
            for (auto& kv : item.node->children) {
                NodePtr child = kv.second;
                float prob = item.prob * child->count /
                    static_cast<float>(item.node->count);
                if (prob >= min_token_prob) {
                    queue.emplace(prob, child, 0, item.parent);
                }
            }
        }
    }
    return ret;
}

// Ukkonen algorithm helper methods
int SuffixTree::_get_edge_length(NodePtr node, int seq_id, int pos) {
    if (node == _root) return 0;

    // For leaf nodes, length extends to current position in the current sequence being processed
    if (node->length == -1) {
        // For multi-sequence trees, we need to ensure this calculation is correct
        // The pos parameter refers to the current position in seq_id being processed
        if (node->seq_id == seq_id) {
            // Same sequence - normal calculation
            return pos - node->start + 1;
        } else {
            // Different sequence - use the full edge length to the end of node's sequence
            auto& node_seq = _seqs.at(node->seq_id);
            return static_cast<int>(node_seq.size()) - node->start;
        }
    }
    return node->length;
}

bool SuffixTree::_walk_down(NodePtr node, UkkonenState& state, int seq_id, int pos) {
    if (node == _root) return false;

    int edge_length = _get_edge_length(node, seq_id, pos);
    if (state.active_length >= edge_length) {
        state.active_edge += edge_length;
        state.active_length -= edge_length;
        state.active_node = node;
        return true;
    }
    return false;
}

// Optimized version that avoids repeated edge length calculations
bool SuffixTree::_walk_down_optimized(NodePtr node, UkkonenState& state, int edge_length) {
    if (node == _root) return false;

    if (state.active_length >= edge_length) {
        state.active_edge += edge_length;
        state.active_length -= edge_length;
        state.active_node = node;
        return true;
    }
    return false;
}

NodePtr SuffixTree::_split_edge(NodePtr node, int split_pos, int seq_id, int pos) {
    (void)seq_id;  // Suppress unused parameter warning
    (void)pos;     // Suppress unused parameter warning

    auto s_manager = _alloc.get_segment_manager();
    ChildrenMapAllocator children_map_alloc(s_manager);

    // Create new internal node
    NodePtr split_node = _create_node_from_bulk(children_map_alloc);
    split_node->parent = node->parent;
    split_node->seq_id = node->seq_id;
    split_node->start = node->start;
    split_node->length = split_pos;
    // split_node->count = 1;  // Will be updated by caller
    split_node->suffix_link = nullptr;

    // Update parent's child pointer to point to split node
    auto& split_seq = _seqs.at(split_node->seq_id);
    int first_char = split_seq[split_node->start];
    if (split_node->parent != nullptr) {
        split_node->parent->children[first_char] = split_node;
    }

    // Update original node
    node->parent = split_node;
    node->start += split_pos;
    if (node->length != -1) {
        node->length -= split_pos;
    }

    // Add original node as child of split node
    auto& node_seq = _seqs.at(node->seq_id);
    int split_char = node_seq[node->start];
    split_node->children[split_char] = node;

    return split_node;
}

NodePtr SuffixTree::_create_leaf_node(int seq_id, int start_pos, NodePtr parent) {
    auto s_manager = _alloc.get_segment_manager();
    ChildrenMapAllocator children_map_alloc(s_manager);

    NodePtr leaf = _create_node_from_bulk(children_map_alloc);
    leaf->parent = parent;
    leaf->seq_id = seq_id;
    leaf->start = start_pos;
    leaf->length = -1;  // Leaf node extends to end
    leaf->count = 1;
    leaf->suffix_link = nullptr;

    return leaf;
}

NodePtr SuffixTree::_create_internal_node(NodePtr parent, int seq_id, int start, int length) {
    auto s_manager = _alloc.get_segment_manager();
    ChildrenMapAllocator children_map_alloc(s_manager);

    NodePtr node = _create_node_from_bulk(children_map_alloc);
    node->parent = parent;
    node->seq_id = seq_id;
    node->start = start;
    node->length = length;
    node->count = 1;
    node->suffix_link = nullptr;

    return node;
}

void SuffixTree::_extend_tree(int seq_id, int pos) {
    auto& seq = _seqs.at(seq_id);
    auto& state = _ukkonen_states.at(seq_id);

    int current_char = seq[pos];
    NodePtr last_new_node = nullptr;
    state.remaining_suffixes++;

    while (state.remaining_suffixes > 0) {
        if (state.active_length == 0) {
            state.active_edge = pos;
        }

        // Determine which character to look for
        int search_char = (state.active_length == 0) ? current_char : seq[state.active_edge];

        // Find child node
        auto it = state.active_node->children.find(search_char);
        NodePtr child = (it != state.active_node->children.end()) ? it->second : nullptr;

        // If no child with this character exists
        if (child == nullptr) {
            // Create new leaf node
            NodePtr leaf = _create_leaf_node(seq_id, pos, state.active_node);
            state.active_node->children[current_char] = leaf;

            // Update count for this node
            // state.active_node->count++;

            // Set suffix link
            if (last_new_node != nullptr) {
                last_new_node->suffix_link = state.active_node;
                last_new_node = nullptr;
            }
        } else {
            // Child exists, check if we need to walk down
            int edge_length = _get_edge_length(child, seq_id, pos);

            // Walk down if active_length is too long
            if (state.active_length >= edge_length) {
                state.active_edge += edge_length;
                state.active_length -= edge_length;
                state.active_node = child;
                continue; // Continue the loop to process with new active point
            }

            // Check character match on the edge
            int edge_char_pos = child->start + state.active_length;

            // Use the correct sequence for this node (child may be from a different sequence)
            auto& child_seq = _seqs.at(child->seq_id);

            // Validate edge position bounds
            if (child->length == -1) {
                // Leaf node: check against sequence length
                if (edge_char_pos >= static_cast<int>(child_seq.size())) {
                    // Should not happen in correct implementation
                    break;
                }
            } else {
                // Internal node: check against edge length
                if (state.active_length >= child->length) {
                    // Should not happen due to walk_down logic above
                    break;
                }
            }

            int edge_char = child_seq[edge_char_pos];

                         if (edge_char == current_char) {
                 // Character matches - increment active_length
                 state.active_length++;

                 // Set suffix link if needed
                 if (last_new_node != nullptr && state.active_node != _root) {
                     last_new_node->suffix_link = state.active_node;
                     last_new_node = nullptr;
                 }

                 // Showstopper rule - no more suffixes to add for this extension
                 break;
            } else {
                // Character doesn't match - split the edge
                NodePtr split_node = _split_edge(child, state.active_length, seq_id, pos);

                // Create new leaf from split point
                NodePtr leaf = _create_leaf_node(seq_id, pos, split_node);
                split_node->children[current_char] = leaf;

                // Update counts
                // split_node->count++;

                // Set suffix link
                if (last_new_node != nullptr) {
                    last_new_node->suffix_link = split_node;
                }
                last_new_node = split_node;
            }
        }

        state.remaining_suffixes--;

        // Move to next suffix
        if (state.active_node == _root && state.active_length > 0) {
            state.active_length--;
            state.active_edge = pos - state.remaining_suffixes + 1;
        } else if (state.active_node != _root) {
            state.active_node = state.active_node->suffix_link ? state.active_node->suffix_link : _root;
        }
    }

    // Set final suffix link
    if (last_new_node != nullptr) {
        last_new_node->suffix_link = _root;
    }
}

// Private helper method to update node counts after tree construction
int SuffixTree::_update_node_counts(NodePtr node, int seq_id) {
    (void)seq_id;  // Suppress unused parameter warning
    if (node == nullptr) return 0;

    int total_count = 0;

    // If this is a leaf node, it represents one suffix
    if (node->children.empty()) {
        node->count = 1;
        return 1;
    }

    // For internal nodes, count is sum of children counts
    for (auto& kv : node->children) {
        total_count += _update_node_counts(kv.second, seq_id);
    }

    node->count = total_count;
    return total_count;
}

void SuffixTree::_allocate_bulk_memory(size_t estimated_nodes) {
    if (_bulk_memory.memory_block) return; // Already allocated

    auto s_manager = _alloc.get_segment_manager();
    _bulk_memory.segment_manager = s_manager;

    // Estimate memory needed: nodes + some extra space
    // Each Node contains children map which needs extra space
    size_t node_size = sizeof(Node);
    size_t extra_space_per_node = 128; // For children map overhead
    _bulk_memory.total_size = estimated_nodes * (node_size + extra_space_per_node);

    // Allocate bulk memory block
    _bulk_memory.memory_block = s_manager->allocate(_bulk_memory.total_size);
    _bulk_memory.used_size = 0;
    _bulk_memory.node_count = 0;
}

NodePtr SuffixTree::_create_node_from_bulk(const ChildrenMap::allocator_type& alloc) {
    if (!_bulk_memory.memory_block) {
        // Fallback to regular allocation if bulk memory not available
        auto s_manager = _alloc.get_segment_manager();
        return s_manager->construct<Node>(anonymous_instance)(alloc);
    }

    // Calculate position in bulk memory
    char* memory_pos = static_cast<char*>(_bulk_memory.memory_block) + _bulk_memory.used_size;

    // Align memory to Node boundary
    size_t alignment = alignof(Node);
    size_t space = _bulk_memory.total_size - _bulk_memory.used_size;
    void* aligned_ptr = memory_pos;
    if (std::align(alignment, sizeof(Node), aligned_ptr, space)) {
        // Construct Node in-place
        Node* node = new (aligned_ptr) Node(alloc);
        _bulk_memory.used_size += (static_cast<char*>(aligned_ptr) + sizeof(Node)) - memory_pos;
        _bulk_memory.node_count++;

        // Return offset_ptr pointing to the node
        return NodePtr(node);
    }

    // Fallback if alignment fails
    auto s_manager = _alloc.get_segment_manager();
    return s_manager->construct<Node>(anonymous_instance)(alloc);
}

// Copy local tree to shared memory
NodePtr SuffixTree::_copy_local_tree_to_shared(LocalNode* local_node,
                                               const std::map<int, std::vector<int>>& local_seqs,
                                               std::unordered_map<LocalNode*, NodePtr>& node_map) {
    if (!local_node) return nullptr;

    // Check if already copied
    auto it = node_map.find(local_node);
    if (it != node_map.end()) {
        return it->second;
    }

    // Create shared memory node
    auto s_manager = _alloc.get_segment_manager();
    ChildrenMapAllocator children_map_alloc(s_manager);
    NodePtr shared_node = _create_node_from_bulk(children_map_alloc);

    // Copy node data
    shared_node->count = local_node->count;
    shared_node->seq_id = local_node->seq_id;
    shared_node->start = local_node->start;
    shared_node->length = local_node->length;

    // Add to mapping
    node_map[local_node] = shared_node;

    // Copy children (recursive)
    for (const auto& [token, child] : local_node->children) {
        NodePtr shared_child = _copy_local_tree_to_shared(child.get(), local_seqs, node_map);
        shared_node->children[token] = shared_child;
        shared_child->parent = shared_node;
    }

    // Set suffix links (need to do this after all nodes are created)
    // This is a simplified approach - in practice you might need a second pass
    if (local_node->suffix_link) {
        auto suffix_it = node_map.find(local_node->suffix_link);
        if (suffix_it != node_map.end()) {
            shared_node->suffix_link = suffix_it->second;
        }
    }

    return shared_node;
}

// Local suffix tree implementation
void LocalSuffixTree::extend(int seq_id, const std::vector<int>& tokens) {
    if (tokens.empty()) return;

    // Create root lazily
    if (!_root) {
        _root = std::make_unique<LocalNode>();
        _root->suffix_link = _root.get();
    }

    // Initialize sequence and state
    _seqs[seq_id] = tokens;
    auto& state = _states[seq_id];
    state.active_node = _root.get();
    state.active_edge = -1;
    state.active_length = 0;
    state.remaining_suffixes = 0;

    // Build tree using Ukkonen algorithm
    for (int pos = 0; pos < static_cast<int>(tokens.size()); pos++) {
        _extend_tree_local(seq_id, pos);
    }

    // Update counts
    if (!tokens.empty()) {
        _update_node_counts_local(_root.get());
    }
}

void LocalSuffixTree::_extend_tree_local(int seq_id, int pos) {
    auto& seq = _seqs[seq_id];
    auto& state = _states[seq_id];
    int c = seq[pos];

    state.remaining_suffixes++;
    LocalNode* last_new_node = nullptr;

    while (state.remaining_suffixes > 0) {
        if (state.active_length == 0) {
            state.active_edge = pos;
        }

        // Check if there's an outgoing edge with the current character
        auto it = state.active_node->children.find(seq[state.active_edge]);

        if (it == state.active_node->children.end()) {
            // No edge exists, create a new leaf node
            auto new_leaf = _create_leaf_node_local(seq_id, pos, state.active_node);
            state.active_node->children[c] = std::unique_ptr<LocalNode>(new_leaf);

            if (last_new_node) {
                last_new_node->suffix_link = state.active_node;
            }
            last_new_node = state.active_node;
        } else {
            LocalNode* next = it->second.get();

            if (_walk_down_local(next, state, seq_id, pos)) {
                continue;
            }

            // Check if current character already exists on the edge
            int edge_pos = next->start + state.active_length;
            if (seq[edge_pos] == c) {
                if (last_new_node && state.active_node != _root.get()) {
                    last_new_node->suffix_link = state.active_node;
                }
                state.active_length++;
                break;
            }

            // Split the edge
            LocalNode* split_node = _split_edge_local(next, state.active_length);
            auto new_leaf = _create_leaf_node_local(seq_id, pos, split_node);
            split_node->children[c] = std::unique_ptr<LocalNode>(new_leaf);

            if (last_new_node) {
                last_new_node->suffix_link = split_node;
            }
            last_new_node = split_node;
        }

        state.remaining_suffixes--;

        if (state.active_node == _root.get() && state.active_length > 0) {
            state.active_length--;
            state.active_edge = pos - state.remaining_suffixes + 1;
        } else if (state.active_node != _root.get()) {
            state.active_node = state.active_node->suffix_link;
        }
    }
}

LocalNode* LocalSuffixTree::_create_leaf_node_local(int seq_id, int start_pos, LocalNode* parent) {
    auto node = new LocalNode();
    node->seq_id = seq_id;
    node->start = start_pos;
    node->length = -1;  // Leaf node extends to end
    node->parent = parent;
    return node;
}

LocalNode* LocalSuffixTree::_create_internal_node_local(LocalNode* parent, int seq_id, int start, int length) {
    auto node = new LocalNode();
    node->seq_id = seq_id;
    node->start = start;
    node->length = length;
    node->parent = parent;
    return node;
}

LocalNode* LocalSuffixTree::_split_edge_local(LocalNode* node, int split_pos) {
    // Create new internal node
    LocalNode* split_node = _create_internal_node_local(node->parent, node->seq_id, node->start, split_pos);

    // Update original node
    node->start += split_pos;
    node->length = (node->length == -1) ? -1 : node->length - split_pos;
    node->parent = split_node;

    // Connect split node to parent
    if (split_node->parent) {
        int first_char = _seqs[node->seq_id][split_node->start];
        split_node->parent->children[first_char] = std::unique_ptr<LocalNode>(split_node);
    }

    // Connect original node to split node
    int first_char = _seqs[node->seq_id][node->start];
    split_node->children[first_char] = std::unique_ptr<LocalNode>(node);

    return split_node;
}

int LocalSuffixTree::_get_edge_length_local(LocalNode* node, int seq_id, int pos) {
    if (node->length == -1) {
        return pos - node->start + 1;
    }
    return node->length;
}

bool LocalSuffixTree::_walk_down_local(LocalNode* node, LocalUkkonenState& state, int seq_id, int pos) {
    int edge_length = _get_edge_length_local(node, seq_id, pos);
    if (state.active_length >= edge_length) {
        state.active_edge += edge_length;
        state.active_length -= edge_length;
        state.active_node = node;
        return true;
    }
    return false;
}

int LocalSuffixTree::_update_node_counts_local(LocalNode* node) {
    if (!node) return 0;

    int count = 0;
    if (node->children.empty()) {
        // Leaf node
        count = 1;
    } else {
        // Internal node
        for (const auto& [token, child] : node->children) {
            count += _update_node_counts_local(child.get());
        }
    }

    node->count = count;
    return count;
}
