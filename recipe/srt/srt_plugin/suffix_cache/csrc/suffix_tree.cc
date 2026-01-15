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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "suffix_tree.h"

#define CHECK_OR_RETURN(cond) \
    if (!(cond)) return "Integrity check failed at line " + \
                        std::to_string(__LINE__) + ": " + #cond;

SuffixTree::SuffixTree(int max_depth)
    : _max_depth(max_depth), _root(new Node()) {
}

void _remove_from_siblings(Node* node) {
    // Remove a node from the siblings and groups linked lists.
    assert(node->parent);  // Should only be called on non-root nodes.
    // Take care of the groups linked list.
    Group* group = node->group.get();
    if (group->head == node) {
        if (node->next_sibling && node->next_sibling->count == node->count) {
            // There are other nodes in the same group, update its head and
            // remove the node from the group.
            group->head = node->next_sibling;
            node->group.reset();
        } else {
            // Otherwise, the node is the only member of its group. Remove the
            // group together with the node.
            if (group->prev) {
                group->prev->next = group->next;
            }
            if (group->next) {
                group->next->prev = group->prev;
            }
            group->prev = group->next = nullptr;
        }
    } else {
        // The node is not the head of its group, just remove it.
        node->group.reset();
    }
    // Take care of the siblings linked list.
    if (node->next_sibling) {
        node->next_sibling->prev_sibling = node->prev_sibling;
    } else {
        node->parent->tail_child = node->prev_sibling;
    }
    if (node->prev_sibling) {
        node->prev_sibling->next_sibling = node->next_sibling;
    } else {
        node->parent->head_child = node->next_sibling;
    }
    node->prev_sibling = node->next_sibling = nullptr;
}

void _insert_into_siblings_before(Node* node, Node* other) {
    // Insert a node before another in the siblings and groups linked lists.
    assert(node->parent);  // Should only be called on non-root nodes.
    assert(node->parent == other->parent);  // Should be siblings.
    // Take care of the siblings linked list.
    if (other->prev_sibling) {
        other->prev_sibling->next_sibling = node;
    } else {
        node->parent->head_child = node;
    }
    node->next_sibling = other;
    node->prev_sibling = other->prev_sibling;
    other->prev_sibling = node;
    // Take care of the groups linked list.
    Node* prev_sibling = node->prev_sibling;
    if (prev_sibling && node->count == prev_sibling->count) {
        // If the previous sibling has the same count, just join its group.
        node->group = prev_sibling->group;  // std::shared_ptr assignment
    } else if (node->count == other->count) {
        // Previous sibling has different count, but next sibling has the same
        // count. Join as the head of the next sibling's group.
        node->group = other->group;  // std::shared_ptr assignment
        node->group->head = node;
    } else {
        // Previous and next siblings both have different counts. The node
        // belongs in a group by itself.
        Group* group = node->group.get();
        if (!group) {
            // The node does not come with a group, create a new one.
            group = new Group(node);
            node->group.reset(group);  // std::shared_ptr assignment
        }
        assert(group->head == node && !group->next && !group->prev);
        // Insert the node's group into the linked list.
        if (prev_sibling) {
            group->prev = prev_sibling->group.get();
            group->prev->next = group;
        }
        group->next = other->group.get();
        group->next->prev = group;
    }
}

void _insert_into_siblings_after(Node* node, Node* other) {
    // Insert a node after another in the siblings and groups linked lists.
    assert(node->parent);  // Should only be called on non-root nodes.
    assert(node->parent == other->parent);  // Should be siblings.
    // Take care of the siblings linked list.
    if (other->next_sibling) {
        other->next_sibling->prev_sibling = node;
    } else {
        node->parent->tail_child = node;
    }
    node->prev_sibling = other;
    node->next_sibling = other->next_sibling;
    other->next_sibling = node;
    // Take care of the groups linked list.
    Node* next_sibling = node->next_sibling;
    if (next_sibling && node->count == next_sibling->count) {
        // If the next sibling has the same count, join its group and maybe
        // update the head of the group.
        node->group = next_sibling->group;  // std::shared_ptr assignment
        if (node->group->head == next_sibling) {
            node->group->head = node;
        }
    } else if (node->count == other->count) {
        // Next sibling has different count, but previous sibling has the same
        // count. Join as the tail of the previous sibling's group.
        node->group = other->group;  // std::shared_ptr assignment
    } else {
        // Previous and next siblings both have different counts. The node
        // belongs in a group by itself.
        Group* group = node->group.get();
        if (!group) {
            // The node does not come with a group, create a new one.
            group = new Group(node);
            node->group.reset(group);  // std::shared_ptr assignment
        }
        assert(group->head == node && !group->next && !group->prev);
        // Insert the node's group into the linked list.
        if (next_sibling) {
            group->next = next_sibling->group.get();
            group->next->prev = group;
        }
        group->prev = other->group.get();
        group->prev->next = group;
    }
}

void _replace_in_siblings(Node* old_node, Node* new_node) {
    // Replace a node with another in the siblings and groups linked lists.
    assert(old_node->count == new_node->count);  // Should have the same count.
    assert(old_node->parent);  // Should only be called on non-root nodes.
    // Take care of the siblings linked list.
    if (old_node->next_sibling) {
        old_node->next_sibling->prev_sibling = new_node;
    } else {
        old_node->parent->tail_child = new_node;
    }
    if (old_node->prev_sibling) {
        old_node->prev_sibling->next_sibling = new_node;
    } else {
        old_node->parent->head_child = new_node;
    }
    new_node->prev_sibling = old_node->prev_sibling;
    new_node->next_sibling = old_node->next_sibling;
    old_node->prev_sibling = old_node->next_sibling = nullptr;
    // Take care of the groups linked list.
    Group* group = old_node->group.get();
    if (group->head == old_node) {
        group->head = new_node;
    }
    new_node->group = old_node->group;  // std::shared_ptr assignment
    old_node->group.reset();
}

void _increment_count(Node* node) {
    // Increment the count of a node by 1, and update its position in the
    // sibling and group linked lists if necessary.
    if (!node->parent) {
        // Root node has no siblings, update its count and return.
        node->count += 1;
        return;
    }
    if (!node->prev_sibling || node->prev_sibling->count > node->count + 1) {
        // The node does not need to move, and will not join the previous group
        // after its count is incremented.
        assert(node->group->head == node);
        if (!node->next_sibling || node->next_sibling->count < node->count) {
            // The node should be the only member of its group and will not
            // join the previous group, so just update its count.
            assert(node->group.use_count() == 1);
            node->count += 1;
        } else {
            // The node will split off from its current group to a new group.
            assert(node->next_sibling->count == node->count);
            Group* orig_group = node->group.get();
            orig_group->head = node->next_sibling;
            Group* new_group = new Group(node);
            new_group->next = orig_group;
            if (orig_group->prev) {
                new_group->prev = orig_group->prev;
                new_group->prev->next = new_group;
            }
            orig_group->prev = new_group;
            node->group.reset(new_group);  // std::shared_ptr assignment
            node->count += 1;
        }
    } else {
        // The node needs to be moved.
        assert(node->prev_sibling->count >= node->count);
        Node* other_node = node->prev_sibling->group->head;
        _remove_from_siblings(node);
        node->count += 1;
        _insert_into_siblings_before(node, other_node);
    }
}

void _decrement_count(Node* node) {
    // Decrement the count of a node by 1, and update its position in the
    // sibling and group linked lists if necessary.
    assert(node->count > 0);
    if (!node->parent) {
        // Root node has no siblings, update its count and return.
        node->count -= 1;
        return;
    }
    if (!node->next_sibling || node->next_sibling->count < node->count - 1) {
        // The node does not need to move, and will not join the next group
        // after its count is decremented.
        if (!node->prev_sibling || node->prev_sibling->count > node->count) {
            // The node should be the only member of its group and will not
            // join the next group, so just update its count.
            assert(node->group.use_count() == 1);
            node->count -= 1;
        } else {
            // The node will split off from its current group to a new group.
            assert(node->prev_sibling->count == node->count);
            Group* orig_group = node->group.get();
            Group* new_group = new Group(node);
            new_group->prev = orig_group;
            if (orig_group->next) {
                new_group->next = orig_group->next;
                new_group->next->prev = new_group;
            }
            orig_group->next = new_group;
            node->group.reset(new_group);  // std::shared_ptr assignment
            node->count -= 1;
        }
    } else if (node->next_sibling->count == node->count - 1) {
        // The node does not need to move, and will join the same group as its
        // next sibling.
        assert(node->next_sibling->group->head == node->next_sibling);
        node->next_sibling->group->head = node;
        if (node->group->head == node) {
            // The node is the head of its group, so the group will be removed.
            assert(node->group.use_count() == 1);
            Group* group = node->group.get();
            if (group->prev) {
                group->prev->next = group->next;
            }
            group->next->prev = group->prev;
        }
        node->group = node->next_sibling->group;  // std::shared_ptr assignment
        node->count -= 1;
    } else {
        // The node needs to be moved to the next group.
        assert(node->next_sibling->count == node->count);
        Group* other_group = node->group->next;
        _remove_from_siblings(node);
        node->count -= 1;
        if (!other_group) {
            // No next group, insert at the end of the siblings list.
            _insert_into_siblings_after(node, node->parent->tail_child);
        } else {
            // Insert as the head of the next group.
            _insert_into_siblings_before(node, other_group->head);
        }
    }
}

// Append a new element to a new or existing sequence.
void SuffixTree::append(int seq_id, int token) {
    // Initialize the sequence if it doesn't exist.
    if (!_seqs.contains(seq_id)) {
        assert(!_active_nodes.contains(seq_id));
        _seqs.emplace(seq_id);
        _active_nodes.emplace(seq_id);
    }

    // Keep references to the seq and active nodes for efficiency.
    std::vector<int32_t>& seq = _seqs[seq_id];
    std::deque<Node*>& active_nodes = _active_nodes[seq_id];

    // Insert a new active node at the root.
    active_nodes.push_back(_root.get());
    _root->endpoints[seq_id] = static_cast<int32_t>(seq.size());
    _root->count += 1;

    // Ensure the number of active nodes doesn't exceed max_depth.
    if (active_nodes.size() > static_cast<size_t>(_max_depth)) {
        active_nodes.pop_front();
    }
    seq.push_back(token);
    int32_t seq_len = static_cast<int32_t>(seq.size());
    
    // Iterate over all active nodes for this sequence.
    for (Node*& active_node : active_nodes) {
        Node* node = active_node;
        Node* child = nullptr;
        auto it = node->children.find(token);
        if (it != node->children.end()) {
            child = it->second.get();
        }

        assert(node->endpoints.contains(seq_id));
        assert(node->endpoints[seq_id] == seq.size() - 1);

        if (child == nullptr) {
            // Case 1: No existing child node for the new token.
            if (node->count == 1 && node != _root.get()) {
                // Case 1a: The active node has count = 1, which means the only
                // suffix that ends here is the one that's being extended right
                // now. Then this node should be a leaf node, and we can simply
                // extend the length of this node.
                assert(node->children.empty());
                assert(node->ref_seq == seq_id);
                node->length += 1;
                node->endpoints[seq_id] += 1;
            } else {
                // Case 1b: Either this is the root node, or the current suffix
                // is not the only one that passes through this node. Need to
                // extend the current suffix into a new child.

                // Create the new child node.
                Node* new_child = new Node(
                    1, // count
                    token,
                    1, // length
                    seq_id,
                    seq_len - 1);
                new_child->parent = node;
                new_child->endpoints[seq_id] = seq_len;

                // Add new child to active node.
                node->children.emplace(token, new_child);

                // Move the endpoint for the sequence from the active node to
                // the new child node.
                node->endpoints.erase(seq_id);

                // Link the new child node into the siblings list.
                if (node->children.size() == 1) {
                    // This should be the first child being added.
                    assert(!node->head_child && !node->tail_child);
                    node->head_child = node->tail_child = new_child;
                    new_child->group.reset(new Group(new_child));
                } else {
                    assert(node->tail_child);
                    _insert_into_siblings_after(new_child, node->tail_child);
                }

                // Update the active node to the new child node.
                active_node = new_child;
            }
        } else if (node->count == child->count + 1 && node != _root.get()) {
            // Case 2: The active node has a child for the new token, and that
            // child's count is exactly one fewer than the active node's count.
            // Since the suffix for the active node ends here, then all other
            // suffixes that pass through this node must go to that child.
            assert(node->children.size() == 1);  // Should have only one child.
            assert(node->endpoints.size() == 1);  // The current seq ends here.
            if (child->length == 1) {
                // Case 2a: The child has length 1. If we append the new token
                // to the current suffix, then it will perfectly overlap with
                // that child. Fuse the current active node with that child.

                // Update child to take the place of the current node.
                child->count += 1;  // Active node extends into the child node.
                child->token = node->token;
                child->length = node->length + 1;
                child->ref_seq = seq_id;
                child->ref_idx = seq_len - child->length;
                child->endpoints[seq_id] = seq_len;
                child->parent = node->parent;

                // Replace the current node with the child in the sibling list.
                // Must be done before changing any of the node's pointers.
                _replace_in_siblings(node, child);

                // Remove the current node from the suffix tree.
                Node* parent = node->parent;
                assert(parent->children.contains(node->token));
                assert(parent->children[node->token].get() == node);
                // Do it in two steps to avoid undefined evaluation order.
                Node* tmp = node->children[token].release();
                parent->children[child->token].reset(tmp);

                // Replace active node with child node.
                active_node = child;
            } else {
                // Case 2b: The child has length > 1. If we append the new
                // token to the current suffix, then it still does not reach
                // the child node. In this case, we keep both nodes but extend
                // the length of the current node by 1 into the child node.

                // Extend the length of the current node by 1.
                node->length += 1;
                node->endpoints[seq_id] += 1;  // Advance endpoint for the seq.
                node->ref_seq = seq_id;  // Need to update the ref sequence.
                node->ref_idx = seq_len - node->length;

                // Child should shrink by 1 at the beginning.
                child->length -= 1;
                child->ref_idx += 1;

                // The child's first token must be updated to its second token.
                child->token = _seqs[child->ref_seq][child->ref_idx];
                if (child->token != token) {
                    // Need to update the key in the parent's children map.
                    Node* tmp = node->children[token].release();
                    node->children.emplace(child->token, tmp);
                    node->children.erase(token);
                }

                // Active node stays the same.
            }
        } else {
            // Case 3: There exists a child node for the new token, and the
            // active node should move into that child.
            if (child->length == 1) {
                // Case 3a: The child node has length 1, just update the active
                // node pointer to it.

                // Move the endpoint for the sequence to the child.
                node->endpoints.erase(seq_id);
                child->endpoints[seq_id] = seq_len;

                // Increment the child count and update siblings list.
                _increment_count(child);

                // Replace active node with child node.
                active_node = child;
            } else {
                // Case 3b: The child node has length > 1. If the suffix is
                // extended into it, then it must split into a segment of
                // length 1 and another segment with the remainder.

                // Create the new intermediate node.
                Node* new_node = new Node(
                    child->count,
                    token,
                    1,  // length
                    seq_id,
                    seq_len - 1);
                new_node->parent = node;

                // Replace the child with the new node in the siblings list.
                // Must be done before changing any of the child's pointers.
                _replace_in_siblings(child, new_node);

                // Replace child with new node in the children map.
                node->children[token].release();  // Should be child.
                node->children[token].reset(new_node);

                // Child should shrink by 1 at the beginning.
                child->length -= 1;
                child->ref_idx += 1;

                // Child's first token must be updated to its second token.
                child->token = _seqs[child->ref_seq][child->ref_idx];

                // Insert the child into the new node's children map.
                new_node->children.emplace(child->token, child);
                child->parent = new_node;

                // Move the endpoint for the sequence to the new node.
                node->endpoints.erase(seq_id);
                new_node->endpoints[seq_id] = seq_len;

                // Create a new group for the child node.
                new_node->head_child = new_node->tail_child = child;
                child->group.reset(new Group(child));

                // Increment the new node count and update siblings lists.
                _increment_count(new_node);

                // Update active node to the new intermediate node.
                active_node = new_node;
            }
        }
    }
}

// Extend a new or existing sequence.
void SuffixTree::extend(int seq_id, std::span<const int32_t> tokens) {
    for (int token : tokens) {
        append(seq_id, token);
    }
}

// Remove an existing sequence.
void SuffixTree::remove(int seq_id) {
    const std::vector<int>& seq = _seqs[seq_id];
    std::vector<Node*> path;  // Declare here to avoid repeated allocations.
    // Loop through all suffix starting indices.
    for (int start = 0; start < seq.size(); start++) {
        Node *node = _root.get();
        node->count--;
        int idx = start;
        path.clear();
        // Loop through the nodes for this suffix.
        while (idx < seq.size()) {
            int token = seq[idx];
            if (!node->children.contains(token)) {
                break;
            }
            Node* child = node->children[token].get();
            if (child->count > 1) {
                _decrement_count(child);
            } else {
                assert(child->count == 1);
                // Remove the child along with its entire subtree.
                _remove_from_siblings(child);
                node->children.erase(token);
                break;
            }
            if (child->endpoints.contains(seq_id)) {
                child->endpoints.erase(seq_id);
            }
            idx += child->length;
            node = child;
            path.push_back(node);
        }
        // The last visited node may be mergeable with its child.
        if (node != _root.get() && node->children.size() == 1) {
            const auto& it = *node->children.begin();
            std::unique_ptr<Node>& child_uptr = node->children[it.first];
            if (node->count == child_uptr->count) {
                // Merge node into child and eliminate node.
                child_uptr->token = node->token;
                child_uptr->length += node->length;
                child_uptr->ref_idx -= node->length;
                child_uptr->parent = node->parent;
                _replace_in_siblings(node, child_uptr.get());
                path.back() = node = child_uptr.release();
                node->parent->children[node->token].reset(node);
            }
        }
        // ref_seq and ref_idx of all nodes in the path may need to be updated.
        // 1. Go to an arbitrary leaf to get its endpoints.
        Node* leaf = node;
        int distance = 0;  // Distance from node to leaf.
        while (!leaf->children.empty()) {
            leaf = (*leaf->children.begin()).second.get();
            distance += leaf->length;
        }
        // 2. Pick an arbitrary endpoint for the reference sequence and index.
        if (leaf->endpoints.empty() || leaf->endpoints.contains(seq_id)) {
            // Still need to visit this leaf later when removing this sequence.
            // We can skip updating the refs until the next time it's visited.
            continue;
        }
        const auto& ref = *leaf->endpoints.begin();
        // 3. Go back up the path to update all nodes' refs.
        int32_t ref_seq = ref.first;
        int32_t ref_idx = ref.second - distance;
        while (!path.empty()) {
            Node* n = path.back();
            path.pop_back();
            ref_idx -= n->length;
            if (n->ref_seq == seq_id) {
                n->ref_seq = ref_seq;
                n->ref_idx = ref_idx;
            }
        }
    }
    _seqs.erase(seq_id);
    _active_nodes.erase(seq_id);
}

Draft SuffixTree::speculate(std::span<const int32_t> context,
                            int max_spec_tokens,
                            float max_spec_factor,
                            float max_spec_offset,
                            float min_token_prob,
                            bool use_tree_spec) {
    Draft best_draft;
    for (int match_len = 1; match_len < context.size(); match_len++) {
        auto[node, idx] = _match_context(
            context.subspan(context.size() - match_len, match_len));
        if (node == nullptr) {
            break;
        }
        int max_tokens = std::min(max_spec_tokens,
                                  static_cast<int>(match_len * max_spec_factor
                                                   + max_spec_offset + 1e-6));
        max_tokens = std::max(max_tokens, 0);
        Draft draft;
        if (use_tree_spec) {
            draft = _speculate_tree(node, idx, max_tokens, min_token_prob);
        } else {
            draft = _speculate_path(node, idx, max_tokens, min_token_prob);
        }
        if (draft.score >= best_draft.score) {
            best_draft = std::move(draft);
            best_draft.match_len = match_len;
        }
    }
    return best_draft;
}

std::string SuffixTree::check_integrity() {
    // 1. Check structural integrity of all nodes.
    std::queue<Node*> queue;
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        std::string ret = _check_node_integrity(node);
        if (!ret.empty()) {
            return ret;
        }
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    // 2. Check all sequences are represented in the tree.
    std::unordered_map<Node*, int64_t> visit_count;
    for (int seq_id = 0; seq_id < _seqs.size(); seq_id++) {
        const std::vector<int>& seq = _seqs[seq_id];
        // Loop through all suffix starting indices.
        for (int start = 0; start < seq.size(); start++) {
            int idx = start;
            // Traverse the tree along this suffix.
            Node* node = _root.get();
            visit_count[node]++;
            while (idx < seq.size() && idx - start < _max_depth) {
                // There should be a child for the next token.
                CHECK_OR_RETURN(node->children.contains(seq[idx]));
                node = node->children[seq[idx]].get();
                visit_count[node]++;
                // Sequence should not end in the middle of a node.
                CHECK_OR_RETURN(idx + node->length <= seq.size());
                for (int i = 0; i < node->length; ++i) {
                    int ref_seq = node->ref_seq;
                    int ref_idx = node->ref_idx + i;
                    // Reference tokens should match sequence tokens.
                    CHECK_OR_RETURN(seq[idx + i] == _seqs[ref_seq][ref_idx]);
                }
                idx += node->length;
            }
            // The last node on this path should have an endpoint.
            CHECK_OR_RETURN(node->endpoints.contains(seq_id));
        }
    }
    // 3. Check all nodes were visited the correct number of times.
    assert(queue.empty());
    queue.push(_root.get());
    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();
        // The visit count should match the node count.
        CHECK_OR_RETURN(node->count == visit_count[node]);
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
    return "";
}

std::string SuffixTree::_check_node_integrity(Node* node) {
    int64_t children_count = 0;
    for (const auto& [token, child] : node->children) {
        // All children should have the correct parent pointer.
        CHECK_OR_RETURN(child->parent == node);
        children_count++;
    }
    // Node count should be at least the sum of all children counts.
    CHECK_OR_RETURN(children_count <= node->count);
    if (node == _root.get()) {
        // Root node should not contain any tokens, do some basic checks.
        CHECK_OR_RETURN(node->count >= 0);
        CHECK_OR_RETURN(node->parent == nullptr);
        CHECK_OR_RETURN(node->length == 0);
        CHECK_OR_RETURN(node->endpoints.empty());
        CHECK_OR_RETURN(node->ref_idx == -1);
    } else {
        // Node length should be positive.
        CHECK_OR_RETURN(node->length > 0);
        // Node count should be positive.
        CHECK_OR_RETURN(node->count > 0);
        // Each child count should be strictly less than the node count.
        // Otherwise, the node and the child should have been merged into a
        // single node.
        for (const auto& [token, child] : node->children) {
            CHECK_OR_RETURN(child->count < node->count);
        }
        // Internal nodes must have a valid reference sequence and index.
        CHECK_OR_RETURN(_seqs.contains(node->ref_seq));
        CHECK_OR_RETURN(node->ref_idx >= 0);
        CHECK_OR_RETURN(
            node->ref_idx + node->length <= _seqs[node->ref_seq].size());
        // Check the first token of the node is correct.
        CHECK_OR_RETURN(node->token == _seqs[node->ref_seq][node->ref_idx]);
        // Check the node is in its parent's children map.
        CHECK_OR_RETURN(node->parent->children.contains(node->token));
        CHECK_OR_RETURN(node->parent->children[node->token].get() == node);
        // Check all endpoint references are correct.
        for (auto [seq_id, end_idx] : node->endpoints) {
            // Endpoint should refer to a sequence id that exists.
            CHECK_OR_RETURN(_seqs.contains(seq_id));
            // Endpoint index should be within the sequence length.
            CHECK_OR_RETURN(end_idx > 0 && end_idx <= _seqs[seq_id].size());
            // Check all tokens from the start of the suffix to the endpoint.
            Node* n = node;
            int idx = end_idx;
            // Walk up the tree and check all tokens agree with the suffix
            // ending at this endpoint.
            do {
                // Check the index in the sequence is not underflowed.
                CHECK_OR_RETURN(n->length <= idx);
                idx -= n->length;
                for (int i = 0; i < n->length; ++i) {
                    int tok = _seqs[n->ref_seq][n->ref_idx + i];
                    // Check each token in this node agrees with the sequence.
                    CHECK_OR_RETURN(_seqs[seq_id][idx + i] == tok);
                }
                n = n->parent;
            } while (n != nullptr);
        }
    }
    // Check siblings list integrity.
    if (!node->head_child && !node->tail_child) {
        CHECK_OR_RETURN(node->children.empty());
    } else {
        // If there is a child then there must be both a head and a tail child.
        CHECK_OR_RETURN(node->head_child && node->tail_child);
        // Check head and tail child pointers are correct.
        CHECK_OR_RETURN(node->head_child->prev_sibling == nullptr);
        CHECK_OR_RETURN(node->tail_child->next_sibling == nullptr);
        // Check all children are in the siblings linked list.
        int count = 0;
        Node* child = node->head_child;
        Node* prev_child = nullptr;
        while (child != nullptr) {
            count++;
            // Check the child is in the children map.
            CHECK_OR_RETURN(node->children.contains(child->token));
            // Check the group pointer is valid.
            CHECK_OR_RETURN(child->group != nullptr);
            if (prev_child) {
                // Check the siblings are ordered in nonincreasing count.
                CHECK_OR_RETURN(child->count <= prev_child->count);
                // Check the sibling pointers are correct.
                CHECK_OR_RETURN(child->prev_sibling == prev_child);
                CHECK_OR_RETURN(prev_child->next_sibling == child);
                // Check the group pointers are correct.
                if (child->count == prev_child->count) {
                    // If the next sibling has the same count, they should be
                    // in the same group.
                    CHECK_OR_RETURN(child->group == prev_child->group);
                } else {
                    // Otherwise, they should be in different groups.
                    CHECK_OR_RETURN(child->group != prev_child->group);
                    // The child should be the head of its group.
                    CHECK_OR_RETURN(child->group->head == child);
                    // Check group pointers are correct.
                    CHECK_OR_RETURN(
                        child->group->prev == prev_child->group.get());
                    CHECK_OR_RETURN(
                        prev_child->group->next == child->group.get());

                }
            } else {
                CHECK_OR_RETURN(child == node->head_child);
            }
            prev_child = child;
            child = child->next_sibling;
        }
        // Check the last child reached is the tail child.
        CHECK_OR_RETURN(prev_child == node->tail_child);
        // Check the number of children matches the size of the children map.
        CHECK_OR_RETURN(count == node->children.size());
    }
    return "";
}

std::pair<Node*, int> SuffixTree::_match_context(
        std::span<const int32_t> context) {
    Node* node = _root.get();
    int idx = 0;
    const int32_t* ref_data = nullptr;
    for (int32_t token : context) {
        if (idx >= node->length) {
            auto it = node->children.find(token);
            if (it == node->children.end()) {
                return {nullptr, -1};
            }
            node = it->second.get();
            // Keep a pointer directly to the reference data for efficiency.
            ref_data = _seqs[node->ref_seq].data() + node->ref_idx;
            idx = 0;
        }
        assert(idx < node->length);
        if (ref_data[idx] != token) {
            return {nullptr, -1};
        }
        idx++;
    }
    return {node, idx};
}

Draft SuffixTree::_speculate_path(Node* node, int idx,
                                  int max_spec_tokens,
                                  float min_token_prob) {
    Draft ret;
    float prob = 1.0f;
    const int32_t* ref_data = _seqs[node->ref_seq].data() + node->ref_idx;
    while (ret.token_ids.size() < max_spec_tokens && prob >= min_token_prob) {
        if (idx < node->length) {
            // Use previous token index as parent; if none, mark as -1.
            ret.parents.push_back(static_cast<int>(ret.token_ids.size()) - 1);
            ret.token_ids.push_back(ref_data[idx]);
            ret.probs.push_back(prob);
            ret.score += prob;
            idx++;
        } else {
            Node* child = node->head_child;
            if (child == nullptr) {
                break;
            }
            int64_t count = child->count;
            prob *= static_cast<float>(count) / node->count;
            node = child;
            // Keep a pointer directly to the reference data for efficiency.
            ref_data = _seqs[node->ref_seq].data() + node->ref_idx;
            idx = 0;
        }
    }
    return ret;
}

struct HeapItem {
    float prob;
    Node* node;
    int idx;
    int parent;   // index in the draft token list; -1 if none.

    HeapItem(float p, Node* n, int i, int par)
        : prob(p), node(n), idx(i), parent(par) {}
};

struct HeapItemCmp {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        // In C++ priority_queue by default returns the largest element.
        // Thus, we compare probabilities so that the highest prob is returned.
        return a.prob < b.prob;
    }
};

// Get a draft token tree using a priority queue.
Draft SuffixTree::_speculate_tree(Node* node, int idx,
                                  int max_spec_tokens,
                                  float min_token_prob) {
    Draft ret;
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapItemCmp> queue;
    queue.emplace(1.0, node, idx, -1);
    while (ret.token_ids.size() < max_spec_tokens && !queue.empty()) {
        HeapItem it = queue.top();
        queue.pop();
        if (it.idx < it.node->length) {
            int32_t token = _seqs[it.node->ref_seq][it.node->ref_idx + it.idx];
            ret.token_ids.push_back(token);
            ret.parents.push_back(it.parent);
            ret.probs.push_back(it.prob);
            ret.score += it.prob;
            queue.emplace(it.prob, it.node, it.idx + 1,
                          static_cast<int>(ret.token_ids.size()) - 1);
        } else {
            Node* child = it.node->head_child;
            while (child) {
                float prob = it.prob * child->count /
                    static_cast<float>(it.node->count);
                if (prob < min_token_prob) {
                    break;
                }
                queue.emplace(prob, child, 0, it.parent);
                child = child->next_sibling;
            }
        }
    }
    return ret;
}

size_t SuffixTree::estimate_memory() const {
    size_t total = sizeof(*this);
    std::vector<Node*> stack;
    stack.push_back(_root.get());
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        total += node->memory_usage();
        if (node->head_child) {
            Group* group = node->head_child->group.get();
            while (group) {
                total += sizeof(*group);
                group = group->next;
            }
        }
        for (const auto& [token, child] : node->children) {
            stack.push_back(child.get());
        }
    }
    for (const auto& [seq_id, seq] : _seqs) {
        total += sizeof(seq) * seq.capacity();
    }
    for (const auto& [seq_id, nodes] : _active_nodes) {
        total += sizeof(nodes) * nodes.size();
    }
    return total;
}

// ============================================================================
// Tree Visualization
// ============================================================================

std::string SuffixTree::dump() const {
    std::ostringstream oss;

    // Helper lambda to format tokens as string
    auto tokens_to_string = [](const std::vector<int32_t>& seq, int ref_idx, int length) {
        std::ostringstream s;
        s << "[";
        for (int i = 0; i < length; i++) {
            if (i > 0) s << ", ";
            s << seq[static_cast<size_t>(ref_idx + i)];
        }
        s << "]";
        return s.str();
    };

    oss << "=== SuffixTree (max_depth=" << _max_depth << ") ===\n";

    // Print sequences
    oss << "\nSequences (" << _seqs.size() << "):\n";
    for (const auto& [seq_id, tokens] : _seqs) {
        oss << "  [" << seq_id << "]: " << tokens_to_string(tokens, 0, static_cast<int>(tokens.size())) << "\n";
    }

    // Print tree structure using iterative DFS with explicit stack
    oss << "\nTree Structure:\n";

    struct StackItem {
        const Node* node;
        std::string prefix;
        bool is_root;
        bool is_last;
    };

    std::vector<StackItem> stack;
    stack.push_back({_root.get(), "", true, true});

    while (!stack.empty()) {
        auto [node, prefix, is_root, is_last] = stack.back();
        stack.pop_back();

        if (is_root) {
            oss << "(root) count=" << node->count << "\n";
        } else {
            // Build edge label by looking up in _seqs
            std::string edge_label;
            auto it = _seqs.find(node->ref_seq);
            if (node->ref_seq >= 0 && it != _seqs.end()) {
                edge_label = tokens_to_string(it->second, node->ref_idx, node->length);
            } else {
                edge_label = "[" + std::to_string(node->token) + "...]";
            }

            std::string connector = is_last ? "└── " : "├── ";
            oss << prefix << connector << edge_label
                << " (count=" << node->count
                << ", len=" << node->length
                << ", ref=" << node->ref_seq << ":" << node->ref_idx << ")";

            // Show endpoints if any
            if (!node->endpoints.empty()) {
                oss << " [endpoints: ";
                bool first = true;
                for (const auto& [seq_id, end_idx] : node->endpoints) {
                    if (!first) oss << ", ";
                    oss << seq_id << "@" << end_idx;
                    first = false;
                }
                oss << "]";
            }
            oss << "\n";
        }

        // Collect children sorted by count (desc) then token (asc) for consistent output
        std::vector<std::pair<int, const Node*>> children_sorted;
        for (const auto& [token, child] : node->children) {
            children_sorted.push_back({token, child.get()});
        }
        std::sort(children_sorted.begin(), children_sorted.end(),
                  [](const auto& a, const auto& b) {
                      if (a.second->count != b.second->count)
                          return a.second->count > b.second->count;
                      return a.first < b.first;
                  });

        // Add children to stack in reverse order (so first child is processed first)
        std::string new_prefix = prefix + (is_root ? "" : (is_last ? "    " : "│   "));
        for (int i = static_cast<int>(children_sorted.size()) - 1; i >= 0; i--) {
            bool child_is_last = (i == static_cast<int>(children_sorted.size()) - 1);
            stack.push_back({children_sorted[static_cast<size_t>(i)].second, new_prefix, false, child_is_last});
        }
    }

    return oss.str();
}

// ============================================================================
// Serialization/Deserialization Implementation
// ============================================================================

namespace {

// Binary format version
constexpr uint32_t SNAPSHOT_VERSION = 1;

// Helper functions for writing primitives to buffer (little-endian)
void write_u32(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(val & 0xFF);
    buf.push_back((val >> 8) & 0xFF);
    buf.push_back((val >> 16) & 0xFF);
    buf.push_back((val >> 24) & 0xFF);
}

void write_i32(std::vector<uint8_t>& buf, int32_t val) {
    write_u32(buf, static_cast<uint32_t>(val));
}

void write_i64(std::vector<uint8_t>& buf, int64_t val) {
    write_u32(buf, static_cast<uint32_t>(val & 0xFFFFFFFF));
    write_u32(buf, static_cast<uint32_t>((val >> 32) & 0xFFFFFFFF));
}

// Helper functions for reading primitives from buffer (little-endian)
uint32_t read_u32(std::span<const uint8_t> data, size_t& offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("Snapshot data truncated");
    }
    uint32_t val = static_cast<uint32_t>(data[offset]) |
                   (static_cast<uint32_t>(data[offset + 1]) << 8) |
                   (static_cast<uint32_t>(data[offset + 2]) << 16) |
                   (static_cast<uint32_t>(data[offset + 3]) << 24);
    offset += 4;
    return val;
}

int32_t read_i32(std::span<const uint8_t> data, size_t& offset) {
    return static_cast<int32_t>(read_u32(data, offset));
}

int64_t read_i64(std::span<const uint8_t> data, size_t& offset) {
    uint32_t low = read_u32(data, offset);
    uint32_t high = read_u32(data, offset);
    return static_cast<int64_t>(low) |
           (static_cast<int64_t>(high) << 32);
}

}  // anonymous namespace

std::vector<uint8_t> SuffixTree::create_snapshot() const {
    std::vector<uint8_t> buffer;

    // =========================================================================
    // PHASE 1: BFS traversal to assign indices to nodes
    // =========================================================================
    std::vector<Node*> nodes_list;
    std::unordered_map<Node*, uint32_t> node_to_index;

    std::queue<Node*> bfs_queue;
    bfs_queue.push(_root.get());

    while (!bfs_queue.empty()) {
        Node* node = bfs_queue.front();
        bfs_queue.pop();

        node_to_index[node] = static_cast<uint32_t>(nodes_list.size());
        nodes_list.push_back(node);

        // Add children in token order for deterministic output
        std::vector<std::pair<int, Node*>> children_sorted;
        for (const auto& [token, child] : node->children) {
            children_sorted.push_back({token, child.get()});
        }
        std::sort(children_sorted.begin(), children_sorted.end());

        for (const auto& [token, child] : children_sorted) {
            bfs_queue.push(child);
        }
    }

    // =========================================================================
    // PHASE 2: Write header (16 bytes)
    // =========================================================================
    write_u32(buffer, SNAPSHOT_VERSION);                          // version
    write_u32(buffer, static_cast<uint32_t>(_max_depth));         // max_depth
    write_u32(buffer, static_cast<uint32_t>(_seqs.size()));       // num_sequences
    write_u32(buffer, static_cast<uint32_t>(nodes_list.size()));  // num_nodes

    // =========================================================================
    // PHASE 3: Write sequences
    // =========================================================================
    for (const auto& [seq_id, tokens] : _seqs) {
        write_i32(buffer, seq_id);
        write_u32(buffer, static_cast<uint32_t>(tokens.size()));
        for (int32_t token : tokens) {
            write_i32(buffer, token);
        }

        // Write active nodes for this sequence
        auto it = _active_nodes.find(seq_id);
        if (it != _active_nodes.end()) {
            const std::deque<Node*>& active = it->second;
            write_u32(buffer, static_cast<uint32_t>(active.size()));
            for (Node* node : active) {
                write_u32(buffer, node_to_index[node]);
            }
        } else {
            write_u32(buffer, 0);  // no active nodes
        }
    }

    // =========================================================================
    // PHASE 4: Write nodes (BFS order)
    // =========================================================================
    for (Node* node : nodes_list) {
        write_i64(buffer, node->count);
        write_i32(buffer, node->token);
        write_i32(buffer, node->length);
        write_i32(buffer, node->ref_seq);
        write_i32(buffer, node->ref_idx);

        // Write endpoints
        write_u32(buffer, static_cast<uint32_t>(node->endpoints.size()));
        for (const auto& [seq_id, end_idx] : node->endpoints) {
            write_i32(buffer, seq_id);
            write_i32(buffer, end_idx);
        }

        // Write children (sorted by token for determinism)
        std::vector<std::pair<int, Node*>> children_sorted;
        for (const auto& [token, child] : node->children) {
            children_sorted.push_back({token, child.get()});
        }
        std::sort(children_sorted.begin(), children_sorted.end());

        write_u32(buffer, static_cast<uint32_t>(children_sorted.size()));
        for (const auto& [token, child] : children_sorted) {
            write_u32(buffer, node_to_index[child]);
        }
    }

    return buffer;
}

void SuffixTree::_rebuild_siblings(Node* parent) {
    if (parent->children.empty()) {
        parent->head_child = nullptr;
        parent->tail_child = nullptr;
        return;
    }

    // Collect all children
    std::vector<Node*> children;
    for (const auto& [token, child] : parent->children) {
        children.push_back(child.get());
    }

    // Sort by count (descending), then by token (for determinism)
    std::sort(children.begin(), children.end(), [](Node* a, Node* b) {
        if (a->count != b->count) return a->count > b->count;
        return a->token < b->token;
    });

    // Build doubly-linked sibling list
    for (size_t i = 0; i < children.size(); i++) {
        Node* child = children[i];
        child->prev_sibling = (i > 0) ? children[i - 1] : nullptr;
        child->next_sibling = (i < children.size() - 1) ? children[i + 1] : nullptr;
    }

    parent->head_child = children.front();
    parent->tail_child = children.back();

    // Build groups (nodes with same count share a group)
    std::shared_ptr<Group> current_group = nullptr;
    Group* prev_group = nullptr;

    for (size_t i = 0; i < children.size(); i++) {
        Node* child = children[i];

        bool new_group_needed = (i == 0) ||
                                (child->count != children[i - 1]->count);

        if (new_group_needed) {
            // Create new group
            current_group = std::make_shared<Group>(child);

            // Link to previous group
            if (prev_group) {
                prev_group->next = current_group.get();
                current_group->prev = prev_group;
            }
            prev_group = current_group.get();
        }

        child->group = current_group;
    }
}

void SuffixTree::_rebuild_all_siblings(Node* root) {
    std::queue<Node*> queue;
    queue.push(root);

    while (!queue.empty()) {
        Node* node = queue.front();
        queue.pop();

        // Rebuild siblings for this node's children
        _rebuild_siblings(node);

        // Add children to queue
        for (const auto& [token, child] : node->children) {
            queue.push(child.get());
        }
    }
}

SuffixTree SuffixTree::restore_snapshot(std::span<const uint8_t> data) {
    size_t offset = 0;

    // =========================================================================
    // PHASE 1: Read and validate header
    // =========================================================================
    uint32_t version = read_u32(data, offset);
    if (version != SNAPSHOT_VERSION) {
        throw std::runtime_error("Unsupported snapshot version: " +
                                 std::to_string(version));
    }

    uint32_t max_depth = read_u32(data, offset);
    uint32_t num_seqs = read_u32(data, offset);
    uint32_t num_nodes = read_u32(data, offset);

    // Create tree with correct max_depth
    SuffixTree tree(static_cast<int>(max_depth));

    // Clear the auto-created root, we'll restore it from snapshot
    tree._root.reset();

    // =========================================================================
    // PHASE 2: Pre-allocate nodes
    // =========================================================================
    std::vector<std::unique_ptr<Node>> nodes(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        nodes[i] = std::make_unique<Node>();
    }

    // =========================================================================
    // PHASE 3: Read sequences and store active node indices
    // =========================================================================
    std::vector<std::pair<int32_t, std::vector<uint32_t>>> active_indices;

    for (uint32_t s = 0; s < num_seqs; s++) {
        int32_t seq_id = read_i32(data, offset);
        uint32_t num_tokens = read_u32(data, offset);

        std::vector<int32_t> tokens(num_tokens);
        for (uint32_t t = 0; t < num_tokens; t++) {
            tokens[t] = read_i32(data, offset);
        }
        tree._seqs.emplace(seq_id, std::move(tokens));

        // Read active node indices (resolve later)
        uint32_t num_active = read_u32(data, offset);
        std::vector<uint32_t> indices(num_active);
        for (uint32_t a = 0; a < num_active; a++) {
            indices[a] = read_u32(data, offset);
        }
        active_indices.push_back({seq_id, std::move(indices)});
    }

    // =========================================================================
    // PHASE 4: Read nodes and store child indices
    // =========================================================================
    std::vector<std::vector<uint32_t>> children_indices(num_nodes);

    for (uint32_t i = 0; i < num_nodes; i++) {
        Node* node = nodes[i].get();

        node->count = read_i64(data, offset);
        node->token = read_i32(data, offset);
        node->length = read_i32(data, offset);
        node->ref_seq = read_i32(data, offset);
        node->ref_idx = read_i32(data, offset);

        // Read endpoints
        uint32_t num_endpoints = read_u32(data, offset);
        for (uint32_t e = 0; e < num_endpoints; e++) {
            int32_t seq_id = read_i32(data, offset);
            int32_t end_idx = read_i32(data, offset);
            node->endpoints.emplace(seq_id, end_idx);
        }

        // Read child indices (resolve later)
        uint32_t num_children = read_u32(data, offset);
        children_indices[i].resize(num_children);
        for (uint32_t c = 0; c < num_children; c++) {
            children_indices[i][c] = read_u32(data, offset);
        }
    }

    // =========================================================================
    // PHASE 5: Build index-to-node map BEFORE transferring ownership
    // =========================================================================
    std::vector<Node*> index_to_node(num_nodes);
    for (uint32_t i = 0; i < num_nodes; i++) {
        index_to_node[i] = nodes[i].get();
    }

    // =========================================================================
    // PHASE 6: Resolve node pointers and build children maps
    // We must be careful about the order - process from back to front
    // so that we don't move a node before its children have been processed.
    // Actually, simpler: set parent pointers first, then transfer ownership.
    // =========================================================================

    // First pass: set all parent pointers
    for (uint32_t i = 0; i < num_nodes; i++) {
        Node* parent = index_to_node[i];
        for (uint32_t child_idx : children_indices[i]) {
            index_to_node[child_idx]->parent = parent;
        }
    }

    // Second pass: transfer ownership (process in reverse order to handle
    // nested structures correctly - children before parents)
    for (int i = static_cast<int>(num_nodes) - 1; i >= 0; i--) {
        for (uint32_t child_idx : children_indices[i]) {
            Node* child = index_to_node[child_idx];
            Node* parent = index_to_node[i];
            parent->children.emplace(child->token, std::move(nodes[child_idx]));
        }
    }

    // Root node is index 0
    tree._root = std::move(nodes[0]);

    // =========================================================================
    // PHASE 7: Rebuild sibling/group linked lists
    // =========================================================================
    _rebuild_all_siblings(tree._root.get());

    // =========================================================================
    // PHASE 8: Resolve active nodes using the index-to-node map
    // =========================================================================
    for (auto& [seq_id, indices] : active_indices) {
        std::deque<Node*> active;
        for (uint32_t idx : indices) {
            active.push_back(index_to_node[idx]);
        }
        tree._active_nodes.emplace(seq_id, std::move(active));
    }

    return tree;
}
