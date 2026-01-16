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

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <utility>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include "suffix_tree.h"
#include "rollout_cache_server.h" // For SharedTreeMap definition

// Use SuffixTreePtr from rollout_cache_server.h
using boost::interprocess::offset_ptr;

/**
 * A struct representing the result of a speculation using SuffixDecoding.
 */
struct SuffixSpecResult {
    std::vector<int> token_ids;    // List of token IDs in the speculation result
    std::vector<int> parents;      // List of parent indices for each token
    std::vector<float> probs;      // List of estimated probabilities for each token
    float score = 0.0f;            // Overall score of the suffix match
    int match_len = 0;             // Length of the pattern match

    // Create SuffixSpecResult from Candidate
    static SuffixSpecResult from_candidate(const Candidate& candidate);
};

/**
 * A class that provides caching functionality for suffix-based speculative decoding.
 */
class SuffixCache {
public:
    SuffixCache();

    /**
     * Destructor for SuffixCache.
     */
    ~SuffixCache();

    // fetch_responses_by_prompt method removed as it's no longer used

    /**
     * Fetches responses associated with multiple prompts from the cache in batch.
     *
     * @param req_ids The list of unique identifiers for the requests
     * @param prompts The list of token ID vectors representing the prompts
     */
    void fetch_responses_by_prompts_batch(const std::vector<std::string>& req_ids,
                                         const std::vector<std::vector<int>>& prompts);

    /**
     * Updates the speculation length for a request based on validation results.
     *
     * @param req_id The unique identifier for the request
     * @param valid_len The number of valid tokens in the last speculation
     */
    void update_spec_len(const std::string& req_id, int valid_len);

    /**
     * Evicts all responses associated with the given request ID from the cache.
     *
     * @param req_id The unique identifier for the request whose responses should be evicted
     */
    void evict_responses(const std::string& req_id);

    /**
     * Performs speculation for a given pattern.
     *
     * @param req_id The unique identifier for the request
     * @param pattern The sequence of token IDs to speculate from
     * @param min_token_prob The minimum probability threshold for token selection (default: 0.1)
     * @param use_tree_spec Whether to use tree-based speculation (default: false)
     * @return A SuffixSpecResult containing the speculation results
     */
    std::vector<std::vector<int>>  speculate(
        const std::vector<std::string>& req_ids,
        const std::vector<std::vector<int>>& patterns,
        float min_token_prob = 0.1f,
        bool use_tree_spec = false);

private:
    // Shared memory related members
    boost::interprocess::managed_shared_memory* shared_memory_segment_ = nullptr;
    SharedTreeMap* shared_tree_map_ = nullptr;

    // Maps prompt IDs to their corresponding SuffixTree objects
    std::unordered_map<std::string, SuffixTree*> req_id_to_responses_;

    // Maps request IDs to their speculation lengths
    std::unordered_map<std::string, int> req_id_to_spec_len_;

    // Helper method to convert prompt tokens to hash string
    std::string compute_prompt_hash(const std::vector<int>& prompt);
};
