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

#include "suffix_cache.h"
#include <xxhash.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include <ctime>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <omp.h>
#include <stdlib.h>

const int SPEC_START_LEN = 2;
const int SPEC_MAX_LEN = 16;

#include <mutex>

SuffixSpecResult SuffixSpecResult::from_candidate(const Candidate& candidate) {
    SuffixSpecResult result;
    result.token_ids = candidate.token_ids;
    result.parents = candidate.parents;
    result.probs = candidate.probs;
    result.score = candidate.score;
    result.match_len = candidate.match_len;
    return result;
}

// Helper method to compute prompt hash
std::string SuffixCache::compute_prompt_hash(const std::vector<int>& prompt) {
    if (prompt.empty()) {
        return "";
    }

    XXH64_hash_t hash = XXH64(prompt.data(), prompt.size() * sizeof(int), 0);
    std::stringstream ss;
    ss << std::hex << hash;
    return ss.str();
}

SuffixCache::SuffixCache() {
    // Open the existing shared memory segment
    shared_memory_segment_ = new boost::interprocess::managed_shared_memory(
        boost::interprocess::open_only,
        SHARED_MEMORY_NAME
    );

    // Find the tree map in shared memory
    shared_tree_map_ = shared_memory_segment_->find<SharedTreeMap>("tree_map").first;

    if (!shared_tree_map_) {
        std::cerr << "Failed to find tree_map in shared memory" << std::endl;
        delete shared_memory_segment_;
        shared_memory_segment_ = nullptr;
    } else {
        std::cout << "Successfully connected to shared memory cache" << std::endl;
    }

    setenv("OMP_WAIT_POLICY", "ACTIVE", 1);
    omp_set_num_threads(8);
}

SuffixCache::~SuffixCache() {
    delete shared_memory_segment_;
    shared_memory_segment_ = nullptr;
    shared_tree_map_ = nullptr;
}


void SuffixCache::fetch_responses_by_prompts_batch(const std::vector<std::string>& req_ids,
                                                  const std::vector<std::vector<int>>& prompts) {
    if (req_ids.size() != prompts.size()) {
        std::cerr << "Error: req_ids and prompts size mismatch" << std::endl;
        return;
    }

    std::vector<uint64_t> req_hashes;
    std::vector<size_t> indices_to_process;

    // First pass: process each prompt and identify which ones need lookup
    for (size_t i = 0; i < req_ids.size(); ++i) {
        const std::string& req_id = req_ids[i];
        const std::vector<int>& prompt = prompts[i];

        // Check if we already have responses for this request
        if (req_id_to_responses_.find(req_id) == req_id_to_responses_.end()) {
            req_id_to_spec_len_[req_id] = SPEC_START_LEN;
            uint64_t req_hash = XXH64(prompt.data(), prompt.size() * sizeof(int), 0);
            req_hashes.push_back(req_hash);
            indices_to_process.push_back(i);
        }
    }



    auto mutex = shared_memory_segment_->find<interprocess_mutex>("mutex").first;
    // Lock the mutex to safely access the shared tree map
    scoped_lock<interprocess_mutex> shm_lock(*mutex);

    uint64_t shared_mem_base = (uint64_t)shared_memory_segment_->get_address();
    // Check each prompt in the shared memory
    for (size_t i = 0; i < indices_to_process.size(); ++i) {
        const uint64_t req_hash = req_hashes[i];
        size_t idx = indices_to_process[i];
        const std::string& req_id = req_ids[idx];

        auto it = shared_tree_map_->find(req_hash);
        if (it != shared_tree_map_->end()) {
                req_id_to_responses_[req_id] = (SuffixTree*)(it->second + shared_mem_base);
            } else {
                req_id_to_responses_[req_id] = nullptr;
            }
        }
}

void SuffixCache::update_spec_len(const std::string& req_id, const int valid_len) {
    // Parse request ID
    if (req_id_to_spec_len_.find(req_id) == req_id_to_spec_len_.end()) {
        std::cerr << "Key not found in req_id_to_spec_len_: " << req_id << std::endl;
        return;
    }

    const int current_spec_len = req_id_to_spec_len_[req_id];

    if (valid_len > current_spec_len) {
        req_id_to_spec_len_[req_id] = std::min(current_spec_len * 2, SPEC_MAX_LEN);
    } else {
        req_id_to_spec_len_[req_id] = std::max(SPEC_START_LEN, current_spec_len / 2);
    }
}

void SuffixCache::evict_responses(const std::string& req_id) {
    // Remove from req_id_to_spec_len_
    req_id_to_spec_len_.erase(req_id);
    req_id_to_responses_.erase(req_id);
}

std::vector<std::vector<int>> SuffixCache::speculate(
    const std::vector<std::string>& req_ids,
    const std::vector<std::vector<int>>& patterns,
    float min_token_prob,
    bool use_tree_spec) {


    assert(req_ids.size() == patterns.size());

    std::vector<std::vector<int>> results(req_ids.size());

    #pragma omp parallel for
    for (size_t i = 0; i < req_ids.size(); ++i) {
        const std::string& req_id = req_ids[i];
        const std::vector<int>& pattern = patterns[i];

        if (pattern.empty()) {
            continue;
        }

        // Check if prompt exists
        auto prompt_it = req_id_to_responses_.find(req_id);
        if (prompt_it == req_id_to_responses_.end()) {
            throw std::invalid_argument("Prompt does not exist for request '" + req_id + "'");
        }

        // Check if pattern is not empty
        if (pattern.empty()) {
            throw std::invalid_argument("Pattern must not be empty");
        }

        // Get spec_len
        auto spec_len_it = req_id_to_spec_len_.find(req_id);
        if (spec_len_it == req_id_to_spec_len_.end()) {
            throw std::invalid_argument("Spec length not found for request '" + req_id + "'");
        }

        auto suffix_tree = prompt_it->second;
        if (suffix_tree == nullptr) {
            continue;
        }
        // Speculate
        Candidate candidate = suffix_tree->speculate(
            pattern,
            spec_len_it->second,
            min_token_prob,
            use_tree_spec);


        results[i] = candidate.token_ids;
    }

    // if (!candidate.token_ids.empty()) {
    // // Debug output
    //     std::cout << "req_id: " << req_id;
    //     std::cout << ", speculate pattern: [";
    //     for (int token : pattern)
    //         std::cout << token << " ";
    //     std::cout << "], speculate result: [";
    //     for (int token : candidate.token_ids)
    //         std::cout << token << " ";
    //     std::cout << "]";
    //     std::cout << std::endl;
    // }


    return results;
}
