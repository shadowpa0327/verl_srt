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

#include "suffix_cache_updater.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>

SuffixCacheUpdater::SuffixCacheUpdater() {
    std::cout << "Initializing suffix cache updater..." << std::endl;
    server_addresses_ = extract_addresses_from_env();
    initialize_stubs();
}

SuffixCacheUpdater::SuffixCacheUpdater(const std::vector<std::string>& server_addresses)
    : server_addresses_(server_addresses) {
    initialize_stubs();
}

SuffixCacheUpdater::~SuffixCacheUpdater() {
    // gRPC stubs will be automatically cleaned up
}

std::vector<std::string> SuffixCacheUpdater::extract_addresses_from_env() {
    // Get the ARNOLD_WORKER_HOSTS environment variable
    const char* arnold_worker_hosts = std::getenv("ARNOLD_WORKER_HOSTS");
    if (!arnold_worker_hosts) {
        // If environment variable is not set, use default localhost
        return {"localhost:6378"};
    }

    // Use regex to extract IPv6 hosts from the format [host]:port
    std::regex pattern(R"(\[([\da-f:]+)\]:\d+)", std::regex::icase);

    std::string hosts_str(arnold_worker_hosts);
    std::smatch match;
    std::set<std::string> unique_ips;  // Use set to avoid duplicates

    std::string::const_iterator search_start(hosts_str.cbegin());
    while (std::regex_search(search_start, hosts_str.cend(), match, pattern)) {
        std::string ip = match[1].str();
        unique_ips.insert(ip);
        search_start = match.suffix().first;
    }

    std::vector<std::string> addresses;
    for (const auto& ip : unique_ips) {
        // Add port 6378 for rollout cache server
        addresses.push_back("[" + ip + "]:6378");
    }

    if (addresses.empty()) {
        // Fallback to localhost if no matches found
        addresses.push_back("localhost:6378");
    }

    return addresses;
}

void SuffixCacheUpdater::initialize_stubs() {
    stubs_.clear();
    for (const auto& address : server_addresses_) {
        auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
        stubs_.push_back(RolloutCacheService::NewStub(channel));
        std::cout << "Connected to rollout cache server at " << address << std::endl;
    }
}

void SuffixCacheUpdater::update_response_cache(
    const std::vector<std::vector<int>>& prompts,
    const std::vector<std::vector<int>>& responses,
    const std::vector<float>& prompt_lengths,
    const std::vector<float>& response_lengths,
    int responses_per_prompt) {

    // Calculate number of prompts based on total responses and responses per prompt
    int prompts_num = responses.size() / responses_per_prompt;

    // Prepare all requests first
    std::vector<UpdateCacheRequest> requests;
    requests.resize(prompts_num);

    for (int i = 0; i < prompts_num; ++i) {
        UpdateCacheRequest& request = requests[i];

        // Calculate indices
        int prompt_idx = i * responses_per_prompt;

        // Get the prompt and its length
        const std::vector<int>& prompt = prompts[prompt_idx];
        int prompt_len = int(prompt_lengths[prompt_idx]);

        // Make sure prompt_len doesn't exceed the prompt's size
        if (prompt_len > static_cast<int>(prompt.size())) {
            prompt_len = static_cast<int>(prompt.size());
        }

        const int* token_data = prompt.data() + (prompt.size() - prompt_len);
        size_t bytes_size = prompt_len * sizeof(int);
        uint64_t hash = XXH64(token_data, bytes_size, 0);

        request.set_prompt_hash(hash);

        TokenList* prompt_tokens = request.mutable_prompt();
        for (int k = prompt.size() - prompt_len; k < static_cast<int>(prompt.size()); ++k) {
            prompt_tokens->add_tokens(prompt[k]);
        }

        // Collect the batch of responses for this prompt
        std::vector<std::vector<int>> resp_batch;
        for (int j = 0; j < responses_per_prompt; ++j) {
            int resp_idx = prompt_idx + j;
            TokenList* response_tokens = request.add_responses();
            for (int k = 0; k < response_lengths[resp_idx]; ++k) {
                response_tokens->add_tokens(responses[resp_idx][k]);
            }
        }
    }

    // Now send all requests to all servers asynchronously
    grpc::CompletionQueue cq;

    // Structure to hold async call state
    struct AsyncCallData {
        grpc::ClientContext context;
        UpdateCacheResponse response;
        grpc::Status status;
        std::unique_ptr<grpc::ClientAsyncResponseReader<UpdateCacheResponse>> response_reader;
        size_t server_index;
        size_t request_index;
    };

    std::vector<std::unique_ptr<AsyncCallData>> call_data_list;

    // Start all async calls for all requests to all servers
    for (size_t req_idx = 0; req_idx < requests.size(); ++req_idx) {
        for (size_t server_idx = 0; server_idx < stubs_.size(); ++server_idx) {
            auto call_data = std::make_unique<AsyncCallData>();
            call_data->server_index = server_idx;
            call_data->request_index = req_idx;

            // Start the async RPC call
            call_data->response_reader = stubs_[server_idx]->AsyncUpdateCache(
                &call_data->context, requests[req_idx], &cq);

            // Request that, upon completion of the RPC, "call_data" be "returned"
            // to us via the completion queue
            call_data->response_reader->Finish(
                &call_data->response,
                &call_data->status,
                call_data.get());

            call_data_list.push_back(std::move(call_data));
        }
    }

    // Wait for all calls to complete and check results
    size_t total_calls = requests.size() * stubs_.size();
    for (size_t completed = 0; completed < total_calls; ++completed) {
        void* got_tag;
        bool ok;

        // Block until the next result is available in the completion queue
        if (cq.Next(&got_tag, &ok)) {
            AsyncCallData* call_data = static_cast<AsyncCallData*>(got_tag);

            if (ok && call_data->status.ok() && call_data->response.success()) {
                // std::cout << "Successfully updated cache on server "
                //           << server_addresses_[call_data->server_index]
                //           << " for request " << call_data->request_index << std::endl;
            } else {
                std::cerr << "Failed to update cache on server "
                          << server_addresses_[call_data->server_index]
                          << " for request " << call_data->request_index
                          << ": " << call_data->status.error_message() << std::endl;
            }
        }
    }
}
