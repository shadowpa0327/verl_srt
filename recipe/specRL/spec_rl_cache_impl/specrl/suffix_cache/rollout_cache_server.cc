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

#include "rollout_cache_server.h"
#include <chrono>

// Mutex for thread-safe console output

// Implementation of RolloutCacheService
grpc::Status RolloutCacheServiceImpl::UpdateCache(
    grpc::ServerContext* context,
    const UpdateCacheRequest* request,
    UpdateCacheResponse* response) {
    // Use mutex to ensure thread-safe console output

    // std::cout << "Received UpdateCacheRequest:" << std::endl;
    // std::cout << "  Prompt hash: " << request->prompt_hash() << std::endl;
    // std::cout << "  Number of responses: " << request->responses_size() << std::endl;

    ShmemAllocator alloc(segment_->get_segment_manager());
    SuffixTree* tree = segment_->construct<SuffixTree>(anonymous_instance)(alloc);

    const int prefix_tokens_to_include = 5;
    int tokens_len = request->prompt().tokens_size() + 1;
    for (int i = 0; i < request->responses_size(); i++) {
        tokens_len += request->responses(i).tokens_size() + prefix_tokens_to_include + 1;
    }

    std::vector<int> tokens;
    tokens.reserve(tokens_len);
    std::vector<int> prefix_tokens;
    prefix_tokens.reserve(5);
    if (request->has_prompt() && request->prompt().tokens_size() > 0) {
        const auto& token_list = request->prompt();
        for (int j = 0; j < token_list.tokens_size(); j++)
            tokens.push_back(token_list.tokens(j));
        tokens.push_back(-1);

        int start_idx = token_list.tokens_size() >= prefix_tokens_to_include
                            ? token_list.tokens_size() - prefix_tokens_to_include
                            : 0;
        for (size_t j = start_idx; j < token_list.tokens_size(); j++)
            prefix_tokens.push_back(token_list.tokens(j));
    }

    // Process each response and add to the suffix tree
    for (int i = 0; i < request->responses_size(); i++) {
        const auto& token_list = request->responses(i);
        tokens.insert(tokens.end(), prefix_tokens.begin(), prefix_tokens.end());
        for (int j = 0; j < token_list.tokens_size(); j++)
            tokens.push_back(token_list.tokens(j));
        tokens.push_back(-1);
        // std::cout << "  Response " << i << " added to suffix tree, size:" << token_list.tokens_size() << std::endl;
    }

    tree->extend(0, tokens);
    SuffixTree* existing_tree_ptr = nullptr;
    {
        const uint64_t prompt_hash = request->prompt_hash();
        auto mutex = segment_->find<interprocess_mutex>("mutex").first;
        scoped_lock<interprocess_mutex> shm_lock(*mutex);
        auto it = tree_map_->find(prompt_hash);
        uint64_t segment_base = (uint64_t)segment_->get_address();
        uint64_t tree_ptr = (uint64_t)tree - segment_base;
        if (it != tree_map_->end()) {
            // Delete the existing tree
            uint64_t existing_tree = it->second;
            it->second = tree_ptr;
            existing_tree_ptr = (SuffixTree*)(existing_tree + segment_base);
        }
        tree_map_->emplace(prompt_hash, tree_ptr);
    }

    if (existing_tree_ptr) {
        segment_->destroy_ptr(existing_tree_ptr);
    }
    response->set_success(true);

    return grpc::Status::OK;
}

// RolloutCacheServer methods implementation
RolloutCacheServer::RolloutCacheServer(const std::string& server_address,
                                       unsigned long long shared_memory_size_gb)
    : server_address_(server_address.empty() ? "[::]:6378" : server_address),
      shared_memory_size_(shared_memory_size_gb > 0
                          ? shared_memory_size_gb * 1024ULL * 1024ULL * 1024ULL
                          : DEFAULT_SHARED_MEMORY_SIZE),
      segment_(nullptr), tree_map_(nullptr), service_(nullptr) {}

RolloutCacheServer::~RolloutCacheServer() {
    Shutdown();
}

// Initialize the shared memory and create the service
bool RolloutCacheServer::Initialize() {
    try {
        // First try to remove any existing shared memory
        shared_memory_object::remove(SHARED_MEMORY_NAME);

        // Create and initialize the shared memory segment
        segment_ = new managed_shared_memory(create_only, SHARED_MEMORY_NAME, shared_memory_size_);

        // Create a mutex in shared memory
        segment_->construct<interprocess_mutex>("mutex")();

        // Create the allocator for the map
        TreeMapAllocator alloc(segment_->get_segment_manager());

        // Create the map in shared memory
        tree_map_ = segment_->construct<SharedTreeMap>("tree_map")(std::less<uint64_t>(), alloc);

        // Create the service implementation
        service_ = new RolloutCacheServiceImpl(segment_, tree_map_);

        std::cout << "Shared memory initialized: " << SHARED_MEMORY_NAME
                  << " (" << shared_memory_size_ / (1024 * 1024) << " MB)" << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize shared memory: " << e.what() << std::endl;
        if (segment_) {
            delete segment_;
            segment_ = nullptr;
        }
        shared_memory_object::remove(SHARED_MEMORY_NAME);
        return false;
    }
}

// Start the server
bool RolloutCacheServer::Start() {
    // Make sure we've initialized our resources
    if (!segment_ || !tree_map_ || !service_) {
        std::cerr << "Cannot start server: resources not initialized" << std::endl;
        return false;
    }

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    grpc::ServerBuilder builder;
    // Listen on the given address without any authentication
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());

    // Register the service
    builder.RegisterService(service_);

    // Set max threads to twice the number of available CPU cores
    builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, 40);
    // builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, std::thread::hardware_concurrency() * 2);

    // Build and start the server
    server_ = builder.BuildAndStart();
    if (!server_) {
        std::cerr << "Failed to start server on " << server_address_ << std::endl;
        return false;
    }

    std::cout << "Server listening on " << server_address_ << std::endl;
    return true;
}

// Wait for the server to shutdown
void RolloutCacheServer::Wait() {
    if (server_) {
        server_->Wait();
    }
}

// Shutdown the server
void RolloutCacheServer::Shutdown() {
    if (server_) {
        server_->Shutdown();
        server_->Wait(); // Wait for completion
        server_.reset();
    }

    // Clean up service
    if (service_) {
        delete service_;
        service_ = nullptr;
    }

    uint64_t segment_base = (uint64_t)segment_->get_address();
    // Clean up shared memory
    if (segment_) {
        // Clean up all suffix trees
        if (tree_map_) {
            for (auto& pair : *tree_map_) {
                segment_->destroy_ptr((SuffixTree*)(segment_base + pair.second));
            }
            segment_->destroy<SharedTreeMap>("tree_map");
        }

        // Destroy mutex
        segment_->destroy<interprocess_mutex>("mutex");

        delete segment_;
        segment_ = nullptr;

        // Remove the shared memory from the system
        shared_memory_object::remove(SHARED_MEMORY_NAME);
        std::cout << "Shared memory removed: " << SHARED_MEMORY_NAME << std::endl;
    }
}
