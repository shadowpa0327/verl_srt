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

#ifndef ROLLOUT_CACHE_SERVER_H
#define ROLLOUT_CACHE_SERVER_H

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <vector>

// Boost.Interprocess headers
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/containers/map.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "rollout-cache.grpc.pb.h"
#include "suffix_tree.h"

// Mutex for thread-safe console output
extern std::mutex console_mutex;

// Constants
const char* const SHARED_MEMORY_NAME = "SUFFIX_CACHE";
const unsigned long long DEFAULT_SHARED_MEMORY_SIZE = 500ULL * 1024ULL * 1024ULL * 1024ULL;  // 500GB default

// Using declarations for Boost.Interprocess
using namespace boost::interprocess;

// Using specrl namespace for proto types
using namespace specrl;

// Shared memory hash map for suffix trees
typedef offset_ptr<SuffixTree> SuffixTreePtr;
typedef std::pair<const uint64_t, uint64_t> TreeMapPair;
typedef allocator<TreeMapPair, managed_shared_memory::segment_manager> TreeMapAllocator;
typedef map<uint64_t, uint64_t, std::less<uint64_t>, TreeMapAllocator> SharedTreeMap;

// Implementation of RolloutCacheService
class RolloutCacheServiceImpl final : public RolloutCacheService::Service {
public:
    RolloutCacheServiceImpl(managed_shared_memory* segment, SharedTreeMap* tree_map)
        : segment_(segment), tree_map_(tree_map){}

    grpc::Status UpdateCache(grpc::ServerContext* context,
                            const UpdateCacheRequest* request,
                            UpdateCacheResponse* response) override;

private:
    managed_shared_memory* segment_;
    SharedTreeMap* tree_map_;
};

// RolloutCacheServer class to manage the gRPC server
class RolloutCacheServer {
public:
    RolloutCacheServer(const std::string& server_address,
                       unsigned long long shared_memory_size_gb = 0);
    ~RolloutCacheServer();

    // Initialize the shared memory and create the service
    bool Initialize();

    // Start the server
    bool Start();

    // Wait for the server to shutdown
    void Wait();

    // Shutdown the server
    void Shutdown();

private:
    std::string server_address_;
    unsigned long long shared_memory_size_;
    managed_shared_memory* segment_;
    SharedTreeMap* tree_map_;
    RolloutCacheServiceImpl* service_;
    std::unique_ptr<grpc::Server> server_;
};

#endif // ROLLOUT_CACHE_SERVER_H
