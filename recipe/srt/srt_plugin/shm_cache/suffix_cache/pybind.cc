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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "suffix_tree.h"
#include "suffix_cache.h"
#include "rollout_cache_server.h"  // Include the RolloutCacheServer implementation

// Declare the constant from suffix_cache.cc to make it accessible

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {

    py::class_<SuffixSpecResult>(m, "SuffixSpecResult")
        .def(py::init<>())
        .def_readwrite("token_ids", &SuffixSpecResult::token_ids)
        .def_readwrite("parents", &SuffixSpecResult::parents)
        .def_readwrite("probs", &SuffixSpecResult::probs)
        .def_readwrite("score", &SuffixSpecResult::score)
        .def_readwrite("match_len", &SuffixSpecResult::match_len)
        .def_static("from_candidate", &SuffixSpecResult::from_candidate,
                   py::call_guard<py::gil_scoped_release>());

    py::class_<SuffixCache>(m, "SuffixCache")
        .def(py::init<const std::string&, int, int>(),
             py::arg("shared_memory_name") = "",
             py::arg("spec_start_len") = 2,
             py::arg("spec_max_len") = 16,
             "Create a SuffixCache.\n\n"
             "Args:\n"
             "    shared_memory_name: Name of the shared memory segment to connect to.\n"
             "                        Defaults to 'SUFFIX_CACHE' if empty.\n"
             "    spec_start_len: Initial/minimum speculation length (default: 2).\n"
             "    spec_max_len: Maximum speculation length (default: 16).")
        .def("fetch_responses_by_prompts_batch", &SuffixCache::fetch_responses_by_prompts_batch,
             py::call_guard<py::gil_scoped_release>())
        .def("update_spec_len", &SuffixCache::update_spec_len,
             py::call_guard<py::gil_scoped_release>())
        .def("evict_responses", &SuffixCache::evict_responses,
             py::call_guard<py::gil_scoped_release>())
        .def("speculate", &SuffixCache::speculate,
             py::arg("req_id"),
             py::arg("pattern"),
             py::arg("min_token_prob") = 0.1f,
             py::arg("use_tree_spec") = false,
             py::call_guard<py::gil_scoped_release>());

    // Register the RolloutCacheServer class
    py::class_<RolloutCacheServer>(m, "RolloutCacheServer")
        .def(py::init<const std::string&, unsigned long long, const std::string&>(),
             py::arg("server_address"),
             py::arg("shared_memory_size_gb") = 0,
             py::arg("shared_memory_name") = "",
             "Create a RolloutCacheServer.\n\n"
             "Args:\n"
             "    server_address: gRPC server address (e.g., '[::]:6378')\n"
             "    shared_memory_size_gb: Shared memory size in GB (default: 500GB)\n"
             "    shared_memory_name: Name for the shared memory segment (default: 'SUFFIX_CACHE')")
        .def("initialize", &RolloutCacheServer::Initialize,
             py::call_guard<py::gil_scoped_release>(),
             "Initialize the shared memory and create the service")
        .def("start", &RolloutCacheServer::Start,
             py::call_guard<py::gil_scoped_release>(),
             "Start the gRPC server")
        .def("wait", &RolloutCacheServer::Wait,
             py::call_guard<py::gil_scoped_release>(),
             "Wait for the server to shutdown")
        .def("shutdown", &RolloutCacheServer::Shutdown,
             py::call_guard<py::gil_scoped_release>(),
             "Shutdown the server and clean up resources")
        .def("get_shared_memory_name", &RolloutCacheServer::get_shared_memory_name,
             "Get the name of the shared memory segment");
}
