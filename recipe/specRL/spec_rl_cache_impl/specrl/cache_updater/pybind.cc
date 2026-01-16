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

#include "suffix_cache_updater.h"

// Declare the constant from suffix_cache.cc to make it accessible

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    py::class_<SuffixCacheUpdater>(m, "SuffixCacheUpdater")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>(), py::arg("server_addresses"))
        .def("update_response_cache", &SuffixCacheUpdater::update_response_cache,
             py::arg("prompts"),
             py::arg("responses"),
             py::arg("prompt_lengths"),
             py::arg("response_lengths"),
             py::arg("responses_per_prompt"),
             py::call_guard<py::gil_scoped_release>());
}
