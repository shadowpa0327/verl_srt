# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for vLLM suffix decoding patches."""

import importlib.util


def _has_module(module_name: str) -> bool:
    """Check if a module is available."""
    return importlib.util.find_spec(module_name) is not None


def has_arctic_inference() -> bool:
    """Check optional arctic_inference package availability."""
    return _has_module("arctic_inference")
