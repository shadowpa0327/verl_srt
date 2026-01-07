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
"""
Run-ahead rollout module.

This module provides run-ahead (speculative execution) support for the agent loop,
filling GPU bubbles by executing secondary requests while waiting for primary
requests to complete.

Components:
- RunaheadConfig: Configuration dataclass
- Types: SecondaryOutput, RunaheadResult, RunaheadMetrics, SecondaryWorkItem, RunaheadBatchResult

The router owns the secondary queue and handles admission/retries internally.
The manager submits all secondary items at once via start_runahead_batch(),
runs the primary loop, then calls stop_runahead_batch() to collect results.
"""

from verl.experimental.agent_loop.runahead.config import RunaheadConfig
from verl.experimental.agent_loop.runahead.types import (
    RunaheadBatchResult,
    RunaheadMetrics,
    RunaheadResult,
    SecondaryOutput,
    SecondaryWorkItem,
)

__all__ = [
    "RunaheadConfig",
    "RunaheadBatchResult",
    "SecondaryOutput",
    "RunaheadResult",
    "RunaheadMetrics",
    "SecondaryWorkItem",
]
