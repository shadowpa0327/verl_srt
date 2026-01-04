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
Type definitions for run-ahead rollout.
"""
from dataclasses import dataclass, field
from typing import Literal, Optional

from verl.protocol import DataProto
from verl.workers.rollout.replica import TokenOutput


@dataclass
class SecondaryOutput:
    """Single secondary (runahead) output.

    Attributes:
        sample_id: Unique identifier for this sample.
        output: Token output if started, None if never started.
        status: Current status of the secondary request.
        tokens_generated: Number of tokens generated (partial if aborted).
    """

    sample_id: str
    output: Optional[TokenOutput] = None
    status: Literal["completed", "aborted", "rejected", "pending"] = "pending"
    tokens_generated: int = 0


@dataclass
class RunaheadMetrics:
    """Observability metrics for runahead rollout.

    Attributes:
        primary_time_s: Total time for primary batch completion (seconds).
        secondary_started: Number of secondary requests started.
        secondary_completed: Number of secondary requests completed.
        secondary_aborted: Number of secondary requests aborted.
        secondary_rejected: Number of secondary requests rejected by admission.
        bubble_utilization: Fraction of bubble time utilized (0.0 to 1.0).
    """

    primary_time_s: float = 0.0
    secondary_started: int = 0
    secondary_completed: int = 0
    secondary_aborted: int = 0
    secondary_rejected: int = 0
    bubble_utilization: float = 0.0


@dataclass
class RunaheadResult:
    """Result of runahead rollout.

    Attributes:
        primary_outputs: Primary outputs (guaranteed complete), as a DataProto.
        secondary_outputs: Secondary outputs (may be partial/aborted).
        metrics: Observability metrics.
    """

    primary_outputs: Optional[DataProto] = None
    secondary_outputs: list[SecondaryOutput] = field(default_factory=list)
    metrics: RunaheadMetrics = field(default_factory=RunaheadMetrics)


@dataclass
class SecondaryWorkItem:
    """Work item for runahead secondary generation.

    Attributes:
        sample_id: Unique identifier for this sample.
        prompt_ids: Token IDs for the prompt.
        sampling_params: Sampling parameters for generation.
        retry_count: Number of times this item has been retried.
    """

    sample_id: str
    prompt_ids: list[int]
    sampling_params: dict
    retry_count: int = 0
