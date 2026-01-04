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
Configuration for run-ahead rollout.

Run-ahead rollout fills GPU bubbles by speculatively executing secondary (future batch)
requests while waiting for primary requests to complete.
"""
from dataclasses import dataclass

@dataclass
class RunaheadConfig:
    """Configuration for run-ahead rollout.

    Attributes:
        enabled: Whether run-ahead is enabled.
        load_threshold: Admit secondary requests when server_load < threshold.
            Server load is the count of in-flight requests (running + waiting).
        poll_interval_s: How often the Manager checks for completed tasks and drip-feeds
            secondary requests (seconds).
        max_retries: Maximum retry attempts for rejected secondary requests.
        max_secondary_concurrent: Maximum number of secondary requests in flight at once.
            This is a hard cap to prevent secondary explosion under long primaries.
        use_kv_cache_admission: If True, use vLLM workload polling (kv_cache_usage) as an
            additional safety gate for secondary admission.
        kv_cache_threshold: Reject secondary when kv_cache_usage >= threshold (0.0-1.0).
        workload_poll_interval_s: Background polling interval for server.get_workload().
        workload_staleness_threshold_s: Treat workload metrics as stale after this many seconds.
        require_fresh_workload: If True, reject secondaries until fresh workload metrics exist.
        abort_grace_s: After issuing abort_requests(), wait up to this many seconds for
            in-flight secondary tasks to return partial outputs before force-canceling.
        wait_for_primary_start: If True, delay secondary submission until at least one
            primary request has reached the router (prevents a startup race where
            server_load doesn't yet reflect primary workload).
    """

    enabled: bool = False
    load_threshold: int = 32
    poll_interval_s: float = 0.1
    max_retries: int = 3
    max_secondary_concurrent: int = 8

    # Optional workload-aware admission (kv cache as a coarse safety valve)
    use_kv_cache_admission: bool = False
    kv_cache_threshold: float = 0.85
    workload_poll_interval_s: float = 0.5
    workload_staleness_threshold_s: float = 2.0
    require_fresh_workload: bool = False

    # Abort handling: allow aborted secondaries to return partial outputs.
    abort_grace_s: float = 1.0

    # Admission warm-up: avoid starting secondaries before primaries register load.
    wait_for_primary_start: bool = True
