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

    The router owns the secondary queue and handles admission internally.
    The manager submits all secondary items at once via start_runahead_batch(),
    runs the primary loop, then calls stop_runahead_batch() to collect results.

    With the router-owned queue model, capacity-based rejection is eliminated:
    items wait in the queue until slack exists. No retry logic is needed.

    Attributes:
        enabled: Whether run-ahead is enabled.
        load_threshold: Admit secondary requests when server_load < threshold.
            Server load is the count of in-flight requests (running + waiting).
            This per-server gate naturally limits total secondaries in flight.
        max_queue_size: Maximum number of pending secondary items in the router queue.
            Oldest items are dropped if queue overflows.
        admit_loop_poll_s: How often the router's admit loop polls for slack (seconds).
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

    # Router queue settings
    max_queue_size: int = 256
    admit_loop_poll_s: float = 0.05

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

    # Priority scheduling (lower value = higher priority)
    # vLLM scheduler must be set to "priority" policy for these to take effect.
    # With priority scheduling, primary requests are processed first and runahead
    # requests are preempted when KV cache is exhausted.
    primary_priority: int = 0        # Primary batch requests get highest priority
    secondary_priority: int = 10     # Runahead requests get lower priority
