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
"""Per-rollout speculative decoding metrics using delta calculation.

vLLM's Prometheus counters are monotonically increasing. To get per-rollout
statistics, we snapshot counters before/after each rollout and compute deltas.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodeSnapshot:
    """Snapshot of spec decode counters at a point in time."""

    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    per_pos_accepted: List[int] = field(default_factory=list)

    @classmethod
    def from_vllm_metrics(cls, metrics: List[Any]) -> "SpecDecodeSnapshot":
        """Parse vLLM metrics list into snapshot."""
        snapshot = cls()
        for metric in metrics:
            name = getattr(metric, "name", None)
            if name == "vllm:spec_decode_num_drafts":
                snapshot.num_drafts = int(getattr(metric, "value", 0))
            elif name == "vllm:spec_decode_num_draft_tokens":
                snapshot.num_draft_tokens = int(getattr(metric, "value", 0))
            elif name == "vllm:spec_decode_num_accepted_tokens":
                snapshot.num_accepted_tokens = int(getattr(metric, "value", 0))
            elif name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                snapshot.per_pos_accepted = [int(v) for v in getattr(metric, "values", [])]
        return snapshot

    def __sub__(self, other: "SpecDecodeSnapshot") -> "SpecDecodeSnapshot":
        """Compute delta between two snapshots."""
        # Handle different length per_pos arrays
        max_len = max(len(self.per_pos_accepted), len(other.per_pos_accepted))
        per_pos_delta = []
        for i in range(max_len):
            a = self.per_pos_accepted[i] if i < len(self.per_pos_accepted) else 0
            b = other.per_pos_accepted[i] if i < len(other.per_pos_accepted) else 0
            per_pos_delta.append(a - b)

        return SpecDecodeSnapshot(
            num_drafts=self.num_drafts - other.num_drafts,
            num_draft_tokens=self.num_draft_tokens - other.num_draft_tokens,
            num_accepted_tokens=self.num_accepted_tokens - other.num_accepted_tokens,
            per_pos_accepted=per_pos_delta,
        )


@dataclass
class SpecDecodeRolloutStats:
    """Computed statistics for a single rollout."""

    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    acceptance_rate: float  # accepted_tokens / draft_tokens
    mean_acceptance_length: float  # 1 + (accepted_tokens / num_drafts)
    per_position_rates: List[float]  # acceptance probability at each position

    @classmethod
    def from_delta(cls, delta: SpecDecodeSnapshot) -> "SpecDecodeRolloutStats":
        """Compute rollout statistics from a delta snapshot."""
        # Acceptance rate: what fraction of drafted tokens were accepted
        if delta.num_draft_tokens > 0:
            acceptance_rate = delta.num_accepted_tokens / delta.num_draft_tokens
        else:
            acceptance_rate = 0.0

        # Mean acceptance length: average tokens accepted per draft + 1 (bonus token)
        if delta.num_drafts > 0:
            mean_acceptance_length = 1.0 + (delta.num_accepted_tokens / delta.num_drafts)
            per_pos_rates = [cnt / delta.num_drafts for cnt in delta.per_pos_accepted]
        else:
            mean_acceptance_length = 1.0
            per_pos_rates = []

        return cls(
            num_drafts=delta.num_drafts,
            num_draft_tokens=delta.num_draft_tokens,
            num_accepted_tokens=delta.num_accepted_tokens,
            acceptance_rate=acceptance_rate,
            mean_acceptance_length=mean_acceptance_length,
            per_position_rates=per_pos_rates,
        )


class SpecDecodeMetricsTracker:
    """Track spec decode metrics across rollouts using delta calculation.

    Usage:
        tracker = SpecDecodeMetricsTracker(llm_engine)

        # Before rollout
        tracker.start_rollout()

        # ... run generation ...

        # After rollout
        stats = tracker.end_rollout()
        print(f"Acceptance rate: {stats.acceptance_rate:.3f}")
    """

    def __init__(self, engine):
        """Initialize tracker with vLLM LLM engine.

        Args:
            engine: vLLM LLM instance (must have llm_engine.get_metrics())
        """
        self.engine = engine
        self._last_snapshot: Optional[SpecDecodeSnapshot] = None
        self._metrics_enabled = True

        # Verify metrics are available
        try:
            _ = self.engine.llm_engine.get_metrics()
        except (AttributeError, AssertionError) as e:
            logger.warning(f"Metrics not available, tracker disabled: {e}")
            self._metrics_enabled = False

    def _get_snapshot(self) -> SpecDecodeSnapshot:
        """Get current snapshot from engine."""
        if not self._metrics_enabled:
            return SpecDecodeSnapshot()

        try:
            metrics = self.engine.llm_engine.get_metrics()
            return SpecDecodeSnapshot.from_vllm_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to get metrics snapshot: {e}")
            return SpecDecodeSnapshot()

    def start_rollout(self):
        """Call before starting a rollout to mark baseline.

        This snapshots the current counter values so we can compute
        the delta after the rollout completes.
        """
        self._last_snapshot = self._get_snapshot()

    def end_rollout(self) -> SpecDecodeRolloutStats:
        """Call after rollout to get per-rollout statistics.

        Returns:
            SpecDecodeRolloutStats with metrics for just this rollout.
        """
        current = self._get_snapshot()

        if self._last_snapshot is None:
            # First call without start_rollout, use zero baseline
            self._last_snapshot = SpecDecodeSnapshot()

        delta = current - self._last_snapshot
        self._last_snapshot = current  # Update baseline for next rollout

        return SpecDecodeRolloutStats.from_delta(delta)

    def is_enabled(self) -> bool:
        """Check if metrics tracking is enabled."""
        return self._metrics_enabled
