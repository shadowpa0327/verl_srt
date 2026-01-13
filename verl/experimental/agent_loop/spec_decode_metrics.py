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

This module is designed to work with the HTTP Prometheus /metrics endpoint
exposed by vLLM's async server mode.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodeSnapshot:
    """Snapshot of spec decode counters at a point in time.

    These values come from vLLM's Prometheus metrics endpoint and are
    monotonically increasing counters.
    """

    num_drafts: float = 0.0
    num_draft_tokens: float = 0.0
    num_accepted_tokens: float = 0.0
    per_pos_accepted: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_prometheus_dict(cls, metrics: dict[str, Any]) -> "SpecDecodeSnapshot":
        """Parse from get_spec_decode_metrics() response.

        Args:
            metrics: Dictionary from vLLMReplica.get_spec_decode_metrics()
                containing spec_decode_* keys.

        Returns:
            SpecDecodeSnapshot with counter values.
        """
        if "error" in metrics:
            logger.debug(f"Metrics fetch returned error: {metrics['error']}")
            return cls()

        return cls(
            num_drafts=metrics.get("spec_decode_num_drafts", 0.0),
            num_draft_tokens=metrics.get("spec_decode_num_draft_tokens", 0.0),
            num_accepted_tokens=metrics.get("spec_decode_num_accepted_tokens", 0.0),
            per_pos_accepted=dict(metrics.get("spec_decode_num_accepted_tokens_per_pos", {})),
        )

    @classmethod
    def aggregate(cls, snapshots: list["SpecDecodeSnapshot"]) -> "SpecDecodeSnapshot":
        """Aggregate multiple snapshots (e.g., from multiple replicas).

        Args:
            snapshots: List of snapshots to aggregate.

        Returns:
            Combined snapshot with summed values.
        """
        aggregated = cls()
        for snapshot in snapshots:
            aggregated.num_drafts += snapshot.num_drafts
            aggregated.num_draft_tokens += snapshot.num_draft_tokens
            aggregated.num_accepted_tokens += snapshot.num_accepted_tokens
            for pos, count in snapshot.per_pos_accepted.items():
                aggregated.per_pos_accepted[pos] = (
                    aggregated.per_pos_accepted.get(pos, 0.0) + count
                )
        return aggregated

    def __sub__(self, other: "SpecDecodeSnapshot") -> "SpecDecodeSnapshot":
        """Compute delta between two snapshots.

        Args:
            other: Earlier snapshot to subtract.

        Returns:
            New snapshot representing the difference.
        """
        # Handle per-position differences
        all_positions = set(self.per_pos_accepted.keys()) | set(other.per_pos_accepted.keys())
        per_pos_delta = {}
        for pos in all_positions:
            delta = self.per_pos_accepted.get(pos, 0.0) - other.per_pos_accepted.get(pos, 0.0)
            if delta != 0:
                per_pos_delta[pos] = delta

        return SpecDecodeSnapshot(
            num_drafts=self.num_drafts - other.num_drafts,
            num_draft_tokens=self.num_draft_tokens - other.num_draft_tokens,
            num_accepted_tokens=self.num_accepted_tokens - other.num_accepted_tokens,
            per_pos_accepted=per_pos_delta,
        )

    def is_empty(self) -> bool:
        """Check if snapshot has no meaningful data."""
        return (
            self.num_drafts == 0
            and self.num_draft_tokens == 0
            and self.num_accepted_tokens == 0
        )


@dataclass
class SpecDecodeRolloutStats:
    """Computed statistics for a single rollout.

    These are the meaningful metrics derived from counter deltas.
    """

    num_drafts: int
    """Number of speculative decoding draft rounds."""

    num_draft_tokens: int
    """Total number of tokens drafted (proposed by the drafter)."""

    num_accepted_tokens: int
    """Total number of drafted tokens that were accepted."""

    acceptance_rate: float
    """Fraction of drafted tokens accepted: accepted_tokens / draft_tokens."""

    mean_accepted_length: float
    """Average tokens accepted per draft round: accepted_tokens / num_drafts.
    Note: Does not include the bonus token from target model."""

    tokens_per_step: float
    """Average tokens generated per forward pass: (accepted + bonus) / drafts.
    This includes the guaranteed bonus token from the target model."""

    per_position_rates: dict[int, float]
    """Acceptance rate at each draft position."""

    @classmethod
    def from_delta(cls, delta: SpecDecodeSnapshot) -> "SpecDecodeRolloutStats":
        """Compute rollout statistics from a delta snapshot.

        Args:
            delta: Difference between after and before snapshots.

        Returns:
            Computed statistics for the rollout.
        """
        num_drafts = int(delta.num_drafts)
        num_draft_tokens = int(delta.num_draft_tokens)
        num_accepted_tokens = int(delta.num_accepted_tokens)

        # Acceptance rate: fraction of drafted tokens accepted
        if num_draft_tokens > 0:
            acceptance_rate = num_accepted_tokens / num_draft_tokens
        else:
            acceptance_rate = 0.0

        # Mean accepted length per draft
        if num_drafts > 0:
            mean_accepted_length = num_accepted_tokens / num_drafts
            # Tokens per step includes the bonus token from target model
            tokens_per_step = (num_accepted_tokens + num_drafts) / num_drafts
        else:
            mean_accepted_length = 0.0
            tokens_per_step = 1.0  # At least 1 token per step without speculation

        # Per-position acceptance rates
        per_position_rates = {}
        if num_drafts > 0:
            for pos, count in sorted(delta.per_pos_accepted.items()):
                per_position_rates[pos] = count / num_drafts

        return cls(
            num_drafts=num_drafts,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
            acceptance_rate=acceptance_rate,
            mean_accepted_length=mean_accepted_length,
            tokens_per_step=tokens_per_step,
            per_position_rates=per_position_rates,
        )

    def to_metrics_dict(self, prefix: str = "spec_decode") -> dict[str, float]:
        """Convert to dictionary for logging/wandb.

        Args:
            prefix: Prefix for metric names (default: "spec_decode").

        Returns:
            Dictionary with metric names and values.
        """
        metrics = {
            f"{prefix}/num_drafts": float(self.num_drafts),
            f"{prefix}/num_draft_tokens": float(self.num_draft_tokens),
            f"{prefix}/num_accepted_tokens": float(self.num_accepted_tokens),
            f"{prefix}/acceptance_rate": self.acceptance_rate,
            f"{prefix}/mean_accepted_length": self.mean_accepted_length,
            f"{prefix}/tokens_per_step": self.tokens_per_step,
        }

        # Add per-position rates (limit to first few positions to avoid metric explosion)
        max_positions = 10
        for pos, rate in sorted(self.per_position_rates.items())[:max_positions]:
            metrics[f"{prefix}/acceptance_rate_pos_{pos}"] = rate

        return metrics

    def is_active(self) -> bool:
        """Check if speculative decoding was active during this rollout."""
        return self.num_drafts > 0
