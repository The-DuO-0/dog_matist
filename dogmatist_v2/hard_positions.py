from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HardPositionCandidate:
    position_key: str
    opening_bucket: str
    phase: str
    result_score: float
    value_error: float
    policy_surprise: float
    repeat_count: int = 1

    def __post_init__(self) -> None:
        if not self.position_key:
            raise ValueError("position_key must be non-empty")
        if not 0.0 <= self.result_score <= 1.0:
            raise ValueError("result_score must be in [0, 1]")
        if self.value_error < 0.0 or self.policy_surprise < 0.0:
            raise ValueError("difficulty signals must be non-negative")
        if self.repeat_count <= 0:
            raise ValueError("repeat_count must be positive")

    @property
    def priority(self) -> float:
        loss_pressure = 1.0 - self.result_score
        repeated_failure = min(1.0, 0.15 * max(0, self.repeat_count - 1))
        return (
            0.40 * loss_pressure
            + 0.35 * min(1.0, self.value_error)
            + 0.20 * min(1.0, self.policy_surprise)
            + 0.05 * repeated_failure
        )


class HardPositionMiner:
    """Select difficult training positions while retaining opening diversity."""

    def select(
        self,
        candidates: Iterable[HardPositionCandidate],
        *,
        limit: int,
        max_per_opening_bucket: int = 8,
    ) -> tuple[HardPositionCandidate, ...]:
        if limit <= 0:
            return ()
        if max_per_opening_bucket <= 0:
            raise ValueError("max_per_opening_bucket must be positive")

        best_by_position: dict[str, HardPositionCandidate] = {}
        for candidate in candidates:
            previous = best_by_position.get(candidate.position_key)
            if previous is None or candidate.priority > previous.priority:
                best_by_position[candidate.position_key] = candidate

        selected: list[HardPositionCandidate] = []
        bucket_counts: dict[str, int] = {}
        ordered = sorted(best_by_position.values(), key=lambda row: row.priority, reverse=True)
        for candidate in ordered:
            count = bucket_counts.get(candidate.opening_bucket, 0)
            if count >= max_per_opening_bucket:
                continue
            selected.append(candidate)
            bucket_counts[candidate.opening_bucket] = count + 1
            if len(selected) >= limit:
                break
        return tuple(selected)
