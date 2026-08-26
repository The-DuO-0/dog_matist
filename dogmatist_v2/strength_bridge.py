from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .strength_store import HardPositionEvidence, StrengthStore


@dataclass(frozen=True)
class PositionObservation:
    """Minimal production-trainer trace needed by the Strength Lab.

    Values are from the side-to-move perspective. ``predicted_value`` and
    ``outcome_value`` use [-1, 1]; policy uncertainty uses [0, 1].
    """

    fen: str
    opening_bucket: str
    source_generation: int
    round_index: int
    predicted_value: float
    outcome_value: float
    policy_uncertainty: float
    source_kind: str = "selfplay"

    def __post_init__(self) -> None:
        if not self.fen.strip():
            raise ValueError("fen must be non-empty")
        if self.source_generation < 0 or self.round_index < 0:
            raise ValueError("generation and round ids must be non-negative")
        if not -1.0 <= self.predicted_value <= 1.0:
            raise ValueError("predicted_value must be in [-1, 1]")
        if not -1.0 <= self.outcome_value <= 1.0:
            raise ValueError("outcome_value must be in [-1, 1]")
        if not 0.0 <= self.policy_uncertainty <= 1.0:
            raise ValueError("policy_uncertainty must be in [0, 1]")

    @property
    def value_error(self) -> float:
        return abs(self.predicted_value - self.outcome_value) / 2.0

    @property
    def severity(self) -> float:
        return (1.0 - self.outcome_value) / 2.0

    def to_evidence(self) -> HardPositionEvidence:
        return HardPositionEvidence(
            fen=self.fen,
            opening_bucket=self.opening_bucket or "unknown",
            source_generation=self.source_generation,
            source_kind=self.source_kind,
            severity=self.severity,
            uncertainty=self.policy_uncertainty,
            value_error=self.value_error,
            round_index=self.round_index,
        )


@dataclass(frozen=True)
class StrengthCapturePolicy:
    minimum_value_error: float = 0.25
    minimum_policy_uncertainty: float = 0.65
    minimum_loss_severity: float = 0.75
    max_positions_per_game: int = 12
    max_per_opening_bucket: int = 128

    def __post_init__(self) -> None:
        thresholds = (
            self.minimum_value_error,
            self.minimum_policy_uncertainty,
            self.minimum_loss_severity,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("capture thresholds must be in [0, 1]")
        if self.max_positions_per_game <= 0 or self.max_per_opening_bucket <= 0:
            raise ValueError("capture caps must be positive")

    def qualifies(self, row: PositionObservation) -> bool:
        return (
            row.value_error >= self.minimum_value_error
            or row.policy_uncertainty >= self.minimum_policy_uncertainty
            or row.severity >= self.minimum_loss_severity
        )


class StrengthEvidenceBridge:
    """Convert real game traces into a bounded, persistent weakness memory."""

    def __init__(
        self,
        store: StrengthStore,
        policy: StrengthCapturePolicy | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or StrengthCapturePolicy()

    def ingest_game(
        self,
        observations: Iterable[PositionObservation],
        *,
        observed_at: datetime,
    ) -> tuple[HardPositionEvidence, ...]:
        candidates = [row for row in observations if self.policy.qualifies(row)]
        candidates.sort(
            key=lambda row: (
                row.value_error * 0.45
                + row.policy_uncertainty * 0.25
                + row.severity * 0.30
            ),
            reverse=True,
        )
        selected = candidates[: self.policy.max_positions_per_game]
        evidence_rows: list[HardPositionEvidence] = []
        for row in selected:
            evidence = row.to_evidence()
            self.store.upsert_hard_position(
                evidence,
                observed_at=observed_at,
                max_per_bucket=self.policy.max_per_opening_bucket,
            )
            evidence_rows.append(evidence)
        return tuple(evidence_rows)
