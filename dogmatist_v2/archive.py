from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable


class ArchiveTier(StrEnum):
    """How much of a generation is retained.

    ACTIVE is allowed to be resident in RAM.  All other tiers are cold by
    definition and must only be loaded explicitly by the trainer/evaluator.
    """

    ACTIVE = "active"
    IMMORTAL = "immortal"
    PRESERVED = "preserved"
    COLD = "cold"
    HISTORY_ONLY = "history_only"


@dataclass(frozen=True)
class ArchiveEntry:
    generation_id: int
    tier: ArchiveTier
    checkpoint_path: Path | None
    checkpoint_bytes: int = 0
    ever_champion: bool = False
    specialist_score: float = 0.0
    protected: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if self.checkpoint_bytes < 0:
            raise ValueError("checkpoint_bytes must be non-negative")
        if self.tier is ArchiveTier.HISTORY_ONLY and self.checkpoint_path is not None:
            raise ValueError("history-only entries cannot retain a checkpoint path")
        if self.tier is ArchiveTier.HISTORY_ONLY and self.checkpoint_bytes:
            raise ValueError("history-only entries cannot retain checkpoint bytes")
        if self.checkpoint_path is None and self.checkpoint_bytes:
            raise ValueError("checkpoint_bytes requires checkpoint_path")

    @property
    def has_model_body(self) -> bool:
        return self.checkpoint_path is not None

    @property
    def ram_resident(self) -> bool:
        return self.tier is ArchiveTier.ACTIVE

    @property
    def auto_deletable(self) -> bool:
        # Historic champions and explicit protected specimens are museum pieces.
        return self.has_model_body and not self.ever_champion and not self.protected


@dataclass(frozen=True)
class ArchivePolicy:
    """Deterministic disk-budget policy for saved model bodies.

    The policy intentionally avoids delta-checkpoint chains in the first
    Dynasty build.  Compact model-only checkpoints are safer: optimizer state
    is omitted from cold storage, while the active training checkpoint can keep
    whatever full state it needs.
    """

    disk_budget_bytes: int
    minimum_specialist_score: float = 0.0

    def __post_init__(self) -> None:
        if self.disk_budget_bytes < 0:
            raise ValueError("disk_budget_bytes must be non-negative")

    def retained_bytes(self, entries: Iterable[ArchiveEntry]) -> int:
        return sum(entry.checkpoint_bytes for entry in entries if entry.has_model_body)

    def over_budget_bytes(self, entries: Iterable[ArchiveEntry]) -> int:
        return max(0, self.retained_bytes(entries) - self.disk_budget_bytes)

    def prune_plan(self, entries: Iterable[ArchiveEntry]) -> tuple[int, ...]:
        """Return generation ids whose model bodies may be dropped.

        Low-value, large, non-champion specimens are removed first.  The caller
        is responsible for converting them to HISTORY_ONLY atomically after the
        file deletion succeeds.
        """

        snapshot = tuple(entries)
        excess = self.over_budget_bytes(snapshot)
        if excess <= 0:
            return ()

        candidates = [
            e
            for e in snapshot
            if e.auto_deletable and e.tier is not ArchiveTier.ACTIVE
        ]
        candidates.sort(
            key=lambda e: (
                e.specialist_score >= self.minimum_specialist_score,
                e.specialist_score,
                -e.checkpoint_bytes,
                e.generation_id,
            )
        )

        freed = 0
        selected: list[int] = []
        for entry in candidates:
            selected.append(entry.generation_id)
            freed += entry.checkpoint_bytes
            if freed >= excess:
                break
        return tuple(selected)


def choose_archive_tier(
    *,
    active: bool,
    ever_champion: bool,
    preserve_requested: bool,
    specialist_score: float,
    minimum_specialist_score: float,
) -> ArchiveTier:
    if active:
        return ArchiveTier.ACTIVE
    if ever_champion:
        return ArchiveTier.IMMORTAL
    if preserve_requested:
        return ArchiveTier.PRESERVED
    if specialist_score >= minimum_specialist_score:
        return ArchiveTier.COLD
    return ArchiveTier.HISTORY_ONLY


@dataclass(frozen=True)
class CompactCheckpointPlan:
    """Describes what to serialize when a model leaves the active pool."""

    generation_id: int
    keep_model_weights: bool
    keep_optimizer_state: bool = False
    compression: bool = True

    @classmethod
    def for_tier(cls, generation_id: int, tier: ArchiveTier) -> "CompactCheckpointPlan":
        return cls(
            generation_id=generation_id,
            keep_model_weights=tier is not ArchiveTier.HISTORY_ONLY,
            # Optimizer moments can dwarf the useful historical artifact and
            # are unnecessary for evaluation/sparring/inheritance-by-weights.
            keep_optimizer_state=tier is ArchiveTier.ACTIVE,
            compression=tier is not ArchiveTier.ACTIVE,
        )
