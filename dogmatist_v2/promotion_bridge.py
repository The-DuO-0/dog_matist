from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .archive import ArchiveEntry, ArchiveTier
from .chronicle_store import ChronicleStore
from .opentree_promotion import PromotionDecision


@dataclass(frozen=True)
class ChampionCheckpoint:
    generation_id: int
    path: Path
    size_bytes: int

    def __post_init__(self) -> None:
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


class PromotionChronicleBridge:
    """Commit a confirmed promotion into the Chronicle without bypassing gates.

    This class deliberately accepts a *PromotionDecision* rather than raw match
    scores. The OpenTree coordinator remains the authority that decides whether
    a candidate is allowed to replace the champion.
    """

    def __init__(self, store: ChronicleStore) -> None:
        self.store = store

    def apply(
        self,
        decision: PromotionDecision,
        *,
        outgoing_checkpoint: ChampionCheckpoint,
        incoming_checkpoint: ChampionCheckpoint,
        occurred_at: datetime,
        evidence: dict[str, object] | None = None,
    ) -> bool:
        if decision.action != "promote":
            raise ValueError("Chronicle succession can only consume a confirmed promote decision")
        if outgoing_checkpoint.generation_id != decision.champion_generation:
            raise ValueError("outgoing checkpoint does not match promotion champion")
        if incoming_checkpoint.generation_id != decision.candidate_generation:
            raise ValueError("incoming checkpoint does not match promotion candidate")

        outgoing = ArchiveEntry(
            generation_id=decision.champion_generation,
            tier=ArchiveTier.IMMORTAL,
            checkpoint_path=outgoing_checkpoint.path,
            checkpoint_bytes=outgoing_checkpoint.size_bytes,
            ever_champion=True,
            protected=True,
            reason="historical champion; preserved after succession",
        )
        incoming = ArchiveEntry(
            generation_id=decision.candidate_generation,
            tier=ArchiveTier.ACTIVE,
            checkpoint_path=incoming_checkpoint.path,
            checkpoint_bytes=incoming_checkpoint.size_bytes,
            ever_champion=True,
            protected=True,
            reason="current champion",
        )
        return self.store.record_champion_succession(
            outgoing=outgoing,
            incoming=incoming,
            occurred_at=occurred_at,
            replacement_reason=decision.reason,
            evidence={
                "promotion_action": decision.action,
                **(evidence or {}),
            },
        )
