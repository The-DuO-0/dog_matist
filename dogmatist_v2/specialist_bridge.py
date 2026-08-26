from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from .archive import ArchiveEntry, ArchiveTier
from .chronicle_store import ChronicleStore
from .specialists import SpecialistRecord


_GEN_RE = re.compile(r"^(?:gen)?\s*(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SpecialistCheckpoint:
    path: Path
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


def parse_generation_id(candidate_id: str) -> int:
    match = _GEN_RE.match(candidate_id.strip())
    if match is None:
        raise ValueError(f"candidate id is not a generation id: {candidate_id!r}")
    return int(match.group(1))


class SpecialistChronicleBridge:
    """Persist useful non-champions as traits and optional cold model bodies."""

    def __init__(
        self,
        store: ChronicleStore,
        *,
        minimum_preserve_advantage: float = 0.08,
    ) -> None:
        if minimum_preserve_advantage < 0.0:
            raise ValueError("minimum_preserve_advantage must be non-negative")
        self.store = store
        self.minimum_preserve_advantage = minimum_preserve_advantage

    def apply(
        self,
        records: Iterable[SpecialistRecord],
        *,
        checkpoints: Mapping[str, SpecialistCheckpoint] | None = None,
        occurred_at: datetime,
    ) -> tuple[int, ...]:
        checkpoint_map = checkpoints or {}
        preserved: set[int] = set()

        for record in records:
            generation_id = parse_generation_id(record.candidate_id)
            self.store.record_trait(
                generation_id,
                "opening",
                record.bucket.key,
                record.advantage_over_reference,
                sample_games=record.games,
                evidence={
                    "score": record.score,
                    "points": record.points,
                    "advantage_over_reference": record.advantage_over_reference,
                    "bucket_name": record.bucket.name,
                },
                updated_at=occurred_at,
            )

            if record.advantage_over_reference < self.minimum_preserve_advantage:
                continue

            checkpoint = checkpoint_map.get(record.candidate_id)
            if checkpoint is None:
                entry = ArchiveEntry(
                    generation_id=generation_id,
                    tier=ArchiveTier.HISTORY_ONLY,
                    checkpoint_path=None,
                    checkpoint_bytes=0,
                    specialist_score=record.advantage_over_reference,
                    reason=f"opening specialist {record.bucket.key}; metadata retained, checkpoint unavailable",
                )
            else:
                entry = ArchiveEntry(
                    generation_id=generation_id,
                    tier=ArchiveTier.COLD,
                    checkpoint_path=checkpoint.path,
                    checkpoint_bytes=checkpoint.size_bytes,
                    specialist_score=record.advantage_over_reference,
                    protected=True,
                    reason=f"opening specialist {record.bucket.key}",
                )
                preserved.add(generation_id)
            self.store.upsert_archive_entry(entry, archived_at=occurred_at)

        return tuple(sorted(preserved))
