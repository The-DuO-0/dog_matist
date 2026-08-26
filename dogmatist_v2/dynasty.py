from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Iterable


class HistoricalRole(StrEnum):
    CHAMPION = "champion"
    SPECIALIST = "specialist"
    ELITE = "elite"
    ORDINARY = "ordinary"


@dataclass(frozen=True)
class GenerationLife:
    """Durable identity and lifetime boundaries for one generation.

    A generation can outlive its champion reign.  `retired_at=None` means it is
    still active somewhere in the league/archive rather than that its age is
    unknown.
    """

    generation_id: int
    parent_id: int | None
    born_at: datetime
    retired_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if self.parent_id is not None and self.parent_id < 0:
            raise ValueError("parent_id must be non-negative when present")
        _require_aware(self.born_at, "born_at")
        if self.retired_at is not None:
            _require_aware(self.retired_at, "retired_at")
            if self.retired_at < self.born_at:
                raise ValueError("retired_at cannot precede born_at")

    def lifetime_seconds(self, now: datetime | None = None) -> float:
        end = self.retired_at or now or datetime.now(timezone.utc)
        _require_aware(end, "now")
        if end < self.born_at:
            raise ValueError("lifetime end cannot precede born_at")
        return (end - self.born_at).total_seconds()


@dataclass(frozen=True)
class ChampionReign:
    """One contiguous champion reign.

    Keeping reigns separate from generation lifetime lets the Chronicle answer
    both "how long did this model exist?" and "how long did it rule?" without
    conflating the two.
    """

    generation_id: int
    started_at: datetime
    ended_at: datetime | None = None
    dethroned_by: int | None = None
    challengers_faced: int = 0
    games_during_reign: int = 0

    def __post_init__(self) -> None:
        if self.generation_id < 0:
            raise ValueError("generation_id must be non-negative")
        if self.dethroned_by is not None and self.dethroned_by < 0:
            raise ValueError("dethroned_by must be non-negative when present")
        if self.challengers_faced < 0 or self.games_during_reign < 0:
            raise ValueError("reign counters must be non-negative")
        _require_aware(self.started_at, "started_at")
        if self.ended_at is not None:
            _require_aware(self.ended_at, "ended_at")
            if self.ended_at < self.started_at:
                raise ValueError("ended_at cannot precede started_at")

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def duration_seconds(self, now: datetime | None = None) -> float:
        end = self.ended_at or now or datetime.now(timezone.utc)
        _require_aware(end, "now")
        if end < self.started_at:
            raise ValueError("reign end cannot precede started_at")
        return (end - self.started_at).total_seconds()


@dataclass(frozen=True)
class HistoricalEvent:
    event_id: int | None
    occurred_at: datetime
    kind: str
    generation_id: int | None
    text: str
    related_generation_id: int | None = None

    def __post_init__(self) -> None:
        _require_aware(self.occurred_at, "occurred_at")
        if not self.kind.strip():
            raise ValueError("kind is required")
        if not self.text.strip():
            raise ValueError("text is required")


@dataclass(frozen=True)
class GenerationChronicle:
    life: GenerationLife
    roles: tuple[HistoricalRole, ...]
    reigns: tuple[ChampionReign, ...] = ()
    noteworthy_traits: tuple[str, ...] = ()

    @property
    def total_reign_seconds(self) -> float:
        return sum(reign.duration_seconds() for reign in self.reigns if not reign.active)

    @property
    def ever_champion(self) -> bool:
        return HistoricalRole.CHAMPION in self.roles or bool(self.reigns)


def build_lineage_path(
    generation_id: int,
    generations: Iterable[GenerationLife],
) -> tuple[int, ...]:
    """Return oldest-known ancestor -> requested generation.

    Cycle detection is explicit because corrupt lineage is worse than an
    incomplete UI.  Missing ancestors terminate the path rather than inventing
    data.
    """

    by_id = {g.generation_id: g for g in generations}
    path: list[int] = []
    seen: set[int] = set()
    current: int | None = generation_id
    while current is not None:
        if current in seen:
            raise ValueError(f"lineage cycle detected at generation {current}")
        seen.add(current)
        path.append(current)
        row = by_id.get(current)
        if row is None:
            break
        current = row.parent_id
    return tuple(reversed(path))


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
