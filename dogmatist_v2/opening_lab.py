from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class OpeningBucketSignal:
    """Model-free evidence that an opening branch deserves repair work.

    Opening names are telemetry labels only. They never prescribe moves and are
    deliberately not treated as an opening book. Unknown/new lines can therefore
    coexist with named buckets and remain eligible for exploration.
    """

    opening_bucket: str
    specialist_generation: int | None = None
    specialist_score: float | None = None
    specialist_games: int = 0
    hard_positions: int = 0
    hard_times_seen: int = 0
    mean_priority: float = 0.0
    max_priority: float = 0.0
    last_seen_round: int = 0

    @property
    def specialist_gap(self) -> float:
        if self.specialist_score is None:
            return 0.0
        raw = max(0.0, float(self.specialist_score) - 0.5) * 2.0
        confidence = min(1.0, max(0, int(self.specialist_games)) / 6.0)
        return min(1.0, raw * confidence)

    @property
    def hard_pressure(self) -> float:
        repeat_bonus = min(0.25, math.log1p(max(0, self.hard_times_seen)) / 12.0)
        pressure = 0.55 * max(0.0, self.max_priority) + 0.30 * max(0.0, self.mean_priority) + repeat_bonus
        return min(1.0, pressure)

    @property
    def weakness_score(self) -> float:
        # A specialist repeatedly outperforming the champion is the strongest
        # signal. Hard-position pressure keeps unnamed/frontier lines visible.
        return min(1.0, 0.65 * self.specialist_gap + 0.35 * self.hard_pressure)

    def as_dict(self) -> dict[str, object]:
        return {
            "opening_bucket": self.opening_bucket,
            "specialist_generation": self.specialist_generation,
            "specialist_score": self.specialist_score,
            "specialist_games": self.specialist_games,
            "hard_positions": self.hard_positions,
            "hard_times_seen": self.hard_times_seen,
            "mean_priority": self.mean_priority,
            "max_priority": self.max_priority,
            "last_seen_round": self.last_seen_round,
            "specialist_gap": self.specialist_gap,
            "hard_pressure": self.hard_pressure,
            "weakness_score": self.weakness_score,
        }


@dataclass(frozen=True)
class OpeningRepairPlan:
    focus_openings: tuple[str, ...]
    signals: tuple[OpeningBucketSignal, ...]
    focus_fraction: float = 0.60
    reason: str = "opening evidence"

    def __post_init__(self) -> None:
        if not 0.0 <= self.focus_fraction <= 0.75:
            raise ValueError("opening focus_fraction must stay between 0 and 0.75")

    @property
    def active(self) -> bool:
        return bool(self.focus_openings)

    def ui_payload(self) -> dict[str, object]:
        return {
            "active": self.active,
            "focus_openings": list(self.focus_openings),
            "focus_fraction": self.focus_fraction,
            "reason": self.reason,
            "signals": [row.as_dict() for row in self.signals],
            "book_moves_injected": False,
            "novel_openings_allowed": True,
        }


class OpeningWeaknessController:
    """Rank weak openings without turning dog_matist into a book engine."""

    def __init__(
        self,
        *,
        max_focus_openings: int = 3,
        minimum_weakness: float = 0.08,
        focus_fraction: float = 0.60,
    ) -> None:
        if max_focus_openings <= 0:
            raise ValueError("max_focus_openings must be positive")
        if minimum_weakness < 0.0:
            raise ValueError("minimum_weakness must be non-negative")
        if not 0.0 <= focus_fraction <= 0.75:
            raise ValueError("focus_fraction must stay between 0 and 0.75")
        self.max_focus_openings = int(max_focus_openings)
        self.minimum_weakness = float(minimum_weakness)
        self.focus_fraction = float(focus_fraction)

    @staticmethod
    def _get(row: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
        try:
            return row[key]
        except Exception:
            return getattr(row, key, default)

    def plan(
        self,
        *,
        specialists: Iterable[Mapping[str, Any] | Any] = (),
        hard_bucket_stats: Iterable[Mapping[str, Any] | Any] = (),
    ) -> OpeningRepairPlan:
        merged: dict[str, dict[str, Any]] = {}

        for row in hard_bucket_stats:
            bucket = str(self._get(row, "opening_bucket", "") or "").strip()
            if not bucket:
                continue
            data = merged.setdefault(bucket, {"opening_bucket": bucket})
            data.update(
                hard_positions=int(self._get(row, "hard_positions", 0) or 0),
                hard_times_seen=int(self._get(row, "hard_times_seen", 0) or 0),
                mean_priority=float(self._get(row, "mean_priority", 0.0) or 0.0),
                max_priority=float(self._get(row, "max_priority", 0.0) or 0.0),
                last_seen_round=int(self._get(row, "last_seen_round", 0) or 0),
            )

        for row in specialists:
            bucket = str(self._get(row, "opening_name", "") or "").strip()
            if not bucket:
                continue
            score = float(self._get(row, "score", 0.0) or 0.0)
            games = int(self._get(row, "games", 0) or 0)
            generation_raw = self._get(row, "generation", None)
            generation = int(generation_raw) if generation_raw is not None else None
            data = merged.setdefault(bucket, {"opening_bucket": bucket})
            previous = data.get("specialist_score")
            if previous is None or score > float(previous):
                data.update(
                    specialist_generation=generation,
                    specialist_score=score,
                    specialist_games=games,
                )

        signals = tuple(
            OpeningBucketSignal(**data)
            for data in merged.values()
        )
        ranked = tuple(
            sorted(
                signals,
                key=lambda row: (row.weakness_score, row.last_seen_round, row.opening_bucket),
                reverse=True,
            )
        )
        focus = tuple(
            row.opening_bucket
            for row in ranked
            if row.weakness_score >= self.minimum_weakness
        )[: self.max_focus_openings]
        reason = (
            "specialist gap + persistent hard-position pressure"
            if focus
            else "no opening bucket has enough evidence yet; preserve broad exploration"
        )
        return OpeningRepairPlan(
            focus_openings=focus,
            signals=ranked[: max(6, self.max_focus_openings)],
            focus_fraction=self.focus_fraction,
            reason=reason,
        )
