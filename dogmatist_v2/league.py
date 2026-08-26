from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    """One member of an evolution population.

    A candidate is intentionally more than a scalar generation number: its role
    lets the trainer pursue different niches without requiring separate full-size
    GPU trainers.
    """

    candidate_id: str
    parent_id: str
    role: str = "balanced"
    checkpoint: str | None = None


@dataclass(frozen=True)
class MatchResult:
    white_id: str
    black_id: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    opening: str | None = None
    plies: int | None = None

    def score_for(self, candidate_id: str) -> float:
        if candidate_id not in (self.white_id, self.black_id):
            raise ValueError(f"{candidate_id!r} did not play this match")
        if self.result == "1/2-1/2":
            return 0.5
        white_won = self.result == "1-0"
        if self.result not in ("1-0", "0-1"):
            raise ValueError(f"invalid chess result: {self.result}")
        return 1.0 if (candidate_id == self.white_id) == white_won else 0.0


@dataclass
class LeagueRow:
    candidate_id: str
    games: int = 0
    points: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    rating: float = 1500.0

    @property
    def score(self) -> float:
        return self.points / self.games if self.games else 0.0


@dataclass
class LeagueTable:
    rows: dict[str, LeagueRow] = field(default_factory=dict)
    k_factor: float = 20.0

    def _row(self, candidate_id: str) -> LeagueRow:
        return self.rows.setdefault(candidate_id, LeagueRow(candidate_id))

    @staticmethod
    def _expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def add(self, match: MatchResult) -> None:
        white = self._row(match.white_id)
        black = self._row(match.black_id)
        ws = match.score_for(match.white_id)
        bs = 1.0 - ws
        we = self._expected(white.rating, black.rating)
        be = 1.0 - we
        old_wr, old_br = white.rating, black.rating

        white.games += 1
        black.games += 1
        white.points += ws
        black.points += bs
        for row, score in ((white, ws), (black, bs)):
            if score == 1.0:
                row.wins += 1
            elif score == 0.5:
                row.draws += 1
            else:
                row.losses += 1

        # Update symmetrically from pre-game expectations.
        white.rating = old_wr + self.k_factor * (ws - we)
        black.rating = old_br + self.k_factor * (bs - be)

    @classmethod
    def from_matches(cls, matches: Iterable[MatchResult]) -> "LeagueTable":
        table = cls()
        for match in matches:
            table.add(match)
        return table

    def ranking(self) -> list[LeagueRow]:
        return sorted(
            self.rows.values(),
            key=lambda r: (r.rating, r.score, r.games),
            reverse=True,
        )


def select_survivors(
    candidates: Iterable[Candidate],
    table: LeagueTable,
    *,
    champion_id: str,
    elite_count: int = 2,
    minimum_games: int = 4,
) -> list[Candidate]:
    """Return a small breeding pool rather than a winner-take-all survivor.

    The reigning champion is retained unless it is absent from the supplied
    candidate set. Remaining slots go to the highest league performers that
    have enough evaluation games. Specialist retention is handled separately
    by SpecialistArchive so an opening expert can survive even without making
    this overall elite set.
    """

    by_id = {c.candidate_id: c for c in candidates}
    survivors: list[Candidate] = []
    if champion_id in by_id:
        survivors.append(by_id[champion_id])

    for row in table.ranking():
        if len(survivors) >= max(1, elite_count):
            break
        if row.games < minimum_games or row.candidate_id == champion_id:
            continue
        candidate = by_id.get(row.candidate_id)
        if candidate is not None:
            survivors.append(candidate)
    return survivors
