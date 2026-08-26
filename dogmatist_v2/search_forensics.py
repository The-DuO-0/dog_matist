from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SearchForensicRow:
    fen: str
    ply: int
    baseline_move: str | None
    candidate_move: str | None
    full_deeper_move: str | None
    full_best_score_cp: float | None = None
    full_baseline_score_cp: float | None = None
    full_candidate_score_cp: float | None = None

    @property
    def candidate_matches_full(self) -> bool:
        return self.candidate_move is not None and self.candidate_move == self.full_deeper_move

    @property
    def baseline_matches_full(self) -> bool:
        return self.baseline_move is not None and self.baseline_move == self.full_deeper_move

    @property
    def candidate_regret_cp(self) -> float | None:
        if self.full_best_score_cp is None or self.full_candidate_score_cp is None:
            return None
        return max(0.0, float(self.full_best_score_cp) - float(self.full_candidate_score_cp))

    @property
    def baseline_regret_cp(self) -> float | None:
        if self.full_best_score_cp is None or self.full_baseline_score_cp is None:
            return None
        return max(0.0, float(self.full_best_score_cp) - float(self.full_baseline_score_cp))

    @property
    def classification(self) -> str:
        if self.candidate_matches_full and not self.baseline_matches_full:
            return "helpful_flip"
        if self.baseline_matches_full and not self.candidate_matches_full:
            return "harmful_flip"
        if self.candidate_matches_full and self.baseline_matches_full:
            return "same_as_full"
        return "both_differ_from_full"


@dataclass(frozen=True)
class SearchForensicsSummary:
    positions: int
    helpful_flips: int
    harmful_flips: int
    same_as_full: int
    both_differ_from_full: int
    candidate_full_match_rate: float
    baseline_full_match_rate: float
    mean_candidate_regret_cp: float | None
    mean_baseline_regret_cp: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "positions": self.positions,
            "helpful_flips": self.helpful_flips,
            "harmful_flips": self.harmful_flips,
            "same_as_full": self.same_as_full,
            "both_differ_from_full": self.both_differ_from_full,
            "candidate_full_match_rate": self.candidate_full_match_rate,
            "baseline_full_match_rate": self.baseline_full_match_rate,
            "mean_candidate_regret_cp": self.mean_candidate_regret_cp,
            "mean_baseline_regret_cp": self.mean_baseline_regret_cp,
        }


def summarize_search_forensics(rows: Iterable[SearchForensicRow]) -> SearchForensicsSummary:
    seq = tuple(rows)
    n = len(seq)
    counts = {
        "helpful_flip": 0,
        "harmful_flip": 0,
        "same_as_full": 0,
        "both_differ_from_full": 0,
    }
    for row in seq:
        counts[row.classification] += 1

    candidate_matches = sum(1 for row in seq if row.candidate_matches_full)
    baseline_matches = sum(1 for row in seq if row.baseline_matches_full)
    candidate_regrets = [row.candidate_regret_cp for row in seq if row.candidate_regret_cp is not None]
    baseline_regrets = [row.baseline_regret_cp for row in seq if row.baseline_regret_cp is not None]

    return SearchForensicsSummary(
        positions=n,
        helpful_flips=counts["helpful_flip"],
        harmful_flips=counts["harmful_flip"],
        same_as_full=counts["same_as_full"],
        both_differ_from_full=counts["both_differ_from_full"],
        candidate_full_match_rate=(candidate_matches / n) if n else 0.0,
        baseline_full_match_rate=(baseline_matches / n) if n else 0.0,
        mean_candidate_regret_cp=(sum(candidate_regrets) / len(candidate_regrets)) if candidate_regrets else None,
        mean_baseline_regret_cp=(sum(baseline_regrets) / len(baseline_regrets)) if baseline_regrets else None,
    )
