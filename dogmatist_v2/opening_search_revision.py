from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .opening_stability import OpeningSearchStabilityReport


@dataclass(frozen=True)
class CandidateScore:
    move: str
    score_cp: float


@dataclass(frozen=True)
class OpeningSearchEvidence:
    """Cheap evidence available after the normal shallow opening search.

    This object deliberately contains no opening names or human-book concepts.
    The revision is allowed to spend more search, but it is never allowed to
    force a known opening move.
    """

    ply: int
    base_depth: int
    best_move: str
    best_score_cp: float | None = None
    candidates: tuple[CandidateScore, ...] = ()
    previous_iteration_move: str | None = None
    previous_iteration_score_cp: float | None = None

    @property
    def candidate_margin_cp(self) -> float | None:
        if len(self.candidates) < 2:
            return None
        ordered = sorted((row.score_cp for row in self.candidates), reverse=True)
        return float(ordered[0] - ordered[1])

    @property
    def iteration_move_flip(self) -> bool:
        return (
            self.previous_iteration_move is not None
            and self.previous_iteration_move != self.best_move
        )

    @property
    def iteration_score_swing_cp(self) -> float | None:
        if self.best_score_cp is None or self.previous_iteration_score_cp is None:
            return None
        return abs(float(self.best_score_cp) - float(self.previous_iteration_score_cp))


@dataclass(frozen=True)
class OpeningDeepeningDecision:
    deepen: bool
    target_depth: int
    reason: str


@dataclass(frozen=True)
class OpeningSearchR2Policy:
    """Confidence-gated opening-only +1-ply stabilization candidate.

    r2a verified too many early plies and cost ~1.65x wall time. r2b removed
    unconditional verification, but the first Mac smoke still cost ~1.61x.
    r2c keeps this confidence gate and separately limits the *root moves* that
    receive the expensive +1-ply verification.

    No opening names, book moves, or Stockfish knowledge are used here. The
    shallow search must provide its own evidence that another ply is worth the
    compute.
    """

    opening_plies: int = 8
    always_verify_plies: int = 0
    extra_depth: int = 1
    candidate_margin_cp: float = 18.0
    iteration_swing_cp: float = 60.0
    move_flip_min_swing_cp: float = 45.0
    max_extra_searches: int = 3

    def __post_init__(self) -> None:
        if self.opening_plies <= 0:
            raise ValueError("opening_plies must be positive")
        if self.always_verify_plies < 0 or self.always_verify_plies > self.opening_plies:
            raise ValueError("always_verify_plies must be within opening_plies")
        if self.extra_depth <= 0:
            raise ValueError("extra_depth must be positive")
        if (
            self.candidate_margin_cp < 0
            or self.iteration_swing_cp < 0
            or self.move_flip_min_swing_cp < 0
        ):
            raise ValueError("thresholds must be non-negative")
        if self.max_extra_searches < 0:
            raise ValueError("max_extra_searches must be non-negative")

    def decide(
        self,
        evidence: OpeningSearchEvidence,
        *,
        extra_searches_used: int = 0,
    ) -> OpeningDeepeningDecision:
        target = int(evidence.base_depth) + int(self.extra_depth)
        if evidence.ply <= 0:
            raise ValueError("ply must be positive")
        if evidence.base_depth <= 0:
            raise ValueError("base_depth must be positive")
        if evidence.ply > self.opening_plies:
            return OpeningDeepeningDecision(False, target, "outside opening stabilization window")
        if extra_searches_used >= self.max_extra_searches:
            return OpeningDeepeningDecision(False, target, "per-game extra-search budget exhausted")
        if evidence.ply <= self.always_verify_plies:
            return OpeningDeepeningDecision(True, target, "configured root opening verification")

        swing = evidence.iteration_score_swing_cp
        margin = evidence.candidate_margin_cp
        if (
            evidence.iteration_move_flip
            and swing is not None
            and swing >= self.move_flip_min_swing_cp
        ):
            return OpeningDeepeningDecision(True, target, "move flip with meaningful score swing")
        if swing is not None and swing >= self.iteration_swing_cp:
            return OpeningDeepeningDecision(True, target, "large iterative score swing")
        if margin is not None and margin <= self.candidate_margin_cp:
            return OpeningDeepeningDecision(True, target, "shallow candidate margin is small")
        return OpeningDeepeningDecision(False, target, "shallow opening search looks stable")


@dataclass
class OpeningSearchR2Session:
    """Track the revision's per-game compute budget for one searcher instance."""

    policy: OpeningSearchR2Policy
    extra_searches_used: int = 0
    last_absolute_ply: int | None = None
    total_extra_searches: int = 0

    def prepare_position(self, absolute_ply: int) -> None:
        if absolute_ply <= 0:
            raise ValueError("absolute_ply must be positive")
        if self.last_absolute_ply is not None and absolute_ply <= self.last_absolute_ply:
            self.extra_searches_used = 0
        self.last_absolute_ply = int(absolute_ply)

    def decide(self, evidence: OpeningSearchEvidence) -> OpeningDeepeningDecision:
        self.prepare_position(evidence.ply)
        decision = self.policy.decide(
            evidence,
            extra_searches_used=self.extra_searches_used,
        )
        if decision.deepen:
            self.extra_searches_used += 1
            self.total_extra_searches += 1
        return decision


def absolute_game_ply(*, fullmove_number: int, white_to_move: bool) -> int:
    """Return a 1-based absolute half-move index from FEN counters."""

    if fullmove_number <= 0:
        raise ValueError("fullmove_number must be positive")
    completed = (int(fullmove_number) - 1) * 2
    return completed + (1 if white_to_move else 2)


def select_verification_candidates(
    candidates: Iterable[CandidateScore],
    *,
    previous_iteration_move: str | None = None,
    min_candidates: int = 4,
    max_candidates: int = 8,
    score_window_cp: float = 90.0,
) -> tuple[str, ...]:
    """Choose a bounded root set for the expensive +1-ply verification.

    The shallow search remains the source of all candidates. We always retain a
    small top set, then keep additional near-best moves up to ``max_candidates``.
    The previous iterative-deepening best move is also preserved because the real
    Gen54 pathology can hide a good move below the shallow top few. This is not an
    opening book: no move is named or preferred by chess theory.
    """

    if min_candidates < 1:
        raise ValueError("min_candidates must be positive")
    if max_candidates < min_candidates:
        raise ValueError("max_candidates must be >= min_candidates")
    if score_window_cp < 0:
        raise ValueError("score_window_cp must be non-negative")

    rows = sorted(
        (CandidateScore(str(row.move), float(row.score_cp)) for row in candidates),
        key=lambda row: row.score_cp,
        reverse=True,
    )
    if not rows:
        return ()

    best_score = rows[0].score_cp
    selected: list[str] = []
    for index, row in enumerate(rows):
        if len(selected) >= max_candidates:
            break
        if index < min_candidates or row.score_cp >= best_score - score_window_cp:
            if row.move not in selected:
                selected.append(row.move)

    previous = str(previous_iteration_move) if previous_iteration_move else None
    if previous and previous not in selected:
        if len(selected) < max_candidates:
            selected.append(previous)
        else:
            selected[-1] = previous

    deduped: list[str] = []
    for move in selected:
        if move not in deduped:
            deduped.append(move)
    return tuple(deduped[:max_candidates])


def select_holdout_opening_names(
    opening_names: Iterable[str],
    *,
    excluded_names: Iterable[str],
    pair_count: int,
    seed: int,
) -> tuple[str, ...]:
    """Select a deterministic, disjoint opening holdout for engine-revision gates.

    This helper is deliberately ignorant of move quality. It only prevents the
    final gate from reusing opening names already seen during revision tuning.
    The returned names are shuffled by a dedicated seed and contain no duplicates.
    """

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    excluded = {str(name) for name in excluded_names}
    pool = sorted({str(name) for name in opening_names if str(name) not in excluded})
    if len(pool) < pair_count:
        raise ValueError("not enough disjoint opening names for requested holdout")
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    return tuple(pool[:pair_count])


@dataclass(frozen=True)
class OpeningSearchRevisionPlan:
    revision_id: str
    generation: int
    enabled: bool
    policy: OpeningSearchR2Policy
    evidence_positions: int
    observed_flip_rate: float
    observed_horizon_sensitive: int
    reason: str

    @classmethod
    def from_stability_report(
        cls,
        report: OpeningSearchStabilityReport,
        *,
        revision_id: str = "search-r2c-selective-root",
        policy: OpeningSearchR2Policy | None = None,
    ) -> "OpeningSearchRevisionPlan":
        chosen = policy or OpeningSearchR2Policy()
        enabled = bool(report.early_search_unstable)
        reason = (
            f"opening probe unstable: flips={report.move_flips}/{len(report.observations)}, "
            f"horizon_sensitive={report.horizon_sensitive}"
            if enabled
            else "opening probe did not justify extra search"
        )
        return cls(
            revision_id=revision_id,
            generation=int(report.generation),
            enabled=enabled,
            policy=chosen,
            evidence_positions=len(report.observations),
            observed_flip_rate=float(report.flip_rate),
            observed_horizon_sensitive=int(report.horizon_sensitive),
            reason=reason,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "revision_id": self.revision_id,
            "generation": self.generation,
            "enabled": self.enabled,
            "evidence_positions": self.evidence_positions,
            "observed_flip_rate": self.observed_flip_rate,
            "observed_horizon_sensitive": self.observed_horizon_sensitive,
            "reason": self.reason,
            "policy": {
                "opening_plies": self.policy.opening_plies,
                "always_verify_plies": self.policy.always_verify_plies,
                "extra_depth": self.policy.extra_depth,
                "candidate_margin_cp": self.policy.candidate_margin_cp,
                "iteration_swing_cp": self.policy.iteration_swing_cp,
                "move_flip_min_swing_cp": self.policy.move_flip_min_swing_cp,
                "max_extra_searches": self.policy.max_extra_searches,
                "book_moves_injected": False,
            },
        }


def candidate_scores(rows: Iterable[tuple[str, float]]) -> tuple[CandidateScore, ...]:
    return tuple(CandidateScore(str(move), float(score)) for move, score in rows)
