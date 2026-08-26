from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TrialEvidence:
    """Evidence collected after trying one OpenTree curriculum policy.

    Strength is deliberately measured by paired held-out chess evaluation, not
    by training loss. Diversity metrics are secondary objectives: a prettier
    tree is not allowed to justify a material strength regression.

    `reference_id` identifies the fixed opponent/reference used for the Arena
    score. Baseline and trial evidence must use the same reference; otherwise a
    score delta is not interpretable as curriculum evidence.
    """

    arena_score: float
    arena_games: int
    effective_branches: float
    viable_frontier: int
    collapse_warning: bool = False
    reference_id: str = "champion"

    def __post_init__(self) -> None:
        if not isfinite(self.arena_score) or not 0.0 <= self.arena_score <= 1.0:
            raise ValueError("arena_score must be in [0, 1]")
        if self.arena_games < 0 or self.viable_frontier < 0:
            raise ValueError("counts must be non-negative")
        if self.arena_games % 2 != 0:
            raise ValueError("paired Arena evidence must contain an even number of games")
        if not isfinite(self.effective_branches) or self.effective_branches < 0.0:
            raise ValueError("effective_branches must be finite and non-negative")
        if not self.reference_id:
            raise ValueError("reference_id must be non-empty")


@dataclass(frozen=True)
class GuardDecision:
    accept_policy: bool
    strength_delta: float
    diversity_delta: float
    reason: str


class OpenTreeStrengthGuard:
    """Gate adaptive curriculum changes with chess-strength evidence.

    The adaptive OpenTree controller can propose more exploration when the tree
    collapses. This guard prevents that policy from becoming a diversity-only
    optimizer. A trial policy is accepted only after a minimum paired Arena
    sample and only if its strength is not materially worse than the baseline.

    Small negative deltas are tolerated because short Arenas are noisy. A trial
    that improves diversity but exceeds the configured strength-loss budget is
    rolled back. Conversely, a strength-neutral trial need not improve diversity
    immediately when it is recovering from an already-collapsed tree.
    """

    def __init__(
        self,
        *,
        minimum_games: int = 12,
        max_strength_drop: float = 0.06,
        minimum_diversity_gain: float = 0.05,
        frontier_gain_weight: float = 0.002,
    ) -> None:
        if minimum_games < 2 or minimum_games % 2 != 0:
            raise ValueError("minimum_games must be an even number >= 2")
        if not 0.0 <= max_strength_drop <= 0.5:
            raise ValueError("max_strength_drop must be in [0, 0.5]")
        if minimum_diversity_gain < 0.0 or frontier_gain_weight < 0.0:
            raise ValueError("diversity parameters must be non-negative")
        self.minimum_games = minimum_games
        self.max_strength_drop = max_strength_drop
        self.minimum_diversity_gain = minimum_diversity_gain
        self.frontier_gain_weight = frontier_gain_weight

    def _diversity_score(self, evidence: TrialEvidence) -> float:
        return evidence.effective_branches + self.frontier_gain_weight * evidence.viable_frontier

    def decide(self, baseline: TrialEvidence, trial: TrialEvidence) -> GuardDecision:
        strength_delta = trial.arena_score - baseline.arena_score
        diversity_delta = self._diversity_score(trial) - self._diversity_score(baseline)

        if baseline.reference_id != trial.reference_id:
            return GuardDecision(
                False,
                strength_delta,
                diversity_delta,
                "Arena reference changed; score delta is not comparable",
            )

        if baseline.arena_games < self.minimum_games or trial.arena_games < self.minimum_games:
            return GuardDecision(
                False,
                strength_delta,
                diversity_delta,
                "insufficient paired Arena evidence; keep baseline policy",
            )

        if strength_delta < -self.max_strength_drop:
            return GuardDecision(
                False,
                strength_delta,
                diversity_delta,
                "strength regression exceeds exploration budget; rollback policy",
            )

        # If the baseline tree is already collapsed, a strength-safe trial is
        # allowed some time to work even before entropy visibly rebounds.
        if baseline.collapse_warning:
            return GuardDecision(
                True,
                strength_delta,
                diversity_delta,
                "strength-safe trial while recovering from opening collapse",
            )

        if diversity_delta >= self.minimum_diversity_gain:
            return GuardDecision(
                True,
                strength_delta,
                diversity_delta,
                "strength preserved and OpenTree coverage improved",
            )

        if strength_delta > 0.03:
            return GuardDecision(
                True,
                strength_delta,
                diversity_delta,
                "clear strength improvement; retain policy despite flat diversity",
            )

        return GuardDecision(
            False,
            strength_delta,
            diversity_delta,
            "no meaningful strength or diversity gain; keep baseline policy",
        )
