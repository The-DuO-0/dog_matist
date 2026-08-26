from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PromotionAction = Literal["promote", "reject", "defer"]


@dataclass(frozen=True)
class PromotionEvidence:
    """Evidence required before a candidate may replace the live champion.

    During an adaptive OpenTree policy trial, candidates are treated as shadow
    generations. Passing the ordinary challenger-vs-champion Arena is necessary
    but not sufficient: the fixed-reference strength guard must also accept the
    curriculum trial before champion state can move.
    """

    candidate_generation: int
    champion_generation: int
    standard_gate_promoted: bool
    standard_gate_score: float
    policy_trial_active: bool = False
    policy_guard_accepts: bool | None = None
    policy_guard_reason: str = ""

    def __post_init__(self) -> None:
        if self.candidate_generation < 0 or self.champion_generation < 0:
            raise ValueError("generation ids must be non-negative")
        if not 0.0 <= self.standard_gate_score <= 1.0:
            raise ValueError("standard_gate_score must be in [0, 1]")
        if not self.policy_trial_active and self.policy_guard_accepts is not None:
            raise ValueError("policy_guard_accepts is only meaningful during a policy trial")


@dataclass(frozen=True)
class PromotionDecision:
    action: PromotionAction
    candidate_generation: int
    champion_generation: int
    reason: str


class OpenTreePromotionCoordinator:
    """Prevent adaptive-curriculum experiments from mutating champion early."""

    def decide(self, evidence: PromotionEvidence) -> PromotionDecision:
        if not evidence.standard_gate_promoted:
            return PromotionDecision(
                "reject",
                evidence.candidate_generation,
                evidence.champion_generation,
                "candidate failed the ordinary champion Arena",
            )

        if not evidence.policy_trial_active:
            return PromotionDecision(
                "promote",
                evidence.candidate_generation,
                evidence.champion_generation,
                "ordinary promotion gate passed; no curriculum trial is active",
            )

        if evidence.policy_guard_accepts is None:
            return PromotionDecision(
                "defer",
                evidence.candidate_generation,
                evidence.champion_generation,
                "curriculum trial is active but fixed-reference evidence is incomplete",
            )

        if evidence.policy_guard_accepts:
            return PromotionDecision(
                "promote",
                evidence.candidate_generation,
                evidence.champion_generation,
                evidence.policy_guard_reason or "ordinary gate and fixed-reference strength guard both passed",
            )

        return PromotionDecision(
            "reject",
            evidence.candidate_generation,
            evidence.champion_generation,
            evidence.policy_guard_reason or "curriculum strength guard rejected the shadow candidate",
        )
