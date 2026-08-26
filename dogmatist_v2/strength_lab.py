from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class StrengthMode(str, Enum):
    NORMAL = "normal"
    PLATEAU = "plateau"


@dataclass(frozen=True)
class RoundStrengthEvidence:
    round_index: int
    champion_generation: int
    promoted: bool
    fixed_reference_score: float
    paired_games: int


@dataclass(frozen=True)
class StrengthCurriculumMix:
    natural_selfplay: float
    hard_positions: float
    specialist_sparring: float
    deep_search_teacher: float

    def __post_init__(self) -> None:
        values = (self.natural_selfplay, self.hard_positions, self.specialist_sparring, self.deep_search_teacher)
        if any(value < 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("invalid curriculum mix")


@dataclass(frozen=True)
class TrainingBatchBudget:
    natural_selfplay: int
    hard_positions: int
    specialist_sparring: int
    deep_search_teacher: int

    @property
    def total(self) -> int:
        return self.natural_selfplay + self.hard_positions + self.specialist_sparring + self.deep_search_teacher

    def as_dict(self) -> dict[str, int]:
        return {
            "natural_selfplay": self.natural_selfplay,
            "hard_positions": self.hard_positions,
            "specialist_sparring": self.specialist_sparring,
            "deep_search_teacher": self.deep_search_teacher,
            "total": self.total,
        }


@dataclass(frozen=True)
class StrengthLabPlan:
    mode: StrengthMode
    curriculum: StrengthCurriculumMix
    teacher_fraction: float
    teacher_search_multiplier: float
    reason: str

    def batch_budget(self, total_examples: int) -> TrainingBatchBudget:
        """Turn fractional curriculum weights into exact integer quotas.

        Largest-remainder allocation keeps the sum exact and deterministic. The
        production trainer can therefore ask for N examples and receive a concrete
        recipe instead of a descriptive percentage that drifts over time.
        """
        if total_examples <= 0:
            raise ValueError("total_examples must be positive")
        weights = (
            ("natural_selfplay", self.curriculum.natural_selfplay),
            ("hard_positions", self.curriculum.hard_positions),
            ("specialist_sparring", self.curriculum.specialist_sparring),
            ("deep_search_teacher", self.curriculum.deep_search_teacher),
        )
        raw = [(name, total_examples * weight) for name, weight in weights]
        counts = {name: int(value) for name, value in raw}
        left = total_examples - sum(counts.values())
        remainder_order = sorted(
            raw,
            key=lambda item: (item[1] - int(item[1]), item[1]),
            reverse=True,
        )
        for index in range(left):
            counts[remainder_order[index % len(remainder_order)][0]] += 1
        return TrainingBatchBudget(
            natural_selfplay=counts["natural_selfplay"],
            hard_positions=counts["hard_positions"],
            specialist_sparring=counts["specialist_sparring"],
            deep_search_teacher=counts["deep_search_teacher"],
        )

    def ui_payload(self, *, total_examples: int | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "phase": "strength_lab",
            "mode": self.mode.value,
            "reason": self.reason,
            "teacher_fraction": self.teacher_fraction,
            "teacher_search_multiplier": self.teacher_search_multiplier,
            "curriculum": {
                "natural_selfplay": self.curriculum.natural_selfplay,
                "hard_positions": self.curriculum.hard_positions,
                "specialist_sparring": self.curriculum.specialist_sparring,
                "deep_search_teacher": self.curriculum.deep_search_teacher,
            },
        }
        if total_examples is not None:
            payload["batch_budget"] = self.batch_budget(total_examples).as_dict()
        return payload


class PlateauDetector:
    def __init__(self, window_rounds: int = 4, minimum_reference_gain: float = 0.03) -> None:
        self.window_rounds = window_rounds
        self.minimum_reference_gain = minimum_reference_gain

    def is_plateau(self, history: Sequence[RoundStrengthEvidence]) -> bool:
        eligible = [row for row in history if row.paired_games >= 4]
        if len(eligible) < self.window_rounds:
            return False
        window = eligible[-self.window_rounds:]
        promotions = sum(1 for row in window if row.promoted)
        reference_gain = max(row.fixed_reference_score for row in window) - window[0].fixed_reference_score
        return promotions == 0 and reference_gain < self.minimum_reference_gain


class StrengthLabController:
    def __init__(self, detector: PlateauDetector | None = None) -> None:
        self.detector = detector or PlateauDetector()

    def plan(self, history: Sequence[RoundStrengthEvidence]) -> StrengthLabPlan:
        if self.detector.is_plateau(history):
            return StrengthLabPlan(
                StrengthMode.PLATEAU,
                StrengthCurriculumMix(0.40, 0.30, 0.15, 0.15),
                0.15,
                3.0,
                "strength plateau: focus extra compute on difficult positions",
            )
        return StrengthLabPlan(
            StrengthMode.NORMAL,
            StrengthCurriculumMix(0.55, 0.20, 0.15, 0.10),
            0.10,
            2.0,
            "normal strength-growth curriculum",
        )


class EngineGateAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    DEFER = "defer"


@dataclass(frozen=True)
class EngineTrialEvidence:
    candidate_revision_id: str
    paired_games: int
    score_vs_baseline: float
    fixed_reference_delta: float
    compute_cost_ratio: float


@dataclass(frozen=True)
class EngineGateDecision:
    action: EngineGateAction
    reason: str


class EngineRevisionGate:
    """A/B-test search/engine changes on fixed model weights before adoption."""

    def decide(self, evidence: EngineTrialEvidence) -> EngineGateDecision:
        if evidence.compute_cost_ratio > 1.60:
            return EngineGateDecision(EngineGateAction.REJECT, "compute cost is too high")
        if evidence.fixed_reference_delta < -0.03:
            return EngineGateDecision(EngineGateAction.REJECT, "fixed-reference strength regressed")
        if evidence.paired_games < 12:
            return EngineGateDecision(EngineGateAction.DEFER, "need more paired A/B games")
        if evidence.score_vs_baseline < 0.48:
            return EngineGateDecision(EngineGateAction.REJECT, "paired A/B result is weaker")
        if evidence.score_vs_baseline >= 0.55 and evidence.fixed_reference_delta >= 0.0 and evidence.compute_cost_ratio <= 1.30:
            return EngineGateDecision(EngineGateAction.ACCEPT, "stronger at acceptable compute cost")
        return EngineGateDecision(EngineGateAction.DEFER, "evidence is not yet strong enough")
