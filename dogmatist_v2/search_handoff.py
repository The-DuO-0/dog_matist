from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HandoffDiagnosis(str, Enum):
    CANDIDATE_SUPPORTED = "candidate_supported"
    CONTINUATION_MISMATCH = "continuation_mismatch"
    CANDIDATE_HARMFUL = "candidate_harmful"
    MIXED = "mixed"


@dataclass(frozen=True)
class HandoffBranchEvidence:
    """Counterfactual outcomes after forcing r1 or r2c at one divergence.

    All scores are from the perspective of the side that made the forced move.
    A common continuation policy is used after the fork so the move itself can be
    separated from the asymmetric r1-vs-r2c match that originally exposed it.
    """

    shallow_baseline_score: float
    shallow_candidate_score: float
    deep_baseline_score: float
    deep_candidate_score: float

    def __post_init__(self) -> None:
        for value in (
            self.shallow_baseline_score,
            self.shallow_candidate_score,
            self.deep_baseline_score,
            self.deep_candidate_score,
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("branch scores must be in [0, 1]")

    @property
    def shallow_delta(self) -> float:
        return float(self.shallow_candidate_score - self.shallow_baseline_score)

    @property
    def deep_delta(self) -> float:
        return float(self.deep_candidate_score - self.deep_baseline_score)

    @property
    def diagnosis(self) -> HandoffDiagnosis:
        shallow = self.shallow_delta
        deep = self.deep_delta
        if shallow >= 0.0 and deep >= 0.0:
            return HandoffDiagnosis.CANDIDATE_SUPPORTED
        if shallow < 0.0 and deep >= 0.0:
            return HandoffDiagnosis.CONTINUATION_MISMATCH
        if shallow < 0.0 and deep < 0.0:
            return HandoffDiagnosis.CANDIDATE_HARMFUL
        return HandoffDiagnosis.MIXED

    def as_dict(self) -> dict[str, object]:
        return {
            "shallow_baseline_score": self.shallow_baseline_score,
            "shallow_candidate_score": self.shallow_candidate_score,
            "shallow_delta": self.shallow_delta,
            "deep_baseline_score": self.deep_baseline_score,
            "deep_candidate_score": self.deep_candidate_score,
            "deep_delta": self.deep_delta,
            "diagnosis": self.diagnosis.value,
        }
