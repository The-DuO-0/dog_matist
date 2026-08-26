from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .opentree_guard import GuardDecision, OpenTreeStrengthGuard, TrialEvidence
from .opentree_policy import CurriculumMix, OpenTreePolicy


@dataclass(frozen=True)
class PolicyTrial:
    trial_id: int
    baseline_policy: OpenTreePolicy
    trial_policy: OpenTreePolicy
    baseline_evidence: TrialEvidence


@dataclass(frozen=True)
class TrialResult:
    trial_id: int
    accepted_policy: OpenTreePolicy
    decision: GuardDecision
    rolled_back: bool


def _policy_to_dict(policy: OpenTreePolicy) -> dict[str, Any]:
    return {
        "mix": policy.mix.as_dict(),
        "early_temperature_scale": policy.early_temperature_scale,
        "frontier_gap_cp": policy.frontier_gap_cp,
        "reason": policy.reason,
    }


def _policy_from_dict(data: dict[str, Any]) -> OpenTreePolicy:
    mix = data["mix"]
    return OpenTreePolicy(
        mix=CurriculumMix(
            float(mix["natural"]),
            float(mix["frontier"]),
            float(mix["specialist"]),
            float(mix["anchor"]),
        ),
        early_temperature_scale=float(data["early_temperature_scale"]),
        frontier_gap_cp=int(data["frontier_gap_cp"]),
        reason=str(data.get("reason", "restored OpenTree policy")),
    )


def _evidence_to_dict(evidence: TrialEvidence) -> dict[str, Any]:
    return {
        "arena_score": evidence.arena_score,
        "arena_games": evidence.arena_games,
        "effective_branches": evidence.effective_branches,
        "viable_frontier": evidence.viable_frontier,
        "collapse_warning": evidence.collapse_warning,
        "reference_id": evidence.reference_id,
    }


def _evidence_from_dict(data: dict[str, Any]) -> TrialEvidence:
    return TrialEvidence(
        arena_score=float(data["arena_score"]),
        arena_games=int(data["arena_games"]),
        effective_branches=float(data["effective_branches"]),
        viable_frontier=int(data["viable_frontier"]),
        collapse_warning=bool(data.get("collapse_warning", False)),
        reference_id=str(data.get("reference_id", "champion")),
    )


class OpenTreePolicyTrialManager:
    """Two-phase rollout for adaptive opening curriculum changes.

    A controller proposal is not treated as permanent immediately. The runtime
    starts a trial, runs one bounded training/evaluation window, then asks the
    strength guard whether the new policy earned the right to persist.

    Rejected trials enter a short cooldown. This prevents repeatedly retrying a
    diversity-heavy policy that just failed the chess-strength gate. The small
    state machine can also be snapshotted to JSON/SQLite metadata so a laptop
    sleep/restart does not silently forget an active experiment.
    """

    SNAPSHOT_VERSION = 1

    def __init__(
        self,
        *,
        guard: OpenTreeStrengthGuard | None = None,
        rejection_cooldown_rounds: int = 2,
    ) -> None:
        if rejection_cooldown_rounds < 0:
            raise ValueError("rejection_cooldown_rounds must be non-negative")
        self.guard = guard or OpenTreeStrengthGuard()
        self.rejection_cooldown_rounds = rejection_cooldown_rounds
        self._active: PolicyTrial | None = None
        self._next_id = 1
        self._cooldown = 0

    @property
    def active(self) -> PolicyTrial | None:
        return self._active

    @property
    def cooldown_rounds(self) -> int:
        return self._cooldown

    @property
    def can_start(self) -> bool:
        return self._active is None and self._cooldown == 0

    def tick_round(self) -> None:
        if self._active is None and self._cooldown > 0:
            self._cooldown -= 1

    def start(
        self,
        *,
        baseline_policy: OpenTreePolicy,
        trial_policy: OpenTreePolicy,
        baseline_evidence: TrialEvidence,
    ) -> PolicyTrial:
        if self._active is not None:
            raise RuntimeError("an OpenTree policy trial is already active")
        if self._cooldown > 0:
            raise RuntimeError("OpenTree policy trials are cooling down after a rejection")
        trial = PolicyTrial(
            trial_id=self._next_id,
            baseline_policy=baseline_policy,
            trial_policy=trial_policy,
            baseline_evidence=baseline_evidence,
        )
        self._next_id += 1
        self._active = trial
        return trial

    def finish(self, trial_evidence: TrialEvidence) -> TrialResult:
        trial = self._active
        if trial is None:
            raise RuntimeError("no active OpenTree policy trial")
        decision = self.guard.decide(trial.baseline_evidence, trial_evidence)
        accepted = trial.trial_policy if decision.accept_policy else trial.baseline_policy
        rolled_back = not decision.accept_policy
        if rolled_back:
            self._cooldown = self.rejection_cooldown_rounds
        self._active = None
        return TrialResult(
            trial_id=trial.trial_id,
            accepted_policy=accepted,
            decision=decision,
            rolled_back=rolled_back,
        )

    def snapshot(self) -> dict[str, Any]:
        active: dict[str, Any] | None = None
        if self._active is not None:
            active = {
                "trial_id": self._active.trial_id,
                "baseline_policy": _policy_to_dict(self._active.baseline_policy),
                "trial_policy": _policy_to_dict(self._active.trial_policy),
                "baseline_evidence": _evidence_to_dict(self._active.baseline_evidence),
            }
        return {
            "version": self.SNAPSHOT_VERSION,
            "next_id": self._next_id,
            "cooldown_rounds": self._cooldown,
            "rejection_cooldown_rounds": self.rejection_cooldown_rounds,
            "active": active,
        }

    @classmethod
    def restore(
        cls,
        data: dict[str, Any],
        *,
        guard: OpenTreeStrengthGuard | None = None,
    ) -> "OpenTreePolicyTrialManager":
        if int(data.get("version", 0)) != cls.SNAPSHOT_VERSION:
            raise ValueError("unsupported OpenTree policy-trial snapshot version")
        manager = cls(
            guard=guard,
            rejection_cooldown_rounds=int(data.get("rejection_cooldown_rounds", 2)),
        )
        manager._next_id = max(1, int(data.get("next_id", 1)))
        manager._cooldown = max(0, int(data.get("cooldown_rounds", 0)))
        active = data.get("active")
        if active is not None:
            manager._active = PolicyTrial(
                trial_id=int(active["trial_id"]),
                baseline_policy=_policy_from_dict(active["baseline_policy"]),
                trial_policy=_policy_from_dict(active["trial_policy"]),
                baseline_evidence=_evidence_from_dict(active["baseline_evidence"]),
            )
            manager._next_id = max(manager._next_id, manager._active.trial_id + 1)
        return manager
