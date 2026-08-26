from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class OpenTreeRoundTrace:
    """One compact research trace row for an isolated OpenTree round.

    The trace intentionally stores aggregate scalars only. It is designed to be
    emitted as JSONL from the Mac harness without retaining the opening graph in
    RAM. Comparable traces must use the same fixed reference_id.
    """

    round_id: int
    reference_id: str
    arena_score: float
    arena_games: int
    nodes: int
    edges: int
    viable_frontier: int
    strict_holdout: int
    effective_branches: float
    root_top_move_share: float
    branch_survival_ratio: float
    db_bytes: int
    collapse_warning: bool = False
    policy_trial_status: str = "baseline"
    policy_trial_id: int | None = None
    policy_reason: str = ""
    champion_generation: int = 0
    candidate_generation: int | None = None
    arena_wins: int = 0
    arena_draws: int = 0
    arena_losses: int = 0
    training_loss: float | None = None
    policy_natural: float = 0.45
    policy_frontier: float = 0.30
    policy_specialist: float = 0.15
    policy_anchor: float = 0.10
    promotion_action: str = "none"
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.round_id < 0:
            raise ValueError("round_id must be non-negative")
        if not self.reference_id:
            raise ValueError("reference_id is required")
        if not isfinite(self.arena_score) or not 0.0 <= self.arena_score <= 1.0:
            raise ValueError("arena_score must be in [0, 1]")
        if self.arena_games < 0 or self.arena_games % 2 != 0:
            raise ValueError("arena_games must be a non-negative even paired-game count")
        for value, name in (
            (self.nodes, "nodes"),
            (self.edges, "edges"),
            (self.viable_frontier, "viable_frontier"),
            (self.strict_holdout, "strict_holdout"),
            (self.db_bytes, "db_bytes"),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isfinite(self.effective_branches) or self.effective_branches < 0.0:
            raise ValueError("effective_branches must be finite and non-negative")
        for value, name in (
            (self.root_top_move_share, "root_top_move_share"),
            (self.branch_survival_ratio, "branch_survival_ratio"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.policy_trial_status not in {
            "baseline",
            "active",
            "accepted",
            "rolled_back",
            "cooldown",
        }:
            raise ValueError("invalid policy_trial_status")
        if self.policy_trial_id is not None and self.policy_trial_id < 1:
            raise ValueError("policy_trial_id must be positive when present")
        if self.champion_generation < 0:
            raise ValueError("champion_generation must be non-negative")
        if self.candidate_generation is not None and self.candidate_generation < 0:
            raise ValueError("candidate_generation must be non-negative when present")
        if any(v < 0 for v in (self.arena_wins, self.arena_draws, self.arena_losses)):
            raise ValueError("arena W/D/L must be non-negative")
        if self.training_loss is not None and (
            not isfinite(self.training_loss) or self.training_loss < 0.0
        ):
            raise ValueError("training_loss must be finite and non-negative when present")
        policy_values = (
            self.policy_natural,
            self.policy_frontier,
            self.policy_specialist,
            self.policy_anchor,
        )
        if any((not isfinite(v) or v < 0.0) for v in policy_values):
            raise ValueError("policy mix must be finite and non-negative")
        if abs(sum(policy_values) - 1.0) > 1e-6:
            raise ValueError("policy mix must sum to 1")
        if self.promotion_action not in {"none", "promote", "reject", "defer"}:
            raise ValueError("invalid promotion_action")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "OpenTreeRoundTrace":
        return cls(**json.loads(payload))


@dataclass(frozen=True)
class OpenTreeExperimentSummary:
    verdict: str
    rounds: int
    reference_id: str
    arena_score_start: float
    arena_score_end: float
    strength_delta: float
    node_growth: int
    edge_growth: int
    frontier_delta: int
    effective_branch_delta: float
    mean_branch_survival: float
    db_growth_bytes: int
    rollback_count: int
    collapse_rounds: int
    reasons: tuple[str, ...]


class OpenTreeExperimentReport:
    """Summarize a fixed-reference, multi-round OpenTree experiment.

    Verdicts are deliberately conservative:
    - fail: material held-out strength regression;
    - watch: insufficient rounds/evaluation, collapse, poor branch survival, or no graph growth;
    - pass: strength-safe with measurable healthy tree behavior.

    The report is an experiment gate, not a chess-rating estimator.
    """

    def __init__(
        self,
        traces: Iterable[OpenTreeRoundTrace] = (),
        *,
        minimum_rounds: int = 4,
        minimum_arena_games: int = 12,
        max_strength_drop: float = 0.06,
        minimum_branch_survival: float = 0.12,
    ) -> None:
        if minimum_rounds < 2:
            raise ValueError("minimum_rounds must be >= 2")
        if minimum_arena_games < 2 or minimum_arena_games % 2 != 0:
            raise ValueError("minimum_arena_games must be an even number >= 2")
        if not 0.0 <= max_strength_drop <= 0.5:
            raise ValueError("max_strength_drop must be in [0, 0.5]")
        if not 0.0 <= minimum_branch_survival <= 1.0:
            raise ValueError("minimum_branch_survival must be in [0, 1]")
        self.minimum_rounds = minimum_rounds
        self.minimum_arena_games = minimum_arena_games
        self.max_strength_drop = max_strength_drop
        self.minimum_branch_survival = minimum_branch_survival
        self._traces: list[OpenTreeRoundTrace] = []
        for trace in traces:
            self.add(trace)

    @property
    def traces(self) -> tuple[OpenTreeRoundTrace, ...]:
        return tuple(self._traces)

    def add(self, trace: OpenTreeRoundTrace) -> None:
        if self._traces:
            if trace.reference_id != self._traces[0].reference_id:
                raise ValueError("all experiment rounds must use the same fixed reference_id")
            if trace.round_id <= self._traces[-1].round_id:
                raise ValueError("round_id must increase monotonically")
        self._traces.append(trace)

    def summarize(self) -> OpenTreeExperimentSummary:
        if not self._traces:
            raise ValueError("cannot summarize an empty experiment")
        first = self._traces[0]
        last = self._traces[-1]
        strength_delta = last.arena_score - first.arena_score
        survival_values = [
            trace.branch_survival_ratio
            for trace in self._traces[1:]
            if trace.edges > 0
        ]
        mean_survival = mean(survival_values) if survival_values else 0.0
        rollback_count = sum(t.policy_trial_status == "rolled_back" for t in self._traces)
        collapse_rounds = sum(t.collapse_warning for t in self._traces)

        reasons: list[str] = []
        verdict = "pass"
        enough_strength_evidence = all(
            trace.arena_games >= self.minimum_arena_games for trace in self._traces
        )
        if enough_strength_evidence and strength_delta < -self.max_strength_drop:
            verdict = "fail"
            reasons.append("held-out strength regression exceeds budget")
        else:
            if len(self._traces) < self.minimum_rounds:
                verdict = "watch"
                reasons.append("not enough rounds for migration evidence")
            if not enough_strength_evidence:
                verdict = "watch"
                reasons.append("not enough paired Arena evidence in every round")
            if last.collapse_warning:
                verdict = "watch"
                reasons.append("opening concentration warning remains active")
            if last.nodes <= first.nodes or last.edges <= first.edges:
                verdict = "watch"
                reasons.append("OpenTree did not show net graph growth")
            if len(self._traces) >= self.minimum_rounds and mean_survival < self.minimum_branch_survival:
                verdict = "watch"
                reasons.append("new branches are not surviving across rounds")
            if rollback_count >= max(2, len(self._traces) // 2):
                verdict = "watch"
                reasons.append("adaptive policy is rolling back too frequently")

        if not reasons:
            reasons.append("strength-safe tree growth with acceptable branch survival")

        return OpenTreeExperimentSummary(
            verdict=verdict,
            rounds=len(self._traces),
            reference_id=first.reference_id,
            arena_score_start=first.arena_score,
            arena_score_end=last.arena_score,
            strength_delta=strength_delta,
            node_growth=last.nodes - first.nodes,
            edge_growth=last.edges - first.edges,
            frontier_delta=last.viable_frontier - first.viable_frontier,
            effective_branch_delta=last.effective_branches - first.effective_branches,
            mean_branch_survival=mean_survival,
            db_growth_bytes=last.db_bytes - first.db_bytes,
            rollback_count=rollback_count,
            collapse_rounds=collapse_rounds,
            reasons=tuple(reasons),
        )

    def write_jsonl(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for trace in self._traces:
                fh.write(trace.to_json() + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, **kwargs) -> "OpenTreeExperimentReport":
        traces: list[OpenTreeRoundTrace] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    traces.append(OpenTreeRoundTrace.from_json(line))
        return cls(traces, **kwargs)
