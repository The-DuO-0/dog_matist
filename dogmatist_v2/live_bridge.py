from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable

from .strength_pipeline import DeepSearchTeacherRequest
from .strength_store import HardPositionEvidence, StrengthStore


@dataclass(frozen=True)
class CapturedHardPosition:
    evidence: HardPositionEvidence
    priority: float
    ply_index: int


@dataclass(frozen=True)
class TeacherSearchBudget:
    max_depth: int
    time_limit_s: float


@dataclass(frozen=True)
class TeacherReplayTarget:
    fen: str
    move_uci: str
    value_target: float
    policy_weight: float
    priority: float
    teacher_score_cp: float
    baseline_score_cp: float
    teacher_depth: int
    teacher_nodes: int
    teacher_elapsed_s: float


def cp_to_value(score_cp: float, *, cp_scale: float = 650.0) -> float:
    if cp_scale <= 0:
        raise ValueError("cp_scale must be positive")
    return math.tanh(float(score_cp) / cp_scale)


class LiveGameEvidenceBridge:
    """Convert production GameRecord data into compact StrengthStore evidence.

    Ordinary hard-position mining skips the stochastic opening exploration window.
    A separate opening lane observes early positions, but does not call a
    played-vs-best mismatch a mistake: production self-play may choose a different
    legal move on purpose to explore. Opening severity comes from the *early search
    evaluation*, not merely the eventual game result, so a move-40 blunder cannot
    make an otherwise healthy opening look automatically terrible.
    """

    _GENERIC_OPENING_LABELS = {"", "unknown", "initial position", "start", "starting position"}

    def __init__(
        self,
        store: StrengthStore,
        *,
        cp_scale: float = 650.0,
        skip_opening_plies: int = 6,
        max_positions_per_game: int = 12,
        minimum_priority: float = 0.24,
        opening_max_plies: int = 12,
        opening_positions_per_game: int = 6,
        opening_minimum_priority: float = 0.16,
    ) -> None:
        if cp_scale <= 0:
            raise ValueError("cp_scale must be positive")
        if skip_opening_plies < 0:
            raise ValueError("skip_opening_plies must be non-negative")
        if max_positions_per_game <= 0:
            raise ValueError("max_positions_per_game must be positive")
        if opening_max_plies <= 0 or opening_positions_per_game <= 0:
            raise ValueError("opening evidence limits must be positive")
        self.store = store
        self.cp_scale = cp_scale
        self.skip_opening_plies = skip_opening_plies
        self.max_positions_per_game = max_positions_per_game
        self.minimum_priority = minimum_priority
        self.opening_max_plies = opening_max_plies
        self.opening_positions_per_game = opening_positions_per_game
        self.opening_minimum_priority = opening_minimum_priority

    @staticmethod
    def _move_text(example: Any, name: str) -> str | None:
        raw = getattr(example, name, None)
        if raw is None:
            return None
        text = raw.uci() if hasattr(raw, "uci") else str(raw)
        return text if text else None

    def opening_bucket_for_record(self, record: Any) -> str:
        metadata = getattr(record, "metadata", {}) or {}
        named = str(metadata.get("opening_name") or metadata.get("opening_family") or "").strip()
        if named.lower() not in self._GENERIC_OPENING_LABELS:
            return named

        # Unknown/new openings remain first-class. Hash a short move trace instead
        # of forcing it into an existing human opening name or injecting book moves.
        trace: list[str] = []
        for example in list(getattr(record, "examples", ()) or ())[:8]:
            move = self._move_text(example, "played_move_uci") or self._move_text(example, "move_uci")
            if move:
                trace.append(move)
            if len(trace) >= 6:
                break
        seed = " ".join(trace) or "unclassified"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
        return f"frontier:{digest}"

    def _evidence_from_example(
        self,
        example: Any,
        *,
        opening: str,
        generation: int,
        round_index: int,
        source_kind: str,
        suppress_exploration_regret: bool,
        opening_evidence: bool = False,
    ) -> HardPositionEvidence | None:
        search_score = getattr(example, "search_score_cp", None)
        best_score = getattr(example, "best_score_cp", None)
        if search_score is None:
            return None
        value_target = float(getattr(example, "value_target", 0.0))
        predicted = cp_to_value(float(search_score), cp_scale=self.cp_scale)
        value_error = min(1.0, abs(predicted - value_target) / 2.0)

        # General hard-position mining can use the eventual result as loss
        # pressure. The opening lane must instead ask whether the *early position*
        # was already bad. Otherwise every later tactical loss would mark all
        # opening plies as severity=1 and poison the opening diagnosis.
        severity = max(0.0, -predicted) if opening_evidence else max(0.0, -value_target)

        policy_surprise = 0.0
        if best_score is not None:
            played = self._move_text(example, "played_move_uci")
            best_move = self._move_text(example, "move_uci")
            deliberate_exploration = bool(played and best_move and played != best_move)
            if not (suppress_exploration_regret and deliberate_exploration):
                policy_surprise = min(
                    1.0,
                    max(0.0, float(best_score) - float(search_score)) / 300.0,
                )

        return HardPositionEvidence(
            fen=str(getattr(example, "fen")),
            opening_bucket=opening,
            source_generation=generation,
            source_kind=source_kind,
            severity=severity,
            uncertainty=policy_surprise,
            value_error=value_error,
            round_index=round_index,
        )

    def candidates_from_record(
        self,
        record: Any,
        *,
        generation: int,
        round_index: int,
        source_kind: str = "selfplay",
    ) -> tuple[CapturedHardPosition, ...]:
        opening = self.opening_bucket_for_record(record)
        rows: list[CapturedHardPosition] = []
        examples = list(getattr(record, "examples", ()) or ())
        for ply_index, example in enumerate(examples):
            if ply_index < self.skip_opening_plies:
                continue
            evidence = self._evidence_from_example(
                example,
                opening=opening,
                generation=generation,
                round_index=round_index,
                source_kind=source_kind,
                suppress_exploration_regret=False,
                opening_evidence=False,
            )
            if evidence is not None:
                rows.append(CapturedHardPosition(evidence, evidence.priority, ply_index))
        rows.sort(key=lambda item: (item.priority, -item.ply_index), reverse=True)
        return tuple(row for row in rows if row.priority >= self.minimum_priority)[: self.max_positions_per_game]

    def opening_candidates_from_record(
        self,
        record: Any,
        *,
        generation: int,
        round_index: int,
        source_kind: str = "selfplay",
    ) -> tuple[CapturedHardPosition, ...]:
        """Observe early positions without treating stochastic exploration as error."""
        opening = self.opening_bucket_for_record(record)
        rows: list[CapturedHardPosition] = []
        examples = list(getattr(record, "examples", ()) or ())
        for ply_index, example in enumerate(examples[: self.opening_max_plies]):
            evidence = self._evidence_from_example(
                example,
                opening=opening,
                generation=generation,
                round_index=round_index,
                source_kind=f"{source_kind}_opening",
                suppress_exploration_regret=True,
                opening_evidence=True,
            )
            if evidence is not None:
                rows.append(CapturedHardPosition(evidence, evidence.priority, ply_index))
        rows.sort(key=lambda item: (item.priority, -item.ply_index), reverse=True)
        return tuple(row for row in rows if row.priority >= self.opening_minimum_priority)[: self.opening_positions_per_game]

    def persist_record(
        self,
        record: Any,
        *,
        generation: int,
        round_index: int,
        observed_at: Any,
        source_kind: str = "selfplay",
        max_per_bucket: int = 128,
    ) -> int:
        combined = (
            *self.opening_candidates_from_record(
                record,
                generation=generation,
                round_index=round_index,
                source_kind=source_kind,
            ),
            *self.candidates_from_record(
                record,
                generation=generation,
                round_index=round_index,
                source_kind=source_kind,
            ),
        )
        # One FEN should count once per game even when it lies in both the early
        # lane and the ordinary post-opening lane.
        best_by_position: dict[str, CapturedHardPosition] = {}
        for row in combined:
            key = row.evidence.position_key
            old = best_by_position.get(key)
            if old is None or row.priority > old.priority:
                best_by_position[key] = row
        rows = sorted(best_by_position.values(), key=lambda item: item.priority, reverse=True)
        for row in rows:
            self.store.upsert_hard_position(
                row.evidence,
                observed_at=observed_at,
                max_per_bucket=max_per_bucket,
            )
        return len(rows)


class AlphaBetaTeacherAdapter:
    """Bounded deep-search self-teaching for the live iterative alpha-beta searcher.

    `search_multiplier` must not be interpreted as multiplying alpha-beta depth:
    depth grows exponentially. Instead we allow at least one extra iterative-
    deepening ply while a wall-clock cap scales from the measured baseline search.
    """

    def __init__(
        self,
        *,
        cp_scale: float = 650.0,
        minimum_teacher_time_s: float = 0.03,
        maximum_teacher_time_s: float = 2.0,
    ) -> None:
        if cp_scale <= 0:
            raise ValueError("cp_scale must be positive")
        if minimum_teacher_time_s <= 0 or maximum_teacher_time_s < minimum_teacher_time_s:
            raise ValueError("invalid teacher time bounds")
        self.cp_scale = cp_scale
        self.minimum_teacher_time_s = minimum_teacher_time_s
        self.maximum_teacher_time_s = maximum_teacher_time_s

    def budget_for(
        self,
        request: DeepSearchTeacherRequest,
        *,
        base_depth: int,
        baseline_elapsed_s: float,
    ) -> TeacherSearchBudget:
        if base_depth <= 0:
            raise ValueError("base_depth must be positive")
        multiplier = max(1.0, float(request.search_multiplier))
        extra_depth = 1 if multiplier <= 3.0 else 2
        measured = max(self.minimum_teacher_time_s, float(baseline_elapsed_s))
        time_limit = min(self.maximum_teacher_time_s, measured * multiplier)
        time_limit = max(self.minimum_teacher_time_s, time_limit)
        return TeacherSearchBudget(base_depth + extra_depth, time_limit)

    def execute(
        self,
        request: DeepSearchTeacherRequest,
        *,
        searcher: Any,
        board_factory: Callable[[str], Any],
        base_depth: int,
        top_n: int = 5,
    ) -> TeacherReplayTarget | None:
        board = board_factory(request.fen)
        baseline = searcher.search(board, depth=base_depth, top_n=top_n)
        budget = self.budget_for(
            request,
            base_depth=base_depth,
            baseline_elapsed_s=float(getattr(baseline, "elapsed_s", 0.0)),
        )
        teacher = searcher.search(
            board,
            depth=budget.max_depth,
            time_limit_s=budget.time_limit_s,
            top_n=top_n,
        )
        move = getattr(teacher, "move", None)
        if move is None:
            return None
        move_uci = move.uci() if hasattr(move, "uci") else str(move)
        teacher_score = float(getattr(teacher, "score_cp", 0.0))
        baseline_score = float(getattr(baseline, "score_cp", 0.0))
        disagreement = abs(teacher_score - baseline_score)
        return TeacherReplayTarget(
            fen=request.fen,
            move_uci=move_uci,
            value_target=cp_to_value(teacher_score, cp_scale=self.cp_scale),
            policy_weight=1.25,
            priority=1.0 + min(2.0, disagreement / 250.0),
            teacher_score_cp=teacher_score,
            baseline_score_cp=baseline_score,
            teacher_depth=int(getattr(teacher, "depth", 0)),
            teacher_nodes=int(getattr(teacher, "nodes", 0)),
            teacher_elapsed_s=float(getattr(teacher, "elapsed_s", 0.0)),
        )
