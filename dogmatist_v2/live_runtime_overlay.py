from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from .live_bridge import AlphaBetaTeacherAdapter, LiveGameEvidenceBridge, TeacherReplayTarget
from .live_replay import LiveReplayMixSampler, LiveReplayOverride
from .opening_lab import OpeningRepairPlan, OpeningWeaknessController
from .strength_lab import StrengthLabController, StrengthLabPlan
from .strength_pipeline import StrengthPipelinePlanner, StrengthRoundRecipe
from .strength_store import StrengthStore


@dataclass(frozen=True)
class LiveReplayExample:
    """Structural twin of production ``darwinchess.memory.ReplayExample``."""

    fen: str
    move_uci: str
    value_target: float
    policy_weight: float = 1.0
    priority: float = 1.0
    played_move_uci: str | None = None
    search_score_cp: float | None = None
    best_score_cp: float | None = None


@dataclass(frozen=True)
class LiveStrengthRoundReport:
    round_index: int
    generation: int
    plan: StrengthLabPlan
    recipe: StrengthRoundRecipe
    captured_positions: int
    teacher_examples: int
    teacher_game_ids: tuple[str, ...]
    opening_repair: OpeningRepairPlan | None = None

    def ui_payload(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "generation": self.generation,
            "captured_positions": self.captured_positions,
            "teacher_examples": self.teacher_examples,
            "teacher_game_ids": list(self.teacher_game_ids),
            "plan": self.plan.ui_payload(),
            "recipe": self.recipe.ui_payload(),
            "opening_repair": (
                self.opening_repair.ui_payload() if self.opening_repair is not None else None
            ),
        }


class LiveStrengthCoordinator:
    """Narrow production overlay for the supplied dog_matist-2.0 runtime.

    The live runtime keeps its MemoryStore, AlphaBetaSearcher, ContinualTrainer and
    optimizer. V2 adds compact hard-position memory, opening-weakness targeting and
    bounded deeper self-search. Opening names are telemetry labels only; no book
    moves are injected and unknown/frontier openings remain eligible for training.
    """

    def __init__(
        self,
        runtime: Any,
        store: StrengthStore,
        *,
        controller: StrengthLabController | None = None,
        opening_controller: OpeningWeaknessController | None = None,
        evidence_bridge: LiveGameEvidenceBridge | None = None,
        teacher_adapter: AlphaBetaTeacherAdapter | None = None,
        board_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.controller = controller or StrengthLabController()
        self.opening_controller = opening_controller or OpeningWeaknessController()
        self.evidence_bridge = evidence_bridge or LiveGameEvidenceBridge(store)
        self.teacher_adapter = teacher_adapter or AlphaBetaTeacherAdapter()
        self._board_factory = board_factory
        self.last_opening_repair: OpeningRepairPlan | None = None

    @property
    def memory(self) -> Any:
        return self.runtime.memory

    @staticmethod
    def _row_get(row: Any, key: str, default: Any = None) -> Any:
        try:
            return row[key]
        except Exception:
            return getattr(row, key, default)

    def _load_saved_record(self, game_id: str) -> tuple[Any, int, str]:
        conn = self.memory.conn
        game = conn.execute(
            "SELECT generation, source, metadata_json FROM games WHERE id=?",
            (game_id,),
        ).fetchone()
        if game is None:
            raise LookupError(f"unknown game id: {game_id}")
        rows = conn.execute(
            """
            SELECT fen, move_uci, played_move_uci, search_score_cp, best_score_cp,
                   value_target, policy_weight, priority
            FROM examples WHERE game_id=? ORDER BY ply
            """,
            (game_id,),
        ).fetchall()
        try:
            metadata = json.loads(game["metadata_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        examples = [
            SimpleNamespace(
                fen=row["fen"],
                move_uci=row["move_uci"],
                played_move_uci=row["played_move_uci"],
                search_score_cp=row["search_score_cp"],
                best_score_cp=row["best_score_cp"],
                value_target=row["value_target"],
                policy_weight=row["policy_weight"],
                priority=row["priority"],
            )
            for row in rows
        ]
        generation = int(
            game["generation"]
            if game["generation"] is not None
            else self.runtime.champion_info()["id"]
        )
        return SimpleNamespace(examples=examples, metadata=metadata), generation, str(game["source"])

    def capture_saved_games(
        self,
        game_ids: Iterable[str],
        *,
        round_index: int,
        observed_at: datetime,
    ) -> int:
        captured = 0
        for game_id in game_ids:
            record, generation, source = self._load_saved_record(str(game_id))
            captured += self.evidence_bridge.persist_record(
                record,
                generation=generation,
                round_index=round_index,
                observed_at=observed_at,
                source_kind=source or "selfplay",
            )
        return captured

    def specialist_example_count(self, *, limit: int = 64) -> int:
        if not hasattr(self.memory, "active_specialists"):
            return 0
        specialists = list(self.memory.active_specialists(limit=limit))
        generations = sorted({int(row["generation"]) for row in specialists})
        openings = sorted({str(row["opening_name"]) for row in specialists if row["opening_name"]})
        if not generations or not openings:
            return 0
        gq = ",".join("?" for _ in generations)
        oq = ",".join("?" for _ in openings)
        row = self.memory.conn.execute(
            f"""
            SELECT COUNT(*) AS n FROM examples
            WHERE origin_generation IN ({gq}) AND opening_name IN ({oq})
            """,
            [*generations, *openings],
        ).fetchone()
        return int(row["n"] if row is not None else 0)

    def _current_champion_specialist_evidence(self, *, limit: int = 64) -> tuple[Any, ...]:
        """Use specialist *gaps* only when they were measured vs the current throne.

        Historical specialists remain available as replay/donors, but a Gen31 win
        against an old Gen15 must not be misread as proof that today's Gen54 is weak
        in the same opening.
        """
        if not hasattr(self.memory, "active_specialists"):
            return ()
        champion = int(self.runtime.champion_info()["id"])
        selected: list[Any] = []
        for row in self.memory.active_specialists(limit=limit):
            raw = self._row_get(row, "evidence_json", None)
            try:
                evidence = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except (TypeError, ValueError, json.JSONDecodeError):
                evidence = {}
            measured_against = evidence.get("champion_generation")
            if measured_against is None:
                continue
            try:
                if int(measured_against) != champion:
                    continue
            except (TypeError, ValueError):
                continue
            selected.append(row)
        return tuple(selected)

    def opening_repair_plan(self) -> OpeningRepairPlan:
        plan = self.opening_controller.plan(
            specialists=self._current_champion_specialist_evidence(),
            hard_bucket_stats=self.store.opening_bucket_stats(),
        )
        self.last_opening_repair = plan
        return plan

    def build_recipe(
        self,
        *,
        targeted_examples: int,
        hard_position_bucket_cap: int = 8,
    ) -> tuple[StrengthLabPlan, StrengthRoundRecipe]:
        if targeted_examples <= 0:
            raise ValueError("targeted_examples must be positive")
        plan = self.controller.plan(self.store.round_history())
        opening = self.opening_repair_plan()
        recipe = StrengthPipelinePlanner(self.store).build_recipe(
            plan,
            total_examples=targeted_examples,
            available_specialist_examples=self.specialist_example_count(),
            hard_position_bucket_cap=hard_position_bucket_cap,
            opening_focus_buckets=opening.focus_openings,
            opening_focus_fraction=opening.focus_fraction if opening.active else 0.0,
        )
        return plan, recipe

    def training_override(
        self,
        recipe: StrengthRoundRecipe,
        *,
        sampler: LiveReplayMixSampler | None = None,
    ) -> LiveReplayOverride:
        """Temporarily mix opening/hard/specialist examples into the real trainer."""
        return LiveReplayOverride(self.memory, recipe, sampler=sampler)

    def _default_board_factory(self) -> Callable[[str], Any]:
        if self._board_factory is not None:
            return self._board_factory
        try:
            import chess  # type: ignore
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError("python-chess is required to execute live teacher searches") from exc
        return chess.Board

    def execute_teacher_requests(
        self,
        recipe: StrengthRoundRecipe,
        *,
        generation: int,
        request_cap: int = 8,
        searcher: Any | None = None,
    ) -> tuple[int, tuple[str, ...]]:
        if request_cap <= 0 or not recipe.teacher_requests:
            return 0, ()
        base_depth = int(self.runtime.config["search"]["depth"])
        searcher = searcher or self.runtime.make_searcher()
        board_factory = self._default_board_factory()

        grouped: dict[str, list[tuple[Any, TeacherReplayTarget]]] = {}
        for request in recipe.teacher_requests[:request_cap]:
            target = self.teacher_adapter.execute(
                request,
                searcher=searcher,
                board_factory=board_factory,
                base_depth=base_depth,
            )
            if target is None:
                continue
            grouped.setdefault(request.opening_bucket or "Strength Lab", []).append((request, target))

        game_ids: list[str] = []
        teacher_examples = 0
        for opening, rows in grouped.items():
            examples = [
                LiveReplayExample(
                    fen=target.fen,
                    move_uci=target.move_uci,
                    value_target=target.value_target,
                    policy_weight=target.policy_weight,
                    priority=target.priority,
                    search_score_cp=target.baseline_score_cp,
                    best_score_cp=target.teacher_score_cp,
                )
                for _, target in rows
            ]
            if not examples:
                continue
            gid = self.memory.add_game(
                source="strength_teacher",
                generation=generation,
                white_agent=f"dog_matist-g{generation}",
                black_agent="dog_matist-self-teacher",
                result="*",
                termination="deep_search_self_teacher",
                pgn="",
                plies=len(examples),
                examples=examples,
                metadata={
                    "opening_name": opening,
                    "opening_family": "strength_teacher",
                    "teacher_search_multiplier": rows[0][0].search_multiplier,
                    "source_generations": sorted(
                        {r.source_generation for r, _ in rows if r.source_generation is not None}
                    ),
                    "opening_repair": opening in set(recipe.opening_focus_buckets),
                },
            )
            game_ids.append(str(gid))
            teacher_examples += len(examples)

        if teacher_examples and hasattr(self.memory, "add_insight"):
            self.memory.add_insight(
                generation,
                "strength_teacher",
                f"Strength Lab added {teacher_examples} bounded self-teacher replay labels.",
                {"game_ids": game_ids, "teacher_examples": teacher_examples},
            )
        return teacher_examples, tuple(game_ids)

    def run_pretraining_stage(
        self,
        game_ids: Iterable[str],
        *,
        round_index: int,
        observed_at: datetime,
        targeted_examples: int = 64,
        teacher_request_cap: int = 8,
        persist_teacher: bool = False,
        generation: int | None = None,
        searcher: Any | None = None,
    ) -> LiveStrengthRoundReport:
        """Prepare one live Strength Lab stage before population training."""
        captured = self.capture_saved_games(
            game_ids,
            round_index=round_index,
            observed_at=observed_at,
        )
        plan, recipe = self.build_recipe(targeted_examples=targeted_examples)
        opening_repair = self.last_opening_repair
        generation = int(
            generation if generation is not None else self.runtime.champion_info()["id"]
        )
        if persist_teacher:
            teacher_examples, teacher_game_ids = self.execute_teacher_requests(
                recipe,
                generation=generation,
                request_cap=teacher_request_cap,
                searcher=searcher,
            )
        else:
            teacher_examples, teacher_game_ids = 0, ()
        return LiveStrengthRoundReport(
            round_index=round_index,
            generation=generation,
            plan=plan,
            recipe=recipe,
            captured_positions=captured,
            teacher_examples=teacher_examples,
            teacher_game_ids=teacher_game_ids,
            opening_repair=opening_repair,
        )
