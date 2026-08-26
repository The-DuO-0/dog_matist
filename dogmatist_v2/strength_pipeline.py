from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .strength_lab import StrengthLabPlan, TrainingBatchBudget
from .strength_store import HardPositionEvidence, StrengthStore


@dataclass(frozen=True)
class DeepSearchTeacherRequest:
    position_key: str
    fen: str
    opening_bucket: str
    search_multiplier: float
    source_generation: int | None


@dataclass(frozen=True)
class StrengthRoundRecipe:
    requested: TrainingBatchBudget
    natural_selfplay_examples: int
    hard_positions: tuple[HardPositionEvidence, ...]
    specialist_examples: int
    teacher_requests: tuple[DeepSearchTeacherRequest, ...]
    backfilled_examples: int
    opening_focus_buckets: tuple[str, ...] = ()
    focused_hard_positions: int = 0
    focused_teacher_requests: int = 0

    @property
    def effective_total(self) -> int:
        return (
            self.natural_selfplay_examples
            + len(self.hard_positions)
            + self.specialist_examples
            + len(self.teacher_requests)
        )

    def ui_payload(self) -> dict[str, object]:
        return {
            "requested": self.requested.as_dict(),
            "effective": {
                "natural_selfplay": self.natural_selfplay_examples,
                "hard_positions": len(self.hard_positions),
                "specialist_sparring": self.specialist_examples,
                "deep_search_teacher": len(self.teacher_requests),
                "total": self.effective_total,
            },
            "backfilled_examples": self.backfilled_examples,
            "teacher_search_multiplier": (
                self.teacher_requests[0].search_multiplier if self.teacher_requests else None
            ),
            "opening_focus": {
                "buckets": list(self.opening_focus_buckets),
                "focused_hard_positions": self.focused_hard_positions,
                "focused_teacher_requests": self.focused_teacher_requests,
                "book_moves_injected": False,
                "novel_openings_allowed": True,
            },
        }


class StrengthPipelinePlanner:
    """Turn Strength Lab policy into bounded production training work.

    Weak-opening evidence can reserve part of the hard-position/self-teacher quota,
    but never more than 75 percent. The remaining targeted quota deliberately
    prefers other buckets, so repairing one bad opening cannot collapse opening
    diversity or turn dog_matist into a fixed opening-book engine.
    """

    def __init__(self, store: StrengthStore) -> None:
        self.store = store

    @staticmethod
    def _dedupe(
        rows: Iterable[HardPositionEvidence],
        used: set[str],
        limit: int,
    ) -> tuple[HardPositionEvidence, ...]:
        selected: list[HardPositionEvidence] = []
        for row in rows:
            key = row.position_key
            if key in used:
                continue
            used.add(key)
            selected.append(row)
            if len(selected) >= limit:
                break
        return tuple(selected)

    def build_recipe(
        self,
        plan: StrengthLabPlan,
        *,
        total_examples: int,
        available_specialist_examples: int = 0,
        hard_position_bucket_cap: int = 8,
        opening_focus_buckets: Iterable[str] = (),
        opening_focus_fraction: float = 0.0,
    ) -> StrengthRoundRecipe:
        if available_specialist_examples < 0:
            raise ValueError("available_specialist_examples must be non-negative")
        if not 0.0 <= float(opening_focus_fraction) <= 0.75:
            raise ValueError("opening_focus_fraction must stay between 0 and 0.75")

        requested = plan.batch_budget(total_examples)
        focus = tuple(dict.fromkeys(str(x) for x in opening_focus_buckets if str(x)))
        focus_set = set(focus)
        fraction = float(opening_focus_fraction) if focus else 0.0

        focus_hard_goal = min(
            requested.hard_positions,
            int(round(requested.hard_positions * fraction)),
        )
        focus_teacher_goal = min(
            requested.deep_search_teacher,
            int(round(requested.deep_search_teacher * fraction)),
        )

        used: set[str] = set()
        focused_teacher_rows: tuple[HardPositionEvidence, ...] = ()
        focused_hard_rows: tuple[HardPositionEvidence, ...] = ()
        if focus and (focus_hard_goal or focus_teacher_goal):
            focused_pool = self.store.sample_hard_positions(
                focus_hard_goal + focus_teacher_goal,
                per_bucket_cap=hard_position_bucket_cap,
                opening_buckets=focus,
            )
            # Give the highest-priority focused positions to deeper self-search;
            # ordinary hard replay receives the next slice.
            focused_teacher_rows = self._dedupe(focused_pool, used, focus_teacher_goal)
            focused_hard_rows = self._dedupe(focused_pool, used, focus_hard_goal)

        hard_remaining = requested.hard_positions - len(focused_hard_rows)
        teacher_remaining = requested.deep_search_teacher - len(focused_teacher_rows)

        # The non-focused share explicitly prefers the rest of opening space.
        # If evidence is sparse, we backfill with natural self-play rather than
        # silently exceeding the focus cap and overfitting one opening family.
        global_pool = self.store.sample_hard_positions(
            hard_remaining + teacher_remaining + len(focus) * hard_position_bucket_cap + 8,
            per_bucket_cap=hard_position_bucket_cap,
        )
        nonfocus_pool = tuple(row for row in global_pool if row.opening_bucket not in focus_set)
        global_hard_rows = self._dedupe(nonfocus_pool, used, hard_remaining)
        global_teacher_rows = self._dedupe(nonfocus_pool, used, teacher_remaining)

        hard_rows = tuple((*focused_hard_rows, *global_hard_rows))
        teacher_rows = tuple((*focused_teacher_rows, *global_teacher_rows))
        teacher_requests = tuple(
            DeepSearchTeacherRequest(
                position_key=row.position_key,
                fen=row.fen,
                opening_bucket=row.opening_bucket,
                search_multiplier=plan.teacher_search_multiplier,
                source_generation=row.source_generation,
            )
            for row in teacher_rows
        )

        specialist_examples = min(requested.specialist_sparring, available_specialist_examples)
        missing_hard = requested.hard_positions - len(hard_rows)
        missing_teacher = requested.deep_search_teacher - len(teacher_requests)
        missing_specialist = requested.specialist_sparring - specialist_examples
        backfill = missing_hard + missing_teacher + missing_specialist

        natural = requested.natural_selfplay + backfill
        recipe = StrengthRoundRecipe(
            requested=requested,
            natural_selfplay_examples=natural,
            hard_positions=hard_rows,
            specialist_examples=specialist_examples,
            teacher_requests=teacher_requests,
            backfilled_examples=backfill,
            opening_focus_buckets=focus,
            focused_hard_positions=len(focused_hard_rows),
            focused_teacher_requests=len(focused_teacher_rows),
        )
        if recipe.effective_total != total_examples:
            raise AssertionError("Strength Lab recipe must preserve the requested total")
        return recipe


@dataclass(frozen=True)
class EngineABTrialPlan:
    """Paired engine-revision experiment on frozen model weights."""

    baseline_revision_id: str
    candidate_revision_id: str
    frozen_checkpoint: str
    start_fens: tuple[str, ...]

    @property
    def paired_games(self) -> int:
        return len(self.start_fens) * 2

    def __post_init__(self) -> None:
        if not self.baseline_revision_id or not self.candidate_revision_id:
            raise ValueError("engine revision ids must be non-empty")
        if self.baseline_revision_id == self.candidate_revision_id:
            raise ValueError("candidate engine revision must differ from baseline")
        if not self.frozen_checkpoint:
            raise ValueError("A/B engine trials require a frozen model checkpoint")
        if not self.start_fens:
            raise ValueError("A/B engine trials require paired start positions")

    def game_specs(self) -> tuple[dict[str, str], ...]:
        games: list[dict[str, str]] = []
        for index, fen in enumerate(self.start_fens):
            pair_id = f"engine-ab-{index}"
            games.append(
                {
                    "pair_id": pair_id,
                    "fen": fen,
                    "white_revision": self.candidate_revision_id,
                    "black_revision": self.baseline_revision_id,
                    "checkpoint": self.frozen_checkpoint,
                }
            )
            games.append(
                {
                    "pair_id": pair_id,
                    "fen": fen,
                    "white_revision": self.baseline_revision_id,
                    "black_revision": self.candidate_revision_id,
                    "checkpoint": self.frozen_checkpoint,
                }
            )
        return tuple(games)
