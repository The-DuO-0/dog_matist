from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from .runtime import ComputeBudgetClock, LeaguePairScheduler
from .strength_lab import StrengthLabPlan
from .strength_pipeline import StrengthRoundRecipe


class EvolutionStage(str, Enum):
    SELF_PLAY = "self_play"
    STRENGTH_LAB = "strength_lab"
    POPULATION_TRAIN = "population_train"
    LEAGUE = "league"
    ARENA = "arena"
    STRENGTH_GUARD = "strength_guard"
    PROMOTION = "promotion"
    ARCHIVE = "archive"
    NEXT_ROUND = "next_round"
    COMPLETE = "complete"


_STAGE_LABELS: tuple[tuple[EvolutionStage, str], ...] = (
    (EvolutionStage.SELF_PLAY, "Self-play"),
    (EvolutionStage.STRENGTH_LAB, "Strength Lab · hard positions + deep-search teacher"),
    (EvolutionStage.POPULATION_TRAIN, "Population train"),
    (EvolutionStage.LEAGUE, "League · 2–3 parallel games"),
    (EvolutionStage.ARENA, "Arena"),
    (EvolutionStage.STRENGTH_GUARD, "Fixed-reference strength guard"),
    (EvolutionStage.PROMOTION, "Promote / Reject"),
    (EvolutionStage.ARCHIVE, "Archive + Chronicle"),
    (EvolutionStage.NEXT_ROUND, "Next round"),
)


@dataclass(frozen=True)
class EvolutionFlowSnapshot:
    round_index: int
    stage: EvolutionStage
    compute: dict[str, float | bool]
    flow: tuple[dict[str, str], ...]
    league: dict[str, object] | None
    strength_lab: dict[str, object] | None
    status_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "stage": self.stage.value,
            "compute": self.compute,
            "flow": list(self.flow),
            "league": self.league,
            "strength_lab": self.strength_lab,
            "status_text": self.status_text,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"), sort_keys=True)


def _flow_rows(stage: EvolutionStage) -> tuple[dict[str, str], ...]:
    if stage is EvolutionStage.COMPLETE:
        return tuple(
            {"key": key.value, "label": label, "state": "done"}
            for key, label in _STAGE_LABELS
        )

    keys = [key for key, _ in _STAGE_LABELS]
    try:
        current_index = keys.index(stage)
    except ValueError:
        current_index = len(keys)
    rows: list[dict[str, str]] = []
    for index, (key, label) in enumerate(_STAGE_LABELS):
        if index < current_index:
            state = "done"
        elif index == current_index:
            state = "current"
        else:
            state = "pending"
        rows.append({"key": key.value, "label": label, "state": state})
    return tuple(rows)


def build_evolution_flow_snapshot(
    *,
    round_index: int,
    stage: EvolutionStage,
    clock: ComputeBudgetClock,
    league: LeaguePairScheduler | None = None,
    strength_lab_plan: StrengthLabPlan | None = None,
    strength_recipe: StrengthRoundRecipe | None = None,
) -> EvolutionFlowSnapshot:
    league_snapshot = league.snapshot() if league is not None else None
    strength_snapshot = strength_lab_plan.ui_payload() if strength_lab_plan is not None else None
    if strength_snapshot is not None and strength_recipe is not None:
        strength_snapshot = dict(strength_snapshot)
        strength_snapshot["recipe"] = strength_recipe.ui_payload()

    if league is not None and league.draining:
        status = "Compute budget reached — finishing the current colour pair(s), then stopping safely."
    elif stage is EvolutionStage.LEAGUE and league is not None:
        active = len(league.active_games)
        status = f"League running: {active}/{league.parallel_games} games active."
    elif stage is EvolutionStage.STRENGTH_LAB and strength_lab_plan is not None:
        if strength_recipe is not None:
            effective = strength_recipe.ui_payload()["effective"]
            status = (
                f"Strength Lab: {strength_lab_plan.mode.value} mode; "
                f"hard {effective['hard_positions']}, specialist {effective['specialist_sparring']}, "
                f"teacher {effective['deep_search_teacher']} at "
                f"{strength_lab_plan.teacher_search_multiplier:.1f}x search."
            )
        else:
            status = (
                f"Strength Lab: {strength_lab_plan.mode.value} mode; "
                f"deep-search teacher {strength_lab_plan.teacher_search_multiplier:.1f}x on "
                f"{strength_lab_plan.teacher_fraction:.0%} of selected hard positions."
            )
    elif stage is EvolutionStage.COMPLETE:
        status = "Run complete."
    else:
        label = dict(_STAGE_LABELS).get(stage, stage.value.replace("_", " ").title())
        status = f"{label} in progress."

    return EvolutionFlowSnapshot(
        round_index=round_index,
        stage=stage,
        compute=clock.snapshot(),
        flow=_flow_rows(stage),
        league=league_snapshot,
        strength_lab=strength_snapshot,
        status_text=status,
    )


def encode_ui_event(snapshot: EvolutionFlowSnapshot) -> str:
    """One-line stdout protocol consumed by the Studio Evolution page."""
    return "DOGMATIST_UI " + snapshot.to_json()
