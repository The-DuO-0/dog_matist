from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any


@dataclass
class LiveLeagueDrainState:
    draining: bool = False
    reason: str | None = None
    pairs_started: int = 0
    pairs_completed: int = 0

    def request_drain(self, reason: str = "compute_budget_exhausted") -> None:
        if not self.draining:
            self.draining = True
            self.reason = reason

    def ui_payload(self) -> dict[str, object]:
        return {
            "draining": self.draining,
            "reason": self.reason,
            "pairs_started": self.pairs_started,
            "pairs_completed": self.pairs_completed,
        }


@dataclass(frozen=True)
class DrainedArenaResult:
    """Shape-compatible non-promotion result for a budget-drained round."""

    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    score: float = 0.0
    wilson_lower: float = 0.0
    promoted: bool = False


def build_budget_aware_population_arena(
    base_arena_cls: type,
    *,
    clock: Any,
    state: LiveLeagueDrainState,
) -> type:
    """Wrap the supplied production PopulationArena at color-pair boundaries.

    The production `_play_paired_set` already guarantees that one seed means the
    same opening is played twice with colors swapped. We deliberately reuse that
    implementation one seed at a time instead of copying chess/game logic.

    If the compute budget expires during the first leg, `super()` still finishes
    the reverse-color leg. The next seed is never admitted.
    """

    class BudgetAwarePopulationArena(base_arena_cls):
        def _play_paired_set(
            self,
            round_id: int,
            table: Any,
            first_generation: int,
            second_generation: int,
            first_searcher: Any,
            second_searcher: Any,
            seeds: list[Any],
            depth: int,
            max_plies: int,
        ) -> None:
            for seed in seeds:
                if state.draining or bool(clock.expired):
                    state.request_drain("compute_budget_exhausted")
                    return
                state.pairs_started += 1
                super()._play_paired_set(
                    round_id,
                    table,
                    first_generation,
                    second_generation,
                    first_searcher,
                    second_searcher,
                    [seed],
                    depth,
                    max_plies,
                )
                state.pairs_completed += 1
                if bool(clock.expired):
                    state.request_drain("compute_budget_exhausted")
                    return

        def _archive_specialists(self, round_id: int, table: Any, champion_generation: int) -> dict[str, int]:
            # A partial League is not enough evidence to mint a new specialist.
            if state.draining:
                return {}
            return super()._archive_specialists(round_id, table, champion_generation)

    BudgetAwarePopulationArena.__name__ = f"BudgetAware{base_arena_cls.__name__}"
    # Keep provenance pointed at the real production population module. The next
    # process-backed wrapper resolves LeagueTable/LeagueSummary/OpeningCurriculum
    # from `base_arena_cls.__module__`; leaving the default nested-class module here
    # incorrectly points it at dogmatist_v2.live_league_guard.
    BudgetAwarePopulationArena.__module__ = base_arena_cls.__module__
    return BudgetAwarePopulationArena


class LiveLeagueDrainOverride:
    """Temporary production hook for safe stop after the current color pair.

    It patches only the runtime module's `PopulationArena` symbol plus narrow
    runtime/memory methods for the duration of one cycle:

    - partial League -> no specialist harvest;
    - drained League -> no final promotion Arena;
    - candidates from an incomplete round are labelled aborted/budget-drained,
      not chessically rejected;
    - the population round is persisted as `budget_drained`, not `complete`.

    The original production objects are restored on exit, including exceptions.
    """

    def __init__(
        self,
        runtime: Any,
        clock: Any,
        *,
        runtime_module: Any | None = None,
        state: LiveLeagueDrainState | None = None,
    ) -> None:
        self.runtime = runtime
        self.clock = clock
        self.runtime_module = runtime_module
        self.state = state or LiveLeagueDrainState()
        self._base_population_arena: Any | None = None
        self._original_gate: Any | None = None
        self._original_harvest: Any | None = None
        self._gate_had_instance_attr = False
        self._harvest_had_instance_attr = False
        self._gate_instance_value: Any = None
        self._harvest_instance_value: Any = None
        self._memory_originals: dict[str, Any] = {}
        self._memory_had_attrs: dict[str, bool] = {}
        self._memory_previous_attrs: dict[str, Any] = {}

    def _module(self) -> Any:
        if self.runtime_module is not None:
            return self.runtime_module
        return importlib.import_module(self.runtime.__class__.__module__)

    def _patch_memory_method(self, name: str, wrapper: Any) -> None:
        memory = getattr(self.runtime, "memory", None)
        if memory is None or not hasattr(memory, name):
            return
        attrs = getattr(memory, "__dict__", {})
        self._memory_had_attrs[name] = name in attrs
        if name in attrs:
            self._memory_previous_attrs[name] = attrs[name]
        self._memory_originals[name] = getattr(memory, name)
        setattr(memory, name, wrapper)

    def __enter__(self) -> "LiveLeagueDrainOverride":
        module = self._module()
        self.runtime_module = module
        self._base_population_arena = module.PopulationArena
        module.PopulationArena = build_budget_aware_population_arena(
            self._base_population_arena,
            clock=self.clock,
            state=self.state,
        )

        attrs = getattr(self.runtime, "__dict__", {})
        self._gate_had_instance_attr = "gate_challenger" in attrs
        self._harvest_had_instance_attr = "_harvest_specialist_experience" in attrs
        if self._gate_had_instance_attr:
            self._gate_instance_value = attrs["gate_challenger"]
        if self._harvest_had_instance_attr:
            self._harvest_instance_value = attrs["_harvest_specialist_experience"]

        self._original_gate = self.runtime.gate_challenger
        self._original_harvest = getattr(self.runtime, "_harvest_specialist_experience", None)
        original_gate = self._original_gate
        original_harvest = self._original_harvest

        def guarded_gate(challenger_id: int, challenger: Any, *, games: int | None = None) -> Any:
            if self.state.draining or bool(self.clock.expired):
                if not self.state.draining:
                    self.state.request_drain("compute_budget_exhausted")
                memory = getattr(self.runtime, "memory", None)
                if memory is not None and hasattr(memory, "update_generation"):
                    memory.update_generation(int(challenger_id), status="aborted")
                if memory is not None and hasattr(memory, "add_insight"):
                    champion = self.runtime.champion_info()
                    champion_id = int(champion["id"])
                    memory.add_insight(
                        champion_id,
                        "budget_drain",
                        f"Generation {challenger_id} final Arena skipped because the League/evolution run entered safe-drain mode.",
                        self.state.ui_payload(),
                    )
                return DrainedArenaResult()
            return original_gate(challenger_id, challenger, games=games)

        def guarded_harvest(*args: Any, **kwargs: Any) -> int:
            if self.state.draining or bool(self.clock.expired):
                if not self.state.draining:
                    self.state.request_drain("compute_budget_exhausted")
                return 0
            if original_harvest is None:
                return 0
            return int(original_harvest(*args, **kwargs))

        self.runtime.gate_challenger = guarded_gate
        if original_harvest is not None:
            self.runtime._harvest_specialist_experience = guarded_harvest

        memory = getattr(self.runtime, "memory", None)
        if memory is not None:
            original_update_generation = getattr(memory, "update_generation", None)
            if original_update_generation is not None:
                def guarded_update_generation(generation_id: int, **fields: Any) -> Any:
                    if self.state.draining and fields.get("status") == "rejected":
                        fields = dict(fields)
                        fields["status"] = "aborted"
                    return original_update_generation(generation_id, **fields)
                self._patch_memory_method("update_generation", guarded_update_generation)

            original_update_member = getattr(memory, "update_population_member", None)
            if original_update_member is not None:
                def guarded_update_member(round_id: int, generation: int, **fields: Any) -> Any:
                    if self.state.draining and fields.get("status") == "rejected":
                        fields = dict(fields)
                        fields["status"] = "budget_drained"
                    return original_update_member(round_id, generation, **fields)
                self._patch_memory_method("update_population_member", guarded_update_member)

            original_finish_round = getattr(memory, "finish_population_round", None)
            if original_finish_round is not None:
                def guarded_finish_round(round_id: int, status: str = "complete") -> Any:
                    if self.state.draining and status == "complete":
                        status = "budget_drained"
                    return original_finish_round(round_id, status)
                self._patch_memory_method("finish_population_round", guarded_finish_round)

        return self

    def _restore_runtime_method(
        self,
        name: str,
        *,
        had_instance_attr: bool,
        previous: Any,
        original: Any,
    ) -> None:
        if had_instance_attr:
            setattr(self.runtime, name, previous)
            return
        try:
            delattr(self.runtime, name)
        except AttributeError:
            if original is not None:
                setattr(self.runtime, name, original)

    def __exit__(self, *_: object) -> None:
        module = self.runtime_module
        if module is not None and self._base_population_arena is not None:
            module.PopulationArena = self._base_population_arena

        self._restore_runtime_method(
            "gate_challenger",
            had_instance_attr=self._gate_had_instance_attr,
            previous=self._gate_instance_value,
            original=self._original_gate,
        )
        if self._original_harvest is not None:
            self._restore_runtime_method(
                "_harvest_specialist_experience",
                had_instance_attr=self._harvest_had_instance_attr,
                previous=self._harvest_instance_value,
                original=self._original_harvest,
            )

        memory = getattr(self.runtime, "memory", None)
        if memory is not None:
            for name, original in self._memory_originals.items():
                if self._memory_had_attrs.get(name, False):
                    setattr(memory, name, self._memory_previous_attrs[name])
                else:
                    try:
                        delattr(memory, name)
                    except AttributeError:
                        setattr(memory, name, original)
        self._memory_originals.clear()
        self._memory_had_attrs.clear()
        self._memory_previous_attrs.clear()
