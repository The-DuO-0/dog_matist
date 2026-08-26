from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
from typing import Any


@dataclass
class LiveArenaDrainState:
    draining: bool = False
    reason: str | None = None
    pairs_started: int = 0
    pairs_completed: int = 0
    games_completed: int = 0
    linked_state: Any | None = field(default=None, repr=False, compare=False)

    def request_drain(self, reason: str = "compute_budget_exhausted") -> None:
        if not self.draining:
            self.draining = True
            self.reason = reason
        if self.linked_state is not None and hasattr(self.linked_state, "request_drain"):
            self.linked_state.request_drain(reason)

    def ui_payload(self) -> dict[str, object]:
        return {
            "draining": self.draining,
            "reason": self.reason,
            "pairs_started": self.pairs_started,
            "pairs_completed": self.pairs_completed,
            "games_completed": self.games_completed,
        }


def build_budget_aware_arena(
    base_arena_cls: type,
    *,
    clock: Any,
    state: LiveArenaDrainState,
) -> type:
    """Wrap the real held-out Arena without copying its chess/evaluation logic.

    Production ``Arena.compare`` asks ``OpeningCurriculum.arena_pairs`` for an
    iterable and then fully plays both colour legs before requesting the next
    opening. Replacing that iterable with a budget-aware generator therefore gives
    us the exact safe boundary we want: if time expires in leg one, leg two still
    completes; the next opening is never yielded.

    The wrapper only forces ``promoted=False`` when the Arena was actually cut
    short by a budget drain. Adaptive accept/reject decisions that finish at a
    complete pair before the generator needs another opening remain valid.
    """

    class BudgetAwareArena(base_arena_cls):
        def compare(self, *args: Any, **kwargs: Any) -> Any:
            arena_module = importlib.import_module(base_arena_cls.__module__)
            original_curriculum = arena_module.OpeningCurriculum

            class BudgetAwareCurriculum(original_curriculum):
                def arena_pairs(self, count: int):  # type: ignore[override]
                    pairs = super().arena_pairs(count)

                    def iter_pairs():
                        for item in pairs:
                            if state.draining or bool(clock.expired):
                                state.request_drain("compute_budget_exhausted")
                                return
                            state.pairs_started += 1
                            yield item
                            # Production asks for the next opening only after both
                            # colour legs of the current opening have completed.
                            state.pairs_completed += 1
                            if bool(clock.expired):
                                state.request_drain("compute_budget_exhausted")
                                return

                    return iter_pairs()

            arena_module.OpeningCurriculum = BudgetAwareCurriculum
            try:
                result = super().compare(*args, **kwargs)
            finally:
                arena_module.OpeningCurriculum = original_curriculum

            state.games_completed = int(getattr(result, "games", 0))
            state.pairs_completed = max(state.pairs_completed, state.games_completed // 2)
            if state.draining and bool(getattr(result, "promoted", False)):
                try:
                    result = replace(result, promoted=False)
                except TypeError:
                    try:
                        result.promoted = False
                    except Exception:
                        pass
            return result

    BudgetAwareArena.__name__ = f"BudgetAware{base_arena_cls.__name__}"
    return BudgetAwareArena


class LiveArenaDrainOverride:
    """Temporary pair-boundary budget hook for production ``Arena.compare``.

    Minimal/fake runtimes used by smoke tests may not expose the production
    ``Arena`` symbol at all. In that case this context is deliberately a no-op;
    the real Mac runtime does expose it and receives the full pair-boundary guard.
    """

    def __init__(
        self,
        runtime: Any,
        clock: Any,
        *,
        runtime_module: Any | None = None,
        state: LiveArenaDrainState | None = None,
    ) -> None:
        self.runtime = runtime
        self.clock = clock
        self.runtime_module = runtime_module
        self.state = state or LiveArenaDrainState()
        self._base_arena: Any | None = None
        self._original_gate: Any | None = None
        self._gate_had_instance_attr = False
        self._gate_instance_value: Any = None
        self._disabled = False

    def _module(self) -> Any:
        if self.runtime_module is not None:
            return self.runtime_module
        return importlib.import_module(self.runtime.__class__.__module__)

    def __enter__(self) -> "LiveArenaDrainOverride":
        module = self._module()
        self.runtime_module = module
        if not hasattr(module, "Arena") or not hasattr(self.runtime, "gate_challenger"):
            self._disabled = True
            return self

        self._base_arena = module.Arena
        module.Arena = build_budget_aware_arena(
            self._base_arena,
            clock=self.clock,
            state=self.state,
        )

        attrs = getattr(self.runtime, "__dict__", {})
        self._gate_had_instance_attr = "gate_challenger" in attrs
        if self._gate_had_instance_attr:
            self._gate_instance_value = attrs["gate_challenger"]
        self._original_gate = self.runtime.gate_challenger
        original_gate = self._original_gate

        def guarded_gate(challenger_id: int, challenger: Any, *, games: int | None = None) -> Any:
            if bool(self.clock.expired):
                self.state.request_drain("compute_budget_exhausted")
                from .live_league_guard import DrainedArenaResult

                memory = getattr(self.runtime, "memory", None)
                if memory is not None and hasattr(memory, "update_generation"):
                    memory.update_generation(int(challenger_id), status="aborted")
                return DrainedArenaResult()

            memory = getattr(self.runtime, "memory", None)
            buffered_rejections: list[tuple[Any, ...]] = []
            original_add_insight = getattr(memory, "add_insight", None) if memory is not None else None
            had_add_insight_attr = bool(memory is not None and "add_insight" in getattr(memory, "__dict__", {}))
            previous_add_insight = getattr(memory, "__dict__", {}).get("add_insight") if memory is not None else None

            if original_add_insight is not None:
                def filtered_add_insight(generation: Any, kind: str, text: str, evidence: Any = None) -> None:
                    if kind == "rejection":
                        buffered_rejections.append((generation, kind, text, evidence))
                        return
                    original_add_insight(generation, kind, text, evidence)

                memory.add_insight = filtered_add_insight

            try:
                result = original_gate(challenger_id, challenger, games=games)
            finally:
                if memory is not None and original_add_insight is not None:
                    if had_add_insight_attr:
                        memory.add_insight = previous_add_insight
                    else:
                        try:
                            delattr(memory, "add_insight")
                        except AttributeError:
                            pass

            if self.state.draining:
                if memory is not None and hasattr(memory, "update_generation"):
                    memory.update_generation(int(challenger_id), status="aborted")
                if original_add_insight is not None:
                    champion = self.runtime.champion_info()
                    original_add_insight(
                        int(champion["id"]),
                        "budget_drain",
                        f"Generation {challenger_id} held-out Arena stopped after a complete colour pair because the active compute budget expired; no promotion decision was recorded.",
                        self.state.ui_payload(),
                    )
            elif original_add_insight is not None:
                for args in buffered_rejections:
                    original_add_insight(*args)
            return result

        self.runtime.gate_challenger = guarded_gate
        return self

    def __exit__(self, *_: object) -> None:
        if self._disabled:
            return
        module = self.runtime_module
        if module is not None and self._base_arena is not None:
            module.Arena = self._base_arena

        if self._gate_had_instance_attr:
            self.runtime.gate_challenger = self._gate_instance_value
        else:
            try:
                delattr(self.runtime, "gate_challenger")
            except AttributeError:
                if self._original_gate is not None:
                    self.runtime.gate_challenger = self._original_gate
