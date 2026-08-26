from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from pathlib import Path
from typing import Any, Callable

from .fixed_reference import (
    FixedReferenceEvaluator,
    FixedReferenceResult,
    FrozenReferenceManager,
    FrozenStrengthReference,
)
from .live_game_watchdog import LiveGameWatchdogPolicy
from .strength_store import StrengthStore


@dataclass(frozen=True)
class LiveFixedReferenceReport:
    reference: FrozenStrengthReference
    result: FixedReferenceResult | None
    subject_generation: int | None
    skipped_reason: str | None = None
    trend: tuple[dict[str, object], ...] = ()

    def ui_payload(self) -> dict[str, object]:
        return {
            "reference": self.reference.as_dict(),
            "subject_generation": self.subject_generation,
            "result": self.result.ui_payload() if self.result is not None else None,
            "skipped_reason": self.skipped_reason,
            "trend": list(self.trend),
        }


def _walk_values(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_values(child)


def _first_int_for_keys(value: Any, keys: tuple[str, ...]) -> int | None:
    wanted = {key.lower() for key in keys}
    for key, child in _walk_values(value):
        if key.lower() not in wanted:
            continue
        try:
            return int(child)
        except (TypeError, ValueError):
            continue
    return None


def _first_bool_for_keys(value: Any, keys: tuple[str, ...]) -> bool | None:
    wanted = {key.lower() for key in keys}
    for key, child in _walk_values(value):
        if key.lower() not in wanted:
            continue
        if isinstance(child, bool):
            return child
        if isinstance(child, (int, float)):
            return bool(child)
    return None


class _ReferenceClockView:
    """Make user safe-stop behave like admission drain for reference games too."""

    def __init__(self, clock: Any, stop_requested: Callable[[], bool]) -> None:
        self.clock = clock
        self.stop_requested = stop_requested

    @property
    def expired(self) -> bool:
        return bool(self.stop_requested()) or bool(self.clock.expired)

    @property
    def stop_reason(self) -> str | None:
        if self.stop_requested():
            return "user_stop"
        return "compute_budget_exhausted" if bool(self.clock.expired) else None

    @property
    def elapsed_seconds(self) -> float:
        return float(self.clock.elapsed_seconds)

    @property
    def remaining_seconds(self) -> float:
        return float(self.clock.remaining_seconds)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.clock, name)


class LiveFixedReferenceCoordinator:
    """Measure each round's strongest available subject on one frozen ruler."""

    def __init__(
        self,
        runtime: Any,
        store: StrengthStore,
        clock: Any,
        *,
        reference_root: str | Path,
        pair_count: int = 2,
        watchdog_policy: LiveGameWatchdogPolicy | None = None,
        stop_requested: Callable[[], bool] | None = None,
        status_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if pair_count <= 0:
            raise ValueError("pair_count must be positive")
        self.runtime = runtime
        self.store = store
        self.clock = clock
        self.manager = FrozenReferenceManager(reference_root)
        self.pair_count = int(pair_count)
        self.watchdog_policy = watchdog_policy or LiveGameWatchdogPolicy()
        self.stop_requested = stop_requested or (lambda: False)
        self.reference_clock = _ReferenceClockView(clock, self.stop_requested)
        self.status_callback = status_callback
        self.reference: FrozenStrengthReference | None = None
        self.last_report: LiveFixedReferenceReport | None = None

    @property
    def memory(self) -> Any:
        return self.runtime.memory

    def checkpoint_for_generation(self, generation: int) -> str:
        row = self.memory.get_generation(int(generation))
        if row is None:
            raise LookupError(f"unknown generation {generation}")
        path = str(row["checkpoint_path"])
        if not path:
            raise RuntimeError(f"generation {generation} has no checkpoint path")
        return path

    def ensure_reference(self) -> FrozenStrengthReference:
        existing = self.manager.load()
        if existing is not None:
            if not self.manager.verify(existing):
                raise RuntimeError("frozen strength reference checksum mismatch")
            self.reference = existing
            return existing

        champion = self.runtime.champion_info()
        generation = int(champion["id"])
        checkpoint = self.checkpoint_for_generation(generation)
        self.reference = self.manager.freeze(
            checkpoint,
            generation=generation,
            created_at=datetime.now(timezone.utc),
        )
        return self.reference

    def _trend_payload(self, *, limit: int = 8) -> tuple[dict[str, object], ...]:
        rows = self.store.round_history(limit=limit)
        return tuple(
            {
                "round_index": int(row.round_index),
                "champion_generation": int(row.champion_generation),
                "score": float(row.fixed_reference_score),
                "games": int(row.paired_games),
                "promoted": bool(row.promoted),
            }
            for row in rows
        )

    def _subject_generation(self, raw_result: Any) -> int:
        # The League winner is the useful signal under a long-lived Champion: it
        # can improve for several rounds before finally winning promotion.
        subject = _first_int_for_keys(
            raw_result,
            (
                "top_generation",
                "league_top_generation",
                "best_generation",
                "challenger_generation",
            ),
        )
        if subject is not None:
            return subject
        after = _first_int_for_keys(raw_result, ("champion_after", "champion_generation"))
        if after is not None:
            return after
        return int(self.runtime.champion_info()["id"])

    def _promoted(self, raw_result: Any) -> bool:
        explicit = _first_bool_for_keys(raw_result, ("promoted", "promotion"))
        if explicit is not None:
            return explicit
        before = _first_int_for_keys(raw_result, ("champion_before",))
        after = _first_int_for_keys(raw_result, ("champion_after",))
        return before is not None and after is not None and before != after

    def _opening_curriculum_class(self) -> type:
        """Resolve the real production OpeningCurriculum through overlay wrappers."""
        runtime_module = importlib.import_module(self.runtime.__class__.__module__)
        population_cls = getattr(runtime_module, "PopulationArena")
        # During a cycle PopulationArena may be Parallel(BudgetAware(Production)).
        # Walk the MRO until we reach a module that actually owns the curriculum.
        for cls in getattr(population_cls, "__mro__", (population_cls,)):
            try:
                module = importlib.import_module(cls.__module__)
            except Exception:
                continue
            curriculum = getattr(module, "OpeningCurriculum", None)
            if curriculum is not None:
                return curriculum
        raise RuntimeError("could not resolve production OpeningCurriculum")

    def _openings(self, round_index: int) -> list[tuple[str, str]]:
        curriculum_cls = self._opening_curriculum_class()
        base_seed = int(self.runtime.config.get("project", {}).get("seed", 0))
        # Same deterministic recipe regardless of who currently holds the throne.
        curriculum = curriculum_cls(seed=base_seed + 910_009 + int(round_index) * 97)
        rows = curriculum.arena_pairs(self.pair_count)
        return [(board.fen(), str(opening)) for board, opening in rows]

    def evaluate_cycle(self, raw_result: Any, *, round_index: int) -> LiveFixedReferenceReport:
        reference = self.reference or self.ensure_reference()
        if self.stop_requested():
            report = LiveFixedReferenceReport(
                reference,
                None,
                None,
                "safe_stop_requested",
                self._trend_payload(),
            )
            self.last_report = report
            return report
        if bool(self.clock.expired):
            report = LiveFixedReferenceReport(
                reference,
                None,
                None,
                "compute_budget_exhausted_before_reference",
                self._trend_payload(),
            )
            self.last_report = report
            return report

        subject = self._subject_generation(raw_result)
        subject_checkpoint = self.checkpoint_for_generation(subject)
        openings = self._openings(round_index)
        lcfg = self.runtime.config.get("league", {})
        depth = int(
            lcfg.get(
                "depth",
                self.runtime.config.get("arena", {}).get(
                    "depth", self.runtime.config["search"]["depth"]
                ),
            )
        )
        max_plies = int(
            lcfg.get(
                "max_game_plies",
                self.runtime.config.get("arena", {}).get("max_game_plies", 220),
            )
        )
        parallel = int(self.runtime.config.get("runtime", {}).get("league_parallel_games", 2) or 2)
        if parallel not in (2, 3):
            parallel = 2

        def emit(snapshot: dict[str, object]) -> None:
            if self.status_callback is not None:
                self.status_callback({
                    "phase": "fixed_reference",
                    "fixed_reference": {
                        "reference": reference.as_dict(),
                        "subject_generation": subject,
                        "active": snapshot,
                        "trend": list(self._trend_payload()),
                    },
                })

        evaluator = FixedReferenceEvaluator(
            clock=self.reference_clock,
            parallel_games=parallel,
            hard_game_timeout_seconds=self.watchdog_policy.emergency_game_seconds,
            stall_timeout_seconds=self.watchdog_policy.stall_seconds,
            kill_grace_seconds=self.watchdog_policy.kill_grace_seconds,
            status_callback=emit,
        )
        result = evaluator.evaluate(
            round_index=int(round_index),
            subject_generation=subject,
            subject_checkpoint=subject_checkpoint,
            reference=reference,
            config=self.runtime.config,
            openings=openings,
            depth=depth,
            max_plies=max_plies,
            torch_threads=1,
        )

        champion = int(self.runtime.champion_info()["id"])
        evidence = result.to_round_evidence(
            champion_generation=champion,
            promoted=self._promoted(raw_result),
        )
        # Metadata mode mirrors what the next Strength planning pass will infer
        # once this new fixed-reference evidence is visible.
        from .strength_lab import StrengthLabController

        planned_mode = StrengthLabController().plan(
            (*self.store.round_history(), evidence)
        ).mode
        self.store.record_round(
            evidence,
            mode=planned_mode,
            recorded_at=datetime.now(timezone.utc),
        )
        report = LiveFixedReferenceReport(
            reference,
            result,
            subject,
            None,
            self._trend_payload(),
        )
        self.last_report = report
        return report


class LiveFixedReferenceCycleOverride:
    """Attach fixed-reference measurement after each real evolution cycle."""

    METHOD_NAMES = (
        "_population_evolve_cycle_unlocked",
        "_evolve_cycle_unlocked",
        "evolve_cycle",
    )

    def __init__(self, runtime: Any, coordinator: LiveFixedReferenceCoordinator) -> None:
        self.runtime = runtime
        self.coordinator = coordinator
        self._originals: dict[str, Any] = {}
        self._had_instance: dict[str, bool] = {}
        self._previous: dict[str, Any] = {}
        self._round_counter = 0

    def __enter__(self) -> "LiveFixedReferenceCycleOverride":
        attrs = getattr(self.runtime, "__dict__", {})
        for name in self.METHOD_NAMES:
            if not hasattr(self.runtime, name):
                continue
            original = getattr(self.runtime, name)
            self._originals[name] = original
            self._had_instance[name] = name in attrs
            if name in attrs:
                self._previous[name] = attrs[name]

            def make_wrapper(method: Any, method_name: str):
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    raw = method(*args, **kwargs)
                    # Public evolve_cycle may delegate to an already-wrapped private
                    # cycle. The marker prevents a second reference match.
                    marker = raw.get("_v2_fixed_reference_measured") if isinstance(raw, dict) else None
                    if not marker:
                        self._round_counter += 1
                        report = self.coordinator.evaluate_cycle(
                            raw,
                            round_index=self._round_counter,
                        )
                        if isinstance(raw, dict):
                            raw["_v2_fixed_reference_measured"] = True
                            raw["fixed_reference"] = report.ui_payload()
                    return raw

                wrapped.__name__ = f"v2_fixed_reference_{method_name}"
                return wrapped

            setattr(self.runtime, name, make_wrapper(original, name))
        return self

    def __exit__(self, *_: object) -> None:
        for name, original in self._originals.items():
            if self._had_instance.get(name, False):
                setattr(self.runtime, name, self._previous[name])
            else:
                try:
                    delattr(self.runtime, name)
                except AttributeError:
                    setattr(self.runtime, name, original)
        self._originals.clear()
        self._had_instance.clear()
        self._previous.clear()
