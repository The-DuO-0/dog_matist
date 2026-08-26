from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from .live_replay import LiveReplayMixSampler
from .live_runtime_overlay import LiveStrengthCoordinator, LiveStrengthRoundReport


class LiveStrengthCycleOverride:
    """Inject Strength Lab into the supplied runtime without editing trainer.py.

    The production population cycle already has the perfect seam:

    ``selfplay() -> start_population_round() -> train_population()``.

    This context manager temporarily wraps only `selfplay` and `train_population`:

    - remember the game ids returned by the existing self-play method;
    - immediately before population training, mine those saved games, create a
      Strength Lab recipe and optionally persist bounded self-teacher labels;
    - run the unchanged production trainer while `MemoryStore.replay_sample` is
      temporarily routed through the recipe-aware replay mixer.

    The wrappers are restored on exit. By default integration failures fail open
    to the original training path so an experimental Strength Lab adapter cannot
    waste an overnight run.
    """

    def __init__(
        self,
        coordinator: LiveStrengthCoordinator,
        *,
        targeted_examples: int = 64,
        teacher_request_cap: int = 8,
        persist_teacher: bool = False,
        replay_sampler: LiveReplayMixSampler | None = None,
        report_callback: Callable[[LiveStrengthRoundReport], None] | None = None,
        error_callback: Callable[[Exception], None] | None = None,
        fail_open: bool = True,
    ) -> None:
        if targeted_examples <= 0:
            raise ValueError("targeted_examples must be positive")
        if teacher_request_cap < 0:
            raise ValueError("teacher_request_cap must be non-negative")
        self.coordinator = coordinator
        self.runtime = coordinator.runtime
        self.targeted_examples = int(targeted_examples)
        self.teacher_request_cap = int(teacher_request_cap)
        self.persist_teacher = bool(persist_teacher)
        self.replay_sampler = replay_sampler
        self.report_callback = report_callback
        self.error_callback = error_callback
        self.fail_open = bool(fail_open)
        self.latest_game_ids: tuple[str, ...] = ()
        self.last_report: LiveStrengthRoundReport | None = None
        self.last_error: Exception | None = None
        self._original_selfplay: Any | None = None
        self._original_train_population: Any | None = None
        self._selfplay_had_instance_attr = False
        self._train_had_instance_attr = False
        self._selfplay_instance_value: Any = None
        self._train_instance_value: Any = None

    def __enter__(self) -> "LiveStrengthCycleOverride":
        if self._original_selfplay is not None:
            raise RuntimeError("LiveStrengthCycleOverride cannot be entered twice")
        attrs = getattr(self.runtime, "__dict__", {})
        self._selfplay_had_instance_attr = "selfplay" in attrs
        self._train_had_instance_attr = "train_population" in attrs
        if self._selfplay_had_instance_attr:
            self._selfplay_instance_value = attrs["selfplay"]
        if self._train_had_instance_attr:
            self._train_instance_value = attrs["train_population"]

        self._original_selfplay = self.runtime.selfplay
        self._original_train_population = self.runtime.train_population
        original_selfplay = self._original_selfplay
        original_train_population = self._original_train_population

        def wrapped_selfplay(*args: Any, **kwargs: Any) -> list[str]:
            game_ids = original_selfplay(*args, **kwargs)
            self.latest_game_ids = tuple(str(gid) for gid in game_ids)
            return game_ids

        def wrapped_train_population(*, round_id: int, total_steps: int | None = None) -> Any:
            try:
                report = self.coordinator.run_pretraining_stage(
                    self.latest_game_ids,
                    round_index=int(round_id),
                    observed_at=datetime.now(timezone.utc),
                    targeted_examples=self.targeted_examples,
                    teacher_request_cap=self.teacher_request_cap,
                    persist_teacher=self.persist_teacher,
                )
                self.last_report = report
                if self.report_callback is not None:
                    self.report_callback(report)
                with self.coordinator.training_override(
                    report.recipe,
                    sampler=self.replay_sampler,
                ):
                    return original_train_population(round_id=round_id, total_steps=total_steps)
            except Exception as exc:
                self.last_error = exc
                if self.error_callback is not None:
                    self.error_callback(exc)
                if not self.fail_open:
                    raise
                # The production training path is the fallback safety boundary.
                return original_train_population(round_id=round_id, total_steps=total_steps)

        self.runtime.selfplay = wrapped_selfplay
        self.runtime.train_population = wrapped_train_population
        return self

    def _restore(self, name: str, *, had_instance_attr: bool, previous: Any, original: Any) -> None:
        if had_instance_attr:
            setattr(self.runtime, name, previous)
            return
        try:
            delattr(self.runtime, name)
        except AttributeError:
            setattr(self.runtime, name, original)

    def __exit__(self, *_: object) -> None:
        if self._original_selfplay is not None:
            self._restore(
                "selfplay",
                had_instance_attr=self._selfplay_had_instance_attr,
                previous=self._selfplay_instance_value,
                original=self._original_selfplay,
            )
        if self._original_train_population is not None:
            self._restore(
                "train_population",
                had_instance_attr=self._train_had_instance_attr,
                previous=self._train_instance_value,
                original=self._original_train_population,
            )
        self._original_selfplay = None
        self._original_train_population = None
