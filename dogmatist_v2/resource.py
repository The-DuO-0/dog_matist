from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceSample:
    cpu_percent: float
    memory_percent: float
    thermal_pressure: str = "nominal"  # nominal/fair/serious/critical


@dataclass(frozen=True)
class ResourceBudget:
    selfplay_workers: int
    arena_workers: int
    trainer_slots: int = 1
    league_games: int = 2


class ResourceController:
    """Conservative adaptive controller for one-Mac overnight training.

    The important invariant is trainer_slots == 1: MPS batch training gets an
    exclusive lane while CPU-heavy self-play/League/Arena work may scale in
    parallel. League is intentionally capped at 2-3 concurrent games: enough to
    stop one slow game from serialising the tournament, but still small enough
    to preserve headroom on a daily-use Mac.
    """

    def __init__(
        self,
        *,
        min_selfplay_workers: int = 1,
        max_selfplay_workers: int = 4,
        min_arena_workers: int = 1,
        max_arena_workers: int = 2,
        min_league_games: int = 2,
        max_league_games: int = 3,
        cpu_soft_limit: float = 72.0,
        memory_soft_limit: float = 72.0,
    ) -> None:
        if not (2 <= min_league_games <= max_league_games <= 3):
            raise ValueError("League concurrency must stay inside the 2-3 game safety band")
        self.min_selfplay_workers = min_selfplay_workers
        self.max_selfplay_workers = max_selfplay_workers
        self.min_arena_workers = min_arena_workers
        self.max_arena_workers = max_arena_workers
        self.min_league_games = min_league_games
        self.max_league_games = max_league_games
        self.cpu_soft_limit = cpu_soft_limit
        self.memory_soft_limit = memory_soft_limit
        self._budget = ResourceBudget(
            selfplay_workers=min_selfplay_workers,
            arena_workers=min_arena_workers,
            trainer_slots=1,
            league_games=min_league_games,
        )

    @property
    def budget(self) -> ResourceBudget:
        return self._budget

    def update(self, sample: ResourceSample) -> ResourceBudget:
        thermal = sample.thermal_pressure.lower()
        constrained = (
            sample.cpu_percent >= self.cpu_soft_limit
            or sample.memory_percent >= self.memory_soft_limit
            or thermal in {"serious", "critical"}
        )
        comfortable = (
            sample.cpu_percent <= self.cpu_soft_limit - 18.0
            and sample.memory_percent <= self.memory_soft_limit - 15.0
            and thermal == "nominal"
        )

        sp = self._budget.selfplay_workers
        arena = self._budget.arena_workers
        league = self._budget.league_games
        if constrained:
            sp = max(self.min_selfplay_workers, sp - 1)
            arena = max(self.min_arena_workers, arena - 1)
            league = max(self.min_league_games, league - 1)
        elif comfortable:
            # Grow slowly; hysteresis prevents constant worker churn.
            sp = min(self.max_selfplay_workers, sp + 1)
            if sp >= self.max_selfplay_workers - 1:
                arena = min(self.max_arena_workers, arena + 1)
                league = min(self.max_league_games, league + 1)

        self._budget = ResourceBudget(
            selfplay_workers=sp,
            arena_workers=arena,
            trainer_slots=1,
            league_games=league,
        )
        return self._budget
