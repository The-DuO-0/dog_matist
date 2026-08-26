from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Callable, Iterable


class ComputeBudgetClock:
    """An active-runtime clock for long experiments.

    The clock advances only while ``resume()`` is active.  This is intentionally
    different from a wall-clock deadline: a sleeping/suspended/explicitly paused
    run can resume later without losing its remaining compute budget.
    """

    def __init__(
        self,
        budget_seconds: float,
        *,
        now: Callable[[], float] = monotonic,
        autostart: bool = True,
    ) -> None:
        if budget_seconds <= 0:
            raise ValueError("budget_seconds must be positive")
        self.budget_seconds = float(budget_seconds)
        self._now = now
        self._accumulated = 0.0
        self._active_since: float | None = None
        if autostart:
            self.resume()

    @property
    def running(self) -> bool:
        return self._active_since is not None

    def resume(self) -> None:
        if self._active_since is None:
            self._active_since = self._now()

    def pause(self) -> None:
        if self._active_since is None:
            return
        current = self._now()
        self._accumulated += max(0.0, current - self._active_since)
        self._active_since = None

    @property
    def elapsed_seconds(self) -> float:
        elapsed = self._accumulated
        if self._active_since is not None:
            elapsed += max(0.0, self._now() - self._active_since)
        return elapsed

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.budget_seconds - self.elapsed_seconds)

    @property
    def expired(self) -> bool:
        return self.elapsed_seconds >= self.budget_seconds

    def snapshot(self) -> dict[str, float | bool]:
        return {
            "budget_seconds": self.budget_seconds,
            "elapsed_compute_seconds": self.elapsed_seconds,
            "remaining_compute_seconds": self.remaining_seconds,
            "running": self.running,
            "wall_sleep_counts": False,
        }


class GameState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass(frozen=True)
class ColorPairing:
    pairing_id: str
    first_white_id: str
    first_black_id: str
    opening: str | None = None

    def games(self) -> tuple[LeagueGameSpec, LeagueGameSpec]:
        return (
            LeagueGameSpec(
                game_id=f"{self.pairing_id}:w",
                pairing_id=self.pairing_id,
                leg=1,
                white_id=self.first_white_id,
                black_id=self.first_black_id,
                opening=self.opening,
            ),
            LeagueGameSpec(
                game_id=f"{self.pairing_id}:b",
                pairing_id=self.pairing_id,
                leg=2,
                white_id=self.first_black_id,
                black_id=self.first_white_id,
                opening=self.opening,
            ),
        )


@dataclass(frozen=True)
class LeagueGameSpec:
    game_id: str
    pairing_id: str
    leg: int
    white_id: str
    black_id: str
    opening: str | None = None


@dataclass
class LeagueGameStatus:
    spec: LeagueGameSpec
    state: GameState = GameState.QUEUED
    plies: int = 0
    started_compute_seconds: float | None = None
    last_progress_compute_seconds: float | None = None
    ended_compute_seconds: float | None = None
    result: str | None = None
    reason: str | None = None

    def start(self, compute_seconds: float) -> None:
        if self.state is not GameState.QUEUED:
            raise RuntimeError(f"game {self.spec.game_id} is not queued")
        self.state = GameState.RUNNING
        self.started_compute_seconds = compute_seconds
        self.last_progress_compute_seconds = compute_seconds

    def report_progress(self, plies: int, compute_seconds: float) -> None:
        if self.state is not GameState.RUNNING:
            return
        if plies > self.plies:
            self.plies = plies
            self.last_progress_compute_seconds = compute_seconds

    def finish(self, compute_seconds: float, *, result: str | None = None) -> None:
        if self.state is not GameState.RUNNING:
            raise RuntimeError(f"game {self.spec.game_id} is not running")
        self.state = GameState.COMPLETE
        self.ended_compute_seconds = compute_seconds
        self.result = result

    def fail(self, compute_seconds: float, reason: str) -> None:
        if self.state is not GameState.RUNNING:
            raise RuntimeError(f"game {self.spec.game_id} is not running")
        self.state = GameState.FAILED
        self.ended_compute_seconds = compute_seconds
        self.reason = reason

    def timeout(self, compute_seconds: float, reason: str) -> None:
        if self.state is not GameState.RUNNING:
            raise RuntimeError(f"game {self.spec.game_id} is not running")
        self.state = GameState.TIMED_OUT
        self.ended_compute_seconds = compute_seconds
        self.reason = reason

    def runtime_seconds(self, compute_seconds: float) -> float:
        if self.started_compute_seconds is None:
            return 0.0
        end = self.ended_compute_seconds if self.ended_compute_seconds is not None else compute_seconds
        return max(0.0, end - self.started_compute_seconds)

    @property
    def completed_full_moves(self) -> int:
        return self.plies // 2


@dataclass(frozen=True)
class WatchdogTrip:
    game_id: str
    reason: str
    runtime_seconds: float
    plies: int


class GameWatchdog:
    def __init__(self, *, hard_timeout_seconds: float, stall_timeout_seconds: float) -> None:
        if hard_timeout_seconds <= 0 or stall_timeout_seconds <= 0:
            raise ValueError("watchdog timeouts must be positive")
        self.hard_timeout_seconds = float(hard_timeout_seconds)
        self.stall_timeout_seconds = float(stall_timeout_seconds)

    def check(self, game: LeagueGameStatus, compute_seconds: float) -> WatchdogTrip | None:
        if game.state is not GameState.RUNNING or game.started_compute_seconds is None:
            return None
        runtime = game.runtime_seconds(compute_seconds)
        if runtime >= self.hard_timeout_seconds:
            return WatchdogTrip(game.spec.game_id, "hard_game_timeout", runtime, game.plies)
        last_progress = game.last_progress_compute_seconds
        if last_progress is not None and compute_seconds - last_progress >= self.stall_timeout_seconds:
            return WatchdogTrip(game.spec.game_id, "no_move_progress_timeout", runtime, game.plies)
        return None


def _clock_stop_reason(clock: Any) -> str:
    reason = getattr(clock, "stop_reason", None)
    return str(reason) if reason else "compute_budget_exhausted"


@dataclass
class LeaguePairScheduler:
    """Admission controller for 2-3 parallel colour-balanced League games.

    When the clock says stop, admission switches to drain mode. No new pairings
    are opened, but missing reverse-colour legs of every already-started pairing
    are still admitted. The clock may represent a compute budget or a user safe-
    stop request; ``stop_reason`` is propagated when available.
    """

    pairings: Iterable[ColorPairing]
    clock: Any
    parallel_games: int = 2
    hard_game_timeout_seconds: float = 20 * 60
    stall_timeout_seconds: float = 4 * 60
    _pair_order: list[str] = field(init=False, default_factory=list)
    _pending: dict[str, list[LeagueGameStatus]] = field(init=False, default_factory=dict)
    _started_pairs: set[str] = field(init=False, default_factory=set)
    _active: dict[str, LeagueGameStatus] = field(init=False, default_factory=dict)
    _terminal: dict[str, LeagueGameStatus] = field(init=False, default_factory=dict)
    _draining: bool = field(init=False, default=False)
    _stop_reason: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.parallel_games not in (2, 3):
            raise ValueError("League parallel_games must be 2 or 3")
        pairs = list(self.pairings)
        if len({p.pairing_id for p in pairs}) != len(pairs):
            raise ValueError("pairing_id values must be unique")
        self.pairings = pairs
        self._pair_order = [p.pairing_id for p in pairs]
        self._pending = {
            p.pairing_id: [LeagueGameStatus(spec) for spec in p.games()]
            for p in pairs
        }
        self._watchdog = GameWatchdog(
            hard_timeout_seconds=self.hard_game_timeout_seconds,
            stall_timeout_seconds=self.stall_timeout_seconds,
        )

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def active_games(self) -> tuple[LeagueGameStatus, ...]:
        return tuple(self._active.values())

    @property
    def terminal_games(self) -> tuple[LeagueGameStatus, ...]:
        return tuple(self._terminal.values())

    def request_drain(self, reason: str = "safe_stop_requested") -> None:
        if not self._draining:
            self._draining = True
            self._stop_reason = reason

    def _observe_budget(self) -> None:
        if bool(self.clock.expired):
            self.request_drain(_clock_stop_reason(self.clock))

    def _start_status(self, status: LeagueGameStatus) -> LeagueGameStatus:
        now = float(self.clock.elapsed_seconds)
        status.start(now)
        self._active[status.spec.game_id] = status
        return status

    def _next_pending_from_started_pair(self) -> LeagueGameStatus | None:
        for pair_id in self._pair_order:
            if pair_id not in self._started_pairs:
                continue
            pending = self._pending[pair_id]
            if pending:
                return pending.pop(0)
        return None

    def _open_next_pair(self) -> LeagueGameStatus | None:
        if self._draining:
            return None
        for pair_id in self._pair_order:
            if pair_id in self._started_pairs:
                continue
            self._started_pairs.add(pair_id)
            return self._pending[pair_id].pop(0)
        return None

    def poll_startable(self) -> list[LeagueGameStatus]:
        self._observe_budget()
        started: list[LeagueGameStatus] = []
        while len(self._active) < self.parallel_games:
            status = self._next_pending_from_started_pair()
            if status is None:
                status = self._open_next_pair()
            if status is None:
                break
            started.append(self._start_status(status))
        return started

    def report_progress(self, game_id: str, plies: int) -> None:
        game = self._active.get(game_id)
        if game is not None:
            game.report_progress(plies, float(self.clock.elapsed_seconds))

    def _take_active(self, game_id: str) -> LeagueGameStatus:
        try:
            return self._active.pop(game_id)
        except KeyError as exc:
            raise KeyError(f"unknown active League game: {game_id}") from exc

    def complete(self, game_id: str, *, result: str | None = None) -> LeagueGameStatus:
        game = self._take_active(game_id)
        game.finish(float(self.clock.elapsed_seconds), result=result)
        self._terminal[game_id] = game
        self._observe_budget()
        return game

    def fail(self, game_id: str, reason: str) -> LeagueGameStatus:
        game = self._take_active(game_id)
        game.fail(float(self.clock.elapsed_seconds), reason)
        self._terminal[game_id] = game
        self._observe_budget()
        return game

    def poll_watchdogs(self) -> list[WatchdogTrip]:
        now = float(self.clock.elapsed_seconds)
        trips: list[WatchdogTrip] = []
        for game_id, game in list(self._active.items()):
            trip = self._watchdog.check(game, now)
            if trip is None:
                continue
            self._active.pop(game_id)
            game.timeout(now, trip.reason)
            self._terminal[game_id] = game
            trips.append(trip)
        self._observe_budget()
        return trips

    def pair_complete(self, pairing_id: str) -> bool:
        if pairing_id not in self._started_pairs:
            return False
        return not self._pending[pairing_id] and not any(
            game.spec.pairing_id == pairing_id for game in self._active.values()
        )

    @property
    def safe_to_stop(self) -> bool:
        if not self._draining or self._active:
            return False
        return all(self.pair_complete(pair_id) for pair_id in self._started_pairs)

    @property
    def all_pairings_complete(self) -> bool:
        if self._active:
            return False
        return all(not self._pending[pair_id] for pair_id in self._pair_order)

    def snapshot(self) -> dict[str, object]:
        now = float(self.clock.elapsed_seconds)
        active = []
        for game in self._active.values():
            active.append(
                {
                    "game_id": game.spec.game_id,
                    "pairing_id": game.spec.pairing_id,
                    "leg": game.spec.leg,
                    "white_id": game.spec.white_id,
                    "black_id": game.spec.black_id,
                    "opening": game.spec.opening,
                    "plies": game.plies,
                    "completed_full_moves": game.completed_full_moves,
                    "runtime_seconds": game.runtime_seconds(now),
                    "state": game.state.value,
                }
            )
        return {
            "parallel_games": self.parallel_games,
            "draining": self._draining,
            "stop_reason": self._stop_reason,
            "safe_to_stop": self.safe_to_stop,
            "active_games": active,
        }
