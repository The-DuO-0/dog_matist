from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .runtime import WatchdogTrip


class KillableWorker(Protocol):
    """Minimal process boundary required by the League watchdog."""

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...


@dataclass(frozen=True)
class WorkerTermination:
    game_id: str
    watchdog_reason: str
    escalated_to_kill: bool


@dataclass
class LeagueWorkerSupervisor:
    """Own killable League workers and enforce watchdog termination.

    The scheduler is responsible for deciding *when* a game timed out. This
    supervisor only guarantees that a timed-out worker cannot remain wedged in
    the background forever.
    """

    terminate_grace_seconds: float = 2.0
    _workers: dict[str, KillableWorker] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.terminate_grace_seconds < 0.0:
            raise ValueError("terminate_grace_seconds must be non-negative")

    def register(self, game_id: str, worker: KillableWorker) -> None:
        if not game_id:
            raise ValueError("game_id must be non-empty")
        if game_id in self._workers:
            raise RuntimeError(f"worker already registered for {game_id}")
        self._workers[game_id] = worker

    def release(self, game_id: str) -> None:
        self._workers.pop(game_id, None)

    def has_worker(self, game_id: str) -> bool:
        return game_id in self._workers

    def enforce(self, trips: list[WatchdogTrip] | tuple[WatchdogTrip, ...]) -> tuple[WorkerTermination, ...]:
        results: list[WorkerTermination] = []
        for trip in trips:
            worker = self._workers.pop(trip.game_id, None)
            if worker is None:
                # The scheduler may report a trip after a worker already exited;
                # absence is safe and should not crash the outer run.
                continue

            escalated = False
            if worker.is_alive():
                worker.terminate()
                worker.join(self.terminate_grace_seconds)
            if worker.is_alive():
                escalated = True
                worker.kill()
                worker.join(self.terminate_grace_seconds)

            results.append(
                WorkerTermination(
                    game_id=trip.game_id,
                    watchdog_reason=trip.reason,
                    escalated_to_kill=escalated,
                )
            )
        return tuple(results)

    def terminate_all(self) -> tuple[str, ...]:
        """Emergency shutdown only; normal budget expiry should use drain mode."""
        stopped: list[str] = []
        for game_id, worker in list(self._workers.items()):
            if worker.is_alive():
                worker.terminate()
                worker.join(self.terminate_grace_seconds)
            if worker.is_alive():
                worker.kill()
                worker.join(self.terminate_grace_seconds)
            stopped.append(game_id)
            self._workers.pop(game_id, None)
        return tuple(stopped)
