from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread
import time
from typing import Callable


@dataclass(frozen=True)
class ComputeSnapshot:
    budget_seconds: float
    elapsed_seconds: float
    remaining_seconds: float
    excluded_sleep_seconds: float
    paused_seconds: float
    expired: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "budget_seconds": self.budget_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_seconds": self.remaining_seconds,
            "excluded_sleep_seconds": self.excluded_sleep_seconds,
            "paused_seconds": self.paused_seconds,
            "expired": self.expired,
            "wall_sleep_counts": False,
        }


class HeartbeatComputeClock:
    """Count active runtime while excluding long process-suspension gaps.

    The current Mac night runner can be suspended by lid-close/manual sleep even
    while a wall-clock deadline keeps advancing. A lightweight heartbeat thread
    gives us a process-local signal: while dog_matist is alive and runnable, the
    thread pulses regularly; during system/process suspension it cannot run.

    On the first pulse after a long gap, all but one normal heartbeat interval is
    excluded from the compute budget. This avoids depending on platform-specific
    semantics of ``time.monotonic`` across system sleep.

    The clock is intentionally conservative: it only classifies gaps larger than
    ``suspension_threshold_seconds`` as suspension, so ordinary scheduling jitter,
    alpha-beta work, PyTorch kernels and child-process waits still count as active
    runtime.
    """

    def __init__(
        self,
        budget_seconds: float,
        *,
        now: Callable[[], float] = time.time,
        heartbeat_interval_seconds: float = 1.0,
        suspension_threshold_seconds: float = 15.0,
    ) -> None:
        if budget_seconds < 0:
            raise ValueError("budget_seconds must be non-negative")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        if suspension_threshold_seconds <= heartbeat_interval_seconds:
            raise ValueError("suspension threshold must exceed heartbeat interval")
        self.budget_seconds = float(budget_seconds)
        self._now = now
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.suspension_threshold_seconds = float(suspension_threshold_seconds)
        started = float(now())
        self._started_at = started
        self._last_pulse_at = started
        self._excluded_sleep = 0.0
        self._paused_total = 0.0
        self._paused_at: float | None = None
        self._forced_expiry = False
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None

    def pulse(self) -> float:
        """Record that the process is currently runnable; return detected sleep."""
        with self._lock:
            current = float(self._now())
            gap = max(0.0, current - self._last_pulse_at)
            excluded = 0.0
            if self._paused_at is None and gap > self.suspension_threshold_seconds:
                excluded = max(0.0, gap - self.heartbeat_interval_seconds)
                self._excluded_sleep += excluded
            self._last_pulse_at = current
            return excluded

    def request_expiry(self) -> None:
        """Force admission-budget expiry without terminating active work.

        This is primarily useful for copied-state validation that needs to prove
        the safe-drain path at a deterministic point. Runtime guards still decide
        when already-started chess games may finish; this method never kills a
        game or worker.
        """
        self.pulse()
        with self._lock:
            self._forced_expiry = True

    def pause(self) -> None:
        self.pulse()
        with self._lock:
            if self._paused_at is None:
                self._paused_at = float(self._now())

    def resume(self) -> None:
        with self._lock:
            if self._paused_at is None:
                return
            current = float(self._now())
            self._paused_total += max(0.0, current - self._paused_at)
            self._paused_at = None
            self._last_pulse_at = current

    def _elapsed_locked(self, current: float) -> tuple[float, float]:
        current_pause = 0.0
        if self._paused_at is not None:
            current_pause = max(0.0, current - self._paused_at)
        paused = self._paused_total + current_pause
        elapsed = max(
            0.0,
            current - self._started_at - self._excluded_sleep - paused,
        )
        return elapsed, paused

    @property
    def elapsed_seconds(self) -> float:
        self.pulse()
        with self._lock:
            elapsed, _ = self._elapsed_locked(float(self._now()))
            return elapsed

    @property
    def remaining_seconds(self) -> float:
        self.pulse()
        with self._lock:
            if self._forced_expiry:
                return 0.0
            elapsed, _ = self._elapsed_locked(float(self._now()))
            return max(0.0, self.budget_seconds - elapsed)

    @property
    def expired(self) -> bool:
        self.pulse()
        with self._lock:
            if self._forced_expiry:
                return True
            elapsed, _ = self._elapsed_locked(float(self._now()))
            return elapsed >= self.budget_seconds

    def snapshot(self) -> dict[str, float | bool]:
        self.pulse()
        with self._lock:
            current = float(self._now())
            elapsed, paused = self._elapsed_locked(current)
            expired = self._forced_expiry or elapsed >= self.budget_seconds
            snapshot = ComputeSnapshot(
                budget_seconds=self.budget_seconds,
                elapsed_seconds=elapsed,
                remaining_seconds=0.0 if self._forced_expiry else max(0.0, self.budget_seconds - elapsed),
                excluded_sleep_seconds=self._excluded_sleep,
                paused_seconds=paused,
                expired=expired,
            )
        return snapshot.as_dict()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_seconds):
            self.pulse()

    def start_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._heartbeat_loop,
            name="dogmatist-compute-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop_background(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, self.heartbeat_interval_seconds * 2.0))
        self._thread = None

    def __enter__(self) -> "HeartbeatComputeClock":
        self.start_background()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_background()
