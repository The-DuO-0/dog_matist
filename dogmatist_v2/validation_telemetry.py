from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class ValidationTelemetry:
    """Small parser for structured V2 stdout during a copied-state Mac run."""

    ui_events: int = 0
    phases: dict[str, int] = field(default_factory=dict)
    max_parallel_games: int = 0
    max_live_games: int = 0
    max_runtime_by_game: dict[str, float] = field(default_factory=dict)
    failed_games: set[str] = field(default_factory=set)
    timed_out_games: set[str] = field(default_factory=set)
    watchdog: dict[str, Any] | None = None
    copy_validation: dict[str, Any] | None = None
    fixed_reference: dict[str, Any] | None = None
    final_compute: dict[str, Any] | None = None

    def feed_line(self, line: str) -> bool:
        prefix = "DOGMATIST_UI "
        if not line.startswith(prefix):
            return False
        try:
            payload = json.loads(line[len(prefix):])
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        self.feed_event(payload)
        return True

    def feed_event(self, payload: dict[str, Any]) -> None:
        self.ui_events += 1
        phase = str(payload.get("phase") or "unknown")
        self.phases[phase] = self.phases.get(phase, 0) + 1

        if isinstance(payload.get("watchdog"), dict):
            self.watchdog = dict(payload["watchdog"])
        if isinstance(payload.get("copy_validation"), dict):
            self.copy_validation = dict(payload["copy_validation"])
        if isinstance(payload.get("fixed_reference"), dict):
            self.fixed_reference = dict(payload["fixed_reference"])
        if isinstance(payload.get("compute"), dict):
            self.final_compute = dict(payload["compute"])

        league = payload.get("league")
        if not isinstance(league, dict):
            parallel = payload.get("parallel_league")
            if isinstance(parallel, dict):
                nested = parallel.get("league")
                league = nested if isinstance(nested, dict) else parallel
        if isinstance(league, dict):
            self._feed_game_snapshot(league)

        fixed = payload.get("fixed_reference")
        if isinstance(fixed, dict) and isinstance(fixed.get("active"), dict):
            self._feed_game_snapshot(fixed["active"])

    def _feed_game_snapshot(self, snapshot: dict[str, Any]) -> None:
        try:
            slots = int(snapshot.get("parallel_games", 0) or 0)
        except (TypeError, ValueError):
            slots = 0
        self.max_parallel_games = max(self.max_parallel_games, slots)

        active = snapshot.get("active_games") or []
        if not isinstance(active, list):
            active = []
        self.max_live_games = max(self.max_live_games, len(active))
        for game in active:
            if not isinstance(game, dict):
                continue
            game_id = str(game.get("game_id") or "unknown")
            try:
                runtime = float(game.get("runtime_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                runtime = 0.0
            self.max_runtime_by_game[game_id] = max(
                runtime,
                self.max_runtime_by_game.get(game_id, 0.0),
            )

        for key, target in (("failed_games", self.failed_games), ("timed_out_games", self.timed_out_games)):
            rows = snapshot.get(key) or []
            if isinstance(rows, (list, tuple, set)):
                target.update(str(row) for row in rows)

    def as_dict(self) -> dict[str, object]:
        runtimes = sorted(
            (
                {"game_id": game_id, "max_observed_runtime_seconds": runtime}
                for game_id, runtime in self.max_runtime_by_game.items()
            ),
            key=lambda row: float(row["max_observed_runtime_seconds"]),
            reverse=True,
        )
        return {
            "ui_events": self.ui_events,
            "phases": dict(sorted(self.phases.items())),
            "max_parallel_games": self.max_parallel_games,
            "max_live_games": self.max_live_games,
            "failed_games": sorted(self.failed_games),
            "timed_out_games": sorted(self.timed_out_games),
            "longest_observed_games": runtimes[:12],
            "watchdog": self.watchdog,
            "copy_validation": self.copy_validation,
            "fixed_reference": self.fixed_reference,
            "final_compute": self.final_compute,
        }
