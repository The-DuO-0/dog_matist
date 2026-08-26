from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


MIN_STALL_SECONDS = 60.0 * 60.0
MIN_EMERGENCY_GAME_SECONDS = 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class LiveGameWatchdogPolicy:
    """Conservative production watchdog for real chess games.

    The session/overnight compute budget is *not* a game timeout. When that budget
    expires, DogMatist only stops admitting new work and lets already-started
    colour pairs finish naturally.

    A worker may be terminated only by this separate bug-watchdog policy. The
    safety floor intentionally errs heavily toward *not* killing chess:

    - at least 60 minutes with no completed move/search progress;
    - at least 24 hours total for one game as a last-resort process-leak ceiling.

    Inputs lower than those floors are automatically raised. This is deliberate:
    a stale old config must not quietly reintroduce a short game timeout. Values
    may still be configured *higher* after real-Mac measurements.
    """

    stall_seconds: float = MIN_STALL_SECONDS
    emergency_game_seconds: float = MIN_EMERGENCY_GAME_SECONDS
    kill_grace_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.stall_seconds <= 0 or self.emergency_game_seconds <= 0:
            raise ValueError("watchdog durations must be positive")
        if self.kill_grace_seconds < 0:
            raise ValueError("kill_grace_seconds must be non-negative")

        # Frozen dataclass, but normalization belongs at policy construction so
        # every caller—including older configs—gets the same bug-only floor.
        object.__setattr__(
            self,
            "stall_seconds",
            max(float(self.stall_seconds), MIN_STALL_SECONDS),
        )
        object.__setattr__(
            self,
            "emergency_game_seconds",
            max(float(self.emergency_game_seconds), MIN_EMERGENCY_GAME_SECONDS),
        )
        if self.emergency_game_seconds <= self.stall_seconds:
            object.__setattr__(
                self,
                "emergency_game_seconds",
                max(MIN_EMERGENCY_GAME_SECONDS, self.stall_seconds * 2.0),
            )

    def ui_payload(self) -> dict[str, float | bool | str]:
        return {
            "budget_interrupts_games": False,
            "stall_seconds": self.stall_seconds,
            "emergency_game_seconds": self.emergency_game_seconds,
            "kill_grace_seconds": self.kill_grace_seconds,
            "policy": "finish_started_games; kill_only_obvious_stall_or_extreme_process_leak",
        }


@contextmanager
def install_live_game_watchdog_policy(
    runtime: Any,
    policy: LiveGameWatchdogPolicy,
) -> Iterator[LiveGameWatchdogPolicy]:
    """Temporarily install conservative watchdog values into production config.

    `LiveParallelLeagueOverride` passes the live config into child workers, so this
    narrow context avoids changing the old trainer/search code or the user's config
    files. Every touched value is restored after the run, including exceptions.
    """

    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        raise ValueError("runtime.config must be a dict")

    had_league = "league" in config
    previous_league = config.get("league")
    if previous_league is None:
        league: dict[str, Any] = {}
        config["league"] = league
    elif not isinstance(previous_league, dict):
        raise ValueError("runtime.config['league'] must be a dict")
    else:
        league = previous_league

    keys = (
        "watchdog_stall_seconds",
        "watchdog_hard_seconds",
        "watchdog_kill_grace_seconds",
        "watchdog_budget_interrupts_games",
    )
    had_key = {key: key in league for key in keys}
    previous = {key: league.get(key) for key in keys}

    league["watchdog_stall_seconds"] = float(policy.stall_seconds)
    league["watchdog_hard_seconds"] = float(policy.emergency_game_seconds)
    league["watchdog_kill_grace_seconds"] = float(policy.kill_grace_seconds)
    league["watchdog_budget_interrupts_games"] = False

    try:
        yield policy
    finally:
        for key in keys:
            if had_key[key]:
                league[key] = previous[key]
            else:
                league.pop(key, None)
        if not had_league:
            config.pop("league", None)
