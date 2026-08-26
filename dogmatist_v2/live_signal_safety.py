from __future__ import annotations

import signal
from typing import Any, Callable

from . import live_parallel_league as _league
from . import live_parallel_population as _population


def _ignore_parent_sigint() -> None:
    """Let the parent process own graceful Ctrl-C / safe-drain semantics.

    Terminal Ctrl-C is delivered to the entire foreground process group on macOS.
    League workers must not raise KeyboardInterrupt independently while the parent
    is deliberately finishing already-started colour pairs. The parent retains
    terminate/kill authority through the watchdog/pool lifecycle.
    """

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, ValueError):
        # Non-main-thread/unit-test environments can reject signal changes.
        pass


def sigint_safe_league_worker(task: Any, event_queue: Any) -> None:
    _ignore_parent_sigint()
    _league._league_worker_main(task, event_queue)


class SigintSafeLeagueProcessPool(_league.LiveLeagueProcessPool):
    """Production pool whose spawned League children ignore foreground SIGINT."""

    def __init__(self, *args: Any, worker_target: Callable[..., None] | None = None, **kwargs: Any) -> None:
        super().__init__(
            *args,
            worker_target=worker_target or sigint_safe_league_worker,
            **kwargs,
        )


def install_parallel_league_signal_safety() -> None:
    """Install the safe pool binding used by the production Population overlay.

    ``live_parallel_population`` resolves its module-level ``LiveLeagueProcessPool``
    binding when a League phase starts, so this explicit one-time substitution is
    enough without copying or monkey-patching any chess/game method.
    """

    _population.LiveLeagueProcessPool = SigintSafeLeagueProcessPool
