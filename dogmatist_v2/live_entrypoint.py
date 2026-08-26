from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from .live_compute import HeartbeatComputeClock
from .live_fixed_reference import LiveFixedReferenceCoordinator, LiveFixedReferenceCycleOverride
from .live_fixed_reference_safe import fixed_reference_fail_open
from .live_game_watchdog import LiveGameWatchdogPolicy, install_live_game_watchdog_policy
from .live_runner import LiveEvolutionRunReport, LiveEvolutionRunner
from .live_runtime_overlay import LiveStrengthCoordinator
from .live_signal_safety import install_parallel_league_signal_safety
from .strength_store import StrengthStore


@dataclass(frozen=True)
class LiveEvolutionOptions:
    """Small production-facing option set for the V2 live overlay."""

    targeted_examples: int = 64
    teacher_request_cap: int = 8
    persist_teacher: bool = False
    enable_parallel_league: bool = True
    strength_fail_open: bool = True
    parallel_league_fail_open: bool = True
    handle_sigint: bool = True
    strength_db_name: str = "strength_v2.sqlite3"
    watchdog_stall_seconds: float = 30.0 * 60.0
    watchdog_emergency_game_seconds: float = 2.0 * 60.0 * 60.0
    watchdog_kill_grace_seconds: float = 2.0
    enable_fixed_reference: bool = True
    fixed_reference_pairs: int = 2
    fixed_reference_fail_open: bool = True
    fixed_reference_dir_name: str = "frozen_strength_reference"

    def __post_init__(self) -> None:
        if self.targeted_examples <= 0:
            raise ValueError("targeted_examples must be positive")
        if self.teacher_request_cap < 0:
            raise ValueError("teacher_request_cap must be non-negative")
        if not self.strength_db_name.strip():
            raise ValueError("strength_db_name must be non-empty")
        if self.fixed_reference_pairs <= 0:
            raise ValueError("fixed_reference_pairs must be positive")
        if not self.fixed_reference_dir_name.strip():
            raise ValueError("fixed_reference_dir_name must be non-empty")
        LiveGameWatchdogPolicy(
            stall_seconds=self.watchdog_stall_seconds,
            emergency_game_seconds=self.watchdog_emergency_game_seconds,
            kill_grace_seconds=self.watchdog_kill_grace_seconds,
        )

    def watchdog_policy(self) -> LiveGameWatchdogPolicy:
        return LiveGameWatchdogPolicy(
            stall_seconds=self.watchdog_stall_seconds,
            emergency_game_seconds=self.watchdog_emergency_game_seconds,
            kill_grace_seconds=self.watchdog_kill_grace_seconds,
        )


def _cycle_only_clock(cycles: int) -> HeartbeatComputeClock:
    """Non-binding safety clock for explicit cycle-count runs."""

    years = max(1, int(cycles))
    return HeartbeatComputeClock(float(years) * 366.0 * 24.0 * 3600.0)


def _copy_validation_parallel_games() -> int:
    """Return the explicitly requested copied-state League width.

    Two games remains the conservative default. Three is allowed only when the
    isolated validation launcher opts in through a dedicated environment flag;
    production/live state never reads this override because the copy-validation
    guard itself must already be enabled.
    """

    raw = os.environ.get("DOGMATIST_V2_COPY_LEAGUE_PARALLEL", "2").strip() or "2"
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("DOGMATIST_V2_COPY_LEAGUE_PARALLEL must be 2 or 3") from exc
    if value not in (2, 3):
        raise ValueError("DOGMATIST_V2_COPY_LEAGUE_PARALLEL must be 2 or 3")
    return value


@contextmanager
def _copy_validation_runtime_overrides(runtime: Any, enabled: bool) -> Iterator[None]:
    """Force conservative settings only inside an isolated copied-state run."""

    if not enabled:
        yield
        return
    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        raise ValueError("runtime.config must be a dict")
    runtime_cfg = config.setdefault("runtime", {})
    if not isinstance(runtime_cfg, dict):
        raise ValueError("runtime.config['runtime'] must be a dict")
    had_parallel = "league_parallel_games" in runtime_cfg
    previous_parallel = runtime_cfg.get("league_parallel_games")
    runtime_cfg["league_parallel_games"] = _copy_validation_parallel_games()
    try:
        yield
    finally:
        if had_parallel:
            runtime_cfg["league_parallel_games"] = previous_parallel
        else:
            runtime_cfg.pop("league_parallel_games", None)


def run_live_evolution(
    runtime: Any,
    *,
    hours: float | None = None,
    cycles: int | None = None,
    options: LiveEvolutionOptions | None = None,
    progress: Callable[[str], None] = print,
    league_status_callback: Callable[[dict[str, object]], None] | None = None,
) -> LiveEvolutionRunReport:
    """Run the real DarwinRuntime through the tested V2 production overlay.

    Strength Lab persists into a separate small SQLite database under the runtime
    state root. The live replay/checkpoint database remains owned by MemoryStore.
    Teacher labels deliberately default OFF until copied-state Mac validation.

    The active-compute budget is only an *admission* budget. Reaching 8/10 hours
    never terminates a chess game already in progress. Existing League/Arena games
    finish naturally (and a missing reverse-colour leg is completed for fairness),
    then no new pair is admitted. A child process is force-stopped only by the
    separate, deliberately generous bug watchdog installed below.

    A frozen reference checkpoint is copied before the first cycle and never moves
    with the throne. Each completed round can therefore measure its best League
    subject against the same ruler even while a dominant Champion remains in power.
    This meter is fail-open during copied-state validation: a reference-integration
    error must not waste the ordinary overnight evolution run.
    """

    if hours is not None and cycles is not None:
        raise ValueError("hours and cycles are mutually exclusive")
    if hours is not None and hours < 0:
        raise ValueError("hours must be non-negative")
    if cycles is not None and cycles < 0:
        raise ValueError("cycles must be non-negative")
    if hours is None and cycles is None:
        cycles = 1

    opts = options or LiveEvolutionOptions()
    paths = getattr(runtime, "paths", None)
    if not isinstance(paths, dict) or "root" not in paths:
        raise ValueError("runtime must expose paths['root'] for Strength Lab state")
    state_root = Path(paths["root"]).expanduser().resolve()
    strength_path = state_root / opts.strength_db_name

    copy_validation = os.environ.get("DOGMATIST_V2_COPY_VALIDATION", "") == "1"
    copy_parallel_games = 2
    expire_on_league_start = False
    if copy_validation:
        isolated_home = Path(os.environ.get("HOME", "")).expanduser().resolve()
        expected_state = (isolated_home / ".darwinchess").resolve()
        if state_root != expected_state:
            raise RuntimeError(
                "copied-state validation refused: runtime state root does not match isolated HOME "
                f"({state_root} != {expected_state})"
            )
        copy_parallel_games = _copy_validation_parallel_games()
        expire_on_league_start = (
            os.environ.get("DOGMATIST_V2_COPY_EXPIRE_ON_LEAGUE_START", "") == "1"
        )
        if opts.persist_teacher:
            progress("[dog_matist][copy-validation] forcing teacher replay persistence OFF")
        opts = replace(opts, persist_teacher=False)
        progress(
            "DOGMATIST_UI " + json.dumps({
                "phase": "copy_validation",
                "copy_validation": {
                    "enabled": True,
                    "state_root": str(state_root),
                    "teacher_persistence": False,
                    "league_parallel_games": copy_parallel_games,
                    "expire_on_league_start": expire_on_league_start,
                },
            }, ensure_ascii=False)
        )

    if opts.enable_parallel_league:
        install_parallel_league_signal_safety()

    if hours is not None:
        clock = HeartbeatComputeClock(float(hours) * 3600.0)
    else:
        clock = _cycle_only_clock(int(cycles or 1))

    watchdog = opts.watchdog_policy()
    progress(
        "[dog_matist][watchdog] run budget never interrupts active games; "
        f"stall={watchdog.stall_seconds:.0f}s emergency_game={watchdog.emergency_game_seconds:.0f}s"
    )
    progress(
        "DOGMATIST_UI " + json.dumps({
            "phase": "watchdog_policy",
            "watchdog": watchdog.ui_payload(),
        }, ensure_ascii=False)
    )

    expiry_probe_triggered = False

    def live_league_status(payload: dict[str, object]) -> None:
        nonlocal expiry_probe_triggered
        if expire_on_league_start and not expiry_probe_triggered:
            league = payload.get("league")
            active = league.get("active_games") if isinstance(league, dict) else None
            if isinstance(active, list) and active:
                expiry_probe_triggered = True
                clock.request_expiry()
                progress(
                    "[dog_matist][copy-validation] injected compute-budget expiry after League start; "
                    "active games must finish naturally"
                )
                progress(
                    "DOGMATIST_UI " + json.dumps({
                        "phase": "validation_budget_expired",
                        "reason": "copied_state_league_start_probe",
                        "active_games_at_expiry": len(active),
                        "compute": clock.snapshot(),
                    }, ensure_ascii=False, default=str)
                )
        if league_status_callback is not None:
            league_status_callback(payload)

    def fixed_status(payload: dict[str, object]) -> None:
        progress("DOGMATIST_UI " + json.dumps(payload, ensure_ascii=False, default=str))
        if league_status_callback is not None:
            league_status_callback(payload)

    with _copy_validation_runtime_overrides(runtime, copy_validation), install_live_game_watchdog_policy(runtime, watchdog), StrengthStore(strength_path) as store:
        coordinator = LiveStrengthCoordinator(runtime, store)
        runner = LiveEvolutionRunner(
            runtime,
            clock,
            strength_coordinator=coordinator,
            targeted_examples=opts.targeted_examples,
            teacher_request_cap=opts.teacher_request_cap,
            persist_teacher=opts.persist_teacher,
            strength_fail_open=opts.strength_fail_open,
            enable_parallel_league=opts.enable_parallel_league,
            parallel_league_fail_open=opts.parallel_league_fail_open,
            league_status_callback=live_league_status,
            handle_sigint=opts.handle_sigint,
        )

        fixed_context: Any = nullcontext()
        fixed_fail_context: Any = nullcontext()
        if opts.enable_fixed_reference:
            fixed = LiveFixedReferenceCoordinator(
                runtime,
                store,
                clock,
                reference_root=state_root / opts.fixed_reference_dir_name,
                pair_count=opts.fixed_reference_pairs,
                watchdog_policy=watchdog,
                stop_requested=lambda: runner._stop_requested,
                status_callback=fixed_status,
            )
            try:
                reference = fixed.ensure_reference()
                progress(
                    "[dog_matist][fixed-reference] "
                    f"reference={reference.reference_id} generation={reference.generation} "
                    f"pairs/round={opts.fixed_reference_pairs}"
                )
                progress(
                    "DOGMATIST_UI " + json.dumps({
                        "phase": "fixed_reference_ready",
                        "fixed_reference": {
                            "reference": reference.as_dict(),
                            "trend": [
                                {
                                    "round_index": row.round_index,
                                    "champion_generation": row.champion_generation,
                                    "score": row.fixed_reference_score,
                                    "games": row.paired_games,
                                    "promoted": row.promoted,
                                }
                                for row in store.round_history(limit=8)
                            ],
                        },
                    }, ensure_ascii=False, default=str)
                )
                fixed_override = LiveFixedReferenceCycleOverride(runtime, fixed)
                history = store.round_history(limit=1)
                if history:
                    fixed_override._round_counter = int(history[-1].round_index)
                fixed_context = fixed_override
                fixed_fail_context = fixed_reference_fail_open(
                    fixed,
                    enabled=opts.fixed_reference_fail_open,
                )
            except Exception as exc:
                if not opts.fixed_reference_fail_open:
                    raise
                progress(
                    "[dog_matist][fixed-reference] disabled for this run: "
                    f"{type(exc).__name__}: {exc}"
                )
                progress(
                    "DOGMATIST_UI " + json.dumps({
                        "phase": "fixed_reference_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "fallback": "ordinary_evolution_continues",
                    }, ensure_ascii=False)
                )

        with fixed_fail_context, fixed_context:
            report = runner.run(cycles=cycles, progress=progress)

    progress(
        "[dog_matist][v2-live] "
        f"cycles={report.cycles_completed} stop={report.stop_reason} "
        f"compute={float(report.compute.get('elapsed_seconds', 0.0)):.1f}s"
    )
    return report
