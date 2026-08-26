from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty
import time
import traceback
from typing import Any, Callable, Mapping

from .runtime import ColorPairing, LeaguePairScheduler


@dataclass(frozen=True)
class LiveLeagueWorkerTask:
    game_id: str
    pairing_id: str
    leg: int
    round_id: int
    white_generation: int
    black_generation: int
    white_checkpoint: str
    black_checkpoint: str
    config: dict[str, Any]
    start_fen: str
    opening_name: str
    depth: int
    max_plies: int
    seed: int
    torch_threads: int = 1


@dataclass(frozen=True)
class LiveLeagueWorkerResult:
    game_id: str
    pairing_id: str
    leg: int
    white_generation: int
    black_generation: int
    opening_name: str
    result: str
    termination: str
    pgn: str
    plies: int
    metadata: dict[str, Any]
    elapsed_s: float

    @property
    def white_score(self) -> float:
        if self.result == "1-0":
            return 1.0
        if self.result == "0-1":
            return 0.0
        return 0.5


@dataclass(frozen=True)
class LiveParallelLeagueExecution:
    results: tuple[LiveLeagueWorkerResult, ...]
    failed_game_ids: tuple[str, ...]
    timed_out_game_ids: tuple[str, ...]
    draining: bool
    stop_reason: str | None
    parallel_games: int
    final_snapshot: dict[str, object]

    @property
    def result_by_game(self) -> dict[str, LiveLeagueWorkerResult]:
        return {row.game_id: row for row in self.results}


class _ProgressSearcher:
    """Child-process proxy that reports one heartbeat after each completed search."""

    def __init__(self, inner: Any, *, game_id: str, event_queue: Any, counter: list[int]) -> None:
        self.inner = inner
        self.game_id = game_id
        self.event_queue = event_queue
        self.counter = counter

    def search(self, *args: Any, **kwargs: Any) -> Any:
        result = self.inner.search(*args, **kwargs)
        self.counter[0] += 1
        try:
            self.event_queue.put({
                "kind": "progress",
                "game_id": self.game_id,
                "plies": self.counter[0],
            })
        except Exception:
            pass
        return result


def _league_worker_main(task: LiveLeagueWorkerTask, event_queue: Any) -> None:
    """Spawn-safe production League worker.

    It intentionally imports the live ``darwinchess`` package only inside the
    child process. The parent process remains the single SQLite writer; workers
    only return compact game records through the queue.
    """

    started = time.monotonic()
    try:
        event_queue.put({"kind": "started", "game_id": task.game_id})

        import chess  # type: ignore
        import torch  # type: ignore
        from darwinchess.evaluator import HybridEvaluator  # type: ignore
        from darwinchess.genome import AgentGenome  # type: ignore
        from darwinchess.network import load_checkpoint  # type: ignore
        from darwinchess.search import AlphaBetaSearcher  # type: ignore
        from darwinchess.selfplay import play_game  # type: ignore

        torch.set_num_threads(max(1, int(task.torch_threads)))
        device = torch.device("cpu")

        def make_searcher(path: str) -> Any:
            model, payload = load_checkpoint(path, device)
            genome = AgentGenome.from_dict(
                payload.get("metadata", {}).get("genome"),
                task.config,
            )
            model.to(device).eval()
            return AlphaBetaSearcher(
                HybridEvaluator(model, task.config, device, genome),
                task.config,
            )

        counter = [0]
        white = _ProgressSearcher(
            make_searcher(task.white_checkpoint),
            game_id=task.game_id,
            event_queue=event_queue,
            counter=counter,
        )
        black = _ProgressSearcher(
            make_searcher(task.black_checkpoint),
            game_id=task.game_id,
            event_queue=event_queue,
            counter=counter,
        )
        board = chess.Board(task.start_fen)
        record = play_game(
            white,
            black,
            task.config,
            white_name=f"league-g{task.white_generation}",
            black_name=f"league-g{task.black_generation}",
            stochastic=False,
            seed=task.seed,
            depth=task.depth,
            max_plies=task.max_plies,
            starting_board=board,
            opening_name=task.opening_name,
            opening_family="league",
        )
        payload = LiveLeagueWorkerResult(
            game_id=task.game_id,
            pairing_id=task.pairing_id,
            leg=task.leg,
            white_generation=task.white_generation,
            black_generation=task.black_generation,
            opening_name=task.opening_name,
            result=record.result,
            termination=record.termination,
            pgn=record.pgn,
            plies=int(record.plies),
            metadata=dict(record.metadata),
            elapsed_s=max(0.0, time.monotonic() - started),
        )
        event_queue.put({"kind": "finished", "game_id": task.game_id, "result": payload})
    except BaseException as exc:
        try:
            event_queue.put({
                "kind": "failed",
                "game_id": task.game_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            })
        except Exception:
            pass


def choose_live_league_parallelism(runtime: Any) -> int:
    """Choose 2 or 3 real League game processes from current Mac headroom."""

    cfg = getattr(runtime, "config", {}) or {}
    explicit = int(cfg.get("runtime", {}).get("league_parallel_games", 0) or 0)
    if explicit in (2, 3):
        return explicit

    budget = getattr(runtime, "_last_resource_budget", None) or {}
    snapshot = budget.get("snapshot", {}) if isinstance(budget, dict) else {}
    cpu_count = int(snapshot.get("cpu_count", 0) or 0)
    load = snapshot.get("load_percent")
    memory = snapshot.get("memory_percent")
    thermal = str(snapshot.get("thermal_pressure") or "unknown")
    reason = str(budget.get("reason") or "") if isinstance(budget, dict) else ""

    load_ok = load is None or float(load) < 65.0
    memory_ok = memory is None or float(memory) < 72.0
    if (
        reason == "headroom available"
        and cpu_count >= 8
        and load_ok
        and memory_ok
        and thermal in {"nominal", "unknown"}
    ):
        return 3
    return 2


def league_worker_threads(runtime: Any, parallel_games: int) -> int:
    cfg = getattr(runtime, "config", {}) or {}
    total = int(cfg.get("runtime", {}).get("torch_threads", 0) or 0)
    if total <= 0:
        return 1
    return max(1, total // max(1, int(parallel_games)))


class LiveLeagueProcessPool:
    """Drive real killable League processes with the tested pair scheduler.

    The scheduler owns admission and fairness. This class owns only process
    lifecycle and IPC. If budget expires, no new pairing is opened, but missing
    reverse-colour legs of already-started pairings are still launched. A worker
    timeout/failure requests the same drain behavior and terminates the offending
    process, preventing a wedged search from hanging the overnight run forever.
    """

    def __init__(
        self,
        pairings: list[ColorPairing],
        tasks: Mapping[str, LiveLeagueWorkerTask],
        *,
        clock: Any,
        parallel_games: int,
        hard_game_timeout_seconds: float = 20 * 60,
        stall_timeout_seconds: float = 4 * 60,
        kill_grace_seconds: float = 1.0,
        poll_interval_seconds: float = 0.10,
        status_callback: Callable[[dict[str, object]], None] | None = None,
        mp_context: Any | None = None,
        worker_target: Callable[[LiveLeagueWorkerTask, Any], None] = _league_worker_main,
    ) -> None:
        if parallel_games not in (2, 3):
            raise ValueError("parallel_games must be 2 or 3")
        if kill_grace_seconds < 0 or poll_interval_seconds <= 0:
            raise ValueError("invalid process-pool timing")
        expected = {game.game_id for pair in pairings for game in pair.games()}
        missing = expected - set(tasks)
        if missing:
            raise ValueError(f"missing worker tasks for games: {sorted(missing)}")
        self.pairings = list(pairings)
        self.tasks = dict(tasks)
        self.clock = clock
        self.parallel_games = parallel_games
        self.kill_grace_seconds = float(kill_grace_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.status_callback = status_callback
        self.context = mp_context or mp.get_context("spawn")
        self.worker_target = worker_target
        self.scheduler = LeaguePairScheduler(
            self.pairings,
            clock,
            parallel_games=parallel_games,
            hard_game_timeout_seconds=hard_game_timeout_seconds,
            stall_timeout_seconds=stall_timeout_seconds,
        )
        self._queue = self.context.Queue()
        self._processes: dict[str, Any] = {}
        self._dead_since: dict[str, float] = {}
        self._results: dict[str, LiveLeagueWorkerResult] = {}
        self._failed: set[str] = set()
        self._timed_out: set[str] = set()
        self._started_pairs: set[str] = set()
        self._completed_pairs: set[str] = set()

    def _snapshot(self) -> dict[str, object]:
        snap = dict(self.scheduler.snapshot())
        snap["worker_pids"] = {
            game_id: int(proc.pid or 0) for game_id, proc in self._processes.items()
        }
        snap["results"] = len(self._results)
        snap["failed_games"] = sorted(self._failed)
        snap["timed_out_games"] = sorted(self._timed_out)
        snap["pairs_started"] = len(self._started_pairs)
        snap["pairs_completed"] = len(self._completed_pairs)
        return snap

    def _emit(self) -> None:
        if self.status_callback is not None:
            self.status_callback(self._snapshot())

    def _spawn(self, game_id: str) -> None:
        task = self.tasks[game_id]
        proc = self.context.Process(
            target=self.worker_target,
            args=(task, self._queue),
            name=f"dogmatist-league-{game_id}",
            daemon=True,
        )
        proc.start()
        self._processes[game_id] = proc
        self._started_pairs.add(task.pairing_id)

    def _reap(self, game_id: str, *, terminate: bool = False) -> None:
        proc = self._processes.pop(game_id, None)
        self._dead_since.pop(game_id, None)
        if proc is None:
            return
        if terminate and proc.is_alive():
            proc.terminate()
            proc.join(timeout=self.kill_grace_seconds)
            if proc.is_alive():
                try:
                    proc.kill()
                except AttributeError:
                    proc.terminate()
        proc.join(timeout=max(0.1, self.kill_grace_seconds))

    def _refresh_completed_pairs(self) -> None:
        for pair in self.pairings:
            if self.scheduler.pair_complete(pair.pairing_id):
                self._completed_pairs.add(pair.pairing_id)

    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "")
        game_id = str(event.get("game_id") or "")
        active_ids = {g.spec.game_id for g in self.scheduler.active_games}
        if not game_id:
            return
        if kind == "progress":
            if game_id in active_ids:
                self.scheduler.report_progress(game_id, int(event.get("plies", 0) or 0))
            return
        if kind == "finished":
            if game_id not in active_ids:
                self._reap(game_id)
                return
            result = event.get("result")
            if not isinstance(result, LiveLeagueWorkerResult):
                self.scheduler.fail(game_id, "invalid_worker_result")
                self._failed.add(game_id)
                self.scheduler.request_drain("worker_failure")
            else:
                self._results[game_id] = result
                self.scheduler.complete(game_id, result=result.result)
            self._reap(game_id)
            self._refresh_completed_pairs()
            return
        if kind == "failed":
            if game_id in active_ids:
                reason = str(event.get("error") or "worker_failure")
                self.scheduler.fail(game_id, reason)
            self._failed.add(game_id)
            self.scheduler.request_drain("worker_failure")
            self._reap(game_id, terminate=True)
            self._refresh_completed_pairs()

    def _drain_events(self) -> bool:
        handled = False
        while True:
            try:
                event = self._queue.get_nowait()
            except Empty:
                break
            if isinstance(event, dict):
                self._handle_event(event)
                handled = True
        return handled

    def _poll_watchdogs(self) -> None:
        trips = self.scheduler.poll_watchdogs()
        if not trips:
            return
        self.scheduler.request_drain("watchdog_timeout")
        for trip in trips:
            self._timed_out.add(trip.game_id)
            self._reap(trip.game_id, terminate=True)
        self._refresh_completed_pairs()

    def _poll_dead_workers(self) -> None:
        active_ids = {g.spec.game_id for g in self.scheduler.active_games}
        now = time.monotonic()
        for game_id, proc in list(self._processes.items()):
            if proc.is_alive() or game_id not in active_ids:
                self._dead_since.pop(game_id, None)
                continue
            first = self._dead_since.setdefault(game_id, now)
            # Give multiprocessing.Queue a short flush window after process exit.
            if now - first < 0.35:
                continue
            self.scheduler.fail(game_id, f"worker_exit_{proc.exitcode}")
            self._failed.add(game_id)
            self.scheduler.request_drain("worker_failure")
            self._reap(game_id)
        self._refresh_completed_pairs()

    def run(self) -> LiveParallelLeagueExecution:
        try:
            while True:
                for status in self.scheduler.poll_startable():
                    self._spawn(status.spec.game_id)
                self._emit()

                if self.scheduler.safe_to_stop or self.scheduler.all_pairings_complete:
                    break

                handled = self._drain_events()
                self._poll_watchdogs()
                self._poll_dead_workers()
                self._emit()

                if not handled:
                    time.sleep(self.poll_interval_seconds)

            # One last IPC drain catches results flushed at process exit.
            self._drain_events()
            self._refresh_completed_pairs()
        finally:
            for game_id in list(self._processes):
                self._reap(game_id, terminate=True)
            try:
                self._queue.close()
                self._queue.join_thread()
            except Exception:
                pass

        return LiveParallelLeagueExecution(
            results=tuple(self._results[key] for key in sorted(self._results)),
            failed_game_ids=tuple(sorted(self._failed)),
            timed_out_game_ids=tuple(sorted(self._timed_out)),
            draining=self.scheduler.draining,
            stop_reason=self.scheduler.stop_reason,
            parallel_games=self.parallel_games,
            final_snapshot=self._snapshot(),
        )
