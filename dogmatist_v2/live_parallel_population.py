from __future__ import annotations

import importlib
import json
import time
from typing import Any, Callable

from .live_league_guard import LiveLeagueDrainState
from .live_parallel_league import (
    LiveLeagueProcessPool,
    LiveLeagueWorkerResult,
    LiveLeagueWorkerTask,
    choose_live_league_parallelism,
    league_worker_threads,
)
from .runtime import ColorPairing


def build_parallel_population_arena(
    base_arena_cls: type,
    *,
    runtime: Any,
    clock: Any,
    state: LiveLeagueDrainState,
    fail_open: bool = True,
    status_callback: Callable[[dict[str, object]], None] | None = None,
) -> type:
    """Create a production PopulationArena using 2-3 killable game processes.

    Candidate checkpoints are already saved before League starts, so workers load
    those immutable files on CPU. SQLite writes remain in the parent process. A
    colour pair contributes to rating/specialist evidence only if *both* legs
    finish successfully; a single timeout can never create one-sided evidence.
    """

    population_module = importlib.import_module(base_arena_cls.__module__)

    class ParallelPopulationArena(base_arena_cls):
        def _checkpoint_path(self, generation: int) -> str:
            row = self.memory.get_generation(int(generation))
            if row is None:
                raise RuntimeError(f"missing generation {generation} for parallel League")
            path = str(row["checkpoint_path"])
            if not path:
                raise RuntimeError(f"generation {generation} has no checkpoint path")
            return path

        def _build_phase(
            self,
            *,
            round_id: int,
            definitions: list[tuple[str, int, int, Any, str]],
            depth: int,
            max_plies: int,
            parallel_games: int,
        ) -> tuple[list[ColorPairing], dict[str, LiveLeagueWorkerTask], dict[str, int]]:
            pairings: list[ColorPairing] = []
            tasks: dict[str, LiveLeagueWorkerTask] = {}
            first_generation: dict[str, int] = {}
            threads = league_worker_threads(runtime, parallel_games)
            checkpoint_cache: dict[int, str] = {}

            def checkpoint(generation: int) -> str:
                if generation not in checkpoint_cache:
                    checkpoint_cache[generation] = self._checkpoint_path(generation)
                return checkpoint_cache[generation]

            for ordinal, (pair_id, first_g, second_g, board, opening) in enumerate(definitions):
                pair = ColorPairing(
                    pairing_id=pair_id,
                    first_white_id=str(first_g),
                    first_black_id=str(second_g),
                    opening=opening,
                )
                pairings.append(pair)
                first_generation[pair_id] = int(first_g)
                for spec in pair.games():
                    white_g = int(spec.white_id)
                    black_g = int(spec.black_id)
                    tasks[spec.game_id] = LiveLeagueWorkerTask(
                        game_id=spec.game_id,
                        pairing_id=pair_id,
                        leg=int(spec.leg),
                        round_id=int(round_id),
                        white_generation=white_g,
                        black_generation=black_g,
                        white_checkpoint=checkpoint(white_g),
                        black_checkpoint=checkpoint(black_g),
                        config=self.config,
                        start_fen=board.fen(),
                        opening_name=opening,
                        depth=int(depth),
                        max_plies=int(max_plies),
                        seed=int(round_id) * 1_000_000 + ordinal * 2 + int(spec.leg) - 1,
                        torch_threads=threads,
                    )
            return pairings, tasks, first_generation

        def _persist_result(
            self,
            row: LiveLeagueWorkerResult,
            *,
            round_id: int,
            source: str,
            pair_valid: bool,
            first_generation: int,
        ) -> str:
            return str(self.memory.add_game(
                source=source,
                generation=int(first_generation),
                white_agent=f"league-g{row.white_generation}",
                black_agent=f"league-g{row.black_generation}",
                result=row.result,
                termination=row.termination,
                pgn=row.pgn,
                plies=row.plies,
                examples=[],
                metadata={
                    **row.metadata,
                    "population_round": int(round_id),
                    "paired_colors": True,
                    "parallel_worker": True,
                    "league_pairing_id": row.pairing_id,
                    "league_leg": row.leg,
                    "pair_valid": bool(pair_valid),
                    "worker_elapsed_s": row.elapsed_s,
                },
            ))

        def _commit_phase(
            self,
            execution: Any,
            pairings: list[ColorPairing],
            first_generation: dict[str, int],
            *,
            round_id: int,
            table: Any,
        ) -> tuple[int, tuple[str, ...]]:
            result_map = execution.result_by_game
            valid_pairs = 0
            invalid: list[str] = []
            for pair in pairings:
                specs = pair.games()
                rows = [result_map.get(spec.game_id) for spec in specs]
                pair_valid = all(row is not None for row in rows)
                if not pair_valid:
                    invalid.append(pair.pairing_id)
                    # Preserve completed compute as history, but never add it to
                    # league_matches/table without the reverse-colour leg.
                    for row in rows:
                        if row is not None:
                            self._persist_result(
                                row,
                                round_id=round_id,
                                source="league_partial",
                                pair_valid=False,
                                first_generation=first_generation[pair.pairing_id],
                            )
                    continue

                valid_pairs += 1
                for row in rows:
                    assert row is not None
                    gid = self._persist_result(
                        row,
                        round_id=round_id,
                        source="league",
                        pair_valid=True,
                        first_generation=first_generation[pair.pairing_id],
                    )
                    self.memory.add_league_match(
                        int(round_id),
                        int(row.white_generation),
                        int(row.black_generation),
                        gid,
                        row.opening_name,
                        row.result,
                        row.white_score,
                    )
                    table.add(
                        int(row.white_generation),
                        int(row.black_generation),
                        row.white_score,
                        row.opening_name,
                    )
                    self._league_played += 1
                    print(
                        f"[dog_matist][stage=league][detail={self._league_played}/{self._league_total}] "
                        f"result={row.result} opening={row.opening_name}",
                        flush=True,
                    )
            return valid_pairs, tuple(invalid)

        def _run_phase(
            self,
            *,
            round_id: int,
            definitions: list[tuple[str, int, int, Any, str]],
            depth: int,
            max_plies: int,
            table: Any,
            parallel_games: int,
        ) -> Any:
            pairings, tasks, first_generation = self._build_phase(
                round_id=round_id,
                definitions=definitions,
                depth=depth,
                max_plies=max_plies,
                parallel_games=parallel_games,
            )
            if not pairings:
                return None

            lcfg = self.config.get("league", {})
            last_emit = [0.0]

            def emit(snapshot: dict[str, object]) -> None:
                now = time.monotonic()
                terminal_change = bool(snapshot.get("failed_games") or snapshot.get("timed_out_games"))
                if not terminal_change and now - last_emit[0] < 0.8:
                    return
                last_emit[0] = now
                payload = {
                    "phase": "league",
                    "league": {
                        **snapshot,
                        "played": self._league_played,
                        "total": self._league_total,
                    },
                }
                print("DOGMATIST_UI " + json.dumps(payload, ensure_ascii=False), flush=True)
                if status_callback is not None:
                    status_callback(payload)

            pool = LiveLeagueProcessPool(
                pairings,
                tasks,
                clock=clock,
                parallel_games=parallel_games,
                hard_game_timeout_seconds=float(lcfg.get("watchdog_hard_seconds", 20 * 60)),
                stall_timeout_seconds=float(lcfg.get("watchdog_stall_seconds", 4 * 60)),
                kill_grace_seconds=float(lcfg.get("watchdog_kill_grace_seconds", 1.0)),
                status_callback=emit,
            )
            execution = pool.run()
            snap = execution.final_snapshot
            state.pairs_started += int(snap.get("pairs_started", 0) or 0)
            state.pairs_completed += int(snap.get("pairs_completed", 0) or 0)
            if execution.draining:
                state.request_drain(execution.stop_reason or "parallel_league_drain")
            valid_pairs, invalid_pairs = self._commit_phase(
                execution,
                pairings,
                first_generation,
                round_id=round_id,
                table=table,
            )
            if invalid_pairs:
                state.request_drain(
                    "watchdog_timeout" if execution.timed_out_game_ids else "worker_failure"
                )
                print(
                    f"[dog_matist][league] invalid colour pairs excluded from evidence: {list(invalid_pairs)}",
                    flush=True,
                )
            return execution

        def run(
            self,
            *,
            round_id: int,
            champion_generation: int,
            members: dict[int, Any],
            candidate_plans: list[Any],
        ) -> Any:
            lcfg = self.config.get("league", {})
            anchor_pairs = max(1, int(lcfg.get("anchor_pairs", 1)))
            playoff_pairs = max(1, int(lcfg.get("playoff_pairs", 1)))
            depth = int(lcfg.get("depth", self.config["arena"].get("depth", self.config["search"]["depth"])))
            max_plies = int(lcfg.get("max_game_plies", self.config["arena"].get("max_game_plies", 220)))
            k_factor = float(lcfg.get("k_factor", 20.0))
            parallel_games = choose_live_league_parallelism(runtime)

            generations = [int(champion_generation)] + [int(p.generation) for p in candidate_plans]
            table = population_module.LeagueTable(generations, k_factor=k_factor)
            self._league_played = 0
            self._league_total = len(candidate_plans) * anchor_pairs * 2
            if len(candidate_plans) >= 2:
                self._league_total += playoff_pairs * 2
            print(
                f"[dog_matist][stage=league][detail=0/{self._league_total} depth={depth} "
                f"max_plies={max_plies} parallel={parallel_games}]",
                flush=True,
            )

            try:
                seed = int(self.config["project"].get("seed", 0)) + int(round_id) * 10007
                curriculum = population_module.OpeningCurriculum(seed=seed)
                anchors = curriculum.arena_pairs(anchor_pairs * max(1, len(candidate_plans)))
                definitions: list[tuple[str, int, int, Any, str]] = []
                cursor = 0
                for plan in candidate_plans:
                    selected = anchors[cursor:cursor + anchor_pairs]
                    cursor += anchor_pairs
                    for local_index, (board, opening) in enumerate(selected):
                        definitions.append((
                            f"r{round_id}-anchor-g{int(plan.generation)}-{local_index}",
                            int(plan.generation),
                            int(champion_generation),
                            board,
                            opening,
                        ))

                self._run_phase(
                    round_id=round_id,
                    definitions=definitions,
                    depth=depth,
                    max_plies=max_plies,
                    table=table,
                    parallel_games=parallel_games,
                )

                if not state.draining:
                    challengers = [r for r in table.ranking() if r.generation != champion_generation]
                    if len(challengers) >= 2:
                        a, b = int(challengers[0].generation), int(challengers[1].generation)
                        playoff = population_module.OpeningCurriculum(seed=seed + 7919).arena_pairs(playoff_pairs)
                        playoff_defs = [
                            (f"r{round_id}-playoff-{index}", a, b, board, opening)
                            for index, (board, opening) in enumerate(playoff)
                        ]
                        self._run_phase(
                            round_id=round_id,
                            definitions=playoff_defs,
                            depth=depth,
                            max_plies=max_plies,
                            table=table,
                            parallel_games=parallel_games,
                        )
            except Exception as exc:
                if fail_open and self._league_played == 0:
                    print(f"[dog_matist][league] process-worker fallback to serial: {exc}", flush=True)
                    return super().run(
                        round_id=round_id,
                        champion_generation=champion_generation,
                        members=members,
                        candidate_plans=candidate_plans,
                    )
                state.request_drain("parallel_league_error")
                raise

            ranking_rows = table.ranking()
            partial = bool(state.draining)
            for row in ranking_rows:
                if int(row.generation) == int(champion_generation):
                    continue
                self.memory.update_population_member(
                    round_id,
                    row.generation,
                    league_score=row.score,
                    rating=row.rating,
                    status="league_partial" if partial else "league_complete",
                )

            specialists = {} if partial else self._archive_specialists(
                round_id, table, champion_generation
            )
            top = next(
                (int(r.generation) for r in ranking_rows if int(r.generation) != int(champion_generation)),
                int(candidate_plans[0].generation) if candidate_plans else int(champion_generation),
            )
            return population_module.LeagueSummary(
                round_id=round_id,
                ranking=[
                    {
                        "generation": r.generation,
                        "games": r.games,
                        "score": r.score,
                        "rating": r.rating,
                        "wins": r.wins,
                        "draws": r.draws,
                        "losses": r.losses,
                    }
                    for r in ranking_rows
                ],
                top_generation=top,
                specialist_generations=specialists,
            )

    ParallelPopulationArena.__name__ = f"Parallel{base_arena_cls.__name__}"
    return ParallelPopulationArena


class LiveParallelLeagueOverride:
    """Install the real process-backed League only for the active cycle."""

    def __init__(
        self,
        runtime: Any,
        clock: Any,
        *,
        runtime_module: Any | None = None,
        state: LiveLeagueDrainState | None = None,
        fail_open: bool = True,
        status_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.clock = clock
        self.runtime_module = runtime_module
        self.state = state or LiveLeagueDrainState()
        self.fail_open = bool(fail_open)
        self.status_callback = status_callback
        self._base: Any | None = None
        self.latest_ui: dict[str, object] | None = None

    def _module(self) -> Any:
        if self.runtime_module is not None:
            return self.runtime_module
        return importlib.import_module(self.runtime.__class__.__module__)

    def __enter__(self) -> "LiveParallelLeagueOverride":
        module = self._module()
        self.runtime_module = module
        self._base = module.PopulationArena

        def capture(payload: dict[str, object]) -> None:
            self.latest_ui = payload
            if self.status_callback is not None:
                self.status_callback(payload)

        module.PopulationArena = build_parallel_population_arena(
            self._base,
            runtime=self.runtime,
            clock=self.clock,
            state=self.state,
            fail_open=self.fail_open,
            status_callback=capture,
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self.runtime_module is not None and self._base is not None:
            self.runtime_module.PopulationArena = self._base
