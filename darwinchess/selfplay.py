from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import random

import chess
import chess.pgn

from .memory import ReplayExample
from .opening_curriculum import CurriculumMix, OpeningCurriculum
from .search import AlphaBetaSearcher, SearchResult


@dataclass
class PositionTrace:
    fen: str
    policy_move_uci: str
    played_move_uci: str
    turn: chess.Color
    played_score_cp: float
    best_score_cp: float


@dataclass
class GameRecord:
    result: str
    winner: chess.Color | None
    termination: str
    plies: int
    pgn: str
    examples: list[ReplayExample]
    metadata: dict[str, Any]


def _result_string(winner: chess.Color | None) -> str:
    if winner is chess.WHITE:
        return "1-0"
    if winner is chess.BLACK:
        return "0-1"
    return "1/2-1/2"


def _choose_stochastic(result: SearchResult, ply: int, config: dict[str, Any], rng: random.Random) -> chess.Move:
    if result.move is None:
        raise RuntimeError("Search returned no move in a non-terminal position")
    sp = config["selfplay"]
    if ply >= int(sp.get("temperature_plies", 16)) or len(result.candidates) <= 1:
        return result.move
    temperature = float(sp.get("opening_temperature", 0.8))
    top_k = int(sp.get("top_k", 5))
    candidates = result.candidates[:top_k]
    if temperature <= 1e-6:
        return candidates[0].move
    denom = max(20.0, temperature * 160.0)
    m = max(c.score_cp for c in candidates)
    weights = [math.exp(max(-12.0, min(12.0, (c.score_cp - m) / denom))) for c in candidates]
    return rng.choices([c.move for c in candidates], weights=weights, k=1)[0]


def build_pgn(
    moves: list[chess.Move],
    result: str,
    white_name: str,
    black_name: str,
    termination: str,
    *,
    initial_board: chess.Board | None = None,
    opening_name: str | None = None,
) -> str:
    board = (initial_board or chess.Board()).copy(stack=False)
    game = chess.pgn.Game()
    game.headers["Event"] = "dog_matist"
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    if opening_name:
        game.headers["Opening"] = opening_name
    if board.fen() != chess.Board().fen():
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = board.fen()
        game.setup(board)

    node = game
    for move in moves:
        if move not in board.legal_moves:
            break
        node = node.add_variation(move)
        board.push(move)
    return str(game)


def _sample_selfplay_opening(config: dict[str, Any], seed: int | None) -> tuple[chess.Board, str, str]:
    sp = config.get("selfplay", {})
    ocfg = sp.get("opening_curriculum", {})
    if not bool(ocfg.get("enabled", True)):
        return chess.Board(), "Initial position", "standard"
    mix = CurriculumMix(
        standard=float(ocfg.get("standard", 0.35)),
        curated=float(ocfg.get("curated", 0.35)),
        uncommon=float(ocfg.get("uncommon", 0.20)),
        controlled_random=float(ocfg.get("controlled_random", 0.10)),
    )
    curriculum = OpeningCurriculum(seed=seed, mix=mix)
    return curriculum.sample()


def play_game(
    white_searcher: AlphaBetaSearcher,
    black_searcher: AlphaBetaSearcher,
    config: dict[str, Any],
    *,
    white_name: str,
    black_name: str,
    stochastic: bool = True,
    seed: int | None = None,
    depth: int | None = None,
    max_plies: int | None = None,
    starting_board: chess.Board | None = None,
    opening_name: str = "Initial position",
    opening_family: str = "standard",
) -> GameRecord:
    rng = random.Random(seed)
    if starting_board is None and stochastic:
        starting_board, opening_name, opening_family = _sample_selfplay_opening(config, seed)
    board = (starting_board or chess.Board()).copy(stack=False)
    initial_board = board.copy(stack=False)
    moves: list[chess.Move] = []
    traces: list[PositionTrace] = []
    max_plies = int(max_plies or config["search"].get("max_game_plies", 240))
    sp = config["selfplay"]
    resign_threshold = float(sp.get("resign_threshold_cp", -1100))
    resign_consecutive = int(sp.get("resign_consecutive", 5))
    allow_resign_after = int(sp.get("allow_resign_after_ply", 50))
    losing_run = {chess.WHITE: 0, chess.BLACK: 0}
    forced_winner: chess.Color | None = None
    termination = "normal"

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        forced_winner = outcome.winner if outcome else None
        termination = outcome.termination.name.lower() if outcome else "terminal_seed"

    for ply in range(max_plies):
        if termination != "normal":
            break
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            forced_winner = outcome.winner
            termination = outcome.termination.name.lower()
            break

        searcher = white_searcher if board.turn == chess.WHITE else black_searcher
        result = searcher.search(board, depth=depth, top_n=max(5, int(sp.get("top_k", 5))))
        if result.move is None:
            forced_winner = None
            termination = "no_legal_move"
            break
        if ply >= allow_resign_after:
            if result.score_cp <= resign_threshold:
                losing_run[board.turn] += 1
            else:
                losing_run[board.turn] = 0
            if losing_run[board.turn] >= resign_consecutive:
                forced_winner = not board.turn
                termination = "resignation"
                break

        move = _choose_stochastic(result, ply, config, rng) if stochastic else result.move
        best_candidate = result.candidates[0] if result.candidates else None
        best_move = best_candidate.move if best_candidate is not None else result.move
        best_score = best_candidate.score_cp if best_candidate is not None else result.score_cp
        chosen_score = next((c.score_cp for c in result.candidates if c.move == move), result.score_cp)
        traces.append(PositionTrace(
            board.fen(), best_move.uci(), move.uci(), board.turn, chosen_score, best_score
        ))
        board.push(move)
        moves.append(move)
    else:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            forced_winner = outcome.winner
            termination = outcome.termination.name.lower()
        else:
            forced_winner = None
            termination = "max_plies_draw"

    if termination == "normal":
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            forced_winner = outcome.winner
            termination = outcome.termination.name.lower()
        else:
            forced_winner = None
            termination = "max_plies_draw"

    result_str = _result_string(forced_winner)
    examples: list[ReplayExample] = []
    for tr in traces:
        if forced_winner is None:
            target = 0.0
        else:
            target = 1.0 if tr.turn == forced_winner else -1.0
        gap = max(0.0, tr.best_score_cp - tr.played_score_cp)
        priority = 1.0 + min(2.0, gap / 250.0)
        examples.append(ReplayExample(
            tr.fen,
            tr.policy_move_uci,
            target,
            1.0,
            priority,
            played_move_uci=tr.played_move_uci,
            search_score_cp=tr.played_score_cp,
            best_score_cp=tr.best_score_cp,
        ))

    pgn = build_pgn(
        moves,
        result_str,
        white_name,
        black_name,
        termination,
        initial_board=initial_board,
        opening_name=opening_name,
    )
    return GameRecord(
        result=result_str,
        winner=forced_winner,
        termination=termination,
        plies=len(traces),
        pgn=pgn,
        examples=examples,
        metadata={
            "stochastic": stochastic,
            "depth": depth,
            "trace_count": len(traces),
            "opening_name": opening_name,
            "opening_family": opening_family,
            "start_fen": initial_board.fen(),
            "seeded_start": initial_board.fen() != chess.Board().fen(),
        },
    )
