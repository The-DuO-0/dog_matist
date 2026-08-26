from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any
import math

import chess
import chess.polyglot

from dogmatist_v2.opening_search_revision import (
    OpeningSearchEvidence,
    OpeningSearchR2Policy,
    OpeningSearchR2Session,
    absolute_game_ply,
    candidate_scores,
    select_verification_candidates,
)

from .evaluator import HybridEvaluator, PIECE_VALUES

MATE_SCORE = 100000.0
INF = 10**9


@dataclass
class Candidate:
    move: chess.Move
    score_cp: float
    pv: list[chess.Move] = field(default_factory=list)


@dataclass
class SearchResult:
    move: chess.Move | None
    score_cp: float
    depth: int
    nodes: int
    pv: list[chess.Move]
    candidates: list[Candidate]
    elapsed_s: float
    engine_revision: str = "search-r1"
    opening_stabilized: bool = False
    opening_stabilization_reason: str | None = None
    opening_extra_nodes: int = 0
    opening_verified_moves: int = 0


@dataclass
class TTEntry:
    depth: int
    score: float
    flag: str
    move: chess.Move | None


class SearchTimeout(Exception):
    pass


class AlphaBetaSearcher:
    def __init__(self, evaluator: HybridEvaluator, config: dict[str, Any]):
        self.evaluator = evaluator
        self.config = config
        self.qdepth = int(config["search"].get("quiescence_depth", 3))
        self.tt_limit = int(config["search"].get("transposition_size", 200000))
        self.tt: dict[int, TTEntry] = {}
        self.nodes = 0
        self.deadline: float | None = None

        rcfg = config.get("search", {}).get("opening_stabilization", {}) or {}
        self.opening_stabilization_enabled = bool(rcfg.get("enabled", False))
        self.opening_revision_id = str(rcfg.get("revision_id", "search-r2c-selective-root"))
        self.opening_policy = OpeningSearchR2Policy(
            opening_plies=int(rcfg.get("opening_plies", 8)),
            always_verify_plies=int(rcfg.get("always_verify_plies", 0)),
            extra_depth=int(rcfg.get("extra_depth", 1)),
            candidate_margin_cp=float(rcfg.get("candidate_margin_cp", 18.0)),
            iteration_swing_cp=float(rcfg.get("iteration_swing_cp", 60.0)),
            move_flip_min_swing_cp=float(rcfg.get("move_flip_min_swing_cp", 45.0)),
            max_extra_searches=int(rcfg.get("max_extra_searches", 3)),
        )
        self.opening_session = OpeningSearchR2Session(self.opening_policy)
        self.opening_verify_min_candidates = int(rcfg.get("verification_min_candidates", 4))
        self.opening_verify_max_candidates = int(rcfg.get("verification_max_candidates", 8))
        self.opening_verify_score_window_cp = float(rcfg.get("verification_score_window_cp", 90.0))

    def _key(self, board: chess.Board) -> int:
        return chess.polyglot.zobrist_hash(board)

    def _check_time(self) -> None:
        if self.deadline is not None and monotonic() >= self.deadline:
            raise SearchTimeout

    def _terminal_score(self, board: chess.Board, ply: int) -> float | None:
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0.0
        return None

    def _move_order_score(self, board: chess.Board, move: chess.Move, tt_move: chess.Move | None, policy: dict[chess.Move, float] | None) -> float:
        score = 0.0
        if tt_move is not None and move == tt_move:
            score += 1_000_000
        if move.promotion:
            score += 30_000 + PIECE_VALUES.get(move.promotion, 0)
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_value = 100 if victim is None else PIECE_VALUES[victim.piece_type]
            attacker_value = 100 if attacker is None else PIECE_VALUES[attacker.piece_type]
            score += 20_000 + 10 * victim_value - attacker_value
        if board.gives_check(move):
            score += 8_000
        if policy is not None:
            score += 200.0 * policy.get(move, -20.0)
        return score

    def _ordered_moves(self, board: chess.Board, *, tt_move: chess.Move | None = None, policy: dict[chess.Move, float] | None = None, captures_only: bool = False) -> list[chess.Move]:
        moves = list(board.legal_moves)
        if captures_only:
            moves = [m for m in moves if board.is_capture(m) or m.promotion or board.gives_check(m)]
        moves.sort(key=lambda m: self._move_order_score(board, m, tt_move, policy), reverse=True)
        return moves

    def _qsearch(self, board: chess.Board, alpha: float, beta: float, depth: int, ply: int) -> float:
        self.nodes += 1
        if (self.nodes & 255) == 0:
            self._check_time()
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal

        in_check = board.is_check()
        if not in_check:
            stand_pat = self.evaluator.evaluate(board)
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            if depth <= 0:
                return alpha
        elif depth <= 0:
            return self.evaluator.evaluate(board)

        for move in self._ordered_moves(board, captures_only=not in_check):
            board.push(move)
            score = -self._qsearch(board, -beta, -alpha, depth - 1, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> tuple[float, list[chess.Move]]:
        self.nodes += 1
        if (self.nodes & 255) == 0:
            self._check_time()
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal, []
        if depth <= 0:
            return self._qsearch(board, alpha, beta, self.qdepth, ply), []

        key = self._key(board)
        original_alpha = alpha
        original_beta = beta
        entry = self.tt.get(key)
        tt_move = entry.move if entry else None
        if entry is not None and entry.depth >= depth:
            if entry.flag == "exact":
                return entry.score, [entry.move] if entry.move else []
            if entry.flag == "lower":
                alpha = max(alpha, entry.score)
            elif entry.flag == "upper":
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score, [entry.move] if entry.move else []

        best_score = -INF
        best_move: chess.Move | None = None
        best_pv: list[chess.Move] = []
        for move in self._ordered_moves(board, tt_move=tt_move):
            board.push(move)
            child_score, child_pv = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            score = -child_score
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + child_pv
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best_move is None:
            return self.evaluator.evaluate(board), []
        flag = "exact"
        if best_score <= original_alpha:
            flag = "upper"
        elif best_score >= original_beta:
            flag = "lower"
        if len(self.tt) >= self.tt_limit:
            self.tt.clear()
        self.tt[key] = TTEntry(depth, best_score, flag, best_move)
        return best_score, best_pv

    def _root(self, board: chess.Board, depth: int, top_n: int) -> tuple[Candidate | None, list[Candidate]]:
        policy = self.evaluator.policy_scores(board) if self.config["search"].get("policy_ordering", True) else None
        entry = self.tt.get(self._key(board))
        tt_move = entry.move if entry else None
        candidates: list[Candidate] = []
        for move in self._ordered_moves(board, tt_move=tt_move, policy=policy):
            self._check_time()
            board.push(move)
            score, child_pv = self._negamax(board, depth - 1, -INF, INF, 1)
            score = -score
            board.pop()
            candidates.append(Candidate(move=move, score_cp=score, pv=[move] + child_pv))
        candidates.sort(key=lambda c: c.score_cp, reverse=True)
        return (candidates[0] if candidates else None), candidates[:top_n]

    def _root_subset(self, board: chess.Board, depth: int, moves: list[chess.Move], top_n: int) -> tuple[Candidate | None, list[Candidate]]:
        candidates: list[Candidate] = []
        for move in moves:
            self._check_time()
            if move not in board.legal_moves:
                continue
            board.push(move)
            score, child_pv = self._negamax(board, depth - 1, -INF, INF, 1)
            score = -score
            board.pop()
            candidates.append(Candidate(move=move, score_cp=score, pv=[move] + child_pv))
        candidates.sort(key=lambda c: c.score_cp, reverse=True)
        return (candidates[0] if candidates else None), candidates[:top_n]

    def _opening_revision_decision(
        self,
        board: chess.Board,
        *,
        base_depth: int,
        best: Candidate,
        candidates: list[Candidate],
        previous_best: Candidate | None,
    ):
        ply = absolute_game_ply(
            fullmove_number=int(board.fullmove_number),
            white_to_move=bool(board.turn == chess.WHITE),
        )
        evidence = OpeningSearchEvidence(
            ply=ply,
            base_depth=base_depth,
            best_move=best.move.uci(),
            best_score_cp=float(best.score_cp),
            candidates=candidate_scores((c.move.uci(), c.score_cp) for c in candidates),
            previous_iteration_move=previous_best.move.uci() if previous_best is not None else None,
            previous_iteration_score_cp=float(previous_best.score_cp) if previous_best is not None else None,
        )
        return self.opening_session.decide(evidence)

    def _verification_moves(
        self,
        board: chess.Board,
        candidates: list[Candidate],
        previous_best: Candidate | None,
    ) -> list[chess.Move]:
        selected = select_verification_candidates(
            candidate_scores((c.move.uci(), c.score_cp) for c in candidates),
            previous_iteration_move=previous_best.move.uci() if previous_best is not None else None,
            min_candidates=self.opening_verify_min_candidates,
            max_candidates=self.opening_verify_max_candidates,
            score_window_cp=self.opening_verify_score_window_cp,
        )
        legal = set(board.legal_moves)
        moves: list[chess.Move] = []
        for uci in selected:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move in legal and move not in moves:
                moves.append(move)
        return moves

    def search(self, board: chess.Board, depth: int | None = None, *, time_limit_s: float | None = None, top_n: int = 5) -> SearchResult:
        start = monotonic()
        self.nodes = 0
        self.deadline = None if time_limit_s is None else start + max(0.01, time_limit_s)
        max_depth = int(depth or self.config["search"].get("depth", 2))
        internal_top_n = max(
            int(top_n),
            self.opening_verify_max_candidates if self.opening_stabilization_enabled else int(top_n),
        )
        best: Candidate | None = None
        best_candidates: list[Candidate] = []
        completed_depth = 0
        previous_iteration_best: Candidate | None = None

        terminal = self._terminal_score(board, 0)
        if terminal is not None:
            return SearchResult(None, terminal, 0, 0, [], [], monotonic() - start)

        for d in range(1, max_depth + 1):
            try:
                root_best, candidates = self._root(board, d, internal_top_n)
            except SearchTimeout:
                break
            if root_best is not None:
                previous_iteration_best = best
                best = root_best
                best_candidates = candidates
                completed_depth = d

        opening_stabilized = False
        stabilization_reason: str | None = None
        opening_extra_nodes = 0
        opening_verified_moves = 0
        revision_id = "search-r1"

        if (
            self.opening_stabilization_enabled
            and best is not None
            and completed_depth == max_depth
        ):
            decision = self._opening_revision_decision(
                board,
                base_depth=max_depth,
                best=best,
                candidates=best_candidates,
                previous_best=previous_iteration_best,
            )
            stabilization_reason = decision.reason
            if decision.deepen:
                verification_moves = self._verification_moves(board, best_candidates, previous_iteration_best)
                opening_verified_moves = len(verification_moves)
                nodes_before = self.nodes
                try:
                    deeper_best, deeper_candidates = self._root_subset(
                        board,
                        decision.target_depth,
                        verification_moves,
                        internal_top_n,
                    )
                except SearchTimeout:
                    deeper_best = None
                    deeper_candidates = []
                    stabilization_reason = decision.reason + "; selective verification timed out"
                opening_extra_nodes = max(0, self.nodes - nodes_before)
                if deeper_best is not None:
                    best = deeper_best
                    best_candidates = deeper_candidates
                    completed_depth = decision.target_depth
                    opening_stabilized = True
                    revision_id = self.opening_revision_id

        if best is None:
            legal = next(iter(board.legal_moves), None)
            best = Candidate(legal, self.evaluator.evaluate(board), [legal] if legal else [])
        return SearchResult(
            move=best.move,
            score_cp=best.score_cp,
            depth=completed_depth,
            nodes=self.nodes,
            pv=best.pv,
            candidates=best_candidates[:top_n],
            elapsed_s=monotonic() - start,
            engine_revision=revision_id,
            opening_stabilized=opening_stabilized,
            opening_stabilization_reason=stabilization_reason,
            opening_extra_nodes=opening_extra_nodes,
            opening_verified_moves=opening_verified_moves,
        )
