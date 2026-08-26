from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import chess
import torch

from .encoding import encode_board, legal_move_log_scores
from .network import ChessNet
from .genome import AgentGenome


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER = {chess.D4, chess.E4, chess.D5, chess.E5}


def _relative_rank(square: int, color: chess.Color) -> int:
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _center_distance(square: int) -> float:
    return abs(chess.square_file(square) - 3.5) + abs(chess.square_rank(square) - 3.5)


def _is_passed_pawn(board: chess.Board, square: int, color: chess.Color) -> bool:
    f = chess.square_file(square)
    r = chess.square_rank(square)
    enemy = board.pieces(chess.PAWN, not color)
    for ef in range(max(0, f - 1), min(7, f + 1) + 1):
        for er in range(8):
            esq = chess.square(ef, er)
            if esq not in enemy:
                continue
            if color == chess.WHITE and er > r:
                return False
            if color == chess.BLACK and er < r:
                return False
    return True


def _pawn_structure(board: chess.Board, color: chess.Color) -> int:
    pawns = list(board.pieces(chess.PAWN, color))
    files = [chess.square_file(s) for s in pawns]
    score = 0
    for f in range(8):
        n = files.count(f)
        if n > 1:
            score -= 14 * (n - 1)
    for sq in pawns:
        f = chess.square_file(sq)
        # Isolated pawn: no friendly pawn on neighboring file.
        if not any(abs(f - of) == 1 for of in files):
            score -= 10
        rr = _relative_rank(sq, color)
        if _is_passed_pawn(board, sq, color):
            score += [0, 5, 10, 18, 32, 55, 95, 0][rr]
    return score


def _king_shelter(board: chess.Board, color: chess.Color) -> int:
    king = board.king(color)
    if king is None:
        return 0
    kf = chess.square_file(king)
    kr = chess.square_rank(king)
    pawns = board.pieces(chess.PAWN, color)
    score = 0
    direction = 1 if color == chess.WHITE else -1
    for df in (-1, 0, 1):
        f = kf + df
        if not 0 <= f < 8:
            continue
        for step, bonus in ((1, 10), (2, 5)):
            r = kr + direction * step
            if 0 <= r < 8 and chess.square(f, r) in pawns:
                score += bonus
                break
    return score


def _piece_activity(board: chess.Board, square: int, piece: chess.Piece, endgame: bool) -> float:
    rr = _relative_rank(square, piece.color)
    cd = _center_distance(square)
    p = piece.piece_type
    if p == chess.PAWN:
        file = chess.square_file(square)
        return rr * 3.0 + (5.0 if file in (3, 4) else 0.0)
    if p == chess.KNIGHT:
        return 30.0 - 8.0 * cd + 1.0 * len(board.attacks(square))
    if p == chess.BISHOP:
        return 22.0 - 3.0 * cd + 1.2 * len(board.attacks(square))
    if p == chess.ROOK:
        return 0.6 * len(board.attacks(square)) + (12.0 if rr == 6 else 0.0)
    if p == chess.QUEEN:
        return 0.35 * len(board.attacks(square)) - (1.5 * cd if not endgame else 0.5 * cd)
    if p == chess.KING:
        if endgame:
            return 28.0 - 7.0 * cd
        # In middlegames central kings are vulnerable; castled files are rewarded.
        f = chess.square_file(square)
        home_rank = chess.square_rank(square) == (0 if piece.color == chess.WHITE else 7)
        return (18.0 if home_rank and f in (2, 6) else 0.0) + 4.0 * cd
    return 0.0


def classical_white_cp(board: chess.Board) -> float:
    """Transparent centipawn evaluator used as the stable ancestral baseline.

    It intentionally contains no opening book or Stockfish knowledge. The learned
    network can progressively replace it, but only through the Arena gate.
    """
    if board.is_checkmate():
        return -100000.0 if board.turn == chess.WHITE else 100000.0
    outcome = board.outcome(claim_draw=False)
    if outcome is not None and outcome.winner is None:
        return 0.0

    nonpawn_material = 0
    for color in (chess.WHITE, chess.BLACK):
        for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            nonpawn_material += PIECE_VALUES[p] * len(board.pieces(p, color))
    endgame = nonpawn_material <= 2600

    score = 0.0
    piece_map = board.piece_map()
    for color, sign in ((chess.WHITE, 1.0), (chess.BLACK, -1.0)):
        for piece_type, value in PIECE_VALUES.items():
            score += sign * value * len(board.pieces(piece_type, color))

        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 28
        score += sign * _pawn_structure(board, color)
        if not endgame:
            score += sign * _king_shelter(board, color)

        # Piece activity / piece-square features.
        for sq, piece in piece_map.items():
            if piece.color == color:
                score += sign * _piece_activity(board, sq, piece, endgame)

        # Rooks prefer open/semi-open files.
        own_pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        for rook_sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(rook_sq)
            own_on_file = any(chess.square_file(s) == f for s in own_pawns)
            enemy_on_file = any(chess.square_file(s) == f for s in enemy_pawns)
            if not own_on_file and not enemy_on_file:
                score += sign * 18
            elif not own_on_file:
                score += sign * 10

        # Mild development pressure in the opening.
        if nonpawn_material >= 5000:
            starts = (
                (chess.B1, chess.G1, chess.C1, chess.F1)
                if color == chess.WHITE
                else (chess.B8, chess.G8, chess.C8, chess.F8)
            )
            for sq in starts:
                piece = board.piece_at(sq)
                if piece is not None and piece.color == color and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                    score -= sign * 7

    # Tempo matters slightly, but not enough to override a real positional edge.
    score += 8.0 if board.turn == chess.WHITE else -8.0
    return score


@dataclass
class EvalBreakdown:
    cp_side_to_move: float
    classical_cp_side_to_move: float
    neural_value: float
    neural_cp_side_to_move: float


class HybridEvaluator:
    def __init__(self, model: ChessNet, config: dict[str, Any], device: torch.device, genome: AgentGenome | None = None):
        self.model = model
        self.config = config
        self.device = device
        genome = genome or AgentGenome(
            classical_mix=float(config.get("evolution", {}).get("initial_classical_mix", 1.0)),
            neural_cp_scale=float(config["model"].get("neural_cp_scale", 650.0)),
        )
        self.genome = genome
        self.classical_mix = float(genome.classical_mix)
        self.neural_cp_scale = float(genome.neural_cp_scale)
        self.uses_neural = self.classical_mix < 1.0 - 1e-9
        self.model.eval()

    @torch.inference_mode()
    def neural_outputs(self, board: chess.Board) -> dict[str, torch.Tensor]:
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        out = self.model(x)
        return {k: v[0] for k, v in out.items()}

    @torch.inference_mode()
    def evaluate(self, board: chess.Board) -> float:
        return self.breakdown(board).cp_side_to_move

    @torch.inference_mode()
    def breakdown(self, board: chess.Board) -> EvalBreakdown:
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            if outcome.winner is None:
                return EvalBreakdown(0.0, 0.0, 0.0, 0.0)
            stm_wins = outcome.winner == board.turn
            terminal = 100000.0 if stm_wins else -100000.0
            return EvalBreakdown(terminal, terminal, 1.0 if stm_wins else -1.0, terminal)
        white_cp = classical_white_cp(board)
        classical_stm = white_cp if board.turn == chess.WHITE else -white_cp
        # A pure-classical genome must not spend a forward pass to multiply the
        # neural result by zero. This matters enormously for generation 0
        # self-play and keeps the first overnight run productive without
        # changing any capability or future training path.
        if not self.uses_neural:
            return EvalBreakdown(float(classical_stm), float(classical_stm), 0.0, 0.0)
        out = self.neural_outputs(board)
        neural_value = float(out["value"].detach().float().cpu())
        neural_cp = neural_value * self.neural_cp_scale
        mixed = self.classical_mix * classical_stm + (1.0 - self.classical_mix) * neural_cp
        return EvalBreakdown(float(mixed), float(classical_stm), neural_value, float(neural_cp))

    @torch.inference_mode()
    def policy_scores(self, board: chess.Board) -> dict[chess.Move, float]:
        if not self.uses_neural:
            return {}
        out = self.neural_outputs(board)
        return legal_move_log_scores(board, out["from_logits"], out["to_logits"], out["promo_logits"])
