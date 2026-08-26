from __future__ import annotations

import chess
import numpy as np
import torch

# 12 piece planes + side to move + 4 castling planes + en-passant plane.
BOARD_PLANES = 18
PROMO_TO_INDEX = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
INDEX_TO_PROMO = {v: k for k, v in PROMO_TO_INDEX.items()}


def encode_board(board: chess.Board) -> np.ndarray:
    x = np.zeros((BOARD_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        x[plane, rank, file] = 1.0

    if board.turn == chess.WHITE:
        x[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        x[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[16, :, :] = 1.0
    if board.ep_square is not None:
        r = chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        x[17, r, f] = 1.0
    return x


def encode_boards(boards: list[chess.Board], device: torch.device | None = None) -> torch.Tensor:
    arr = np.stack([encode_board(b) for b in boards])
    return torch.from_numpy(arr).to(device=device)


def move_targets(move: chess.Move) -> tuple[int, int, int]:
    return move.from_square, move.to_square, PROMO_TO_INDEX.get(move.promotion, 0)


def legal_move_log_scores(
    board: chess.Board,
    from_logits: torch.Tensor,
    to_logits: torch.Tensor,
    promo_logits: torch.Tensor,
) -> dict[chess.Move, float]:
    fp = torch.log_softmax(from_logits.detach().float().cpu(), dim=-1)
    tp = torch.log_softmax(to_logits.detach().float().cpu(), dim=-1)
    pp = torch.log_softmax(promo_logits.detach().float().cpu(), dim=-1)
    scores: dict[chess.Move, float] = {}
    for move in board.legal_moves:
        promo_idx = PROMO_TO_INDEX.get(move.promotion, 0)
        scores[move] = float(fp[move.from_square] + tp[move.to_square] + pp[promo_idx])
    return scores
