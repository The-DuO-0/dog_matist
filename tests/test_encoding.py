import chess
import torch

from darwinchess.encoding import BOARD_PLANES, encode_board, legal_move_log_scores, move_targets


def test_initial_board_encoding():
    board = chess.Board()
    x = encode_board(board)
    assert x.shape == (BOARD_PLANES, 8, 8)
    assert x[:12].sum() == 32
    assert x[12].sum() == 64  # white to move


def test_promotion_target_is_distinct():
    move = chess.Move.from_uci("e7e8q")
    frm, to, promo = move_targets(move)
    assert frm == chess.E7
    assert to == chess.E8
    assert promo == 1


def test_policy_scores_only_legal_moves():
    board = chess.Board()
    logits64 = torch.zeros(64)
    promo = torch.zeros(5)
    scores = legal_move_log_scores(board, logits64, logits64, promo)
    assert len(scores) == board.legal_moves.count()
    assert chess.Move.from_uci("e2e4") in scores
