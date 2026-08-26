import chess
import torch

from darwinchess.config import load_config
from darwinchess.evaluator import HybridEvaluator
from darwinchess.network import ChessNet
from darwinchess.search import AlphaBetaSearcher


def test_search_finds_mate_in_one():
    # White: Kg6,Qg7; Black: Kh8. Qh7# is available.
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    cfg = load_config()
    cfg["model"]["channels"] = 16
    cfg["model"]["residual_blocks"] = 1
    model = ChessNet(16, 1).eval()
    ev = HybridEvaluator(model, cfg, torch.device("cpu"))
    result = AlphaBetaSearcher(ev, cfg).search(board, depth=1)
    assert result.move is not None
    board.push(result.move)
    assert board.is_checkmate()
