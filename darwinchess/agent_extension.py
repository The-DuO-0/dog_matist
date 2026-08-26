"""Minimal example: use DarwinChess as a persistent skill inside another agent."""

import chess
from darwinchess.api import DarwinChessAgent

with DarwinChessAgent(mode="normal") as chess_skill:
    board = chess.Board()
    move = chess_skill.best_move(board.fen(), depth=2)
    print(move["move_san"], move["explanation"])
    print(chess_skill.talk("你最近学到了什么？"))
