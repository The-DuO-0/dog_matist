from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import chess
import chess.engine

from .memory import MemoryStore, ReplayExample


@dataclass
class TeacherStatus:
    available: bool
    path: str | None
    reason: str


def find_stockfish(config: dict[str, Any]) -> TeacherStatus:
    configured = config.get("teacher", {}).get("stockfish_path")
    candidates = [
        configured,
        shutil.which("stockfish"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return TeacherStatus(True, str(Path(candidate).expanduser()), "found")
    return TeacherStatus(False, None, "Stockfish not configured/found")


class StockfishTeacher:
    """Optional external teacher. It is not required for self-evolution."""

    def __init__(self, path: str, depth: int = 14):
        self.path = path
        self.depth = depth

    def annotate(self, fens: list[str]) -> list[ReplayExample]:
        examples: list[ReplayExample] = []
        engine = chess.engine.SimpleEngine.popen_uci(self.path)
        try:
            for fen in fens:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                pv = info.get("pv", [])
                if not pv:
                    continue
                score = info["score"].pov(board.turn).score(mate_score=100000) or 0
                # Soft value target: tanh-like centipawn compression without numpy dependency.
                value = max(-1.0, min(1.0, float(score) / 1200.0))
                examples.append(ReplayExample(fen, pv[0].uci(), value, policy_weight=2.0, priority=2.0))
        finally:
            engine.quit()
        return examples
