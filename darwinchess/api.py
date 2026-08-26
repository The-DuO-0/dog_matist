from __future__ import annotations

from typing import Any

import chess

from .dialogue import DialogueAgent, explain_search
from .network import load_checkpoint
from .runtime import DarwinRuntime


class DarwinChessAgent:
    """Stable programmatic boundary for embedding dog_matist in a larger agent.

    The public class name stays compatible with DarwinChess 1.x. Embedded
    agents are treated as interactive, so they keep normal process scheduling
    priority while separate Evolution worker processes yield according to their
    resource profile.
    """

    def __init__(
        self,
        config_path: str | None = None,
        *,
        mode: str = "normal",
        device: str | None = None,
        search_device: str | None = None,
    ):
        self.runtime = DarwinRuntime(
            config_path,
            mode=mode,
            device=device,
            search_device=search_device,
            apply_nice=False,
        )
        self.dialogue = DialogueAgent(self.runtime)
        self._game_searcher = None
        self._game_generation: int | None = None

    def close(self) -> None:
        self.end_game()
        self.runtime.close()

    def __enter__(self) -> "DarwinChessAgent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def status(self) -> dict[str, Any]:
        status = self.runtime.status()
        status["pinned_game_generation"] = self._game_generation
        return status

    def begin_game(self) -> dict[str, Any]:
        """Pin one generation/checkpoint pair for the complete human game."""
        champion = self.runtime.champion_info()
        generation = int(champion["id"])
        checkpoint = str(champion["checkpoint_path"])
        model, payload = load_checkpoint(checkpoint, self.runtime.search_device)
        genome = self.runtime.genome_from_payload(payload)
        self._game_searcher = self.runtime.make_searcher(
            model,
            genome=genome,
            device=self.runtime.search_device,
        )
        self._game_generation = generation
        return {"generation": generation, "checkpoint": checkpoint}

    def end_game(self) -> None:
        self._game_searcher = None
        self._game_generation = None

    def best_move(self, fen: str, *, depth: int | None = None) -> dict[str, Any]:
        board = chess.Board(fen)
        if self._game_searcher is not None:
            result = self._game_searcher.search(board, depth=depth, top_n=5)
            generation = self._game_generation
        else:
            result = self.runtime.analyze(fen, depth=depth, top_n=5)
            generation = int(self.runtime.champion_info()["id"])
        if result.move is None:
            return {
                "move_uci": None,
                "move_san": None,
                "score_cp": result.score_cp,
                "terminal": True,
                "generation": generation,
                "explanation": explain_search(board, result),
            }
        return {
            "move_uci": result.move.uci(),
            "move_san": board.san(result.move),
            "score_cp": result.score_cp,
            "depth": result.depth,
            "nodes": result.nodes,
            "pv_uci": [m.uci() for m in result.pv],
            "candidates": [
                {"move_uci": c.move.uci(), "move_san": board.san(c.move), "score_cp": c.score_cp}
                for c in result.candidates
            ],
            "terminal": False,
            "generation": generation,
            "explanation": explain_search(board, result),
        }

    def record_human_game(
        self,
        *,
        pgn: str,
        result: str,
        termination: str,
        plies: int,
        human_color: str,
        takebacks: int = 0,
        generation: int | None = None,
    ) -> dict[str, Any]:
        if human_color not in {"white", "black"}:
            raise ValueError("human_color must be 'white' or 'black'")
        if generation is None:
            generation = self._game_generation
        if generation is None:
            generation = int(self.runtime.champion_info()["id"])
        dog = f"dog_matist-g{generation}"
        white_agent = "Human" if human_color == "white" else dog
        black_agent = dog if human_color == "white" else "Human"
        gid = self.runtime.memory.add_game(
            source="human",
            generation=int(generation),
            white_agent=white_agent,
            black_agent=black_agent,
            result=result,
            termination=termination,
            pgn=pgn,
            plies=int(plies),
            examples=[],
            metadata={
                "human_color": human_color,
                "takebacks": int(takebacks),
                "training_replay": False,
                "studio": True,
            },
        )
        return {"game_id": gid, "generation": int(generation), "training_replay": False}

    def talk(self, message: str) -> str:
        return self.dialogue.answer(message)

    def chat(self, message: str) -> str:
        return self.talk(message)

    def evolve_once(self) -> dict[str, Any]:
        return self.runtime.evolve_cycle()

    def remember_status(self) -> dict[str, Any]:
        return self.status()


DogMatistAgent = DarwinChessAgent
