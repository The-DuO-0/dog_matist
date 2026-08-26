from pathlib import Path

import pytest

from darwinchess.api import DogMatistAgent
from darwinchess.locks import EvolutionAlreadyRunning, EvolutionLock
from darwinchess.runtime import DarwinRuntime


def _config(tmp_path: Path) -> Path:
    cfg = tmp_path / "test.yaml"
    cfg.write_text(
        f"""
project:
  state_dir: {tmp_path / 'state'}
model:
  channels: 16
  residual_blocks: 1
resources:
  default_mode: eco
""",
        encoding="utf-8",
    )
    return cfg


def test_second_runtime_does_not_abort_active_challenger(tmp_path):
    cfg = _config(tmp_path)
    with DarwinRuntime(cfg, mode="eco", device="cpu", search_device="cpu") as rt:
        champion = rt.champion_info()
        rt.memory.add_generation(
            1,
            0,
            champion["checkpoint_path"],
            "challenger",
            notes="simulated in-flight challenger",
        )

    # Opening a separate Play/status runtime used to mark every challenger
    # aborted. Studio 2.0 must leave the evolution-owned row untouched.
    with DarwinRuntime(cfg, mode="eco", device="cpu", search_device="cpu") as rt:
        assert rt.memory.get_generation(1)["status"] == "challenger"


def test_only_one_evolution_writer_can_hold_lock(tmp_path):
    with EvolutionLock(tmp_path):
        with pytest.raises(EvolutionAlreadyRunning):
            with EvolutionLock(tmp_path):
                pass


def test_human_game_keeps_pinned_generation_after_promotion(tmp_path):
    cfg = _config(tmp_path)
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    with DogMatistAgent(str(cfg), mode="eco", device="cpu", search_device="cpu") as agent:
        pinned = agent.begin_game()
        assert pinned["generation"] == 0

        champion = agent.runtime.champion_info()
        agent.runtime.memory.add_generation(
            1,
            0,
            champion["checkpoint_path"],
            "challenger",
            notes="simulated promoted successor",
        )
        agent.runtime.memory.promote_generation(0, 1, champion["checkpoint_path"])

        during_game = agent.best_move(fen, depth=1)
        assert during_game["generation"] == 0

        agent.end_game()
        next_game = agent.best_move(fen, depth=1)
        assert next_game["generation"] == 1
