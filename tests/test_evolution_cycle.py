from darwinchess.api import DarwinChessAgent


def test_complete_evolution_cycle_and_outer_agent_api(tmp_path):
    cfg = tmp_path / "cycle.yaml"
    cfg.write_text(
        f"""
project:
  state_dir: {tmp_path / 'state'}
  seed: 7
model:
  channels: 8
  residual_blocks: 1
evolution:
  mix_mutation_min_examples: 1
training:
  batch_size: 4
  min_examples_before_training: 1
search:
  quiescence_depth: 1
  transposition_size: 1000
  max_game_plies: 4
selfplay:
  allow_resign_after_ply: 99
  temperature_plies: 2
arena:
  depth: 1
  max_game_plies: 4
resources:
  eco:
    search_device: cpu
    search_depth: 1
    selfplay_games_per_cycle: 1
    training_steps_per_cycle: 1
    arena_games: 2
    nice: 0
    torch_threads: 2
""",
        encoding="utf-8",
    )

    with DarwinChessAgent(str(cfg), mode="eco", device="cpu", search_device="cpu") as agent:
        result = agent.evolve_once()
        assert result["trained"] is True
        assert result["selfplay_games"] == 1
        assert result["challenger"] == 1
        assert result["arena"]["games"] == 2
        status = agent.status()
        assert status["games"] == 3  # 1 self-play + 2 held-out Arena games
        assert status["replay_examples"] > 0
        assert "champion" in agent.chat("status")

        mate = agent.best_move("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", depth=1)
        assert mate["move_uci"] is not None
        assert mate["move_san"].endswith("#")
