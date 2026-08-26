from darwinchess.api import DogMatistAgent


def test_completed_human_game_is_memory_not_replay(tmp_path):
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
    with DogMatistAgent(str(cfg), mode="eco", device="cpu", search_device="cpu") as agent:
        pinned = agent.begin_game()
        before = agent.runtime.memory.count_examples()
        saved = agent.record_human_game(
            pgn='[Event "test"]\n[Result "1-0"]\n\n1. e4 e5 1-0',
            result="1-0",
            termination="human_resignation",
            plies=2,
            human_color="white",
            takebacks=1,
            generation=pinned["generation"],
        )
        row = agent.runtime.memory.conn.execute(
            "SELECT * FROM games WHERE id=?", (saved["game_id"],)
        ).fetchone()
        assert row["source"] == "human"
        assert row["generation"] == pinned["generation"]
        assert agent.runtime.memory.count_examples() == before
        assert saved["training_replay"] is False
