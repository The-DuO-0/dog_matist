from pathlib import Path

from darwinchess.runtime import DarwinRuntime


def test_runtime_initializes_persistent_champion(tmp_path):
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
    with DarwinRuntime(cfg, mode="eco", device="cpu") as rt:
        status = rt.status()
        assert status["champion_generation"] == 0
        assert Path(status["champion_checkpoint"]).exists()
