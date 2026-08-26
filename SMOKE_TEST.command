#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Missing .venv. Run ./setup_mac.sh first."
  exit 1
fi
source .venv/bin/activate

TMP="$(mktemp -d "${TMPDIR:-/tmp}/dog_matist_smoke.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
STATE="$TMP/state"
CFG="$TMP/smoke.yaml"

cat > "$CFG" <<YAML
project:
  state_dir: $STATE
  seed: 4242
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
YAML

echo "dog_matist 2.0 isolated smoke test"
echo "Temporary state: $STATE"
echo "Your real ~/.darwinchess state will not be used by this evolution cycle."
echo

DARWINCHESS_HOME="$STATE" dog-matist \
  --config "$CFG" --mode eco --device cpu --search-device cpu \
  evolve --cycles 1

echo
DARWINCHESS_HOME="$STATE" python - "$CFG" <<'PY'
import sys
from pathlib import Path
from darwinchess.api import DogMatistAgent

cfg = sys.argv[1]
with DogMatistAgent(cfg, mode="eco", device="cpu", search_device="cpu") as agent:
    status = agent.status()
    assert status["games"] >= 3, status
    assert status["replay_examples"] > 0, status
    pinned = agent.begin_game()
    assert pinned["generation"] == status["champion_generation"]
    mate = agent.best_move("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", depth=1)
    assert mate["move_uci"] is not None
    assert mate["generation"] == pinned["generation"]
    agent.end_game()
    print("Temporary evolution + pinned-play API: OK")
PY

QT_QPA_PLATFORM=offscreen python - <<'PY'
import studio.app
import studio.backend
import studio.pages.play
import studio.pages.evolution
print("Studio imports: OK")
PY

echo
echo "SMOKE TEST PASSED"
echo "Temporary state is being deleted; real lifetime state was not modified by this test."
