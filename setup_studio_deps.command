#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "dog_matist .venv was not found in this folder."
  echo "Run ./setup_mac.sh first."
  exit 1
fi

source .venv/bin/activate
python -m pip install --upgrade "PySide6>=6.7,<7" "matplotlib>=3.8"
QT_QPA_PLATFORM=offscreen python - <<'PY'
import chess
import PySide6
import matplotlib
from darwinchess.api import DogMatistAgent
import studio.app
print("dog_matist Studio dependencies OK")
PY
chmod +x run_studio.command setup_studio_deps.command

echo
echo "dog_matist Studio dependencies are ready."
echo "Double-click run_studio.command to launch it."
