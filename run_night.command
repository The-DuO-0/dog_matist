#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
HOURS="${1:-8}"
BIN=".venv/bin/dog-matist"
if [ ! -x "$BIN" ]; then BIN=".venv/bin/darwinchess"; fi
if [ ! -x "$BIN" ]; then
  echo "dog_matist is not installed yet. Run ./setup_mac.sh first."
  exit 1
fi
mkdir -p "$HOME/.darwinchess/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$HOME/.darwinchess/logs/night_${STAMP}.log"
echo "dog_matist NIGHT mode for about ${HOURS} hour(s)."
echo "Log: $LOG"
echo "Play/status can run at the same time; a second Evolution writer is blocked."
echo "Ctrl-C stops safely at the runtime boundary."
caffeinate -i "$BIN" --mode night evolve --hours "$HOURS" 2>&1 | tee "$LOG"
