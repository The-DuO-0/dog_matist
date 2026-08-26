#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
BIN=".venv/bin/dog-matist"
if [ ! -x "$BIN" ]; then BIN=".venv/bin/darwinchess"; fi
if [ ! -x "$BIN" ]; then
  echo "Run ./setup_mac.sh first."
  exit 1
fi
"$BIN" --mode normal evolve --cycles "${1:-1}"
