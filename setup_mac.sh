#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.11+ first (Homebrew or python.org), then rerun."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ required; found {sys.version.split()[0]}")
print("Python", sys.version.split()[0])
PY

if command -v pgrep >/dev/null 2>&1; then
  if pgrep -f 'darwinchess.*evolve|dog-matist.*evolve' >/dev/null 2>&1; then
    echo "An Evolution process is still running. Use Stop safely first, then rerun setup_mac.sh."
    exit 2
  fi
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

root = Path(os.environ.get("DARWINCHESS_HOME") or os.environ.get("DARWINCHESS_STATE_DIR") or "~/.darwinchess").expanduser()
lock = root / "evolution.lock"
if lock.exists():
    try:
        import fcntl
        fh = lock.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except BlockingIOError:
            fh.seek(0)
            owner = fh.read().strip() or "another process"
            raise SystemExit(f"Evolution is still running ({owner}). Use Stop safely, then run setup again.")
        finally:
            fh.close()
    except ImportError:
        pass
PY

# Preserve the existing lifetime database before the first v2 launch. SQLite's
# online backup API creates a consistent snapshot without duplicating the large
# checkpoint directory. v2 intentionally keeps using ~/.darwinchess.
"$PYTHON_BIN" - <<'PY'
from datetime import datetime
from pathlib import Path
import os
import sqlite3

root = Path(os.environ.get("DARWINCHESS_HOME") or os.environ.get("DARWINCHESS_STATE_DIR") or "~/.darwinchess").expanduser()
db = root / "darwinchess.sqlite3"
if db.exists():
    backups = root / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backups / f"pre_dog_matist_2_{stamp}.sqlite3"
    src = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=30)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    print(f"Lifetime DB backup: {target}")
else:
    print("No previous lifetime DB found; this looks like a fresh install.")
PY

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,studio]"
python -m pytest

QT_QPA_PLATFORM=offscreen python - <<'PY'
import studio.app
import studio.backend
import studio.pages.play
import studio.pages.evolution
print("Studio import smoke test: OK")
PY

chmod +x run_studio.command run_normal.command run_night.command setup_studio_deps.command SMOKE_TEST.command 2>/dev/null || true

echo
echo "dog_matist 2.0 installed. Running hardware/state doctor..."
dog-matist --mode normal doctor

echo
echo "Upgrade complete. Existing champion lineage remains in ~/.darwinchess."
echo "Before touching the real lineage with 2.0 Evolution, run the isolated test:"
echo "  ./SMOKE_TEST.command"
echo
echo "Start Studio with:"
echo "  ./run_studio.command"
echo
echo "Start an overnight evolution run with:"
echo "  ./run_night.command 8"
