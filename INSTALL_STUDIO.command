#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "dog_matist Studio 2.0 installer"
echo "This installer preserves the existing ~/.darwinchess champion/checkpoint/database state."
echo

candidates=()
while IFS= read -r -d '' f; do
  d="$(dirname "$f")"
  if [ -f "$d/pyproject.toml" ] && grep -Eqi 'dog-matist|darwinchess' "$d/pyproject.toml" 2>/dev/null; then
    candidates+=("$d")
  fi
done < <(find "$HOME/Downloads" "$HOME/Desktop" "$HOME/Documents" -maxdepth 4 -type f -name 'setup_mac.sh' -print0 2>/dev/null || true)

if [ -f "$HERE/setup_mac.sh" ] && [ -f "$HERE/pyproject.toml" ]; then
  target="$HERE"
elif [ ${#candidates[@]} -eq 1 ]; then
  target="${candidates[0]}"
elif [ ${#candidates[@]} -gt 1 ]; then
  echo "I found several dog_matist / DarwinChess project folders:"
  for i in "${!candidates[@]}"; do echo "$((i+1))) ${candidates[$i]}"; done
  echo
  read -r -p "Choose a number: " pick
  idx=$((pick-1))
  target="${candidates[$idx]}"
else
  echo "I couldn't automatically find the project folder."
  echo "Drag the dog_matist project folder into this Terminal window, then press Return:"
  read -r target
  target="${target%/}"
  target="${target#\'}"; target="${target%\'}"
  target="${target#\"}"; target="${target%\"}"
fi

if [ ! -f "$target/setup_mac.sh" ] || [ ! -f "$target/pyproject.toml" ]; then
  echo "That does not look like the dog_matist project folder: $target"
  exit 1
fi

cd "$target"
chmod +x setup_mac.sh run_studio.command run_normal.command run_night.command setup_studio_deps.command 2>/dev/null || true
./setup_mac.sh

echo
echo "Installation finished. Your old champion lineage was not reset."
echo "Open Studio with: $target/run_studio.command"
