# dog_matist 2.0

**Persistent, self-evolving, conversational chess agent for local machines.**

`dog_matist` is the 2.0 continuation of DarwinChess. The project name and Studio identity change, but the learned lifetime does not: 2.0 intentionally reuses the existing `~/.darwinchess` champion checkpoints, SQLite history, replay experience, metrics, and lineage.

> Current branch status: development candidate. Run the automated tests and the Mac upgrade smoke test before treating it as the replacement for a working 1.x installation.

## Core loop

```text
SELF-PLAY
   ↓
PERSISTENT REPLAY
   ↓
TRAIN CHALLENGER
   ↓
PAIRED-OPENING ARENA
   ↓
PROMOTE / REJECT
   ↓
repeat
```

A challenger always starts from the current champion. Training alone never replaces the champion. A candidate must pass the held-out Arena gate first.

## What changed in 2.0

### Opening diversity

Self-play no longer starts every game from the same initial board and relies only on temperature. The opening curriculum mixes:

- **35%** standard initial-position free exploration
- **35%** curated sound openings
- **20%** uncommon but legal openings
- **10%** controlled-random legal continuations from sound positions

The curriculum is **not an opening book**. It only chooses a starting position; dog_matist calculates every move from there.

Arena uses **paired openings**. Candidate and Champion play the same starting position twice with colors swapped. Odd Arena game requests are rounded up so a color pair is never left incomplete.

### Play while Evolution runs

Human Play and Evolution are designed to coexist. At the beginning of a game, Studio pins the exact current champion checkpoint and generation. If Evolution promotes a successor while you are still playing, your game keeps the original opponent. The next game receives the new champion.

Evolution has a cross-process single-writer lock: a second Evolution/Challenge writer is refused, while Play, status, and conversation can continue.

### Human game memory

Completed Studio games are saved as PGNs under:

```text
~/.darwinchess/studio_games/
```

They are also written to lifetime SQLite memory as `source=human`, with the pinned generation and takeback count. They deliberately add **zero replay examples**, so playing a human never silently teaches the network to imitate that human.

`Abort without saving` writes neither PGN nor lifetime memory.

### Studio 2.0

The desktop Studio includes:

- graphical click-to-move chessboard
- legal destination hints
- last-move and check highlighting
- Play White / Play Black and board flip
- full-turn undo, resign, and abort-without-saving
- move, capture, check, and game-over sounds with mute control
- explicit Self-play / Training / Arena / Promote-or-Reject status
- current-run progress and elapsed time
- hover / pressed / disabled button feedback
- lightweight vector dog_matist mascot with thinking animation during Evolution

## Existing 1.x state is preserved

Default state remains:

```text
~/.darwinchess/
```

The first 2.0 setup makes a consistent SQLite backup at:

```text
~/.darwinchess/backups/pre_dog_matist_2_<timestamp>.sqlite3
```

It does not duplicate the potentially large checkpoint directory. The installer refuses to upgrade while an old DarwinChess or new dog_matist Evolution process is active; stop Evolution safely first.

The old `darwinchess` CLI command and `DarwinChessAgent` Python class remain compatibility aliases during migration. New integrations may use `dog-matist` and `DogMatistAgent`.

## macOS setup

From Terminal inside the 2.0 project folder:

```bash
./setup_mac.sh
```

The script:

1. checks Python 3.11+;
2. refuses to modify the installation while Evolution is active;
3. backs up the existing lifetime SQLite database if present;
4. creates/reuses `.venv`;
5. installs core + Studio dependencies;
6. runs the full test suite;
7. imports the Studio modules headlessly as a smoke test;
8. runs `dog-matist --mode normal doctor` against the real state.

Then launch Studio:

```bash
./run_studio.command
```

Or run one normal Evolution cycle:

```bash
./run_normal.command 1
```

For a plugged-in overnight run:

```bash
./run_night.command 8
```

The night launcher uses macOS `caffeinate -i`, logs under `~/.darwinchess/logs/`, and still respects the same single-Evolution-writer rule.

## CLI

```bash
dog-matist status
dog-matist doctor
dog-matist selfplay --games 10
dog-matist --mode normal evolve --cycles 1
dog-matist --mode night evolve --hours 8
dog-matist challenge --steps 120 --games 10
dog-matist play --color white
dog-matist chat
dog-matist export
```

Terminal Play also pins its champion at game start. It supports `undo`, `resign`, and `quit`; quitting/interrupting aborts without writing a completed game.

## Architecture

The learned engine remains under the internal `darwinchess` Python package for backwards compatibility. Major components are:

- `darwinchess/network.py` — residual CNN value/policy network
- `darwinchess/evaluator.py` — classical + neural hybrid evaluation
- `darwinchess/search.py` — iterative-deepening alpha-beta / quiescence search
- `darwinchess/memory.py` — durable SQLite lifetime/replay store
- `darwinchess/opening_curriculum.py` — diversified starting positions
- `darwinchess/selfplay.py` — self-play and replay targets
- `darwinchess/trainer.py` — continual AdamW training
- `darwinchess/arena.py` — paired-opening promotion gate
- `darwinchess/locks.py` — single-writer Evolution protection
- `darwinchess/runtime.py` — persistent lifecycle/evolution orchestration
- `darwinchess/api.py` — stable agent boundary and pinned human games
- `studio/` — dog_matist Studio 2.0

See [`DOG_MATIST_2.md`](DOG_MATIST_2.md) for migration/safety notes and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the original research architecture background.

## Scientific caveat

`dog_matist` can continue updating from its accumulated experience without restarting training from zero, but no finite architecture guarantees unlimited improvement. The useful research question is whether persistent replay, continual updates, diverse self-play, and held-out promotion tests continue producing measurable gains under a fixed local compute budget. The project keeps the lineage and metrics needed to test that rather than assuming the answer.
