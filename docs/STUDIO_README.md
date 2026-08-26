# dog_matist Studio 2.0

Desktop UI for the persistent dog_matist chess agent. Studio uses the same learned lifetime state at `~/.darwinchess`; it does not reset or copy the champion lineage.

## Install / upgrade on macOS

From the dog_matist 2.0 project folder:

```bash
./setup_mac.sh
./run_studio.command
```

`setup_mac.sh` refuses to upgrade while Evolution is active, backs up the existing SQLite lifetime database, installs core + Studio dependencies, runs tests, performs a headless Studio import check, and runs the hardware/state doctor.

## Play

The Play page uses the real champion. A new game first **pins one exact champion generation/checkpoint**. Background Evolution can continue; a promotion only changes the opponent for the next game.

The graphical board provides legal-move highlighting, last-move/check highlighting, board flip, full-turn undo, resign, abort-without-saving, and local move/capture/check/end sounds.

Completed games are saved to:

```text
~/.darwinchess/studio_games/
```

and to lifetime SQLite memory as `source=human`. Human games add **zero training replay examples**. Abort writes neither PGN nor a completed-game record.

## Evolution

Evolution shows explicit runtime stages:

```text
SELF-PLAY → TRAINING → ARENA → PROMOTE / REJECT
```

The Current Run card reports progress and elapsed time. Self-play uses the opening curriculum; Arena evaluates color-swapped pairs from the same opening position.

Only one Evolution/Challenge/replay-writing process may own the single-writer lock. Play, status, and conversation remain available while Evolution runs. Interactive Studio search keeps normal process scheduling priority; heavy background Evolution yields CPU priority according to the selected resource profile.

`Stop safely` sends the worker an interrupt so already committed lifetime state and the active champion are not replaced by an unproven partial candidate.

## Research

Research reads the same SQLite state and export directory. Arena games remain held out from training replay. Human games are visible as lifetime encounters but also remain outside replay unless a future feature explicitly opts them in.

## Conversation

Conversation uses the same embedded `DogMatistAgent` and can inspect the agent's real persistent status. The legacy `DarwinChessAgent` class remains an API compatibility alias during the 1.x → 2.0 migration.

## Safety rule during migration

Do not run the old DarwinChess 1.x Evolution loop and dog_matist 2.0 Evolution at the same time. Stop the old evolution process before running `setup_mac.sh`. The 2.0 installer checks for this automatically.
