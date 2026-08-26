# dog_matist 2.0

`dog_matist` is the 2.0 continuation of the DarwinChess lineage. The name changes; the learned lifetime does not.

## State compatibility

2.0 intentionally keeps using `~/.darwinchess` so existing champion checkpoints, SQLite history, replay experience, metrics, insights, and exports remain available. The package also keeps a `darwinchess` CLI alias and the `DarwinChessAgent` class as compatibility shims while new code may use `dog-matist` and `DogMatistAgent`.

Before an upgrade, `setup_mac.sh` refuses to proceed if a legacy or v2 Evolution process is active. It then creates a consistent SQLite snapshot under:

```text
~/.darwinchess/backups/pre_dog_matist_2_<timestamp>.sqlite3
```

Checkpoint files are not duplicated because they can be large and are not rewritten by the installer.

## Opening diversity

Self-play no longer relies only on temperature from the normal initial board. Its curriculum mixes:

- 35% free play from the standard initial position
- 35% curated sound opening positions
- 20% uncommon but legal opening positions
- 10% controlled-random legal continuations from sound seeds

The seed is a starting position, not an opening book. All moves after the seed are chosen by dog_matist's own search/policy.

Arena uses paired openings: Candidate vs Champion plays the same opening with colors swapped before moving to the next opening. This reduces color/opening bias and makes specialization in one repeated line less useful.

## Human play while Evolution runs

Human Play and Evolution are designed to coexist.

At the beginning of every human game, Studio pins a snapshot of the current champion generation. If Evolution promotes a successor during that game, the current game keeps the pinned model. The new champion is used on the next game.

Evolution has a cross-process single-writer lock. A second Evolution/Challenge writer is refused, while Play/status/conversation may continue.

Completed Studio games are stored both as PGN files and as `source=human` lifetime-memory records. They deliberately add **zero replay examples**, so playing a human does not silently teach the network to imitate the human. `Abort without saving` writes neither PGN nor lifetime memory.

## Studio 2.0

Studio includes:

- graphical click-to-move chessboard and legal-move hints
- last-move and check highlighting
- board flip, full-turn undo, resign, abort without saving
- local move/capture/check/game-over sounds with mute control
- explicit Self-play / Training / Arena / Promote-or-Reject runtime stages
- current-run progress and elapsed time
- button hover, pressed, and disabled feedback
- lightweight vector dog_matist mascot with a thinking animation during Evolution

## Release gate

Do not treat a branch build as a release only because it imports. The intended release gate is:

1. automated core tests pass;
2. Studio modules pass an offscreen import smoke test;
3. opening/concurrency/human-memory regression tests pass;
4. `setup_mac.sh` succeeds on the target Mac and `dog-matist doctor` sees the existing champion;
5. one short self-play/training/Arena smoke cycle completes without resetting the lineage;
6. one disposable human game confirms board input, sound, pinned generation, PGN saving, and abort behavior.

Only after those checks should 2.0 replace the working 1.x installation for normal use.
