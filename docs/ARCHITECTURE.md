# DarwinChess Architecture

## 1. Separation of concerns

DarwinChess deliberately separates **competence**, **learning**, **selection**, **memory**, and **language**.

- `search.py`: tactical/strategic decision procedure (iterative alpha-beta + quiescence).
- `evaluator.py`: transparent classical ancestor + learned neural value.
- `network.py`: compact residual policy/value model.
- `trainer.py`: continual replay training; optimizer state is inherited when compatible.
- `genome.py`: non-weight behavior parameters that are inherited/mutated and Arena-tested.
- `selfplay.py`: exploration and durable experience generation.
- `arena.py`: held-out champion/challenger selection.
- `memory.py`: SQLite WAL lifetime database.
- `reflection.py`: durable summaries from real game history.
- `dialogue.py`: chess-native language layer; optional Ollama generation.
- `teacher.py`: optional UCI/Stockfish distillation.
- `runtime.py`: lifecycle orchestration.

## 2. What a generation contains

A generation is not just a `.pt` file. It is:

```text
generation
├── neural weights
├── optimizer state (when compatible)
├── genome
│   ├── classical_mix
│   └── neural_cp_scale
├── parent generation
├── training loss
├── arena record
└── status: champion / challenger / rejected / retired
```

Generation 0 starts with `classical_mix=1.0`. Neural policy/value heads are zeroed. It is therefore a deterministic ancestral baseline rather than a random neural player.

A child inherits the champion weights and optimizer state, trains on the persistent replay memory, and proposes a genome mutation. By default the mutation reduces `classical_mix` by 0.05, down to a configured floor. This mutation only becomes active if the child passes the same held-out Arena as its weight changes.

This prevents an important failure mode: silently trusting the learned model more just because more data exists. Trust itself must survive selection.

## 3. Experience semantics

A self-play replay row stores both:

- the **played move** (which may be stochastic during exploration);
- the **search-best move** (used as the policy target);
- search score of the played move;
- search score of the best move;
- final side-to-move game result as the value target.

This means later research can distinguish deliberate exploration from search disagreement instead of losing that information.

Human games are remembered but are not automatically added to training replay. This avoids blindly imitating a human opponent. Arena games are also held out from replay so promotion is not evaluated on its training set.

## 4. Crash/data-safety properties

- SQLite runs in WAL mode.
- Each completed game is committed immediately.
- The active champion is never overwritten by an untested challenger.
- Challenger promotion is an atomic SQLite champion switch after the candidate checkpoint is durable.
- Missing champion checkpoints cause a hard error rather than silently resetting lifetime history.
- Replay can be pruned while PGNs/generation history remain durable.

## 5. Compute model on Apple Silicon

Search and training can use different devices. Default profiles keep the small, branch-heavy single-position search network on CPU while allowing batch training to use PyTorch MPS when available. This avoids forcing every alpha-beta leaf through a GPU synchronization boundary.

All resource modes share the same database, champion, network and lineage. `eco`, `normal`, and `night` only alter compute budget.

## 6. Why this can improve without full retraining

A promoted generation becomes the starting point of the next challenger. Weights are not reset. Compatible optimizer state is also inherited. The replay database survives every process restart.

That is **continual training**, not a promise of mathematically unbounded improvement. A finite architecture/search budget can plateau. DarwinChess is designed so plateaus are measurable and future architecture changes can reuse the same games, labels and lineage rather than discarding them.
