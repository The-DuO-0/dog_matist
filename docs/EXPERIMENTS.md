# Research Experiments

DarwinChess is instrumented so you can turn development into small, defensible experiments instead of relying on generation numbers as proof of progress.

## Baseline experiment

Keep one fixed configuration for several nights and export after each session:

```bash
darwinchess export
```

Track:

- champion generation over wall-clock time;
- candidate acceptance rate;
- Arena score and Wilson lower bound;
- training loss;
- number of self-play positions;
- champion `classical_mix` lineage.

Do **not** call generation count an Elo rating. Absolute Elo requires calibration against fixed external opponents.

## Ablations

### A. Persistent replay vs recent-only replay

Change replay capacity/recent fraction. Hypothesis: persistent replay reduces catastrophic forgetting but slows adaptation.

### B. Pure self-play vs Stockfish teacher

Run matched compute budgets. Use `darwinchess teacher --positions N` only in the teacher condition.

### C. Fixed classical trust vs evolving trust

Set `classical_mix_step: 0` for control. Compare with the default Arena-gated mix mutation.

### D. Search depth teacher/student

Generate labels with a deeper search and play with a shallower search. Measure whether policy distillation recovers part of the depth gap.

### E. Promotion gate strictness

Vary `promotion_score` and `promotion_wilson_z`. Measure false-looking promotions, rejections and lineage survival.

## External strength calibration

For an actual rating curve, build a fixed opponent ladder (for example several Stockfish skill/limit settings), never train on the evaluation games, alternate colors, and keep openings controlled. Store calibration as separate metrics rather than modifying self-play Arena results.

## Reproducibility

Keep the YAML config, seed, exported `generations.csv`, `metrics.csv`, and PGNs for each experiment. The SQLite DB is the source of truth; CSV/PGN exports are analysis artifacts.
