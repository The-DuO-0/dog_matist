# Validation

DarwinChess 1.0 was validated before packaging with the following checks:

- Python compile check for package, tests, and examples.
- Shell syntax check for macOS setup/night/normal launchers.
- 9 automated tests passing, including board encoding, model shapes, persistent SQLite replay, atomic generation promotion, mate-in-one search, persistent runtime initialization, and a complete self-play -> replay -> gradient update -> challenger -> held-out Arena -> outer-agent API cycle.
- Manual CLI smoke checks for `doctor`, `status`, `analyze`, and research `export`.
- Manual outer-agent API smoke checks for `status`, `best_move`, `chat`, and durable memory reads.
- Generation-0 performance check confirming the pure-classical genome skips neural forward passes that would otherwise be multiplied by zero.

The validation environment was a Linux CPU container. The macOS launcher and MPS selection path are included and checked syntactically, but the actual Apple MPS runtime can only be verified on the target Mac by `./setup_mac.sh` / `darwinchess doctor`.
