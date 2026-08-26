from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import chess


@dataclass(frozen=True)
class OpeningSeed:
    name: str
    moves: tuple[str, ...]
    family: str = "curated"
    weight: float = 1.0

    def board(self) -> chess.Board:
        board = chess.Board()
        for san in self.moves:
            board.push_san(san)
        return board


# Intentionally broad rather than deep.  The learner must solve the resulting
# positions; this is a curriculum, not a move-selection opening book.
CURATED_OPENINGS: tuple[OpeningSeed, ...] = (
    OpeningSeed("Open Game", ("e4", "e5", "Nf3", "Nc6"), "e4"),
    OpeningSeed("Italian", ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"), "e4"),
    OpeningSeed("Ruy Lopez", ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"), "e4"),
    OpeningSeed("Scotch", ("e4", "e5", "Nf3", "Nc6", "d4", "exd4"), "e4"),
    OpeningSeed("Sicilian", ("e4", "c5", "Nf3", "d6"), "e4"),
    OpeningSeed("Sicilian Najdorf shell", ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"), "e4"),
    OpeningSeed("French", ("e4", "e6", "d4", "d5"), "e4"),
    OpeningSeed("Caro-Kann", ("e4", "c6", "d4", "d5"), "e4"),
    OpeningSeed("Pirc", ("e4", "d6", "d4", "Nf6", "Nc3", "g6"), "e4"),
    OpeningSeed("Queen's Gambit", ("d4", "d5", "c4"), "d4"),
    OpeningSeed("QGD", ("d4", "d5", "c4", "e6", "Nc3", "Nf6"), "d4"),
    OpeningSeed("Slav", ("d4", "d5", "c4", "c6"), "d4"),
    OpeningSeed("King's Indian", ("d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6"), "d4"),
    OpeningSeed("Nimzo-Indian", ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"), "d4"),
    OpeningSeed("Grunfeld", ("d4", "Nf6", "c4", "g6", "Nc3", "d5"), "d4"),
    OpeningSeed("English", ("c4", "e5", "Nc3", "Nf6"), "flank"),
    OpeningSeed("Symmetrical English", ("c4", "c5", "Nc3", "Nc6"), "flank"),
    OpeningSeed("Reti", ("Nf3", "d5", "g3", "Nf6", "Bg2", "g6"), "flank"),
    OpeningSeed("Bird", ("f4", "d5", "Nf3", "Nf6"), "uncommon", 0.65),
    OpeningSeed("Scandinavian", ("e4", "d5", "exd5", "Qxd5", "Nc3", "Qd8"), "uncommon", 0.75),
)


@dataclass(frozen=True)
class CurriculumMix:
    standard: float = 0.35
    curated: float = 0.35
    uncommon: float = 0.20
    controlled_random: float = 0.10

    def normalized(self) -> tuple[float, float, float, float]:
        vals = [max(0.0, self.standard), max(0.0, self.curated), max(0.0, self.uncommon), max(0.0, self.controlled_random)]
        total = sum(vals) or 1.0
        return tuple(v / total for v in vals)  # type: ignore[return-value]


class OpeningCurriculum:
    """Produces legal starting boards for self-play and paired Arena testing.

    It deliberately does *not* prescribe moves after the seed.  Once the seed
    position is reached, normal dog_matist search/policy chooses everything.
    """

    def __init__(self, seed: int | None = None, mix: CurriculumMix | None = None):
        self.rng = random.Random(seed)
        self.mix = mix or CurriculumMix()

    def sample(self) -> tuple[chess.Board, str, str]:
        standard, curated, uncommon, controlled = self.mix.normalized()
        r = self.rng.random()
        if r < standard:
            return chess.Board(), "Initial position", "standard"
        if r < standard + curated:
            pool = [o for o in CURATED_OPENINGS if o.family not in {"uncommon"}]
            opening = self._weighted_choice(pool)
            return opening.board(), opening.name, opening.family
        if r < standard + curated + uncommon:
            pool = [o for o in CURATED_OPENINGS if o.family == "uncommon"]
            opening = self._weighted_choice(pool or list(CURATED_OPENINGS))
            return opening.board(), opening.name, "uncommon"
        board = self._controlled_random_board()
        return board, "Controlled random", "controlled_random"

    def arena_pairs(self, pair_count: int) -> list[tuple[chess.Board, str]]:
        """Return distinct-ish seeds; caller plays each board twice with colors swapped."""
        if pair_count <= 0:
            return []
        pool = list(CURATED_OPENINGS)
        self.rng.shuffle(pool)
        out: list[tuple[chess.Board, str]] = []
        while len(out) < pair_count:
            for opening in pool:
                out.append((opening.board(), opening.name))
                if len(out) >= pair_count:
                    break
            self.rng.shuffle(pool)
        return out

    def _weighted_choice(self, pool: Iterable[OpeningSeed]) -> OpeningSeed:
        seq = list(pool)
        if not seq:
            raise RuntimeError("opening pool is empty")
        return self.rng.choices(seq, weights=[max(0.01, x.weight) for x in seq], k=1)[0]

    def _controlled_random_board(self) -> chess.Board:
        # Start from a sound position, then make a short legal random walk.
        # Filtering keeps this from becoming a garbage-position generator.
        base = self._weighted_choice(CURATED_OPENINGS).board()
        target_plies = self.rng.randint(2, 8)
        for _ in range(target_plies):
            legal = list(base.legal_moves)
            if not legal or base.is_game_over():
                break
            quietish = []
            for move in legal:
                # Avoid intentionally throwing the queen or choosing an immediate
                # terminal blunder merely for diversity. Captures/checks remain possible.
                piece = base.piece_at(move.from_square)
                if piece and piece.piece_type == chess.QUEEN and base.fullmove_number < 7:
                    continue
                quietish.append(move)
            move = self.rng.choice(quietish or legal)
            base.push(move)
        return base


def validate_curriculum() -> list[str]:
    errors: list[str] = []
    seen = set()
    for opening in CURATED_OPENINGS:
        try:
            board = opening.board()
            if board.is_game_over():
                errors.append(f"{opening.name}: seed is already terminal")
            key = board.fen()
            if key in seen:
                errors.append(f"{opening.name}: duplicate final position")
            seen.add(key)
        except Exception as exc:
            errors.append(f"{opening.name}: {exc}")
    return errors
