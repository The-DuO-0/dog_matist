from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class OpeningSearchObservation:
    ply: int
    fen: str
    baseline_move: str
    deeper_move: str
    baseline_score_cp: float | None = None
    deeper_score_cp: float | None = None
    baseline_depth: int | None = None
    deeper_depth: int | None = None
    baseline_elapsed_s: float | None = None
    deeper_elapsed_s: float | None = None

    @property
    def move_changed(self) -> bool:
        return self.baseline_move != self.deeper_move

    @property
    def score_delta_cp(self) -> float | None:
        if self.baseline_score_cp is None or self.deeper_score_cp is None:
            return None
        return abs(float(self.deeper_score_cp) - float(self.baseline_score_cp))

    def status(self, *, watch_cp: float = 80.0, unstable_cp: float = 150.0) -> str:
        delta = self.score_delta_cp
        if self.move_changed and (delta is None or delta >= watch_cp):
            return "HORIZON_SENSITIVE"
        if self.move_changed:
            return "MOVE_FLIP"
        if delta is not None and delta >= unstable_cp:
            return "HORIZON_SENSITIVE"
        if delta is not None and delta >= watch_cp:
            return "WATCH"
        return "STABLE"

    def as_dict(self) -> dict[str, object]:
        return {
            "ply": self.ply,
            "fen": self.fen,
            "baseline_move": self.baseline_move,
            "deeper_move": self.deeper_move,
            "baseline_score_cp": self.baseline_score_cp,
            "deeper_score_cp": self.deeper_score_cp,
            "score_delta_cp": self.score_delta_cp,
            "baseline_depth": self.baseline_depth,
            "deeper_depth": self.deeper_depth,
            "baseline_elapsed_s": self.baseline_elapsed_s,
            "deeper_elapsed_s": self.deeper_elapsed_s,
            "move_changed": self.move_changed,
            "status": self.status(),
        }


@dataclass(frozen=True)
class OpeningSearchStabilityReport:
    generation: int
    baseline_depth: int
    deeper_depth: int
    observations: tuple[OpeningSearchObservation, ...]

    @property
    def move_flips(self) -> int:
        return sum(1 for row in self.observations if row.move_changed)

    @property
    def horizon_sensitive(self) -> int:
        return sum(1 for row in self.observations if row.status() == "HORIZON_SENSITIVE")

    @property
    def flip_rate(self) -> float:
        return self.move_flips / len(self.observations) if self.observations else 0.0

    @property
    def early_search_unstable(self) -> bool:
        if not self.observations:
            return False
        # One isolated move flip can be normal at shallow search. Require either
        # repeated high-signal horizon sensitivity or a meaningful multi-position
        # sample before a raw flip rate can label the opening search unstable.
        enough_positions_for_rate = len(self.observations) >= 4
        return self.horizon_sensitive >= 2 or (
            enough_positions_for_rate and self.flip_rate >= 0.30
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "baseline_depth": self.baseline_depth,
            "deeper_depth": self.deeper_depth,
            "positions": len(self.observations),
            "move_flips": self.move_flips,
            "flip_rate": self.flip_rate,
            "horizon_sensitive": self.horizon_sensitive,
            "early_search_unstable": self.early_search_unstable,
            "book_moves_injected": False,
            "observations": [row.as_dict() for row in self.observations],
        }

    def format_text(self) -> str:
        lines = [
            f"GEN{self.generation} DETERMINISTIC OPENING SEARCH STABILITY",
            "=" * 48,
            f"baseline depth={self.baseline_depth}  deeper depth={self.deeper_depth}",
        ]
        for row in self.observations:
            delta = "?" if row.score_delta_cp is None else f"{row.score_delta_cp:.0f}cp"
            lines.append(
                f"ply {row.ply:>2}: {row.baseline_move:<6} -> {row.deeper_move:<6} "
                f"delta={delta:<7} {row.status()}"
            )
        lines.extend(
            [
                "",
                f"Move flips: {self.move_flips}/{len(self.observations)} ({self.flip_rate:.0%})",
                f"Horizon-sensitive positions: {self.horizon_sensitive}",
                "Conclusion: "
                + (
                    "EARLY SEARCH IS UNSTABLE; test an opening-only search stabilization revision."
                    if self.early_search_unstable
                    else "NO STRONG HORIZON INSTABILITY; prioritize evaluator/training repair over more search."
                ),
                "Book moves: OFF",
            ]
        )
        return "\n".join(lines)


def build_stability_report(
    generation: int,
    baseline_depth: int,
    deeper_depth: int,
    observations: Iterable[OpeningSearchObservation],
) -> OpeningSearchStabilityReport:
    if generation < 0:
        raise ValueError("generation must be non-negative")
    if baseline_depth <= 0 or deeper_depth <= baseline_depth:
        raise ValueError("deeper_depth must be greater than baseline_depth")
    rows = tuple(observations)
    if any(row.ply <= 0 for row in rows):
        raise ValueError("ply numbers must be positive")
    return OpeningSearchStabilityReport(
        generation=int(generation),
        baseline_depth=int(baseline_depth),
        deeper_depth=int(deeper_depth),
        observations=rows,
    )
