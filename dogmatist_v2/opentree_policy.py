from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TreeHealth:
    """Small, runtime-safe summary of OpenTree health.

    This deliberately contains aggregate scalars only. The controller never
    needs the full opening graph in memory.
    """

    root_visits: int
    root_top_move_share: float
    root_effective_branches: float
    viable_frontier: int
    strict_holdout: int
    branch_revisit_ratio: float = 0.0
    collapse_warning: bool = False

    def __post_init__(self) -> None:
        if self.root_visits < 0 or self.viable_frontier < 0 or self.strict_holdout < 0:
            raise ValueError("counts must be non-negative")
        for value, name in (
            (self.root_top_move_share, "root_top_move_share"),
            (self.branch_revisit_ratio, "branch_revisit_ratio"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isfinite(self.root_effective_branches) or self.root_effective_branches < 0.0:
            raise ValueError("root_effective_branches must be finite and non-negative")


@dataclass(frozen=True)
class CurriculumMix:
    natural: float
    frontier: float
    specialist: float
    anchor: float

    def __post_init__(self) -> None:
        values = (self.natural, self.frontier, self.specialist, self.anchor)
        if any((not isfinite(v) or v < 0.0) for v in values):
            raise ValueError("curriculum weights must be finite and non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("curriculum weights must sum to 1")

    def as_dict(self) -> dict[str, float]:
        return {
            "natural": self.natural,
            "frontier": self.frontier,
            "specialist": self.specialist,
            "anchor": self.anchor,
        }


@dataclass(frozen=True)
class OpenTreePolicy:
    mix: CurriculumMix
    early_temperature_scale: float
    frontier_gap_cp: int
    reason: str


class OpenTreeCurriculumController:
    """Adaptive anti-collapse controller that changes *where* to train.

    It never forces a named opening or a particular first move. If OpenTree
    becomes too concentrated, the controller increases Frontier sampling and
    early stochasticity while preserving the search-safety guard. Natural
    self-play remains the fallback when there is no useful frontier inventory.

    Hysteresis plus bounded per-update movement prevents a single noisy round
    from causing curriculum oscillation.
    """

    def __init__(
        self,
        *,
        base: CurriculumMix | None = None,
        min_root_visits: int = 40,
        collapse_top_share: float = 0.76,
        recovery_top_share: float = 0.64,
        collapse_effective_branches: float = 2.2,
        recovery_effective_branches: float = 2.8,
        max_frontier: float = 0.48,
        min_specialist: float = 0.08,
        min_anchor: float = 0.06,
        max_step: float = 0.06,
        base_frontier_gap_cp: int = 110,
        max_frontier_gap_cp: int = 150,
    ) -> None:
        self.base = base or CurriculumMix(0.45, 0.30, 0.15, 0.10)
        self.min_root_visits = min_root_visits
        self.collapse_top_share = collapse_top_share
        self.recovery_top_share = recovery_top_share
        self.collapse_effective_branches = collapse_effective_branches
        self.recovery_effective_branches = recovery_effective_branches
        self.max_frontier = max_frontier
        self.min_specialist = min_specialist
        self.min_anchor = min_anchor
        self.max_step = max_step
        self.base_frontier_gap_cp = base_frontier_gap_cp
        self.max_frontier_gap_cp = max_frontier_gap_cp
        self._mix = self.base
        self._collapsed = False

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def mix(self) -> CurriculumMix:
        return self._mix

    def _detect(self, health: TreeHealth) -> bool:
        if health.root_visits < self.min_root_visits:
            return self._collapsed
        if self._collapsed:
            recovered = (
                health.root_top_move_share <= self.recovery_top_share
                and health.root_effective_branches >= self.recovery_effective_branches
            )
            return not recovered
        return bool(
            health.collapse_warning
            or health.root_top_move_share >= self.collapse_top_share
            or health.root_effective_branches <= self.collapse_effective_branches
        )

    @staticmethod
    def _normalize(natural: float, frontier: float, specialist: float, anchor: float) -> CurriculumMix:
        total = natural + frontier + specialist + anchor
        if total <= 0.0:
            raise ValueError("curriculum total must be positive")
        return CurriculumMix(
            natural / total,
            frontier / total,
            specialist / total,
            anchor / total,
        )

    def _recover_toward_base(self, cur: CurriculumMix) -> CurriculumMix:
        current = (cur.natural, cur.frontier, cur.specialist, cur.anchor)
        target = (self.base.natural, self.base.frontier, self.base.specialist, self.base.anchor)
        max_abs_delta = max(abs(t - v) for v, t in zip(current, target))
        if max_abs_delta <= 1e-12:
            return self.base
        alpha = min(1.0, self.max_step / max_abs_delta)
        moved = tuple(v + alpha * (t - v) for v, t in zip(current, target))
        # Convex interpolation preserves the exact unit total up to float noise.
        return self._normalize(*moved)

    def update(self, health: TreeHealth) -> OpenTreePolicy:
        self._collapsed = self._detect(health)
        cur = self._mix

        if self._collapsed:
            # Spend more budget on organically expanding the tree. We do not
            # choose a human opening or force a root move.
            frontier_room = max(0.0, self.max_frontier - cur.frontier)
            add_frontier = min(self.max_step, frontier_room)
            specialist_room = max(0.0, cur.specialist - self.min_specialist)
            anchor_room = max(0.0, cur.anchor - self.min_anchor)
            take_specialist = min(add_frontier * 0.65, specialist_room)
            take_anchor = min(add_frontier - take_specialist, anchor_room)
            funded = take_specialist + take_anchor

            # If protected floors cannot fund the desired frontier increase,
            # shift the remainder from Natural only very gently.
            take_natural = min(max(0.0, add_frontier - funded), self.max_step * 0.20, cur.natural)
            actual_add = funded + take_natural
            natural = cur.natural - take_natural
            frontier = cur.frontier + actual_add
            specialist = cur.specialist - take_specialist
            anchor = cur.anchor - take_anchor
            self._mix = self._normalize(natural, frontier, specialist, anchor)
            severity = max(
                max(0.0, health.root_top_move_share - self.recovery_top_share),
                max(0.0, self.recovery_effective_branches - health.root_effective_branches) / 4.0,
            )
            temp_scale = min(1.35, 1.08 + severity)
            gap_cp = min(self.max_frontier_gap_cp, self.base_frontier_gap_cp + int(80 * severity))
            reason = "root-concentration: expand frontier coverage"
        else:
            self._mix = self._recover_toward_base(cur)
            temp_scale = 1.0
            gap_cp = self.base_frontier_gap_cp
            reason = "tree-health nominal"

        # No frontier inventory means there is nothing useful to sample. Move
        # that fraction back to Natural instead of wasting games/retrying.
        if health.viable_frontier == 0 and self._mix.frontier > 0.0:
            f = self._mix.frontier
            self._mix = CurriculumMix(
                self._mix.natural + f,
                0.0,
                self._mix.specialist,
                self._mix.anchor,
            )
            reason += "; frontier inventory empty -> Natural fallback"

        return OpenTreePolicy(
            mix=self._mix,
            early_temperature_scale=temp_scale,
            frontier_gap_cp=gap_cp,
            reason=reason,
        )
