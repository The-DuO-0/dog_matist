from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Iterable

from .strength_pipeline import StrengthRoundRecipe


@dataclass(frozen=True)
class ReplayBatchQuota:
    natural_selfplay: int
    hard_positions: int
    specialist_sparring: int
    deep_search_teacher: int

    @property
    def total(self) -> int:
        return self.natural_selfplay + self.hard_positions + self.specialist_sparring + self.deep_search_teacher

    def as_dict(self) -> dict[str, int]:
        return {
            "natural_selfplay": self.natural_selfplay,
            "hard_positions": self.hard_positions,
            "specialist_sparring": self.specialist_sparring,
            "deep_search_teacher": self.deep_search_teacher,
            "total": self.total,
        }


class LiveReplayMixSampler:
    """Sample the existing production replay DB using a Strength Lab recipe.

    This does not duplicate hard positions. It selects the original replay rows by
    FEN, teacher rows by their game source, and specialist rows by the metadata the
    current production MemoryStore already writes. Any unavailable quota is filled
    from the ordinary lifetime replay sampler.

    Opening-focused population branches keep their identity: when the production
    trainer asks for a particular opening, targeted hard/teacher/specialist rows are
    restricted to that opening where possible, and the ordinary fallback delegates
    to the original production sampler with all of its existing focus arguments.
    """

    def __init__(self, *, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()

    def quota_for(self, recipe: StrengthRoundRecipe, batch_size: int) -> ReplayBatchQuota:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        weights = (
            ("natural_selfplay", recipe.natural_selfplay_examples),
            ("hard_positions", len(recipe.hard_positions)),
            ("specialist_sparring", recipe.specialist_examples),
            ("deep_search_teacher", len(recipe.teacher_requests)),
        )
        total = sum(value for _, value in weights)
        if total <= 0:
            return ReplayBatchQuota(batch_size, 0, 0, 0)
        raw = [(name, batch_size * value / total) for name, value in weights]
        counts = {name: int(value) for name, value in raw}
        left = batch_size - sum(counts.values())
        order = sorted(raw, key=lambda item: (item[1] - int(item[1]), item[1]), reverse=True)
        for index in range(left):
            counts[order[index % len(order)][0]] += 1
        return ReplayBatchQuota(
            counts["natural_selfplay"],
            counts["hard_positions"],
            counts["specialist_sparring"],
            counts["deep_search_teacher"],
        )

    def _choose(self, pool: Iterable[Any], take: int, seen: set[int]) -> list[Any]:
        if take <= 0:
            return []
        candidates = [row for row in pool if int(row["id"]) not in seen]
        if not candidates:
            return []
        chosen = self.rng.sample(candidates, min(take, len(candidates)))
        seen.update(int(row["id"]) for row in chosen)
        return chosen

    @staticmethod
    def _focus_values(values: Iterable[str] | None) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(value) for value in (values or ()) if value))

    def _hard_rows(
        self,
        memory: Any,
        recipe: StrengthRoundRecipe,
        take: int,
        seen: set[int],
        *,
        opening_names: Iterable[str] | None = None,
    ) -> list[Any]:
        focus = set(self._focus_values(opening_names))
        hard_rows = [
            row for row in recipe.hard_positions
            if row.fen and (not focus or row.opening_bucket in focus)
        ]
        fens = list(dict.fromkeys(row.fen for row in hard_rows))
        if take <= 0 or not fens:
            return []
        # Production targeted_examples is intentionally small (tens, not thousands),
        # so a bounded IN query is simpler and safer than a new schema/index here.
        fens = fens[:256]
        placeholders = ",".join("?" for _ in fens)
        pool = memory.conn.execute(
            f"SELECT * FROM examples WHERE fen IN ({placeholders}) ORDER BY priority DESC, id DESC",
            fens,
        ).fetchall()
        return self._choose(pool, take, seen)

    def _teacher_rows(
        self,
        memory: Any,
        take: int,
        seen: set[int],
        *,
        opening_names: Iterable[str] | None = None,
    ) -> list[Any]:
        if take <= 0:
            return []
        focus = self._focus_values(opening_names)
        params: list[Any] = []
        focus_sql = ""
        if focus:
            placeholders = ",".join("?" for _ in focus)
            focus_sql = f" AND e.opening_name IN ({placeholders})"
            params.extend(focus)
        params.append(max(64, take * 12))
        pool = memory.conn.execute(
            f"""
            SELECT e.* FROM examples e
            JOIN games g ON g.id=e.game_id
            WHERE g.source='strength_teacher'{focus_sql}
            ORDER BY e.id DESC LIMIT ?
            """,
            params,
        ).fetchall()
        return self._choose(pool, take, seen)

    def _specialist_rows(
        self,
        memory: Any,
        take: int,
        seen: set[int],
        *,
        opening_names: Iterable[str] | None = None,
        generations: Iterable[int] | None = None,
    ) -> list[Any]:
        if take <= 0 or not hasattr(memory, "active_specialists"):
            return []
        specialists = list(memory.active_specialists(limit=64))
        specialist_generations = {int(row["generation"]) for row in specialists}
        specialist_openings = {str(row["opening_name"]) for row in specialists if row["opening_name"]}

        requested_openings = set(self._focus_values(opening_names))
        requested_generations = {int(value) for value in (generations or ())}
        if requested_openings:
            specialist_openings &= requested_openings
        if requested_generations:
            specialist_generations &= requested_generations

        selected_generations = sorted(specialist_generations)
        selected_openings = sorted(specialist_openings)
        if not selected_generations or not selected_openings:
            return []
        gq = ",".join("?" for _ in selected_generations)
        oq = ",".join("?" for _ in selected_openings)
        pool = memory.conn.execute(
            f"""
            SELECT * FROM examples
            WHERE origin_generation IN ({gq}) AND opening_name IN ({oq})
            ORDER BY id DESC LIMIT ?
            """,
            [*selected_generations, *selected_openings, max(128, take * 16)],
        ).fetchall()
        return self._choose(pool, take, seen)

    def sample(
        self,
        memory: Any,
        recipe: StrengthRoundRecipe,
        *,
        batch_size: int,
        recent_fraction: float = 0.35,
        opening_names: Iterable[str] | None = None,
        generations: Iterable[int] | None = None,
        ordinary_sampler: Callable[..., Iterable[Any]] | None = None,
        ordinary_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        quota = self.quota_for(recipe, batch_size)
        rows: list[Any] = []
        seen: set[int] = set()

        rows.extend(
            self._hard_rows(
                memory,
                recipe,
                quota.hard_positions,
                seen,
                opening_names=opening_names,
            )
        )
        rows.extend(
            self._specialist_rows(
                memory,
                quota.specialist_sparring,
                seen,
                opening_names=opening_names,
                generations=generations,
            )
        )
        rows.extend(
            self._teacher_rows(
                memory,
                quota.deep_search_teacher,
                seen,
                opening_names=opening_names,
            )
        )

        # Missing targeted evidence is deliberately backfilled with the existing
        # lifetime replay policy instead of repeated copies of the same weakness.
        remaining = batch_size - len(rows)
        if remaining > 0:
            sampler = ordinary_sampler or memory.replay_sample
            kwargs = dict(ordinary_kwargs or {})
            ordinary = sampler(max(batch_size * 2, remaining * 3), recent_fraction, **kwargs)
            for row in ordinary:
                row_id = int(row["id"])
                if row_id in seen:
                    continue
                seen.add(row_id)
                rows.append(row)
                if len(rows) >= batch_size:
                    break

        if len(rows) < batch_size:
            pool = memory.conn.execute(
                "SELECT * FROM examples ORDER BY id DESC LIMIT ?",
                (batch_size * 4,),
            ).fetchall()
            for row in pool:
                row_id = int(row["id"])
                if row_id in seen:
                    continue
                seen.add(row_id)
                rows.append(row)
                if len(rows) >= batch_size:
                    break
        return rows[:batch_size]


class LiveReplayOverride:
    """Temporarily route the existing ContinualTrainer through Strength Lab.

    The supplied production trainer already calls ``memory.replay_sample`` on
    every optimization step. Replacing that one bound instance method inside a
    context manager lets us reuse the live trainer unchanged. The original method
    is restored even when training raises an exception.

    Existing opening-focus arguments are preserved for the ordinary fallback and
    are also used to keep targeted rows compatible with specialist branches.
    """

    def __init__(
        self,
        memory: Any,
        recipe: StrengthRoundRecipe,
        *,
        sampler: LiveReplayMixSampler | None = None,
    ) -> None:
        self.memory = memory
        self.recipe = recipe
        self.sampler = sampler or LiveReplayMixSampler()
        self._original: Callable[..., Any] | None = None
        self._had_instance_attr = False
        self._previous_instance_attr: Any = None

    def __enter__(self) -> "LiveReplayOverride":
        if self._original is not None:
            raise RuntimeError("LiveReplayOverride cannot be entered twice")
        self._original = self.memory.replay_sample
        attrs = getattr(self.memory, "__dict__", {})
        self._had_instance_attr = "replay_sample" in attrs
        if self._had_instance_attr:
            self._previous_instance_attr = attrs["replay_sample"]

        original = self._original

        def strength_replay_sample(
            batch_size: int,
            recent_fraction: float = 0.35,
            *,
            opening_names: Iterable[str] | None = None,
            opening_fraction: float = 0.0,
            generations: Iterable[int] | None = None,
        ) -> list[Any]:
            ordinary_kwargs = {
                "opening_names": opening_names,
                "opening_fraction": opening_fraction,
                "generations": generations,
            }
            return self.sampler.sample(
                self.memory,
                self.recipe,
                batch_size=batch_size,
                recent_fraction=recent_fraction,
                opening_names=opening_names,
                generations=generations,
                ordinary_sampler=original,
                ordinary_kwargs=ordinary_kwargs,
            )

        self.memory.replay_sample = strength_replay_sample
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._original is None:
            return
        if self._had_instance_attr:
            self.memory.replay_sample = self._previous_instance_attr
        else:
            try:
                delattr(self.memory, "replay_sample")
            except AttributeError:
                # A highly dynamic production wrapper may have removed it itself;
                # restoring the captured bound method is still safer than leaving
                # the Strength Lab closure installed.
                self.memory.replay_sample = self._original
        self._original = None
