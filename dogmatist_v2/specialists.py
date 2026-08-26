from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable

from .league import MatchResult


@dataclass(frozen=True)
class OpeningBucket:
    """Stable opening label used for specialist accounting.

    Prefer ECO codes when available; a normalized move-prefix string is a safe
    fallback for games that do not have an ECO classifier attached yet.
    """

    key: str
    name: str | None = None


@dataclass(frozen=True)
class SpecialistRecord:
    candidate_id: str
    bucket: OpeningBucket
    games: int
    points: float
    score: float
    advantage_over_reference: float


@dataclass
class SpecialistArchive:
    records: dict[str, list[SpecialistRecord]] = field(default_factory=dict)

    def update_from_matches(
        self,
        matches: Iterable[MatchResult],
        *,
        reference_id: str,
        minimum_games: int = 4,
        minimum_advantage: float = 0.08,
        top_per_bucket: int = 2,
    ) -> list[SpecialistRecord]:
        """Discover candidates that outperform the reference in an opening.

        This is deliberately independent from overall promotion. A challenger
        can lose the league and still become a durable specialist donor.
        """

        stats: dict[tuple[str, str], list[float]] = defaultdict(list)
        for match in matches:
            if not match.opening:
                continue
            for cid in (match.white_id, match.black_id):
                stats[(match.opening, cid)].append(match.score_for(cid))

        by_bucket: dict[str, list[SpecialistRecord]] = defaultdict(list)
        for (opening, cid), values in stats.items():
            if cid == reference_id or len(values) < minimum_games:
                continue
            ref_values = stats.get((opening, reference_id), [])
            if len(ref_values) < minimum_games:
                continue
            score = sum(values) / len(values)
            ref_score = sum(ref_values) / len(ref_values)
            advantage = score - ref_score
            if advantage < minimum_advantage:
                continue
            by_bucket[opening].append(
                SpecialistRecord(
                    candidate_id=cid,
                    bucket=OpeningBucket(opening),
                    games=len(values),
                    points=sum(values),
                    score=score,
                    advantage_over_reference=advantage,
                )
            )

        accepted: list[SpecialistRecord] = []
        for opening, records in by_bucket.items():
            records.sort(
                key=lambda r: (r.advantage_over_reference, r.games, r.score),
                reverse=True,
            )
            kept = records[:top_per_bucket]
            self.records[opening] = kept
            accepted.extend(kept)
        return accepted

    def donors_for(self, opening: str) -> list[SpecialistRecord]:
        return list(self.records.get(opening, ()))

    def candidate_weights(self) -> dict[str, float]:
        """Produce replay-sampling weights for specialist inheritance.

        These are curriculum weights, not neural weight interpolation.
        """

        weights: dict[str, float] = defaultdict(float)
        for records in self.records.values():
            for record in records:
                weights[record.candidate_id] += max(
                    0.0,
                    record.advantage_over_reference,
                ) * max(1, record.games)
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {cid: value / total for cid, value in weights.items()}
