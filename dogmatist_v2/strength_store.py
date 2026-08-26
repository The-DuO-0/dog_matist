from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .strength_lab import EngineGateDecision, EngineTrialEvidence, RoundStrengthEvidence, StrengthMode


_SCHEMA = """
CREATE TABLE IF NOT EXISTS strength_rounds (
    round_index INTEGER PRIMARY KEY,
    champion_generation INTEGER NOT NULL,
    promoted INTEGER NOT NULL,
    fixed_reference_score REAL NOT NULL,
    paired_games INTEGER NOT NULL,
    mode TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hard_positions (
    position_key TEXT PRIMARY KEY,
    fen TEXT NOT NULL,
    opening_bucket TEXT NOT NULL,
    source_generation INTEGER,
    source_kind TEXT NOT NULL,
    severity REAL NOT NULL,
    uncertainty REAL NOT NULL,
    value_error REAL NOT NULL,
    times_seen INTEGER NOT NULL DEFAULT 1,
    first_seen_round INTEGER NOT NULL,
    last_seen_round INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hard_positions_bucket
    ON hard_positions(opening_bucket, severity DESC, value_error DESC, uncertainty DESC);

CREATE TABLE IF NOT EXISTS engine_revisions (
    revision_id TEXT PRIMARY KEY,
    parent_revision_id TEXT,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    adopted_at TEXT
);

CREATE TABLE IF NOT EXISTS engine_trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id TEXT NOT NULL,
    baseline_revision_id TEXT NOT NULL,
    paired_games INTEGER NOT NULL,
    score_vs_baseline REAL NOT NULL,
    fixed_reference_delta REAL NOT NULL,
    compute_cost_ratio REAL NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_engine_trials_revision
    ON engine_trials(revision_id, recorded_at DESC);
"""


@dataclass(frozen=True)
class HardPositionEvidence:
    fen: str
    opening_bucket: str = "unknown"
    source_generation: int | None = None
    source_kind: str = "selfplay"
    severity: float = 0.0
    uncertainty: float = 0.0
    value_error: float = 0.0
    round_index: int = 0

    @property
    def position_key(self) -> str:
        # Ignore halfmove/fullmove clocks so the same board state deduplicates.
        fields = self.fen.strip().split()
        normalized = " ".join(fields[:4]) if len(fields) >= 4 else self.fen.strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    @property
    def priority(self) -> float:
        return max(0.0, self.severity) * 0.50 + max(0.0, self.value_error) * 0.35 + max(0.0, self.uncertainty) * 0.15


class StrengthStore:
    """Persistent, model-free evidence store for continuous strength growth."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StrengthStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def record_round(
        self,
        evidence: RoundStrengthEvidence,
        *,
        mode: StrengthMode,
        recorded_at: datetime,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO strength_rounds(
                round_index, champion_generation, promoted, fixed_reference_score,
                paired_games, mode, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(round_index) DO UPDATE SET
                champion_generation=excluded.champion_generation,
                promoted=excluded.promoted,
                fixed_reference_score=excluded.fixed_reference_score,
                paired_games=excluded.paired_games,
                mode=excluded.mode,
                recorded_at=excluded.recorded_at
            """,
            (
                evidence.round_index,
                evidence.champion_generation,
                int(evidence.promoted),
                evidence.fixed_reference_score,
                evidence.paired_games,
                mode.value,
                _iso(recorded_at),
            ),
        )
        self._conn.commit()

    def round_history(self, limit: int = 32) -> tuple[RoundStrengthEvidence, ...]:
        if limit <= 0:
            return ()
        rows = self._conn.execute(
            "SELECT * FROM strength_rounds ORDER BY round_index DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return tuple(
            RoundStrengthEvidence(
                round_index=row["round_index"],
                champion_generation=row["champion_generation"],
                promoted=bool(row["promoted"]),
                fixed_reference_score=row["fixed_reference_score"],
                paired_games=row["paired_games"],
            )
            for row in reversed(rows)
        )

    def upsert_hard_position(
        self,
        evidence: HardPositionEvidence,
        *,
        observed_at: datetime,
        max_per_bucket: int = 128,
    ) -> None:
        if max_per_bucket <= 0:
            raise ValueError("max_per_bucket must be positive")
        values = (evidence.severity, evidence.uncertainty, evidence.value_error)
        if any(value < 0.0 for value in values):
            raise ValueError("hard-position scores must be non-negative")
        timestamp = _iso(observed_at)
        self._conn.execute(
            """
            INSERT INTO hard_positions(
                position_key, fen, opening_bucket, source_generation, source_kind,
                severity, uncertainty, value_error, times_seen,
                first_seen_round, last_seen_round, first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            ON CONFLICT(position_key) DO UPDATE SET
                opening_bucket=excluded.opening_bucket,
                source_generation=COALESCE(excluded.source_generation, hard_positions.source_generation),
                source_kind=excluded.source_kind,
                severity=MAX(hard_positions.severity, excluded.severity),
                uncertainty=MAX(hard_positions.uncertainty, excluded.uncertainty),
                value_error=MAX(hard_positions.value_error, excluded.value_error),
                times_seen=hard_positions.times_seen + 1,
                last_seen_round=MAX(hard_positions.last_seen_round, excluded.last_seen_round),
                last_seen_at=excluded.last_seen_at
            """,
            (
                evidence.position_key,
                evidence.fen,
                evidence.opening_bucket or "unknown",
                evidence.source_generation,
                evidence.source_kind,
                evidence.severity,
                evidence.uncertainty,
                evidence.value_error,
                evidence.round_index,
                evidence.round_index,
                timestamp,
                timestamp,
            ),
        )
        self._conn.execute(
            """
            DELETE FROM hard_positions
            WHERE position_key IN (
                SELECT position_key FROM hard_positions
                WHERE opening_bucket=?
                ORDER BY
                    (severity * 0.50 + value_error * 0.35 + uncertainty * 0.15) DESC,
                    times_seen DESC,
                    last_seen_round DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (evidence.opening_bucket or "unknown", max_per_bucket),
        )
        self._conn.commit()

    def sample_hard_positions(
        self,
        limit: int,
        *,
        per_bucket_cap: int = 8,
        opening_buckets: Iterable[str] | None = None,
    ) -> tuple[HardPositionEvidence, ...]:
        if limit <= 0 or per_bucket_cap <= 0:
            return ()
        allowed = None
        if opening_buckets is not None:
            allowed = {str(bucket) for bucket in opening_buckets if str(bucket)}
            if not allowed:
                return ()
        rows = self._conn.execute(
            """
            SELECT * FROM hard_positions
            ORDER BY
                (severity * 0.50 + value_error * 0.35 + uncertainty * 0.15) DESC,
                times_seen DESC,
                last_seen_round DESC
            """
        ).fetchall()
        selected: list[HardPositionEvidence] = []
        bucket_counts: dict[str, int] = {}
        for row in rows:
            bucket = row["opening_bucket"]
            if allowed is not None and bucket not in allowed:
                continue
            if bucket_counts.get(bucket, 0) >= per_bucket_cap:
                continue
            selected.append(
                HardPositionEvidence(
                    fen=row["fen"],
                    opening_bucket=bucket,
                    source_generation=row["source_generation"],
                    source_kind=row["source_kind"],
                    severity=row["severity"],
                    uncertainty=row["uncertainty"],
                    value_error=row["value_error"],
                    round_index=row["last_seen_round"],
                )
            )
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if len(selected) >= limit:
                break
        return tuple(selected)

    def opening_bucket_stats(self, limit: int = 64) -> tuple[dict[str, object], ...]:
        """Aggregate only the dedicated early-opening evidence lane.

        A middlegame failure from a game labelled 'Queen's Gambit' must not by
        itself prove that the opening was bad. Only rows captured by the explicit
        opening lane (`*_opening`) feed this weakness meter.
        """
        if limit <= 0:
            return ()
        rows = self._conn.execute(
            """
            SELECT
                opening_bucket,
                COUNT(*) AS hard_positions,
                SUM(times_seen) AS hard_times_seen,
                AVG(severity * 0.50 + value_error * 0.35 + uncertainty * 0.15) AS mean_priority,
                MAX(severity * 0.50 + value_error * 0.35 + uncertainty * 0.15) AS max_priority,
                MAX(last_seen_round) AS last_seen_round
            FROM hard_positions
            WHERE source_kind LIKE '%_opening'
            GROUP BY opening_bucket
            ORDER BY max_priority DESC, hard_times_seen DESC, last_seen_round DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(
            {
                "opening_bucket": str(row["opening_bucket"]),
                "hard_positions": int(row["hard_positions"] or 0),
                "hard_times_seen": int(row["hard_times_seen"] or 0),
                "mean_priority": float(row["mean_priority"] or 0.0),
                "max_priority": float(row["max_priority"] or 0.0),
                "last_seen_round": int(row["last_seen_round"] or 0),
            }
            for row in rows
        )

    def hard_position_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM hard_positions").fetchone()
        return int(row["n"])

    def register_engine_revision(
        self,
        revision_id: str,
        *,
        parent_revision_id: str | None,
        description: str,
        created_at: datetime,
        status: str = "candidate",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO engine_revisions(
                revision_id, parent_revision_id, description, status, created_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(revision_id) DO UPDATE SET
                parent_revision_id=excluded.parent_revision_id,
                description=excluded.description
            """,
            (revision_id, parent_revision_id, description, status, _iso(created_at)),
        )
        self._conn.commit()

    def record_engine_trial(
        self,
        evidence: EngineTrialEvidence,
        *,
        baseline_revision_id: str,
        decision: EngineGateDecision,
        recorded_at: datetime,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO engine_trials(
                revision_id, baseline_revision_id, paired_games,
                score_vs_baseline, fixed_reference_delta, compute_cost_ratio,
                decision, reason, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.candidate_revision_id,
                baseline_revision_id,
                evidence.paired_games,
                evidence.score_vs_baseline,
                evidence.fixed_reference_delta,
                evidence.compute_cost_ratio,
                decision.action.value,
                decision.reason,
                _iso(recorded_at),
            ),
        )
        self._conn.commit()

    def adopt_engine_revision(self, revision_id: str, *, adopted_at: datetime) -> None:
        row = self._conn.execute(
            "SELECT revision_id FROM engine_revisions WHERE revision_id=?",
            (revision_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"unknown engine revision: {revision_id}")

        latest_trial = self._conn.execute(
            """
            SELECT decision, reason FROM engine_trials
            WHERE revision_id=? ORDER BY id DESC LIMIT 1
            """,
            (revision_id,),
        ).fetchone()
        if latest_trial is None:
            raise RuntimeError("engine revision has no recorded A/B gate evidence")
        if latest_trial["decision"] != "accept":
            raise RuntimeError(
                f"engine revision cannot be adopted: latest gate={latest_trial['decision']} "
                f"({latest_trial['reason']})"
            )

        self._conn.execute("UPDATE engine_revisions SET status='retired' WHERE status='active'")
        self._conn.execute(
            "UPDATE engine_revisions SET status='active', adopted_at=? WHERE revision_id=?",
            (_iso(adopted_at), revision_id),
        )
        self._conn.commit()

    def active_engine_revision(self) -> str | None:
        row = self._conn.execute(
            "SELECT revision_id FROM engine_revisions WHERE status='active' ORDER BY adopted_at DESC LIMIT 1"
        ).fetchone()
        return str(row["revision_id"]) if row is not None else None


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("StrengthStore datetimes must be timezone-aware")
    return value.isoformat()
