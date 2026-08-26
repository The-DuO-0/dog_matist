from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .archive import ArchiveEntry, ArchiveTier
from .dynasty import ChampionReign, HistoricalEvent


SCHEMA_VERSION = 1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chronicle_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS champion_reigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    dethroned_by INTEGER,
    challengers_faced INTEGER NOT NULL DEFAULT 0,
    games_during_reign INTEGER NOT NULL DEFAULT 0,
    replacement_reason TEXT,
    UNIQUE(generation_id, started_at)
);
CREATE INDEX IF NOT EXISTS idx_champion_reigns_generation
    ON champion_reigns(generation_id);
CREATE INDEX IF NOT EXISTS idx_champion_reigns_started
    ON champion_reigns(started_at);

CREATE TABLE IF NOT EXISTS generation_archive (
    generation_id INTEGER PRIMARY KEY,
    tier TEXT NOT NULL,
    checkpoint_path TEXT,
    checkpoint_bytes INTEGER NOT NULL DEFAULT 0,
    ever_champion INTEGER NOT NULL DEFAULT 0,
    specialist_score REAL NOT NULL DEFAULT 0.0,
    protected INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL DEFAULT '',
    archived_at TEXT,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_generation_archive_tier
    ON generation_archive(tier);

CREATE TABLE IF NOT EXISTS generation_traits (
    generation_id INTEGER NOT NULL,
    trait_kind TEXT NOT NULL,
    trait_key TEXT NOT NULL,
    score REAL NOT NULL,
    sample_games INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY(generation_id, trait_kind, trait_key)
);
CREATE INDEX IF NOT EXISTS idx_generation_traits_lookup
    ON generation_traits(trait_kind, trait_key, score DESC);

CREATE TABLE IF NOT EXISTS historical_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    generation_id INTEGER,
    related_generation_id INTEGER,
    text TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_historical_events_time
    ON historical_events(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_historical_events_generation
    ON historical_events(generation_id, occurred_at DESC);
"""


class ChronicleStore:
    """Small SQLite store for durable evolutionary history.

    This store is intentionally independent from model loading. Querying the
    Chronicle must never cause a checkpoint to enter RAM.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO chronicle_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ChronicleStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _archive_values(
        entry: ArchiveEntry,
        archived_at: datetime | None,
        last_used_at: datetime | None,
    ) -> tuple[object, ...]:
        return (
            entry.generation_id,
            entry.tier.value,
            str(entry.checkpoint_path) if entry.checkpoint_path else None,
            entry.checkpoint_bytes,
            int(entry.ever_champion),
            entry.specialist_score,
            int(entry.protected),
            entry.reason,
            _iso(archived_at),
            _iso(last_used_at),
        )

    def _upsert_archive_entry_no_commit(
        self,
        entry: ArchiveEntry,
        *,
        archived_at: datetime | None = None,
        last_used_at: datetime | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO generation_archive(
                generation_id, tier, checkpoint_path, checkpoint_bytes,
                ever_champion, specialist_score, protected, reason,
                archived_at, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                tier=excluded.tier,
                checkpoint_path=excluded.checkpoint_path,
                checkpoint_bytes=excluded.checkpoint_bytes,
                ever_champion=MAX(generation_archive.ever_champion, excluded.ever_champion),
                specialist_score=MAX(generation_archive.specialist_score, excluded.specialist_score),
                protected=MAX(generation_archive.protected, excluded.protected),
                reason=excluded.reason,
                archived_at=COALESCE(excluded.archived_at, generation_archive.archived_at),
                last_used_at=COALESCE(excluded.last_used_at, generation_archive.last_used_at)
            """,
            self._archive_values(entry, archived_at, last_used_at),
        )

    def upsert_archive_entry(
        self,
        entry: ArchiveEntry,
        *,
        archived_at: datetime | None = None,
        last_used_at: datetime | None = None,
    ) -> None:
        self._upsert_archive_entry_no_commit(
            entry,
            archived_at=archived_at,
            last_used_at=last_used_at,
        )
        self._conn.commit()

    def archive_entries(self) -> tuple[ArchiveEntry, ...]:
        rows = self._conn.execute(
            "SELECT * FROM generation_archive ORDER BY generation_id"
        ).fetchall()
        return tuple(
            ArchiveEntry(
                generation_id=row["generation_id"],
                tier=ArchiveTier(row["tier"]),
                checkpoint_path=Path(row["checkpoint_path"]) if row["checkpoint_path"] else None,
                checkpoint_bytes=row["checkpoint_bytes"],
                ever_champion=bool(row["ever_champion"]),
                specialist_score=row["specialist_score"],
                protected=bool(row["protected"]),
                reason=row["reason"],
            )
            for row in rows
        )

    def active_reign(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT * FROM champion_reigns
            WHERE ended_at IS NULL
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row is not None else None

    def start_reign(self, reign: ChampionReign) -> None:
        if not reign.active:
            raise ValueError("start_reign expects an active reign")
        existing = self.active_reign()
        if existing is not None:
            raise RuntimeError(
                f"cannot start Gen{reign.generation_id}; Gen{existing['generation_id']} already has an active reign"
            )
        self._conn.execute(
            """
            INSERT INTO champion_reigns(
                generation_id, started_at, challengers_faced, games_during_reign
            ) VALUES (?, ?, ?, ?)
            """,
            (
                reign.generation_id,
                _iso(reign.started_at),
                reign.challengers_faced,
                reign.games_during_reign,
            ),
        )
        self._conn.commit()

    def end_reign(
        self,
        generation_id: int,
        *,
        ended_at: datetime,
        dethroned_by: int | None,
        replacement_reason: str = "",
    ) -> None:
        cursor = self._conn.execute(
            """
            UPDATE champion_reigns
            SET ended_at=?, dethroned_by=?, replacement_reason=?
            WHERE id=(
                SELECT id FROM champion_reigns
                WHERE generation_id=? AND ended_at IS NULL
                ORDER BY started_at DESC LIMIT 1
            )
            """,
            (_iso(ended_at), dethroned_by, replacement_reason, generation_id),
        )
        if cursor.rowcount != 1:
            self._conn.rollback()
            raise LookupError(f"no active reign found for generation {generation_id}")
        self._conn.commit()

    def record_champion_succession(
        self,
        *,
        outgoing: ArchiveEntry,
        incoming: ArchiveEntry,
        occurred_at: datetime,
        replacement_reason: str,
        evidence: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically close one reign, open the next and persist both bodies.

        Returns ``False`` when the exact succession was already committed. This
        makes a production promotion callback safe to retry after a crash or
        uncertain acknowledgement.
        """
        if outgoing.generation_id == incoming.generation_id:
            raise ValueError("champion succession requires two different generations")
        if outgoing.tier is not ArchiveTier.IMMORTAL or not outgoing.ever_champion or not outgoing.protected:
            raise ValueError("outgoing champion must enter protected IMMORTAL archive")
        if incoming.tier is not ArchiveTier.ACTIVE or not incoming.ever_champion:
            raise ValueError("incoming champion must be ACTIVE and marked ever_champion")

        timestamp = _iso(occurred_at)
        payload = json.dumps(evidence or {}, separators=(",", ":"), sort_keys=True)
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            current = self._conn.execute(
                """
                SELECT * FROM champion_reigns
                WHERE ended_at IS NULL
                ORDER BY started_at DESC, id DESC
                LIMIT 1
                """
            ).fetchone()

            if current is not None and current["generation_id"] == incoming.generation_id:
                self._conn.rollback()
                return False
            if current is None or current["generation_id"] != outgoing.generation_id:
                actual = None if current is None else current["generation_id"]
                raise RuntimeError(
                    f"succession expected active Gen{outgoing.generation_id}, found {actual}"
                )

            cursor = self._conn.execute(
                """
                UPDATE champion_reigns
                SET ended_at=?, dethroned_by=?, replacement_reason=?
                WHERE id=? AND ended_at IS NULL
                """,
                (timestamp, incoming.generation_id, replacement_reason, current["id"]),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("active champion reign changed during succession")

            self._conn.execute(
                """
                INSERT INTO champion_reigns(
                    generation_id, started_at, challengers_faced, games_during_reign
                ) VALUES (?, ?, 0, 0)
                """,
                (incoming.generation_id, timestamp),
            )
            self._upsert_archive_entry_no_commit(outgoing, archived_at=occurred_at)
            self._upsert_archive_entry_no_commit(incoming, last_used_at=occurred_at)
            self._conn.execute(
                """
                INSERT INTO historical_events(
                    occurred_at, kind, generation_id, related_generation_id,
                    text, evidence_json
                ) VALUES (?, 'champion_succession', ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    outgoing.generation_id,
                    incoming.generation_id,
                    f"Gen{outgoing.generation_id} was succeeded by Gen{incoming.generation_id}: {replacement_reason}",
                    payload,
                ),
            )
            self._conn.commit()
            return True
        except Exception:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    def increment_reign_activity(
        self,
        generation_id: int,
        *,
        challengers: int = 0,
        games: int = 0,
    ) -> None:
        if challengers < 0 or games < 0:
            raise ValueError("activity increments must be non-negative")
        cursor = self._conn.execute(
            """
            UPDATE champion_reigns
            SET challengers_faced=challengers_faced + ?,
                games_during_reign=games_during_reign + ?
            WHERE id=(
                SELECT id FROM champion_reigns
                WHERE generation_id=? AND ended_at IS NULL
                ORDER BY started_at DESC LIMIT 1
            )
            """,
            (challengers, games, generation_id),
        )
        if cursor.rowcount != 1:
            self._conn.rollback()
            raise LookupError(f"no active reign found for generation {generation_id}")
        self._conn.commit()

    def record_trait(
        self,
        generation_id: int,
        trait_kind: str,
        trait_key: str,
        score: float,
        *,
        sample_games: int = 0,
        evidence: dict[str, Any] | None = None,
        updated_at: datetime,
    ) -> None:
        if sample_games < 0:
            raise ValueError("sample_games must be non-negative")
        self._conn.execute(
            """
            INSERT INTO generation_traits(
                generation_id, trait_kind, trait_key, score,
                sample_games, evidence_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(generation_id, trait_kind, trait_key) DO UPDATE SET
                score=excluded.score,
                sample_games=excluded.sample_games,
                evidence_json=excluded.evidence_json,
                updated_at=excluded.updated_at
            """,
            (
                generation_id,
                trait_kind,
                trait_key,
                score,
                sample_games,
                json.dumps(evidence or {}, separators=(",", ":"), sort_keys=True),
                _iso(updated_at),
            ),
        )
        self._conn.commit()

    def trait_records(
        self,
        generation_id: int | None = None,
        *,
        trait_kind: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        where: list[str] = []
        values: list[object] = []
        if generation_id is not None:
            where.append("generation_id=?")
            values.append(generation_id)
        if trait_kind is not None:
            where.append("trait_kind=?")
            values.append(trait_kind)
        clause = " WHERE " + " AND ".join(where) if where else ""
        rows = self._conn.execute(
            "SELECT * FROM generation_traits" + clause + " ORDER BY score DESC, sample_games DESC",
            values,
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def record_event(
        self,
        event: HistoricalEvent,
        *,
        evidence: dict[str, Any] | None = None,
    ) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO historical_events(
                occurred_at, kind, generation_id, related_generation_id,
                text, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _iso(event.occurred_at),
                event.kind,
                event.generation_id,
                event.related_generation_id,
                event.text,
                json.dumps(evidence or {}, separators=(",", ":"), sort_keys=True),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def recent_events(self, limit: int = 50) -> tuple[dict[str, Any], ...]:
        if limit <= 0:
            return ()
        rows = self._conn.execute(
            "SELECT * FROM historical_events ORDER BY occurred_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return tuple(dict(row) for row in rows)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("chronicle datetimes must be timezone-aware")
    return value.isoformat()
