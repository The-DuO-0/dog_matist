from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import random
import sqlite3
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    parent_id INTEGER,
    checkpoint_path TEXT NOT NULL,
    status TEXT NOT NULL,
    games_seen INTEGER NOT NULL DEFAULT 0,
    examples_seen INTEGER NOT NULL DEFAULT 0,
    arena_score REAL,
    arena_wins INTEGER,
    arena_draws INTEGER,
    arena_losses INTEGER,
    training_loss REAL,
    genome_json TEXT NOT NULL DEFAULT '{}',
    notes TEXT,
    FOREIGN KEY(parent_id) REFERENCES generations(id)
);

CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    generation INTEGER,
    white_agent TEXT NOT NULL,
    black_agent TEXT NOT NULL,
    result TEXT NOT NULL,
    termination TEXT,
    plies INTEGER NOT NULL,
    pgn TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    ply INTEGER NOT NULL,
    fen TEXT NOT NULL,
    move_uci TEXT NOT NULL,
    played_move_uci TEXT,
    search_score_cp REAL,
    best_score_cp REAL,
    value_target REAL NOT NULL,
    policy_weight REAL NOT NULL DEFAULT 1.0,
    priority REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(game_id) REFERENCES games(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_examples_game ON examples(game_id);
CREATE INDEX IF NOT EXISTS idx_examples_created ON examples(created_at);
CREATE INDEX IF NOT EXISTS idx_games_created ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_games_generation ON games(generation);

CREATE TABLE IF NOT EXISTS arena_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    challenger_generation INTEGER NOT NULL,
    champion_generation INTEGER NOT NULL,
    game_id TEXT,
    challenger_color TEXT NOT NULL,
    result_for_challenger REAL NOT NULL,
    FOREIGN KEY(game_id) REFERENCES games(id)
);

CREATE TABLE IF NOT EXISTS insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    generation INTEGER,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    generation INTEGER,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS population_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    champion_generation INTEGER NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS population_members (
    round_id INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    role TEXT NOT NULL,
    status TEXT NOT NULL,
    focus_json TEXT NOT NULL DEFAULT '[]',
    league_score REAL,
    rating REAL,
    PRIMARY KEY(round_id, generation),
    FOREIGN KEY(round_id) REFERENCES population_rounds(id),
    FOREIGN KEY(generation) REFERENCES generations(id)
);

CREATE TABLE IF NOT EXISTS league_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    round_id INTEGER NOT NULL,
    white_generation INTEGER NOT NULL,
    black_generation INTEGER NOT NULL,
    game_id TEXT,
    opening_name TEXT,
    result TEXT NOT NULL,
    white_score REAL NOT NULL,
    FOREIGN KEY(round_id) REFERENCES population_rounds(id),
    FOREIGN KEY(game_id) REFERENCES games(id)
);
CREATE INDEX IF NOT EXISTS idx_league_round ON league_matches(round_id);

CREATE TABLE IF NOT EXISTS specialists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    generation INTEGER NOT NULL,
    opening_name TEXT NOT NULL,
    score REAL NOT NULL,
    games INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(generation, opening_name),
    FOREIGN KEY(generation) REFERENCES generations(id)
);
CREATE INDEX IF NOT EXISTS idx_specialists_active ON specialists(active, opening_name);
"""


@dataclass
class ReplayExample:
    fen: str
    move_uci: str  # policy/search target
    value_target: float
    policy_weight: float = 1.0
    priority: float = 1.0
    played_move_uci: str | None = None
    search_score_cp: float | None = None
    best_score_cp: float | None = None


class MemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        # Keep lifetime databases forward-compatible as the agent evolves.
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(examples)").fetchall()}
        additions = {
            "played_move_uci": "TEXT",
            "search_score_cp": "REAL",
            "best_score_cp": "REAL",
            "opening_name": "TEXT",
            "opening_family": "TEXT",
            "origin_generation": "INTEGER",
        }
        for name, type_sql in additions.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE examples ADD COLUMN {name} {type_sql}")
        gcols = {row[1] for row in self.conn.execute("PRAGMA table_info(generations)").fetchall()}
        if "genome_json" not in gcols:
            self.conn.execute("ALTER TABLE generations ADD COLUMN genome_json TEXT NOT NULL DEFAULT '{}'")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def add_generation(
        self,
        generation_id: int,
        parent_id: int | None,
        checkpoint_path: str,
        status: str,
        *,
        notes: str = "",
        training_loss: float | None = None,
        genome: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO generations
               (id,created_at,parent_id,checkpoint_path,status,notes,training_loss,genome_json)
               VALUES(?,?,?,?,?,?,?,?)""",
            (generation_id, utc_now(), parent_id, checkpoint_path, status, notes, training_loss,
             json.dumps(genome or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def update_generation(self, generation_id: int, **fields: Any) -> None:
        allowed = {
            "status", "games_seen", "examples_seen", "arena_score", "arena_wins",
            "arena_draws", "arena_losses", "training_loss", "notes", "checkpoint_path", "genome_json",
        }
        pairs = [(k, v) for k, v in fields.items() if k in allowed]
        if not pairs:
            return
        sql = "UPDATE generations SET " + ",".join(f"{k}=?" for k, _ in pairs) + " WHERE id=?"
        self.conn.execute(sql, [v for _, v in pairs] + [generation_id])
        self.conn.commit()

    def get_generation(self, generation_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM generations WHERE id=?", (generation_id,)).fetchone()

    def promote_generation(self, old_champion_id: int, new_champion_id: int, new_checkpoint_path: str) -> None:
        """Atomically switch active champion in SQLite after checkpoint is durable."""
        with self.conn:
            self.conn.execute("UPDATE generations SET status='retired' WHERE id=? AND status='champion'", (old_champion_id,))
            self.conn.execute(
                "UPDATE generations SET status='champion', checkpoint_path=? WHERE id=?",
                (new_checkpoint_path, new_champion_id),
            )


    def champion_generation(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM generations WHERE status='champion' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def next_generation_id(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(id),-1)+1 AS n FROM generations").fetchone()
        return int(row["n"])

    def add_game(
        self,
        *,
        source: str,
        generation: int | None,
        white_agent: str,
        black_agent: str,
        result: str,
        termination: str,
        pgn: str,
        plies: int,
        examples: Iterable[ReplayExample] = (),
        metadata: dict[str, Any] | None = None,
        game_id: str | None = None,
    ) -> str:
        gid = game_id or str(uuid.uuid4())
        created = utc_now()
        with self.conn:
            self.conn.execute(
                """INSERT INTO games
                   (id,created_at,source,generation,white_agent,black_agent,result,termination,plies,pgn,metadata_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    gid, created, source, generation, white_agent, black_agent, result,
                    termination, plies, pgn, json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            self.conn.executemany(
                """INSERT INTO examples
                   (game_id,ply,fen,move_uci,played_move_uci,search_score_cp,best_score_cp,value_target,policy_weight,priority,created_at,opening_name,opening_family,origin_generation)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (gid, ply, ex.fen, ex.move_uci, ex.played_move_uci, ex.search_score_cp, ex.best_score_cp,
                     ex.value_target, ex.policy_weight, ex.priority, created,
                     (metadata or {}).get("opening_name"), (metadata or {}).get("opening_family"), generation)
                    for ply, ex in enumerate(examples)
                ],
            )
        return gid

    def add_arena_match(
        self,
        challenger_generation: int,
        champion_generation: int,
        game_id: str,
        challenger_color: str,
        result_for_challenger: float,
    ) -> None:
        self.conn.execute(
            """INSERT INTO arena_matches
               (created_at,challenger_generation,champion_generation,game_id,challenger_color,result_for_challenger)
               VALUES(?,?,?,?,?,?)""",
            (utc_now(), challenger_generation, champion_generation, game_id, challenger_color, result_for_challenger),
        )
        self.conn.commit()

    def add_insight(self, generation: int | None, kind: str, text: str, evidence: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO insights(created_at,generation,kind,text,evidence_json) VALUES(?,?,?,?,?)",
            (utc_now(), generation, kind, text, json.dumps(evidence or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def add_metric(self, generation: int | None, name: str, value: float, context: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO metrics(created_at,generation,name,value,context_json) VALUES(?,?,?,?,?)",
            (utc_now(), generation, name, float(value), json.dumps(context or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def start_population_round(self, champion_generation: int, metadata: dict[str, Any] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO population_rounds(created_at,champion_generation,status,metadata_json) VALUES(?,?,?,?)",
            (utc_now(), int(champion_generation), "running", json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_population_round(self, round_id: int, status: str = "complete") -> None:
        self.conn.execute("UPDATE population_rounds SET status=? WHERE id=?", (status, int(round_id)))
        self.conn.commit()

    def add_population_member(self, round_id: int, generation: int, role: str, focus: Iterable[str] = ()) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO population_members
               (round_id,generation,role,status,focus_json) VALUES(?,?,?,?,?)""",
            (int(round_id), int(generation), str(role), "candidate", json.dumps(list(focus), ensure_ascii=False)),
        )
        self.conn.commit()

    def update_population_member(self, round_id: int, generation: int, **fields: Any) -> None:
        allowed = {"status", "league_score", "rating", "role", "focus_json"}
        pairs = [(k, v) for k, v in fields.items() if k in allowed]
        if not pairs:
            return
        sql = "UPDATE population_members SET " + ",".join(f"{k}=?" for k, _ in pairs) + " WHERE round_id=? AND generation=?"
        self.conn.execute(sql, [v for _, v in pairs] + [int(round_id), int(generation)])
        self.conn.commit()

    def add_league_match(
        self, round_id: int, white_generation: int, black_generation: int, game_id: str,
        opening_name: str, result: str, white_score: float,
    ) -> None:
        self.conn.execute(
            """INSERT INTO league_matches
               (created_at,round_id,white_generation,black_generation,game_id,opening_name,result,white_score)
               VALUES(?,?,?,?,?,?,?,?)""",
            (utc_now(), int(round_id), int(white_generation), int(black_generation), game_id,
             str(opening_name), str(result), float(white_score)),
        )
        self.conn.commit()

    def upsert_specialist(
        self, generation: int, opening_name: str, score: float, games: int, evidence: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO specialists(created_at,generation,opening_name,score,games,active,evidence_json)
               VALUES(?,?,?,?,?,1,?)
               ON CONFLICT(generation,opening_name) DO UPDATE SET
                 score=excluded.score,games=excluded.games,active=1,evidence_json=excluded.evidence_json""",
            (utc_now(), int(generation), str(opening_name), float(score), int(games),
             json.dumps(evidence or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def recent_population_round(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM population_rounds ORDER BY id DESC LIMIT 1"
        ).fetchone()

    def population_members(self, round_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT pm.*, g.checkpoint_path, g.training_loss, g.arena_score
               FROM population_members pm JOIN generations g ON g.id=pm.generation
               WHERE pm.round_id=? ORDER BY COALESCE(pm.rating,0) DESC, pm.generation ASC""",
            (int(round_id),),
        ).fetchall()

    def active_specialists(self, limit: int = 12) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT * FROM specialists WHERE active=1
               ORDER BY score DESC, games DESC, id DESC LIMIT ?""", (int(limit),)
        ).fetchall()

    def league_opening_scores(self, round_id: int, generation: int) -> dict[str, tuple[float, int]]:
        rows = self.conn.execute(
            """SELECT opening_name, white_generation, black_generation, white_score
               FROM league_matches WHERE round_id=? AND (white_generation=? OR black_generation=?)""",
            (int(round_id), int(generation), int(generation)),
        ).fetchall()
        acc: dict[str, list[float]] = {}
        for row in rows:
            score = float(row["white_score"]) if int(row["white_generation"]) == int(generation) else 1.0 - float(row["white_score"])
            acc.setdefault(str(row["opening_name"] or "Unknown"), []).append(score)
        return {name: (sum(vals) / len(vals), len(vals)) for name, vals in acc.items()}

    def recent_opening_counts(self, limit_games: int = 500) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT metadata_json FROM games WHERE source IN ('selfplay','specialist_selfplay') ORDER BY created_at DESC LIMIT ?",
            (int(limit_games),),
        ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            try:
                name = json.loads(row["metadata_json"] or "{}").get("opening_name")
            except (TypeError, json.JSONDecodeError):
                name = None
            if name:
                counts[str(name)] = counts.get(str(name), 0) + 1
        return counts

    def count_examples(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM examples").fetchone()["n"])

    def count_games(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()["n"])

    def replay_sample(
        self,
        batch_size: int,
        recent_fraction: float = 0.35,
        *,
        opening_names: Iterable[str] | None = None,
        opening_fraction: float = 0.0,
        generations: Iterable[int] | None = None,
    ) -> list[sqlite3.Row]:
        """Sample lifetime replay with optional specialist/curriculum focus.

        The ordinary path stays O(batch) and avoids ORDER BY RANDOM over the full
        replay table. Focused rows are drawn first from indexed-ish metadata columns
        added in v2, then the remainder uses the original recent/random mixture.
        """
        total = self.count_examples()
        if total == 0:
            return []
        batch_size = max(1, int(batch_size))
        names = [str(x) for x in (opening_names or []) if str(x)]
        gens = [int(x) for x in (generations or [])]
        focus_take = min(batch_size, int(round(batch_size * max(0.0, min(1.0, opening_fraction))))) if names else 0
        rows: list[sqlite3.Row] = []
        seen: set[int] = set()

        if focus_take:
            name_q = ",".join("?" for _ in names)
            clauses = [f"opening_name IN ({name_q})"]
            args: list[Any] = list(names)
            if gens:
                gen_q = ",".join("?" for _ in gens)
                clauses.append(f"origin_generation IN ({gen_q})")
                args.extend(gens)
            sql = (
                "SELECT * FROM examples WHERE " + " AND ".join(clauses) +
                " ORDER BY id DESC LIMIT ?"
            )
            args.append(max(focus_take * 12, 256))
            pool = self.conn.execute(sql, args).fetchall()
            if pool:
                chosen = random.sample(pool, min(focus_take, len(pool)))
                rows.extend(chosen)
                seen.update(int(r["id"]) for r in chosen)

        remaining = batch_size - len(rows)
        if remaining <= 0:
            return rows[:batch_size]
        recent_n = min(total, max(remaining * 8, 1000))
        recent_take = min(remaining, int(round(remaining * recent_fraction)))
        random_take = remaining - recent_take
        if recent_take:
            pool = self.conn.execute(
                "SELECT * FROM (SELECT * FROM examples ORDER BY id DESC LIMIT ?) ORDER BY RANDOM() LIMIT ?",
                (recent_n, recent_take * 2),
            ).fetchall()
            for row in pool:
                if int(row["id"]) not in seen:
                    rows.append(row); seen.add(int(row["id"]))
                    if len(rows) >= batch_size - random_take:
                        break
        if random_take and len(rows) < batch_size:
            max_id = int(self.conn.execute("SELECT MAX(id) AS m FROM examples").fetchone()["m"] or 0)
            attempts = 0
            while len(rows) < batch_size and attempts < random_take * 16 + 30:
                attempts += 1
                probe = random.randint(1, max_id)
                row = self.conn.execute("SELECT * FROM examples WHERE id>=? ORDER BY id LIMIT 1", (probe,)).fetchone()
                if row is not None and int(row["id"]) not in seen:
                    seen.add(int(row["id"])); rows.append(row)
        if len(rows) < batch_size:
            extra = self.conn.execute("SELECT * FROM examples ORDER BY id DESC LIMIT ?", (batch_size * 2,)).fetchall()
            for row in extra:
                if int(row["id"]) not in seen:
                    rows.append(row); seen.add(int(row["id"]))
                    if len(rows) >= batch_size:
                        break
        return rows[:batch_size]

    def recent_games(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM games ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()

    def recent_insights(self, limit: int = 8) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM insights ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def prune_replay(self, capacity: int) -> int:
        count = self.count_examples()
        if count <= capacity:
            return 0
        remove = count - capacity
        cutoff = self.conn.execute(
            "SELECT id FROM examples ORDER BY id LIMIT 1 OFFSET ?", (remove - 1,)
        ).fetchone()
        if cutoff is None:
            return 0
        with self.conn:
            self.conn.execute("DELETE FROM examples WHERE id<=?", (int(cutoff["id"]),))
        return remove

    def status_snapshot(self) -> dict[str, Any]:
        champion = self.champion_generation()
        recent = self.conn.execute(
            "SELECT result, COUNT(*) AS n FROM games GROUP BY result"
        ).fetchall()
        return {
            "games": self.count_games(),
            "examples": self.count_examples(),
            "champion": dict(champion) if champion else None,
            "results": {r["result"]: int(r["n"]) for r in recent},
            "insights": [dict(r) for r in self.recent_insights(5)],
        }
