from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path
from typing import Any


def state_dir() -> Path:
    override = os.environ.get("DARWINCHESS_HOME") or os.environ.get("DARWINCHESS_STATE_DIR")
    return Path(override).expanduser() if override else Path.home() / ".darwinchess"


def db_path() -> Path:
    base = state_dir()
    preferred = base / "darwinchess.sqlite3"
    if preferred.exists():
        return preferred
    matches = sorted(base.glob("*.sqlite*")) if base.exists() else []
    return matches[0] if matches else preferred


def exports_dir() -> Path:
    return state_dir() / "exports"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class ReadOnlyStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or db_path()

    def connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=1.5)
        conn.row_factory = sqlite3.Row
        return conn

    def tables(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.connect() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return [r[0] for r in rows if not str(r[0]).startswith("sqlite_")]

    def columns(self, table: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(f'PRAGMA table_info("{table.replace(chr(34), chr(34)*2)}")').fetchall()
        return [str(r[1]) for r in rows]

    def count(self, table: str) -> int:
        safe = table.replace('"', '""')
        with self.connect() as conn:
            return int(conn.execute(f'SELECT COUNT(*) FROM "{safe}"').fetchone()[0])

    def rows(self, table: str, limit: int = 100, descending: bool = True) -> list[dict[str, Any]]:
        safe = table.replace('"', '""')
        cols = self.columns(table)
        order = ""
        candidates = [c for c in cols if c.lower() in {"id", "generation", "generation_id", "created_at", "timestamp"}]
        if candidates:
            order = f' ORDER BY "{candidates[0].replace(chr(34), chr(34)*2)}" {"DESC" if descending else "ASC"}'
        with self.connect() as conn:
            result = conn.execute(f'SELECT * FROM "{safe}"{order} LIMIT ?', (limit,)).fetchall()
        return [dict(r) for r in result]

    def find_table(self, *needles: str) -> str | None:
        tables = self.tables()
        for needle in needles:
            for table in tables:
                if needle.lower() == table.lower():
                    return table
        for needle in needles:
            for table in tables:
                if needle.lower() in table.lower():
                    return table
        return None

    def generations(self, limit: int = 300) -> list[dict[str, Any]]:
        table = self.find_table("generations", "generation", "lineage")
        return self.rows(table, limit, descending=False) if table else []

    def metrics(self, limit: int = 1000) -> list[dict[str, Any]]:
        table = self.find_table("metrics", "metric")
        return self.rows(table, limit, descending=False) if table else []

    def overview_counts(self) -> dict[str, int]:
        out = {}
        for table in self.tables():
            try:
                out[table] = self.count(table)
            except sqlite3.Error:
                pass
        return out


def first_value(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    lower = {k.lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lower and lower[name.lower()] is not None:
            return lower[name.lower()]
    for name in names:
        for key, value in lower.items():
            if name.lower() in key and value is not None:
                return value
    return default


def numeric_series(rows: list[dict[str, Any]], x_names: tuple[str, ...], y_names: tuple[str, ...]):
    xs, ys = [], []
    for i, row in enumerate(rows):
        x = first_value(row, *x_names, default=i)
        y = first_value(row, *y_names)
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        xs.append(xf)
        ys.append(yf)
    return xs, ys


def named_metric_series(rows: list[dict[str, Any]], metric_needles: tuple[str, ...]):
    """Read key/value style metric tables: name|metric + value + generation|step."""
    xs, ys = [], []
    for i, row in enumerate(rows):
        name = first_value(row, "name", "metric", "key", "kind", default="")
        if not any(n.lower() in str(name).lower() for n in metric_needles):
            continue
        x = first_value(row, "generation", "step", "iteration", "id", default=i)
        y = first_value(row, "value", "metric_value", "score")
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        xs.append(xf)
        ys.append(yf)
    return xs, ys
