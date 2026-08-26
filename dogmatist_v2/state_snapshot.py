from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable


@dataclass(frozen=True)
class SnapshotCheckpoint:
    generation: int
    source: str
    copied: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class StateSnapshotManifest:
    created_at: str
    source_root: str
    snapshot_root: str
    database: str
    champion_generation: int | None
    checkpoints: tuple[SnapshotCheckpoint, ...]
    strength_store_copied: bool

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checkpoints"] = [asdict(row) for row in self.checkpoints]
        return payload


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sqlite_backup(source: Path, destination: Path) -> None:
    """Take a transactionally consistent read-only SQLite backup, including WAL."""

    source_uri = f"file:{source.resolve()}?mode=ro"
    src = sqlite3.connect(source_uri, uri=True, timeout=30.0)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()


def _integrity(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    if row is None or str(row[0]).lower() != "ok":
        raise RuntimeError(f"snapshot database integrity_check failed: {row}")


def create_validation_snapshot(
    source_root: str | Path,
    snapshot_root: str | Path,
    *,
    db_name: str = "darwinchess.sqlite3",
    strength_db_name: str = "strength_v2.sqlite3",
    include_statuses: Iterable[str] | None = None,
) -> StateSnapshotManifest:
    """Clone live dog_matist state without allowing copied DB paths back to live.

    The source database is opened SQLite `mode=ro` and copied with the backup API,
    so a WAL-mode database can be snapshotted while Studio/status readers exist.
    Every generation checkpoint selected for the snapshot is copied underneath the
    destination root, then the *copied* database is rewritten to point only at
    those copied files. The source database/checkpoints are never modified.

    By default every checkpoint referenced by ``generations`` is copied. A caller
    may restrict `include_statuses` for a smaller validation fixture, but any row
    left outside the snapshot has its checkpoint path cleared to a non-live
    sentinel so accidental access cannot fall through to the source tree.
    """

    source_root = Path(source_root).expanduser().resolve()
    snapshot_root = Path(snapshot_root).expanduser().resolve()
    if source_root == snapshot_root:
        raise ValueError("snapshot_root must differ from source_root")
    if _inside(snapshot_root, source_root):
        raise ValueError("snapshot_root must not live inside the live state root")
    if snapshot_root.exists() and any(snapshot_root.iterdir()):
        raise FileExistsError(f"snapshot destination is not empty: {snapshot_root}")

    source_db = source_root / db_name
    if not source_db.is_file():
        raise FileNotFoundError(f"live database not found: {source_db}")

    snapshot_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = snapshot_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_db = snapshot_root / db_name
    _sqlite_backup(source_db, snapshot_db)

    statuses = set(str(x) for x in include_statuses) if include_statuses is not None else None
    copied_rows: list[SnapshotCheckpoint] = []
    conn = sqlite3.connect(snapshot_db)
    conn.row_factory = sqlite3.Row
    try:
        _integrity(conn)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(generations)").fetchall()}
        if not {"id", "checkpoint_path", "status"}.issubset(columns):
            raise RuntimeError("snapshot database does not contain the expected generations schema")

        rows = conn.execute(
            "SELECT id, checkpoint_path, status FROM generations ORDER BY id"
        ).fetchall()
        for row in rows:
            generation = int(row["id"])
            status = str(row["status"])
            raw_path = str(row["checkpoint_path"] or "")
            if statuses is not None and status not in statuses:
                conn.execute(
                    "UPDATE generations SET checkpoint_path=? WHERE id=?",
                    (str(snapshot_root / "unavailable" / f"generation_{generation}.pt"), generation),
                )
                continue
            if not raw_path:
                raise RuntimeError(f"generation {generation} has an empty checkpoint path")
            source_checkpoint = Path(raw_path).expanduser()
            if not source_checkpoint.is_absolute():
                source_checkpoint = source_root / source_checkpoint
            source_checkpoint = source_checkpoint.resolve()
            if not source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"checkpoint referenced by generation {generation} is missing: {source_checkpoint}"
                )
            suffix = "".join(source_checkpoint.suffixes) or ".pt"
            copied_checkpoint = checkpoint_dir / f"generation_{generation}{suffix}"
            shutil.copy2(source_checkpoint, copied_checkpoint)
            copied_checkpoint = copied_checkpoint.resolve()
            conn.execute(
                "UPDATE generations SET checkpoint_path=? WHERE id=?",
                (str(copied_checkpoint), generation),
            )
            copied_rows.append(
                SnapshotCheckpoint(
                    generation=generation,
                    source=str(source_checkpoint),
                    copied=str(copied_checkpoint),
                    bytes=copied_checkpoint.stat().st_size,
                    sha256=_sha256(copied_checkpoint),
                )
            )

        conn.commit()
        _integrity(conn)
        champion = conn.execute(
            "SELECT id FROM generations WHERE status='champion' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        champion_generation = int(champion[0]) if champion is not None else None
    finally:
        conn.close()

    source_strength = source_root / strength_db_name
    strength_copied = False
    if source_strength.is_file():
        _sqlite_backup(source_strength, snapshot_root / strength_db_name)
        strength_copied = True

    manifest = StateSnapshotManifest(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_root=str(source_root),
        snapshot_root=str(snapshot_root),
        database=str(snapshot_db.resolve()),
        champion_generation=champion_generation,
        checkpoints=tuple(copied_rows),
        strength_store_copied=strength_copied,
    )
    manifest_path = snapshot_root / "SNAPSHOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    validate_snapshot_isolation(manifest)
    return manifest


def validate_snapshot_isolation(manifest: StateSnapshotManifest) -> None:
    """Fail if any copied generation can still resolve to the live source tree."""

    source_root = Path(manifest.source_root).resolve()
    snapshot_root = Path(manifest.snapshot_root).resolve()
    database = Path(manifest.database).resolve()
    if not _inside(database, snapshot_root):
        raise RuntimeError("snapshot database escaped the snapshot root")

    conn = sqlite3.connect(database)
    try:
        _integrity(conn)
        rows = conn.execute("SELECT id, checkpoint_path FROM generations").fetchall()
        for generation, raw_path in rows:
            path = Path(str(raw_path)).expanduser().resolve()
            if _inside(path, source_root):
                raise RuntimeError(
                    f"snapshot generation {generation} still points into live state: {path}"
                )
            if "unavailable" not in path.parts and not _inside(path, snapshot_root):
                raise RuntimeError(
                    f"snapshot generation {generation} checkpoint escaped snapshot root: {path}"
                )
    finally:
        conn.close()
