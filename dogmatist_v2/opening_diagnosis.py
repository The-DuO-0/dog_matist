from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

from .opening_lab import OpeningBucketSignal, OpeningWeaknessController


@dataclass(frozen=True)
class OpeningDiagnosisRow:
    opening_bucket: str
    pressure: float
    confidence: float
    status: str
    hard_positions: int
    observations: int
    mean_priority: float
    max_priority: float
    mean_severity: float
    max_severity: float
    mean_value_error: float
    max_value_error: float
    last_seen_round: int

    def as_dict(self) -> dict[str, object]:
        return {
            "opening_bucket": self.opening_bucket,
            "pressure": self.pressure,
            "confidence": self.confidence,
            "status": self.status,
            "hard_positions": self.hard_positions,
            "observations": self.observations,
            "mean_priority": self.mean_priority,
            "max_priority": self.max_priority,
            "mean_severity": self.mean_severity,
            "max_severity": self.max_severity,
            "mean_value_error": self.mean_value_error,
            "max_value_error": self.max_value_error,
            "last_seen_round": self.last_seen_round,
        }


@dataclass(frozen=True)
class OpeningDiagnosisReport:
    generation: int
    rows: tuple[OpeningDiagnosisRow, ...]
    focus_openings: tuple[str, ...]
    focus_fraction: float
    observed_generations: tuple[int, ...]
    strength_db: str
    book_moves_injected: bool = False
    novel_openings_allowed: bool = True

    @property
    def evidence_positions(self) -> int:
        return sum(row.hard_positions for row in self.rows)

    @property
    def evidence_observations(self) -> int:
        return sum(row.observations for row in self.rows)

    @property
    def ready(self) -> bool:
        return bool(self.rows)

    @property
    def clean_generation_only(self) -> bool:
        return not self.observed_generations or self.observed_generations == (self.generation,)

    def as_dict(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "ready": self.ready,
            "clean_generation_only": self.clean_generation_only,
            "observed_generations": list(self.observed_generations),
            "evidence_positions": self.evidence_positions,
            "evidence_observations": self.evidence_observations,
            "focus_openings": list(self.focus_openings),
            "focus_fraction": self.focus_fraction,
            "book_moves_injected": self.book_moves_injected,
            "novel_openings_allowed": self.novel_openings_allowed,
            "strength_db": self.strength_db,
            "openings": [row.as_dict() for row in self.rows],
            "interpretation": (
                "Pressure is an internal opening-repair signal, not an Elo score or an external-engine evaluation. "
                "WEAK/WATCH labels require early-position evidence from the requested generation."
            ),
        }

    def format_text(self, *, width: int = 22) -> str:
        title = f"GEN{self.generation} OPENING DIAGNOSIS"
        lines = [title, "=" * len(title)]
        if not self.rows:
            lines.append("No generation-specific early-opening evidence was captured.")
            return "\n".join(lines)
        for row in self.rows:
            name = row.opening_bucket[:width].ljust(width)
            bar_n = max(0, min(10, int(round(row.pressure * 10))))
            bar = ("█" * bar_n).ljust(10, "·")
            lines.append(
                f"{name} {bar} {row.pressure:0.2f}  {row.status:<16} "
                f"evidence={row.observations}"
            )
        lines.extend(
            [
                "",
                "Repair focus: " + (", ".join(self.focus_openings) if self.focus_openings else "broad exploration"),
                f"Opening focus ceiling this plan: {self.focus_fraction:.0%}",
                "Book moves: OFF",
                "Novel exploration: ON",
            ]
        )
        if not self.clean_generation_only:
            lines.append(
                "Note: the Strength DB also contains other generations; this table itself is filtered to the requested generation."
            )
        return "\n".join(lines)


def _status(pressure: float, confidence: float) -> str:
    # Do not call a one-off noisy position a confirmed opening weakness.
    if pressure >= 0.55 and confidence >= 0.25:
        return "WEAK"
    if pressure >= 0.28:
        return "WATCH"
    if confidence >= 0.45:
        return "NO CLEAR WEAKNESS"
    return "LOW EVIDENCE"


def _confidence(hard_positions: int, observations: int) -> float:
    position_term = min(1.0, max(0, hard_positions) / 6.0)
    repeat_term = min(1.0, max(0, observations) / 12.0)
    return min(1.0, 0.55 * position_term + 0.45 * repeat_term)


def diagnose_openings(
    strength_db: str | Path,
    *,
    generation: int,
    limit: int = 12,
    focus_fraction: float = 0.60,
) -> OpeningDiagnosisReport:
    """Read a StrengthStore database without mutating it and diagnose one generation.

    This intentionally reads only `*_opening` rows. A later middlegame blunder in a
    game carrying an opening label therefore cannot by itself make that opening look
    weak. The generation filter is equally important for a long-lived archive: old
    Champion evidence must not be silently attributed to the current throne.
    """

    path = Path(strength_db).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Strength Lab database not found: {path}")
    if int(generation) < 0:
        raise ValueError("generation must be non-negative")
    if limit <= 0:
        raise ValueError("limit must be positive")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hard_positions'"
        ).fetchone()
        if table is None:
            raise RuntimeError("Strength Lab database has no hard_positions table")

        generation_rows = conn.execute(
            """
            SELECT DISTINCT source_generation
            FROM hard_positions
            WHERE source_kind LIKE '%_opening' AND source_generation IS NOT NULL
            ORDER BY source_generation
            """
        ).fetchall()
        observed_generations = tuple(int(row[0]) for row in generation_rows)

        rows = conn.execute(
            """
            SELECT
                opening_bucket,
                COUNT(*) AS hard_positions,
                SUM(times_seen) AS observations,
                AVG(severity * 0.50 + value_error * 0.35 + uncertainty * 0.15) AS mean_priority,
                MAX(severity * 0.50 + value_error * 0.35 + uncertainty * 0.15) AS max_priority,
                AVG(severity) AS mean_severity,
                MAX(severity) AS max_severity,
                AVG(value_error) AS mean_value_error,
                MAX(value_error) AS max_value_error,
                MAX(last_seen_round) AS last_seen_round
            FROM hard_positions
            WHERE source_kind LIKE '%_opening' AND source_generation=?
            GROUP BY opening_bucket
            ORDER BY max_priority DESC, observations DESC, last_seen_round DESC
            LIMIT ?
            """,
            (int(generation), int(limit)),
        ).fetchall()
    finally:
        conn.close()

    hard_stats: list[dict[str, object]] = []
    raw: dict[str, sqlite3.Row] = {}
    for row in rows:
        bucket = str(row["opening_bucket"])
        raw[bucket] = row
        hard_stats.append(
            {
                "opening_bucket": bucket,
                "hard_positions": int(row["hard_positions"] or 0),
                "hard_times_seen": int(row["observations"] or 0),
                "mean_priority": float(row["mean_priority"] or 0.0),
                "max_priority": float(row["max_priority"] or 0.0),
                "last_seen_round": int(row["last_seen_round"] or 0),
            }
        )

    controller = OpeningWeaknessController(focus_fraction=focus_fraction)
    plan = controller.plan(hard_bucket_stats=hard_stats)
    diagnosis_rows: list[OpeningDiagnosisRow] = []
    for signal in plan.signals:
        row = raw.get(signal.opening_bucket)
        if row is None:
            continue
        positions = int(row["hard_positions"] or 0)
        observations = int(row["observations"] or 0)
        confidence = _confidence(positions, observations)
        pressure = float(signal.hard_pressure)
        diagnosis_rows.append(
            OpeningDiagnosisRow(
                opening_bucket=signal.opening_bucket,
                pressure=pressure,
                confidence=confidence,
                status=_status(pressure, confidence),
                hard_positions=positions,
                observations=observations,
                mean_priority=float(row["mean_priority"] or 0.0),
                max_priority=float(row["max_priority"] or 0.0),
                mean_severity=float(row["mean_severity"] or 0.0),
                max_severity=float(row["max_severity"] or 0.0),
                mean_value_error=float(row["mean_value_error"] or 0.0),
                max_value_error=float(row["max_value_error"] or 0.0),
                last_seen_round=int(row["last_seen_round"] or 0),
            )
        )

    return OpeningDiagnosisReport(
        generation=int(generation),
        rows=tuple(diagnosis_rows),
        focus_openings=plan.focus_openings,
        focus_fraction=plan.focus_fraction,
        observed_generations=observed_generations,
        strength_db=str(path),
    )


def reset_copied_strength_state(
    snapshot_state: str | Path,
    *,
    db_name: str = "strength_v2.sqlite3",
) -> tuple[str, ...]:
    """Delete only the copied Strength Lab DB so an opening diagnosis starts clean.

    The path must itself be a `.darwinchess` snapshot directory. The helper refuses
    arbitrary directories and never follows a database path outside that snapshot.
    It removes SQLite WAL/SHM companions as well, returning the paths that existed.
    """

    snapshot = Path(snapshot_state).expanduser().resolve()
    if snapshot.name != ".darwinchess":
        raise ValueError("snapshot_state must end in /.darwinchess")
    if not snapshot.is_dir():
        raise FileNotFoundError(f"copied state directory not found: {snapshot}")
    if not db_name or Path(db_name).name != db_name:
        raise ValueError("db_name must be a simple filename")

    removed: list[str] = []
    for suffix in ("", "-wal", "-shm"):
        candidate = (snapshot / f"{db_name}{suffix}").resolve()
        if candidate.parent != snapshot:
            raise RuntimeError("refusing to remove a Strength DB outside the copied snapshot")
        if candidate.exists():
            if not candidate.is_file():
                raise RuntimeError(f"refusing to remove non-file Strength state: {candidate}")
            candidate.unlink()
            removed.append(str(candidate))
    return tuple(removed)


def write_opening_diagnosis(
    report: OpeningDiagnosisReport,
    destination: str | Path,
) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
