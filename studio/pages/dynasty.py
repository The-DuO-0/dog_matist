from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data import ReadOnlyStore, state_dir


def _parse_dt(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fmt_dt(value: Any) -> str:
    dt = _parse_dt(value)
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _fmt_duration(start: Any, end: Any = None) -> str:
    a = _parse_dt(start)
    if a is None:
        return "—"
    b = _parse_dt(end) if end else datetime.now(timezone.utc)
    if b is None:
        return "—"
    seconds = max(0, int((b - a).total_seconds()))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _short_path(value: Any) -> str:
    if not value:
        return "—"
    return Path(str(value)).name


def _table_item(value: Any) -> QTableWidgetItem:
    return QTableWidgetItem("—" if value is None else str(value))


class DynastyPage(QWidget):
    """Read-only dynasty/lineage view over durable dog_matist state.

    The production SQLite lineage remains the source of truth for old generations.
    If the optional V2 Chronicle database exists, archive/body counts are surfaced
    without loading model checkpoints into RAM.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.store = ReadOnlyStore()
        self.refresh_clock = QTimer(self)
        self.refresh_clock.setInterval(5000)
        self.refresh_clock.timeout.connect(self.refresh)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        head = QHBoxLayout()
        title = QLabel("Dynasty Archive")
        title.setObjectName("SectionTitle")
        head.addWidget(title)
        head.addStretch()
        self.source_badge = QLabel("READ ONLY")
        self.source_badge.setObjectName("RunBadge")
        head.addWidget(self.source_badge)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        head.addWidget(self.refresh_btn)
        root.addLayout(head)

        note = QLabel(
            "Every generation stays visible in lineage history. Champion/reign data is reconstructed from the durable "
            "production SQLite record; archived checkpoints are never loaded merely to render this page."
        )
        note.setObjectName("InfoNote")
        note.setWordWrap(True)
        root.addWidget(note)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        self.current_champion = self._card(cards, 0, "CURRENT CHAMPION")
        self.dynasties = self._card(cards, 1, "CHAMPION DYNASTIES")
        self.generations_count = self._card(cards, 2, "GENERATIONS RECORDED")
        self.archive_count = self._card(cards, 3, "ARCHIVE BODIES")
        root.addLayout(cards)

        dynasty_panel = QFrame()
        dynasty_panel.setObjectName("Panel")
        dl = QVBoxLayout(dynasty_panel)
        dl.setContentsMargins(14, 12, 14, 14)
        dl.addWidget(QLabel("CHAMPION REIGNS"))
        self.reigns = QTableWidget(0, 10)
        self.reigns.setHorizontalHeaderLabels(
            ["Gen", "Born", "Reign start", "Reign end", "Reign", "Challengers", "Arena games", "Arena score", "Status", "Checkpoint"]
        )
        self._prepare_table(self.reigns, stretch_column=9)
        self.reigns.setMinimumHeight(190)
        dl.addWidget(self.reigns)
        root.addWidget(dynasty_panel)

        gens_panel = QFrame()
        gens_panel.setObjectName("Panel")
        gl = QVBoxLayout(gens_panel)
        gl.setContentsMargins(14, 12, 14, 14)
        gl.addWidget(QLabel("GENERATION LINEAGE"))
        self.generations = QTableWidget(0, 10)
        self.generations.setHorizontalHeaderLabels(
            ["Gen", "Parent", "Status", "Born", "Lifetime", "Train loss", "Arena", "W/D/L", "Games seen", "Notes"]
        )
        self._prepare_table(self.generations, stretch_column=9)
        self.generations.setMinimumHeight(240)
        gl.addWidget(self.generations)
        root.addWidget(gens_panel, 1)

        events_panel = QFrame()
        events_panel.setObjectName("Panel")
        el = QVBoxLayout(events_panel)
        el.setContentsMargins(14, 12, 14, 14)
        el.addWidget(QLabel("RECENT CHRONICLE EVENTS"))
        self.events = QTableWidget(0, 4)
        self.events.setHorizontalHeaderLabels(["Time", "Kind", "Generation", "Event"])
        self._prepare_table(self.events, stretch_column=3)
        self.events.setMinimumHeight(170)
        el.addWidget(self.events)
        root.addWidget(events_panel)

        self.status_note = QLabel("Waiting for lineage data…")
        self.status_note.setObjectName("Subtle")
        self.status_note.setWordWrap(True)
        root.addWidget(self.status_note)

        self.refresh()
        self.refresh_clock.start()

    @staticmethod
    def _prepare_table(table: QTableWidget, *, stretch_column: int) -> None:
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(stretch_column, QHeaderView.Stretch)

    @staticmethod
    def _card(layout: QGridLayout, column: int, label: str) -> QLabel:
        card = QFrame()
        card.setObjectName("Card")
        box = QVBoxLayout(card)
        box.setContentsMargins(16, 12, 16, 12)
        key = QLabel(label)
        key.setObjectName("Subtle")
        value = QLabel("—")
        value.setObjectName("CardValue")
        box.addWidget(key)
        box.addWidget(value)
        layout.addWidget(card, 0, column)
        return value

    def _main_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.store.path.exists():
            return [], [], []
        generations = self.store.generations(limit=5000)
        insights_table = self.store.find_table("insights")
        arena_table = self.store.find_table("arena_matches")
        insights = self.store.rows(insights_table, 2000, descending=False) if insights_table else []
        arena = self.store.rows(arena_table, 20000, descending=False) if arena_table else []
        return generations, insights, arena

    @staticmethod
    def _promotion_times(insights: list[dict[str, Any]]) -> dict[int, str]:
        out: dict[int, str] = {}
        for row in insights:
            if str(row.get("kind") or "").lower() != "promotion":
                continue
            try:
                generation = int(row.get("generation"))
            except (TypeError, ValueError):
                continue
            out[generation] = str(row.get("created_at") or "")
        return out

    @staticmethod
    def _arena_summary(arena: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
        challengers: dict[int, set[int]] = {}
        games: dict[int, int] = {}
        for row in arena:
            try:
                champion = int(row.get("champion_generation"))
                challenger = int(row.get("challenger_generation"))
            except (TypeError, ValueError):
                continue
            challengers.setdefault(champion, set()).add(challenger)
            games[champion] = games.get(champion, 0) + 1
        return {gen: (len(challengers.get(gen, set())), games.get(gen, 0)) for gen in set(challengers) | set(games)}

    @staticmethod
    def _archive_body_count() -> tuple[int, str]:
        chronicle = state_dir() / "chronicle_v2.sqlite3"
        if not chronicle.exists():
            return 0, "Chronicle mirror not created yet; lineage is still durable in the production database."
        try:
            uri = f"file:{chronicle.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1.0) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM generation_archive WHERE checkpoint_path IS NOT NULL"
                ).fetchone()
            return int(row[0] if row else 0), f"Chronicle: {chronicle.name}"
        except sqlite3.Error as exc:
            return 0, f"Chronicle read warning: {exc}"

    def refresh(self) -> None:
        try:
            rows, insights, arena = self._main_rows()
        except sqlite3.Error as exc:
            self.status_note.setText(f"Could not read lineage database: {exc}")
            return

        rows = sorted(rows, key=lambda row: int(row.get("id", -1)))
        champions = [row for row in rows if str(row.get("status") or "").lower() in {"champion", "retired"}]
        promotions = self._promotion_times(insights)
        activity = self._arena_summary(arena)
        archive_bodies, archive_note = self._archive_body_count()

        current = next((row for row in reversed(rows) if str(row.get("status") or "").lower() == "champion"), None)
        self.current_champion.setText(f"Gen{current['id']}" if current else "—")
        self.dynasties.setText(str(len(champions)))
        self.generations_count.setText(str(len(rows)))
        self.archive_count.setText(str(archive_bodies))

        starts: list[str] = []
        for row in champions:
            gid = int(row.get("id", -1))
            starts.append(promotions.get(gid) or str(row.get("created_at") or ""))

        self.reigns.setRowCount(len(champions))
        for i, row in enumerate(champions):
            gid = int(row.get("id", -1))
            start = starts[i]
            end = starts[i + 1] if i + 1 < len(starts) else None
            challengers, games = activity.get(gid, (0, 0))
            values = [
                f"Gen{gid}",
                _fmt_dt(row.get("created_at")),
                _fmt_dt(start),
                _fmt_dt(end) if end else "ACTIVE",
                _fmt_duration(start, end),
                challengers,
                games,
                "—" if row.get("arena_score") is None else f"{float(row['arena_score']):.3f}",
                str(row.get("status") or ""),
                _short_path(row.get("checkpoint_path")),
            ]
            for col, value in enumerate(values):
                self.reigns.setItem(i, col, _table_item(value))

        display_rows = list(reversed(rows[-300:]))
        self.generations.setRowCount(len(display_rows))
        now = datetime.now(timezone.utc)
        for i, row in enumerate(display_rows):
            born = _parse_dt(row.get("created_at"))
            lifetime = _fmt_duration(row.get("created_at"), now.isoformat()) if born else "—"
            wins = row.get("arena_wins")
            draws = row.get("arena_draws")
            losses = row.get("arena_losses")
            wdl = "—" if wins is None else f"{wins}/{draws or 0}/{losses or 0}"
            arena_score = "—" if row.get("arena_score") is None else f"{float(row['arena_score']):.3f}"
            loss = "—" if row.get("training_loss") is None else f"{float(row['training_loss']):.4f}"
            values = [
                row.get("id"),
                row.get("parent_id"),
                row.get("status"),
                _fmt_dt(row.get("created_at")),
                lifetime,
                loss,
                arena_score,
                wdl,
                row.get("games_seen", 0),
                row.get("notes") or "",
            ]
            for col, value in enumerate(values):
                self.generations.setItem(i, col, _table_item(value))

        recent = list(reversed(insights[-120:]))
        self.events.setRowCount(len(recent))
        for i, row in enumerate(recent):
            values = [
                _fmt_dt(row.get("created_at")),
                row.get("kind"),
                row.get("generation"),
                row.get("text") or "",
            ]
            for col, value in enumerate(values):
                self.events.setItem(i, col, _table_item(value))

        self.source_badge.setText("READ ONLY · LIVE REFRESH")
        self.status_note.setText(
            f"Source: {self.store.path} · {archive_note} · No checkpoint was loaded to render this page."
        )
