from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import QElapsedTimer, QTimer, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data import ReadOnlyStore, numeric_series
from ..widgets.charts import LineChart


def _hms(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


class EvolutionPage(QWidget):
    """Production V2 Evolution view built against the uploaded Studio source."""

    _loss_re = re.compile(r"^(\d+)/(\d+).*?loss=([0-9eE+.-]+)")
    FLOW = [
        ("SELF-PLAY", "self-play"),
        ("STRENGTH LAB", "strength-lab"),
        ("POPULATION TRAIN", "population-train"),
        ("LEAGUE", "league"),
        ("ARENA", "arena"),
        ("STRENGTH GUARD", "strength-guard"),
        ("PROMOTE / REJECT", "promotion"),
        ("ARCHIVE", "archive"),
    ]

    def __init__(self, process, parent=None):
        super().__init__(parent)
        self.process = process
        self.elapsed = QElapsedTimer()
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)
        self.store = ReadOnlyStore()
        self.live_loss_x: list[float] = []
        self.live_loss_y: list[float] = []
        self.latest_ui: dict[str, Any] = {}
        self.latest_compute: dict[str, Any] = {}
        self.chart_clock = QTimer(self)
        self.chart_clock.setInterval(3500)
        self.chart_clock.timeout.connect(self._reload_charts)
        self.chart_clock.start()

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        title_row = QHBoxLayout()
        title = QLabel("Evolution")
        title.setObjectName("SectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        self.status = QLabel("● IDLE")
        self.status.setObjectName("RunBadge")
        title_row.addWidget(self.status)
        root.addLayout(title_row)

        controls = QFrame()
        controls.setObjectName("Panel")
        grid = QGridLayout(controls)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.addWidget(QLabel("Resource mode"), 0, 0)
        grid.addWidget(QLabel("Cycles"), 0, 1)
        grid.addWidget(QLabel("Active compute hours (0 = cycles mode)"), 0, 2)
        self.mode = QComboBox()
        self.mode.addItems(["eco", "normal", "night"])
        self.mode.setCurrentText("night")
        self.cycles = QSpinBox()
        self.cycles.setRange(1, 999)
        self.cycles.setValue(1)
        self.hours = QDoubleSpinBox()
        self.hours.setRange(0, 72)
        self.hours.setDecimals(1)
        self.hours.setValue(8.0)
        self.start_btn = QPushButton("Start evolution")
        self.start_btn.setObjectName("Primary")
        self.stop_btn = QPushButton("Stop safely")
        self.stop_btn.setObjectName("Danger")
        for button in (self.start_btn, self.stop_btn):
            button.setCursor(Qt.PointingHandCursor)
        grid.addWidget(self.mode, 1, 0)
        grid.addWidget(self.cycles, 1, 1)
        grid.addWidget(self.hours, 1, 2)
        grid.addWidget(self.start_btn, 1, 3)
        grid.addWidget(self.stop_btn, 1, 4)
        root.addWidget(controls)

        run_card = QFrame()
        run_card.setObjectName("Card")
        rl = QVBoxLayout(run_card)
        rl.setContentsMargins(18, 16, 18, 16)
        run_head = QHBoxLayout()
        run_head.addWidget(QLabel("CURRENT RUN"))
        run_head.addStretch()
        self.wall_label = QLabel("wall 00:00:00")
        self.wall_label.setObjectName("Subtle")
        run_head.addWidget(self.wall_label)
        rl.addLayout(run_head)

        self.stage_title = QLabel("Waiting")
        self.stage_title.setObjectName("CardValue")
        rl.addWidget(self.stage_title)
        self.detail = QLabel("The current stage will appear here as soon as the process starts.")
        self.detail.setObjectName("Subtle")
        self.detail.setWordWrap(True)
        rl.addWidget(self.detail)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(18)
        metrics.addWidget(QLabel("ACTIVE COMPUTE"), 0, 0)
        metrics.addWidget(QLabel("REMAINING"), 0, 1)
        metrics.addWidget(QLabel("SLEEP EXCLUDED"), 0, 2)
        metrics.addWidget(QLabel("SAFE STOP"), 0, 3)
        self.compute_used = QLabel("—")
        self.compute_left = QLabel("—")
        self.sleep_excluded = QLabel("00:00:00")
        self.safe_stop = QLabel("No")
        for col, widget in enumerate((self.compute_used, self.compute_left, self.sleep_excluded, self.safe_stop)):
            widget.setObjectName("CardValue")
            metrics.addWidget(widget, 1, col)
        rl.addLayout(metrics)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        rl.addWidget(self.progress)
        root.addWidget(run_card)

        flow = QFrame()
        flow.setObjectName("Panel")
        fl = QHBoxLayout(flow)
        fl.setContentsMargins(10, 12, 10, 12)
        fl.setSpacing(5)
        self.stage_boxes: dict[str, QLabel] = {}
        for index, (text, key) in enumerate(self.FLOW):
            label = QLabel(text)
            label.setObjectName("StageBox")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(50)
            label.setWordWrap(True)
            fl.addWidget(label, 1)
            self.stage_boxes[key] = label
            if index < len(self.FLOW) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("Subtle")
                fl.addWidget(arrow)
        root.addWidget(flow)

        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(12)
        info_grid.setVerticalSpacing(12)

        strength = QFrame()
        strength.setObjectName("Panel")
        sl = QVBoxLayout(strength)
        sl.setContentsMargins(14, 12, 14, 12)
        sh = QHBoxLayout()
        sh.addWidget(QLabel("STRENGTH LAB"))
        sh.addStretch()
        self.strength_mode = QLabel("waiting")
        self.strength_mode.setObjectName("RunBadge")
        sh.addWidget(self.strength_mode)
        sl.addLayout(sh)
        self.strength_reason = QLabel("Hard-position memory and self-teacher telemetry will appear here.")
        self.strength_reason.setObjectName("Subtle")
        self.strength_reason.setWordWrap(True)
        sl.addWidget(self.strength_reason)
        sg = QGridLayout()
        for col, text in enumerate(("Natural", "Hard", "Specialist", "Deep teacher", "Backfill")):
            sg.addWidget(QLabel(text), 0, col)
        self.strength_values = [QLabel("—") for _ in range(5)]
        for col, widget in enumerate(self.strength_values):
            widget.setObjectName("CardValue")
            sg.addWidget(widget, 1, col)
        sl.addLayout(sg)
        self.teacher_detail = QLabel("Teacher search: —")
        self.teacher_detail.setObjectName("Subtle")
        sl.addWidget(self.teacher_detail)
        info_grid.addWidget(strength, 0, 0)

        league = QFrame()
        league.setObjectName("Panel")
        ll = QVBoxLayout(league)
        ll.setContentsMargins(14, 12, 14, 12)
        lh = QHBoxLayout()
        lh.addWidget(QLabel("LIVE LEAGUE"))
        lh.addStretch()
        self.league_summary = QLabel("waiting")
        self.league_summary.setObjectName("RunBadge")
        lh.addWidget(self.league_summary)
        ll.addLayout(lh)
        self.league_table = QTableWidget(0, 9)
        self.league_table.setHorizontalHeaderLabels(
            ["Game", "Pair", "White", "Black", "Opening", "Leg", "Ply", "Runtime", "State"]
        )
        self.league_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.league_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.league_table.verticalHeader().setVisible(False)
        header = self.league_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.league_table.setMinimumHeight(150)
        ll.addWidget(self.league_table)
        info_grid.addWidget(league, 0, 1)

        reference = QFrame()
        reference.setObjectName("Panel")
        rf = QVBoxLayout(reference)
        rf.setContentsMargins(14, 12, 14, 12)
        rh = QHBoxLayout()
        rh.addWidget(QLabel("CIVILIZATION STRENGTH · FROZEN REFERENCE"))
        rh.addStretch()
        self.reference_badge = QLabel("waiting")
        self.reference_badge.setObjectName("RunBadge")
        rh.addWidget(self.reference_badge)
        rf.addLayout(rh)
        rg = QGridLayout()
        for col, text in enumerate(("Reference", "Subject", "Score", "Games")):
            rg.addWidget(QLabel(text), 0, col)
        self.reference_generation = QLabel("—")
        self.reference_subject = QLabel("—")
        self.reference_score = QLabel("—")
        self.reference_games = QLabel("—")
        for col, widget in enumerate((self.reference_generation, self.reference_subject, self.reference_score, self.reference_games)):
            widget.setObjectName("CardValue")
            rg.addWidget(widget, 1, col)
        rf.addLayout(rg)
        self.reference_trend = QLabel("Trend: waiting for fixed-reference rounds")
        self.reference_trend.setObjectName("Subtle")
        self.reference_trend.setWordWrap(True)
        rf.addWidget(self.reference_trend)
        info_grid.addWidget(reference, 1, 0)

        safety = QFrame()
        safety.setObjectName("Panel")
        sf = QVBoxLayout(safety)
        sf.setContentsMargins(14, 12, 14, 12)
        s_head = QHBoxLayout()
        s_head.addWidget(QLabel("GAME SAFETY"))
        s_head.addStretch()
        self.watchdog_badge = QLabel("budget never kills games")
        self.watchdog_badge.setObjectName("RunBadge")
        s_head.addWidget(self.watchdog_badge)
        sf.addLayout(s_head)
        wg = QGridLayout()
        wg.addWidget(QLabel("No-progress bug threshold"), 0, 0)
        wg.addWidget(QLabel("Emergency single-game ceiling"), 0, 1)
        wg.addWidget(QLabel("Kill grace"), 0, 2)
        self.watchdog_stall = QLabel("—")
        self.watchdog_emergency = QLabel("—")
        self.watchdog_grace = QLabel("—")
        for col, widget in enumerate((self.watchdog_stall, self.watchdog_emergency, self.watchdog_grace)):
            widget.setObjectName("CardValue")
            wg.addWidget(widget, 1, col)
        sf.addLayout(wg)
        self.watchdog_note = QLabel(
            "Healthy active games finish naturally even after the Night compute budget is reached."
        )
        self.watchdog_note.setObjectName("Subtle")
        self.watchdog_note.setWordWrap(True)
        sf.addWidget(self.watchdog_note)
        info_grid.addWidget(safety, 1, 1)

        info_grid.setColumnStretch(0, 1)
        info_grid.setColumnStretch(1, 2)
        root.addLayout(info_grid)

        note = QLabel(
            "Night time is an admission budget, not a chess clock. Lid-close/process suspension is excluded; "
            "when the budget is reached, already-started games and their reverse-colour fairness legs finish naturally. "
            "Only the separate conservative bug watchdog may terminate an obviously wedged worker."
        )
        note.setObjectName("InfoNote")
        note.setWordWrap(True)
        root.addWidget(note)

        charts = QFrame()
        charts.setObjectName("Panel")
        chart_layout = QVBoxLayout(charts)
        chart_layout.setContentsMargins(14, 12, 14, 14)
        chart_head = QHBoxLayout()
        chart_head.addWidget(QLabel("TRAINING CHARTS"))
        chart_head.addStretch()
        self.chart_note = QLabel("SQLite lineage + live trainer telemetry")
        self.chart_note.setObjectName("Subtle")
        chart_head.addWidget(self.chart_note)
        chart_layout.addLayout(chart_head)
        chart_grid = QGridLayout()
        chart_grid.setHorizontalSpacing(10)
        self.live_loss_chart = LineChart("CURRENT RUN · TRAINING LOSS")
        self.gen_loss_chart = LineChart("LINEAGE · TRAINING LOSS")
        self.arena_chart = LineChart("LINEAGE · ARENA SCORE", percent=True)
        chart_grid.addWidget(self.live_loss_chart, 0, 0)
        chart_grid.addWidget(self.gen_loss_chart, 0, 1)
        chart_grid.addWidget(self.arena_chart, 0, 2)
        chart_layout.addLayout(chart_grid)
        root.addWidget(charts)

        root.addWidget(QLabel("Live process log"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        root.addWidget(self.log, 1)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self.process.stop_safely)
        self.process.output.connect(self._append)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.state_changed.connect(self._running_changed)
        self.process.stage_changed.connect(self._stage_changed)
        if hasattr(self.process, "ui_event"):
            self.process.ui_event.connect(self._ui_event)
        self._running_changed(self.process.running)
        QTimer.singleShot(250, self._reload_charts)

    def _start(self):
        ok = self.process.start_evolution(self.mode.currentText(), self.cycles.value(), self.hours.value())
        if not ok:
            self._append("[Studio] A process is already running; Start was ignored.")

    def _running_changed(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.mode.setEnabled(not running)
        self.cycles.setEnabled(not running)
        self.hours.setEnabled(not running)

    def _started(self, label: str):
        self.elapsed.start()
        self.clock.start()
        self.latest_ui.clear()
        self.latest_compute.clear()
        self.live_loss_x.clear()
        self.live_loss_y.clear()
        self.live_loss_chart.clear()
        self.league_table.setRowCount(0)
        self.status.setText(f"● RUNNING · {label}")
        self.status.setProperty("active", True)
        self._restyle(self.status)
        self._tick()

    def _finished(self, code: int):
        self.clock.stop()
        self.status.setText("● IDLE" if code == 0 else f"● EXITED · code {code}")
        self.status.setProperty("active", False)
        self._restyle(self.status)
        self.stage_title.setText("Finished" if code == 0 else "Process stopped")
        self.detail.setText("Safe boundary reached." if code == 0 else "Check the final log lines for details.")
        self.progress.setVisible(False)
        self._highlight(None)
        self._reload_charts()

    def _tick(self):
        if not self.elapsed.isValid():
            return
        self.wall_label.setText(f"wall {_hms(self.elapsed.elapsed() / 1000)}")

    def _stage_changed(self, stage: str, detail: str):
        normalized = stage.strip().lower().replace("_", "-")
        if normalized in {"cycle", "cycle-complete"}:
            return
        label = normalized.upper().replace("-", " ")
        self.status.setText(f"● {label}" + (f" · {detail}" if detail else ""))
        self.stage_title.setText(label)
        self.detail.setText(detail or self._stage_explanation(normalized))
        self._highlight(normalized)

        if normalized in {"training", "population-train"} and detail:
            match = self._loss_re.search(detail)
            if match:
                step = float(match.group(1))
                loss = float(match.group(3))
                self.live_loss_x.append(step)
                self.live_loss_y.append(loss)
                self.live_loss_chart.set_series(self.live_loss_x, self.live_loss_y)

        if detail and "/" in detail:
            try:
                token = detail.split()[0]
                cur, total = [int(x) for x in token.split("/", 1)]
                self.progress.setRange(0, total)
                self.progress.setValue(cur)
                self.progress.setVisible(True)
            except ValueError:
                self.progress.setVisible(False)
        else:
            self.progress.setVisible(normalized not in ("idle", "promoted", "rejected"))
            if self.progress.isVisible():
                self.progress.setRange(0, 0)

        if normalized in {"arena", "promoted", "rejected", "fixed-reference"}:
            self._reload_charts()

    def _ui_event(self, payload: object):
        if not isinstance(payload, dict):
            return
        self.latest_ui = payload
        compute = payload.get("compute")
        if isinstance(compute, dict):
            self._apply_compute(compute)

        phase = str(payload.get("phase") or "").replace("_", "-")
        if phase == "strength-lab" or "strength_lab" in payload:
            strength = payload if phase == "strength-lab" else payload.get("strength_lab")
            if isinstance(strength, dict):
                self._apply_strength(strength)
        league = payload.get("league")
        if isinstance(league, dict):
            self._apply_league(league)
        parallel = payload.get("parallel_league")
        if isinstance(parallel, dict):
            nested = parallel.get("league")
            self._apply_league(nested if isinstance(nested, dict) else parallel)

        fixed = payload.get("fixed_reference")
        if isinstance(fixed, dict):
            self._apply_fixed_reference(fixed)
        watchdog = payload.get("watchdog")
        if isinstance(watchdog, dict):
            self._apply_watchdog(watchdog)

        if phase == "fixed-reference-error":
            self.reference_badge.setText("DISABLED THIS RUN")
            self.reference_trend.setText(str(payload.get("error") or "Reference integration error"))
        if phase == "safe-stop-requested":
            self.safe_stop.setText("Requested")
        for key in ("league_drain", "arena_drain"):
            drain = payload.get(key)
            if isinstance(drain, dict) and drain.get("draining"):
                self.safe_stop.setText(str(drain.get("reason") or "Draining"))

    def _apply_compute(self, compute: dict[str, Any]):
        self.latest_compute = compute
        used = compute.get("elapsed_seconds", compute.get("elapsed_compute_seconds"))
        budget = compute.get("budget_seconds")
        remaining = compute.get("remaining_seconds", compute.get("remaining_compute_seconds"))
        self.compute_used.setText(f"{_hms(used)} / {_hms(budget)}")
        self.compute_left.setText(_hms(remaining))
        self.sleep_excluded.setText(_hms(compute.get("excluded_sleep_seconds", 0)))
        if compute.get("safe_stop_requested"):
            self.safe_stop.setText(str(compute.get("safe_stop_reason") or "Requested"))

    def _apply_strength(self, strength: dict[str, Any]):
        plan = strength.get("plan") if isinstance(strength.get("plan"), dict) else strength
        recipe = strength.get("recipe") if isinstance(strength.get("recipe"), dict) else {}
        mode = str(plan.get("mode") or strength.get("mode") or "active")
        self.strength_mode.setText(mode.upper())
        self.strength_reason.setText(str(plan.get("reason") or "Strength Lab active"))
        effective = recipe.get("effective") if isinstance(recipe.get("effective"), dict) else {}
        requested = recipe.get("requested") if isinstance(recipe.get("requested"), dict) else {}
        values = [
            effective.get("natural_selfplay", requested.get("natural_selfplay")),
            effective.get("hard_positions", requested.get("hard_positions")),
            effective.get("specialist_sparring", requested.get("specialist_sparring")),
            effective.get("deep_search_teacher", requested.get("deep_search_teacher")),
            recipe.get("backfilled_examples", 0),
        ]
        for widget, value in zip(self.strength_values, values):
            widget.setText("—" if value is None else str(value))
        mult = recipe.get("teacher_search_multiplier", plan.get("teacher_search_multiplier"))
        teacher_examples = strength.get("teacher_examples")
        captured = strength.get("captured_positions")
        bits = [f"Teacher search: {mult}×" if mult else "Teacher search: —"]
        if captured is not None:
            bits.append(f"captured {captured}")
        if teacher_examples is not None:
            bits.append(f"labels {teacher_examples}")
        self.teacher_detail.setText(" · ".join(bits))

    def _apply_league(self, league: dict[str, Any]):
        active = league.get("active_games") or []
        if not isinstance(active, list):
            active = []
        slots = league.get("parallel_games", "—")
        draining = bool(league.get("draining"))
        reason = league.get("stop_reason") or league.get("reason")
        state = f"{slots} slots · {len(active)} live"
        if draining:
            state += f" · DRAINING ({reason or 'safe stop'})"
        self.league_summary.setText(state)
        self.league_table.setRowCount(len(active))
        for row_index, game in enumerate(active):
            if not isinstance(game, dict):
                continue
            values = [
                game.get("game_id", ""),
                game.get("pairing_id", ""),
                game.get("white_id", ""),
                game.get("black_id", ""),
                game.get("opening", ""),
                game.get("leg", ""),
                game.get("plies", ""),
                _hms(game.get("runtime_seconds", 0)),
                game.get("state", ""),
            ]
            for col, value in enumerate(values):
                self.league_table.setItem(row_index, col, QTableWidgetItem(str(value)))

    def _apply_fixed_reference(self, fixed: dict[str, Any]):
        reference = fixed.get("reference") if isinstance(fixed.get("reference"), dict) else {}
        result = fixed.get("result") if isinstance(fixed.get("result"), dict) else None
        active = fixed.get("active") if isinstance(fixed.get("active"), dict) else None
        trend = fixed.get("trend") if isinstance(fixed.get("trend"), list) else []
        ref_generation = reference.get("generation")
        subject = fixed.get("subject_generation")
        self.reference_generation.setText("—" if ref_generation is None else f"Gen{ref_generation} · FROZEN")
        self.reference_subject.setText("—" if subject is None else f"Gen{subject}")

        if result is not None:
            self.reference_badge.setText("MEASURED")
            self.reference_score.setText(_pct(result.get("score")))
            self.reference_games.setText(str(result.get("games", "—")))
        elif active is not None:
            self.reference_badge.setText("MEASURING")
            self.reference_score.setText("…")
            self.reference_games.setText(str(active.get("results", 0)))
        elif fixed.get("skipped_reason"):
            self.reference_badge.setText("SKIPPED")
            self.reference_score.setText("—")
            self.reference_games.setText("—")
        else:
            self.reference_badge.setText("READY")

        if trend:
            pieces = []
            for row in trend[-8:]:
                if not isinstance(row, dict):
                    continue
                pieces.append(f"R{row.get('round_index', '?')} {_pct(row.get('score'))}")
            self.reference_trend.setText("Trend: " + " → ".join(pieces) if pieces else "Trend: waiting")
        else:
            self.reference_trend.setText("Trend: waiting for fixed-reference rounds")

    def _apply_watchdog(self, watchdog: dict[str, Any]):
        self.watchdog_stall.setText(_hms(watchdog.get("stall_seconds")))
        self.watchdog_emergency.setText(_hms(watchdog.get("emergency_game_seconds")))
        grace = watchdog.get("kill_grace_seconds")
        self.watchdog_grace.setText("—" if grace is None else f"{float(grace):.1f}s")
        if watchdog.get("budget_interrupts_games") is False:
            self.watchdog_badge.setText("BUDGET ≠ GAME TIMEOUT")
            self.watchdog_note.setText(
                "Healthy games are never stopped because the Night budget is exhausted; only an obvious stall or emergency ceiling can terminate a worker."
            )

    def _reload_charts(self):
        try:
            generations = self.store.generations(limit=500)
        except Exception as exc:
            self.chart_note.setText(f"Chart data unavailable: {exc}")
            return
        gx, gl = numeric_series(generations, ("id", "generation"), ("training_loss",))
        ax, ar = numeric_series(generations, ("id", "generation"), ("arena_score",))
        self.gen_loss_chart.set_series(gx, gl)
        self.arena_chart.set_series(ax, ar, threshold=0.55)
        self.chart_note.setText(
            f"{len(generations)} generations in lifetime SQLite" if generations else "Waiting for generation history"
        )

    def _stage_explanation(self, stage: str) -> str:
        return {
            "starting": "Launching the existing champion lineage without resetting lifetime state.",
            "self-play": "Generating durable experience across the opening curriculum.",
            "strength-lab": "Mining difficult positions and preparing targeted replay/self-teacher work.",
            "training": "Running the unchanged continual trainer through the Strength-aware replay mixer.",
            "population-train": "Shared training plus role-specific population branches.",
            "league": "2–3 colour-balanced League games run in killable worker processes.",
            "arena": "Held-out challenger vs champion test; run time never terminates a healthy active game.",
            "fixed-reference": "Measuring the strongest round subject against one immutable historical checkpoint.",
            "strength-guard": "Checking fixed-reference/engine evidence before adoption.",
            "promoted": "The challenger passed the gate and became champion.",
            "rejected": "The challenger failed the gate; the old champion remains active.",
            "safe-stop-requested": "Stop requested; already-started games and fairness legs finish naturally.",
            "specialist-harvest": "Preserving useful specialist experience without discarding the overall loser.",
        }.get(stage, "")

    def _highlight(self, stage: str | None):
        mapping = {
            "self-play": "self-play",
            "strength-lab": "strength-lab",
            "training": "population-train",
            "population-train": "population-train",
            "league": "league",
            "arena": "arena",
            "fixed-reference": "strength-guard",
            "strength-guard": "strength-guard",
            "promoted": "promotion",
            "rejected": "promotion",
            "archive": "archive",
            "specialist-harvest": "archive",
        }
        active = mapping.get(stage or "")
        for key, label in self.stage_boxes.items():
            label.setProperty("active", key == active)
            self._restyle(label)

    @staticmethod
    def _restyle(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _append(self, text: str):
        self.log.appendPlainText(text)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
