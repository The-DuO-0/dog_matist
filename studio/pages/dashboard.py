from __future__ import annotations

import json
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..backend import find_state_value


class DashboardPage(QWidget):
    def __init__(self, agent, process, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.process = process
        self.pending = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)
        row = QHBoxLayout()
        title = QLabel("Dashboard")
        title.setObjectName("SectionTitle")
        row.addWidget(title)
        row.addStretch()
        self.refresh = QPushButton("Refresh")
        self.refresh.clicked.connect(self.reload)
        row.addWidget(self.refresh)
        root.addLayout(row)

        cards = QGridLayout()
        self.values = {}
        for i, (key, label) in enumerate([
            ("generation", "Champion generation"),
            ("games", "Lifetime games"),
            ("replay", "Replay examples"),
            ("device", "Training device"),
        ]):
            card = QFrame(); card.setObjectName("Card")
            layout = QVBoxLayout(card)
            lab = QLabel(label); lab.setObjectName("Subtle")
            val = QLabel("—"); val.setObjectName("CardValue")
            layout.addWidget(lab); layout.addWidget(val)
            cards.addWidget(card, i // 2, i % 2)
            self.values[key] = val
        root.addLayout(cards)

        self.run = QLabel("Evolution process: idle")
        self.run.setObjectName("InfoNote")
        root.addWidget(self.run)
        self.raw = QLabel("Waiting for core status…")
        self.raw.setObjectName("Subtle")
        self.raw.setWordWrap(True)
        root.addWidget(self.raw)
        root.addStretch()

        self.agent.status_ready.connect(self._status)
        self.agent.error.connect(self._error)
        self.process.stage_changed.connect(self._stage)
        self.process.state_changed.connect(lambda running: self.run.setText("Evolution process: running" if running else "Evolution process: idle"))
        QTimer.singleShot(600, self.reload)

    def reload(self):
        self.pending = self.agent.request_status()

    def _status(self, request_id, state):
        if request_id != self.pending:
            return
        self.pending = None
        self.values["generation"].setText(str(find_state_value(state, "champion_generation", "generation", default="—")))
        self.values["games"].setText(str(find_state_value(state, "games", "game_count", default="—")))
        self.values["replay"].setText(str(find_state_value(state, "replay_examples", "examples", default="—")))
        self.values["device"].setText(str(find_state_value(state, "training_device", "device", default="—")))
        self.raw.setText("State source: existing persistent chess core. dog_matist does not reset the database or checkpoints.")

    def _stage(self, stage, detail):
        self.run.setText(f"Evolution process: {stage}" + (f" · {detail}" if detail else ""))

    def _error(self, request_id, message):
        if request_id == self.pending:
            self.pending = None
            self.raw.setText(f"Status error: {message}")
