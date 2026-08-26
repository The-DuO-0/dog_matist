from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class ResearchPage(QWidget):
    def __init__(self, process, parent=None):
        super().__init__(parent)
        self.process = process
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        row = QHBoxLayout()
        title = QLabel("Research")
        title.setObjectName("SectionTitle")
        row.addWidget(title); row.addStretch()
        self.export = QPushButton("Export research data")
        self.export.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.export)
        root.addLayout(row)
        note = QLabel("Exports continue to use the existing persistent database and champion lineage. Arena remains held out from training replay.")
        note.setObjectName("InfoNote"); note.setWordWrap(True)
        root.addWidget(note)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        root.addWidget(self.log, 1)
        self.export.clicked.connect(self._export)
        self.process.output.connect(self.log.appendPlainText)
        self.process.state_changed.connect(lambda running: self.export.setEnabled(not running))

    def _export(self):
        if not self.process.start(["export"], "Research export"):
            self.log.appendPlainText("A process is already running.")
