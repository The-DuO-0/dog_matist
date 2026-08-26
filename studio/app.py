from __future__ import annotations

import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .backend import AgentBridge, ProcessController
from .mascot import DogMatistMascot
from .pages.chat import ChatPage
from .pages.dashboard import DashboardPage
from .pages.evolution import EvolutionPage
from .pages.dynasty import DynastyPage
from .scroll_host import scroll_page
from .pages.play import PlayPage
from .pages.research import ResearchPage
from .theme import APP_QSS


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dog_matist Studio")
        self.resize(1500, 940)
        self.setMinimumSize(1100, 720)

        self.agent = AgentBridge(mode="normal", parent=self)
        self.process = ProcessController(self)
        self._close_when_finished = False
        self.process.finished.connect(self._finish_pending_close)

        shell = QWidget()
        main = QHBoxLayout(shell)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(16, 18, 16, 18)
        sl.setSpacing(8)

        mascot_row = QHBoxLayout()
        self.mascot = DogMatistMascot()
        mascot_row.addWidget(self.mascot)
        mascot_row.addStretch()
        sl.addLayout(mascot_row)

        brand = QLabel("dog_matist")
        brand.setObjectName("Brand")
        sl.addWidget(brand)
        sub = QLabel("STUDIO 2.0 · DEV")
        sub.setObjectName("Subtle")
        sl.addWidget(sub)
        sl.addSpacing(16)

        self.stack = QStackedWidget()
        self.pages = [
            ("Dashboard", DashboardPage(self.agent, self.process)),
            ("Play dog_matist", PlayPage(self.agent)),
            ("Evolution", scroll_page(EvolutionPage(self.process), min_content_height=1220)),
            ("Dynasty Archive", scroll_page(DynastyPage(), min_content_height=1080)),
            ("Research", ResearchPage(self.process)),
            ("Conversation", ChatPage(self.agent)),
        ]
        self.nav = []
        for i, (name, page) in enumerate(self.pages):
            btn = QPushButton(name)
            btn.setObjectName("Nav")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, idx=i: self.set_page(idx))
            sl.addWidget(btn)
            self.nav.append(btn)
            self.stack.addWidget(page)
        sl.addStretch()

        self.activity = QLabel("● dog_matist idle")
        self.activity.setObjectName("SidebarStatus")
        sl.addWidget(self.activity)
        footer = QLabel("Same champion lineage,\nSQLite memory & checkpoints.")
        footer.setObjectName("Subtle")
        footer.setWordWrap(True)
        sl.addWidget(footer)

        main.addWidget(sidebar)
        main.addWidget(self.stack, 1)
        self.setCentralWidget(shell)
        self.set_page(0)

        self.process.state_changed.connect(self._process_state)
        self.process.stage_changed.connect(self._process_stage)
        self._process_state(self.process.running)

    def set_page(self, idx):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav):
            btn.setChecked(i == idx)

    def _process_state(self, running: bool):
        self.mascot.set_active(running)
        if not running:
            self.activity.setText("● dog_matist idle")
            self.activity.setProperty("active", False)
            self._restyle(self.activity)

    def _process_stage(self, stage: str, detail: str):
        label = stage.replace("_", " ").upper()
        self.activity.setText(f"● {label}")
        self.activity.setToolTip(detail or label)
        self.activity.setProperty("active", stage not in {"idle", "promoted", "rejected"})
        self._restyle(self.activity)

    @staticmethod
    def _restyle(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def closeEvent(self, event):
        if self.process.running and not self._close_when_finished:
            choice = QMessageBox.question(
                self,
                "Evolution is running",
                "Stop dog_matist safely and close Studio when it reaches the safe boundary?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choice != QMessageBox.Yes:
                event.ignore()
                return
            self._close_when_finished = True
            self.process.stop_safely()
            event.ignore()
            return
        if self.process.running:
            event.ignore()
            return
        self.agent.close()
        super().closeEvent(event)

    def _finish_pending_close(self, _exit_code):
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("dog_matist Studio")
    app.setStyleSheet(APP_QSS)
    window = MainWindow()
    window.show()
    return app.exec()
