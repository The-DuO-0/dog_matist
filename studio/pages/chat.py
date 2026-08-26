from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget


class ChatPage(QWidget):
    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.pending = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        title = QLabel("Conversation")
        title.setObjectName("SectionTitle")
        root.addWidget(title)
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setPlaceholderText("Talk to dog_matist about its real chess memory and positions.")
        root.addWidget(self.history, 1)
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask dog_matist something…")
        self.send = QPushButton("Send")
        self.send.setObjectName("Primary")
        self.send.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.input, 1); row.addWidget(self.send)
        root.addLayout(row)
        self.send.clicked.connect(self._send)
        self.input.returnPressed.connect(self._send)
        self.agent.talk_ready.connect(self._reply)
        self.agent.error.connect(self._error)

    def _send(self):
        text = self.input.text().strip()
        if not text or self.pending:
            return
        self.history.append(f"<b>you</b> · {text}")
        self.input.clear()
        self.pending = self.agent.request_talk(text)
        self.send.setEnabled(False)

    def _reply(self, request_id, text):
        if request_id != self.pending:
            return
        self.pending = None
        self.history.append(f"<b>dog_matist</b> · {text}<br>")
        self.send.setEnabled(True)

    def _error(self, request_id, message):
        if request_id != self.pending:
            return
        self.pending = None
        self.history.append(f"<b>error</b> · {message}<br>")
        self.send.setEnabled(True)
