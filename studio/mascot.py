from __future__ import annotations

from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class DogMatistMascot(QWidget):
    """Tiny vector mascot: a dog wearing a chess-knight badge.

    It stays intentionally lightweight: no image assets, no network, and only a
    subtle thinking-dot animation while evolution is running.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(116, 92)
        self._active = False
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.setInterval(360)
        self._timer.timeout.connect(self._advance)

    def set_active(self, active: bool):
        self._active = bool(active)
        if self._active and not self._timer.isActive():
            self._timer.start()
        elif not self._active:
            self._timer.stop()
            self._frame = 0
        self.update()

    def _advance(self):
        self._frame = (self._frame + 1) % 3
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # ears
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#6f5849"))
        p.drawEllipse(QPointF(28, 29), 15, 22)
        p.drawEllipse(QPointF(76, 29), 15, 22)

        # head
        p.setBrush(QColor("#b58a69"))
        p.drawEllipse(QPointF(52, 43), 35, 31)

        # muzzle
        p.setBrush(QColor("#d7b89c"))
        p.drawEllipse(QPointF(52, 54), 21, 14)

        # eyes
        p.setBrush(QColor("#13161c"))
        p.drawEllipse(QPointF(40, 39), 3.8, 4.6)
        p.drawEllipse(QPointF(64, 39), 3.8, 4.6)
        p.drawEllipse(QPointF(52, 51), 4.2, 3.4)

        # collar + chess badge
        p.setBrush(QColor("#4169e1"))
        p.drawRoundedRect(24, 70, 56, 11, 5, 5)
        p.setPen(QPen(QColor("#ffffff"), 1))
        font = QFont("Arial Unicode MS")
        font.setPixelSize(19)
        font.setBold(True)
        p.setFont(font)
        p.drawText(61, 85, "♞")

        if self._active:
            p.setPen(Qt.NoPen)
            for i in range(3):
                alpha = 235 if i == self._frame else 90
                p.setBrush(QColor(95, 139, 255, alpha))
                p.drawEllipse(QPointF(91 + i * 8, 20 - i * 5), 3.1, 3.1)
