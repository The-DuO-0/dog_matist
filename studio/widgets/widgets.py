from __future__ import annotations

from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget


class DogMatistMascot(QWidget):
    """Tiny vector mascot; no external image assets required."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self._phase = 0
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setInterval(420)
        self._timer.timeout.connect(self._tick)

    def set_busy(self, busy: bool):
        self._busy = bool(busy)
        if self._busy and not self._timer.isActive():
            self._timer.start()
        elif not self._busy:
            self._timer.stop()
            self._phase = 0
        self.update()

    def _tick(self):
        self._phase = (self._phase + 1) % 4
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.translate(4, 4)

        # ears
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#7b5b43"))
        p.drawEllipse(5, 12, 18, 28)
        p.drawEllipse(41, 12, 18, 28)

        # head
        p.setBrush(QColor("#d6a56f"))
        p.drawEllipse(12, 10, 44, 46)
        p.setBrush(QColor("#f1d2ae"))
        p.drawEllipse(20, 29, 28, 22)

        # eyes
        p.setBrush(QColor("#111318"))
        p.drawEllipse(23, 25, 4, 5)
        p.drawEllipse(41, 25, 4, 5)
        p.drawEllipse(31, 34, 7, 5)

        # tiny knight badge
        p.setPen(QPen(QColor("#8fb0ff"), 2))
        p.setBrush(QColor("#20335f"))
        p.drawRoundedRect(23, 52, 22, 10, 5, 5)
        p.setPen(QColor("#eaf0ff"))
        p.drawText(23, 51, 22, 11, Qt.AlignCenter, "♞")

        if self._busy:
            p.setPen(Qt.NoPen)
            for i in range(3):
                alpha = 255 if i == (self._phase % 3) else 80
                c = QColor("#7ea2ff")
                c.setAlpha(alpha)
                p.setBrush(c)
                p.drawEllipse(QPointF(49 + i * 6, 8 - i * 3), 2.2, 2.2)
