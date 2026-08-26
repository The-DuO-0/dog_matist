from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class LineChart(QWidget):
    """Small dependency-free line chart for dog_matist Studio."""

    def __init__(self, title: str, *, percent: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.percent = percent
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.threshold: float | None = None
        self.setMinimumHeight(180)

    def set_series(self, xs, ys, *, threshold: float | None = None) -> None:
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]
        self.threshold = threshold
        self.update()

    def clear(self) -> None:
        self.set_series([], [])

    def _fmt(self, value: float) -> str:
        return f"{value * 100:.1f}%" if self.percent else f"{value:.4f}"

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        bg = QColor("#121720")
        border = QColor("#2a3342")
        text = QColor("#d8deea")
        subtle = QColor("#7f8a9c")
        grid = QColor("#28303d")
        line = QColor("#5f8bff")
        point = QColor("#b9ccff")
        gate = QColor("#e2b857")

        outer = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(border, 1))
        p.setBrush(bg)
        p.drawRoundedRect(outer, 10, 10)

        title_font = QFont()
        title_font.setBold(True)
        title_font.setPixelSize(13)
        p.setFont(title_font)
        p.setPen(text)
        p.drawText(QRectF(14, 10, self.width() - 28, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        if self.ys:
            value_font = QFont()
            value_font.setBold(True)
            value_font.setPixelSize(16)
            p.setFont(value_font)
            p.drawText(QRectF(14, 31, self.width() - 28, 24), Qt.AlignLeft | Qt.AlignVCenter, self._fmt(self.ys[-1]))
        else:
            p.setPen(subtle)
            p.drawText(QRectF(14, 32, self.width() - 28, 22), Qt.AlignLeft | Qt.AlignVCenter, "waiting for data")

        plot = QRectF(42, 62, max(10, self.width() - 58), max(10, self.height() - 88))
        p.setPen(QPen(grid, 1))
        for i in range(4):
            y = plot.top() + i * plot.height() / 3
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        if not self.ys:
            return

        xmin, xmax = min(self.xs), max(self.xs)
        ymin, ymax = min(self.ys), max(self.ys)
        if self.threshold is not None:
            ymin = min(ymin, self.threshold)
            ymax = max(ymax, self.threshold)
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            pad = max(abs(ymin) * 0.05, 0.05)
            ymin -= pad
            ymax += pad
        else:
            pad = (ymax - ymin) * 0.12
            ymin -= pad
            ymax += pad

        def map_point(x: float, y: float) -> QPointF:
            px = plot.left() + (x - xmin) / (xmax - xmin) * plot.width()
            py = plot.bottom() - (y - ymin) / (ymax - ymin) * plot.height()
            return QPointF(px, py)

        axis_font = QFont()
        axis_font.setPixelSize(10)
        p.setFont(axis_font)
        p.setPen(subtle)
        p.drawText(QRectF(3, plot.top() - 7, 36, 14), Qt.AlignRight, self._fmt(ymax))
        p.drawText(QRectF(3, plot.bottom() - 7, 36, 14), Qt.AlignRight, self._fmt(ymin))
        p.drawText(QRectF(plot.left(), plot.bottom() + 5, 70, 15), Qt.AlignLeft, f"{xmin:g}")
        p.drawText(QRectF(plot.right() - 70, plot.bottom() + 5, 70, 15), Qt.AlignRight, f"{xmax:g}")

        if self.threshold is not None and ymin <= self.threshold <= ymax:
            y = map_point(xmin, self.threshold).y()
            pen = QPen(gate, 1)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            p.drawText(QRectF(plot.left() + 5, y - 18, 90, 16), Qt.AlignLeft, f"gate {self._fmt(self.threshold)}")

        path = QPainterPath()
        for i, (x, y) in enumerate(zip(self.xs, self.ys)):
            pt = map_point(x, y)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        p.setPen(QPen(line, 2.2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

        p.setPen(Qt.NoPen)
        p.setBrush(point)
        for x, y in zip(self.xs[-24:], self.ys[-24:]):
            p.drawEllipse(map_point(x, y), 2.8, 2.8)
