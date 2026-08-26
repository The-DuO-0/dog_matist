from __future__ import annotations

import chess
import chess.svg
from PySide6.QtCore import QByteArray, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget


class ChessBoardWidget(QWidget):
    move_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.flipped = False
        self.selected: chess.Square | None = None
        self.last_move: chess.Move | None = None
        self.enabled_for_user = True
        self._renderers = {
            piece.symbol(): QSvgRenderer(QByteArray(chess.svg.piece(piece, size=128).encode("utf-8")))
            for piece in [chess.Piece(pt, color) for color in (chess.WHITE, chess.BLACK) for pt in range(1, 7)]
        }
        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(650, 650)

    def set_board(self, board: chess.Board, last_move: chess.Move | None = None):
        self.board = board
        self.last_move = last_move
        self.selected = None
        self.update()

    def set_flipped(self, flipped: bool):
        self.flipped = flipped
        self.update()

    def set_user_enabled(self, enabled: bool):
        self.enabled_for_user = enabled
        self.update()

    def _geometry(self):
        side = min(self.width(), self.height())
        left = (self.width() - side) / 2
        top = (self.height() - side) / 2
        return left, top, side, side / 8

    def _display_coords(self, square: chess.Square):
        f = chess.square_file(square)
        r = chess.square_rank(square)
        if self.flipped:
            return 7 - f, r
        return f, 7 - r

    def _square_rect(self, square: chess.Square):
        left, top, _side, cell = self._geometry()
        x, y = self._display_coords(square)
        return QRectF(left + x * cell, top + y * cell, cell, cell)

    def _square_at(self, pos):
        left, top, side, cell = self._geometry()
        if not (left <= pos.x() < left + side and top <= pos.y() < top + side):
            return None
        x = int((pos.x() - left) // cell)
        y = int((pos.y() - top) // cell)
        if self.flipped:
            file_ = 7 - x
            rank = y
        else:
            file_ = x
            rank = 7 - y
        return chess.square(file_, rank)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        light = QColor("#d9d3c5")
        dark = QColor("#697b8c")
        selected = QColor(79, 120, 228, 150)
        last = QColor(239, 205, 96, 115)
        legal = QColor(60, 65, 74, 120)
        check = QColor(210, 72, 84, 145)

        for sq in chess.SQUARES:
            rect = self._square_rect(sq)
            color = light if (chess.square_file(sq) + chess.square_rank(sq)) % 2 else dark
            p.fillRect(rect, color)

        if self.last_move:
            p.fillRect(self._square_rect(self.last_move.from_square), last)
            p.fillRect(self._square_rect(self.last_move.to_square), last)

        if self.board.is_check():
            king = self.board.king(self.board.turn)
            if king is not None:
                p.fillRect(self._square_rect(king), check)

        if self.selected is not None:
            p.fillRect(self._square_rect(self.selected), selected)
            for mv in self.board.legal_moves:
                if mv.from_square == self.selected:
                    rect = self._square_rect(mv.to_square)
                    radius = rect.width() * (0.14 if self.board.piece_at(mv.to_square) is None else 0.42)
                    p.setPen(Qt.NoPen)
                    p.setBrush(legal)
                    p.drawEllipse(rect.center(), radius, radius)

        for sq, piece in self.board.piece_map().items():
            rect = self._square_rect(sq).adjusted(4, 4, -4, -4)
            self._renderers[piece.symbol()].render(p, rect)

        # coordinates
        p.setPen(QPen(QColor("#505867")))
        font = p.font(); font.setPointSize(9); p.setFont(font)
        for i in range(8):
            display_file = 7 - i if self.flipped else i
            display_rank = i if self.flipped else 7 - i
            left, top, side, cell = self._geometry()
            p.drawText(QRectF(left + i*cell + 4, top + side - 18, 20, 15), Qt.AlignLeft, chr(ord('a') + display_file))
            p.drawText(QRectF(left + 3, top + i*cell + 3, 18, 15), Qt.AlignLeft, str(display_rank + 1))

        if not self.enabled_for_user:
            p.fillRect(QRectF(*self._geometry()[:3], self._geometry()[2]), QColor(10, 12, 16, 25))

    def mousePressEvent(self, event):
        if not self.enabled_for_user or event.button() != Qt.LeftButton:
            return
        sq = self._square_at(event.position())
        if sq is None:
            return
        if self.selected is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self.selected = sq
                self.update()
            return

        if sq == self.selected:
            self.selected = None
            self.update()
            return

        candidates = [m for m in self.board.legal_moves if m.from_square == self.selected and m.to_square == sq]
        if candidates:
            if len(candidates) == 1:
                move = candidates[0]
                self.move_requested.emit(chess.square_name(move.from_square), chess.square_name(move.to_square) + (chess.piece_symbol(move.promotion) if move.promotion else ""))
            else:
                self.move_requested.emit(chess.square_name(self.selected), chess.square_name(sq) + "?")
            self.selected = None
            self.update()
            return

        piece = self.board.piece_at(sq)
        self.selected = sq if piece and piece.color == self.board.turn else None
        self.update()
