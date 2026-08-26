from __future__ import annotations

from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from PySide6.QtCore import QRectF, Signal, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..data import state_dir
from ..sound import MoveSoundBank


_PIECE_FILES = {
    chess.KING: "K",
    chess.QUEEN: "Q",
    chess.ROOK: "R",
    chess.BISHOP: "B",
    chess.KNIGHT: "N",
    chess.PAWN: "P",
}


class BoardWidget(QWidget):
    move_chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.flipped = False
        self.selected: int | None = None
        self.last_move: chess.Move | None = None
        self.input_enabled = True
        self.setMinimumSize(560, 560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.PointingHandCursor)

        piece_dir = Path(__file__).resolve().parent.parent / "assets" / "pieces" / "cburnett"
        self._piece_renderers: dict[tuple[bool, int], QSvgRenderer] = {}
        for color in (chess.WHITE, chess.BLACK):
            prefix = "w" if color == chess.WHITE else "b"
            for piece_type, code in _PIECE_FILES.items():
                renderer = QSvgRenderer(str(piece_dir / f"{prefix}{code}.svg"))
                if not renderer.isValid():
                    raise RuntimeError(f"Invalid/missing chess piece SVG: {prefix}{code}.svg")
                self._piece_renderers[(color, piece_type)] = renderer

    def set_board(self, board: chess.Board, last_move: chess.Move | None = None):
        self.board = board.copy(stack=False)
        self.last_move = last_move
        self.selected = None
        self.update()

    def toggle_flip(self):
        self.flipped = not self.flipped
        self.selected = None
        self.update()

    def _square_from_xy(self, x: float, y: float) -> int | None:
        side = min(self.width(), self.height())
        ox = (self.width() - side) / 2
        oy = (self.height() - side) / 2
        if not (ox <= x < ox + side and oy <= y < oy + side):
            return None
        cell = side / 8
        file_display = int((x - ox) / cell)
        rank_display = int((y - oy) / cell)
        if self.flipped:
            file_idx = 7 - file_display
            rank_idx = rank_display
        else:
            file_idx = file_display
            rank_idx = 7 - rank_display
        return chess.square(file_idx, rank_idx)

    def mousePressEvent(self, event):
        if not self.input_enabled:
            return
        sq = self._square_from_xy(event.position().x(), event.position().y())
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if self.selected is None:
            if piece is not None and piece.color == self.board.turn:
                self.selected = sq
                self.update()
            return
        if sq == self.selected:
            self.selected = None
            self.update()
            return
        candidates = [m for m in self.board.legal_moves if m.from_square == self.selected and m.to_square == sq]
        if candidates:
            # Promotion chooser is a separate UI polish item; queen remains the
            # safe default until that dialog is added.
            move = next((m for m in candidates if m.promotion == chess.QUEEN), candidates[0])
            self.move_chosen.emit(move.uci())
            self.selected = None
            return
        self.selected = sq if piece is not None and piece.color == self.board.turn else None
        self.update()

    def _draw_piece(self, painter: QPainter, piece: chess.Piece, square_rect: QRectF) -> None:
        margin = square_rect.width() * 0.045
        target = square_rect.adjusted(margin, margin, -margin, -margin)
        self._piece_renderers[(piece.color, piece.piece_type)].render(painter, target)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        side = min(self.width(), self.height())
        ox = (self.width() - side) / 2
        oy = (self.height() - side) / 2
        cell = side / 8
        light = QColor("#e6d8be")
        dark = QColor("#8b6b52")
        selected = QColor(92, 142, 255, 150)
        legal = QColor(77, 199, 132, 150)
        last = QColor(235, 197, 77, 125)
        check = QColor(226, 78, 78, 165)
        legal_targets = {
            m.to_square for m in self.board.legal_moves if m.from_square == self.selected
        } if self.selected is not None else set()
        checked_king = self.board.king(self.board.turn) if self.board.is_check() else None

        for display_rank in range(8):
            for display_file in range(8):
                if self.flipped:
                    file_idx = 7 - display_file
                    rank_idx = display_rank
                else:
                    file_idx = display_file
                    rank_idx = 7 - display_rank
                sq = chess.square(file_idx, rank_idx)
                rect = QRectF(ox + display_file * cell, oy + display_rank * cell, cell, cell)
                p.fillRect(rect, light if (file_idx + rank_idx) % 2 == 0 else dark)
                if self.last_move and sq in (self.last_move.from_square, self.last_move.to_square):
                    p.fillRect(rect, last)
                if sq == self.selected:
                    p.fillRect(rect, selected)
                if sq == checked_king:
                    p.fillRect(rect, check)
                if sq in legal_targets:
                    p.setPen(Qt.NoPen)
                    p.setBrush(legal)
                    radius = cell * (0.13 if self.board.piece_at(sq) is None else 0.34)
                    p.drawEllipse(rect.center(), radius, radius)
                piece = self.board.piece_at(sq)
                if piece:
                    self._draw_piece(p, piece, rect)

        coord = QFont()
        coord.setPixelSize(max(9, int(cell * 0.14)))
        coord.setBold(True)
        p.setFont(coord)
        p.setPen(QColor(30, 30, 30, 160))
        for i in range(8):
            file_idx = 7 - i if self.flipped else i
            rank_idx = i if self.flipped else 7 - i
            p.drawText(
                QRectF(ox + i * cell + 4, oy + side - 18, cell - 6, 14),
                Qt.AlignLeft,
                chess.FILE_NAMES[file_idx],
            )
            p.drawText(
                QRectF(ox + 3, oy + i * cell + 2, 18, 14),
                Qt.AlignLeft,
                str(rank_idx + 1),
            )


class PlayPage(QWidget):
    def __init__(self, agent, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.board = chess.Board()
        self.human_color = chess.WHITE
        self.pending_move_request: str | None = None
        self.pending_begin_request: str | None = None
        self.pending_record_request: str | None = None
        self.game_active = False
        self.last_move: chess.Move | None = None
        self.sounds = MoveSoundBank(self)
        self.game: chess.pgn.Game | None = None
        self.node = None
        self.pinned_generation: int | None = None
        self.takebacks = 0
        self.last_saved_path: Path | None = None
        self.live_path: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)
        title_row = QHBoxLayout()
        title = QLabel("Play dog_matist")
        title.setObjectName("SectionTitle")
        title_row.addWidget(title)
        title_row.addStretch()
        self.state_label = QLabel("Ready for a new game")
        self.state_label.setObjectName("RunBadge")
        title_row.addWidget(self.state_label)
        root.addLayout(title_row)

        body = QHBoxLayout()
        body.setSpacing(18)
        self.board_widget = BoardWidget()
        self.board_widget.move_chosen.connect(self._human_move)
        body.addWidget(self.board_widget, 4)

        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setFixedWidth(350)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(18, 18, 18, 18)
        pl.setSpacing(10)
        pl.addWidget(QLabel("New game"))
        self.color_box = QComboBox()
        self.color_box.addItems(["Play White", "Play Black"])
        pl.addWidget(self.color_box)
        self.new_btn = QPushButton("New game")
        self.new_btn.setObjectName("Primary")
        self.new_btn.setCursor(Qt.PointingHandCursor)
        self.new_btn.clicked.connect(self.new_game)
        pl.addWidget(self.new_btn)
        pl.addSpacing(6)

        self.undo_btn = QPushButton("↶  Undo full turn")
        self.resign_btn = QPushButton("Resign")
        self.abort_btn = QPushButton("Abort without saving")
        self.flip_btn = QPushButton("Flip board")
        for btn in (self.undo_btn, self.resign_btn, self.abort_btn, self.flip_btn):
            btn.setCursor(Qt.PointingHandCursor)
            pl.addWidget(btn)
        self.resign_btn.setObjectName("Danger")
        self.abort_btn.setObjectName("Danger")
        self.undo_btn.clicked.connect(self.undo_turn)
        self.resign_btn.clicked.connect(self.resign)
        self.abort_btn.clicked.connect(self.abort_game)
        self.flip_btn.clicked.connect(self.board_widget.toggle_flip)

        self.sound_check = QCheckBox("Move sounds")
        self.sound_check.setChecked(True)
        self.sound_check.toggled.connect(lambda enabled: setattr(self.sounds, "enabled", enabled))
        pl.addWidget(self.sound_check)
        pl.addSpacing(8)

        pl.addWidget(QLabel("Moves"))
        self.moves = QPlainTextEdit()
        self.moves.setReadOnly(True)
        self.moves.setMaximumHeight(170)
        self.moves.setPlaceholderText("Moves will appear here as the game is played.")
        pl.addWidget(self.moves)

        self.autosave = QLabel("No active game autosave.")
        self.autosave.setObjectName("Subtle")
        self.autosave.setWordWrap(True)
        pl.addWidget(self.autosave)

        pl.addWidget(QLabel("Game status"))
        self.info = QLabel(
            "Completed games save to PGN + lifetime memory, but never enter training replay automatically."
        )
        self.info.setObjectName("Subtle")
        self.info.setWordWrap(True)
        pl.addWidget(self.info)
        pl.addStretch()

        help_text = QLabel(
            "Every move is shown here and crash-safely autosaved while the game is in progress. "
            "Each game pins one champion generation; background evolution only affects your next game."
        )
        help_text.setObjectName("Subtle")
        help_text.setWordWrap(True)
        pl.addWidget(help_text)
        body.addWidget(panel)
        root.addLayout(body, 1)

        self.agent.move_ready.connect(self._ai_move_ready)
        self.agent.game_ready.connect(self._game_ready)
        self.agent.game_recorded.connect(self._game_recorded)
        self.agent.error.connect(self._agent_error)
        self._refresh_controls()

    def new_game(self):
        if self.game_active:
            answer = QMessageBox.question(
                self,
                "Start new game",
                "Abort the current game and start a new one? Its in-progress autosave will be deleted.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            self._discard_live_snapshot()
            self._end_core_game()
            self.game_active = False
            self.game = None
            self.node = None

        self.board = chess.Board()
        self.last_move = None
        self.human_color = chess.WHITE if self.color_box.currentIndex() == 0 else chess.BLACK
        self.pending_move_request = None
        self.game_active = False
        self.pinned_generation = None
        self.takebacks = 0
        self.last_saved_path = None
        self.live_path = None
        self.moves.clear()
        self.autosave.setText("Preparing game autosave…")
        self.board_widget.flipped = self.human_color == chess.BLACK
        self.board_widget.set_board(self.board)
        self.state_label.setText("● PINNING CHAMPION…")
        self.info.setText("Loading a fixed champion snapshot for this game.")
        self.pending_begin_request = self.agent.request_begin_game()
        self._refresh_controls()

    def _game_ready(self, request_id: str, payload: object):
        if request_id != self.pending_begin_request:
            return
        self.pending_begin_request = None
        data = payload if isinstance(payload, dict) else {}
        self.pinned_generation = int(data.get("generation")) if data.get("generation") is not None else None
        self.game_active = True
        self._new_pgn()
        gen = f"Gen {self.pinned_generation}" if self.pinned_generation is not None else "current champion"
        self.info.setText(f"Pinned opponent: {gen}. Background evolution can continue safely.")
        self.state_label.setText("● YOUR TURN" if self.board.turn == self.human_color else "● dog_matist THINKING…")
        self._refresh_controls()
        if self.board.turn != self.human_color:
            self._request_ai_move()

    def _new_pgn(self):
        self.game = chess.pgn.Game()
        self.game.headers["Event"] = "dog_matist Studio Human vs Champion"
        self.game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        dog = f"dog_matist-g{self.pinned_generation}" if self.pinned_generation is not None else "dog_matist"
        self.game.headers["White"] = "Human" if self.human_color == chess.WHITE else dog
        self.game.headers["Black"] = dog if self.human_color == chess.WHITE else "Human"
        if self.pinned_generation is not None:
            self.game.headers["DogMatistGeneration"] = str(self.pinned_generation)
        self.node = self.game

        folder = state_dir() / "studio_games" / "in_progress"
        folder.mkdir(parents=True, exist_ok=True)
        generation = f"g{self.pinned_generation}" if self.pinned_generation is not None else "gunknown"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.live_path = folder / f"human_{stamp}_{generation}.pgn"
        self._sync_game_process()

    def _refresh_controls(self):
        human_turn = self.game_active and self.pending_move_request is None and self.board.turn == self.human_color
        self.board_widget.input_enabled = human_turn
        busy = self.pending_begin_request is not None or self.pending_record_request is not None
        self.new_btn.setEnabled(not busy)
        self.color_box.setEnabled(not self.game_active and not busy)
        self.undo_btn.setEnabled(self.game_active and self.pending_move_request is None and len(self.board.move_stack) >= 1)
        self.resign_btn.setEnabled(self.game_active)
        self.abort_btn.setEnabled(self.game_active)

    def _push_move(self, move: chess.Move, was_capture: bool):
        if self.node is not None:
            self.node = self.node.add_variation(move)
        self.board.push(move)
        self.last_move = move
        self._post_move_sound(was_capture)
        self.board_widget.set_board(self.board, move)
        self._sync_game_process()

    def _format_moves(self) -> str:
        replay = chess.Board()
        rows: list[str] = []
        current = ""
        for ply, move in enumerate(self.board.move_stack):
            san = replay.san(move)
            if ply % 2 == 0:
                if current:
                    rows.append(current)
                current = f"{ply // 2 + 1}. {san}"
            else:
                current += f"  {san}"
            replay.push(move)
        if current:
            rows.append(current)
        return "\n".join(rows)

    def _build_clean_pgn(self, result: str, termination: str) -> chess.pgn.Game:
        old_headers = dict(self.game.headers) if self.game is not None else {}
        clean = chess.pgn.Game()
        for key, value in old_headers.items():
            clean.headers[key] = value
        clean.headers["Result"] = result
        clean.headers["Termination"] = termination
        clean.headers["Takebacks"] = str(self.takebacks)
        node = clean
        replay = chess.Board()
        for move in self.board.move_stack:
            if move not in replay.legal_moves:
                raise RuntimeError(f"Cannot rebuild PGN: illegal stored move {move.uci()}")
            node = node.add_variation(move)
            replay.push(move)
        return clean

    def _write_live_snapshot(self) -> None:
        if self.live_path is None or self.game is None:
            return
        snapshot = self._build_clean_pgn("*", "in_progress")
        tmp = self.live_path.with_suffix(".tmp")
        try:
            tmp.write_text(str(snapshot) + "\n\n", encoding="utf-8")
            tmp.replace(self.live_path)
            self.autosave.setText(
                f"● Autosaved after every move · {len(self.board.move_stack)} plies\n{self.live_path}"
            )
        except OSError as exc:
            self.autosave.setText(f"⚠ Autosave failed: {exc}")

    def _sync_game_process(self) -> None:
        text = self._format_moves()
        self.moves.setPlainText(text)
        bar = self.moves.verticalScrollBar()
        bar.setValue(bar.maximum())
        self._write_live_snapshot()

    def _discard_live_snapshot(self) -> None:
        if self.live_path is not None:
            try:
                self.live_path.unlink(missing_ok=True)
                self.live_path.with_suffix(".tmp").unlink(missing_ok=True)
            except OSError:
                pass
        self.live_path = None
        self.autosave.setText("No active game autosave.")

    def _human_move(self, uci: str):
        if not self.game_active or self.board.turn != self.human_color or self.pending_move_request:
            return
        try:
            move = chess.Move.from_uci(uci)
            if move not in self.board.legal_moves:
                return
        except ValueError:
            return
        self._push_move(move, self.board.is_capture(move))
        if self._finish_if_over():
            return
        self._request_ai_move()

    def _request_ai_move(self):
        self.state_label.setText("● dog_matist THINKING…")
        self.pending_move_request = self.agent.request_move(self.board.fen())
        self._refresh_controls()

    def _ai_move_ready(self, request_id: str, uci: str):
        if request_id != self.pending_move_request or not self.game_active:
            return
        self.pending_move_request = None
        try:
            move = chess.Move.from_uci(uci)
            if move not in self.board.legal_moves:
                raise ValueError(f"core returned illegal move {uci}")
            self._push_move(move, self.board.is_capture(move))
        except Exception as exc:
            self.state_label.setText("● ENGINE ERROR")
            recovery = f" In-progress PGN remains at {self.live_path}." if self.live_path else ""
            self.info.setText(str(exc) + recovery)
            self._refresh_controls()
            return
        if not self._finish_if_over():
            self.state_label.setText("● YOUR TURN")
        self._refresh_controls()

    def _post_move_sound(self, was_capture: bool):
        if self.board.is_check():
            self.sounds.play("check")
        elif was_capture:
            self.sounds.play("capture")
        else:
            self.sounds.play("move")

    def _finish_if_over(self) -> bool:
        if not self.board.is_game_over(claim_draw=True):
            return False
        result = self.board.result(claim_draw=True)
        outcome = self.board.outcome(claim_draw=True)
        termination = outcome.termination.name.lower() if outcome else "completed"
        self._complete_game(result, termination)
        return True

    def _save_pgn(self, result: str, termination: str):
        if self.game is None:
            return None
        clean = self._build_clean_pgn(result, termination)
        folder = state_dir() / "studio_games"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"human_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pgn"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(str(clean) + "\n\n", encoding="utf-8")
        tmp.replace(path)
        self.game = clean
        self.node = clean.end()
        self._discard_live_snapshot()
        return path

    def _complete_game(self, result: str, termination: str):
        self.game_active = False
        generation = self.pinned_generation
        try:
            path = self._save_pgn(result, termination)
        except OSError as exc:
            path = None
            self.info.setText(
                f"Final PGN save failed: {exc}. The in-progress autosave was kept at {self.live_path}."
            )
        self.last_saved_path = path
        if self.game is not None and generation is not None and path is not None:
            payload = {
                "pgn": str(self.game),
                "result": result,
                "termination": termination,
                "plies": len(self.board.move_stack),
                "human_color": "white" if self.human_color == chess.WHITE else "black",
                "takebacks": self.takebacks,
                "generation": generation,
            }
            self.pending_record_request = self.agent.request_record_game(payload)
        self._end_core_game()
        self.state_label.setText(f"● GAME OVER · {result}")
        if path:
            self.info.setText(f"Saved final PGN: {path}. Adding encounter to lifetime memory…")
        self.sounds.play("end")
        self._refresh_controls()

    def _game_recorded(self, request_id: str, payload: object):
        if request_id != self.pending_record_request:
            return
        self.pending_record_request = None
        data = payload if isinstance(payload, dict) else {}
        gid = str(data.get("game_id", ""))
        short = gid[:8] if gid else "saved"
        location = f"PGN: {self.last_saved_path}. " if self.last_saved_path else ""
        self.info.setText(f"{location}Lifetime memory: {short} · training replay: OFF")
        self._refresh_controls()

    def undo_turn(self):
        if not self.game_active or self.pending_move_request:
            return
        popped = 0
        while self.board.move_stack and popped < 2:
            self.board.pop()
            if self.node is not None and self.node.parent is not None:
                self.node = self.node.parent
            popped += 1
        while self.board.move_stack and self.board.turn != self.human_color:
            self.board.pop()
            if self.node is not None and self.node.parent is not None:
                self.node = self.node.parent
        self.takebacks += 1
        self.last_move = self.board.peek() if self.board.move_stack else None
        self.board_widget.set_board(self.board, self.last_move)
        self._sync_game_process()
        self.state_label.setText("● YOUR TURN")
        self.info.setText(f"Takeback #{self.takebacks} recorded. In-progress PGN was rewritten to the surviving line.")
        self._refresh_controls()

    def resign(self):
        if not self.game_active:
            return
        answer = QMessageBox.question(
            self,
            "Resign",
            "Resign this game?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        result = "0-1" if self.human_color == chess.WHITE else "1-0"
        self._complete_game(result, "human_resignation")
        self.state_label.setText(f"● RESIGNED · {result}")

    def abort_game(self):
        if not self.game_active:
            return
        answer = QMessageBox.question(
            self,
            "Abort game",
            "Abort this game without saving it? The in-progress autosave will also be deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return
        self.game_active = False
        self.pending_move_request = None
        self._discard_live_snapshot()
        self._end_core_game()
        self.board = chess.Board()
        self.last_move = None
        self.game = None
        self.node = None
        self.moves.clear()
        self.board_widget.set_board(self.board)
        self.state_label.setText("● ABORTED")
        self.info.setText("Game discarded: no PGN, no autosave, and no lifetime-memory record remain.")
        self._refresh_controls()

    def _end_core_game(self):
        if self.pinned_generation is not None:
            self.agent.request_end_game()
        self.pinned_generation = None

    def _agent_error(self, request_id: str, message: str):
        relevant = {self.pending_move_request, self.pending_begin_request, self.pending_record_request}
        if request_id not in relevant:
            return
        if request_id == self.pending_begin_request:
            self.pending_begin_request = None
        if request_id == self.pending_move_request:
            self.pending_move_request = None
        if request_id == self.pending_record_request:
            self.pending_record_request = None
            prefix = (
                f"PGN saved at {self.last_saved_path}, but lifetime-memory write failed: "
                if self.last_saved_path
                else "Lifetime-memory write failed: "
            )
            self.info.setText(prefix + message)
            self._refresh_controls()
            return
        self.state_label.setText("● ENGINE ERROR")
        recovery = f" In-progress PGN is safe at {self.live_path}." if self.live_path else ""
        self.info.setText(message + recovery)
        self._refresh_controls()
