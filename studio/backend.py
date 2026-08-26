from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QThread, Signal, Slot


def dog_executable() -> str:
    bindir = Path(sys.executable).resolve().parent
    for name in ("dog-matist", "darwinchess"):
        local = bindir / name
        if local.exists():
            return str(local)
        found = shutil.which(name)
        if found:
            return found
    return "darwinchess"


def flatten_mapping(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, val in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(val, dict):
                out.update(flatten_mapping(val, path))
            else:
                out[path] = val
    else:
        out[prefix or "value"] = value
    return out


def find_state_value(state: dict[str, Any], *aliases: str, default: Any = "—") -> Any:
    flat = flatten_mapping(state)
    aliases = tuple(a.lower().replace(" ", "_") for a in aliases)
    for key, value in flat.items():
        leaf = key.split(".")[-1].lower().replace(" ", "_")
        if leaf in aliases:
            return value
    for key, value in flat.items():
        norm = key.lower().replace(" ", "_")
        if any(a in norm for a in aliases):
            return value
    return default


def normalize_move(answer: Any) -> str:
    if answer is None:
        raise ValueError("dog_matist returned no move")
    if isinstance(answer, dict):
        for key in ("move_uci", "move", "best_move", "uci", "bestmove"):
            if key in answer:
                return normalize_move(answer[key])
    for attr in ("move_uci", "move", "best_move", "uci"):
        if hasattr(answer, attr):
            obj = getattr(answer, attr)
            obj = obj() if callable(obj) else obj
            return normalize_move(obj)
    text = str(answer).strip()
    match = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", text.lower())
    if match:
        return match.group(1)
    return text


class _AgentWorker(QObject):
    ready = Signal()
    status_ready = Signal(str, object)
    move_ready = Signal(str, str)
    talk_ready = Signal(str, str)
    game_ready = Signal(str, object)
    game_ended = Signal(str)
    game_recorded = Signal(str, object)
    error = Signal(str, str)
    stopped = Signal()

    def __init__(self, mode: str = "normal") -> None:
        super().__init__()
        self.mode = mode
        self.agent = None
        self._entered = False

    @Slot()
    def initialize(self) -> None:
        try:
            from darwinchess.api import DogMatistAgent
            obj = DogMatistAgent(mode=self.mode)
            if hasattr(obj, "__enter__"):
                entered = obj.__enter__()
                self.agent = entered if entered is not None else obj
                self._entered = True
            else:
                self.agent = obj
            self.ready.emit()
        except Exception as exc:
            self.error.emit("startup", f"Could not start dog_matist agent: {exc}")

    @Slot(str)
    def get_status(self, request_id: str) -> None:
        try:
            result = self.agent.status()
            if not isinstance(result, dict):
                result = {"status": result}
            self.status_ready.emit(request_id, result)
        except Exception as exc:
            self.error.emit(request_id, str(exc))

    @Slot(str, str)
    def get_best_move(self, request_id: str, fen: str) -> None:
        try:
            self.move_ready.emit(request_id, normalize_move(self.agent.best_move(fen)))
        except Exception as exc:
            self.error.emit(request_id, str(exc))

    @Slot(str, str)
    def talk(self, request_id: str, text: str) -> None:
        try:
            self.talk_ready.emit(request_id, str(self.agent.talk(text)))
        except Exception as exc:
            self.error.emit(request_id, str(exc))

    @Slot(str)
    def begin_game(self, request_id: str) -> None:
        try:
            self.game_ready.emit(request_id, self.agent.begin_game())
        except Exception as exc:
            self.error.emit(request_id, str(exc))

    @Slot(str)
    def end_game(self, request_id: str) -> None:
        try:
            self.agent.end_game()
            self.game_ended.emit(request_id)
        except Exception as exc:
            self.error.emit(request_id, str(exc))

    @Slot(str, object)
    def record_game(self, request_id: str, payload: object) -> None:
        try:
            data = payload if isinstance(payload, dict) else {}
            result = self.agent.record_human_game(**data)
            self.game_recorded.emit(request_id, result)
        except Exception as exc:
            self.error.emit(request_id, str(exc))

    @Slot()
    def shutdown(self) -> None:
        try:
            if self._entered and self.agent is not None and hasattr(self.agent, "__exit__"):
                self.agent.__exit__(None, None, None)
            elif self.agent is not None and hasattr(self.agent, "close"):
                self.agent.close()
        finally:
            self.agent = None
            self.stopped.emit()


class AgentBridge(QObject):
    status_request = Signal(str)
    move_request = Signal(str, str)
    talk_request = Signal(str, str)
    begin_game_request = Signal(str)
    end_game_request = Signal(str)
    record_game_request = Signal(str, object)
    shutdown_request = Signal()

    ready = Signal()
    status_ready = Signal(str, object)
    move_ready = Signal(str, str)
    talk_ready = Signal(str, str)
    game_ready = Signal(str, object)
    game_ended = Signal(str)
    game_recorded = Signal(str, object)
    error = Signal(str, str)

    def __init__(self, mode: str = "normal", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.thread = QThread(self)
        self.worker = _AgentWorker(mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.initialize)
        self.status_request.connect(self.worker.get_status)
        self.move_request.connect(self.worker.get_best_move)
        self.talk_request.connect(self.worker.talk)
        self.begin_game_request.connect(self.worker.begin_game)
        self.end_game_request.connect(self.worker.end_game)
        self.record_game_request.connect(self.worker.record_game)
        self.shutdown_request.connect(self.worker.shutdown)
        self.worker.ready.connect(self.ready)
        self.worker.status_ready.connect(self.status_ready)
        self.worker.move_ready.connect(self.move_ready)
        self.worker.talk_ready.connect(self.talk_ready)
        self.worker.game_ready.connect(self.game_ready)
        self.worker.game_ended.connect(self.game_ended)
        self.worker.game_recorded.connect(self.game_recorded)
        self.worker.error.connect(self.error)
        self.worker.stopped.connect(self.thread.quit)
        self.thread.start()

    def request_status(self) -> str:
        rid = uuid.uuid4().hex
        self.status_request.emit(rid)
        return rid

    def request_move(self, fen: str) -> str:
        rid = uuid.uuid4().hex
        self.move_request.emit(rid, fen)
        return rid

    def request_talk(self, text: str) -> str:
        rid = uuid.uuid4().hex
        self.talk_request.emit(rid, text)
        return rid

    def request_begin_game(self) -> str:
        rid = uuid.uuid4().hex
        self.begin_game_request.emit(rid)
        return rid

    def request_end_game(self) -> str:
        rid = uuid.uuid4().hex
        self.end_game_request.emit(rid)
        return rid

    def request_record_game(self, payload: dict[str, Any]) -> str:
        rid = uuid.uuid4().hex
        self.record_game_request.emit(rid, payload)
        return rid

    def close(self) -> None:
        if self.thread.isRunning():
            self.shutdown_request.emit()
            self.thread.wait(4000)


class ProcessController(QObject):
    output = Signal(str)
    started = Signal(str)
    finished = Signal(int)
    state_changed = Signal(bool)
    stage_changed = Signal(str, str)
    ui_event = Signal(object)

    _stage_re = re.compile(r"\[dog_matist\]\[stage=([^\]]+)\](?:\[detail=([^\]]*)\])?")
    _ui_prefix = "DOGMATIST_UI "

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._on_started)
        self.process.finished.connect(self._on_finished)
        self.label = ""
        self._line_buffer = ""

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def start(self, args: list[str], label: str = "dog_matist") -> bool:
        if self.running:
            return False
        self.label = label
        self._line_buffer = ""
        self.process.setProgram(dog_executable())
        self.process.setArguments(args)
        env = self.process.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.start()
        return True

    def start_evolution(self, mode: str, cycles: int, hours: float) -> bool:
        args = ["--mode", mode, "evolve"]
        if hours > 0:
            args += ["--hours", str(hours)]
            label = f"Evolution ({mode}, {hours:g}h active compute)"
        else:
            args += ["--cycles", str(cycles)]
            label = f"Evolution ({mode}, {cycles} cycle{'s' if cycles != 1 else ''})"
        return self.start(args, label)

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        if line.startswith(self._ui_prefix):
            raw_json = line[len(self._ui_prefix):].strip()
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                # Keep malformed telemetry visible in the ordinary log; never let
                # one partial/old line crash the Studio event loop.
                return
            if isinstance(payload, dict):
                self.ui_event.emit(payload)
                phase = str(payload.get("phase") or "").strip()
                if phase:
                    detail = ""
                    if phase == "league":
                        league = payload.get("league") or {}
                        if isinstance(league, dict):
                            active = league.get("active_games") or []
                            detail = f"{len(active)} live · {league.get('parallel_games', '—')} slots"
                    elif phase == "strength_lab":
                        mode = payload.get("mode") or (payload.get("plan") or {}).get("mode")
                        detail = str(mode or "")
                    elif phase == "safe_stop_requested":
                        detail = "Finishing current colour pair(s)"
                    self.stage_changed.emit(phase.replace("_", "-"), detail)
            return

        match = self._stage_re.search(line)
        if match:
            self.stage_changed.emit(match.group(1).strip(), (match.group(2) or "").strip())
        elif line.startswith("[cycle ") and "self-play" in line:
            self.stage_changed.emit("self-play", "Generating diversified opening games")
        elif '"promoted": true' in line.lower():
            self.stage_changed.emit("promoted", "Candidate became the new champion")
        elif '"promoted": false' in line.lower():
            self.stage_changed.emit("rejected", "Champion retained")

    @Slot()
    def _read_output(self) -> None:
        raw = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not raw:
            return
        self.output.emit(raw.rstrip())
        self._line_buffer += raw
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    @Slot()
    def _on_started(self) -> None:
        self.state_changed.emit(True)
        self.started.emit(self.label)
        self.stage_changed.emit("starting", "Launching dog_matist")

    @Slot(int, QProcess.ExitStatus)
    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if self._line_buffer.strip():
            self._handle_line(self._line_buffer.strip())
        self._line_buffer = ""
        self.state_changed.emit(False)
        self.finished.emit(exit_code)

    def stop_safely(self) -> None:
        if not self.running:
            return
        self.stage_changed.emit("safe-stop-requested", "Finishing current colour pair(s)")
        pid = int(self.process.processId())
        if pid > 0 and os.name == "posix":
            try:
                os.kill(pid, signal.SIGINT)
                self.output.emit(
                    "[Studio] Safe stop requested. dog_matist will finish already-started colour pair(s), then stop."
                )
                return
            except OSError:
                pass
        self.process.terminate()
