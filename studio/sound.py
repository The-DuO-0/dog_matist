from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QSoundEffect


def _tone(path: Path, hz: float, duration: float, volume: float = 0.22):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 22050
    frames = int(rate * duration)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        buf = bytearray()
        for i in range(frames):
            t = i / rate
            env = min(1.0, t / 0.008) * max(0.0, 1.0 - t / duration)
            sample = int(32767 * volume * env * math.sin(2 * math.pi * hz * t))
            buf.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(buf))


class MoveSoundBank:
    def __init__(self, parent=None):
        root = Path.home() / ".darwinchess" / "ui_sounds"
        specs = {
            "move": (520, 0.065),
            "capture": (310, 0.095),
            "check": (760, 0.12),
            "end": (430, 0.20),
        }
        self.effects = {}
        for name, (hz, duration) in specs.items():
            path = root / f"{name}.wav"
            try:
                _tone(path, hz, duration)
                fx = QSoundEffect(parent)
                fx.setSource(QUrl.fromLocalFile(str(path)))
                fx.setVolume(0.55)
                self.effects[name] = fx
            except Exception:
                pass
        self.enabled = True

    def play(self, name: str):
        if not self.enabled:
            return
        fx = self.effects.get(name) or self.effects.get("move")
        if fx is not None:
            fx.play()
