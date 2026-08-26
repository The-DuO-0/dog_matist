from __future__ import annotations

from pathlib import Path
import os


class EvolutionAlreadyRunning(RuntimeError):
    pass


class EvolutionLock:
    """Cross-process single-writer lock for evolution/training.

    Human play and read-only/status runtimes intentionally do not acquire this
    lock. Only evolution owns it, so Play can run at the same time while two
    independent evolution processes cannot mutate the lineage concurrently.
    """

    def __init__(self, state_root: str | Path):
        self.path = Path(state_root) / "evolution.lock"
        self._fh = None

    def __enter__(self) -> "EvolutionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+", encoding="utf-8")
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.seek(0)
            owner = self._fh.read().strip() or "another process"
            self._fh.close()
            self._fh = None
            raise EvolutionAlreadyRunning(
                f"Evolution is already running ({owner}). Play/status may still run normally."
            ) from exc
        except ImportError:
            # dog_matist currently targets macOS/Linux for local training. On a
            # platform without fcntl, fail closed instead of pretending the
            # lineage is protected.
            self._fh.close()
            self._fh = None
            raise RuntimeError("Evolution locking requires POSIX fcntl support")

        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"pid={os.getpid()}")
        self._fh.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is None:
            return
        try:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None
