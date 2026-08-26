from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import platform
import re
import subprocess
from typing import Any


@dataclass(frozen=True)
class ResourceSnapshot:
    load_percent: float | None
    memory_percent: float | None
    thermal_pressure: str
    cpu_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerBudget:
    selfplay_workers: int
    arena_workers: int
    trainer_slots: int
    reason: str
    snapshot: ResourceSnapshot

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["snapshot"] = self.snapshot.to_dict()
        return out


def _run_text(argv: list[str], timeout: float = 1.5) -> str:
    try:
        return subprocess.check_output(argv, stderr=subprocess.DEVNULL, text=True, timeout=timeout).strip()
    except Exception:
        return ""


def _load_percent(cpu_count: int) -> float | None:
    try:
        one_minute = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None
    return max(0.0, 100.0 * one_minute / max(1, cpu_count))


def _linux_memory_percent() -> float | None:
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                key, rest = line.split(":", 1)
                values[key] = int(rest.strip().split()[0])
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        if total <= 0:
            return None
        return 100.0 * (total - available) / total
    except Exception:
        return None


def _mac_memory_percent() -> float | None:
    total_text = _run_text(["sysctl", "-n", "hw.memsize"])
    vm_text = _run_text(["vm_stat"])
    try:
        total = int(total_text)
    except ValueError:
        return None
    if total <= 0 or not vm_text:
        return None
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", vm_text)
    page_size = int(page_match.group(1)) if page_match else 4096
    pages: dict[str, int] = {}
    for line in vm_text.splitlines():
        match = re.match(r"([^:]+):\s*([0-9]+)\.?", line.strip())
        if match:
            pages[match.group(1)] = int(match.group(2))
    # Inactive/file-cache pages are intentionally treated as reclaimable. This
    # estimate is for headroom control, not Activity Monitor accounting.
    used_pages = (
        pages.get("Pages active", 0)
        + pages.get("Pages wired down", 0)
        + pages.get("Pages occupied by compressor", 0)
        + pages.get("Pages speculative", 0)
    )
    return max(0.0, min(100.0, 100.0 * used_pages * page_size / total))


def _thermal_pressure() -> str:
    if platform.system() != "Darwin":
        return "unknown"
    text = _run_text(["pmset", "-g", "therm"])
    if not text:
        return "unknown"
    limits: list[int] = []
    for key in ("CPU_Speed_Limit", "Scheduler_Limit"):
        match = re.search(rf"{key}\s*=\s*(\d+)", text)
        if match:
            limits.append(int(match.group(1)))
    if not limits:
        return "nominal"
    floor = min(limits)
    if floor < 60:
        return "critical"
    if floor < 80:
        return "serious"
    if floor < 95:
        return "fair"
    return "nominal"


def sample_resources() -> ResourceSnapshot:
    cpu_count = max(1, os.cpu_count() or 1)
    system = platform.system()
    if system == "Darwin":
        memory = _mac_memory_percent()
    elif system == "Linux":
        memory = _linux_memory_percent()
    else:
        memory = None
    return ResourceSnapshot(
        load_percent=_load_percent(cpu_count),
        memory_percent=memory,
        thermal_pressure=_thermal_pressure(),
        cpu_count=cpu_count,
    )


def choose_worker_budget(config: dict[str, Any]) -> WorkerBudget:
    runtime = config.get("runtime", {})
    configured_sp = max(1, int(runtime.get("selfplay_workers", 1)))
    configured_arena = max(1, int(runtime.get("arena_workers", 1)))
    adaptive = bool(runtime.get("adaptive_workers", runtime.get("mode") == "night"))
    snapshot = sample_resources()
    if not adaptive:
        return WorkerBudget(configured_sp, configured_arena, 1, "fixed profile", snapshot)

    load = snapshot.load_percent if snapshot.load_percent is not None else 0.0
    memory = snapshot.memory_percent if snapshot.memory_percent is not None else 0.0
    thermal = snapshot.thermal_pressure
    if thermal in {"critical", "serious"} or memory >= 86.0 or load >= 92.0:
        return WorkerBudget(1, 1, 1, "high system pressure", snapshot)
    if thermal == "fair" or memory >= 78.0 or load >= 78.0:
        return WorkerBudget(max(1, configured_sp - 1), 1, 1, "moderate system pressure", snapshot)
    return WorkerBudget(configured_sp, configured_arena, 1, "headroom available", snapshot)
