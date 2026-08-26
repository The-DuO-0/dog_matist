from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class FixedReferenceFailureReport:
    error: str

    def ui_payload(self) -> dict[str, object]:
        return {
            "reference": None,
            "subject_generation": None,
            "result": None,
            "skipped_reason": "fixed_reference_error",
            "error": self.error,
        }


@contextmanager
def fixed_reference_fail_open(coordinator: Any, *, enabled: bool = True) -> Iterator[None]:
    """Keep an experimental reference meter from wasting an overnight run."""

    if not enabled:
        yield
        return
    original = coordinator.evaluate_cycle

    def safe_evaluate(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            report = FixedReferenceFailureReport(f"{type(exc).__name__}: {exc}")
            coordinator.last_report = report
            return report

    coordinator.evaluate_cycle = safe_evaluate
    try:
        yield
    finally:
        try:
            delattr(coordinator, "evaluate_cycle")
        except AttributeError:
            coordinator.evaluate_cycle = original
