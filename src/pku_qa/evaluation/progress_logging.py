"""Small, dependency-free helpers for machine-readable progress logs."""

from __future__ import annotations


def progress_percent(completed: int | float, total: int | float) -> float:
    """Return a bounded percentage; an empty workload is already complete."""

    total_value = float(total)
    if total_value <= 0:
        return 100.0
    value = 100.0 * float(completed) / total_value
    return max(0.0, min(100.0, value))


def progress_fields(completed: int | float, total: int | float) -> str:
    """Format the canonical progress fields appended to runtime log events."""

    return (
        f"progress={completed}/{total} "
        f"progress_percent={progress_percent(completed, total):.2f}%"
    )
