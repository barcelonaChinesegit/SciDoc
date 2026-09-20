"""Expose categorized source trees without requiring a package install."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "src",
    ROOT / "src/pku_qa/evaluation",
    ROOT / "src/pku_qa/services/task_queue",
    ROOT / "src/pku_qa/services/review",
):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
