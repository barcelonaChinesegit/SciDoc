"""PKU multimodal paper-QA evaluation and data-maintenance package.

The evaluation engine retains basename imports because those exact source
bytes are bound into completed publication manifests.  Adding the categorized
engine and service directories to ``sys.path`` preserves those audited bytes
while exposing a conventional ``pku_qa`` package for all new commands.
"""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
for source_dir in (
    PACKAGE_ROOT / "evaluation",
    PACKAGE_ROOT / "services/task_queue",
    PACKAGE_ROOT / "services/review",
):
    value = str(source_dir)
    if value not in sys.path:
        sys.path.insert(0, value)
