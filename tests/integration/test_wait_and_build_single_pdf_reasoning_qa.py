from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/generation/wait_and_build_single_pdf_reasoning_qa.py"
)
SPEC = importlib.util.spec_from_file_location("wait_build_reasoning", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_progress(path: Path, **overrides: object) -> None:
    value = {
        "status": "search_space_exhausted",
        "completed": 2457,
        "total": 3000,
        "exhausted_generation_rounds": 7,
        **overrides,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def test_same_exhausted_round_budget_raises_actionable_error(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress.json"
    write_progress(progress)
    with pytest.raises(MODULE.SearchSpaceExhaustedError, match="increase"):
        MODULE.is_complete_or_raise(progress, 3000, 7)


def test_expanded_round_budget_allows_resume(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    write_progress(progress)
    assert MODULE.is_complete_or_raise(progress, 3000, 12) is False


def test_target_reached_wins_over_stale_exhaustion_state(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    write_progress(progress, completed=3000)
    assert MODULE.is_complete_or_raise(progress, 3000, 7) is True
