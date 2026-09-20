from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/selection/build_hard_benchmark_selection_pool.py"
)
SPEC = importlib.util.spec_from_file_location("build_hard_selection_pool", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(selection_id: str, is_new: bool, score: int) -> dict:
    return {
        "selection_id": selection_id,
        "dataset": selection_id.split(":", 1)[0],
        "paper_or_bundle_id": "p",
        "qa_id": selection_id,
        "is_new": is_new,
        "hardness_priority_score": score,
        "hardness_priority_reasons": [],
        "qa": {},
    }


def test_recommended_selection_keeps_all_new_then_hardest_old() -> None:
    rows = [
        row("new:a", True, 100),
        row("new:b", True, 100),
        row("old:easy", False, 0),
        row("old:hard", False, 75),
    ]
    chosen = MODULE.select_recommended(rows, 3)
    assert {item["selection_id"] for item in chosen} == {
        "new:a",
        "new:b",
        "old:hard",
    }


def test_scoring_prioritizes_manual_and_local_errors() -> None:
    rows = [
        {
            "selection_id": "reasoning:p/QA1",
            "dataset": "reasoning",
            "paper_or_bundle_id": "p",
            "qa_id": "QA1",
            "is_new": False,
            "qa": {"question_type": "Inferential", "evidence_pages": [2, 8]},
        }
    ]
    key = ("reasoning", "p/QA1")
    MODULE.score_rows(rows, {key}, {"4b": {key}, "8b": set()})
    assert rows[0]["hardness_priority_score"] == 133
    assert rows[0]["hardness_priority_reasons"] == [
        "gpt5_6_manual_incorrect",
        "local_qwen_4b_incorrect",
        "inferential",
        "nonlocal_evidence",
    ]
