from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/reporting/prepare_hard_expansion_pilot_eval.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_hard_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def dataset(prefix: str, spans: list[int]) -> dict:
    return {
        str(index): {
            "paper": str(index),
            "QA": {
                f"{prefix}{index}": {
                    "question": f"Question {prefix}{index}?",
                    "answer": f"Answer {prefix}{index}",
                    "evidence_pages": [1, 1 + span],
                    "source_doc_numbers": [1, 2, 3] if index == 1 else [1, 2],
                }
            },
        }
        for index, span in enumerate(spans, start=1)
    }


def test_cohort_dataset_uses_equal_counts_and_preserves_labels() -> None:
    new = dataset("N", [3, 4])
    old = dataset("O", [1, 9, 5])
    result, summary = MODULE.cohort_dataset(new, old, "reasoning")
    items = [qa for paper in result.values() for qa in paper["QA"].values()]
    assert summary["old_baseline"] == summary["new_hard_expansion"] == 2
    assert {qa["benchmark_cohort"] for qa in items} == {
        "old_baseline",
        "new_hard_expansion",
    }
    old_spans = sorted(
        MODULE.evidence_span(qa)
        for qa in items
        if qa["benchmark_cohort"] == "old_baseline"
    )
    assert old_spans == [5, 9]
