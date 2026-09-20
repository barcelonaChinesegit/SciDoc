from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/review/finalize_hard_cross_pdf_qa.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_hard_cross", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def item(bundle: str, index: int, docs: int, reasoning_type: str) -> dict:
    return {
        "question": f"Derived pipeline question {bundle} {index}?",
        "answer": f"Derived answer {bundle} {index}.",
        "source_doc_numbers": list(range(1, docs + 1)),
        "source_document_count": docs,
        "reasoning_type": reasoning_type,
        "evidence_pages": [1, 12, 24][:docs],
        "intermediate_facts": ["bridge"],
        "_bundle_id": bundle,
        "_source_qa_id": f"HQA{index}",
    }


def test_select_items_meets_three_document_quota_and_bundle_cap() -> None:
    eligible = [
        item("b1", 1, 3, "adjacent_module_dependency"),
        item("b1", 2, 3, "metric_reasoning"),
        item("b2", 3, 3, "component_hierarchy"),
        item("b2", 4, 2, "method_transfer"),
        item("b3", 5, 2, "compatibility_judgment"),
    ]
    chosen = MODULE.select_items(eligible, 4, 0.5, 2)
    assert len(chosen) == 4
    assert sum(row["source_document_count"] == 3 for row in chosen) >= 2
    assert max(
        sum(row["_bundle_id"] == bundle for row in chosen)
        for bundle in {row["_bundle_id"] for row in chosen}
    ) <= 2


def test_static_gate_rejects_answer_leakage() -> None:
    qa = item("b1", 1, 3, "component_hierarchy")
    qa["question"] = "Is derived answer b1 1 the result?"
    assert "answer_leakage" in MODULE.candidate_static_errors(qa)


def test_strict_keep_accepts_only_noop_fix() -> None:
    qa = item("b1", 1, 3, "metric_reasoning")
    review = {
        "decision": "FIX",
        "confidence": 0.9,
        "criteria": {name: True for name in MODULE.CRITERIA},
        "corrected_evidence_pages": list(qa["evidence_pages"]),
    }
    assert MODULE.strict_keep(review, qa, 0.8)
    review["corrected_evidence_pages"] = [1, 13, 24]
    assert not MODULE.strict_keep(review, qa, 0.8)
