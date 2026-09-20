from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/cleaning/apply_hard_cross_pdf_review_fixes.py"
)
SPEC = importlib.util.spec_from_file_location("apply_hard_cross_fixes", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_apply_evidence_only_fix_recomputes_document_metadata() -> None:
    qa = {
        "question": "Why are the methods incompatible?",
        "answer": "Their prerequisites conflict.",
        "evidence_pages": [4, 11, 19],
        "source_doc_numbers": [1, 2, 3],
        "source_document_count": 3,
        "evidence_span": 15,
    }
    review = {
        "decision": "FIX",
        "corrected_question": None,
        "corrected_answer": None,
        "corrected_evidence_pages": [4, 10, 18],
    }
    manifest = {
        "sources": [
            {"merged_start_page": 1, "merged_end_page": 7},
            {"merged_start_page": 8, "merged_end_page": 14},
            {"merged_start_page": 15, "merged_end_page": 22},
        ]
    }
    corrected, changed = MODULE.apply_fix(qa, review, manifest)
    assert changed == ["evidence_pages"]
    assert corrected["evidence_pages"] == [4, 10, 18]
    assert corrected["source_doc_numbers"] == [1, 2, 3]
    assert corrected["source_document_count"] == 3
    assert corrected["evidence_span"] == 14
