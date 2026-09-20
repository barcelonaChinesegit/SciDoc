from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/review/finalize_hard_cross_pdf_pool_set.py"
)
SPEC = importlib.util.spec_from_file_location("finalize_hard_cross_pool_set", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def item(pool: str, qa_id: str, question: str, confidence: float) -> dict:
    return {
        "_pool_name": pool,
        "_bundle_id": "xb_0001",
        "_source_qa_id": qa_id,
        "question": question,
        "answer": "One supported decision.",
        "source_document_count": 3,
        "reasoning_type": "compatibility_judgment",
        "dual_model_validation": {
            "claude": {"confidence": confidence},
            "gemini": {"confidence": confidence},
        },
    }


def test_cross_pool_dedup_prefers_higher_dual_confidence() -> None:
    low = item("pool-low", "QA1", "Can the metric satisfy this constraint?", 0.82)
    high = item("pool-high", "QA2", "Can the metric satisfy this constraint?", 0.96)
    distinct = item("pool-low", "QA3", "Which module must run first?", 0.85)
    kept, rejected = MODULE.remove_cross_pool_duplicates([low, high, distinct])
    assert {(row["_pool_name"], row["_source_qa_id"]) for row in kept} == {
        ("pool-high", "QA2"),
        ("pool-low", "QA3"),
    }
    assert rejected == [
        {
            "pool": "pool-low",
            "bundle_id": "xb_0001",
            "qa_id": "QA1",
            "reason": "near_duplicate_across_pools",
            "kept_pool": "pool-high",
            "kept_qa_id": "QA2",
        }
    ]


def test_build_release_records_candidate_pool() -> None:
    chosen = [item("pool-a", "QA1", "Which dependency is required?", 0.9)]
    candidates = {
        "pool-a": {
            "xb_0001": {
                "primary_category": "Computer Science",
                "secondary_category": "AI",
            }
        }
    }
    release = MODULE.build_release(candidates, chosen)
    released = next(iter(release["xb_0001"]["QA"].values()))
    assert released["candidate_pool"] == "pool-a"
    assert released["candidate_qa_id"] == "QA1"
    assert released["release_rank"] == 1
