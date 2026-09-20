from __future__ import annotations

import json
from pathlib import Path

from pku_qa.workflows.generation.build_cross_pdf_quality_v2 import (
    audit_dataset,
    diversify_question_opening,
    mapped_docs,
    select_zero_overlap_bundles,
    validate_generation,
)


def tiny_manifest(bundle_id: str, source_ids: list[str]) -> dict:
    sources = []
    start = 1
    for source_id in source_ids:
        sources.append(
            {
                "paper_id": source_id,
                "merged_start_page": start,
                "merged_end_page": start + 4,
            }
        )
        start += 5
    return {"id": bundle_id, "sources": sources}


def test_mapped_docs_uses_merged_physical_pages() -> None:
    manifest = tiny_manifest("b1", ["a", "b", "c"])
    assert mapped_docs([1, 7, 15], manifest["sources"]) == {1, 2, 3}


def test_selection_never_reuses_source_papers() -> None:
    qa = {
        key: {"QA": {"Q1": {"question": "x", "modal_types": ["text"]}}}
        for key in ("b1", "b2", "b3")
    }
    manifests = {
        "b1": tiny_manifest("b1", ["a", "b", "c"]),
        "b2": tiny_manifest("b2", ["c", "d", "e"]),
        "b3": tiny_manifest("b3", ["f", "g", "h"]),
    }
    selected = select_zero_overlap_bundles(qa, manifests)
    used = [
        source["paper_id"]
        for bundle_id in selected
        for source in manifests[bundle_id]["sources"]
    ]
    assert len(selected) == 2
    assert len(used) == len(set(used))


def test_generation_requires_all_three_physical_page_ranges() -> None:
    manifest = tiny_manifest("b1", ["a", "b", "c"])
    result = {
        "rewrites": [
            {"qa_id": "Q1", "rewritten_question": "A clearer question."}
        ],
        "tri_candidates": [
            {
                "candidate_id": "T1",
                "question": "q",
                "answer": "a",
                "evidence_pages": [1, 7, 12],
            },
            {
                "candidate_id": "T2",
                "question": "q2",
                "answer": "a2",
                "evidence_pages": [2, 8, 14],
            },
        ],
    }
    assert validate_generation(result, ["Q1"], manifest) == []
    result["tri_candidates"][0]["evidence_pages"] = [1, 7]
    assert "T1:pages_do_not_cover_three_docs" in validate_generation(
        result, ["Q1"], manifest
    )
    result["tri_candidates"] = [result["tri_candidates"][1]]
    assert validate_generation(result, ["Q1"], manifest) == []


def test_fixed_question_openings_are_diversified_without_losing_body() -> None:
    rewritten = diversify_question_opening(
        "How do method A and method B differ?", "bundle/Q1"
    )
    assert not rewritten.casefold().startswith("how do ")
    assert "method A and method B differ?" in rewritten
    assert (
        diversify_question_opening("Why is this hard?", "bundle/Q2")
        == "Why is this hard?"
    )
