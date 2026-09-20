from __future__ import annotations

import json
from pathlib import Path

from pku_qa.workflows.generation.build_cross_pdf_challenge_v3 import (
    STRICT_CRITERIA,
    ledger_evidence_items,
    normalize_candidate,
    parse_json_response_lenient,
    strict_new_items,
)


def sources() -> list[dict]:
    return [
        {
            "paper_id": "a",
            "merged_start_page": 1,
            "merged_end_page": 5,
            "source_page_count": 5,
        },
        {
            "paper_id": "b",
            "merged_start_page": 6,
            "merged_end_page": 10,
            "source_page_count": 5,
        },
        {
            "paper_id": "c",
            "merged_start_page": 11,
            "merged_end_page": 15,
            "source_page_count": 5,
        },
    ]


def candidate(candidate_id: str = "C01") -> dict:
    return {
        "candidate_id": candidate_id,
        "question": "Which joint diagnosis follows from all three mechanisms?",
        "answer": "The mechanisms jointly imply the diagnosis.",
        "integrated_conclusion": "Only the combined evidence fixes the diagnosis.",
        "ablation_failure": {
            "without_doc_1": "The initial failure is unknown.",
            "without_doc_2": "The intervention is unknown.",
            "without_doc_3": "The validation result is unknown.",
        },
        "evidence_pages": [2, 7, 12],
        "evidence_items": [
            {
                "doc_number": 1,
                "physical_pdf_page": 2,
                "supported_fact": "Fact A",
            },
            {
                "doc_number": 2,
                "physical_pdf_page": 7,
                "supported_fact": "Fact B",
            },
            {
                "doc_number": 3,
                "physical_pdf_page": 12,
                "supported_fact": "Fact C",
            },
        ],
    }


def test_lenient_json_parser_preserves_invalid_latex_escape() -> None:
    parsed = parse_json_response_lenient(
        r'{"answer":"Use \alpha and \mathbb{F}_2."}'
    )
    assert parsed["answer"] == r"Use \alpha and \mathbb{F}_2."


def test_ledger_repairs_source_local_page_to_merged_physical_page() -> None:
    ledger = {
        "claude_support": [
            {
                "doc_number": 2,
                "merged_page": 3,
                "supported_fact": "A fact on source-local page 3.",
            }
        ]
    }
    evidence, repairs = ledger_evidence_items(ledger, sources())
    assert evidence[0]["physical_pdf_page"] == 8
    assert repairs == [
        {
            "doc_number": 2,
            "reported_page": 3,
            "resolved_physical_pdf_page": 8,
            "reason": (
                "reported page did not map to the declared document "
                "but was valid as a source-local page"
            ),
        }
    ]


def test_candidate_requires_support_from_all_three_documents() -> None:
    row = candidate()
    assert normalize_candidate(row, sources()) is not None
    row["evidence_items"] = row["evidence_items"][:2]
    assert normalize_candidate(row, sources()) is None


def test_manual_allowlist_filters_other_model_accepts(tmp_path: Path) -> None:
    generation_dir = tmp_path / "generation"
    review_dir = tmp_path / "reviews"
    generation_dir.mkdir()
    review_dir.mkdir()
    rows = [candidate("C01"), candidate("C02")]
    (generation_dir / "xb_test.json").write_text(
        json.dumps(
            {
                "valid_candidates": rows,
                "validation_errors": [],
            }
        ),
        encoding="utf-8",
    )
    review_items = []
    for row in rows:
        review_items.append(
            {
                "candidate_id": row["candidate_id"],
                "decision": "KEEP",
                "criteria": {name: True for name in STRICT_CRITERIA},
                "confidence": 0.95,
                "reason": "All mechanisms jointly determine one conclusion.",
                "verified_support": row["evidence_items"],
            }
        )
    (review_dir / "xb_test.json").write_text(
        json.dumps(
            {
                "review": {"items": review_items},
                "validation_errors": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manual_new_acceptance.json").write_text(
        json.dumps(
            {
                "accepted": [
                    {"bundle_id": "xb_test", "candidate_id": "C02"}
                ]
            }
        ),
        encoding="utf-8",
    )

    accepted = strict_new_items(
        {"xb_test": {"sources": sources()}}, tmp_path
    )
    assert [row[0] for row in accepted["xb_test"]] == ["C02"]
