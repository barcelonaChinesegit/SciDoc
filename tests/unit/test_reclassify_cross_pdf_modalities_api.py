import pytest

from pku_qa.workflows.cleaning.reclassify_cross_pdf_modalities_api import (
    REVIEW_VERSION,
    apply_decisions,
    qa_fingerprint,
    validate_response,
)


def qa() -> dict:
    return {
        "question": "What value does the table report?",
        "answer": "42",
        "evidence_pages": [3],
        "modal_types": ["text"],
        "annotation_provenance": {},
    }


def test_validate_response_accepts_canonical_multimodal_decision() -> None:
    qas = {"Q1": qa()}
    result = validate_response(
        "b1",
        qas,
        {
            "bundle_id": "b1",
            "items": [{
                "qa_id": "Q1",
                "modal_types": ["text", "table"],
                "confidence": 0.95,
                "rationale": "The answer uses a row label and its cell value.",
                "evidence": [{"page": 3, "modalities": ["text", "table"], "reason": "row and cell"}],
            }],
        },
    )
    assert result["Q1"]["modal_types"] == ["text", "table"]


def test_validate_response_normalizes_valid_modality_order() -> None:
    qas = {"Q1": qa()}
    result = validate_response(
        "b1",
        qas,
        {
            "bundle_id": "b1",
            "items": [{
                "qa_id": "Q1",
                "modal_types": ["table", "text"],
                "confidence": 0.9,
                "rationale": "table lookup",
                "evidence": [{"page": 3, "modalities": ["table", "text"], "reason": "cell"}],
            }],
        },
    )
    assert result["Q1"]["modal_types"] == ["text", "table"]


def test_validate_response_normalizes_numeric_page_string() -> None:
    result = validate_response(
        "b1",
        {"Q1": qa()},
        {
            "bundle_id": "b1",
            "items": [{
                "qa_id": "Q1",
                "modal_types": ["table"],
                "confidence": 0.9,
                "rationale": "table lookup",
                "evidence": [{"page": "3", "modalities": ["table"], "reason": "cell"}],
            }],
        },
    )
    assert result["Q1"]["evidence"][0]["page"] == 3


def test_validate_response_rejects_modality_supported_only_by_non_cited_page() -> None:
    with pytest.raises(ValueError, match="no cited page-level"):
        validate_response(
            "b1",
            {"Q1": qa()},
            {
                "bundle_id": "b1",
                "items": [{
                    "qa_id": "Q1",
                    "modal_types": ["table"],
                    "confidence": 1,
                    "rationale": "cell lookup",
                    "evidence": [{"page": 4, "modalities": ["table"], "reason": "cell"}],
                }],
            },
        )


def test_apply_decisions_updates_modality_and_provenance() -> None:
    item = {
        "qa_id": "Q1",
        "modal_types": ["text", "table"],
        "confidence": 0.9,
        "rationale": "table lookup",
        "evidence": [{"page": 3, "modalities": ["text", "table"], "reason": "cell"}],
        "status": "completed",
        "review_version": REVIEW_VERSION,
        "input_fingerprint": qa_fingerprint(qa()),
        "page_support_complete": True,
        "page_support_missing_modalities": [],
    }
    dataset = {"b1": {"QA": {"Q1": qa()}}}
    counts = apply_decisions(dataset, {"model": "m", "bundles": {"b1": {"items": {"Q1": item}}}})
    assert dataset["b1"]["QA"]["Q1"]["modal_types"] == ["text", "table"]
    assert dataset["b1"]["QA"]["Q1"]["annotation_provenance"]["modality_api_review"]["model"] == "m"
    assert counts["changed"] == 1
