from __future__ import annotations

import pytest

from pku_qa.workflows.reporting.run_claude_pdf_selection import validate_prediction


def test_validate_prediction_enforces_pdf_contract() -> None:
    assert validate_prediction(
        {"answer_pre": "42", "evidence_pages": [2, 4]}, 4
    ) == ("42", [2, 4], [])
    assert validate_prediction(
        {"answer_pre": "Unanswerable", "evidence_pages": []}, 4
    ) == ("Unanswerable", [], [])

    with pytest.raises(ValueError):
        validate_prediction({"answer_pre": "42", "evidence_pages": []}, 4)
    with pytest.raises(ValueError):
        validate_prediction({"answer_pre": "42", "evidence_pages": [5]}, 4)


def test_validate_prediction_canonicalizes_page_order_and_duplicates() -> None:
    assert validate_prediction(
        {"answer_pre": "answer", "evidence_pages": [3, 1, 3]}, 3
    ) == ("answer", [1, 3], ["sorted_unique_evidence_pages"])
