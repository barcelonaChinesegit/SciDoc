from pku_qa.workflows.reporting.diagnose_external_answer_accuracy import (
    build_rows,
    prediction_from_result,
    row_binding_sha256,
)


def test_prediction_from_result_handles_supported_fields_and_failure() -> None:
    assert prediction_from_result({"prediction_answer": " x "}) == (
        "x",
        "prediction_answer",
    )
    assert prediction_from_result({"answer_pre": " y "}) == (
        "y",
        "answer_pre",
    )
    assert prediction_from_result({"failed": True, "answer_pre": "y"}) == (
        "",
        "failed",
    )


def test_build_rows_joins_by_exact_item_and_filters_unanswerable() -> None:
    gold = {
        "1": {
            "QA": {
                "QA1": {"question": "Q1", "answer": "A1"},
                "UQA1": {"question": "Q2", "answer": "Unanswerable"},
            }
        }
    }
    results = {
        "1": {
            "QA": {
                "QA1": {"question": "Q1", "answer_pre": "P1"},
                "UQA1": {"question": "Q2", "answer_pre": "Unanswerable"},
            }
        }
    }
    rows = build_rows(gold, results, answerable_only=True)
    assert [(row["item_id"], row["prediction_answer"]) for row in rows] == [
        ("1/QA1", "P1")
    ]
    assert rows[0]["binding_sha256"] == row_binding_sha256("Q1", "A1", "P1")
