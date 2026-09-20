from __future__ import annotations

from pku_qa.workflows.cleaning.filter_reasoning_closed_book import (
    filter_dataset,
    flatten_judge,
)


def dataset() -> dict:
    return {
        "1": {
            "paper": "1",
            "QA": {
                "A": {"question": "Question A", "answer": "Answer A"},
                "B": {"question": "Question B", "answer": "Answer B"},
            },
        }
    }


def judge(a_correct: bool, b_correct: bool, model: str) -> dict:
    return {
        "1": {
            "QA": {
                qa_id: {
                    "question": f"Question {qa_id}",
                    "correct_answer": f"Answer {qa_id}",
                    "parsed_answer": f"prediction-{qa_id}",
                    "answer_is_correct": correct,
                    "input_mode": "question_only",
                    "evaluated_model": model,
                    "inference_binding_sha256": f"binding-{model}",
                    "judge_protocol_fingerprint": f"judge-{model}",
                }
                for qa_id, correct in (("A", a_correct), ("B", b_correct))
            }
        }
    }


def test_filter_removes_only_items_both_models_answer_correctly() -> None:
    judges = {
        "Qwen4B": flatten_judge(judge(True, False, "4B"), "Qwen4B"),
        "Qwen8B": flatten_judge(judge(True, True, "8B"), "Qwen8B"),
    }
    filtered, summary = filter_dataset(
        dataset(), judges, {"Qwen4B": "hash4", "Qwen8B": "hash8"}
    )
    assert list(filtered["1"]["QA"]) == ["B"]
    assert summary["retained_questions"] == 1
    assert summary["removed_both_correct"] == 1
    dependency = filtered["1"]["QA"]["B"]["closed_book_pdf_dependency"]
    assert dependency["both_models_correct"] is False
    assert dependency["models"]["Qwen4B"]["answer_is_correct"] is False


def test_filter_rejects_non_question_only_judge() -> None:
    payload = judge(False, False, "4B")
    payload["1"]["QA"]["A"]["input_mode"] = "pdf"
    try:
        flatten_judge(payload, "Qwen4B")
    except ValueError as exc:
        assert "not question_only" in str(exc)
    else:
        raise AssertionError("PDF Judge must not pass the closed-book filter")
