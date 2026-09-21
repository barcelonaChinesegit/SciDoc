import json

import pytest

from evaluation.judge import Judge
from evaluation.metrics import score
from evaluation.validation import Gold, load_submission


def test_hand_calculated_macro_fixed_denominator(tmp_path):
    gold = {f"QA{i:04d}": Gold(f"QA{i:04d}", "Question?", "42", pages, "General",
        "Physics", "Optics", "paper", 9) for i, pages in [(1, (1, 2)), (2, (2, 3)), (4, (2,))]}
    gold["QA0003"] = Gold("QA0003", "Unknown?", "Unanswerable", (), "Unanswerable", "Physics", "Optics", "paper", 9)
    path = tmp_path / "predictions.jsonl"
    path.write_text('\n'.join(json.dumps(row) for row in [
        {"qa_id": "QA0001", "answer_pre": "42", "evidence_pages": [1, 2]},
        {"qa_id": "QA0002", "answer_pre": "43", "evidence_pages": [1, 2]},
        {"qa_id": "QA0003", "answer_pre": "Unanswerable", "evidence_pages": []}]))
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return "INCORRECT" if "\n43\n" in prompt else "CORRECT"

    result = score(gold, load_submission(path, gold), Judge({"max_attempts": 1}, generate=generate))
    assert result["total"] == 4 and result["received"] == 3 and result["missing"] == 1
    assert result["answer_metrics"]["All"]["accuracy"] == 50
    assert result["answer_metrics"]["Unanswerable"]["accuracy"] == 100
    assert result["evidence_metrics"] == pytest.approx({"E-Precision": 62.5, "E-Recall": 62.5, "E-F1": 62.5, "A-Pages": 1})
    assert len(calls) == 3  # sxz also judges Unanswerable; PARTIAL never earns credit.


def test_missing_judge_aborts_instead_of_publishing_zero_score(tmp_path):
    gold = {"QA0001": Gold("QA0001", "Question?", "42", (2,), "General", "Physics", "Optics", "paper", 9)}
    path = tmp_path / "p.jsonl"
    path.write_text('{"qa_id":"QA0001","answer_pre":"42","evidence_pages":[2]}')
    with pytest.raises(ValueError, match="No matching"):
        score(gold, load_submission(path, gold), Judge({}))


def test_refusal_with_pages_is_judged_and_evidence_is_independent(tmp_path):
    gold = {"QA0001": Gold("QA0001", "Question?", "Unanswerable", (), "Unanswerable", "Physics", "Optics", "paper", 9)}
    path = tmp_path / "p.jsonl"
    path.write_text('{"qa_id":"QA0001","answer_pre":"Unanswerable","evidence_pages":[3]}')
    result = score(gold, load_submission(path, gold), Judge({}, generate=lambda _: "CORRECT"))
    assert result["answer_metrics"]["All"]["accuracy"] == 100
    assert result["evidence_metrics"]["E-F1"] == 0
    assert result["rows"][0]["prediction"]["audit"]["producer_format_status"] == "illegal"


def test_recover_fenced_json_and_score_evidence_before_failed_answer(tmp_path):
    gold = {f"QA{i:04d}": Gold(f"QA{i:04d}", "Question?", "42", (2,), "General",
            "Physics", "Optics", "paper", 9) for i in (1, 2, 3)}
    path = tmp_path / "p.jsonl"
    path.write_text('\n'.join(json.dumps(r) for r in [
        {"qa_id": "QA0001", "raw_model_output": '```json\n{"answer_pre":" 42 ","evidence_pages":["2"]}\n```'},
        {"qa_id": "QA0002", "answer_pre": "ERROR: CUDA out of memory", "evidence_pages": [2]},
        {"qa_id": "QA0003", "answer_pre": "part of the answer", "evidence_pages": [2]}]))
    def generate(prompt):
        return "PARTIAL" if "part of the answer" in prompt else "CORRECT"
    result = score(gold, load_submission(path, gold), Judge({}, generate=generate))
    assert result["answer_metrics"]["All"]["correct"] == 1
    assert result["technical_failures"] == 1
    assert result["evidence_metrics"]["E-F1"] == 100
    assert result["rows"][2]["judge"]["decision"] == "PARTIAL"
