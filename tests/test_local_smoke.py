"""Selection regressions for the opt-in real-model diagnostic (no GPU required)."""
import json

import pytest

from evaluation.validation import Gold
from scripts.smoke_local_evaluation import DATASETS, TASKS, select_predictions


def fixtures(tmp_path):
    gold, records = {}, {}
    for i, task in enumerate(TASKS, 1):
        qid = f"QA{i:04d}"
        answer = "Unanswerable" if task == "Unanswerable" else "42"
        pages = [] if task == "Unanswerable" else [1]
        gold[qid] = Gold(qid, "Question?", answer, tuple(pages), task,
                        "Physics", "Optics", "pdf", 3, "paper", qid)
        records[qid] = {"question": "Question?", "answer": answer, "evidence_pages": pages,
                        "answer_pre_raw": json.dumps({"answer_pre": answer, "evidence_pages": pages})}
    for dataset in DATASETS:
        (tmp_path / f"{dataset}.json").write_text("{}")
    (tmp_path / "ordinary1190.json").write_text(json.dumps({"paper": {"QA": records}}))
    return gold, records


def test_select_preserves_raw_and_requires_each_task(tmp_path):
    gold, records = fixtures(tmp_path)
    selected, audit = select_predictions(gold, tmp_path, 1)
    assert len(selected) == 4
    assert audit["selection_counts"] == dict.fromkeys(TASKS, 1)
    assert all(r["raw_model_output"] == records[r["qa_id"]]["answer_pre_raw"] for r in selected)


@pytest.mark.parametrize("change", ["question", "answer", "evidence_pages", "answer_pre_raw"])
def test_selection_rejects_changed_gold_or_illegal_raw(tmp_path, change):
    gold, records = fixtures(tmp_path)
    records["QA0001"][change] = [2] if change == "evidence_pages" else "changed"
    (tmp_path / "ordinary1190.json").write_text(json.dumps({"paper": {"QA": records}}))
    with pytest.raises(ValueError, match="Insufficient"):
        select_predictions(gold, tmp_path, 1)


def test_duplicate_historical_binding_is_fatal(tmp_path):
    gold, records = fixtures(tmp_path)
    (tmp_path / "cross_old400.json").write_text(json.dumps({"paper": {"QA": records}}))
    with pytest.raises(ValueError, match="Duplicate"):
        select_predictions(gold, tmp_path, 1)
