"""Historical audit must not equate shared worksheet gold or outsider IDs."""
import hashlib
from types import SimpleNamespace

import pytest

from evaluation.audit_score_provenance import (
    exact_question_candidates, historical_cache_key, historical_text,
    reconcile_membership, verify_raw_projection,
)


def test_cache_key_binds_the_actual_model_specific_reference():
    entry = dict(judge_rules_sha256="prompt", dataset_id="ordinary1190", question="Q",
                 gold_answer="model-specific reference", pred_answer="A")
    expected = hashlib.sha256(b"qwen36_calibrated_semantic_triclass_v4\n---\nprompt\n---\nordinary1190\n---\nQ\n---\nmodel-specific reference\n---\nA").hexdigest()
    assert historical_cache_key(entry) == expected
    assert historical_cache_key({**entry, "gold_answer": "shared worksheet reference"}) != expected
    assert historical_cache_key({**entry, "judge_rules_sha256": "other prompt"}) != expected


def test_historical_cache_lookup_does_not_modify_input():
    value = " A\n B\u00a0 "
    assert historical_text(value) == "A B"
    assert value == " A\n B\u00a0 "


def test_exact_match_requires_both_document_and_question():
    records = {("ordinary1190", "1", "QA2"): {"question": "Which value?"},
               ("ordinary1190", "2", "QA1"): {"question": "Different question?"}}
    assert exact_question_candidates("ordinary1190", "1", "Which value?", records) == [["ordinary1190", "1", "QA2"]]
    assert exact_question_candidates("ordinary1190", "1", "which value?", records) == []
    assert exact_question_candidates("ordinary1190", "2", "Which value?", records) == []


def test_correct_outsider_cannot_be_added_to_fixed_subject_cohort():
    a, b, c = [("ordinary1190", "1", f"QA{i}") for i in range(1, 4)]
    workbook = {a: {"model": {"Present": 1, "Answer Correct": 1}},
                b: {"model": {"Present": 1, "Answer Correct": 1}},
                c: {"model": {"Present": 0, "Answer Correct": None}}}
    subjects = {a: {"question": "First"}, c: {"question": "New question"}}
    raw = {a: {"question": "First", "answer": "A"}, b: {"question": "Old question", "answer": "B"}}
    result = reconcile_membership(workbook, subjects, "model", raw)
    assert result["recorded_correct"] == 2 and result["subject_join_correct"] == 1
    assert result["correct_outside_subject_cohort"] == 1
    assert result["outside_records"][0]["exact_subject_candidates"] == []
    assert result["subject_items_without_prediction"] == [list(c)]


def test_raw_projection_only_allows_page_sort_and_dedup():
    p = SimpleNamespace(qa_id="QA0001", answer_pre="Answer.", evidence_pages=[1, 3])
    verify_raw_projection('{"answer_pre":"Answer.","evidence_pages":[3,1,3]}', p)
    with pytest.raises(ValueError, match="changed model content"):
        verify_raw_projection('{"answer_pre":"answer","evidence_pages":[1,3]}', p)
    with pytest.raises(ValueError, match="changed model content"):
        verify_raw_projection('{"answer_pre":"Answer.","evidence_pages":[true,3]}', p)
