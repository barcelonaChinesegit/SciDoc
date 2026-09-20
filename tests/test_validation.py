import json
from pathlib import Path

import pytest

from evaluation.validation import Gold, SubmissionError, load_submission, validate_raw


def item(qa_id="QA0001", **kwargs):
    defaults = dict(question="What value?", answer="42", evidence_pages=(2,), task="General",
                    discipline="Physics", field="Optics", pdf_id="paper", page_count=9)
    return Gold(qa_id, **{**defaults, **kwargs})


def test_sort_deduplicate_and_whitespace_are_audited():
    p = validate_raw("QA0001", ' \n{"answer_pre":" 42% ","evidence_pages":[7,3,7]}\n', 9)
    assert p.status == "legal"
    assert p.answer_pre == " 42% "
    assert p.evidence_pages == [3, 7]
    assert set(p.audit["deterministic_normalizations"]) == {
        "outer_whitespace", "evidence_page_sorting", "evidence_page_deduplication"}


@pytest.mark.parametrize("pages,error", [([0], "page_nonpositive"), ([-1], "page_nonpositive"),
    ([10], "page_out_of_range"), (["3"], "page_type"), ([True], "page_type"),
    ([3.0], "page_type"), ("3-5", "pages_type"), (["page 3"], "page_type"), (["07"], "page_type")])
def test_invalid_pages(pages, error):
    p = validate_raw("QA0001", json.dumps({"answer_pre": "42", "evidence_pages": pages}), 9)
    assert p.status == "illegal"
    assert error in [e["type"] for e in p.errors]


@pytest.mark.parametrize("answer,pages,status", [
    ("Unanswerable", [], "legal"), ("Unanswerable", [3], "illegal"),
    ("unanswerable", [], "illegal"), ("UNANSWERABLE", [], "illegal"),
    ("N/A", [], "illegal"), ("Unknown", [3], "illegal"), ("42", [], "illegal"),
    (42, [3], "illegal"), (None, [3], "illegal"), ("", [3], "illegal")])
def test_refusal_and_answer_types(answer, pages, status):
    assert validate_raw("QA0001", json.dumps({"answer_pre": answer, "evidence_pages": pages}), 9).status == status


@pytest.mark.parametrize("raw", ['```json\n{"answer_pre":"42","evidence_pages":[2]}\n```',
    '{"answer_pre":"42","evidence_pages":[2]} explanation', '{} {}', '[]',
    '{"answer_pre":"42","answer_pre":"wrong","evidence_pages":[2]}',
    '{"answer_pre":"42","evidence_pages":[2],"thoughts":"extra"}',
    '{"answer_pre":"42","evidence_pages":[NaN]}'])
def test_no_fuzzy_json_repair(raw):
    assert validate_raw("QA0001", raw, 9).status == "illegal"


@pytest.mark.parametrize("records,match", [
    ([{"qa_id": "QA0001"}, {"qa_id": "QA0001"}], "duplicate"),
    ([{"qa_id": "QA9999"}], "unknown"), ([{}], "qa_id")])
def test_fatal_binding_errors(tmp_path, records, match):
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(SubmissionError, match=match):
        load_submission(path, {"QA0001": item()})


def test_missing_retained_and_raw_envelope(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"qa_id": "QA0001", "raw_model_output": "bad JSON"}))
    result = load_submission(path, {"QA0001": item(), "QA0002": item("QA0002")})
    assert result["QA0001"].status == "illegal"
    assert result["QA0002"].status == "missing"


def test_unbound_bad_json_is_fatal(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"qa_id":"QA0001", broken')
    with pytest.raises(SubmissionError, match="submission_parse_error"):
        load_submission(path, {"QA0001": item()})
