from argparse import Namespace
import hashlib
from unittest.mock import patch

import pytest

from pathlib import Path
JUDGE_PROMPT = (Path(__file__).resolve().parents[2] / "evaluation/prompts/paper_semantic_judge.txt").read_text()
from eval_framework import ProviderError
from run_judge import build_judge_prompt, direct_fill_match, judge_fill, process_judge_paper
from evaluation_protocol import canonicalize_pdf_output_for_storage, parse_canonical_pdf_output
from evaluation.validation import validate_raw


@pytest.mark.parametrize("reference,prediction", [("42", "42"), ("42", "42.0"),
    ("0.42", "42%"), ("AdamW", "adamw"), ("A", "A"), ("42", "Unanswerable")])
def test_no_answerable_semantic_shortcut(reference, prediction):
    assert direct_fill_match(reference, prediction) == (None, "needs_llm_judge")


def test_exact_paper_prompt_bytes():
    assert build_judge_prompt("Q", "R", "P") == JUDGE_PROMPT.format(question="Q", correct="R", model_answer="P")


@pytest.mark.parametrize("raw", ["correct", " CORRECT", "CORRECT\n", "PARTIAL", "WRONG", "It is CORRECT"])
def test_strict_parser_retains_invalid_raw(raw):
    class Provider:
        def generate(self, *args):
            return raw
    with pytest.raises(ProviderError) as error:
        judge_fill(Provider(), "Q", "R", "P", 32)
    assert error.value.raw_response == raw


def test_no_refusal_whitespace_repair():
    assert direct_fill_match("Unanswerable", " Unanswerable ")[0] is False


@pytest.mark.parametrize("raw", [
    '{"answer_pre":" 42 ","evidence_pages":[3,1,3]}',
    '{"answer_pre":"Unanswerable","evidence_pages":[]}',
    '{"answer_pre":" Unanswerable ","evidence_pages":[]}',
    '{"answer_pre":"N/A","evidence_pages":[1]}',
    '{"answer_pre":"42","evidence_pages":[true]}',
    '{"answer_pre":"42","evidence_pages":[0]}',
    '{"answer_pre":"42","evidence_pages":[4]}',
    '{"answer_pre":"42","evidence_pages":["1"]}',
    '{"answer_pre":"42","answer_pre":"43","evidence_pages":[1]}',
    '```json\n{"answer_pre":"42","evidence_pages":[1]}\n```',
])
def test_internal_pdf_parser_agrees_with_public_contract(raw):
    public = validate_raw("QA0001", raw, 3)
    if public.status == "illegal":
        with pytest.raises(ValueError):
            parse_canonical_pdf_output(raw, allowed_pages=[1, 2, 3])
    else:
        assert parse_canonical_pdf_output(raw, allowed_pages=[1, 2, 3]) == (public.answer_pre, public.evidence_pages)


def test_answer_bytes_preserved_and_only_allowed_normalizations_logged():
    raw = '  {"answer_pre":" 42 ","evidence_pages":[3,1,3]}\n'
    result, audit = canonicalize_pdf_output_for_storage(raw)
    assert '"answer_pre":" 42 "' in result
    assert audit == ["outer_whitespace", "evidence_pages_deduplicated", "evidence_pages_sorted"]


@pytest.mark.parametrize("replies,expected", [(["correct", "CORRECT"], True), (["correct", "INCORRECT\n"], False)])
def test_retry_preserves_every_raw_reply_and_failed_item(replies, expected):
    gold = {"QA": {"Q": {"question": "Question?", "answer": "42", "evidence_pages": [1],
                            "answer_format": "Integer", "numeric_tolerance": {"value": 100}}}}
    raw = '{"answer_pre":"42","evidence_pages":[1]}'
    inference = {"QA": {"Q": {"question": "Question?", "correct_answer": "42", "reference_evidence_pages": [1],
        "model_output": raw, "raw_model_output": raw, "raw_model_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "deterministic_normalizations": [], "type": "fill", "input_mode": "pdf", "require_structured_output": True,
        "require_evidence_pages": True, "prompt_style": "pdf", "shown_pdf_pages": [1]}}}
    class Provider:
        def __init__(self):
            self.replies = iter(replies)
        def generate(self, *args):
            return next(self.replies)
    args = Namespace(retry_errors_only=False, worker_id="test", shard_id=0, max_judge_retries=2, max_new_tokens=32)
    with patch("run_judge.time.sleep"):
        row = process_judge_paper("P", inference, gold, None, judge_provider=Provider(), args=args,
                                  retry_match_methods=set())["QA"]["Q"]
    assert [a["raw"] for a in row["judge_attempts"]] == replies
    assert row["answer_is_correct"] is expected
    if not expected:
        assert row["technical_failure"] and row["judge_verdict"] == "ERROR"
