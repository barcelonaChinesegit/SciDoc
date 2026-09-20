from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/review/review_single_pdf_reasoning_qa_api.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("review_reasoning", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source_paper() -> dict:
    return {
        "paper": "1",
        "primary_category": "Economics",
        "secondary_category": "Theory",
        "QA": {
            "QA1": {
                "question": "Which policy raises output?",
                "answer": "Policy A",
                "evidence_pages": [1],
            },
            "QA2": {
                "question": "What does higher output cause?",
                "answer": "Lower prices",
                "evidence_pages": [2],
            },
        },
    }


def candidate_paper() -> dict:
    return {
        "paper": "1",
        "primary_category": "Economics",
        "secondary_category": "Theory",
        "QA": {
            "RQA1": {
                "question": "What price effect follows from the output-raising policy?",
                "answer": "Lower prices",
                "source_qa_ids": ["QA1", "QA2"],
                "reasoning_type": "causal_chain",
                "relation": "output links policy and prices",
                "derivation": [
                    {"source_qa_id": "QA1", "fact": "Policy A raises output"},
                    {"source_qa_id": "QA2", "fact": "output lowers prices"},
                    {"inference": "Policy A lowers prices"},
                ],
            }
        },
    }


def review(decision: str = "KEEP", confidence: float = 0.95) -> dict:
    return {
        "paper_id": "1",
        "items": [
            {
                "qa_id": "RQA1",
                "decision": decision,
                "criteria": {
                    criterion: decision == "KEEP"
                    for criterion in MODULE.CRITERIA
                },
                "confidence": confidence,
                "reason": "Both facts form a supported causal chain.",
                "corrected_question": None,
                "corrected_answer": None,
                "corrected_source_qa_ids": None,
                "corrected_reasoning_type": None,
                "corrected_relation": None,
                "corrected_derivation": None,
            }
        ],
    }


def test_prompts_exclude_evidence_pages() -> None:
    prompt = MODULE.build_prompt("1", source_paper(), candidate_paper())
    assert "evidence_pages" not in prompt
    assert "Which policy raises output?" in prompt
    assert "RQA1" in prompt


def test_hard_prompt_includes_structural_evidence_profile() -> None:
    candidate = candidate_paper()
    candidate["QA"]["RQA1"].update(
        {
            "reasoning_focus": "sequential_reasoning",
            "derived_conclusion": "The output-raising policy lowers prices.",
            "intermediate_facts": ["Output links policy to price."],
            "source_evidence_pages": [1, 5],
            "construction_key_evidence_pages": [1, 4],
            "evidence_span": 4,
        }
    )
    prompt = MODULE.build_prompt(
        "1", source_paper(), candidate, hard_mode=True
    )
    assert "source_evidence_pages" in prompt
    assert "hard expansion" in prompt
    assert "authoritative frozen metadata" in prompt
    assert "Do not speculate that a paper has fewer pages" in prompt
    assert "candidate's exact wording" in prompt
    assert "identity-hidden chain" in prompt
    assert "derived_conclusion" in prompt
    assert "abs(page_i - page_j) >= 3" in prompt
    assert "broader `source_evidence_pages`" in prompt


def test_validate_review_requires_all_ids_and_criteria() -> None:
    assert MODULE.validate_review(review(), "1", ["RQA1"]) == []
    broken = review()
    del broken["items"][0]["criteria"]["all_sources_needed"]
    assert "RQA1:invalid_criteria" in MODULE.validate_review(
        broken, "1", ["RQA1"]
    )


def test_finalize_requires_two_high_confidence_keep_reviews() -> None:
    source = {"1": source_paper()}
    candidates = {"1": candidate_paper()}
    reviews = {
        "claude": {"1": review()},
        "gemini": {"1": review()},
    }
    clean, ledger, summary = MODULE.finalize_clean_dataset(
        source, candidates, reviews, 1, 0.8
    )
    assert summary["status"] == "complete"
    assert summary["accepted_questions"] == 1
    assert ledger[0]["accepted"] is True
    assert clean["1"]["QA"]["RQA1"]["question"].startswith("What price")


def test_finalize_rejects_disagreement() -> None:
    source = {"1": source_paper()}
    candidates = {"1": candidate_paper()}
    reviews = {
        "claude": {"1": review()},
        "gemini": {"1": review("REJECT")},
    }
    clean, ledger, summary = MODULE.finalize_clean_dataset(
        source, candidates, reviews, 1, 0.8
    )
    assert not clean
    assert summary["status"] == "insufficient"
    assert ledger[0]["accepted"] is False


def test_review_one_retries_invalid_success_response(monkeypatch) -> None:
    calls = []

    def fake_call_provider(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {}, ""
        return {}, json.dumps(review())

    monkeypatch.setattr(MODULE, "call_provider", fake_call_provider)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        max_output_tokens=100,
        timeout=10,
        max_retries=2,
    )
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "review.json"
        payload = MODULE.review_one(
            "claude",
            "model",
            "secret",
            "1",
            source_paper(),
            candidate_paper(),
            output,
            args,
        )
        assert output.exists()
        assert payload["response_attempts"] == 2
        assert len(calls) == 2


def test_review_one_does_not_retry_exhausted_quota(monkeypatch) -> None:
    calls = []

    def fake_call_provider(*args, **kwargs):
        calls.append(1)
        raise MODULE.ApiQuotaExhaustedError("insufficient_quota")

    monkeypatch.setattr(MODULE, "call_provider", fake_call_provider)
    args = SimpleNamespace(max_output_tokens=100, timeout=10, max_retries=6)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            MODULE.review_one(
                "claude",
                "model",
                "secret",
                "1",
                source_paper(),
                candidate_paper(),
                Path(tmp) / "review.json",
                args,
            )
        except MODULE.ApiQuotaExhaustedError:
            pass
        else:
            raise AssertionError("Expected quota exhaustion to stop the review")
    assert len(calls) == 1


def test_quota_checkpoint_records_exact_provider_resume_point() -> None:
    candidates = {"1": candidate_paper(), "2": candidate_paper()}
    candidates["2"] = {**candidates["2"], "paper": "2"}
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        saved = {
            "paper_id": "1",
            "provider": "claude",
            "model": "model",
            "review": review(),
        }
        MODULE.atomic_json(output_dir / "claude" / "1.json", saved)
        checkpoint = MODULE.write_quota_checkpoint(
            output_dir,
            candidates,
            "claude",
            "2",
            "insufficient_quota",
        )
        assert checkpoint["completed_by_provider"] == {
            "claude": 1,
            "gemini": 0,
        }
        assert checkpoint["pending_by_provider"] == {
            "claude": 1,
            "gemini": 2,
        }
        progress = json.loads(
            (output_dir / "review_progress.json").read_text(encoding="utf-8")
        )
        assert progress["status"] == "waiting_for_api_quota"
        assert progress["resource_waiting"] is True


def test_saved_review_must_match_current_candidate_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "review.json"
        path.write_text(
            json.dumps({"paper_id": "1", "review": review()}),
            encoding="utf-8",
        )
        assert MODULE.saved_review_is_valid(path, "1", candidate_paper())
        changed = candidate_paper()
        changed["QA"]["RQA2"] = changed["QA"].pop("RQA1")
        assert not MODULE.saved_review_is_valid(path, "1", changed)


def test_final_integrity_requires_exact_dual_keep_target() -> None:
    source = {"1": source_paper()}
    candidates = {"1": candidate_paper()}
    reviews = {
        "claude": {"1": review()},
        "gemini": {"1": review()},
    }
    clean, _, _ = MODULE.finalize_clean_dataset(
        source, candidates, reviews, 1, 0.8
    )
    result = MODULE.validate_final_dataset(clean, 1, 0.8)
    assert result["status"] == "passed"
    clean["1"]["QA"]["RQA1"]["dual_model_validation"]["gemini"][
        "decision"
    ] = "REJECT"
    try:
        MODULE.validate_final_dataset(clean, 1, 0.8)
    except ValueError as exc:
        assert "did not KEEP" in str(exc)
    else:
        raise AssertionError("Expected strict integrity validation to fail")
