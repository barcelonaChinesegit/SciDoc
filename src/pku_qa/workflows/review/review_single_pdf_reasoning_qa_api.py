#!/usr/bin/env python3
"""Dual-model review and cleaning for composed single-paper reasoning QA.

Claude and Gemini independently see the complete original QA set for a paper
plus the Qwen-generated candidates.  PDFs and evidence pages are intentionally
excluded.  API responses are checkpointed per provider and paper.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests

from pku_qa.workflows.review.review_cross_pdf_qa_api import (
    ApiQuotaExhaustedError,
    CLAUDE_BASE_URL,
    GEMINI_BASE_URL,
    load_key,
    parse_json_response,
    post_with_retries,
    response_text_claude,
    response_text_gemini,
)
from pku_qa.workflows.cleaning.restore_reasoning_qa_evidence import (
    restore_reasoning_qa_evidence,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
DEFAULT_CANDIDATES = (
    ROOT
    / "data/qa/3.reasoning/reasoning_qa_200.json"
)
DEFAULT_OUTPUT = ROOT / "data/qa/3.reasoning/dual_review"
DEFAULT_FINAL = (
    ROOT
    / "data/qa/3.reasoning/"
    "work__reasoning__historical_clean__batch00__n100.json"
)
PROVIDERS = ("claude", "gemini")
CRITERIA = (
    "source_facts_supported",
    "all_sources_needed",
    "logical_inference_required",
    "not_simple_concatenation",
    "question_self_contained",
    "answer_correct_and_unique",
    "no_external_knowledge",
    "scientifically_meaningful",
)
DECISIONS = {"KEEP", "FIX", "REJECT"}
API_QUOTA_EXIT_CODE = 75

SYSTEM_PROMPT = """\
You are an independent senior reviewer for a scientific reasoning-QA dataset.
You receive all original QA pairs from one paper and new QA candidates composed
from them. You do not receive the PDF. Judge only whether each candidate is
strictly derivable from the supplied original QA.

KEEP only when every criterion is true:
1. every claimed source fact is present in its original QA;
2. all cited source QA are materially necessary;
3. answering requires a coherent inference across at least two source QA;
4. it is not two old questions joined together and its answer is not a list or
   concatenation of old answers;
5. the question is natural, self-contained, precise, and mentions no QA IDs;
6. the answer is correct, concise, and unique from the supplied facts;
7. no outside fact or unstated assumption is needed;
8. the synthesis is scientifically meaningful rather than arbitrary.

Use FIX only for a small, unambiguous correction that remains completely
derivable from 2-4 supplied source QA. Otherwise REJECT. Do not keep an item to
meet a quota. Return JSON only.
"""

HARD_REVIEW_PROMPT = """\

This run reviews the hard expansion. In addition to the eight base criteria,
KEEP only if the question does not disclose any original QA answer, the cited
evidence pages are distinct and separated by at least three physical pages, the
reasoning focus is one of the declared hard-focus values, and the answer states
the answer-critical intermediate bridge facts instead of jumping to a bare
conclusion. Treat same-page or adjacent-page evidence, answer leakage, or a
missing intermediate fact as REJECT rather than a cosmetic FIX. Test whether
one source fact can answer the candidate's exact wording, not merely whether a
terminal value appears as one original answer. A valid identity-hidden chain
may reuse the terminal source value when the question withholds the bridge
entity, the preceding source is necessary to resolve that entity, and the final
answer explicitly states both the bridge entity and terminal fact. Still REJECT
if the preceding source is decorative, the bridge is exposed in the question,
or a source_necessity_tests counterfactual is false. The
construction_key_evidence_pages must select one page per source QA with every
selected pair at least three pages apart.

The supplied evidence-page fields are authoritative frozen metadata recovered
from the source dataset. Check their numeric separation and source coverage as
given. Do not speculate that a paper has fewer pages, reject because PDF text is
not included in this blind content review, or demand a PDF-only verification.
For page separation, use only `construction_key_evidence_pages`: it contains
exactly one selected key page per cited source QA, and every pair is valid when
`abs(page_i - page_j) >= 3`. The broader `source_evidence_pages` field is the
union of all optional evidence pages and may contain adjacent alternatives; do
not reject an item because that union contains close pages. A difference of
exactly 3 is valid. Do not invent a stricter interpretation based on the number
of pages lying between two selected page numbers.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=ROOT / "data/pdfs",
        help="Local PDFs used only after blind review to validate restored evidence.",
    )
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--claude-model", default="claude-sonnet-5")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=7000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--quota-retries", type=int, default=0)
    parser.add_argument("--quota-retry-wait-seconds", type=int, default=45)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hard-mode",
        action="store_true",
        help="Apply answer-leakage, evidence-distance, and intermediate-fact gates.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not call APIs; rebuild the clean dataset from saved reviews.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def original_qa_payload(paper: dict[str, Any]) -> list[dict[str, str]]:
    """Exclude evidence-page and modality metadata from the API prompt."""
    return [
        {
            "qa_id": str(qa_id),
            "question": str(qa.get("question", "")).strip(),
            "answer": str(qa.get("answer", "")).strip(),
        }
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    ]


def candidate_payload(
    paper: dict[str, Any], hard_mode: bool = False
) -> list[dict[str, Any]]:
    fields = [
        "question",
        "answer",
        "source_qa_ids",
        "reasoning_type",
        "relation",
        "derivation",
    ]
    if hard_mode:
        fields.extend(
            (
                "reasoning_focus",
                "derived_conclusion",
                "intermediate_facts",
                "source_necessity_tests",
                "source_evidence_pages",
                "construction_key_evidence_pages",
                "evidence_span",
                "hard_construction_quality_check",
            )
        )
    return [
        {"qa_id": str(qa_id), **{field: qa.get(field) for field in fields}}
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    ]


def expected_review(paper_id: str, qa_ids: list[str]) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "items": [
            {
                "qa_id": qa_id,
                "decision": "KEEP | FIX | REJECT",
                "criteria": {criterion: True for criterion in CRITERIA},
                "confidence": 0.95,
                "reason": "Specific source-QA-grounded reason.",
                "corrected_question": None,
                "corrected_answer": None,
                "corrected_source_qa_ids": None,
                "corrected_reasoning_type": None,
                "corrected_relation": None,
                "corrected_derivation": None,
            }
            for qa_id in qa_ids
        ],
    }


def build_prompt(
    paper_id: str,
    source_paper: dict[str, Any],
    candidate_paper: dict[str, Any],
    hard_mode: bool = False,
) -> str:
    candidates = candidate_payload(candidate_paper, hard_mode=hard_mode)
    return (
        f"paper_id={paper_id}\n\n"
        "ALL ORIGINAL QA FROM THIS PAPER:\n"
        + json.dumps(
            original_qa_payload(source_paper), ensure_ascii=False, indent=2
        )
        + "\n\nNEW CANDIDATES TO REVIEW:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
        + "\n\nREQUIRED OUTPUT SHAPE:\n"
        + json.dumps(
            expected_review(
                paper_id, [str(item["qa_id"]) for item in candidates]
            ),
            ensure_ascii=False,
            indent=2,
        )
        + (HARD_REVIEW_PROMPT if hard_mode else "")
    )


def validate_review(
    review: dict[str, Any], paper_id: str, qa_ids: list[str]
) -> list[str]:
    errors: list[str] = []
    if str(review.get("paper_id")) != paper_id:
        errors.append("paper_id_mismatch")
    items = review.get("items")
    if not isinstance(items, list):
        return errors + ["items_not_list"]
    actual_ids = [
        str(item.get("qa_id")) for item in items if isinstance(item, dict)
    ]
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("duplicate_qa_ids")
    if set(actual_ids) != set(qa_ids):
        errors.append("qa_id_set_mismatch")
    for item in items:
        if not isinstance(item, dict):
            errors.append("item_not_object")
            continue
        qa_id = str(item.get("qa_id"))
        decision = item.get("decision")
        if decision not in DECISIONS:
            errors.append(f"{qa_id}:invalid_decision")
        criteria = item.get("criteria")
        if not isinstance(criteria, dict) or any(
            not isinstance(criteria.get(name), bool) for name in CRITERIA
        ):
            errors.append(f"{qa_id}:invalid_criteria")
        confidence = item.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            errors.append(f"{qa_id}:invalid_confidence")
        if not str(item.get("reason", "")).strip():
            errors.append(f"{qa_id}:empty_reason")
        if decision == "FIX" and not any(
            item.get(field) is not None
            for field in (
                "corrected_question",
                "corrected_answer",
                "corrected_source_qa_ids",
                "corrected_reasoning_type",
                "corrected_relation",
                "corrected_derivation",
            )
        ):
            errors.append(f"{qa_id}:fix_without_correction")
    return errors


def call_provider(
    provider: str,
    key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], str]:
    if provider == "claude":
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        response: requests.Response = post_with_retries(
            f"{CLAUDE_BASE_URL}/v1/messages",
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body,
            timeout,
            retries,
        )
        raw = response.json()
        return raw, response_text_claude(raw)
    if provider == "gemini":
        body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        }
        response = post_with_retries(
            f"{GEMINI_BASE_URL}/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": key, "content-type": "application/json"},
            body,
            timeout,
            retries,
        )
        raw = response.json()
        return raw, response_text_gemini(raw)
    raise ValueError(f"Unknown provider: {provider}")


def review_one(
    provider: str,
    model: str,
    key: str,
    paper_id: str,
    source_paper: dict[str, Any],
    candidate_paper: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    prompt = build_prompt(
        paper_id,
        source_paper,
        candidate_paper,
        hard_mode=bool(getattr(args, "hard_mode", False)),
    )
    qa_ids = list(candidate_paper.get("QA", {}))
    last_error: Exception | None = None
    raw_api: dict[str, Any] = {}
    text = ""
    review: dict[str, Any] = {}
    attempts = max(1, args.max_retries)
    quota_retries = 0
    for attempt in range(1, attempts + 1):
        try:
            while True:
                try:
                    raw_api, text = call_provider(
                        provider,
                        key,
                        model,
                        prompt,
                        args.max_output_tokens,
                        args.timeout,
                        args.max_retries,
                    )
                    break
                except ApiQuotaExhaustedError:
                    if quota_retries >= max(
                        0, int(getattr(args, "quota_retries", 0))
                    ):
                        raise
                    quota_retries += 1
                    wait_seconds = max(
                        1,
                        int(
                            getattr(
                                args, "quota_retry_wait_seconds", 45
                            )
                        ),
                    )
                    print(
                        json.dumps(
                            {
                                "event": "api_quota_retry_wait",
                                "provider": provider,
                                "paper_id": paper_id,
                                "quota_retry": quota_retries,
                                "wait_seconds": wait_seconds,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    time.sleep(wait_seconds)
            review = parse_json_response(text)
            errors = validate_review(review, paper_id, qa_ids)
            if errors:
                raise ValueError(
                    f"{provider} invalid review for {paper_id}: {errors}"
                )
            break
        except Exception as exc:
            if isinstance(exc, ApiQuotaExhaustedError):
                raise
            last_error = exc
            if attempt >= attempts:
                raise RuntimeError(
                    f"{provider} review failed after {attempts} response "
                    f"attempts for paper {paper_id}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            print(
                json.dumps(
                    {
                        "event": "api_response_retry",
                        "provider": provider,
                        "paper_id": paper_id,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "error_type": type(exc).__name__,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(min(30, 2 ** (attempt - 1)))
    if last_error is not None and not review:
        raise last_error
    payload = {
        "paper_id": paper_id,
        "provider": provider,
        "model": model,
        "elapsed_seconds": round(time.time() - started, 3),
        "response_attempts": attempt,
        "quota_retries": quota_retries,
        "review": review,
        "raw_response_text": text,
        "api_response_metadata": {
            key: value
            for key, value in raw_api.items()
            if key not in {"content", "candidates"}
        },
    }
    atomic_json(output_path, payload)
    return payload


def saved_review_is_valid(
    path: Path,
    paper_id: str,
    candidate_paper: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
        review = payload.get("review")
        if not isinstance(review, dict):
            return False
        return not validate_review(
            review,
            paper_id,
            list(candidate_paper.get("QA", {})),
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def review_index(output_dir: Path, provider: str) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((output_dir / provider).glob("*.json")):
        payload = read_json(path)
        review = payload.get("review")
        if isinstance(review, dict):
            review = dict(review)
            review["_model"] = payload.get("model")
            result[str(payload.get("paper_id", path.stem))] = review
    return result


def item_index(review: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("qa_id")): item
        for item in review.get("items", [])
        if isinstance(item, dict)
    }


def finalize_clean_dataset(
    source: dict[str, Any],
    candidates: dict[str, Any],
    reviews: dict[str, dict[str, dict[str, Any]]],
    target: int,
    min_confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    clean: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    accepted = 0
    provider_decisions: dict[str, Counter[str]] = {
        provider: Counter() for provider in PROVIDERS
    }
    pair_decisions: Counter[str] = Counter()

    for paper_id, candidate_paper in candidates.items():
        provider_items = {
            provider: item_index(reviews.get(provider, {}).get(paper_id, {}))
            for provider in PROVIDERS
        }
        output_qas: dict[str, Any] = {}
        for qa_id, qa in candidate_paper.get("QA", {}).items():
            decisions = {}
            reasons = {}
            confidences = {}
            criteria_ok = {}
            for provider in PROVIDERS:
                review_item = provider_items[provider].get(str(qa_id), {})
                decision = str(review_item.get("decision", "MISSING"))
                decisions[provider] = decision
                reasons[provider] = review_item.get("reason")
                confidences[provider] = review_item.get("confidence")
                criteria = review_item.get("criteria")
                criteria_ok[provider] = isinstance(criteria, dict) and all(
                    criteria.get(name) is True for name in CRITERIA
                )
                provider_decisions[provider][decision] += 1
            pair_key = f"{decisions['claude']}+{decisions['gemini']}"
            pair_decisions[pair_key] += 1
            question_key = normalized_text(qa.get("question"))
            keep = (
                accepted < target
                and decisions == {"claude": "KEEP", "gemini": "KEEP"}
                and all(criteria_ok.values())
                and all(
                    isinstance(confidences[provider], (int, float))
                    and confidences[provider] >= min_confidence
                    for provider in PROVIDERS
                )
                and question_key
                and question_key not in seen_questions
            )
            ledger.append(
                {
                    "paper_id": paper_id,
                    "qa_id": qa_id,
                    "accepted": keep,
                    "decisions": decisions,
                    "confidences": confidences,
                    "criteria_all_true": criteria_ok,
                    "reasons": reasons,
                }
            )
            if not keep:
                continue
            accepted += 1
            seen_questions.add(question_key)
            item = dict(qa)
            item["dual_model_validation"] = {
                provider: {
                    "decision": decisions[provider],
                    "confidence": confidences[provider],
                    "reason": reasons[provider],
                    "model": reviews[provider][paper_id].get(
                        "_model", provider
                    ),
                }
                for provider in PROVIDERS
            }
            output_qas[f"RQA{accepted}"] = item
        if output_qas:
            clean[paper_id] = {
                "paper": candidate_paper.get("paper", paper_id),
                "primary_category": candidate_paper.get(
                    "primary_category", source[paper_id].get("primary_category", "")
                ),
                "secondary_category": candidate_paper.get(
                    "secondary_category",
                    source[paper_id].get("secondary_category", ""),
                ),
                "QA": output_qas,
            }

    summary = {
        "status": "complete" if accepted >= target else "insufficient",
        "target": target,
        "candidate_questions": sum(
            len(paper.get("QA", {})) for paper in candidates.values()
        ),
        "accepted_questions": accepted,
        "papers_used": len(clean),
        "acceptance_rate": (
            accepted
            / max(
                1,
                sum(len(paper.get("QA", {})) for paper in candidates.values()),
            )
        ),
        "policy": (
            "Claude KEEP + Gemini KEEP + all criteria true + both confidence "
            f">= {min_confidence}"
        ),
        "provider_decisions": {
            provider: dict(sorted(counter.items()))
            for provider, counter in provider_decisions.items()
        },
        "paired_decisions": dict(sorted(pair_decisions.items())),
        "pdf_uploaded": False,
        "evidence_pages_used_in_blind_review": False,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    return clean, ledger, summary


def validate_final_dataset(
    dataset: dict[str, Any],
    target: int,
    min_confidence: float,
) -> dict[str, Any]:
    items: list[tuple[str, str, dict[str, Any]]] = []
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Invalid final paper record: {paper_id}")
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(f"Invalid final QA record: {paper_id}/{qa_id}")
            items.append((str(paper_id), str(qa_id), qa))
    if len(items) != target:
        raise ValueError(
            f"Final dataset must contain exactly {target} QA, found {len(items)}"
        )
    normalized_questions: set[str] = set()
    for paper_id, qa_id, qa in items:
        question = normalized_text(qa.get("question"))
        if not question or question in normalized_questions:
            raise ValueError(
                f"Empty or duplicate final question: {paper_id}/{qa_id}"
            )
        normalized_questions.add(question)
        validation = qa.get("dual_model_validation")
        if not isinstance(validation, dict):
            raise ValueError(
                f"Missing dual_model_validation: {paper_id}/{qa_id}"
            )
        for provider in PROVIDERS:
            verdict = validation.get(provider)
            if not isinstance(verdict, dict):
                raise ValueError(
                    f"Missing {provider} validation: {paper_id}/{qa_id}"
                )
            if verdict.get("decision") != "KEEP":
                raise ValueError(
                    f"{provider} did not KEEP {paper_id}/{qa_id}"
                )
            confidence = verdict.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or confidence < min_confidence
            ):
                raise ValueError(
                    f"{provider} confidence below threshold for "
                    f"{paper_id}/{qa_id}"
                )
    return {
        "status": "passed",
        "question_count": len(items),
        "unique_question_count": len(normalized_questions),
        "providers": list(PROVIDERS),
        "minimum_confidence": min_confidence,
        "policy": "both KEEP and both confidence at or above threshold",
    }


def validate_hard_final_dataset(
    dataset: dict[str, Any], min_evidence_span: int = 3
) -> dict[str, Any]:
    """Enforce the hard expansion's leakage and page-separation contract."""
    focus_counts: Counter[str] = Counter()
    spans: list[int] = []
    for paper_id, paper in dataset.items():
        for qa_id, qa in paper.get("QA", {}).items():
            question = re.sub(r"[^\w]+", " ", normalized_text(qa.get("question"))).strip()
            answer = re.sub(r"[^\w]+", " ", normalized_text(qa.get("answer"))).strip()
            if len(answer) >= 4 and answer in question:
                raise ValueError(f"Answer leakage in {paper_id}/{qa_id}")
            if len(re.findall(r"\w+", str(qa.get("answer", "")))) < 8:
                raise ValueError(
                    f"Answer omits an explicit reasoning bridge: {paper_id}/{qa_id}"
                )
            pages = qa.get("evidence_pages")
            if not isinstance(pages, list) or len(set(pages)) < 2:
                raise ValueError(
                    f"Hard reasoning evidence is not on distinct pages: "
                    f"{paper_id}/{qa_id}"
                )
            span = max(pages) - min(pages)
            if span < min_evidence_span:
                raise ValueError(
                    f"Hard reasoning evidence span {span} is below "
                    f"{min_evidence_span}: {paper_id}/{qa_id}"
                )
            intermediate = qa.get("intermediate_facts")
            if not isinstance(intermediate, list) or not intermediate:
                raise ValueError(
                    f"Missing intermediate facts: {paper_id}/{qa_id}"
                )
            focus = str(qa.get("reasoning_focus", ""))
            if focus not in {
                "sequential_reasoning",
                "conditional_filtering",
                "similar_concept_discrimination",
                "model_relationship",
                "metric_reasoning",
            }:
                raise ValueError(
                    f"Invalid hard reasoning focus {focus!r}: {paper_id}/{qa_id}"
                )
            focus_counts[focus] += 1
            spans.append(span)
    return {
        "status": "passed",
        "minimum_evidence_span": min(spans) if spans else None,
        "evidence_span_distribution": dict(sorted(Counter(spans).items())),
        "reasoning_focus_distribution": dict(sorted(focus_counts.items())),
    }


def write_progress(
    output_dir: Path,
    status: str,
    completed: int,
    total: int,
    stage: str,
    **extra: Any,
) -> None:
    atomic_json(
        output_dir / "review_progress.json",
        {
            "status": status,
            "completed": completed,
            "total": total,
            "percent": round(100 * completed / total, 4) if total else 100.0,
            "stage": stage,
            "updated_at": time.time(),
            **extra,
        },
    )


def saved_checkpoint_snapshot(
    output_dir: Path,
    candidates: dict[str, Any],
) -> dict[str, Any]:
    """Describe the exact durable resume point without making API calls."""
    completed_by_provider: dict[str, int] = {}
    next_pending_by_provider: dict[str, str | None] = {}
    pending_by_provider: dict[str, int] = {}
    for provider in PROVIDERS:
        completed = 0
        pending: list[str] = []
        for paper_id, candidate_paper in candidates.items():
            path = output_dir / provider / f"{paper_id}.json"
            if saved_review_is_valid(
                path, str(paper_id), candidate_paper
            ):
                completed += 1
            else:
                pending.append(str(paper_id))
        completed_by_provider[provider] = completed
        pending_by_provider[provider] = len(pending)
        next_pending_by_provider[provider] = pending[0] if pending else None
    return {
        "completed": sum(completed_by_provider.values()),
        "total": len(candidates) * len(PROVIDERS),
        "completed_by_provider": completed_by_provider,
        "pending_by_provider": pending_by_provider,
        "next_pending_by_provider": next_pending_by_provider,
    }


def write_quota_checkpoint(
    output_dir: Path,
    candidates: dict[str, Any],
    provider: str,
    paper_id: str,
    error: str,
) -> dict[str, Any]:
    snapshot = saved_checkpoint_snapshot(output_dir, candidates)
    event = {
        "status": "waiting_for_api_quota",
        "provider_detected": provider,
        "paper_id_detected": paper_id,
        "error": error[:2000],
        **snapshot,
        "detected_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "resume_policy": (
            "After quota renewal, resume the queue task. Valid provider/paper "
            "JSON checkpoints are skipped and only pending reviews are called."
        ),
    }
    history_path = output_dir / "api_quota_history.json"
    try:
        history = read_json(history_path) if history_path.exists() else []
    except (OSError, ValueError, json.JSONDecodeError):
        history = []
    if not isinstance(history, list):
        history = []
    history.append(event)
    atomic_json(history_path, history)
    atomic_json(output_dir / "api_quota_exhausted.json", event)
    write_progress(
        output_dir,
        "waiting_for_api_quota",
        int(snapshot["completed"]),
        int(snapshot["total"]),
        "dual_api_review",
        resource_waiting=True,
        work_state="waiting_for_api_quota",
        blocked_reasons={"api_quota_exhausted": 1},
        quota_checkpoint="api_quota_exhausted.json",
        completed_by_provider=snapshot["completed_by_provider"],
        pending_by_provider=snapshot["pending_by_provider"],
        next_pending_by_provider=snapshot["next_pending_by_provider"],
    )
    return event


def main() -> None:
    args = parse_args()
    source = read_json(args.source)
    candidates = read_json(args.candidates)
    if not isinstance(source, dict) or not isinstance(candidates, dict):
        raise SystemExit("Source and candidates must be JSON objects")
    missing = sorted(set(candidates) - set(source))
    if missing:
        raise SystemExit(f"Candidate paper IDs missing from source: {missing[:10]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (provider, paper_id)
        for paper_id in candidates
        for provider in PROVIDERS
        if args.force
        or not saved_review_is_valid(
            args.output_dir / provider / f"{paper_id}.json",
            str(paper_id),
            candidates[paper_id],
        )
    ]
    total_calls = len(candidates) * len(PROVIDERS)
    already_done = total_calls - len(work)
    write_progress(
        args.output_dir,
        "running",
        already_done,
        total_calls,
        "dual_api_review",
    )
    if not args.finalize_only and work:
        key = load_key(args.api_key_file)
        models = {
            "claude": args.claude_model,
            "gemini": args.gemini_model,
        }
        errors = []
        quota_exhaustion: dict[str, str] | None = None
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            work_iterator = iter(work)
            futures = {}

            def submit_one(provider: str, paper_id: str) -> None:
                output_path = (
                    args.output_dir / provider / f"{paper_id}.json"
                )
                future = pool.submit(
                    review_one,
                    provider,
                    models[provider],
                    key,
                    paper_id,
                    source[paper_id],
                    candidates[paper_id],
                    output_path,
                    args,
                )
                futures[future] = (provider, paper_id)

            for _ in range(min(max(1, args.workers), len(work))):
                submit_one(*next(work_iterator))
            successful = already_done
            attempted = already_done
            work_finished = False
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    provider, paper_id = futures.pop(future)
                    succeeded = False
                    try:
                        future.result()
                        succeeded = True
                    except ApiQuotaExhaustedError as exc:
                        if quota_exhaustion is None:
                            quota_exhaustion = {
                                "provider": provider,
                                "paper_id": str(paper_id),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                    except Exception as exc:
                        errors.append(
                            {
                                "provider": provider,
                                "paper_id": paper_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    attempted += 1
                    if succeeded:
                        successful += 1
                    print(
                        json.dumps(
                            {
                                "event": (
                                    "api_quota_exhausted"
                                    if quota_exhaustion is not None
                                    else "review_complete"
                                ),
                                "provider": provider,
                                "paper_id": paper_id,
                                "completed": successful,
                                "total": total_calls,
                                "attempted": attempted,
                                "succeeded": succeeded,
                                "errors": len(errors),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if quota_exhaustion is not None:
                    for future in futures:
                        future.cancel()
                    break
                while len(futures) < max(1, args.workers):
                    try:
                        submit_one(*next(work_iterator))
                    except StopIteration:
                        work_finished = True
                        break
                if work_finished and not futures:
                    break
                write_progress(
                    args.output_dir,
                    "running",
                    successful,
                    total_calls,
                    "dual_api_review",
                    errors=len(errors),
                    attempted=attempted,
                )
        if quota_exhaustion is not None:
            checkpoint = write_quota_checkpoint(
                args.output_dir,
                candidates,
                quota_exhaustion["provider"],
                quota_exhaustion["paper_id"],
                quota_exhaustion["error"],
            )
            print(
                json.dumps(
                    {"event": "api_quota_checkpoint_saved", **checkpoint},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            raise SystemExit(API_QUOTA_EXIT_CODE)
        atomic_json(args.output_dir / "api_errors.json", errors)
        if errors:
            write_progress(
                args.output_dir,
                "failed",
                successful,
                total_calls,
                "dual_api_review",
                errors=len(errors),
                attempted=attempted,
            )
            raise SystemExit(
                f"{len(errors)} API reviews failed; rerun to resume"
            )

    reviews = {
        provider: review_index(args.output_dir, provider)
        for provider in PROVIDERS
    }
    expected_papers = set(candidates)
    for provider in PROVIDERS:
        missing_reviews = sorted(expected_papers - set(reviews[provider]))
        if missing_reviews:
            raise SystemExit(
                f"{provider} is missing {len(missing_reviews)} paper reviews"
            )
        saved_errors = []
        for paper_id in candidates:
            errors = validate_review(
                reviews[provider][paper_id],
                paper_id,
                list(candidates[paper_id].get("QA", {})),
            )
            if errors:
                saved_errors.append(
                    {"paper_id": paper_id, "errors": errors}
                )
        if saved_errors:
            atomic_json(
                args.output_dir / f"{provider}_saved_review_errors.json",
                saved_errors,
            )
            raise SystemExit(
                f"{provider} has {len(saved_errors)} invalid saved reviews"
            )

    clean, ledger, summary = finalize_clean_dataset(
        source,
        candidates,
        reviews,
        args.target,
        args.min_confidence,
    )
    if summary["status"] == "complete":
        clean, evidence_summary = restore_reasoning_qa_evidence(
            clean,
            source,
            pdf_dir=args.pdf_dir,
            source_dataset_path=args.source,
            source_dataset_sha256=sha256_file(args.source),
            expected_count=args.target,
        )
        summary["evidence_restoration"] = evidence_summary
        summary["final_evidence_restored"] = True
        summary["integrity_validation"] = validate_final_dataset(
            clean,
            args.target,
            args.min_confidence,
        )
        if args.hard_mode:
            summary["hard_integrity_validation"] = validate_hard_final_dataset(
                clean
            )
            summary["evidence_pages_used_in_blind_review"] = True
    atomic_json(args.final_output, clean)
    atomic_json(args.output_dir / "cleaning_ledger.json", ledger)
    atomic_json(args.output_dir / "summary.json", summary)
    write_progress(
        args.output_dir,
        "completed" if summary["status"] == "complete" else "failed",
        summary["accepted_questions"],
        args.target,
        "cleaning_complete",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(
            f"Strict dual review retained only "
            f"{summary['accepted_questions']}/{args.target}; "
            "generate a larger candidate pool and resume review"
        )


if __name__ == "__main__":
    main()
