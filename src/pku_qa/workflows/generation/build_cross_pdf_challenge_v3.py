#!/usr/bin/env python3
"""Build a 400+ item high-quality Cross-PDF Challenge v3 release.

The release combines:

1. the 369-item semantic release, enriched with the already-audited
   per-fact evidence ledger;
2. every cached three-document candidate that passed all strict v2 review
   gates, including candidates omitted only by the v2 zero-overlap policy;
3. newly generated three-document candidates from a focused Gemini generator,
   accepted only after an independent full-paper Claude review.

Generation and review are checkpointed per bundle. API credentials are loaded
through the existing safe loader and are never written to outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from pku_qa.pdf_assets import PDF_ROOT, output_pdf_path, resolve_pdf_path

try:
    from pku_qa.workflows.generation.build_cross_pdf_quality_v2 import (
        TRI_CRITERIA,
        apply_tri_fix,
        evidence_items_from_support,
        mapped_docs,
        normalize_original_support,
        source_text,
    )
    from pku_qa.workflows.review.review_cross_pdf_qa_api import (
        CLAUDE_BASE_URL,
        GEMINI_BASE_URL,
        load_key,
        manifest_index,
        parse_json_response,
        post_with_retries,
        response_text_claude,
        response_text_gemini,
    )
except ModuleNotFoundError:
    from pku_qa.workflows.generation.build_cross_pdf_quality_v2 import (
        TRI_CRITERIA,
        apply_tri_fix,
        evidence_items_from_support,
        mapped_docs,
        normalize_original_support,
        source_text,
    )
    from pku_qa.workflows.review.review_cross_pdf_qa_api import (
        CLAUDE_BASE_URL,
        GEMINI_BASE_URL,
        load_key,
        manifest_index,
        parse_json_response,
        post_with_retries,
        response_text_claude,
        response_text_gemini,
    )


ROOT = Path(__file__).resolve().parents[4]
SOURCE_QA = ROOT / "data/qa/qa_cross_pdf_semantic_verified_20260724.json"
SOURCE_MANIFEST = (
    ROOT
    / "data/qa/4.cross_pdf/semantic_reaudit/final_release/"
    "selected_bundle_manifest.json"
)
SOURCE_LEDGER = (
    ROOT
    / "data/qa/4.cross_pdf/semantic_reaudit/final_release/"
    "release_ledger.jsonl"
)
SOURCE_PDFS = ROOT / "data/pdfs"
V2_ROOT = ROOT / "data/qa/4.cross_pdf/quality"
V2_QA = ROOT / "data/qa/4.cross_pdf/quality/qa_cross_pdf_challenge_v2.json"

OUTPUT_ROOT = ROOT / "data/qa/4.cross_pdf/challenge"
OUTPUT_QA = ROOT / "data/qa/4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json"
OUTPUT_PDFS = ROOT / "data/pdfs"
MANUAL_NEW_ACCEPTANCE = OUTPUT_ROOT / "manual_new_acceptance.json"

GENERATOR_SYSTEM = """\
You create difficult scientific multi-document QA from exactly three supplied
papers. Every question must require substantive facts from Doc 1, Doc 2, and
Doc 3. Removing any one document must make the complete answer incomplete or
non-unique.

Reject shallow designs while generating: no generic common-theme questions,
no three independent summaries joined together, no answer that can be written
from abstracts alone, and no invented causal or equivalence relation. Prefer
mechanism synthesis, conditional transfer, failure diagnosis, quantitative
reconciliation, experimental-constraint analysis, or evidence-led claim
assessment. The question must ask for one joint conclusion, choice, diagnosis,
derived quantity, compatibility judgment, or counterfactual whose derivation
uses all three papers. Do not enumerate one subquestion per paper; do not use
"(1)/(2)/(3)", "respectively", or an answer made of three independent
paragraphs. Use visibly diverse wording and do not refer to "the provided
documents" generically.

The strongest design is a concrete scientific scenario with a dependency
chain: a result, constraint, or failure diagnosed from one paper changes which
mechanism from another paper is applicable, and the third paper resolves the
remaining decision or prediction. Before returning a candidate, apply this
counterfactual test: if its answer can be produced as three self-contained
sentences obtained independently from the papers, discard it. Never manufacture
a dependency by adding unrelated measurements, converting units, or performing
arbitrary arithmetic.

Use only the supplied paper contents. Evidence pages are 1-based physical page
numbers in the merged PDF. Return JSON only in the requested shape.
"""

REVIEWER_SYSTEM = """\
You are the independent senior gatekeeper for a hard three-document scientific
QA benchmark. Another model generated the candidates. Read all three complete
papers and verify every candidate from scratch.

KEEP or FIX only when every strict criterion is true:
1. substantive facts from Doc 1, Doc 2, and Doc 3 are all necessary;
2. removing any one document makes the answer incomplete or non-unique;
3. the task is harder than three parallel lookups and requires real synthesis;
4. the question is specific, natural, unambiguous, and scientifically useful;
5. the answer is correct, complete, concise, and has no invented relationship;
6. every material answer claim has explicit page support;
7. verified support covers all three documents.

REJECT whenever the answer can be produced by independently answering one
subquestion per paper and concatenating the three results. A shared theme or a
three-way comparison is not sufficient; the facts must jointly determine one
conclusion.

FIX is allowed only for a small, unambiguous correction that you can verify
directly. Otherwise REJECT. Return JSON only in the requested shape.
"""

STRICT_CRITERIA = (
    "all_three_documents_required",
    "each_document_individually_necessary",
    "requires_integrated_reasoning",
    "answer_not_parallel_concatenation",
    "question_specific_natural_unambiguous",
    "answer_correct_and_complete",
    "evidence_supports_every_material_claim",
    "verified_support_covers_all_three_documents",
)

_WRITE_LOCK = threading.Lock()
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PARALLEL_QUESTION_RE = re.compile(
    r"(?:\(\s*[123]\s*\)|(?:^|\s)[123]\s*[\).:]|"
    r"\brespectively\b|\bspecifically\s*,?\s*(?:describe|detail|explain|"
    r"identify|analyze)\b)",
    re.IGNORECASE | re.MULTILINE,
)
SYNTHESIS_REJECTION_REASON_RE = re.compile(
    r"(?:three|3)\s+(?:separate|independent)\s+(?:facts|lookups|queries|"
    r"sub-?questions|design\s+decisions|components|failure\s+modes)|"
    r"(?:answer|response)\s+is\s+(?:literally\s+)?(?:a\s+)?concatenation|"
    r"(?:classic|textbook|canonical)\s+(?:case\s+of\s+)?"
    r"(?:parallel|three-way)|"
    r"violates?\s+(?:the\s+)?(?:joint|non-?parallel)|"
    r"(?:facts|claims)\s+do\s+not\s+jointly\s+determine",
    re.IGNORECASE,
)

FOCUSED_GENERATION_HINTS = {
    "xb_0051": (
        "Frame one statistically coherent adaptive-evaluation scenario. For "
        "example, a learned predictor prioritizes tests of a causal claim "
        "while the analyst may stop early: determine which estimator "
        "correction and anytime-valid guarantee are jointly required, and "
        "whether the anomaly system's replay policy supplies either. Verify "
        "the exact mechanisms; do not merely compare three weighting methods."
    ),
    "xb_0098": (
        "Look for a defensible end-to-end dependency among legal-document "
        "retrieval, the decision to search, and span-level uncertainty. A "
        "retrieved fact or uncertainty diagnosis must change the routing or "
        "final decision; do not merely describe the three methods."
    ),
    "xb_0121": (
        "Frame one adaptive evaluation of diffusion unlearning: the "
        "cross-attention surrogate defines the intervention target, a "
        "prediction-powered sampler allocates labels, and anytime-valid "
        "inference decides when evidence is sufficient. Ask for one validity "
        "or design conclusion that genuinely depends on all three."
    ),
    "xb_0175": (
        "Frame one end-to-end missing-data analysis in which the chosen "
        "imputation changes a causal anomaly score and a sequential "
        "evaluation rule determines when the model comparison may stop. The "
        "answer must be one diagnosis or decision, not three tool summaries."
    ),
    "xb_0205": (
        "Look for one evidence-supported failure-analysis chain in which a "
        "code-completion artifact is attributed, responsibility inside a "
        "multi-agent workflow is localized, and a token-level RL mechanism "
        "explains or prevents the failure. Do not assert a cross-paper causal "
        "link unless the supplied mechanisms genuinely support the transfer."
    ),
    "xb_0240": (
        "Look for one judge-audit decision in which measured judge error "
        "changes a self-evolving curator decision and an evaluation-metric "
        "limitation determines whether that conclusion is trustworthy."
    ),
    "xb_0275": (
        "Look for one concrete speech-to-forecasting-to-enterprise-harness "
        "failure or design decision where an upstream observation changes a "
        "downstream uncertainty route and the harness action."
    ),
    "xb_0300": (
        "Look for one document-QA decision chain where contextual retrieval "
        "provides or misses evidence, hallucination control diagnoses or "
        "regularizes the answer, and counterfactual routing determines "
        "whether additional search is useful."
    ),
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_ledger_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[(str(row["bundle_id"]), str(row["qa_id"]))] = row
    return rows


def ledger_evidence_items(
    ledger_row: dict[str, Any],
    sources: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize audited support and conservatively repair local-page leakage."""

    normalized_support = []
    repairs = []
    for row in ledger_row.get("claude_support", []):
        doc_number = int(row["doc_number"])
        source = sources[doc_number - 1]
        raw_page = int(row["merged_page"])
        start = int(source["merged_start_page"])
        end = int(source["merged_end_page"])
        page_count = int(source["source_page_count"])
        if start <= raw_page <= end:
            physical_page = raw_page
        elif 1 <= raw_page <= page_count:
            physical_page = start + raw_page - 1
            repairs.append(
                {
                    "doc_number": doc_number,
                    "reported_page": raw_page,
                    "resolved_physical_pdf_page": physical_page,
                    "reason": (
                        "reported page did not map to the declared document "
                        "but was valid as a source-local page"
                    ),
                }
            )
        else:
            raise ValueError(
                "Unresolvable audited support page: "
                f"doc={doc_number}, page={raw_page}"
            )
        normalized_support.append(
            {
                "doc_number": doc_number,
                "physical_pdf_page": physical_page,
                "supported_fact": str(row["supported_fact"]).strip(),
            }
        )
    return evidence_items_from_support(
        normalized_support, sources
    ), repairs


def response_usage(raw: dict[str, Any]) -> dict[str, Any]:
    return raw.get("usage") or raw.get("usageMetadata") or {}


def parse_json_response_lenient(text: str) -> dict[str, Any]:
    """Repair only non-JSON backslash escapes commonly emitted in LaTeX."""

    try:
        parsed = parse_json_response(text)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})',
            r"\\\\",
            text,
        )
        parsed = parse_json_response(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("API response is not a JSON object")
    return parsed


def call_generator(
    key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        "system_instruction": {"parts": [{"text": GENERATOR_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
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
    parsed = parse_json_response_lenient(response_text_gemini(raw))
    return raw, parsed


def call_reviewer(
    key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": REVIEWER_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = post_with_retries(
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
    parsed = parse_json_response_lenient(response_text_claude(raw))
    return raw, parsed


def generation_schema(count: int) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "candidate_id": f"C{index:02d}",
                "question": "A precise integrated three-document question.",
                "answer": "A concise complete answer synthesizing all documents.",
                "evidence_pages": [1, 20, 40],
                "evidence_items": [
                    {
                        "doc_number": 1,
                        "physical_pdf_page": 1,
                        "supported_fact": "Material fact needed from this page.",
                    }
                ],
                "modal_types": ["text"],
                "question_category": "Specific scientific synthesis category",
                "reasoning_style": (
                    "mechanism_synthesis | conditional_transfer | "
                    "failure_diagnosis | quantitative_reconciliation | "
                    "constraint_analysis | claim_assessment"
                ),
                "necessity": {
                    "doc_1": "Why Doc 1 is indispensable.",
                    "doc_2": "Why Doc 2 is indispensable.",
                    "doc_3": "Why Doc 3 is indispensable.",
                },
                "integrated_conclusion": (
                    "The one conclusion that can be derived only by combining "
                    "all three papers."
                ),
                "ablation_failure": {
                    "without_doc_1": "Why the conclusion becomes impossible.",
                    "without_doc_2": "Why the conclusion becomes impossible.",
                    "without_doc_3": "Why the conclusion becomes impossible.",
                },
            }
            for index in range(1, count + 1)
        ]
    }


def build_generation_prompt(
    bundle_id: str,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    count: int,
) -> tuple[str, dict[str, Any]]:
    text, source_meta = source_text(manifest)
    existing_questions = [
        item["question"] for item in bundle.get("QA", {}).values()
    ]
    prompt = (
        f"bundle_id={bundle_id}\n"
        f"Create exactly {count} candidate questions.\n\n"
        "BUNDLE-SPECIFIC DESIGN BRIEF:\n"
        + FOCUSED_GENERATION_HINTS.get(
            bundle_id,
            (
                "Find a natural dependency chain supported by the papers; "
                "do not force a link when none exists."
            ),
        )
        + "\n\n"
        "SOURCE MAP:\n"
        + json.dumps(source_meta, ensure_ascii=False, indent=2)
        + "\n\nEXISTING QUESTIONS TO AVOID DUPLICATING OR PARAPHRASING:\n"
        + json.dumps(existing_questions, ensure_ascii=False, indent=2)
        + "\n\nREQUIRED OUTPUT SHAPE:\n"
        + json.dumps(generation_schema(count), ensure_ascii=False, indent=2)
        + "\n\nFULL SOURCE PAPERS:\n"
        + text
    )
    return prompt, {
        "bundle_id": bundle_id,
        "source_meta": source_meta,
        "prompt_characters": len(prompt),
        "requested_candidates": count,
    }


def normalize_candidate(
    row: dict[str, Any],
    sources: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    candidate_id = str(row.get("candidate_id", "")).strip()
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    integrated_conclusion = str(
        row.get("integrated_conclusion", "")
    ).strip()
    ablation_failure = row.get("ablation_failure")
    pages = row.get("evidence_pages")
    evidence_items = row.get("evidence_items")
    if (
        not candidate_id
        or not question
        or not answer
        or not integrated_conclusion
        or not isinstance(ablation_failure, dict)
        or any(
            not str(ablation_failure.get(name, "")).strip()
            for name in (
                "without_doc_1",
                "without_doc_2",
                "without_doc_3",
            )
        )
        or PARALLEL_QUESTION_RE.search(question)
        or not isinstance(pages, list)
        or not pages
        or not all(
            isinstance(page, int)
            and not isinstance(page, bool)
            and page >= 1
            for page in pages
        )
        or mapped_docs(pages, sources) != {1, 2, 3}
        or not isinstance(evidence_items, list)
    ):
        return None
    normalized_support = []
    for support in evidence_items:
        if not isinstance(support, dict):
            continue
        doc_number = support.get("doc_number")
        page = support.get("physical_pdf_page")
        fact = str(support.get("supported_fact", "")).strip()
        if (
            not isinstance(doc_number, int)
            or doc_number not in {1, 2, 3}
            or not isinstance(page, int)
            or page not in pages
            or not fact
        ):
            continue
        if mapped_docs([page], sources) != {doc_number}:
            continue
        normalized_support.append(
            {
                "doc_number": doc_number,
                "physical_pdf_page": page,
                "supported_fact": fact,
            }
        )
    if {item["doc_number"] for item in normalized_support} != {1, 2, 3}:
        return None
    result = deepcopy(row)
    result.update(
        {
            "candidate_id": candidate_id,
            "question": question,
            "answer": answer,
            "evidence_pages": sorted(set(pages)),
            "evidence_items": normalized_support,
        }
    )
    return result


def validate_generation(
    result: dict[str, Any],
    sources: list[dict[str, Any]],
    requested: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_rows = result.get("candidates")
    if not isinstance(raw_rows, list):
        return [], ["candidates_not_list"]
    valid = []
    seen: set[str] = set()
    for row in raw_rows:
        normalized = normalize_candidate(row, sources)
        if normalized is None:
            continue
        candidate_id = normalized["candidate_id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        valid.append(normalized)
    errors = []
    if len(raw_rows) != requested:
        errors.append(f"candidate_count_{len(raw_rows)}_expected_{requested}")
    if not valid:
        errors.append("no_structurally_valid_candidates")
    return valid, errors


def review_schema(candidate_ids: list[str]) -> dict[str, Any]:
    return {
        "items": [
            {
                "candidate_id": candidate_id,
                "decision": "KEEP | FIX | REJECT",
                "criteria": {
                    criterion: True for criterion in STRICT_CRITERIA
                },
                "confidence": 0.9,
                "reason": "Specific evidence-grounded decision.",
                "verified_support": [
                    {
                        "doc_number": 1,
                        "physical_pdf_page": 1,
                        "supported_fact": "Verified material fact.",
                    }
                ],
                "corrected_question": None,
                "corrected_answer": None,
                "corrected_evidence_pages": None,
            }
            for candidate_id in candidate_ids
        ]
    }


def build_review_prompt(
    bundle_id: str,
    manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    text, source_meta = source_text(manifest)
    candidate_ids = [row["candidate_id"] for row in candidates]
    prompt = (
        f"bundle_id={bundle_id}\n\n"
        "SOURCE MAP:\n"
        + json.dumps(source_meta, ensure_ascii=False, indent=2)
        + "\n\nCANDIDATES TO REVIEW:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
        + "\n\nREQUIRED OUTPUT SHAPE:\n"
        + json.dumps(review_schema(candidate_ids), ensure_ascii=False, indent=2)
        + "\n\nFULL SOURCE PAPERS:\n"
        + text
    )
    return prompt, {
        "bundle_id": bundle_id,
        "source_meta": source_meta,
        "prompt_characters": len(prompt),
        "candidate_ids": candidate_ids,
    }


def validate_review(
    result: dict[str, Any], candidate_ids: list[str]
) -> list[str]:
    items = result.get("items")
    if not isinstance(items, list):
        return ["items_not_list"]
    actual = [
        str(row.get("candidate_id"))
        for row in items
        if isinstance(row, dict)
    ]
    errors = []
    if len(actual) != len(candidate_ids) or set(actual) != set(candidate_ids):
        errors.append("candidate_id_mismatch")
    for row in items:
        if not isinstance(row, dict):
            errors.append("review_item_not_object")
            continue
        candidate_id = row.get("candidate_id")
        if row.get("decision") not in {"KEEP", "FIX", "REJECT"}:
            errors.append(f"{candidate_id}:invalid_decision")
        criteria = row.get("criteria")
        if not isinstance(criteria, dict) or any(
            not isinstance(criteria.get(name), bool)
            for name in STRICT_CRITERIA
        ):
            errors.append(f"{candidate_id}:invalid_criteria")
        if not isinstance(row.get("verified_support"), list):
            errors.append(f"{candidate_id}:support_not_list")
    return errors


def process_generation(
    bundle_id: str,
    *,
    qa: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = args.output_root / "generation" / f"{bundle_id}.json"
    if path.exists() and not args.force:
        return load_json(path)
    prompt, metadata = build_generation_prompt(
        bundle_id,
        qa[bundle_id],
        manifests[bundle_id],
        args.candidates_per_bundle,
    )
    started = time.time()
    raw, generation = call_generator(
        key,
        args.generator_model,
        prompt,
        args.max_output_tokens,
        args.timeout,
        args.max_retries,
    )
    valid, errors = validate_generation(
        generation,
        manifests[bundle_id]["sources"],
        args.candidates_per_bundle,
    )
    record = {
        "bundle_id": bundle_id,
        "generator_model": args.generator_model,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "duration_seconds": round(time.time() - started, 3),
        "input_fingerprint_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "metadata": metadata,
        "generation": generation,
        "valid_candidates": valid,
        "validation_errors": errors,
        "usage": response_usage(raw),
        "raw_response": raw,
    }
    atomic_json(path, record)
    return record


def process_review(
    bundle_id: str,
    *,
    qa: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    key: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = args.output_root / "reviews" / f"{bundle_id}.json"
    if path.exists() and not args.force:
        return load_json(path)
    generation = process_generation(
        bundle_id,
        qa=qa,
        manifests=manifests,
        key=key,
        args=args,
    )
    candidates = generation.get("valid_candidates", [])
    if not candidates:
        record = {
            "bundle_id": bundle_id,
            "reviewer_model": args.reviewer_model,
            "reviewed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "review": {"items": []},
            "validation_errors": ["generation_not_reviewable"],
            "skipped": True,
        }
        atomic_json(path, record)
        return record
    prompt, metadata = build_review_prompt(
        bundle_id, manifests[bundle_id], candidates
    )
    started = time.time()
    raw, review = call_reviewer(
        key,
        args.reviewer_model,
        prompt,
        args.max_output_tokens,
        args.timeout,
        args.max_retries,
    )
    errors = validate_review(
        review, [row["candidate_id"] for row in candidates]
    )
    record = {
        "bundle_id": bundle_id,
        "reviewer_model": args.reviewer_model,
        "reviewed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "duration_seconds": round(time.time() - started, 3),
        "input_fingerprint_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "metadata": metadata,
        "review": review,
        "validation_errors": errors,
        "usage": response_usage(raw),
        "raw_response": raw,
    }
    atomic_json(path, record)
    return record


def cached_v2_strict_items(
    manifests: dict[str, dict[str, Any]]
) -> dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    accepted: dict[
        str, list[tuple[str, dict[str, Any], dict[str, Any]]]
    ] = {}
    for generation_path in sorted((V2_ROOT / "generation").glob("*.json")):
        bundle_id = generation_path.stem
        review_path = V2_ROOT / "reviews" / f"{bundle_id}.json"
        if not review_path.exists() or bundle_id not in manifests:
            continue
        generation_record = load_json(generation_path)
        review_record = load_json(review_path)
        if (
            generation_record.get("validation_errors")
            or review_record.get("validation_errors")
        ):
            continue
        generated = {
            str(row["candidate_id"]): row
            for row in generation_record["generation"].get(
                "tri_candidates", []
            )
        }
        reviewed = {
            str(row["candidate_id"]): row
            for row in review_record["review"].get("tri_candidates", [])
        }
        for candidate_id, candidate in generated.items():
            decision = reviewed.get(candidate_id, {})
            if decision.get("decision") not in {"KEEP", "FIX"}:
                continue
            if not all(
                decision.get("criteria", {}).get(name) is True
                for name in TRI_CRITERIA
            ):
                continue
            fixed = apply_tri_fix(candidate, decision)
            sources = manifests[bundle_id]["sources"]
            if mapped_docs(fixed.get("evidence_pages", []), sources) != {
                1,
                2,
                3,
            }:
                continue
            support = decision.get("verified_support", [])
            if {
                row.get("doc_number")
                for row in support
                if isinstance(row, dict)
            } != {1, 2, 3}:
                continue
            accepted.setdefault(bundle_id, []).append(
                (candidate_id, fixed, decision)
            )
    return accepted


def apply_review_fix(
    candidate: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(candidate)
    if review.get("decision") == "FIX":
        for source_name, target_name in (
            ("corrected_question", "question"),
            ("corrected_answer", "answer"),
            ("corrected_evidence_pages", "evidence_pages"),
        ):
            value = review.get(source_name)
            if value not in (None, "", []):
                result[target_name] = deepcopy(value)
    return result


def strict_new_items(
    manifests: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    manual_allowlist: set[tuple[str, str]] | None = None
    acceptance_path = output_root / MANUAL_NEW_ACCEPTANCE.name
    if acceptance_path.exists():
        acceptance = load_json(acceptance_path)
        manual_allowlist = {
            (str(row["bundle_id"]), str(row["candidate_id"]))
            for row in acceptance.get("accepted", [])
        }
    accepted: dict[
        str, list[tuple[str, dict[str, Any], dict[str, Any]]]
    ] = {}
    for generation_path in sorted((output_root / "generation").glob("*.json")):
        bundle_id = generation_path.stem
        review_path = output_root / "reviews" / f"{bundle_id}.json"
        if not review_path.exists() or bundle_id not in manifests:
            continue
        generation = load_json(generation_path)
        review = load_json(review_path)
        if review.get("validation_errors"):
            continue
        candidates = {
            str(row["candidate_id"]): row
            for row in generation.get("valid_candidates", [])
        }
        decisions = {
            str(row["candidate_id"]): row
            for row in review.get("review", {}).get("items", [])
        }
        for candidate_id, candidate in candidates.items():
            if (
                manual_allowlist is not None
                and (bundle_id, candidate_id) not in manual_allowlist
            ):
                continue
            if PARALLEL_QUESTION_RE.search(
                str(candidate.get("question", ""))
            ):
                continue
            decision = decisions.get(candidate_id, {})
            if SYNTHESIS_REJECTION_REASON_RE.search(
                str(decision.get("reason", ""))
            ):
                continue
            if decision.get("decision") not in {"KEEP", "FIX"}:
                continue
            if not all(
                decision.get("criteria", {}).get(name) is True
                for name in STRICT_CRITERIA
            ):
                continue
            fixed = apply_review_fix(candidate, decision)
            sources = manifests[bundle_id]["sources"]
            if mapped_docs(fixed.get("evidence_pages", []), sources) != {
                1,
                2,
                3,
            }:
                continue
            support = decision.get("verified_support", [])
            support_docs = {
                row.get("doc_number")
                for row in support
                if isinstance(row, dict)
            }
            if support_docs != {1, 2, 3}:
                continue
            if any(
                not isinstance(row.get("physical_pdf_page"), int)
                or mapped_docs(
                    [row["physical_pdf_page"]], sources
                )
                != {row.get("doc_number")}
                for row in support
                if isinstance(row, dict)
            ):
                continue
            accepted.setdefault(bundle_id, []).append(
                (candidate_id, fixed, decision)
            )
    return accepted


def question_tokens(question: str) -> set[str]:
    return set(TOKEN_RE.findall(question.casefold()))


def is_near_duplicate(
    question: str, existing: list[str]
) -> tuple[bool, str | None, float, float]:
    normalized = " ".join(question.casefold().split())
    tokens = question_tokens(question)
    for other in existing:
        other_normalized = " ".join(other.casefold().split())
        other_tokens = question_tokens(other)
        union = tokens | other_tokens
        jaccard = len(tokens & other_tokens) / len(union) if union else 1.0
        ratio = SequenceMatcher(None, normalized, other_normalized).ratio()
        if normalized == other_normalized or jaccard >= 0.80 or ratio >= 0.82:
            return True, other, jaccard, ratio
    return False, None, 0.0, 0.0


def build_base_release(
    qa: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    ledger: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    release = deepcopy(qa)
    for bundle_id, bundle in release.items():
        sources = manifests[bundle_id]["sources"]
        bundle["pdf_page_numbering"] = "physical_pdf_page_1_based"
        bundle["source_paper_ids"] = [
            str(source["paper_id"]) for source in sources
        ]
        bundle["source_documents"] = [
            {
                "doc_number": number,
                "paper_id": source["paper_id"],
                "arxiv_id": source.get("arxiv_id"),
                "title": source["title"],
                "merged_start_page": source["merged_start_page"],
                "merged_end_page": source["merged_end_page"],
                "source_page_count": source["source_page_count"],
            }
            for number, source in enumerate(sources, start=1)
        ]
        for qa_id, item in bundle["QA"].items():
            ledger_row = ledger[(bundle_id, qa_id)]
            evidence_items, page_repairs = ledger_evidence_items(
                ledger_row, sources
            )
            reviewed_pages = sorted(
                {
                    int(row["physical_pdf_page"])
                    for row in evidence_items
                }
            )
            original_pages = sorted(
                {int(page) for page in item["evidence_pages"]}
            )
            pages = (
                reviewed_pages
                if len(mapped_docs(reviewed_pages, sources)) >= 2
                else original_pages
            )
            docs = sorted(mapped_docs(pages, sources))
            item.update(
                {
                    "qa_uid": f"{bundle_id}/{qa_id}",
                    "evidence_pages": pages,
                    "evidence_page_numbering": (
                        "physical_pdf_page_1_based"
                    ),
                    "evidence_items": evidence_items,
                    "evidence_hops": len(docs),
                    "evidence_source_docs": docs,
                    "evidence_span": max(pages) - min(pages),
                    "source_paper_ids": [
                        sources[number - 1]["paper_id"] for number in docs
                    ],
                    "answer_format": item.get("answer_format", "string"),
                    "answer_aliases": item.get("answer_aliases", []),
                    "answer_unit": item.get("answer_unit"),
                    "numeric_tolerance": item.get("numeric_tolerance"),
                    "review_status": "semantic_full_paper_review_pass",
                    "difficulty_basis": [
                        "inferential",
                        "requires_multiple_source_papers",
                    ],
                    "annotation_provenance": {
                        "source_release": SOURCE_QA.name,
                        "source_release_tier": ledger_row["release_tier"],
                        "semantic_reviewer": "claude-sonnet-5",
                        "semantic_review_confidence": ledger_row.get(
                            "claude_confidence"
                        ),
                        "v3_enrichment": (
                            "embedded audited per-fact evidence ledger"
                        ),
                        "evidence_page_resolution": (
                            "claude_verified_support_pages"
                            if pages == reviewed_pages
                            else "original_release_pages"
                        ),
                        "page_numbering_repairs": page_repairs,
                    },
                }
            )
    return release


def apply_published_v2_items(
    release: dict[str, Any],
) -> tuple[int, int, set[tuple[str, str]]]:
    """Make v3 an exact content superset of the published 39-item v2."""

    v2 = load_json(V2_QA)
    rewrites_applied = 0
    tri_added = 0
    published_tri_candidates: set[tuple[str, str]] = set()
    for bundle_id, bundle in v2.items():
        for qa_id, source_item in bundle["QA"].items():
            item = deepcopy(source_item)
            provenance = deepcopy(item.get("annotation_provenance", {}))
            provenance["v3_inclusion"] = (
                "exact_published_challenge_v2_item"
            )
            item["annotation_provenance"] = provenance
            item["qa_uid"] = f"{bundle_id}/{qa_id}"
            if qa_id.startswith("TRI_"):
                if qa_id in release[bundle_id]["QA"]:
                    raise ValueError(
                        f"Unexpected v2 tri-item id collision: "
                        f"{bundle_id}/{qa_id}"
                    )
                item["review_status"] = (
                    "independent_strict_three_document_review_pass"
                )
                item["difficulty_basis"] = [
                    "all_three_documents_required",
                    "independent_full_paper_review",
                    "published_in_challenge_v2",
                ]
                release[bundle_id]["QA"][qa_id] = item
                published_tri_candidates.add(
                    (bundle_id, qa_id.removeprefix("TRI_"))
                )
                tri_added += 1
            else:
                if qa_id not in release[bundle_id]["QA"]:
                    raise ValueError(
                        f"Published v2 rewrite has no semantic-base item: "
                        f"{bundle_id}/{qa_id}"
                    )
                item["review_status"] = (
                    "semantic_and_v2_independent_review_pass"
                )
                item["difficulty_basis"] = [
                    "inferential",
                    "requires_multiple_source_papers",
                    "v2_question_rewrite_reviewed",
                ]
                release[bundle_id]["QA"][qa_id] = item
                rewrites_applied += 1
    return rewrites_applied, tri_added, published_tri_candidates


def append_strict_items(
    release: dict[str, Any],
    items: dict[
        str, list[tuple[str, dict[str, Any], dict[str, Any]]]
    ],
    manifests: dict[str, dict[str, Any]],
    *,
    id_prefix: str,
    provenance: dict[str, Any],
    existing_questions: list[str],
    duplicate_log: list[dict[str, Any]],
) -> int:
    added = 0
    for bundle_id in sorted(items):
        sources = manifests[bundle_id]["sources"]
        for candidate_id, candidate, decision in items[bundle_id]:
            duplicate, other, jaccard, ratio = is_near_duplicate(
                candidate["question"], existing_questions
            )
            if duplicate:
                duplicate_log.append(
                    {
                        "bundle_id": bundle_id,
                        "candidate_id": candidate_id,
                        "question": candidate["question"],
                        "matched_question": other,
                        "token_jaccard": jaccard,
                        "sequence_ratio": ratio,
                    }
                )
                continue
            qa_id = f"{id_prefix}_{candidate_id}"
            suffix = 2
            while qa_id in release[bundle_id]["QA"]:
                qa_id = f"{id_prefix}_{candidate_id}_{suffix}"
                suffix += 1
            pages = sorted(
                {
                    int(row["physical_pdf_page"])
                    for row in decision["verified_support"]
                }
            )
            docs = sorted(mapped_docs(pages, sources))
            item = {
                "question": str(candidate["question"]).strip(),
                "answer": str(candidate["answer"]).strip(),
                "evidence_pages": pages,
                "modal_types": candidate.get("modal_types", ["text"]),
                "question_type": "Inferential",
                "question_category": candidate.get(
                    "question_category", "Three-document synthesis"
                ),
                "reasoning_style": candidate.get(
                    "reasoning_style", "three_document_synthesis"
                ),
                "qa_uid": f"{bundle_id}/{qa_id}",
                "evidence_page_numbering": "physical_pdf_page_1_based",
                "evidence_items": evidence_items_from_support(
                    decision["verified_support"], sources
                ),
                "evidence_hops": len(docs),
                "evidence_source_docs": docs,
                "evidence_span": max(pages) - min(pages),
                "source_paper_ids": [
                    sources[number - 1]["paper_id"] for number in docs
                ],
                "answer_format": candidate.get("answer_format", "string"),
                "answer_aliases": [],
                "answer_unit": None,
                "numeric_tolerance": None,
                "review_status": (
                    "independent_strict_three_document_review_pass"
                ),
                "difficulty_basis": [
                    "all_three_documents_required",
                    "each_document_individually_necessary",
                    "requires_integrated_reasoning",
                ],
                "annotation_provenance": {
                    **provenance,
                    "review_confidence": decision.get("confidence"),
                    "review_reason": decision.get("reason"),
                    "review_criteria": decision.get("criteria"),
                },
            }
            release[bundle_id]["QA"][qa_id] = item
            existing_questions.append(item["question"])
            added += 1
    return added


def materialize_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"Existing PDF hash mismatch: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise ValueError(f"Copied PDF hash mismatch: {destination}")


def validate_release(
    release: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    pdf_dir: Path,
) -> dict[str, Any]:
    errors = []
    questions: set[str] = set()
    qa_count = 0
    hop_counts: Counter[int] = Counter()
    review_statuses: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    evidence_complete = 0
    page_counts = []
    per_bundle = []
    strict_count = 0
    for bundle_id, bundle in release.items():
        sources = manifests[bundle_id]["sources"]
        for source in sources:
            source_counts[str(source["paper_id"])] += 1
        pdf_path = resolve_pdf_path(bundle_id, [pdf_dir], required=False)
        if pdf_path is None:
            errors.append(f"{bundle_id}:missing_pdf")
            continue
        actual_pages = len(PdfReader(pdf_path).pages)
        page_counts.append(actual_pages)
        if actual_pages != int(manifests[bundle_id]["page_count"]):
            errors.append(f"{bundle_id}:page_count_mismatch")
        per_bundle.append(len(bundle["QA"]))
        for qa_id, item in bundle["QA"].items():
            qa_count += 1
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            normalized = " ".join(question.casefold().split())
            if not question or not answer:
                errors.append(f"{bundle_id}/{qa_id}:blank_question_or_answer")
            if normalized in questions:
                errors.append(f"{bundle_id}/{qa_id}:duplicate_question")
            questions.add(normalized)
            pages = item.get("evidence_pages")
            if (
                not isinstance(pages, list)
                or not pages
                or not all(
                    isinstance(page, int)
                    and not isinstance(page, bool)
                    and 1 <= page <= actual_pages
                    for page in pages
                )
            ):
                errors.append(f"{bundle_id}/{qa_id}:invalid_evidence_pages")
                continue
            docs = mapped_docs(pages, sources)
            hop_counts[len(docs)] += 1
            if len(docs) < 2:
                errors.append(f"{bundle_id}/{qa_id}:not_cross_document")
            if item.get("review_status") == (
                "independent_strict_three_document_review_pass"
            ):
                strict_count += 1
                if docs != {1, 2, 3}:
                    errors.append(
                        f"{bundle_id}/{qa_id}:strict_item_not_three_document"
                    )
            evidence_items = item.get("evidence_items", [])
            item_pages = {
                row.get("physical_pdf_page")
                for row in evidence_items
                if isinstance(row, dict)
            }
            if set(pages).issubset(item_pages):
                evidence_complete += 1
            else:
                errors.append(f"{bundle_id}/{qa_id}:evidence_item_gap")
            review_statuses[item.get("review_status", "missing")] += 1
            modalities.update(item.get("modal_types", []))
    return {
        "status": "passed" if not errors else "failed",
        "bundles": len(release),
        "qa": qa_count,
        "strict_three_document_qa": strict_count,
        "hop_counts": dict(sorted(hop_counts.items())),
        "review_statuses": dict(review_statuses),
        "modalities": dict(modalities),
        "source_references": sum(source_counts.values()),
        "unique_source_papers": len(source_counts),
        "max_source_reuse": max(source_counts.values(), default=0),
        "evidence_items_complete": evidence_complete,
        "evidence_items_complete_rate": (
            evidence_complete / qa_count if qa_count else 0
        ),
        "pdf_physical_pages_total": sum(page_counts),
        "qa_per_pdf_min": min(per_bundle, default=0),
        "qa_per_pdf_max": max(per_bundle, default=0),
        "validation_errors": errors,
    }


def finalize(
    qa: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ledger = semantic_ledger_index(args.source_ledger)
    release = build_base_release(qa, manifests, ledger)
    (
        v2_rewrites_applied,
        v2_tri_added,
        published_v2_tri,
    ) = apply_published_v2_items(release)
    existing_questions = [
        item["question"]
        for bundle in release.values()
        for item in bundle["QA"].values()
    ]
    duplicate_log: list[dict[str, Any]] = []
    cached = cached_v2_strict_items(manifests)
    cached_extra = {
        bundle_id: [
            row
            for row in rows
            if (bundle_id, row[0]) not in published_v2_tri
        ]
        for bundle_id, rows in cached.items()
    }
    cached_extra = {
        bundle_id: rows
        for bundle_id, rows in cached_extra.items()
        if rows
    }
    cached_extra_added = append_strict_items(
        release,
        cached_extra,
        manifests,
        id_prefix="V2TRI",
        provenance={
            "source": "cross_pdf_quality_v2_20260728 cached strict review",
            "generator_model": "gemini-3-flash-preview",
            "reviewer_model": "claude-sonnet-5",
        },
        existing_questions=existing_questions,
        duplicate_log=duplicate_log,
    )
    new = strict_new_items(manifests, args.output_root)
    new_added = append_strict_items(
        release,
        new,
        manifests,
        id_prefix="V3TRI",
        provenance={
            "source": "cross_pdf_challenge_v3_20260729 focused generation",
            "generator_model": args.generator_model,
            "reviewer_model": args.reviewer_model,
        },
        existing_questions=existing_questions,
        duplicate_log=duplicate_log,
    )

    qa_count = sum(len(bundle["QA"]) for bundle in release.values())
    if qa_count < args.minimum_qa:
        summary = {
            "status": "insufficient_accepted_qa",
            "qa": qa_count,
            "minimum_qa": args.minimum_qa,
            "semantic_base": sum(
                len(bundle["QA"]) for bundle in qa.values()
            ),
            "v2_rewrites_applied": v2_rewrites_applied,
            "v2_published_tri_added": v2_tri_added,
            "cached_strict_extra_added": cached_extra_added,
            "cached_strict_added": v2_tri_added + cached_extra_added,
            "new_strict_added": new_added,
            "near_duplicate_candidates_excluded": len(duplicate_log),
        }
        atomic_json(args.output_root / "interim_summary.json", summary)
        atomic_json(
            args.output_root / "near_duplicate_exclusions.json",
            duplicate_log,
        )
        return summary

    expected_pdf_names = {
        output_pdf_path(args.output_pdf_dir, bundle_id).name for bundle_id in release
    }
    args.output_pdf_dir.mkdir(parents=True, exist_ok=True)
    stale = [] if args.output_pdf_dir.resolve() == PDF_ROOT.resolve() else [
        path for path in args.output_pdf_dir.glob("*.pdf")
        if path.name not in expected_pdf_names
    ]
    if stale:
        raise ValueError(
            "Output PDF directory contains stale files: "
            + ", ".join(path.name for path in stale)
        )
    pdf_hashes = {}
    for bundle_id in release:
        source = resolve_pdf_path(bundle_id, [args.source_pdf_dir])
        destination = output_pdf_path(args.output_pdf_dir, bundle_id)
        materialize_pdf(source, destination)
        pdf_hashes[bundle_id] = sha256_file(destination)

    validation = validate_release(release, manifests, args.output_pdf_dir)
    if validation["validation_errors"]:
        raise ValueError(
            "Release validation failed: "
            + json.dumps(
                validation["validation_errors"][:30], ensure_ascii=False
            )
        )
    args.output_qa.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_qa, release)
    selected_manifest = [manifests[bundle_id] for bundle_id in release]
    atomic_json(
        args.output_root / "selected_bundle_manifest.json",
        selected_manifest,
    )
    atomic_json(
        args.output_root / "near_duplicate_exclusions.json",
        duplicate_log,
    )
    summary = {
        **validation,
        "minimum_qa": args.minimum_qa,
        "semantic_base": sum(len(bundle["QA"]) for bundle in qa.values()),
        "v2_rewrites_applied": v2_rewrites_applied,
        "v2_published_tri_added": v2_tri_added,
        "cached_strict_extra_added": cached_extra_added,
        "cached_strict_added": v2_tri_added + cached_extra_added,
        "new_strict_added": new_added,
        "near_duplicate_candidates_excluded": len(duplicate_log),
        "models": {
            "generator": args.generator_model,
            "independent_reviewer": args.reviewer_model,
        },
        "outputs": {
            "qa_json": str(args.output_qa),
            "pdf_dir": str(args.output_pdf_dir),
            "selected_manifest": str(
                args.output_root / "selected_bundle_manifest.json"
            ),
            "manual_new_acceptance": str(
                args.output_root / MANUAL_NEW_ACCEPTANCE.name
            ),
            **(
                {"report": str(args.output_root / "REPORT.md")}
                if (args.output_root / "REPORT.md").exists()
                else {}
            ),
        },
        "output_sha256": {
            "qa_json": sha256_file(args.output_qa),
            "selected_manifest": sha256_file(
                args.output_root / "selected_bundle_manifest.json"
            ),
            "manual_new_acceptance": sha256_file(
                args.output_root / MANUAL_NEW_ACCEPTANCE.name
            ),
            **(
                {
                    "report": sha256_file(
                        args.output_root / "REPORT.md"
                    )
                }
                if (args.output_root / "REPORT.md").exists()
                else {}
            ),
        },
        "pdf_sha256": pdf_hashes,
    }
    atomic_json(args.output_root / "summary.json", summary)
    return summary


def ranked_bundles(
    qa: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    count: int,
) -> list[str]:
    cached = cached_v2_strict_items(manifests)
    scored = []
    for bundle_id, bundle in qa.items():
        items = list(bundle["QA"].values())
        modalities = sum(
            bool(set(item.get("modal_types", [])) - {"text"})
            for item in items
        )
        page_count = int(manifests[bundle_id]["page_count"])
        already_accepted = len(cached.get(bundle_id, []))
        score = (
            already_accepted == 0,
            modalities,
            len(items),
            page_count,
            bundle_id,
        )
        scored.append((score, bundle_id))
    return [
        bundle_id
        for _, bundle_id in sorted(scored, reverse=True)[:count]
    ]


def run_parallel(
    bundle_ids: list[str],
    worker,
    concurrency: int,
    output_root: Path,
    label: str,
) -> list[dict[str, Any]]:
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(worker, bundle_id): bundle_id
            for bundle_id in bundle_ids
        }
        for position, future in enumerate(as_completed(futures), start=1):
            bundle_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{label} {position}/{len(bundle_ids)}] {bundle_id}: "
                    f"validation_errors="
                    f"{len(result.get('validation_errors', []))}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "bundle_id": bundle_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                }
                errors.append(error)
                print(
                    f"[{label} {position}/{len(bundle_ids)}] {bundle_id}: "
                    f"ERROR {error['error_type']}: {error['error']}",
                    flush=True,
                )
    with _WRITE_LOCK:
        atomic_json(
            output_root / f"last_{label}_run.json",
            {"completed": len(results), "errors": errors},
        )
    if errors:
        raise RuntimeError(f"{len(errors)} {label} operations failed")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("select", "generate", "review", "finalize", "all"),
    )
    parser.add_argument("--source-qa", type=Path, default=SOURCE_QA)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--source-ledger", type=Path, default=SOURCE_LEDGER)
    parser.add_argument("--source-pdf-dir", type=Path, default=SOURCE_PDFS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-qa", type=Path, default=OUTPUT_QA)
    parser.add_argument("--output-pdf-dir", type=Path, default=OUTPUT_PDFS)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--generator-model", default="gemini-3-flash-preview"
    )
    parser.add_argument("--reviewer-model", default="claude-sonnet-5")
    parser.add_argument("--bundle-count", type=int, default=24)
    parser.add_argument("--bundle-id", action="append", default=[])
    parser.add_argument("--candidates-per-bundle", type=int, default=6)
    parser.add_argument("--minimum-qa", type=int, default=420)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    qa = load_json(args.source_qa)
    manifests = manifest_index(args.manifest)
    if args.bundle_id:
        missing = sorted(set(args.bundle_id) - set(qa))
        if missing:
            raise SystemExit(f"Unknown bundle ids: {missing}")
        selected = list(dict.fromkeys(args.bundle_id))
    else:
        selected = ranked_bundles(
            qa, manifests, max(0, args.bundle_count)
        )
    selection = {
        "selection_rule": (
            "prefer bundles without cached strict tri items, then multimodal "
            "coverage, verified QA count, and PDF length"
        ),
        "bundle_ids": selected,
        "bundle_count": len(selected),
        "candidates_per_bundle": args.candidates_per_bundle,
    }
    atomic_json(args.output_root / "generation_selection.json", selection)
    cached_count = sum(
        len(items) for items in cached_v2_strict_items(manifests).values()
    )
    print(
        json.dumps(
            {
                "action": args.action,
                "semantic_base_qa": sum(
                    len(bundle["QA"]) for bundle in qa.values()
                ),
                "cached_strict_tri": cached_count,
                "base_total_before_new_api": (
                    sum(len(bundle["QA"]) for bundle in qa.values())
                    + cached_count
                ),
                "selected_bundles": len(selected),
                "requested_new_candidates": (
                    len(selected) * args.candidates_per_bundle
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.action == "select":
        return 0
    if args.action == "finalize":
        summary = finalize(qa, manifests, args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary.get("qa", 0) >= args.minimum_qa else 2

    key = load_key(args.api_key_file)
    if args.action in {"generate", "all"}:
        run_parallel(
            selected,
            lambda bundle_id: process_generation(
                bundle_id,
                qa=qa,
                manifests=manifests,
                key=key,
                args=args,
            ),
            args.concurrency,
            args.output_root,
            "generate",
        )
    if args.action in {"review", "all"}:
        run_parallel(
            selected,
            lambda bundle_id: process_review(
                bundle_id,
                qa=qa,
                manifests=manifests,
                key=key,
                args=args,
            ),
            args.concurrency,
            args.output_root,
            "review",
        )
    if args.action == "all":
        summary = finalize(qa, manifests, args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary.get("qa", 0) >= args.minimum_qa else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
