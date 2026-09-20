#!/usr/bin/env python3
"""Build a harder, leakage-safe multi-document QA release.

The pipeline deliberately separates generation (Gemini) from independent
review (Claude), checkpoints every bundle, and never logs API credentials.
Evidence pages are always 1-based physical pages in the merged PDF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pypdf import PdfReader
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from pku_qa.pdf_assets import PDF_ROOT, output_pdf_path, resolve_pdf_path

try:
    from pku_qa.workflows.review.review_cross_pdf_qa_api import (
        CLAUDE_BASE_URL,
        call_gemini,
        extract_source_text,
        load_key,
        manifest_index,
        parse_json_response,
        post_with_retries,
        response_text_claude,
        source_rows,
    )
except ModuleNotFoundError:
    from pku_qa.workflows.review.review_cross_pdf_qa_api import (
        CLAUDE_BASE_URL,
        call_gemini,
        extract_source_text,
        load_key,
        manifest_index,
        parse_json_response,
        post_with_retries,
        response_text_claude,
        source_rows,
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
OUTPUT_ROOT = ROOT / "data/qa/4.cross_pdf/quality"
OUTPUT_QA = ROOT / "data/qa/4.cross_pdf/quality/qa_cross_pdf_challenge_v2.json"
OUTPUT_PDFS = PDF_ROOT

GENERATOR_SYSTEM = """\
You design difficult scientific multi-document QA, but you do not decide what
is finally accepted. Use only the supplied paper text.

Create three candidate questions whose complete answers each necessarily use
material facts from all three papers. A candidate fails if any one paper can be
removed and the question can still be answered completely. Prefer synthesis of
mechanisms, experimental constraints, quantitative findings, tables, figures,
or formulas over abstract-level topical comparison.
Use three visibly different question structures. At most one candidate may
start with "How", "What", or "Compare"; prefer a concrete decision, diagnosis,
claim assessment, design constraint, or conditional transfer where the papers
jointly determine the answer.

Also propose stylistically diverse rewrites of the supplied verified questions.
A rewrite must preserve the exact answer and evidence requirements. Avoid
repetitive openings such as "How do", "How does", "Compare the", and "What is".
Use natural styles such as decision under constraints, failure diagnosis,
claim assessment, method transfer, conditional reasoning, or evidence-led
synthesis. Do not make the question longer merely to sound difficult.

Use 1-based physical page numbers in the merged PDF. Return JSON only.
"""

REVIEWER_SYSTEM = """\
You are the independent senior gatekeeper for a hard multi-document scientific
QA benchmark. The candidates were produced by another model. Read all supplied
source-paper pages and verify every item from scratch.

For a three-document candidate, KEEP only when:
1. the complete answer necessarily uses substantive facts from Doc 1, Doc 2,
   and Doc 3; removing any one document makes the answer incomplete or
   non-unique;
2. the question is specific, natural, scientifically useful, and not a generic
   "common theme" or three parallel abstract summaries;
3. the answer is correct, complete, concise, and contains no invented relation;
4. every material answer claim is supported by the listed physical PDF pages;
5. evidence covers all three documents and the support list identifies the
   fact contributed by each document;
6. the item is meaningfully harder than two independent lookups.

For a rewrite, verify that its meaning, answer, and evidence needs are exactly
equivalent to the supplied already-verified original. Prefer the rewrite only
when it is genuinely less templated and remains clear.

FIX is allowed only for a small unambiguous correction. Otherwise REJECT.
Return JSON only, with no markdown.
"""

TRI_CRITERIA = (
    "all_three_documents_required",
    "question_specific_and_natural",
    "answer_correct",
    "answer_complete",
    "evidence_sufficient",
    "no_unsupported_claim",
    "harder_than_parallel_lookup",
)
REWRITE_CRITERIA = (
    "meaning_preserved",
    "answer_unchanged_and_correct",
    "evidence_requirements_preserved",
    "wording_clear",
    "template_diversity_improved",
)
TEMPLATE_PREFIXES = (
    "how do ",
    "how does ",
    "compare the ",
    "what is ",
)
_WRITE_LOCK = threading.Lock()


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


def question_prefix(question: str) -> str:
    words = re.findall(r"[A-Za-z]+", question.casefold())
    return " ".join(words[:2])


def mapped_docs(
    pages: list[int], sources: list[dict[str, Any]]
) -> set[int]:
    docs: set[int] = set()
    for page in pages:
        for number, source in enumerate(sources, start=1):
            if int(source["merged_start_page"]) <= page <= int(
                source["merged_end_page"]
            ):
                docs.add(number)
    return docs


def select_zero_overlap_bundles(
    qa: dict[str, Any], manifests: dict[str, dict[str, Any]]
) -> list[str]:
    """Solve maximum 3-set packing, then prefer richer existing bundles."""

    bundle_ids = sorted(qa)
    paper_ids = sorted(
        {
            str(source["paper_id"])
            for bundle_id in bundle_ids
            for source in manifests[bundle_id]["sources"]
        }
    )
    paper_index = {paper_id: index for index, paper_id in enumerate(paper_ids)}
    matrix = lil_matrix((len(paper_ids), len(bundle_ids)), dtype=float)
    quality = []
    for column, bundle_id in enumerate(bundle_ids):
        items = list(qa[bundle_id]["QA"].values())
        non_text = sum(
            bool(set(item.get("modal_types", [])) - {"text"})
            for item in items
        )
        prefixes = len({question_prefix(item["question"]) for item in items})
        quality.append(min(len(items), 10) * 100 + non_text * 10 + prefixes)
        for source in manifests[bundle_id]["sources"]:
            matrix[paper_index[str(source["paper_id"])], column] = 1

    # One more selected bundle is always worth more than all tie-break scores.
    objective = -np.array(
        [1_000_000 + value for value in quality], dtype=float
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(bundle_ids)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(
            matrix.tocsr(),
            lb=np.zeros(len(paper_ids)),
            ub=np.ones(len(paper_ids)),
        ),
        options={"time_limit": 120},
    )
    if result.x is None:
        raise RuntimeError(f"Bundle selection MILP failed: {result.message}")
    selected = [
        bundle_id
        for bundle_id, value in zip(bundle_ids, result.x, strict=True)
        if value >= 0.5
    ]
    used = [
        str(source["paper_id"])
        for bundle_id in selected
        for source in manifests[bundle_id]["sources"]
    ]
    if len(used) != len(set(used)):
        raise AssertionError("Zero-overlap selection reused a source paper")
    return sorted(selected)


def release_ledger() -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    with SOURCE_LEDGER.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[(str(row["bundle_id"]), str(row["qa_id"]))] = row
    return rows


def choose_existing_items(bundle: dict[str, Any], limit: int = 2) -> list[str]:
    """Prefer multimodal and less-templated verified items."""

    def score(row: tuple[str, dict[str, Any]]) -> tuple[int, int, int, str]:
        qa_id, item = row
        non_text = bool(set(item.get("modal_types", [])) - {"text"})
        templated = item["question"].casefold().startswith(TEMPLATE_PREFIXES)
        evidence_count = len(set(item.get("evidence_pages", [])))
        return (int(non_text), int(not templated), evidence_count, qa_id)

    ranked = sorted(bundle["QA"].items(), key=score, reverse=True)
    selected: list[str] = []
    prefixes: set[str] = set()
    for qa_id, item in ranked:
        prefix = question_prefix(item["question"])
        if prefix in prefixes and len(ranked) > limit:
            continue
        selected.append(qa_id)
        prefixes.add(prefix)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for qa_id, _ in ranked:
            if qa_id not in selected:
                selected.append(qa_id)
            if len(selected) == limit:
                break
    return selected


def source_text(
    bundle_manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    blocks = []
    metadata = []
    for source in source_rows(bundle_manifest):
        text, page_count, errors = extract_source_text(source)
        blocks.append(
            f"\n######## DOC {source.doc_number}: {source.title} ########\n"
            f"paper_id={source.paper_id}; merged physical pages "
            f"{source.merged_start}-{source.merged_end}\n{text}"
        )
        metadata.append(
            {
                "doc_number": source.doc_number,
                "paper_id": source.paper_id,
                "title": source.title,
                "merged_start_page": source.merged_start,
                "merged_end_page": source.merged_end,
                "source_page_count": page_count,
                "extraction_errors": errors,
            }
        )
    return "".join(blocks), metadata


def generator_shape(rewrite_ids: list[str]) -> dict[str, Any]:
    return {
        "rewrites": [
            {
                "qa_id": qa_id,
                "rewritten_question": "Semantically equivalent but less templated.",
                "style": (
                    "constraint | diagnosis | claim_assessment | method_transfer "
                    "| conditional | evidence_synthesis"
                ),
            }
            for qa_id in rewrite_ids
        ],
        "tri_candidates": [
            {
                "candidate_id": f"T{index}",
                "question": "Question requiring all three documents.",
                "answer": "Complete concise answer.",
                "evidence_pages": [1, 20, 40],
                "evidence_items": [
                    {
                        "doc_number": 1,
                        "physical_pdf_page": 1,
                        "supported_fact": "Material fact from this page.",
                    }
                ],
                "modal_types": ["text"],
                "question_category": "Specific scientific synthesis category",
                "answer_format": "string",
                "necessity": {
                    "doc_1": "Why Doc 1 is necessary.",
                    "doc_2": "Why Doc 2 is necessary.",
                    "doc_3": "Why Doc 3 is necessary.",
                },
            }
            for index in range(1, 4)
        ],
    }


def build_generation_prompt(
    bundle_id: str,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    rewrite_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    text, sources = source_text(manifest)
    verified = {
        qa_id: bundle["QA"][qa_id]
        for qa_id in rewrite_ids
    }
    prompt = (
        f"bundle_id={bundle_id}\n"
        "Source map:\n"
        + json.dumps(sources, ensure_ascii=False, indent=2)
        + "\n\nVerified items to rewrite:\n"
        + json.dumps(verified, ensure_ascii=False, indent=2)
        + "\n\nRequired output shape:\n"
        + json.dumps(generator_shape(rewrite_ids), ensure_ascii=False, indent=2)
        + "\n\nFull source papers:\n"
        + text
    )
    return prompt, {"source_meta": sources, "prompt_chars": len(prompt)}


def validate_generation(
    result: dict[str, Any],
    rewrite_ids: list[str],
    manifest: dict[str, Any],
) -> list[str]:
    errors = []
    rewrites = result.get("rewrites")
    if not isinstance(rewrites, list):
        errors.append("rewrites_not_list")
    else:
        actual = [str(row.get("qa_id")) for row in rewrites if isinstance(row, dict)]
        if set(actual) != set(rewrite_ids) or len(actual) != len(rewrite_ids):
            errors.append("rewrite_id_mismatch")
        for row in rewrites:
            if not isinstance(row, dict) or not str(
                row.get("rewritten_question", "")
            ).strip():
                errors.append("invalid_rewrite")

    candidates = result.get("tri_candidates")
    if not isinstance(candidates, list) or len(candidates) < 1:
        errors.append("no_tri_candidates")
        return errors
    seen: set[str] = set()
    for row in candidates:
        if not isinstance(row, dict):
            errors.append("candidate_not_object")
            continue
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            errors.append("invalid_candidate_id")
        seen.add(candidate_id)
        if not str(row.get("question", "")).strip():
            errors.append(f"{candidate_id}:blank_question")
        if not str(row.get("answer", "")).strip():
            errors.append(f"{candidate_id}:blank_answer")
        pages = row.get("evidence_pages")
        if (
            not isinstance(pages, list)
            or not pages
            or not all(
                isinstance(page, int)
                and not isinstance(page, bool)
                and page >= 1
                for page in pages
            )
        ):
            errors.append(f"{candidate_id}:invalid_pages")
        elif mapped_docs(pages, manifest["sources"]) != {1, 2, 3}:
            errors.append(f"{candidate_id}:pages_do_not_cover_three_docs")
    return errors


def structurally_valid_tri_candidates(
    result: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    valid = []
    seen: set[str] = set()
    for row in result.get("tri_candidates") or []:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id", ""))
        pages = row.get("evidence_pages")
        if (
            not candidate_id
            or candidate_id in seen
            or not str(row.get("question", "")).strip()
            or not str(row.get("answer", "")).strip()
            or not isinstance(pages, list)
            or not pages
            or not all(
                isinstance(page, int)
                and not isinstance(page, bool)
                and page >= 1
                for page in pages
            )
            or mapped_docs(pages, manifest["sources"]) != {1, 2, 3}
        ):
            continue
        seen.add(candidate_id)
        valid.append(row)
    return valid


def call_generator(
    key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, text = call_gemini(
        key,
        model,
        prompt,
        [],
        max_tokens,
        timeout,
        retries,
    )
    return raw, parse_json_response(text)


def reviewer_shape(
    rewrite_ids: list[str], candidate_ids: list[str]
) -> dict[str, Any]:
    return {
        "rewrites": [
            {
                "qa_id": qa_id,
                "decision": "KEEP_REWRITE | KEEP_ORIGINAL | REJECT",
                "criteria": {name: True for name in REWRITE_CRITERIA},
                "confidence": 0.95,
                "reason": "Paper-grounded reason.",
                "verified_support": [
                    {
                        "doc_number": 1,
                        "physical_pdf_page": 1,
                        "supported_fact": "Verified fact.",
                    }
                ],
            }
            for qa_id in rewrite_ids
        ],
        "tri_candidates": [
            {
                "candidate_id": candidate_id,
                "decision": "KEEP | FIX | REJECT",
                "criteria": {name: True for name in TRI_CRITERIA},
                "confidence": 0.95,
                "reason": "Paper-grounded reason.",
                "verified_support": [
                    {
                        "doc_number": 1,
                        "physical_pdf_page": 1,
                        "supported_fact": "Verified fact.",
                    }
                ],
                "corrected_question": None,
                "corrected_answer": None,
                "corrected_evidence_pages": None,
            }
            for candidate_id in candidate_ids
        ],
    }


def build_review_prompt(
    bundle_id: str,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    generation: dict[str, Any],
    rewrite_ids: list[str],
) -> str:
    text, sources = source_text(manifest)
    payload = {
        "verified_originals": {
            qa_id: bundle["QA"][qa_id] for qa_id in rewrite_ids
        },
        "generator_output": generation,
    }
    candidate_ids = [
        str(row["candidate_id"])
        for row in generation.get("tri_candidates", [])
    ]
    return (
        f"bundle_id={bundle_id}\n"
        "Source map:\n"
        + json.dumps(sources, ensure_ascii=False, indent=2)
        + "\n\nItems to review:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nRequired output shape:\n"
        + json.dumps(
            reviewer_shape(rewrite_ids, candidate_ids),
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nFull source papers:\n"
        + text
    )


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
    return raw, parse_json_response(response_text_claude(raw))


def validate_review(
    result: dict[str, Any],
    rewrite_ids: list[str],
    candidate_ids: list[str],
) -> list[str]:
    errors = []
    rewrites = result.get("rewrites")
    if not isinstance(rewrites, list):
        errors.append("rewrites_not_list")
    else:
        actual = [str(row.get("qa_id")) for row in rewrites if isinstance(row, dict)]
        if set(actual) != set(rewrite_ids) or len(actual) != len(rewrite_ids):
            errors.append("rewrite_id_mismatch")
        for row in rewrites:
            qa_id = row.get("qa_id")
            if row.get("decision") not in {
                "KEEP_REWRITE",
                "KEEP_ORIGINAL",
                "REJECT",
            }:
                errors.append(f"{qa_id}:invalid_rewrite_decision")
            criteria = row.get("criteria")
            if not isinstance(criteria, dict) or any(
                not isinstance(criteria.get(name), bool)
                for name in REWRITE_CRITERIA
            ):
                errors.append(f"{qa_id}:invalid_rewrite_criteria")

    candidates = result.get("tri_candidates")
    if not isinstance(candidates, list):
        return errors + ["tri_candidates_not_list"]
    actual_candidates = [
        str(row.get("candidate_id"))
        for row in candidates
        if isinstance(row, dict)
    ]
    if (
        set(actual_candidates) != set(candidate_ids)
        or len(actual_candidates) != len(candidate_ids)
    ):
        errors.append("candidate_id_mismatch")
    for row in candidates:
        candidate_id = row.get("candidate_id")
        if row.get("decision") not in {"KEEP", "FIX", "REJECT"}:
            errors.append(f"{candidate_id}:invalid_candidate_decision")
        criteria = row.get("criteria")
        if not isinstance(criteria, dict) or any(
            not isinstance(criteria.get(name), bool) for name in TRI_CRITERIA
        ):
            errors.append(f"{candidate_id}:invalid_tri_criteria")
        support = row.get("verified_support")
        if not isinstance(support, list):
            errors.append(f"{candidate_id}:support_not_list")
    return errors


def usage(raw: dict[str, Any], provider: str) -> dict[str, Any]:
    if provider == "gemini":
        return raw.get("usageMetadata", {})
    return raw.get("usage", {})


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
    rewrite_ids = choose_existing_items(qa[bundle_id], 2)
    prompt, metadata = build_generation_prompt(
        bundle_id,
        qa[bundle_id],
        manifests[bundle_id],
        rewrite_ids,
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
    generation["tri_candidates"] = structurally_valid_tri_candidates(
        generation, manifests[bundle_id]
    )
    errors = validate_generation(
        generation, rewrite_ids, manifests[bundle_id]
    )
    record = {
        "bundle_id": bundle_id,
        "generator_model": args.generator_model,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "duration_seconds": time.time() - started,
        "input_fingerprint_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "metadata": metadata,
        "rewrite_ids": rewrite_ids,
        "generation": generation,
        "validation_errors": errors,
        "usage": usage(raw, "gemini"),
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
    generation_record = process_generation(
        bundle_id,
        qa=qa,
        manifests=manifests,
        key=key,
        args=args,
    )
    if generation_record.get("validation_errors"):
        record = {
            "bundle_id": bundle_id,
            "reviewer_model": args.reviewer_model,
            "reviewed_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "review": {"rewrites": [], "tri_candidates": []},
            "validation_errors": [
                "generation_not_reviewable",
                *generation_record["validation_errors"],
            ],
            "skipped": True,
        }
        atomic_json(path, record)
        return record
    generation = generation_record["generation"]
    rewrite_ids = generation_record["rewrite_ids"]
    candidate_ids = [
        str(row["candidate_id"])
        for row in generation["tri_candidates"]
    ]
    prompt = build_review_prompt(
        bundle_id,
        qa[bundle_id],
        manifests[bundle_id],
        generation,
        rewrite_ids,
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
    errors = validate_review(review, rewrite_ids, candidate_ids)
    record = {
        "bundle_id": bundle_id,
        "reviewer_model": args.reviewer_model,
        "reviewed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "duration_seconds": time.time() - started,
        "input_fingerprint_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "review": review,
        "validation_errors": errors,
        "usage": usage(raw, "claude"),
        "raw_response": raw,
    }
    atomic_json(path, record)
    return record


def run_parallel(
    bundle_ids: list[str],
    worker,
    concurrency: int,
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
                error_count = len(result.get("validation_errors", []))
                print(
                    f"[{position}/{len(bundle_ids)}] {bundle_id}: "
                    f"saved, validation_errors={error_count}",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "bundle_id": bundle_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[{position}/{len(bundle_ids)}] {bundle_id}: "
                    f"ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )
    atomic_json(
        OUTPUT_ROOT / "last_parallel_run.json",
        {"completed": len(results), "errors": errors},
    )
    if errors:
        raise RuntimeError(f"{len(errors)} bundle operations failed")
    return results


def apply_tri_fix(
    candidate: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(candidate)
    if review["decision"] == "FIX":
        for source_name, target_name in (
            ("corrected_question", "question"),
            ("corrected_answer", "answer"),
            ("corrected_evidence_pages", "evidence_pages"),
        ):
            value = review.get(source_name)
            if value not in (None, "", []):
                result[target_name] = deepcopy(value)
    return result


def evidence_items_from_support(
    support: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in support:
        page = int(row["physical_pdf_page"])
        doc_number = int(row["doc_number"])
        source = sources[doc_number - 1]
        result.append(
            {
                "physical_pdf_page": page,
                "source_doc_number": doc_number,
                "source_page": page - int(source["merged_start_page"]) + 1,
                "source_paper_id": source["paper_id"],
                "supported_fact": str(row["supported_fact"]).strip(),
            }
        )
    return result


def diversify_question_opening(question: str, qa_uid: str) -> str:
    """Apply a meaning-preserving opening rewrite to fixed templates."""

    stripped = question.strip()
    lowered = stripped.casefold()
    templates = {
        "how do ": (
            "Explain how ",
            "Describe how ",
            "Trace how ",
            "Using the cited evidence, determine how ",
            "Across the cited studies, establish how ",
        ),
        "how does ": (
            "Explain how ",
            "Clarify how ",
            "Describe how ",
            "Using the cited evidence, determine how ",
            "In the cited setting, establish how ",
        ),
        "compare the ": (
            "Contrast the ",
            "Set the following in contrast: the ",
            "Distinguish the ",
        ),
        "what is ": (
            "Identify ",
            "State ",
            "Determine ",
        ),
    }
    for prefix, replacements in templates.items():
        if not lowered.startswith(prefix):
            continue
        digest = hashlib.sha256(qa_uid.encode("utf-8")).digest()
        replacement = replacements[digest[0] % len(replacements)]
        remainder = stripped[len(prefix) :]
        return replacement + remainder
    return stripped


def normalize_original_support(
    ledger_row: dict[str, Any],
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return evidence_items_from_support(
        [
            {
                "doc_number": row["doc_number"],
                "physical_pdf_page": row["merged_page"],
                "supported_fact": row["supported_fact"],
            }
            for row in ledger_row.get("claude_support", [])
        ],
        sources,
    )


def enriched_item(
    item: dict[str, Any],
    *,
    bundle_id: str,
    qa_id: str,
    sources: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(item)
    original_question = str(result["question"]).strip()
    normalized_question = diversify_question_opening(
        original_question, f"{bundle_id}/{qa_id}"
    )
    provenance = deepcopy(provenance)
    if normalized_question != original_question:
        result["question"] = normalized_question
        result["question_style_normalization"] = (
            "meaning_preserving_opening_diversification_v1"
        )
        provenance["pre_style_normalization_question"] = original_question
        provenance["style_normalization"] = (
            "deterministic meaning-preserving opening rewrite"
        )
    pages = sorted({int(page) for page in result["evidence_pages"]})
    docs = sorted(mapped_docs(pages, sources))
    result.update(
        {
            "qa_uid": f"{bundle_id}/{qa_id}",
            "evidence_pages": pages,
            "evidence_page_numbering": "physical_pdf_page_1_based",
            "evidence_items": evidence_items,
            "evidence_hops": len(docs),
            "evidence_source_docs": docs,
            "evidence_span": max(pages) - min(pages) if pages else 0,
            "source_paper_ids": [
                sources[number - 1]["paper_id"] for number in docs
            ],
            "answer_format": result.get("answer_format", "string"),
            "answer_aliases": result.get("answer_aliases", []),
            "answer_unit": result.get("answer_unit"),
            "numeric_tolerance": result.get("numeric_tolerance"),
            "annotation_provenance": provenance,
            "review_status": "independent_model_review_pass",
        }
    )
    return result


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


def finalize(
    selected: list[str],
    qa: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    ledger_index = release_ledger()
    output: dict[str, Any] = {}
    release_manifest = []
    audit_rows = []
    exclusion_counts: Counter[str] = Counter()

    for bundle_id in selected:
        generation_record = load_json(
            args.output_root / "generation" / f"{bundle_id}.json"
        )
        review_record = load_json(
            args.output_root / "reviews" / f"{bundle_id}.json"
        )
        if generation_record.get("validation_errors"):
            exclusion_counts["invalid_generation_bundle"] += 1
            continue
        if review_record.get("validation_errors"):
            exclusion_counts["invalid_review_bundle"] += 1
            continue
        manifest = manifests[bundle_id]
        sources = manifest["sources"]
        generation = generation_record["generation"]
        review = review_record["review"]
        rewrite_by_id = {
            str(row["qa_id"]): row for row in generation["rewrites"]
        }
        rewrite_review = {
            str(row["qa_id"]): row for row in review["rewrites"]
        }
        tri_by_id = {
            str(row["candidate_id"]): row
            for row in generation["tri_candidates"]
        }
        tri_review = {
            str(row["candidate_id"]): row
            for row in review["tri_candidates"]
        }

        accepted_tri = []
        for candidate_id, candidate in tri_by_id.items():
            decision = tri_review.get(candidate_id, {})
            criteria = decision.get("criteria", {})
            if decision.get("decision") not in {"KEEP", "FIX"}:
                exclusion_counts["tri_rejected"] += 1
                continue
            if not all(criteria.get(name) is True for name in TRI_CRITERIA):
                exclusion_counts["tri_failed_criteria"] += 1
                continue
            fixed = apply_tri_fix(candidate, decision)
            pages = fixed.get("evidence_pages", [])
            if mapped_docs(pages, sources) != {1, 2, 3}:
                exclusion_counts["tri_not_three_doc_after_review"] += 1
                continue
            support = decision.get("verified_support", [])
            support_docs = {
                int(row["doc_number"])
                for row in support
                if isinstance(row, dict)
                and isinstance(row.get("doc_number"), int)
            }
            if support_docs != {1, 2, 3}:
                exclusion_counts["tri_review_support_not_three_doc"] += 1
                continue
            accepted_tri.append((candidate_id, fixed, decision))

        desired_existing = min(
            2, max(0, args.qa_per_bundle - len(accepted_tri[:2]))
        )
        existing_ids = generation_record["rewrite_ids"][:desired_existing]
        items: list[tuple[str, dict[str, Any]]] = []
        for qa_id in existing_ids:
            original = deepcopy(qa[bundle_id]["QA"][qa_id])
            decision = rewrite_review.get(qa_id, {})
            rewritten = rewrite_by_id.get(qa_id, {})
            if (
                decision.get("decision") == "KEEP_REWRITE"
                and all(
                    decision.get("criteria", {}).get(name) is True
                    for name in REWRITE_CRITERIA
                )
            ):
                original["question"] = rewritten["rewritten_question"].strip()
                original["question_style"] = rewritten.get(
                    "style", "model_rewrite"
                )
                rewrite_status = "accepted"
            else:
                original["question_style"] = "verified_original"
                rewrite_status = "original_retained"
            ledger_row = ledger_index[(bundle_id, qa_id)]
            verified_support = decision.get("verified_support", [])
            verified_pages = sorted(
                {
                    int(row["physical_pdf_page"])
                    for row in verified_support
                    if isinstance(row, dict)
                    and isinstance(row.get("physical_pdf_page"), int)
                }
            )
            if len(mapped_docs(verified_pages, sources)) >= 2:
                original["evidence_pages"] = verified_pages
                existing_evidence_items = evidence_items_from_support(
                    verified_support, sources
                )
            else:
                existing_evidence_items = normalize_original_support(
                    ledger_row, sources
                )
            item = enriched_item(
                original,
                bundle_id=bundle_id,
                qa_id=qa_id,
                sources=sources,
                evidence_items=existing_evidence_items,
                provenance={
                    "source_release": SOURCE_QA.name,
                    "source_release_tier": ledger_row["release_tier"],
                    "rewrite_generator": args.generator_model,
                    "rewrite_reviewer": args.reviewer_model,
                    "rewrite_status": rewrite_status,
                },
            )
            items.append((qa_id, item))

        tri_needed = args.qa_per_bundle - len(items)
        for candidate_id, candidate, decision in accepted_tri[:tri_needed]:
            qa_id = f"TRI_{candidate_id}"
            verified_pages = sorted(
                {
                    int(row["physical_pdf_page"])
                    for row in decision["verified_support"]
                }
            )
            item = {
                "question": candidate["question"].strip(),
                "answer": candidate["answer"].strip(),
                "evidence_pages": verified_pages,
                "modal_types": candidate.get("modal_types", ["text"]),
                "question_type": "Inferential",
                "question_category": candidate.get(
                    "question_category", "Three-document synthesis"
                ),
                "question_style": "three_document_synthesis",
                "answer_format": candidate.get("answer_format", "string"),
            }
            item = enriched_item(
                item,
                bundle_id=bundle_id,
                qa_id=qa_id,
                sources=sources,
                evidence_items=evidence_items_from_support(
                    decision["verified_support"], sources
                ),
                provenance={
                    "generator_model": args.generator_model,
                    "reviewer_model": args.reviewer_model,
                    "review_confidence": decision.get("confidence"),
                    "review_reason": decision.get("reason"),
                    "review_criteria": decision.get("criteria"),
                },
            )
            items.append((qa_id, item))

        if len(items) < args.qa_per_bundle or not any(
            item["evidence_hops"] == 3 for _, item in items
        ):
            exclusion_counts["bundle_insufficient_final_items"] += 1
            continue

        payload = {
            key: deepcopy(value)
            for key, value in qa[bundle_id].items()
            if key != "QA"
        }
        payload.update(
            {
                "paper": bundle_id,
                "pdf_page_numbering": "physical_pdf_page_1_based",
                "source_paper_ids": [
                    source["paper_id"] for source in sources
                ],
                "source_documents": [
                    {
                        "doc_number": number,
                        "paper_id": source["paper_id"],
                        "arxiv_id": source["arxiv_id"],
                        "title": source["title"],
                        "merged_start_page": source["merged_start_page"],
                        "merged_end_page": source["merged_end_page"],
                        "source_page_count": source["source_page_count"],
                    }
                    for number, source in enumerate(sources, start=1)
                ],
                "QA": dict(items[: args.qa_per_bundle]),
            }
        )
        output[bundle_id] = payload
        release_manifest.append(manifest)
        for qa_id, item in payload["QA"].items():
            audit_rows.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "evidence_hops": item["evidence_hops"],
                    "source_paper_ids": item["source_paper_ids"],
                    "question_style": item["question_style"],
                    "review_status": item["review_status"],
                }
            )
    accepted_before_overlap_pruning = len(output)
    if output:
        leakage_safe_ids = set(
            select_zero_overlap_bundles(
                output,
                {bundle_id: manifests[bundle_id] for bundle_id in output},
            )
        )
        output = {
            bundle_id: bundle
            for bundle_id, bundle in output.items()
            if bundle_id in leakage_safe_ids
        }
        release_manifest = [
            row for row in release_manifest if row["id"] in leakage_safe_ids
        ]
        audit_rows = [
            row
            for row in audit_rows
            if row["bundle_id"] in leakage_safe_ids
        ]
    for bundle_id in output:
        source = resolve_pdf_path(bundle_id, [SOURCE_PDFS])
        materialize_pdf(
            source,
            output_pdf_path(args.output_pdf_dir, bundle_id),
        )
    if args.output_pdf_dir.is_dir() and args.output_pdf_dir.resolve() != PDF_ROOT.resolve():
        for pdf_path in args.output_pdf_dir.glob("*.pdf"):
            if pdf_path.stem not in output:
                pdf_path.unlink()
    atomic_json(args.output_qa, output)
    atomic_json(args.output_root / "selected_bundle_manifest.json", release_manifest)
    audit_path = args.output_root / "release_ledger.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in audit_rows
        ),
        encoding="utf-8",
    )
    summary = audit_dataset(
        output, release_manifest, args.output_pdf_dir
    )
    summary["selection"] = {
        "candidate_bundles": len(qa),
        "generation_scope_bundles": len(selected),
        "accepted_before_overlap_pruning": accepted_before_overlap_pruning,
        "zero_overlap_selected": len(output),
        "released_bundles": len(output),
        "excluded": dict(exclusion_counts),
    }
    summary["models"] = {
        "generator": args.generator_model,
        "independent_reviewer": args.reviewer_model,
    }
    atomic_json(args.output_root / "summary.json", summary)
    return summary


def audit_dataset(
    qa: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    pdf_dir: Path,
) -> dict[str, Any]:
    manifests = {row["id"]: row for row in manifest_rows}
    qa_count = 0
    hop_counts: Counter[int] = Counter()
    modalities: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    page_counts = []
    per_pdf_qa = []
    errors = []
    questions: set[str] = set()
    for bundle_id, bundle in qa.items():
        manifest = manifests[bundle_id]
        sources = manifest["sources"]
        for source in sources:
            source_counts[str(source["paper_id"])] += 1
        pdf_path = resolve_pdf_path(bundle_id, [pdf_dir], required=False)
        if pdf_path is None:
            errors.append(f"{bundle_id}:missing_pdf")
            continue
        actual_pages = len(PdfReader(pdf_path).pages)
        page_counts.append(actual_pages)
        if actual_pages != int(manifest["page_count"]):
            errors.append(f"{bundle_id}:pdf_page_count_mismatch")
        per_pdf_qa.append(len(bundle["QA"]))
        for qa_id, item in bundle["QA"].items():
            qa_count += 1
            normalized = " ".join(item["question"].casefold().split())
            if normalized in questions:
                errors.append(f"{bundle_id}/{qa_id}:duplicate_question")
            questions.add(normalized)
            pages = item.get("evidence_pages", [])
            docs = mapped_docs(pages, sources)
            hop_counts[len(docs)] += 1
            modalities.update(item.get("modal_types", []))
            prefixes[question_prefix(item["question"])] += 1
            if any(page < 1 or page > actual_pages for page in pages):
                errors.append(f"{bundle_id}/{qa_id}:evidence_out_of_range")
            if item.get("evidence_page_numbering") != (
                "physical_pdf_page_1_based"
            ):
                errors.append(f"{bundle_id}/{qa_id}:page_numbering_missing")
            evidence_items = item.get("evidence_items", [])
            evidence_item_pages = {
                row.get("physical_pdf_page") for row in evidence_items
            }
            if not set(pages).issubset(evidence_item_pages):
                errors.append(f"{bundle_id}/{qa_id}:evidence_item_gap")
    return {
        "status": "passed" if not errors else "failed",
        "bundles": len(qa),
        "qa": qa_count,
        "unique_source_papers": len(source_counts),
        "source_reference_count": sum(source_counts.values()),
        "max_source_reuse": max(source_counts.values(), default=0),
        "hop_counts": dict(sorted(hop_counts.items())),
        "modalities": dict(modalities),
        "top_question_prefixes": prefixes.most_common(15),
        "pdf_physical_pages_total": sum(page_counts),
        "pdf_physical_pages_min": min(page_counts, default=0),
        "pdf_physical_pages_max": max(page_counts, default=0),
        "qa_per_pdf_min": min(per_pdf_qa, default=0),
        "qa_per_pdf_max": max(per_pdf_qa, default=0),
        "validation_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "select",
            "revalidate",
            "generate",
            "review",
            "finalize",
            "all",
        ),
    )
    parser.add_argument("--source-qa", type=Path, default=SOURCE_QA)
    parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-qa", type=Path, default=OUTPUT_QA)
    parser.add_argument("--output-pdf-dir", type=Path, default=OUTPUT_PDFS)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--generator-model", default="gemini-3-flash-preview"
    )
    parser.add_argument("--reviewer-model", default="claude-sonnet-5")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--qa-per-bundle", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bundle-id", action="append", default=[])
    parser.add_argument(
        "--selection-mode",
        choices=("zero-overlap", "all"),
        default="zero-overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa = load_json(args.source_qa)
    manifests = manifest_index(args.manifest)
    selected_path = args.output_root / "zero_overlap_selection.json"
    if args.selection_mode == "all":
        selected = sorted(qa)
    elif selected_path.exists():
        selected = load_json(selected_path)["bundle_ids"]
    else:
        selected = select_zero_overlap_bundles(qa, manifests)
        atomic_json(
            selected_path,
            {
                "selection_rule": (
                    "maximum bundle count subject to each source paper "
                    "appearing in at most one bundle"
                ),
                "bundle_ids": selected,
            },
        )
    if args.bundle_id:
        requested = set(args.bundle_id)
        selected = [bundle_id for bundle_id in selected if bundle_id in requested]
    print(
        json.dumps(
            {
                "selected_bundles": len(selected),
                "selected_source_papers": len(
                    {
                        str(source["paper_id"])
                        for bundle_id in selected
                        for source in manifests[bundle_id]["sources"]
                    }
                ),
                "action": args.action,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.action == "select":
        return
    if args.action == "revalidate":
        for bundle_id in selected:
            path = args.output_root / "generation" / f"{bundle_id}.json"
            if not path.exists():
                continue
            record = load_json(path)
            generation = record["generation"]
            generation["tri_candidates"] = structurally_valid_tri_candidates(
                generation, manifests[bundle_id]
            )
            record["validation_errors"] = validate_generation(
                generation,
                record["rewrite_ids"],
                manifests[bundle_id],
            )
            atomic_json(path, record)
        return
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
        )
    if args.action in {"finalize", "all"}:
        summary = finalize(selected, qa, manifests, args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
