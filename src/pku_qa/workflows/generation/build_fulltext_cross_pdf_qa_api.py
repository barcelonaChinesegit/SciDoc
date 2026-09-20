#!/usr/bin/env python3
"""Generate grounded Cross-PDF QA from complete source-paper text.

This is a resumable candidate generator, not a publication shortcut. Its output
must still pass the independent Claude/Gemini full-PDF review and strict
finalizer before release.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pku_qa.workflows.review.review_cross_pdf_qa_api import (
    CLAUDE_BASE_URL,
    GEMINI_BASE_URL,
    extract_source_text,
    load_json,
    load_key,
    manifest_index,
    parse_json_response,
    post_with_retries,
    response_text_claude,
    response_text_gemini,
    source_rows,
)


ROOT = Path(__file__).resolve().parents[4]
ALLOWED_REASONING_TYPES = {
    "adjacent_module_dependency",
    "metric_reasoning",
    "component_hierarchy",
    "method_transfer",
    "compatibility_judgment",
    "conflict_resolution",
}
SYSTEM_PROMPT = """\
You construct publication-quality multi-paper scientific QA. Read every supplied
source-paper page before writing. Generate only questions whose one complete
answer necessarily uses substantive facts from every source paper and derives a
scientifically useful comparison, selection, taxonomy, invariant, compatibility
decision, or conflict resolution.

Reject arbitrary arithmetic, invented pipelines, unrelated external constraints,
parallel subquestions, and answers that merely concatenate one fact per paper.
An explicit selection constraint is valid only when it is natural for this exact
shared research topic. Do not claim the papers collaborated or evaluated one
another unless the text says so. Use no outside knowledge. Cite exact merged page
numbers that directly support every material answer claim. Return JSON only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("claude", "gemini"), default="claude")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--items-per-bundle", type=int, default=6)
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=14000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--bundle-id", action="append", default=[])
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--revalidate-existing", action="store_true")
    parser.add_argument("--generation-round", type=int, default=1)
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def expected_shape(bundle_id: str, source_count: int) -> dict[str, Any]:
    return {
        "bundle_id": bundle_id,
        "items": [
            {
                "qa_id": "FTQA_unique_stable_name",
                "question": "one coherent question",
                "answer": "complete but concise answer",
                "evidence_pages": list(range(1, source_count + 1)),
                "source_doc_numbers": list(range(1, source_count + 1)),
                "source_document_count": source_count,
                "reasoning_type": "one allowed type",
                "relation": "why every source is indispensable",
                "derived_conclusion": "one result derived from all sources",
                "intermediate_facts": [
                    {"doc_number": number, "merged_page": number, "fact": "fact"}
                    for number in range(1, source_count + 1)
                ],
            }
        ],
    }


def build_prompt(
    bundle: dict[str, Any], item_count: int, generation_round: int = 1
) -> tuple[str, dict[str, Any]]:
    sources = source_rows(bundle)
    blocks: list[str] = []
    errors: list[str] = []
    for source in sources:
        text, page_count, extraction_errors = extract_source_text(source)
        errors.extend(f"Doc {source.doc_number}: {error}" for error in extraction_errors)
        blocks.append(
            f"\n\n######## PAPER {source.doc_number}: {source.title} ########\n"
            f"paper_id={source.paper_id}; merged pages "
            f"{source.merged_start}-{source.merged_end}; source_pages={page_count}\n{text}"
        )
    source_count = len(sources)
    prompt = (
        f"BUNDLE: {bundle['id']}\nGENERATION ROUND: {generation_round}\n"
        f"Generate up to {item_count} genuinely distinct items. Return fewer or no "
        "items when the papers do not support that many natural syntheses. Every "
        f"item must require all {source_count} papers.\n\n"
        "Prefer questions that reconcile different theorem scopes, explain distinct "
        "thresholds for related phenomena, classify methods under a shared native "
        "criterion, or resolve an apparent conflict. The final answer must state the "
        "answer-critical bridge facts from all papers. Do not ask which paper says "
        "what. Do not infer that absence of a statement proves a method lacks a "
        "property. Avoid generic similarities/differences.\n\n"
        "This round must use materially different questions and inference angles "
        "from lower-numbered rounds; do not repeat the most obvious comparison. "
        "Each intermediate fact needs its exact merged page. evidence_pages is the "
        "sorted union of answer-critical pages and must include at least one page "
        "from each paper. source_doc_numbers must contain every paper number. "
        "Allowed reasoning_type values: "
        + ", ".join(sorted(ALLOWED_REASONING_TYPES))
        + ".\n\nREQUIRED OUTPUT SHAPE:\n"
        + json.dumps(expected_shape(str(bundle["id"]), source_count), ensure_ascii=False, indent=2)
        + "\n\nFULL SOURCE PAPERS FOLLOW:"
        + "".join(blocks)
    )
    return prompt, {
        "source_count": source_count,
        "prompt_characters": len(prompt),
        "extraction_errors": errors,
    }


def normalize_item(
    item: dict[str, Any], bundle: dict[str, Any], index: int
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    sources = source_rows(bundle)
    source_count = len(sources)
    result = dict(item)
    question = str(result.get("question", "")).strip()
    answer = str(result.get("answer", "")).strip()
    if not question or not answer:
        errors.append("empty_question_or_answer")
    if answer.casefold() in question.casefold() and len(answer) >= 5:
        errors.append("answer_leakage")
    if result.get("reasoning_type") not in ALLOWED_REASONING_TYPES:
        errors.append("invalid_reasoning_type")
    docs = result.get("source_doc_numbers")
    expected_docs = list(range(1, source_count + 1))
    if not isinstance(docs, list) or sorted(set(docs)) != expected_docs:
        errors.append("source_docs_do_not_cover_bundle")
    if result.get("source_document_count") != source_count:
        errors.append("wrong_source_document_count")
    pages = result.get("evidence_pages")
    if not isinstance(pages, list) or not pages or not all(
        isinstance(page, int) and not isinstance(page, bool) for page in pages
    ):
        errors.append("invalid_evidence_pages")
        pages = []
    else:
        result["evidence_pages"] = sorted(set(pages))
    covered = {
        source.doc_number
        for source in sources
        if any(source.merged_start <= page <= source.merged_end for page in pages)
    }
    if covered != set(expected_docs):
        errors.append("evidence_does_not_cover_all_sources")
    facts = result.get("intermediate_facts")
    if not isinstance(facts, list) or {
        row.get("doc_number") for row in facts if isinstance(row, dict)
    } != set(expected_docs):
        errors.append("intermediate_facts_do_not_cover_all_sources")
    result["qa_id"] = "FTQA_" + hashlib.sha256(
        (str(bundle["id"]) + "\0" + question).encode("utf-8")
    ).hexdigest()[:16]
    result["question_type"] = "cross_document_reasoning"
    result["question_category"] = "cross_pdf_hard"
    result["modal_types"] = ["text"]
    result.setdefault("source_qa_ids", [])
    result["generation_rank"] = index
    return (None if errors else result), errors


def process_bundle(
    bundle: dict[str, Any], args: argparse.Namespace, key: str
) -> dict[str, Any]:
    output_path = args.output_dir / "checkpoints" / f"{bundle['id']}.json"
    if output_path.exists() and not args.force:
        existing = load_json(output_path)
        if not args.revalidate_existing:
            return existing
        raw = existing.get("raw_response", {})
        response_text = (
            response_text_claude(raw)
            if existing.get("provider") == "claude"
            else response_text_gemini(raw)
        )
        parsed = parse_json_response(response_text)
        raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []
        valid_items: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        seen_questions: list[str] = []
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                issues.append({"item": index, "errors": ["item_is_not_object"]})
                continue
            item, errors = normalize_item(raw_item, bundle, index)
            question = re.sub(
                r"\s+", " ", str(raw_item.get("question", ""))
            ).casefold()
            if question in seen_questions:
                errors.append("duplicate_question")
                item = None
            if question:
                seen_questions.append(question)
            if errors:
                issues.append(
                    {"item": index, "qa_id": raw_item.get("qa_id"), "errors": errors}
                )
            if item is not None:
                valid_items.append(item)
        existing["valid_items"] = valid_items
        existing["validation_issues"] = issues
        existing["revalidated_at_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        atomic_json(output_path, existing)
        return existing
    started = time.monotonic()
    prompt, metadata = build_prompt(
        bundle, args.items_per_bundle, args.generation_round
    )
    if args.provider == "claude":
        response = post_with_retries(
            f"{CLAUDE_BASE_URL}/v1/messages",
            {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": args.model,
                "max_tokens": args.max_output_tokens,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
            },
            args.timeout,
            args.max_retries,
        )
        raw = response.json()
        response_text = response_text_claude(raw)
    else:
        response = post_with_retries(
            f"{GEMINI_BASE_URL}/v1beta/models/{args.model}:generateContent",
            {"x-goog-api-key": key, "content-type": "application/json"},
            {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": args.max_output_tokens,
                    "responseMimeType": "application/json",
                },
            },
            args.timeout,
            args.max_retries,
        )
        raw = response.json()
        response_text = response_text_gemini(raw)
    parsed = parse_json_response(response_text)
    raw_items = parsed.get("items", []) if isinstance(parsed, dict) else []
    valid_items: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_questions: list[str] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            issues.append({"item": index, "errors": ["item_is_not_object"]})
            continue
        item, errors = normalize_item(raw_item, bundle, index)
        question = re.sub(r"\s+", " ", str(raw_item.get("question", ""))).casefold()
        if question in seen_questions:
            errors.append("duplicate_question")
            item = None
        if question:
            seen_questions.append(question)
        if errors:
            issues.append({"item": index, "qa_id": raw_item.get("qa_id"), "errors": errors})
        if item is not None:
            valid_items.append(item)
    record = {
        "bundle_id": bundle["id"],
        "provider": args.provider,
        "model": args.model,
        "metadata": metadata,
        "valid_items": valid_items,
        "validation_issues": issues,
        "raw_response": raw,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(output_path, record)
    return record


def aggregate(
    bundles: list[dict[str, Any]], records: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    manifest_by_id = {str(bundle["id"]): bundle for bundle in bundles}
    dataset: dict[str, Any] = {}
    issue_rows: list[dict[str, Any]] = []
    for record in records:
        bundle_id = str(record["bundle_id"])
        manifest = manifest_by_id[bundle_id]
        qas = {item["qa_id"]: item for item in record.get("valid_items", [])}
        if qas:
            dataset[bundle_id] = {
                "paper": bundle_id,
                "primary_category": manifest.get("primary_category", ""),
                "secondary_category": manifest.get("secondary_category", ""),
                "QA": qas,
            }
        issue_rows.extend(
            {"bundle_id": bundle_id, **row}
            for row in record.get("validation_issues", [])
        )
    qa_count = sum(len(bundle["QA"]) for bundle in dataset.values())
    output = args.output_dir / "fulltext_cross_pdf_candidates.json"
    atomic_json(output, dataset)
    atomic_json(args.output_dir / "validation_issues.json", issue_rows)
    summary = {
        "status": "complete",
        "manifest_json": str(args.manifest_json),
        "provider": args.provider,
        "model": args.model,
        "requested_bundles": len(bundles),
        "output_bundles": len(dataset),
        "candidate_questions": qa_count,
        "source_document_distribution": dict(
            sorted(Counter(
                item["source_document_count"]
                for bundle in dataset.values() for item in bundle["QA"].values()
            ).items())
        ),
        "validation_issue_count": len(issue_rows),
        "output": str(output),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.items_per_bundle <= 0 or args.api_workers <= 0:
        raise SystemExit("items-per-bundle and api-workers must be positive")
    all_bundles = load_json(args.manifest_json)
    if not isinstance(all_bundles, list):
        raise SystemExit("manifest JSON must be a list")
    selected = all_bundles
    if args.bundle_id:
        requested = set(args.bundle_id)
        selected = [row for row in selected if str(row.get("id")) in requested]
        missing = requested - {str(row.get("id")) for row in selected}
        if missing:
            raise SystemExit(f"unknown bundle ids: {sorted(missing)}")
    if args.sample:
        selected = selected[: args.sample]
    key = load_key(args.api_key_file)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.api_workers) as pool:
        future_map = {pool.submit(process_bundle, row, args, key): row for row in selected}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            row = future_map[future]
            try:
                record = future.result()
            except Exception as exc:
                print(f"[{completed}/{len(selected)}] {row['id']}: ERROR {type(exc).__name__}: {exc}")
                continue
            records.append(record)
            print(
                f"[{completed}/{len(selected)}] {row['id']}: "
                f"{len(record.get('valid_items', []))} valid"
            )
    existing_records = {str(record["bundle_id"]): record for record in records}
    for row in selected:
        path = args.output_dir / "checkpoints" / f"{row['id']}.json"
        if str(row["id"]) not in existing_records and path.is_file():
            existing_records[str(row["id"])] = load_json(path)
    summary = aggregate(selected, list(existing_records.values()), args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
