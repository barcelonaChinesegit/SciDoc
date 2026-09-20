#!/usr/bin/env python3
"""Select an exact dual-KEEP hard Cross-PDF release from API reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
CRITERIA = (
    "cross_document_required",
    "question_coherent",
    "question_specific",
    "answer_correct",
    "answer_complete",
    "evidence_sufficient",
    "no_unsupported_claim",
)
PRIORITY_TYPES = (
    "adjacent_module_dependency",
    "metric_reasoning",
    "component_hierarchy",
    "method_transfer",
    "compatibility_judgment",
    "conflict_resolution",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--claude-dir", type=Path, required=True)
    parser.add_argument("--gemini-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-three-doc-ratio", type=float, default=0.6)
    parser.add_argument("--max-per-bundle", type=int, default=4)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def content_text(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", normalized(value)).strip()


def load_review(directory: Path, bundle_id: str) -> dict[str, Any] | None:
    path = directory / f"{bundle_id}.json"
    if not path.is_file():
        return None
    record = read_json(path)
    if record.get("validation_errors"):
        return None
    return record


def review_index(record: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not record:
        return {}
    return {
        str(item.get("qa_id")): item
        for item in record.get("review", {}).get("items", [])
        if isinstance(item, dict)
    }


def correction_is_noop(item: dict[str, Any], qa: dict[str, Any]) -> bool:
    seen = False
    for correction, field in (
        ("corrected_question", "question"),
        ("corrected_answer", "answer"),
        ("corrected_evidence_pages", "evidence_pages"),
    ):
        value = item.get(correction)
        if value in (None, "", []):
            continue
        seen = True
        if field == "evidence_pages":
            if not isinstance(value, list) or sorted(set(value)) != sorted(
                set(qa.get(field, []))
            ):
                return False
        elif normalized(value) != normalized(qa.get(field)):
            return False
    return seen


def strict_keep(
    item: dict[str, Any] | None, qa: dict[str, Any], min_confidence: float
) -> bool:
    if not isinstance(item, dict) or item.get("decision") not in {"KEEP", "FIX"}:
        return False
    if item.get("decision") == "FIX" and not correction_is_noop(item, qa):
        return False
    confidence = item.get("confidence")
    criteria = item.get("criteria")
    return (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and confidence >= min_confidence
        and isinstance(criteria, dict)
        and all(criteria.get(name) is True for name in CRITERIA)
    )


def candidate_static_errors(qa: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    question = content_text(qa.get("question"))
    answer = content_text(qa.get("answer"))
    if not question or not answer:
        errors.append("empty_question_or_answer")
    if len(answer) >= 5 and answer in question:
        errors.append("answer_leakage")
    docs = qa.get("source_doc_numbers")
    if not isinstance(docs, list) or len(set(docs)) not in {2, 3}:
        errors.append("invalid_source_document_count")
    pages = qa.get("evidence_pages")
    if not isinstance(pages, list) or len(set(pages)) < 2:
        errors.append("insufficient_evidence_pages")
    if qa.get("reasoning_type") not in PRIORITY_TYPES:
        errors.append("invalid_reasoning_type")
    if not isinstance(qa.get("intermediate_facts"), list):
        errors.append("missing_intermediate_facts")
    return errors


def collect(
    candidates: dict[str, Any],
    claude_dir: Path,
    gemini_dir: Path,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen_questions: list[str] = []
    for bundle_id, bundle in candidates.items():
        claude_record = load_review(claude_dir, bundle_id)
        gemini_record = load_review(gemini_dir, bundle_id)
        indexes = {
            "claude": review_index(claude_record),
            "gemini": review_index(gemini_record),
        }
        for qa_id, qa in bundle.get("QA", {}).items():
            provider_items = {
                provider: index.get(str(qa_id)) for provider, index in indexes.items()
            }
            static_errors = candidate_static_errors(qa)
            duplicate = any(
                SequenceMatcher(None, normalized(qa.get("question")), prior).ratio() >= 0.92
                for prior in seen_questions
            )
            if duplicate:
                static_errors.append("near_duplicate_question")
            accepted = not static_errors and all(
                strict_keep(item, qa, min_confidence)
                for item in provider_items.values()
            )
            row = {
                "bundle_id": bundle_id,
                "qa_id": qa_id,
                "accepted_before_quota": accepted,
                "static_errors": static_errors,
                "decisions": {
                    provider: (item or {}).get("decision")
                    for provider, item in provider_items.items()
                },
                "confidences": {
                    provider: (item or {}).get("confidence")
                    for provider, item in provider_items.items()
                },
                "reasons": {
                    provider: (item or {}).get("reason")
                    for provider, item in provider_items.items()
                },
                "selected": False,
            }
            ledger.append(row)
            if not accepted:
                continue
            seen_questions.append(normalized(qa.get("question")))
            item = dict(qa)
            item["_bundle_id"] = bundle_id
            item["_source_qa_id"] = qa_id
            item["dual_model_validation"] = {
                provider: {
                    "decision": provider_items[provider]["decision"],
                    "confidence": provider_items[provider]["confidence"],
                    "reason": provider_items[provider]["reason"],
                    "model": (
                        claude_record if provider == "claude" else gemini_record
                    ).get("model"),
                }
                for provider in ("claude", "gemini")
            }
            support = []
            for provider, review_item in provider_items.items():
                for evidence in review_item.get("support", []):
                    if isinstance(evidence, dict):
                        support.append({"reviewer": provider, **evidence})
            item["review_evidence_ledger"] = support
            eligible.append(item)
    return eligible, ledger


def select_items(
    eligible: list[dict[str, Any]],
    target: int,
    three_ratio: float,
    max_per_bundle: int,
) -> list[dict[str, Any]]:
    if not 0 <= three_ratio <= 1:
        raise ValueError("three-document ratio must be between 0 and 1")
    type_counts: Counter[str] = Counter()
    bundle_counts: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    wanted_three = int(round(target * three_ratio))
    remaining = list(eligible)
    while remaining and len(chosen) < target:
        need_three = sum(item["source_document_count"] == 3 for item in chosen) < wanted_three
        viable = [
            item
            for item in remaining
            if bundle_counts[item["_bundle_id"]] < max_per_bundle
            and (not need_three or item["source_document_count"] == 3)
        ]
        if not viable:
            viable = [
                item
                for item in remaining
                if bundle_counts[item["_bundle_id"]] < max_per_bundle
            ]
        if not viable:
            break
        viable.sort(
            key=lambda item: (
                type_counts[item["reasoning_type"]],
                bundle_counts[item["_bundle_id"]],
                -item["source_document_count"],
                item["_bundle_id"],
            )
        )
        item = viable[0]
        chosen.append(item)
        remaining.remove(item)
        type_counts[item["reasoning_type"]] += 1
        bundle_counts[item["_bundle_id"]] += 1
    return chosen


def build_release(candidates: dict[str, Any], chosen: list[dict[str, Any]]) -> dict[str, Any]:
    release: dict[str, Any] = {}
    for index, item in enumerate(chosen, start=1):
        bundle_id = item.pop("_bundle_id")
        source_qa_id = item.pop("_source_qa_id")
        bucket = release.setdefault(
            bundle_id,
            {
                "paper": bundle_id,
                "primary_category": candidates[bundle_id].get("primary_category", ""),
                "secondary_category": candidates[bundle_id].get("secondary_category", ""),
                "QA": {},
            },
        )
        digest = hashlib.sha256(
            (bundle_id + "\0" + normalized(item["question"])).encode("utf-8")
        ).hexdigest()[:16]
        item["candidate_qa_id"] = source_qa_id
        item["release_rank"] = index
        bucket["QA"][f"CQA_{digest}"] = item
    return release


def main() -> None:
    args = parse_args()
    candidates = read_json(args.candidates)
    eligible, ledger = collect(
        candidates, args.claude_dir, args.gemini_dir, args.min_confidence
    )
    chosen = select_items(
        eligible, args.target, args.min_three_doc_ratio, args.max_per_bundle
    )
    if len(chosen) != args.target:
        raise SystemExit(
            f"insufficient dual-KEEP candidates: selected={len(chosen)}/{args.target}, "
            f"eligible={len(eligible)}"
        )
    selected_keys = {
        (item["_bundle_id"], item["_source_qa_id"]) for item in chosen
    }
    for row in ledger:
        row["selected"] = (row["bundle_id"], row["qa_id"]) in selected_keys
    release = build_release(candidates, chosen)
    items = [qa for bundle in release.values() for qa in bundle["QA"].values()]
    three_count = sum(qa["source_document_count"] == 3 for qa in items)
    summary = {
        "status": "complete",
        "target": args.target,
        "released_questions": len(items),
        "released_bundles": len(release),
        "dual_keep_eligible": len(eligible),
        "three_document_questions": three_count,
        "three_document_ratio": three_count / len(items),
        "reasoning_type_distribution": dict(
            sorted(Counter(qa["reasoning_type"] for qa in items).items())
        ),
        "policy": (
            "Claude/Gemini KEEP or criteria-true no-op FIX + every review "
            "criterion true + "
            f"both confidence >= {args.min_confidence} + static hard gates"
        ),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(args.output, release)
    atomic_json(
        args.ledger_output or args.output.with_name(args.output.stem + "_ledger.json"),
        ledger,
    )
    atomic_json(
        args.summary_output or args.output.with_name(args.output.stem + "_summary.json"),
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
