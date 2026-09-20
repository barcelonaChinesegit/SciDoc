#!/usr/bin/env python3
"""Build a source-grounded QA set for independent review of Gemini repairs.

Only complete Gemini repairs of items rejected by Claude are included.  The
corrected evidence must still map to at least two source papers in the bundle.
The output preserves the original bundle/QA identifiers for full traceability.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("data/qa/4.cross_pdf/semantic_reaudit")
DEFAULT_MANIFEST = Path(
    "data/qa_generation/expansion_20260711/cross_pdf_bundle_manifest.json"
)
DEFAULT_PRIOR_QA = Path(
    "data/qa/qa_cross_pdf_manual_verified_500_20260724.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mapped_doc_numbers(
    evidence_pages: list[int], sources: list[dict[str, Any]]
) -> set[int]:
    docs: set[int] = set()
    for page in evidence_pages:
        if not isinstance(page, int) or isinstance(page, bool):
            return set()
        for doc_number, source in enumerate(sources, start=1):
            if source["merged_start_page"] <= page <= source["merged_end_page"]:
                docs.add(doc_number)
                break
    return docs


def apply_complete_gemini_fix(
    original: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any] | None:
    question = review.get("corrected_question")
    answer = review.get("corrected_answer")
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(answer, str) or not answer.strip():
        return None

    result = deepcopy(original)
    result["question"] = question.strip()
    result["answer"] = answer.strip()
    pages = review.get("corrected_evidence_pages")
    if pages not in (None, []):
        if not isinstance(pages, list) or not all(
            isinstance(page, int) and not isinstance(page, bool)
            for page in pages
        ):
            return None
        result["evidence_pages"] = pages
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prior-qa", type=Path, default=DEFAULT_PRIOR_QA)
    parser.add_argument(
        "--remaining-qa",
        type=Path,
        default=DEFAULT_ROOT / "qa_cross_pdf_remaining_1941.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "qa_gemini_fixes_for_claude_rereview.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_ROOT / "gemini_fix_rereview_input_ledger.jsonl",
    )
    args = parser.parse_args()

    manifest = {row["id"]: row for row in load_json(args.manifest)}
    subsets = [
        ("prior_500", load_json(args.prior_qa), args.root / "api_reviews"),
        (
            "remaining_1941",
            load_json(args.remaining_qa),
            args.root / "remaining_1941_api_reviews",
        ),
    ]

    output: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for subset_name, qa_data, reviews_dir in subsets:
        for bundle_id, bundle in qa_data.items():
            gemini_record = load_json(
                reviews_dir
                / "gemini_gemini-3-flash-preview"
                / f"{bundle_id}.json"
            )
            claude_record = load_json(
                reviews_dir / "claude_claude-sonnet-5" / f"{bundle_id}.json"
            )
            if gemini_record.get("validation_errors"):
                raise ValueError(f"Invalid Gemini review: {bundle_id}")
            if claude_record.get("validation_errors"):
                raise ValueError(f"Invalid Claude review: {bundle_id}")
            gemini_items = {
                row["qa_id"]: row for row in gemini_record["review"]["items"]
            }
            claude_items = {
                row["qa_id"]: row for row in claude_record["review"]["items"]
            }

            for qa_id, original in bundle["QA"].items():
                gemini = gemini_items[qa_id]
                claude = claude_items[qa_id]
                if claude["decision"] in {"KEEP", "FIX"}:
                    exclusions["already_claude_accepted"] += 1
                    continue
                if gemini["decision"] != "FIX":
                    exclusions["gemini_not_fix"] += 1
                    continue
                corrected = apply_complete_gemini_fix(original, gemini)
                if corrected is None:
                    exclusions["incomplete_or_malformed_fix"] += 1
                    continue
                doc_numbers = mapped_doc_numbers(
                    corrected["evidence_pages"], manifest[bundle_id]["sources"]
                )
                if len(doc_numbers) < 2:
                    exclusions["corrected_evidence_not_cross_paper"] += 1
                    continue

                if bundle_id not in output:
                    output[bundle_id] = {
                        key: deepcopy(value)
                        for key, value in bundle.items()
                        if key != "QA"
                    }
                    output[bundle_id]["QA"] = {}
                if qa_id in output[bundle_id]["QA"]:
                    raise ValueError(f"Duplicate corrected item: {bundle_id}/{qa_id}")
                output[bundle_id]["QA"][qa_id] = corrected
                ledger.append(
                    {
                        "subset": subset_name,
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "gemini_confidence": gemini["confidence"],
                        "gemini_reason": gemini["reason"],
                        "original": original,
                        "corrected": corrected,
                        "corrected_evidence_doc_numbers": sorted(doc_numbers),
                    }
                )

    item_count = sum(len(bundle["QA"]) for bundle in output.values())
    if item_count != len(ledger):
        raise AssertionError("Output and ledger item counts differ")
    if item_count == 0:
        raise ValueError("No complete cross-paper fixes were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.ledger.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "bundle_count": len(output),
        "qa_count": item_count,
        "exclusions": dict(exclusions),
        "output": str(args.output),
        "ledger": str(args.ledger),
    }
    summary_path = args.output.with_name(
        args.output.stem + "_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
