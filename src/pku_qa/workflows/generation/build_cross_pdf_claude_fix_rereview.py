#!/usr/bin/env python3
"""Build corrected Claude FIX items whose original criteria were inconsistent."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from pku_qa.workflows.generation.build_cross_pdf_gemini_fix_rereview import mapped_doc_numbers
    from pku_qa.workflows.cleaning.finalize_cross_pdf_semantic_reaudit import (
        all_criteria_true,
        apply_review_fix,
        review_items,
    )
except ModuleNotFoundError:
    from pku_qa.workflows.generation.build_cross_pdf_gemini_fix_rereview import mapped_doc_numbers
    from pku_qa.workflows.cleaning.finalize_cross_pdf_semantic_reaudit import (
        all_criteria_true,
        apply_review_fix,
        review_items,
    )


ROOT = Path("data/qa/4.cross_pdf/semantic_reaudit")
MANIFEST = Path(
    "data/qa_generation/expansion_20260711/cross_pdf_bundle_manifest.json"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = {row["id"]: row for row in load_json(MANIFEST)}
    subsets = [
        (
            "prior_500",
            load_json(
                Path("data/qa/qa_cross_pdf_manual_verified_500_20260724.json")
            ),
            ROOT / "api_reviews",
        ),
        (
            "remaining_1941",
            load_json(ROOT / "qa_cross_pdf_remaining_1941.json"),
            ROOT / "remaining_1941_api_reviews",
        ),
    ]
    output: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    for subset_name, qa_data, reviews_dir in subsets:
        for bundle_id, bundle in qa_data.items():
            claude = review_items(
                reviews_dir / "claude_claude-sonnet-5" / f"{bundle_id}.json"
            )
            for qa_id, original in bundle["QA"].items():
                review = claude[qa_id]
                if review["decision"] != "FIX" or all_criteria_true(review):
                    continue
                corrected, changed = apply_review_fix(original, review)
                pages = corrected.get("evidence_pages", [])
                if not isinstance(pages, list) or not all(
                    isinstance(page, int) and not isinstance(page, bool)
                    for page in pages
                ):
                    continue
                docs = mapped_doc_numbers(
                    pages, manifest[bundle_id]["sources"]
                )
                if len(docs) < 2:
                    continue
                if bundle_id not in output:
                    output[bundle_id] = {
                        key: deepcopy(value)
                        for key, value in bundle.items()
                        if key != "QA"
                    }
                    output[bundle_id]["QA"] = {}
                output[bundle_id]["QA"][qa_id] = corrected
                ledger.append(
                    {
                        "source_subset": subset_name,
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "applied_claude_fix_fields": changed,
                        "first_claude_confidence": review["confidence"],
                        "first_claude_reason": review["reason"],
                        "corrected_evidence_doc_numbers": sorted(docs),
                        "corrected": corrected,
                    }
                )

    output_path = ROOT / "qa_claude_fixes_for_claude_rereview.json"
    ledger_path = ROOT / "claude_fix_rereview_input_ledger.jsonl"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "bundle_count": len(output),
        "qa_count": len(ledger),
        "output": str(output_path),
        "ledger": str(ledger_path),
    }
    (ROOT / "qa_claude_fixes_for_claude_rereview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
