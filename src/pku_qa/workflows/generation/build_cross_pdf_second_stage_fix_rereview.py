#!/usr/bin/env python3
"""Apply inconsistent Claude FIX outputs and build one final rereview set."""

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
    inputs = [
        (
            "gemini_repair_first_rereview",
            load_json(ROOT / "qa_gemini_fixes_for_claude_rereview.json"),
            ROOT
            / "gemini_fix_claude_rereviews"
            / "claude_claude-sonnet-5",
        ),
        (
            "claude_fix_first_rereview",
            load_json(ROOT / "qa_claude_fixes_for_claude_rereview.json"),
            ROOT
            / "claude_fix_claude_rereviews"
            / "claude_claude-sonnet-5",
        ),
    ]
    output: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    for stage_name, qa_data, review_dir in inputs:
        for bundle_id, bundle in qa_data.items():
            claude = review_items(review_dir / f"{bundle_id}.json")
            for qa_id, item in bundle["QA"].items():
                review = claude[qa_id]
                if review["decision"] != "FIX" or all_criteria_true(review):
                    continue
                corrected, changed = apply_review_fix(item, review)
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
                if qa_id in output[bundle_id]["QA"]:
                    raise ValueError(f"Duplicate second-stage item: {bundle_id}/{qa_id}")
                output[bundle_id]["QA"][qa_id] = corrected
                ledger.append(
                    {
                        "source_stage": stage_name,
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "applied_claude_fix_fields": changed,
                        "claude_confidence": review["confidence"],
                        "claude_reason": review["reason"],
                        "corrected_evidence_doc_numbers": sorted(docs),
                        "corrected": corrected,
                    }
                )

    output_path = ROOT / "qa_second_stage_fixes_for_claude_rereview.json"
    ledger_path = ROOT / "second_stage_fix_rereview_input_ledger.jsonl"
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
    (ROOT / "qa_second_stage_fixes_for_claude_rereview_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
