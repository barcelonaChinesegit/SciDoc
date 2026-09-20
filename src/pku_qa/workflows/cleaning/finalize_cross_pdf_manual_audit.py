#!/usr/bin/env python3
"""Build the 50-PDF / 500-QA manually reviewed cross-paper release."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pku_qa.pdf_assets import output_pdf_path, resolve_pdf_path


QA_PATH = Path("data/qa/qa_expansion_cross_pdf_cleaned_20260711.json")
MANIFEST_PATH = Path(
    "data/qa_generation/expansion_20260711/cross_pdf_bundle_manifest.json"
)
DECISIONS_PATH = Path(
    "data/qa/6.review/cross_pdf/manual_review_decisions.json"
)
REVIEW_PACKET_PATH = Path(
    "data/qa/6.review/cross_pdf/review_packet.jsonl"
)
SOURCE_PDF_DIR = Path("data/pdfs")
OUTPUT_QA_PATH = Path(
    "data/qa/qa_cross_pdf_manual_verified_500_20260724.json"
)
OUTPUT_DIR = Path("data/qa/6.review/cross_pdf")
OUTPUT_PDF_DIR = Path(
    "data/pdfs"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    qa_data = load_json(QA_PATH)
    manifest_rows = load_json(MANIFEST_PATH)
    manifest = {row["id"]: row for row in manifest_rows}
    decisions = load_json(DECISIONS_PATH)
    review_packet = {}
    with REVIEW_PACKET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            review_packet[(record["bundle_id"], record["qa_id"])] = record

    bundle_decisions = decisions["bundles"]
    if len(bundle_decisions) != 55:
        raise ValueError(f"Expected 55 reviewed bundles, got {len(bundle_decisions)}")

    selected_bundle_ids: list[str] = []
    excluded_bundle_ids: list[str] = []
    output_qa: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []

    for bundle_id, bundle_decision in bundle_decisions.items():
        source_qa = qa_data[bundle_id]["QA"]
        rejected = bundle_decision["reject"]
        unknown_rejections = sorted(set(rejected) - set(source_qa))
        if unknown_rejections:
            raise ValueError(
                f"{bundle_id} has unknown rejected QA IDs: {unknown_rejections}"
            )
        qualified_ids = [qa_id for qa_id in source_qa if qa_id not in rejected]
        selected = len(qualified_ids) == 10
        if selected:
            selected_bundle_ids.append(bundle_id)
            output_qa[bundle_id] = {
                key: value
                for key, value in qa_data[bundle_id].items()
                if key != "QA"
            }
            output_qa[bundle_id]["QA"] = {
                qa_id: source_qa[qa_id] for qa_id in qualified_ids
            }
        else:
            excluded_bundle_ids.append(bundle_id)

        for qa_id in source_qa:
            packet = review_packet[(bundle_id, qa_id)]
            if qa_id in rejected:
                status = "rejected"
                reason = rejected[qa_id]
            elif selected:
                status = "kept"
                reason = (
                    "Requires facts from multiple source papers; answer is "
                    "consistent with the cited cross-paper evidence."
                )
            else:
                status = "qualified_not_selected_bundle"
                reason = (
                    "No item-level rejection recorded, but its PDF had fewer "
                    "than 10 qualified items and was excluded as a whole."
                )
            ledger.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "status": status,
                    "reason": reason,
                    "question": packet["question"],
                    "answer": packet["answer"],
                    "evidence_pages": packet["evidence_pages"],
                    "evidence_doc_numbers": packet["evidence_doc_numbers"],
                    "structural_flags": packet["structural_flags"],
                }
            )

    kept_count = sum(len(record["QA"]) for record in output_qa.values())
    if len(selected_bundle_ids) != 50 or kept_count != 500:
        raise ValueError(
            "Release must contain exactly 50 PDFs and 500 QA; got "
            f"{len(selected_bundle_ids)} PDFs and {kept_count} QA"
        )

    structural_errors: list[dict[str, Any]] = []
    seen_questions: Counter[str] = Counter()
    for bundle_id, record in output_qa.items():
        bundle_manifest = manifest[bundle_id]
        pdf_path = resolve_pdf_path(bundle_id, [SOURCE_PDF_DIR], required=False)
        if pdf_path is None:
            structural_errors.append(
                {"bundle_id": bundle_id, "error": "missing_source_pdf"}
            )
        if len(bundle_manifest["sources"]) < 2:
            structural_errors.append(
                {"bundle_id": bundle_id, "error": "fewer_than_two_sources"}
            )
        for qa_id, qa in record["QA"].items():
            packet = review_packet[(bundle_id, qa_id)]
            seen_questions[qa["question"].strip().casefold()] += 1
            if len(packet["evidence_doc_numbers"]) < 2:
                structural_errors.append(
                    {
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "error": "evidence_not_cross_paper",
                    }
                )
            hard_flags = [
                flag
                for flag in packet["structural_flags"]
                if flag
                in {
                    "evidence_page_not_mapped_to_source",
                    "evidence_page_out_of_bounds",
                    "evidence_spans_fewer_than_two_source_papers",
                }
            ]
            if hard_flags:
                structural_errors.append(
                    {
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "error": "hard_structural_flags",
                        "flags": hard_flags,
                    }
                )
            if not qa.get("question") or not qa.get("answer"):
                structural_errors.append(
                    {
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "error": "blank_question_or_answer",
                    }
                )

    duplicate_questions = [
        question for question, count in seen_questions.items() if count > 1
    ]
    if duplicate_questions:
        structural_errors.append(
            {
                "error": "duplicate_questions",
                "count": len(duplicate_questions),
                "questions": duplicate_questions,
            }
        )
    if structural_errors:
        raise ValueError(
            "Structural release validation failed:\n"
            + json.dumps(structural_errors, ensure_ascii=False, indent=2)
        )

    OUTPUT_QA_PATH.write_text(
        json.dumps(output_qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_manifest = [manifest[bundle_id] for bundle_id in selected_bundle_ids]
    (OUTPUT_DIR / "selected_bundle_manifest.json").write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "review_ledger.jsonl").open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    for bundle_id in selected_bundle_ids:
        source = resolve_pdf_path(bundle_id, [SOURCE_PDF_DIR])
        target = output_pdf_path(OUTPUT_PDF_DIR, bundle_id)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    status_counts = Counter(row["status"] for row in ledger)
    summary = {
        "reviewed_pdf_count": len(bundle_decisions),
        "reviewed_qa_count": len(ledger),
        "selected_pdf_count": len(selected_bundle_ids),
        "selected_qa_count": kept_count,
        "selected_qa_per_pdf": 10,
        "excluded_pdf_count": len(excluded_bundle_ids),
        "selected_pdf_ids": selected_bundle_ids,
        "excluded_pdf_ids": excluded_bundle_ids,
        "ledger_status_counts": dict(status_counts),
        "validation": {
            "all_kept_evidence_spans_multiple_source_papers": True,
            "all_kept_evidence_pages_in_bounds_and_mapped": True,
            "all_kept_questions_and_answers_nonempty": True,
            "duplicate_kept_questions": 0,
            "selected_pdf_files_copied": len(selected_bundle_ids),
        },
        "outputs": {
            "qa_json": str(OUTPUT_QA_PATH),
            "pdf_dir": str(OUTPUT_PDF_DIR),
            "manifest": str(OUTPUT_DIR / "selected_bundle_manifest.json"),
            "review_ledger": str(OUTPUT_DIR / "review_ledger.jsonl"),
            "manual_decisions": str(DECISIONS_PATH),
        },
    }
    (OUTPUT_DIR / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
