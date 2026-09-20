#!/usr/bin/env python3
"""Translate reviewed printed-page references to physical PDF page numbers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pku_qa.workflows.cleaning.sync_human_reviewed_qa import atomic_write_json


DEFAULT_DATASET = ROOT / "data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json"
DEFAULT_LEDGER = (
    ROOT
    / "data/results/reports/human_review_sync_20260731/evidence_page_mappings.json"
)

# Each mapping was checked against text extracted from the local PDF.  The old
# values mix journal printed-page labels, physical pages, and (for a few arXiv
# papers) section/equation references.  The new values are 1-based physical
# pages in data/pdfs/{paper_id}.pdf.
PAGE_MAPPINGS: dict[tuple[str, str], tuple[list[int], list[int], str]] = {
    ("90", "QA8"): ([5, 26], [5, 7], "semantic_text_match"),
    ("339", "QA6"): ([3, 9], [3], "semantic_text_match"),
    ("386", "QA10"): ([1, 15], [1], "title_exact_match"),
    ("419", "QA8"): ([1, 14], [1, 8], "abstract_and_result_match"),
    ("457", "QA7"): ([628, 629, 630], [7, 8, 9], "printed_footer_match"),
    ("469", "QA9"): ([121], [6], "printed_footer_match"),
    ("469", "QA10"): ([116, 128], [1, 13], "printed_footer_match"),
    ("472", "QA5"): ([183, 189], [1, 7], "printed_footer_match"),
    ("473", "QA6"): ([74, 82], [1, 9], "printed_footer_match"),
    ("477", "QA7"): ([299, 300], [5, 6], "printed_footer_match"),
    ("477", "QA8"): ([301], [7], "printed_footer_match"),
    ("479", "QA5"): ([14], [10], "method_exact_match"),
    ("486", "QA9"): ([226], [8], "printed_footer_match"),
    ("506", "QA8"): ([206, 209], [6, 9], "printed_footer_match"),
    ("510", "QA8"): ([207], [4], "printed_header_match"),
    ("515", "QA6"): ([2282, 2283], [12, 13], "printed_header_match"),
    ("515", "QA10"): ([2285], [15], "printed_header_match"),
    ("550", "QA6"): ([213, 214], [3, 4], "printed_footer_match"),
    ("550", "QA7"): ([216, 217], [6, 7], "printed_footer_match"),
    ("550", "QA8"): ([217, 218], [7, 8], "printed_footer_match"),
    ("671", "QA2"): ([1, 18], [1, 10], "abstract_and_conclusion_match"),
    ("671", "QA4"): ([1, 26], [1, 10], "abstract_and_conclusion_match"),
}

UNRESOLVED = {
    ("327", "QA6"): {
        "evidence_pages": [24, 28],
        "reason": (
            "The local 327.pdf has 23 pages and is about black-hole evaporation, "
            "while the QA concerns a Moran-model ring topology; no defensible "
            "physical-page translation exists."
        ),
        "status": "excluded_from_active_enriched_and_challenge_datasets",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def apply_mappings(dataset: dict) -> list[dict]:
    rows = []
    for (paper_id, qa_id), (expected, physical, method) in PAGE_MAPPINGS.items():
        qa = dataset[paper_id]["QA"][qa_id]
        before = qa.get("evidence_pages")
        if before not in (expected, physical):
            raise ValueError(
                f"unexpected evidence pages for {paper_id}:{qa_id}: {before}"
            )
        qa["evidence_pages"] = list(physical)
        rows.append(
            {
                "paper_id": paper_id,
                "qa_id": qa_id,
                "question": qa.get("question"),
                "original_reviewed_pages": expected,
                "physical_pdf_pages": physical,
                "method": method,
                "changed": before != physical,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    rows = apply_mappings(dataset)
    ledger = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve().relative_to(ROOT)),
        "page_basis": "1-based physical page in local PDF",
        "mapped_items": len(rows),
        "changed_items": sum(row["changed"] for row in rows),
        "mappings": rows,
        "unresolved": [
            {"paper_id": key[0], "qa_id": key[1], **value}
            for key, value in UNRESOLVED.items()
        ],
    }
    if args.apply:
        atomic_write_json(args.dataset, dataset)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.ledger, ledger)
    print(json.dumps(ledger, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
