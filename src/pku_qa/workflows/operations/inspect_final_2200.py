"""Read-only release inspection; the default command needs only Python's stdlib."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from pku_qa.workflows.selection.sync_final_2200_manifest import (
    FINAL_DIR, MANIFEST_PATH, ROOT, build_manifest, read_json, sha256,
)


def inspect_release(*, check_pdfs: bool = False) -> dict:
    manifest = build_manifest()
    if read_json(MANIFEST_PATH) != manifest:
        raise ValueError("Final-2200 manifest does not match the four release files")
    modalities: Counter = Counter()
    domains: Counter = Counter()
    document_counts: Counter = Counter()
    papers: dict[str, list] = {}
    single_ids: set[str] = set()
    cross_ids: set[str] = set()
    for component in manifest["components"]:
        data = read_json(FINAL_DIR / component["dataset_id"])
        is_cross = component["dataset_id"] == "cross_pdf_qa.json"
        (cross_ids if is_cross else single_ids).update(data)
        for paper_id, paper in data.items():
            qas = list(paper["QA"].values())
            domains[paper["primary_category"]] += len(qas)
            papers.setdefault(paper_id, []).extend(qas)
            modalities.update("+".join(qa["modal_types"]) for qa in qas)
            if is_cross:
                for qa in qas:
                    count = qa.get("source_document_count", len(qa.get("source_paper_ids", [])))
                    if count not in (2, 3):
                        raise ValueError(f"{paper_id}: missing two/three-document count")
                    document_counts[str(count)] += 1
    result = {
        "status": "valid",
        "qa_count": manifest["qa_count"],
        "components": {c["dataset_id"]: c["qa_count"] for c in manifest["components"]},
        "evaluation_pdfs": len(papers),
        "single_paper_pdfs": len(single_ids),
        "merged_cross_pdfs": len(cross_ids),
        "modality_combinations": dict(sorted(modalities.items())),
        "primary_category_qa_counts": dict(sorted(domains.items())),
        "cross_pdf_source_document_counts": dict(sorted(document_counts.items())),
        "pdf_check": "not_requested",
    }
    if check_pdfs:
        # PDF parsing is deliberately optional for a fresh clone without assets.
        from pypdf import PdfReader

        assets = read_json(ROOT / "data/pdf_assets_manifest.json")
        by_id = {asset["pdf_id"]: asset for asset in assets["assets"]}
        for paper_id, qas in papers.items():
            asset = by_id[paper_id]
            path = ROOT / "data/pdfs" / asset["filename"]
            if sha256(path) != asset["sha256"]:
                raise ValueError(f"PDF hash mismatch: {path}")
            with path.open("rb") as handle:
                page_count = len(PdfReader(handle).pages)
            if not page_count or any(
                page > page_count for qa in qas for page in qa["evidence_pages"]
            ):
                raise ValueError(f"Invalid evidence page or empty PDF: {path}")
        result["pdf_check"] = "hashes_readability_and_evidence_bounds_valid"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pdfs", action="store_true", help="Also verify final PDF hashes, readability and evidence bounds (requires pypdf).")
    args = parser.parse_args()
    print(json.dumps(inspect_release(check_pdfs=args.check_pdfs), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
