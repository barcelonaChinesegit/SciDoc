#!/usr/bin/env python3
"""Maintain categorized single-PDF views of the canonical final release."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = ROOT / "data/qa/7.final_2200"
ORDINARY_SOURCE = FINAL_DIR / "ordinary_qa.json"
UNANSWERABLE_SOURCE = FINAL_DIR / "unanswerable_qa.json"
ORDINARY_VIEW = (
    ROOT / "data/qa/1.base/rel__single_pdf__ordinary__batch01__n1000.json"
)
UNANSWERABLE_VIEW = (
    ROOT
    / "data/qa/2.unanswerable/rel__single_pdf__unanswerable__batch01__n200.json"
)
DEFAULT_RUNTIME_DIR = ROOT / "data/results/evaluations/single_pdf_1200/inputs"
COMBINED_RUNTIME_INPUT = (
    DEFAULT_RUNTIME_DIR
    / "work__single_pdf__ordinary_unanswerable__batch01__n1200.json"
)

RELEASE_VIEWS = {
    "ordinary_qa.json": ORDINARY_VIEW,
    "unanswerable_qa.json": UNANSWERABLE_VIEW,
}


def read_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object-keyed QA dataset: {path}")
    return payload


def qa_count(dataset: dict[str, Any]) -> int:
    return sum(
        len(paper.get("QA", {}))
        for paper in dataset.values()
        if isinstance(paper, dict) and isinstance(paper.get("QA"), dict)
    )


def atomic_write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encoded(dataset: dict[str, Any]) -> bytes:
    return (json.dumps(dataset, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def merge_single_pdf_release(
    ordinary: dict[str, Any], unanswerable: dict[str, Any]
) -> dict[str, Any]:
    """Merge the two canonical single-PDF components without changing QA IDs."""
    merged = copy.deepcopy(ordinary)
    for paper_id, paper in unanswerable.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Malformed unanswerable paper record: {paper_id}")
        if paper_id not in merged:
            merged[paper_id] = copy.deepcopy(paper)
            continue
        current = merged[paper_id]
        if not isinstance(current, dict) or not isinstance(current.get("QA"), dict):
            raise ValueError(f"Malformed ordinary paper record: {paper_id}")
        ordinary_metadata = {key: value for key, value in current.items() if key != "QA"}
        unanswerable_metadata = {key: value for key, value in paper.items() if key != "QA"}
        if ordinary_metadata != unanswerable_metadata:
            raise ValueError(f"Paper metadata differs across final components: {paper_id}")
        duplicate_ids = set(current["QA"]).intersection(paper["QA"])
        if duplicate_ids:
            raise ValueError(
                f"Duplicate QA IDs across final components for {paper_id}: "
                f"{sorted(duplicate_ids)[:5]}"
            )
        current["QA"].update(copy.deepcopy(paper["QA"]))
    if qa_count(merged) != 1200:
        raise ValueError(
            f"Combined single-PDF release must contain 1,200 QA, found {qa_count(merged)}"
        )
    return merged


def build_evidence_ablation_dataset(
    ordinary: dict[str, Any],
) -> dict[str, Any]:
    """Build one leave-one-evidence-page variant for every multi-page gold QA."""
    result: dict[str, Any] = {}
    for paper_id, paper in ordinary.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Malformed ordinary paper record: {paper_id}")
        variants: dict[str, Any] = {}
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(f"Malformed ordinary QA record: {paper_id}/{qa_id}")
            raw_pages = qa.get("evidence_pages")
            if not isinstance(raw_pages, list):
                raise ValueError(f"Invalid evidence_pages: {paper_id}/{qa_id}")
            pages = sorted({int(page) for page in raw_pages})
            if len(pages) < 2:
                continue
            for removed_page in pages:
                variant = copy.deepcopy(qa)
                variant_id = f"{qa_id}__drop_p{removed_page}"
                variant["input_pages"] = [
                    page for page in pages if page != removed_page
                ]
                variant["ablation"] = {
                    "source_qa_id": qa_id,
                    "removed_physical_pdf_page": removed_page,
                    "original_evidence_pages": pages,
                    "decision_rule": (
                        "The removed page is not necessary if the answer remains "
                        "uniquely recoverable from the remaining oracle pages."
                    ),
                }
                variant["review_status"] = "pending_ablation_eval"
                variants[variant_id] = variant
        if variants:
            result[paper_id] = {
                **{key: copy.deepcopy(value) for key, value in paper.items() if key != "QA"},
                "QA": variants,
            }
    return result


def sync_release_views(*, check_only: bool = False) -> dict[str, Any]:
    expected = {
        ORDINARY_VIEW: ORDINARY_SOURCE.read_bytes(),
        UNANSWERABLE_VIEW: UNANSWERABLE_SOURCE.read_bytes(),
    }
    stale = [
        path
        for path, raw in expected.items()
        if not path.is_file() or path.read_bytes() != raw
    ]
    if check_only and stale:
        paths = ", ".join(str(path.relative_to(ROOT)) for path in stale)
        raise ValueError(f"Single-PDF release views are stale: {paths}")
    if not check_only:
        for path, raw in expected.items():
            if path in stale:
                atomic_write_bytes(path, raw)
    return {
        "status": "current",
        "mode": "check_only" if check_only else "sync",
        "views": {
            str(path.relative_to(ROOT)): qa_count(read_dataset(source))
            for source, path in (
                (ORDINARY_SOURCE, ORDINARY_VIEW),
                (UNANSWERABLE_SOURCE, UNANSWERABLE_VIEW),
            )
        },
    }


def write_runtime_inputs(
    *, combined_path: Path = COMBINED_RUNTIME_INPUT,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
) -> tuple[Path, Path]:
    sync_release_views(check_only=True)
    ordinary = read_dataset(ORDINARY_VIEW)
    unanswerable = read_dataset(UNANSWERABLE_VIEW)
    combined = merge_single_pdf_release(ordinary, unanswerable)
    ablation = build_evidence_ablation_dataset(ordinary)
    ablation_path = (
        runtime_dir
        / f"work__single_pdf__evidence_ablation__batch01__n{qa_count(ablation)}.json"
    )
    atomic_write_bytes(combined_path, encoded(combined))
    atomic_write_bytes(ablation_path, encoded(ablation))
    return combined_path, ablation_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = sync_release_views(check_only=args.check_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
