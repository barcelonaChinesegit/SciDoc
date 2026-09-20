#!/usr/bin/env python3
"""Prepare matched old/new pilot datasets for local Qwen evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-cross", type=Path, required=True)
    parser.add_argument("--new-reasoning", type=Path, required=True)
    parser.add_argument(
        "--old-cross",
        type=Path,
        default=ROOT / "data/qa/4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json",
    )
    parser.add_argument(
        "--old-reasoning",
        type=Path,
        default=ROOT / "data/qa/3.reasoning/work__reasoning__historical_clean__batch00__n100.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Dataset must be an object: {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def flatten(dataset: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    return [
        (str(paper_id), str(qa_id), qa, paper)
        for paper_id, paper in dataset.items()
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    ]


def evidence_span(qa: dict[str, Any]) -> int:
    pages = [int(page) for page in qa.get("evidence_pages", [])]
    return max(pages) - min(pages) if pages else 0


def source_doc_count(qa: dict[str, Any]) -> int:
    for field in ("source_doc_numbers", "evidence_source_docs"):
        values = qa.get(field)
        if isinstance(values, list):
            return len(set(values))
    items = qa.get("evidence_items")
    if isinstance(items, list):
        docs = {
            item.get("source_doc_number")
            for item in items
            if isinstance(item, dict) and item.get("source_doc_number") is not None
        }
        if docs:
            return len(docs)
    return 2


def select_old(
    dataset: dict[str, Any], count: int, kind: str
) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    rows = flatten(dataset)
    if kind == "cross":
        rows.sort(
            key=lambda row: (
                source_doc_count(row[2]),
                evidence_span(row[2]),
                len(row[2].get("source_qa_ids", [])),
                row[0],
                row[1],
            ),
            reverse=True,
        )
    else:
        rows.sort(
            key=lambda row: (
                evidence_span(row[2]),
                len(row[2].get("source_qa_ids", [])),
                row[0],
                row[1],
            ),
            reverse=True,
        )
    if len(rows) < count:
        raise ValueError(f"Old {kind} dataset has only {len(rows)} items, need {count}")
    return rows[:count]


def cohort_dataset(
    new: dict[str, Any], old: dict[str, Any], kind: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    new_rows = flatten(new)
    old_rows = select_old(old, len(new_rows), kind)
    output: dict[str, Any] = {}
    for cohort, rows in (("old_baseline", old_rows), ("new_hard_expansion", new_rows)):
        for paper_id, qa_id, qa, paper in rows:
            bucket = output.setdefault(
                paper_id,
                {
                    "paper": str(paper.get("paper", paper_id)),
                    "primary_category": paper.get("primary_category", ""),
                    "secondary_category": paper.get("secondary_category", ""),
                    "QA": {},
                },
            )
            digest = hashlib.sha256(
                (cohort + "\0" + paper_id + "\0" + qa_id).encode("utf-8")
            ).hexdigest()[:16]
            item = dict(qa)
            item["benchmark_cohort"] = cohort
            item["benchmark_source_id"] = f"{paper_id}/{qa_id}"
            bucket["QA"][f"PILOT_{digest}"] = item
    summary = {
        "kind": kind,
        "selection_policy": (
            "same count; old baseline deliberately selects the largest evidence "
            "spans and, for Cross-PDF, the largest source-document counts"
        ),
        "old_baseline": len(old_rows),
        "new_hard_expansion": len(new_rows),
        "total": len(old_rows) + len(new_rows),
    }
    return output, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for kind, new_path, old_path in (
        ("cross", args.new_cross, args.old_cross),
        ("reasoning", args.new_reasoning, args.old_reasoning),
    ):
        dataset, summary = cohort_dataset(
            read_json(new_path), read_json(old_path), kind
        )
        atomic_json(args.output_dir / f"{kind}_old_new_pilot.json", dataset)
        summaries[kind] = summary
    atomic_json(args.output_dir / "preparation_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
