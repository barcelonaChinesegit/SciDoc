#!/usr/bin/env python3
"""Validate the canonical four-file final-2200 release and sync its manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pku_qa.workflows.selection.final_2200_contract import validate_final_dataset
from pku_qa.workflows.selection.normalize_final_2200_release import atomic_json
from pku_qa.workflows.selection.single_pdf_release_views import sync_release_views


ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = ROOT / "data/qa/7.final_2200"
MANIFEST_PATH = FINAL_DIR / "rel__collection__final_2200__manifest.json"
FINAL_COMPONENTS = (
    ("ordinary_1000", "ordinary_qa.json", 1000),
    ("unanswerable_200", "unanswerable_qa.json", 200),
    ("reasoning_200", "reasoning_qa.json", 200),
    ("cross_pdf_800", "cross_pdf_qa.json", 800),
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qa_rows(dataset: dict[str, Any]):
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            continue
        for qa_id, qa in paper["QA"].items():
            yield str(paper_id), str(qa_id), qa


def build_manifest(final_dir: Path = FINAL_DIR) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    global_ids: list[str] = []
    legacy_keys: set[tuple[str, str]] = set()
    for component_id, filename, expected_count in FINAL_COMPONENTS:
        path = final_dir / filename
        dataset = read_json(path)
        validate_final_dataset(dataset, label=filename)
        rows = list(qa_rows(dataset))
        if len(rows) != expected_count:
            raise ValueError(
                f"{filename}: expected {expected_count} QA, found {len(rows)}"
            )
        for paper_id, qa_id, qa in rows:
            global_ids.append(qa_id)
            identity = qa.get("annotation_provenance", {}).get(
                "final_2200_identity"
            )
            if not isinstance(identity, dict):
                raise ValueError(f"{filename}:{paper_id}/{qa_id}: missing final identity")
            if identity.get("global_qa_id") != qa_id:
                raise ValueError(
                    f"{filename}:{paper_id}/{qa_id}: global identity mismatch"
                )
            legacy_key = (
                str(identity.get("legacy_paper_id", "")),
                str(identity.get("legacy_qa_id", "")),
            )
            if not all(legacy_key) or legacy_key in legacy_keys:
                raise ValueError(
                    f"{filename}:{paper_id}/{qa_id}: duplicate or empty legacy identity"
                )
            legacy_keys.add(legacy_key)
        components.append(
            {
                "id": component_id,
                "dataset_id": filename,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "qa_count": len(rows),
                "paper_count": len(dataset),
                "answerable": sum(
                    qa.get("answer") != "Unanswerable" for _, _, qa in rows
                ),
                "unanswerable": sum(
                    qa.get("answer") == "Unanswerable" for _, _, qa in rows
                ),
                "qa_format": "short_answer_only",
            }
        )
    expected_ids = [f"QA{index:04d}" for index in range(1, 2201)]
    if global_ids != expected_ids:
        raise ValueError("Final 2,200 QA IDs must be the ordered QA0001..QA2200 set")
    return {
        "schema_version": 2,
        "collection_id": "final_2200",
        "purpose": "canonical_publication_and_manual_review_release",
        "qa_format": "short_answer_only",
        "qa_identity": "collection_wide_QA0001_to_QA2200",
        "qa_count": 2200,
        "component_count": 4,
        "components": components,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    expected = build_manifest(args.final_dir)
    sync_release_views(check_only=args.check_only)
    if args.check_only:
        if read_json(args.manifest) != expected:
            raise ValueError(f"Final 2,200 manifest is stale: {args.manifest}")
    else:
        atomic_json(args.manifest, expected)
    print(
        json.dumps(
            {
                "status": "current",
                "mode": "check_only" if args.check_only else "sync",
                "manifest": str(args.manifest),
                "files": {
                    component["dataset_id"]: component["qa_count"]
                    for component in expected["components"]
                },
                "qa_count": expected["qa_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
