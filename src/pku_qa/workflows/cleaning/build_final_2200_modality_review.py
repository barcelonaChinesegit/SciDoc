#!/usr/bin/env python3
"""Prepare and atomically apply four-file final-2200 modality API reviews."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pku_qa.workflows.cleaning.reclassify_cross_pdf_modalities_api import (
    REVIEW_VERSION,
    apply_decisions,
    audit_items,
)
from pku_qa.workflows.selection.final_2200_contract import validate_final_dataset
from pku_qa.workflows.selection.normalize_final_2200_release import atomic_json
from pku_qa.workflows.selection.sync_final_2200_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = ROOT / "data/qa/7.final_2200"
MANIFEST_PATH = FINAL_DIR / "rel__collection__final_2200__manifest.json"
AUDIT_DIR = ROOT / "data/qa/6.review/final_2200_modalities"
SINGLE_INPUT = AUDIT_DIR / "work__single_pdf__modality_api_input__n1400__v1.json"
SINGLE_AUDIT = AUDIT_DIR / "work__single_pdf__modality_api_audit__n1400__v1.json"
CROSS_AUDIT = AUDIT_DIR / "work__cross_pdf__modality_api_audit__n800__v1.json"
FINAL_FILES = (
    "ordinary_qa.json",
    "unanswerable_qa.json",
    "reasoning_qa.json",
    "cross_pdf_qa.json",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def merge_review_input(*datasets: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for filename, dataset in datasets:
        for paper_id, source_paper in dataset.items():
            paper_id = str(paper_id)
            paper = merged.setdefault(
                paper_id,
                {key: copy.deepcopy(value) for key, value in source_paper.items() if key != "QA"},
            )
            target_qas = paper.setdefault("QA", {})
            for qa_id, source_qa in source_paper["QA"].items():
                if qa_id in target_qas:
                    raise ValueError(f"Duplicate QA key across final files: {paper_id}/{qa_id}")
                qa = copy.deepcopy(source_qa)
                qa["modality_review_source_file"] = filename
                target_qas[str(qa_id)] = qa
    return merged


def seed_cross_audit() -> dict[str, Any]:
    if CROSS_AUDIT.is_file():
        return read_json(CROSS_AUDIT)
    return {
        "schema_version": 1,
        "review_version": REVIEW_VERSION,
        "source_path": str((FINAL_DIR / "cross_pdf_qa.json").resolve()),
        "model": "claude-sonnet-5",
        "bundles": {},
    }


def prepare() -> dict[str, int]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: read_json(FINAL_DIR / name) for name in FINAL_FILES}
    single = merge_review_input(
        ("ordinary_qa.json", datasets["ordinary_qa.json"]),
        ("unanswerable_qa.json", datasets["unanswerable_qa.json"]),
        ("reasoning_qa.json", datasets["reasoning_qa.json"]),
    )
    atomic_json(SINGLE_INPUT, single)
    if not SINGLE_AUDIT.is_file():
        atomic_json(
            SINGLE_AUDIT,
            {
                "schema_version": 1,
                "review_version": REVIEW_VERSION,
                "source_path": str(SINGLE_INPUT.resolve()),
                "model": "claude-sonnet-5",
                "bundles": {},
            },
        )
    atomic_json(CROSS_AUDIT, seed_cross_audit())
    return {
        "single_pdf_qas": sum(len(paper["QA"]) for paper in single.values()),
        "single_pdf_papers": len(single),
        "cross_pdf_qas": sum(len(paper["QA"]) for paper in datasets["cross_pdf_qa.json"].values()),
        "cross_pdf_seeded_qas": len(audit_items(read_json(CROSS_AUDIT))),
    }


def apply() -> dict[str, Any]:
    datasets = {name: read_json(FINAL_DIR / name) for name in FINAL_FILES}
    single_audit = read_json(SINGLE_AUDIT)
    cross_audit = read_json(CROSS_AUDIT)
    counts: dict[str, dict[str, int]] = {}
    for name in FINAL_FILES:
        audit = cross_audit if name == "cross_pdf_qa.json" else single_audit
        audit_path = CROSS_AUDIT if name == "cross_pdf_qa.json" else SINGLE_AUDIT
        result = apply_decisions(datasets[name], audit, audit_path=audit_path)
        validate_final_dataset(datasets[name], label=name)
        counts[name] = dict(result)

    for name, dataset in datasets.items():
        atomic_json(FINAL_DIR / name, dataset)
    atomic_json(MANIFEST_PATH, build_manifest(FINAL_DIR))
    return {
        "qa_count": 2200,
        "files": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.prepare and not args.apply:
        parser.error("choose --prepare or --apply")
    result = prepare() if args.prepare else apply()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
