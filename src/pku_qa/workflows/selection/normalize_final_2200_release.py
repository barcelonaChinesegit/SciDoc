#!/usr/bin/env python3
"""Normalize and verify the shared core contract of the final 2,200 QA."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from pku_qa.workflows.selection.final_2200_contract import (
    canonical_modal_types,
    category_for_reasoning_type,
    normalize_nonstandard_category,
    ordered_qa,
    representative_source_category,
    validate_final_dataset,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = ROOT / "data/qa/7.final_2200/rel__collection__final_2200__manifest.json"
DEFAULT_BASE = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
NORMALIZATION_VERSION = "2026-09-01.1"
REASONING_COMPONENTS = {"reasoning_200"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_provenance(qa: dict[str, Any]) -> dict[str, Any]:
    provenance = qa.get("annotation_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        qa["annotation_provenance"] = provenance
    return provenance


def normalize_reasoning_qa(
    qa: dict[str, Any], source_paper: dict[str, Any]
) -> dict[str, Any]:
    source_ids = qa.get("source_qa_ids")
    if not isinstance(source_ids, list) or not source_ids:
        raise ValueError("reasoning QA has no source_qa_ids")
    source_qas = []
    for source_id in source_ids:
        source = source_paper.get("QA", {}).get(str(source_id))
        if not isinstance(source, dict):
            raise ValueError(f"unresolved reasoning source QA: {source_id}")
        source_qas.append(source)
    modality_review = (
        qa.get("annotation_provenance", {}).get("modality_api_review")
        if isinstance(qa.get("annotation_provenance"), dict)
        else None
    )
    if (
        isinstance(modality_review, dict)
        and modality_review.get("page_support_complete") is True
        and modality_review.get("modal_types") == qa.get("modal_types")
    ):
        qa["modal_types"] = canonical_modal_types(qa["modal_types"])
        modal_types_method = "vision_api_pdf_page_review"
    else:
        qa["modal_types"] = canonical_modal_types(
            list(
                dict.fromkeys(
                    modality
                    for source in source_qas
                    for modality in source["modal_types"]
                )
            )
        )
        modal_types_method = "ordered_union_of_bound_source_qa_modal_types"
    qa["question_type"] = "Inferential"
    source_categories = [str(source["question_category"]) for source in source_qas]
    qa["question_category"] = representative_source_category(
        source_categories, str(source_paper["primary_category"])
    )
    provenance = ensure_provenance(qa)
    provenance["final_2200_core_normalization"] = {
        "version": NORMALIZATION_VERSION,
        "modal_types_method": modal_types_method,
        "question_type_method": "reasoning_questions_are_inferential",
        "question_category_method": "source_category_majority_first_source_tiebreak",
        "source_qa_ids": [str(value) for value in source_ids],
        "source_question_categories": source_categories,
    }
    return ordered_qa(qa)


def normalize_general_qa(
    qa: dict[str, Any], *, component_id: str, primary_category: str
) -> dict[str, Any]:
    before_modalities = copy.deepcopy(qa.get("modal_types"))
    qa["modal_types"] = canonical_modal_types(before_modalities)
    before_type = qa.get("question_type")
    before_category = str(qa.get("question_category", ""))
    if before_type in {"Unanswerable", "cross_document_reasoning"}:
        qa["question_type"] = (
            "Literal" if before_type == "Unanswerable" else "Inferential"
        )
    if component_id == "cross_pdf_second_400":
        qa["question_category"] = category_for_reasoning_type(
            primary_category, str(qa.get("reasoning_type", ""))
        )
    else:
        qa["question_category"] = normalize_nonstandard_category(
            primary_category, before_category
        )
    qa["evidence_pages"] = sorted(set(qa.get("evidence_pages", [])))
    changed = (
        before_modalities != qa["modal_types"]
        or before_type != qa["question_type"]
        or before_category != qa["question_category"]
    )
    if changed or component_id == "cross_pdf_second_400":
        provenance = ensure_provenance(qa)
        record = {
            "version": NORMALIZATION_VERSION,
            "modal_types_method": (
                "native_fulltext_generation_annotation"
                if component_id == "cross_pdf_second_400"
                else "native_annotation_alias_and_order_normalization"
            ),
            "question_category_method": (
                "discipline_taxonomy_role_from_reasoning_type"
                if component_id == "cross_pdf_second_400"
                else "discipline_taxonomy_normalization"
            ),
        }
        if before_modalities != qa["modal_types"]:
            record["original_modal_types"] = before_modalities
        if before_type != qa["question_type"]:
            record["original_question_type"] = before_type
        if before_category != qa["question_category"]:
            record["original_question_category"] = before_category
        provenance.setdefault("final_2200_core_normalization", record)
    return ordered_qa(qa)


def normalize_component(
    component_id: str, dataset: dict[str, Any], base: dict[str, Any]
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for paper_id, source_record in dataset.items():
        if not isinstance(source_record, dict):
            raise ValueError(f"{component_id}:{paper_id}: invalid paper")
        source_qas = source_record.get("QA")
        if not isinstance(source_qas, dict):
            raise ValueError(f"{component_id}:{paper_id}: invalid QA mapping")
        if not source_qas:
            continue
        paper = copy.deepcopy(source_record)
        paper["paper"] = str(paper_id)
        if component_id in REASONING_COMPONENTS:
            identities = [
                qa.get("annotation_provenance", {}).get("final_2200_identity", {})
                for qa in source_qas.values()
                if isinstance(qa, dict)
            ]
            legacy_paper_ids = {
                str(identity.get("legacy_paper_id", ""))
                for identity in identities
                if isinstance(identity, dict)
            }
            if len(legacy_paper_ids) != 1:
                raise ValueError(
                    f"invalid reasoning legacy paper identity: {paper_id}"
                )
            base_paper = base.get(next(iter(legacy_paper_ids)))
            if not isinstance(base_paper, dict):
                raise ValueError(f"missing base paper for reasoning QA: {paper_id}")
            paper["primary_category"] = base_paper["primary_category"]
            paper["secondary_category"] = base_paper["secondary_category"]
            qas = {
                str(qa_id): normalize_reasoning_qa(copy.deepcopy(qa), base_paper)
                for qa_id, qa in source_qas.items()
            }
        else:
            primary = str(paper.get("primary_category", ""))
            qas = {
                str(qa_id): normalize_general_qa(
                    copy.deepcopy(qa), component_id=component_id, primary_category=primary
                )
                for qa_id, qa in source_qas.items()
            }
        normalized[str(paper_id)] = {
            "paper": paper["paper"],
            "primary_category": paper["primary_category"],
            "secondary_category": paper["secondary_category"],
            **{
                key: value
                for key, value in paper.items()
                if key not in {"paper", "primary_category", "secondary_category", "QA"}
            },
            "QA": qas,
        }
    validate_final_dataset(normalized, label=component_id)
    return normalized


def refresh_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    refreshed = copy.deepcopy(manifest)
    total = 0
    for component in refreshed["components"]:
        path = Path(component["path"])
        dataset = read_json(path)
        rows = [qa for paper in dataset.values() for qa in paper["QA"].values()]
        component["sha256"] = sha256_file(path)
        component["qa_count"] = len(rows)
        component["paper_count"] = len(dataset)
        component["answerable"] = sum(qa["answer"] != "Unanswerable" for qa in rows)
        component["unanswerable"] = sum(qa["answer"] == "Unanswerable" for qa in rows)
        total += len(rows)
    refreshed["qa_count"] = total
    if total != 2200:
        raise ValueError(f"expected 2,200 QA, found {total}")
    return refreshed


def main() -> int:
    args = parse_args()
    manifest = read_json(args.manifest)
    base = read_json(args.base)
    changed_files: list[str] = []
    for component in manifest.get("components", []):
        component_id = str(component["id"])
        path = Path(component["path"])
        current = read_json(path)
        normalized = normalize_component(component_id, current, base)
        if current != normalized:
            changed_files.append(str(path))
            if not args.check_only:
                atomic_json(path, normalized)
    if args.check_only:
        if changed_files:
            raise ValueError(f"final 2200 normalization is stale: {changed_files}")
        expected_manifest = refresh_manifest(manifest)
        if expected_manifest != manifest:
            raise ValueError("final 2200 manifest hashes or counts are stale")
    else:
        refreshed = refresh_manifest(manifest)
        atomic_json(args.manifest, refreshed)
    print(
        json.dumps(
            {"status": "current", "qa_count": 2200, "changed_files": changed_files},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
