#!/usr/bin/env python3
"""Apply explicit API-review fixes to hard Cross-PDF candidates for re-review."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument(
        "--review-dir",
        action="append",
        required=True,
        help="Reviewer directory as NAME=PATH; repeat in precedence order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_review_dirs(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --review-dir value: {value!r}")
        result.append((name.strip(), Path(raw_path.strip())))
    return result


def manifest_index(payload: Any) -> dict[str, dict[str, Any]]:
    rows = payload if isinstance(payload, list) else payload.values()
    return {
        str(row.get("id") or row.get("bundle_id") or row.get("paper")): row
        for row in rows
        if isinstance(row, dict)
    }


def docs_for_pages(pages: list[int], manifest: dict[str, Any]) -> list[int]:
    docs: set[int] = set()
    for page in pages:
        matches = [
            index
            for index, source in enumerate(manifest.get("sources", []), start=1)
            if int(source["merged_start_page"])
            <= page
            <= int(source["merged_end_page"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Page {page} maps to {len(matches)} documents")
        docs.add(matches[0])
    return sorted(docs)


def review_items(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    record = read_json(path)
    if record.get("validation_errors"):
        return {}
    return {
        str(item.get("qa_id")): item
        for item in record.get("review", {}).get("items", [])
        if isinstance(item, dict)
    }


def apply_fix(
    qa: dict[str, Any], review: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    corrected = deepcopy(qa)
    changed: list[str] = []
    for source_field, target_field in (
        ("corrected_question", "question"),
        ("corrected_answer", "answer"),
        ("corrected_evidence_pages", "evidence_pages"),
    ):
        value = review.get(source_field)
        if value is None:
            continue
        if target_field == "evidence_pages":
            if not isinstance(value, list) or not value or not all(
                isinstance(page, int) and not isinstance(page, bool) and page > 0
                for page in value
            ):
                raise ValueError("corrected_evidence_pages must be positive integers")
            value = sorted(set(value))
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source_field} must be a non-empty string")
        if corrected.get(target_field) != value:
            corrected[target_field] = value
            changed.append(target_field)
    if "evidence_pages" in changed:
        corrected["source_doc_numbers"] = docs_for_pages(
            corrected["evidence_pages"], manifest
        )
        corrected["source_document_count"] = len(corrected["source_doc_numbers"])
        corrected["evidence_span"] = (
            max(corrected["evidence_pages"]) - min(corrected["evidence_pages"])
        )
    return corrected, changed


def build_fixed_dataset(
    candidates: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    review_dirs: list[tuple[str, Path]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    for bundle_id, bundle in candidates.items():
        indexes = {
            name: review_items(directory / f"{bundle_id}.json")
            for name, directory in review_dirs
        }
        for qa_id, original in bundle.get("QA", {}).items():
            current = deepcopy(original)
            applied: list[dict[str, Any]] = []
            for reviewer, index in indexes.items():
                review = index.get(str(qa_id))
                if not review or review.get("decision") != "FIX":
                    continue
                current, changed = apply_fix(current, review, manifests[bundle_id])
                if changed:
                    applied.append(
                        {
                            "reviewer": reviewer,
                            "changed_fields": changed,
                            "confidence": review.get("confidence"),
                            "reason": review.get("reason"),
                        }
                    )
            if not applied:
                continue
            if len(set(current.get("source_doc_numbers", []))) < 2:
                raise ValueError(f"Fix collapsed {bundle_id}/{qa_id} to one document")
            bucket = output.setdefault(
                bundle_id,
                {
                    **{
                        key: deepcopy(value)
                        for key, value in bundle.items()
                        if key != "QA"
                    },
                    "QA": {},
                },
            )
            bucket["QA"][qa_id] = current
            ledger.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "applied": applied,
                    "corrected_evidence_pages": current.get("evidence_pages"),
                    "corrected_source_doc_numbers": current.get("source_doc_numbers"),
                }
            )
    return output, ledger


def main() -> None:
    args = parse_args()
    candidates = read_json(args.candidates)
    manifests = manifest_index(read_json(args.manifest_json))
    output, ledger = build_fixed_dataset(
        candidates, manifests, parse_review_dirs(args.review_dir)
    )
    atomic_json(args.output, output)
    atomic_json(
        args.ledger_output
        or args.output.with_name(args.output.stem + "_fix_ledger.json"),
        ledger,
    )
    print(
        json.dumps(
            {
                "fixed_questions": len(ledger),
                "fixed_bundles": len(output),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
