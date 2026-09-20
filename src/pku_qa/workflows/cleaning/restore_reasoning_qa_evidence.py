#!/usr/bin/env python3
"""Restore reasoning-QA evidence pages from the referenced source QAs.

The reasoning dataset records the ordinary-QA identifiers used to construct
each item.  For a reasoning item, its gold evidence is the sorted union of the
1-based physical PDF pages attached to all of those source QAs.  This script
performs that deterministic join, validates the result against the local PDF,
and records enough provenance to reproduce and audit every contributed page.

The input dataset is never modified in place.  By default a sibling
``work__reasoning__historical_clean__batch00__n100.json`` file is created.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pypdf import PdfReader

from pku_qa.pdf_assets import resolve_pdf_path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REASONING_DATASET = (
    ROOT
    / "data/qa/3.reasoning/"
    "work__reasoning__historical_clean__batch00__n100.json"
)
DEFAULT_SOURCE_DATASET = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
DEFAULT_PDF_DIR = ROOT / "data/pdfs"
PROVENANCE_SCHEMA_VERSION = 1


class EvidenceRestoreError(ValueError):
    """Raised when evidence cannot be restored without ambiguity."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def question_answer_sha256(qa: Mapping[str, Any]) -> str:
    """Hash only the immutable semantic question/answer pair."""

    return canonical_sha256(
        {"question": qa.get("question"), "answer": qa.get("answer")}
    )


def dataset_question_answer_sha256(dataset: Mapping[str, Any]) -> str:
    """Hash all reasoning question/answer pairs at their composite grain."""

    rows: list[dict[str, Any]] = []
    for paper_key in sorted(dataset, key=str):
        paper = dataset[paper_key]
        if not isinstance(paper, Mapping):
            raise EvidenceRestoreError(
                f"paper {paper_key!r} must be a JSON object"
            )
        qa_map = paper.get("QA")
        if not isinstance(qa_map, Mapping):
            raise EvidenceRestoreError(
                f"paper {paper_key!r} has no valid QA object"
            )
        for qa_id in sorted(qa_map, key=str):
            qa = qa_map[qa_id]
            if not isinstance(qa, Mapping):
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_key}:{qa_id} must be an object"
                )
            rows.append(
                {
                    "paper_id": str(paper_key),
                    "qa_id": str(qa_id),
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                }
            )
    return canonical_sha256(rows)


def display_path(path: Path) -> str:
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute.resolve().as_posix()


def normalize_pages(
    raw_pages: Any,
    *,
    paper_id: str,
    qa_id: str,
    page_count: int,
) -> list[int]:
    if not isinstance(raw_pages, list) or not raw_pages:
        raise EvidenceRestoreError(
            f"source QA {paper_id}:{qa_id} must have non-empty evidence_pages"
        )

    pages: list[int] = []
    for page in raw_pages:
        # bool is an int subclass and must not silently become page 1/0.
        if type(page) is not int:
            raise EvidenceRestoreError(
                f"source QA {paper_id}:{qa_id} has non-integer page {page!r}"
            )
        if page < 1 or page > page_count:
            raise EvidenceRestoreError(
                f"source QA {paper_id}:{qa_id} page {page} is outside "
                f"the physical PDF range 1..{page_count}"
            )
        pages.append(page)
    return sorted(set(pages))


def _canonical_paper_id(paper_key: Any, paper: Mapping[str, Any]) -> str:
    paper_id = str(paper_key)
    for field in ("paper", "source_pdf_id"):
        if field in paper and str(paper[field]) != paper_id:
            raise EvidenceRestoreError(
                f"paper key {paper_id!r} disagrees with {field}={paper[field]!r}"
            )
    return paper_id


def restore_reasoning_qa_evidence(
    reasoning_dataset: Mapping[str, Any],
    source_dataset: Mapping[str, Any],
    *,
    pdf_dir: Path,
    source_dataset_path: Path,
    source_dataset_sha256: str,
    expected_count: int | None = 100,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an enriched copy and a deterministic validation summary."""

    if not isinstance(reasoning_dataset, Mapping):
        raise EvidenceRestoreError("reasoning dataset must be a JSON object")
    if not isinstance(source_dataset, Mapping):
        raise EvidenceRestoreError("source dataset must be a JSON object")

    restored: dict[str, Any] = copy.deepcopy(dict(reasoning_dataset))
    dataset_qa_hash_before = dataset_question_answer_sha256(reasoning_dataset)
    pdf_cache: dict[str, tuple[Path, int, str]] = {}
    reasoning_count = 0
    source_reference_count = 0
    unique_source_refs: set[tuple[str, str]] = set()
    union_page_reference_count = 0
    union_page_count_distribution: Counter[int] = Counter()

    for paper_key, paper in restored.items():
        if not isinstance(paper, dict):
            raise EvidenceRestoreError(
                f"paper {paper_key!r} must be a JSON object"
            )
        paper_id = _canonical_paper_id(paper_key, paper)
        source_paper = source_dataset.get(paper_id)
        if not isinstance(source_paper, Mapping):
            raise EvidenceRestoreError(
                f"paper {paper_id} is missing from the source QA dataset"
            )
        source_qa_map = source_paper.get("QA")
        if not isinstance(source_qa_map, Mapping):
            raise EvidenceRestoreError(
                f"paper {paper_id} has no valid source QA object"
            )

        if paper_id not in pdf_cache:
            pdf_path = resolve_pdf_path(paper_id, [pdf_dir])
            if not pdf_path.is_file():
                raise EvidenceRestoreError(
                    f"local PDF is missing for paper {paper_id}: {pdf_path}"
                )
            try:
                page_count = len(PdfReader(str(pdf_path)).pages)
            except Exception as exc:
                raise EvidenceRestoreError(
                    f"failed to inspect PDF for paper {paper_id}: {exc}"
                ) from exc
            if page_count < 1:
                raise EvidenceRestoreError(
                    f"PDF for paper {paper_id} has no physical pages"
                )
            pdf_cache[paper_id] = (
                pdf_path,
                page_count,
                sha256_file(pdf_path),
            )
        pdf_path, page_count, pdf_sha256 = pdf_cache[paper_id]

        qa_map = paper.get("QA")
        if not isinstance(qa_map, dict):
            raise EvidenceRestoreError(
                f"paper {paper_id} has no valid reasoning QA object"
            )
        for reasoning_qa_id, qa in qa_map.items():
            if not isinstance(qa, dict):
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} must be an object"
                )
            if not isinstance(qa.get("question"), str) or not qa["question"].strip():
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} has no question"
                )
            if not isinstance(qa.get("answer"), str) or not qa["answer"].strip():
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} has no answer"
                )
            if "source_pdf_id" in qa and str(qa["source_pdf_id"]) != paper_id:
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} has mismatched "
                    f"source_pdf_id={qa['source_pdf_id']!r}"
                )

            source_ids = qa.get("source_qa_ids")
            if not isinstance(source_ids, list) or not source_ids:
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} has no "
                    "source_qa_ids"
                )
            if any(not isinstance(value, str) or not value for value in source_ids):
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} has malformed "
                    "source_qa_ids"
                )
            if len(set(source_ids)) != len(source_ids):
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} has duplicate "
                    "source_qa_ids"
                )

            item_qa_hash_before = question_answer_sha256(qa)
            contribution_rows: list[dict[str, Any]] = []
            union_pages: set[int] = set()
            for source_qa_id in source_ids:
                source_qa = source_qa_map.get(source_qa_id)
                if not isinstance(source_qa, Mapping):
                    raise EvidenceRestoreError(
                        f"source QA {paper_id}:{source_qa_id} referenced by "
                        f"{reasoning_qa_id} does not exist"
                    )
                pages = normalize_pages(
                    source_qa.get("evidence_pages"),
                    paper_id=paper_id,
                    qa_id=source_qa_id,
                    page_count=page_count,
                )
                union_pages.update(pages)
                contribution_rows.append(
                    {
                        "source_qa_id": source_qa_id,
                        "evidence_pages": pages,
                        "source_question_answer_sha256": question_answer_sha256(
                            source_qa
                        ),
                    }
                )
                source_reference_count += 1
                unique_source_refs.add((paper_id, source_qa_id))

            evidence_pages = sorted(union_pages)
            if not evidence_pages:
                raise EvidenceRestoreError(
                    f"reasoning QA {paper_id}:{reasoning_qa_id} restored no pages"
                )
            qa["evidence_pages"] = evidence_pages
            qa["evidence_provenance"] = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "method": "sorted_union_of_source_qa_evidence_pages",
                "page_basis": "1-based physical page in local PDF",
                "source_dataset": display_path(source_dataset_path),
                "source_dataset_sha256": source_dataset_sha256,
                "source_pdf": display_path(pdf_path),
                "source_pdf_sha256": pdf_sha256,
                "source_pdf_page_count": page_count,
                "reasoning_question_answer_sha256": item_qa_hash_before,
                "source_qa_contributions": contribution_rows,
                "validation": {
                    "all_source_qa_ids_resolved": True,
                    "pages_are_positive_integers": True,
                    "pages_are_sorted_and_unique": True,
                    "pages_within_pdf_bounds": True,
                },
            }
            if question_answer_sha256(qa) != item_qa_hash_before:
                raise AssertionError(
                    f"question/answer changed for {paper_id}:{reasoning_qa_id}"
                )
            reasoning_count += 1
            union_page_reference_count += len(evidence_pages)
            union_page_count_distribution[len(evidence_pages)] += 1

    if expected_count is not None and reasoning_count != expected_count:
        raise EvidenceRestoreError(
            f"expected {expected_count} reasoning QAs, found {reasoning_count}"
        )

    qa_hash_after = dataset_question_answer_sha256(restored)
    if qa_hash_after != dataset_qa_hash_before:
        raise AssertionError("dataset question/answer hash changed during restore")

    report: dict[str, Any] = {
        "method": "sorted_union_of_source_qa_evidence_pages",
        "page_basis": "1-based physical page in local PDF",
        "reasoning_qa_count": reasoning_count,
        "resolved_reasoning_qa_count": reasoning_count,
        "source_qa_reference_count": source_reference_count,
        "unique_source_qa_count": len(unique_source_refs),
        "source_pdf_count": len(pdf_cache),
        "union_evidence_page_reference_count": union_page_reference_count,
        "union_evidence_page_count_distribution": {
            str(key): union_page_count_distribution[key]
            for key in sorted(union_page_count_distribution)
        },
        "source_dataset": display_path(source_dataset_path),
        "source_dataset_sha256": source_dataset_sha256,
        "input_question_answer_sha256": dataset_qa_hash_before,
        "output_question_answer_sha256": qa_hash_after,
        "question_answer_hash_unchanged": dataset_qa_hash_before == qa_hash_after,
        "all_source_qa_ids_resolved": True,
        "all_evidence_pages_valid": True,
    }
    return restored, report


def restore_from_paths(
    reasoning_dataset_path: Path,
    source_dataset_path: Path,
    pdf_dir: Path,
    *,
    expected_count: int | None = 100,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        reasoning_dataset = json.loads(
            reasoning_dataset_path.read_text(encoding="utf-8")
        )
        source_dataset = json.loads(
            source_dataset_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceRestoreError(f"failed to load input dataset: {exc}") from exc
    return restore_reasoning_qa_evidence(
        reasoning_dataset,
        source_dataset,
        pdf_dir=pdf_dir,
        source_dataset_path=source_dataset_path,
        source_dataset_sha256=sha256_file(source_dataset_path),
        expected_count=expected_count,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reasoning-dataset", type=Path, default=DEFAULT_REASONING_DATASET
    )
    parser.add_argument(
        "--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional rebuilt copy; omit to verify the canonical dataset",
    )
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and print the report without writing an output dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file; the input still cannot be replaced",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reasoning_path = args.reasoning_dataset.resolve()
    output_path = args.output.resolve() if args.output else None
    if output_path == reasoning_path:
        raise SystemExit("refusing to overwrite the input reasoning dataset")
    if (
        output_path is not None
        and output_path.exists()
        and not args.overwrite
        and not args.check_only
    ):
        raise SystemExit(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )

    restored, report = restore_from_paths(
        reasoning_path,
        args.source_dataset.resolve(),
        args.pdf_dir.resolve(),
        expected_count=args.expected_count,
    )
    if output_path is not None and not args.check_only:
        atomic_write_json(output_path, restored)
        report["output"] = display_path(output_path)
        report["output_sha256"] = sha256_file(output_path)
    else:
        report["output"] = None
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
