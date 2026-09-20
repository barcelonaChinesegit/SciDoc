#!/usr/bin/env python3
"""Synchronize project QA datasets from a human-reviewed gold subset.

The human-reviewed file is authoritative for matching (paper, question)
pairs. Historical model outputs, reports, task state, and review transcripts
are deliberately excluded: rewriting them would destroy evaluation lineage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pku_qa.pdf_assets import resolve_pdf_path


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_AUTHORITY = ROOT / "data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json"
DEFAULT_PDF_DIR = ROOT / "data/pdfs"
DEFAULT_REPORT_DIR = ROOT / "data/results/reports/human_review_sync_20260731"
CORE_FIELDS = (
    "question",
    "answer",
    "evidence_pages",
    "modal_types",
    "question_type",
    "question_category",
)
EXCLUDED_PATH_PARTS = {
    ".git",
    ".next",
    "__pycache__",
    "ablation_finalization",
    "audit_outputs",
    "api_reviews",
    "claude_fix_claude_rereviews",
    "dual_review",
    "dual_review_v2",
    "gemini_fix_claude_rereviews",
    "logs",
    "node_modules",
    "novel_guardian",
    "output",
    "raw",
    "remaining_1941_api_reviews",
    "reports",
    "reviews",
    "second_stage_fix_claude_rereviews",
    "task_queue",
    "three_hop_ablation_finalization",
    "czj1_output",
}


@dataclass
class QARef:
    paper_id: str | None
    qa_id: str | None
    path: tuple[str, ...]
    qa: dict[str, Any]


@dataclass
class GoldRef:
    paper_id: str
    qa_id: str
    qa: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def normalize_pages(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    pages: list[int] = []
    for value in raw:
        if isinstance(value, bool):
            continue
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page >= 1 and page not in pages:
            pages.append(page)
    return sorted(pages)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    newline = "\n"
    if path.exists():
        with path.open("rb") as existing:
            if b"\r\n" in existing.read(64 * 1024):
                newline = "\r\n"
    with temp.open("w", encoding="utf-8", newline=newline) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def iter_qa_refs(payload: Any) -> Iterable[QARef]:
    """Yield QA records while retaining a standard dataset's paper id."""

    def walk(
        node: Any,
        path: tuple[str, ...],
        paper_id: str | None,
        qa_id: str | None,
    ) -> Iterable[QARef]:
        if isinstance(node, dict):
            inferred_paper = paper_id
            for key in ("paper_id", "source_paper_id"):
                if node.get(key) not in (None, ""):
                    inferred_paper = str(node[key])
                    break
            if isinstance(node.get("question"), str) and "answer" in node:
                yield QARef(inferred_paper, qa_id, path, node)
            for key, value in node.items():
                if key == "QA" and isinstance(value, dict):
                    for child_id, child in value.items():
                        yield from walk(
                            child,
                            path + (key, str(child_id)),
                            inferred_paper,
                            str(child_id),
                        )
                else:
                    yield from walk(
                        value, path + (str(key),), inferred_paper, qa_id
                    )
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(
                    value, path + (str(index),), paper_id, qa_id
                )

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict) and isinstance(value.get("QA"), dict):
                yield from walk(value, (str(key),), str(key), None)
            else:
                yield from walk(value, (str(key),), None, None)
    else:
        yield from walk(payload, (), None, None)


def authority_index(
    payload: Any,
) -> tuple[
    dict[tuple[str, str], GoldRef],
    dict[str, GoldRef],
    dict[str, set[int]],
]:
    by_paper_question: dict[tuple[str, str], GoldRef] = {}
    by_question: dict[str, GoldRef] = {}
    question_counts = Counter()
    pages_by_paper: defaultdict[str, set[int]] = defaultdict(set)
    for ref in iter_qa_refs(payload):
        if ref.paper_id is None or ref.qa_id is None:
            raise ValueError(f"authority QA lacks paper/QA identity: {ref.path}")
        question = normalize_text(ref.qa.get("question"))
        if not question or not normalize_text(ref.qa.get("answer")):
            raise ValueError(f"authority QA is empty: {ref.path}")
        key = (ref.paper_id, question)
        if key in by_paper_question:
            raise ValueError(f"duplicate authority paper/question: {key}")
        gold = GoldRef(ref.paper_id, ref.qa_id, deepcopy(ref.qa))
        by_paper_question[key] = gold
        by_question[question] = gold
        question_counts[question] += 1
        pages_by_paper[ref.paper_id].update(
            normalize_pages(ref.qa.get("evidence_pages"))
        )
    duplicates = [key for key, count in question_counts.items() if count > 1]
    for key in duplicates:
        by_question.pop(key, None)
    return by_paper_question, by_question, dict(pages_by_paper)


def discover_json_files(project_root: Path, authority: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_root.rglob("*.json"):
        if path == authority or any(part in EXCLUDED_PATH_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def selected_option(qa: dict[str, Any]) -> dict[str, Any] | None:
    options = qa.get("options")
    if not isinstance(options, list):
        return None
    answer = normalize_text(qa.get("answer"))
    for option in options:
        if isinstance(option, dict) and normalize_text(option.get("id")) == answer:
            return option
    return None


def effective_answer(qa: dict[str, Any]) -> Any:
    option = selected_option(qa)
    return option.get("text") if option is not None else qa.get("answer")


def match_gold(
    ref: QARef,
    by_paper_question: dict[tuple[str, str], GoldRef],
    by_question: dict[str, GoldRef],
) -> GoldRef | None:
    question = normalize_text(ref.qa.get("question"))
    if ref.paper_id is not None:
        return by_paper_question.get((ref.paper_id, question))
    return by_question.get(question)


def apply_core_gold(qa: dict[str, Any], gold: GoldRef) -> dict[str, list[Any]]:
    changes: dict[str, list[Any]] = {}
    option = selected_option(qa)
    for field in CORE_FIELDS:
        if field == "answer" and option is not None:
            before = option.get("text")
            after = deepcopy(gold.qa.get("answer"))
            if before != after:
                option["text"] = after
                changes["effective_answer"] = [before, after]
            continue
        before = deepcopy(qa.get(field))
        after = deepcopy(gold.qa.get(field))
        if before != after:
            qa[field] = after
            changes[field] = [before, after]
    return changes


def refresh_enrichment(
    qa: dict[str, Any],
    gold: GoldRef,
    pdf_dir: Path,
    pages_by_paper: dict[str, set[int]],
    pdf_cache: dict[str, Any],
    authority_path: Path,
    authority_hash: str,
) -> None:
    from scripts.single_pdf_challenge_pipeline import (
        answer_format,
        answer_unit,
        enrich_qa,
        load_pdf_info,
        numeric_tolerance,
    )

    if gold.paper_id not in pdf_cache:
        pdf_path = resolve_pdf_path(gold.paper_id, [pdf_dir])
        if not pdf_path.exists():
            raise FileNotFoundError(f"missing PDF for enriched QA: {pdf_path}")
        pdf_cache[gold.paper_id] = load_pdf_info(
            pdf_path, pages_by_paper.get(gold.paper_id, set())
        )
    old_provenance = deepcopy(qa.get("annotation_provenance", {}))
    old_review_status = qa.get("review_status")
    enriched = enrich_qa(gold.paper_id, gold.qa_id, qa, pdf_cache[gold.paper_id])
    answer = str(enriched.get("answer", ""))
    fmt = answer_format(answer)
    enriched["answer_format"] = fmt
    enriched["answer_unit"] = answer_unit(answer)
    enriched["numeric_tolerance"] = numeric_tolerance(answer, fmt)
    enriched["annotation_provenance"] = {
        **old_provenance,
        "human_review_source": str(authority_path.relative_to(ROOT)),
        "human_review_source_key": f"{gold.paper_id}:{gold.qa_id}",
        "human_review_source_sha256": authority_hash,
    }
    if old_review_status is not None:
        enriched["review_status"] = old_review_status
    qa.clear()
    qa.update(enriched)


def sync_unanswerable(
    ref: QARef,
    by_paper_question: dict[tuple[str, str], GoldRef],
    by_question: dict[str, GoldRef],
) -> dict[str, list[Any]]:
    qa = ref.qa
    if normalize_text(qa.get("answer")) != "unanswerable":
        return {}
    changes: dict[str, list[Any]] = {}
    for field, desired in (
        ("evidence_pages", []),
        ("oracle_pages", []),
        ("evidence_items", []),
        ("evidence_hops", 0),
        ("evidence_span", 0),
        ("evidence_span_ratio", 0),
    ):
        if field in qa and qa.get(field) != desired:
            changes[field] = [deepcopy(qa.get(field)), deepcopy(desired)]
            qa[field] = deepcopy(desired)
    construction = qa.get("unanswerable_construction")
    if not isinstance(construction, dict):
        return changes
    original_question = normalize_text(construction.get("original_question"))
    gold = None
    if ref.paper_id is not None:
        gold = by_paper_question.get((ref.paper_id, original_question))
    elif original_question:
        gold = by_question.get(original_question)
    if gold is None:
        return changes
    for field, desired in (
        ("original_question", gold.qa.get("question")),
        ("original_answer", gold.qa.get("answer")),
        ("nearest_neighbor_pages", normalize_pages(gold.qa.get("evidence_pages"))),
    ):
        before = deepcopy(construction.get(field))
        if before != desired:
            construction[field] = deepcopy(desired)
            changes[f"unanswerable_construction.{field}"] = [before, desired]
    return changes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sync_file(
    path: Path,
    by_paper_question: dict[tuple[str, str], GoldRef],
    by_question: dict[str, GoldRef],
    pages_by_paper: dict[str, set[int]],
    pdf_dir: Path,
    pdf_cache: dict[str, Any],
    authority_path: Path,
    authority_hash: str,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    refs = list(iter_qa_refs(payload))
    matched = 0
    changed_records = 0
    unanswerable_records = 0
    field_counts = Counter()
    detail_rows: list[dict[str, Any]] = []
    for ref in refs:
        record_changes = sync_unanswerable(
            ref, by_paper_question, by_question
        )
        if normalize_text(ref.qa.get("answer")) == "unanswerable":
            unanswerable_records += 1
        gold = match_gold(ref, by_paper_question, by_question)
        if gold is not None:
            matched += 1
            before_effective = deepcopy(effective_answer(ref.qa))
            record_changes.update(apply_core_gold(ref.qa, gold))
            if "evidence_items" in ref.qa and "ablation" not in ref.qa:
                refresh_enrichment(
                    ref.qa,
                    gold,
                    pdf_dir,
                    pages_by_paper,
                    pdf_cache,
                    authority_path,
                    authority_hash,
                )
            after_effective = deepcopy(effective_answer(ref.qa))
            if before_effective != after_effective:
                record_changes.setdefault(
                    "effective_answer", [before_effective, after_effective]
                )
        if record_changes:
            changed_records += 1
            field_counts.update(record_changes.keys())
            detail_rows.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "paper_id": ref.paper_id,
                    "qa_id": ref.qa_id,
                    "question": ref.qa.get("question", ""),
                    "changed_fields": "|".join(sorted(record_changes)),
                    "changes_json": json.dumps(
                        record_changes, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    stats = {
        "file": str(path.relative_to(ROOT)),
        "qa_records": len(refs),
        "authority_matches": matched,
        "changed_records": changed_records,
        "unanswerable_records": unanswerable_records,
        "field_change_counts": dict(sorted(field_counts.items())),
    }
    return payload, stats, detail_rows


def main() -> None:
    args = parse_args()
    authority_path = args.authority.resolve()
    authority = load_json(authority_path)
    by_paper_question, by_question, pages_by_paper = authority_index(authority)
    authority_hash = sha256(authority_path)
    files = discover_json_files(args.project_root.resolve(), authority_path)
    pdf_cache: dict[str, Any] = {}
    file_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    changed_files = 0
    scanned_qa_records = 0
    for path in files:
        try:
            payload, stats, details = sync_file(
                path,
                by_paper_question,
                by_question,
                pages_by_paper,
                args.pdf_dir.resolve(),
                pdf_cache,
                authority_path,
                authority_hash,
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        scanned_qa_records += stats["qa_records"]
        if stats["authority_matches"] or stats["unanswerable_records"]:
            file_rows.append(stats)
        if stats["changed_records"]:
            changed_files += 1
            if args.apply:
                atomic_write_json(path, payload)
        detail_rows.extend(details)


    authority_papers = len({paper for paper, _ in by_paper_question})
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "authority": str(authority_path.relative_to(ROOT)),
        "authority_sha256": authority_hash,
        "authority_qa_count": len(by_paper_question),
        "authority_paper_count": authority_papers,
        "authority_unique_question_count": len(by_question),
        "matching_rule": (
            "normalized exact question within the same paper; globally unique "
            "question only for records without paper identity"
        ),
        "scanned_json_files": len(files),
        "scanned_qa_records": scanned_qa_records,
        "files_with_relevant_records": len(file_rows),
        "changed_files": changed_files,
        "changed_records": len(detail_rows),
        "excluded_path_parts": sorted(EXCLUDED_PATH_PARTS),
        "files": file_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "sync_manifest.json", manifest)
    write_csv(args.output_dir / "changed_records.csv", detail_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
