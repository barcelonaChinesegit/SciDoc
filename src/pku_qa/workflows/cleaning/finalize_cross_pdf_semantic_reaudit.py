#!/usr/bin/env python3
"""Build the publication-grade cross-paper QA release from audited decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pku_qa.pdf_assets import output_pdf_path, resolve_pdf_path

try:
    from pku_qa.workflows.generation.build_cross_pdf_gemini_fix_rereview import mapped_doc_numbers
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from pku_qa.workflows.generation.build_cross_pdf_gemini_fix_rereview import mapped_doc_numbers


ROOT = Path("data/qa/4.cross_pdf/semantic_reaudit")
MANIFEST = Path(
    "data/qa_generation/expansion_20260711/cross_pdf_bundle_manifest.json"
)
PRIOR_QA = Path("data/qa/qa_cross_pdf_manual_verified_500_20260724.json")
SOURCE_PDF_DIR = Path("data/pdfs")
OUTPUT_QA = Path("data/qa/qa_cross_pdf_semantic_verified_20260724.json")
OUTPUT_PDF_DIR = Path("data/pdfs")
OUTPUT_DIR = ROOT / "final_release"
CRITERIA = (
    "cross_document_required",
    "question_coherent",
    "question_specific",
    "answer_correct",
    "answer_complete",
    "evidence_sufficient",
    "no_unsupported_claim",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_items(path: Path) -> dict[str, dict[str, Any]]:
    record = load_json(path)
    if record.get("validation_errors"):
        raise ValueError(
            f"Review has validation errors: {path}: "
            f"{record['validation_errors']}"
        )
    items = record["review"]["items"]
    result = {str(item["qa_id"]): item for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate QA IDs in review: {path}")
    return result


def all_criteria_true(review: dict[str, Any]) -> bool:
    return all(review["criteria"].get(name) is True for name in CRITERIA)


def support_doc_numbers(review: dict[str, Any]) -> set[int]:
    return {
        int(row["doc_number"])
        for row in review.get("support", [])
        if isinstance(row.get("doc_number"), int)
        and not isinstance(row.get("doc_number"), bool)
    }


def apply_review_fix(
    item: dict[str, Any], review: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    result = deepcopy(item)
    changed: list[str] = []
    if review["decision"] != "FIX":
        return result, changed
    for source_name, target_name in (
        ("corrected_question", "question"),
        ("corrected_answer", "answer"),
        ("corrected_evidence_pages", "evidence_pages"),
    ):
        value = review.get(source_name)
        if value not in (None, "", []):
            result[target_name] = deepcopy(value)
            changed.append(target_name)
    if not changed:
        raise ValueError("FIX decision contains no actual correction")
    return result, changed


def structurally_valid_item(
    item: dict[str, Any], sources: list[dict[str, Any]]
) -> tuple[bool, list[int], str | None]:
    if not isinstance(item.get("question"), str) or not item["question"].strip():
        return False, [], "blank_question"
    if not isinstance(item.get("answer"), str) or not item["answer"].strip():
        return False, [], "blank_answer"
    pages = item.get("evidence_pages")
    if (
        not isinstance(pages, list)
        or not pages
        or not all(
            isinstance(page, int) and not isinstance(page, bool)
            for page in pages
        )
    ):
        return False, [], "malformed_evidence_pages"
    docs = mapped_doc_numbers(pages, sources)
    if len(docs) < 2:
        return False, sorted(docs), "evidence_spans_fewer_than_two_papers"
    return True, sorted(docs), None


def add_release_item(
    *,
    release: dict[str, Any],
    source_bundle: dict[str, Any],
    bundle_id: str,
    qa_id: str,
    item: dict[str, Any],
) -> None:
    if bundle_id not in release:
        release[bundle_id] = {
            key: deepcopy(value)
            for key, value in source_bundle.items()
            if key != "QA"
        }
        release[bundle_id]["QA"] = {}
    if qa_id in release[bundle_id]["QA"]:
        raise ValueError(f"Duplicate release item: {bundle_id}/{qa_id}")
    release[bundle_id]["QA"][qa_id] = item


def find_near_duplicate_questions(
    release: dict[str, Any],
    *,
    jaccard_threshold: float = 0.80,
    sequence_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    rows: list[tuple[str, str, str, str, set[str]]] = []
    for bundle_id, bundle in release.items():
        for qa_id, item in bundle["QA"].items():
            question = item["question"].strip()
            normalized = " ".join(
                re.findall(r"[a-z0-9]+", question.casefold())
            )
            rows.append(
                (
                    bundle_id,
                    qa_id,
                    question,
                    normalized,
                    set(normalized.split()),
                )
            )
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            union = left[4] | right[4]
            jaccard = len(left[4] & right[4]) / max(1, len(union))
            sequence = SequenceMatcher(None, left[3], right[3]).ratio()
            if jaccard >= jaccard_threshold or sequence >= sequence_threshold:
                pairs.append(
                    {
                        "left": {
                            "bundle_id": left[0],
                            "qa_id": left[1],
                            "question": left[2],
                        },
                        "right": {
                            "bundle_id": right[0],
                            "qa_id": right[1],
                            "question": right[2],
                        },
                        "token_jaccard": jaccard,
                        "sequence_ratio": sequence,
                    }
                )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--prior-qa", type=Path, default=PRIOR_QA)
    parser.add_argument(
        "--remaining-qa",
        type=Path,
        default=ROOT / "qa_cross_pdf_remaining_1941.json",
    )
    parser.add_argument(
        "--repair-qa",
        type=Path,
        default=ROOT / "qa_gemini_fixes_for_claude_rereview.json",
    )
    parser.add_argument(
        "--repair-reviews",
        type=Path,
        default=(
            ROOT
            / "gemini_fix_claude_rereviews"
            / "claude_claude-sonnet-5"
        ),
    )
    parser.add_argument(
        "--claude-fix-qa",
        type=Path,
        default=ROOT / "qa_claude_fixes_for_claude_rereview.json",
    )
    parser.add_argument(
        "--claude-fix-reviews",
        type=Path,
        default=(
            ROOT
            / "claude_fix_claude_rereviews"
            / "claude_claude-sonnet-5"
        ),
    )
    parser.add_argument(
        "--second-stage-fix-qa",
        type=Path,
        default=ROOT / "qa_second_stage_fixes_for_claude_rereview.json",
    )
    parser.add_argument(
        "--second-stage-fix-reviews",
        type=Path,
        default=(
            ROOT
            / "second_stage_fix_claude_rereviews"
            / "claude_claude-sonnet-5"
        ),
    )
    parser.add_argument("--output-qa", type=Path, default=OUTPUT_QA)
    parser.add_argument("--output-pdf-dir", type=Path, default=OUTPUT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--minimum-qa", type=int, default=350)
    args = parser.parse_args()

    manifest_rows = load_json(args.manifest)
    manifest = {row["id"]: row for row in manifest_rows}
    release: dict[str, Any] = {}
    ledger: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()

    direct_subsets = [
        (
            "prior_500",
            load_json(args.prior_qa),
            args.root / "api_reviews",
        ),
        (
            "remaining_1941",
            load_json(args.remaining_qa),
            args.root / "remaining_1941_api_reviews",
        ),
    ]
    for subset_name, qa_data, reviews_dir in direct_subsets:
        for bundle_id, bundle in qa_data.items():
            gemini = review_items(
                reviews_dir
                / "gemini_gemini-3-flash-preview"
                / f"{bundle_id}.json"
            )
            claude = review_items(
                reviews_dir / "claude_claude-sonnet-5" / f"{bundle_id}.json"
            )
            for qa_id, original in bundle["QA"].items():
                claude_item = claude[qa_id]
                gemini_item = gemini[qa_id]
                if claude_item["decision"] not in {"KEEP", "FIX"}:
                    exclusion_counts["direct_claude_reject"] += 1
                    continue
                if not all_criteria_true(claude_item):
                    exclusion_counts[
                        "direct_claude_accept_with_inconsistent_criteria"
                    ] += 1
                    continue
                if len(support_doc_numbers(claude_item)) < 2:
                    exclusion_counts["direct_claude_support_fewer_than_two_docs"] += 1
                    continue
                selected, changed = apply_review_fix(original, claude_item)
                valid, evidence_docs, structural_error = structurally_valid_item(
                    selected, manifest[bundle_id]["sources"]
                )
                if not valid:
                    exclusion_counts[f"direct_{structural_error}"] += 1
                    continue
                add_release_item(
                    release=release,
                    source_bundle=bundle,
                    bundle_id=bundle_id,
                    qa_id=qa_id,
                    item=selected,
                )
                ledger.append(
                    {
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "release_tier": "claude_direct_accept",
                        "source_subset": subset_name,
                        "applied_claude_fix_fields": changed,
                        "evidence_doc_numbers": evidence_docs,
                        "gemini_decision": gemini_item["decision"],
                        "gemini_confidence": gemini_item["confidence"],
                        "gemini_reason": gemini_item["reason"],
                        "claude_decision": claude_item["decision"],
                        "claude_confidence": claude_item["confidence"],
                        "claude_reason": claude_item["reason"],
                        "claude_support": claude_item["support"],
                        "released_item": selected,
                    }
                )

    repaired_qa = load_json(args.repair_qa)
    for bundle_id, bundle in repaired_qa.items():
        claude = review_items(args.repair_reviews / f"{bundle_id}.json")
        for qa_id, gemini_repaired in bundle["QA"].items():
            claude_item = claude[qa_id]
            if claude_item["decision"] not in {"KEEP", "FIX"}:
                exclusion_counts["repair_claude_reject"] += 1
                continue
            if not all_criteria_true(claude_item):
                exclusion_counts[
                    "repair_claude_accept_with_inconsistent_criteria"
                ] += 1
                continue
            if len(support_doc_numbers(claude_item)) < 2:
                exclusion_counts["repair_claude_support_fewer_than_two_docs"] += 1
                continue
            selected, changed = apply_review_fix(
                gemini_repaired, claude_item
            )
            valid, evidence_docs, structural_error = structurally_valid_item(
                selected, manifest[bundle_id]["sources"]
            )
            if not valid:
                exclusion_counts[f"repair_{structural_error}"] += 1
                continue
            add_release_item(
                release=release,
                source_bundle=bundle,
                bundle_id=bundle_id,
                qa_id=qa_id,
                item=selected,
            )
            ledger.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "release_tier": "gemini_repair_claude_rereview_accept",
                    "source_subset": "gemini_repair_rereview",
                    "applied_claude_fix_fields": changed,
                    "evidence_doc_numbers": evidence_docs,
                    "claude_decision": claude_item["decision"],
                    "claude_confidence": claude_item["confidence"],
                    "claude_reason": claude_item["reason"],
                    "claude_support": claude_item["support"],
                    "released_item": selected,
                }
            )

    claude_fixed_qa = load_json(args.claude_fix_qa)
    for bundle_id, bundle in claude_fixed_qa.items():
        claude = review_items(args.claude_fix_reviews / f"{bundle_id}.json")
        for qa_id, first_claude_repaired in bundle["QA"].items():
            claude_item = claude[qa_id]
            if claude_item["decision"] not in {"KEEP", "FIX"}:
                exclusion_counts["claude_fix_rereview_reject"] += 1
                continue
            if not all_criteria_true(claude_item):
                exclusion_counts[
                    "claude_fix_rereview_accept_with_inconsistent_criteria"
                ] += 1
                continue
            if len(support_doc_numbers(claude_item)) < 2:
                exclusion_counts[
                    "claude_fix_rereview_support_fewer_than_two_docs"
                ] += 1
                continue
            selected, changed = apply_review_fix(
                first_claude_repaired, claude_item
            )
            valid, evidence_docs, structural_error = structurally_valid_item(
                selected, manifest[bundle_id]["sources"]
            )
            if not valid:
                exclusion_counts[
                    f"claude_fix_rereview_{structural_error}"
                ] += 1
                continue
            add_release_item(
                release=release,
                source_bundle=bundle,
                bundle_id=bundle_id,
                qa_id=qa_id,
                item=selected,
            )
            ledger.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "release_tier": "claude_fix_claude_rereview_accept",
                    "source_subset": "claude_fix_rereview",
                    "applied_claude_fix_fields": changed,
                    "evidence_doc_numbers": evidence_docs,
                    "claude_decision": claude_item["decision"],
                    "claude_confidence": claude_item["confidence"],
                    "claude_reason": claude_item["reason"],
                    "claude_support": claude_item["support"],
                    "released_item": selected,
                }
            )

    second_stage_qa = load_json(args.second_stage_fix_qa)
    for bundle_id, bundle in second_stage_qa.items():
        claude = review_items(
            args.second_stage_fix_reviews / f"{bundle_id}.json"
        )
        for qa_id, twice_repaired in bundle["QA"].items():
            claude_item = claude[qa_id]
            if claude_item["decision"] not in {"KEEP", "FIX"}:
                exclusion_counts["second_stage_rereview_reject"] += 1
                continue
            if not all_criteria_true(claude_item):
                exclusion_counts[
                    "second_stage_rereview_accept_with_inconsistent_criteria"
                ] += 1
                continue
            if len(support_doc_numbers(claude_item)) < 2:
                exclusion_counts[
                    "second_stage_rereview_support_fewer_than_two_docs"
                ] += 1
                continue
            selected, changed = apply_review_fix(twice_repaired, claude_item)
            valid, evidence_docs, structural_error = structurally_valid_item(
                selected, manifest[bundle_id]["sources"]
            )
            if not valid:
                exclusion_counts[
                    f"second_stage_rereview_{structural_error}"
                ] += 1
                continue
            add_release_item(
                release=release,
                source_bundle=bundle,
                bundle_id=bundle_id,
                qa_id=qa_id,
                item=selected,
            )
            ledger.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "release_tier": "second_stage_fix_claude_rereview_accept",
                    "source_subset": "second_stage_fix_rereview",
                    "applied_claude_fix_fields": changed,
                    "evidence_doc_numbers": evidence_docs,
                    "claude_decision": claude_item["decision"],
                    "claude_confidence": claude_item["confidence"],
                    "claude_reason": claude_item["reason"],
                    "claude_support": claude_item["support"],
                    "released_item": selected,
                }
            )

    qa_count = sum(len(bundle["QA"]) for bundle in release.values())
    if qa_count < args.minimum_qa:
        raise ValueError(
            f"Only {qa_count} QA passed the full audit; "
            f"minimum is {args.minimum_qa}"
        )
    if qa_count != len(ledger):
        raise AssertionError("Release and ledger counts differ")

    normalized_questions: Counter[str] = Counter()
    for bundle in release.values():
        for item in bundle["QA"].values():
            normalized_questions[item["question"].strip().casefold()] += 1
    duplicates = [
        question for question, count in normalized_questions.items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate released questions: {duplicates}")
    near_duplicates = find_near_duplicate_questions(release)
    if near_duplicates:
        raise ValueError(
            "Near-duplicate released questions: "
            + json.dumps(near_duplicates[:20], ensure_ascii=False)
        )

    args.output_qa.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_pdf_dir.mkdir(parents=True, exist_ok=True)
    expected_pdf_names = {
        output_pdf_path(args.output_pdf_dir, bundle_id).name for bundle_id in release
    }

    pdf_hashes: dict[str, str] = {}
    for bundle_id in release:
        source = resolve_pdf_path(bundle_id, [SOURCE_PDF_DIR])
        target = output_pdf_path(args.output_pdf_dir, bundle_id)
        if not target.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        source_hash = sha256_file(source)
        target_hash = sha256_file(target)
        if source_hash != target_hash:
            raise ValueError(f"Copied PDF hash mismatch: {bundle_id}")
        pdf_hashes[bundle_id] = source_hash

    args.output_qa.write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_manifest = [manifest[bundle_id] for bundle_id in release]
    selected_manifest_path = args.output_dir / "selected_bundle_manifest.json"
    selected_manifest_path.write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    release_ledger_path = args.output_dir / "release_ledger.jsonl"
    with release_ledger_path.open("w", encoding="utf-8") as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tier_counts = Counter(row["release_tier"] for row in ledger)
    summary = {
        "selected_pdf_count": len(release),
        "selected_qa_count": qa_count,
        "minimum_required_qa": args.minimum_qa,
        "release_tier_counts": dict(tier_counts),
        "exclusion_counts": dict(exclusion_counts),
        "validation": {
            "all_released_items_claude_accepted": True,
            "all_released_review_criteria_true": True,
            "all_released_support_spans_multiple_source_papers": True,
            "all_released_evidence_spans_multiple_source_papers": True,
            "all_questions_and_answers_nonempty": True,
            "duplicate_questions": 0,
            "near_duplicate_questions": 0,
            "near_duplicate_thresholds": {
                "token_jaccard": 0.80,
                "sequence_ratio": 0.82,
            },
            "pdf_files": len(expected_pdf_names),
            "all_pdf_hashes_match_sources": True,
        },
        "outputs": {
            "qa_json": str(args.output_qa),
            "pdf_dir": str(args.output_pdf_dir),
            "selected_manifest": str(selected_manifest_path),
            "release_ledger": str(release_ledger_path),
        },
        "output_sha256": {
            "qa_json": sha256_file(args.output_qa),
            "selected_manifest": sha256_file(selected_manifest_path),
            "release_ledger": sha256_file(release_ledger_path),
        },
        "pdf_sha256": pdf_hashes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
