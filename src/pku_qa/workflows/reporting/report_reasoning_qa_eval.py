#!/usr/bin/env python3
"""Validate and report the strict 100-item reasoning-QA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_protocol import (
    INFERENCE_PROTOCOL_VERSION,
    SCORING_PROTOCOL_VERSION,
    STRICT_JUDGE_REQUIRED_QA_FIELDS,
    canonical_gold_answer,
    canonical_gold_pages,
    canonicalize_pdf_output_for_storage,
    judge_inference_binding_sha256,
    parse_canonical_pdf_output,
    pdf_corpus_sha256,
    resolve_pdf_path,
    validate_dataset_protocol,
)
from pypdf import PdfReader
from pku_qa.workflows.cleaning.restore_reasoning_qa_evidence import (
    restore_reasoning_qa_evidence,
    sha256_file as restore_sha256_file,
)


PROVIDERS = ("claude", "gemini")
MODELS = ("4B", "8B")
REASONING_TYPES = {
    "causal_chain",
    "conditional_inference",
    "comparison",
    "constraint_intersection",
    "multi_step_calculation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--pdf-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Directory containing <paper_id>.pdf. May be repeated. The "
            "report independently hashes and counts the actual PDFs."
        ),
    )
    parser.add_argument("--expected-pdf-corpus-sha256")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pdf_corpus(
    dataset: dict[str, Any], pdf_dirs: list[Path]
) -> dict[str, Any]:
    """Independently bind the benchmark to the actual PDF bytes and pages."""
    if not pdf_dirs:
        raise ValueError("At least one PDF directory is required")
    papers: dict[str, dict[str, Any]] = {}
    for paper_id in sorted(
        str(value) for value in dataset if not str(value).startswith("__")
    ):
        path = resolve_pdf_path(paper_id, pdf_dirs).resolve()
        try:
            total_pages = len(PdfReader(str(path)).pages)
        except Exception as exc:
            raise ValueError(f"Cannot inspect PDF {path}: {exc}") from exc
        if total_pages <= 0:
            raise ValueError(f"PDF has no pages: {path}")
        papers[paper_id] = {
            "path": str(path),
            "sha256": sha256(path),
            "total_pdf_pages": total_pages,
        }
    return {
        "pdf_corpus_sha256": pdf_corpus_sha256(dataset, pdf_dirs),
        "paper_count": len(papers),
        "papers": papers,
    }


def iter_qas(dataset: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Invalid paper record: {paper_id}")
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(f"Invalid QA record: {paper_id}/{qa_id}")
            yield str(paper_id), str(qa_id), qa


def validate_dataset(
    dataset: dict[str, Any],
    expected_count: int,
    min_confidence: float,
) -> dict[str, Any]:
    rows = list(iter_qas(dataset))
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected exactly {expected_count} QA, found {len(rows)}"
        )
    questions: set[str] = set()
    reasoning_types: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for paper_id, qa_id, qa in rows:
        question = " ".join(str(qa.get("question", "")).split()).casefold()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"Empty question/answer: {paper_id}/{qa_id}")
        if question in questions:
            raise ValueError(f"Duplicate question: {paper_id}/{qa_id}")
        questions.add(question)
        validation = qa.get("dual_model_validation")
        if not isinstance(validation, dict):
            raise ValueError(
                f"Missing dual_model_validation: {paper_id}/{qa_id}"
            )
        for provider in PROVIDERS:
            verdict = validation.get(provider)
            if not isinstance(verdict, dict):
                raise ValueError(
                    f"Missing {provider} validation: {paper_id}/{qa_id}"
                )
            if verdict.get("decision") != "KEEP":
                raise ValueError(
                    f"{provider} decision is not KEEP: {paper_id}/{qa_id}"
                )
            confidence = verdict.get("confidence")
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or confidence < min_confidence
            ):
                raise ValueError(
                    f"{provider} confidence below {min_confidence}: "
                    f"{paper_id}/{qa_id}"
                )
        provenance = qa.get("evidence_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"Missing evidence_provenance: {paper_id}/{qa_id}"
            )
        if provenance.get("method") != "sorted_union_of_source_qa_evidence_pages":
            raise ValueError(
                f"Invalid evidence provenance method: {paper_id}/{qa_id}"
            )
        validation_flags = provenance.get("validation")
        if not isinstance(validation_flags, dict) or not all(
            validation_flags.get(field) is True
            for field in (
                "all_source_qa_ids_resolved",
                "pages_are_positive_integers",
                "pages_are_sorted_and_unique",
                "pages_within_pdf_bounds",
            )
        ):
            raise ValueError(
                f"Evidence provenance validation failed: {paper_id}/{qa_id}"
            )
        reasoning_type = str(qa.get("reasoning_type", "unknown"))
        if reasoning_type not in REASONING_TYPES:
            raise ValueError(
                f"Invalid reasoning_type={reasoning_type!r}: {paper_id}/{qa_id}"
            )
        reasoning_types[reasoning_type] += 1
        categories[str(dataset[paper_id].get("primary_category", "unknown"))] += 1
    protocol_profile = validate_dataset_protocol(dataset, "pdf")
    return {
        "status": "passed",
        "question_count": len(rows),
        "unique_questions": len(questions),
        "paper_count": len(dataset),
        "minimum_review_confidence": min_confidence,
        "reasoning_type_distribution": dict(sorted(reasoning_types.items())),
        "primary_category_distribution": dict(sorted(categories.items())),
        "protocol_validation": protocol_profile,
    }


def audit_evidence_provenance(
    dataset: dict[str, Any], *, expected_count: int
) -> dict[str, Any]:
    """Rebuild all evidence from the recorded source QAs and real PDFs."""
    provenance_rows = [
        qa.get("evidence_provenance")
        for _, _, qa in iter_qas(dataset)
    ]
    source_paths = {
        str(row.get("source_dataset", ""))
        for row in provenance_rows
        if isinstance(row, dict)
    }
    pdf_paths = {
        str(row.get("source_pdf", ""))
        for row in provenance_rows
        if isinstance(row, dict)
    }
    if len(source_paths) != 1 or "" in source_paths:
        raise ValueError("Evidence provenance does not name one source dataset")
    source_path = _resolve_recorded_path(next(iter(source_paths)), ROOT)
    if not source_path.is_file():
        raise ValueError(f"Evidence source dataset is missing: {source_path}")
    resolved_pdfs = {_resolve_recorded_path(value, ROOT) for value in pdf_paths}
    if not resolved_pdfs or any(not path.is_file() for path in resolved_pdfs):
        raise ValueError("Evidence provenance references missing source PDFs")
    pdf_parents = {path.parent for path in resolved_pdfs}
    if len(pdf_parents) != 1:
        raise ValueError("Evidence source PDFs do not share one recorded directory")
    source_dataset = load_json(source_path)
    source_hash = restore_sha256_file(source_path)
    rebuilt, audit = restore_reasoning_qa_evidence(
        dataset,
        source_dataset,
        pdf_dir=next(iter(pdf_parents)),
        source_dataset_path=source_path,
        source_dataset_sha256=source_hash,
        expected_count=expected_count,
    )
    if rebuilt != dataset:
        raise ValueError(
            "Evidence provenance cannot reproduce the submitted benchmark "
            "byte-for-byte at the JSON object level"
        )
    return {**audit, "status": "passed", "deep_equal_rebuild": True}


def result_index(result: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (paper_id, qa_id): qa
        for paper_id, qa_id, qa in iter_qas(result)
    }


def _resolve_recorded_path(raw_path: Any, output_dir: Path) -> Path:
    path = Path(str(raw_path or ""))
    if path.is_absolute():
        return path.resolve()
    candidates = [ROOT / path, output_dir / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def validate_inference_manifest(
    output_dir: Path,
    model: str,
    *,
    source_sha256: str,
    pdf_corpus_hash: str,
) -> str:
    """Require the exact Full-PDF invocation, including max_pdf_pages=0."""
    path = (
        output_dir
        / ".adaptive_queue"
        / f"inference_{model}"
        / "manifest.json"
    )
    manifest = load_json(path)
    if not isinstance(manifest, dict) or int(manifest.get("version", 0)) < 2:
        raise ValueError(f"{model} inference manifest is missing/v1: {path}")
    metadata = manifest.get("metadata")
    config = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"{model} inference manifest has no strict config")
    expected = {
        "qa_source_sha256": source_sha256,
        "pdf_corpus_sha256": pdf_corpus_hash,
        "input_mode": "pdf",
        "page_input_policy": "full",
        "model": model,
        "prompt_style": "pdf",
        "pdf_mode": True,
        "max_pdf_pages": 0,
    }
    mismatches = {
        field: {"expected": value, "found": config.get(field)}
        for field, value in expected.items()
        if config.get(field) != value
    }
    if mismatches:
        raise ValueError(
            f"{model} inference manifest is not the required Full-PDF run: "
            f"{mismatches}"
        )
    if metadata.get("stage") != "inference" or metadata.get(
        "protocol_version"
    ) != INFERENCE_PROTOCOL_VERSION:
        raise ValueError(
            f"{model} inference manifest protocol is not "
            f"v{INFERENCE_PROTOCOL_VERSION}"
        )
    fingerprint = str(metadata.get("protocol_fingerprint", "")).strip()
    if not fingerprint:
        raise ValueError(f"{model} inference manifest has no fingerprint")
    return fingerprint


def validate_judge_manifest(
    output_dir: Path,
    model: str,
    *,
    source_sha256: str,
    pdf_corpus_hash: str,
    inference_fingerprint: str,
    inference_results: dict[str, Any],
) -> str:
    """Verify the Judge implementation/provider and per-QA cache binding."""
    path = output_dir / ".adaptive_queue" / f"judge_{model}" / "manifest.json"
    manifest = load_json(path)
    if not isinstance(manifest, dict) or int(manifest.get("version", 0)) < 2:
        raise ValueError(f"{model} Judge manifest is missing/v1: {path}")
    queue_source = {
        str(paper_id): paper
        for paper_id, paper in inference_results.items()
        if not str(paper_id).startswith("__") and isinstance(paper, dict)
    }
    encoded_source = json.dumps(
        queue_source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if manifest.get("source_sha256") != hashlib.sha256(encoded_source).hexdigest():
        raise ValueError(f"{model} Judge manifest source hash mismatch")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{model} Judge manifest has no metadata")
    expected_metadata = {
        "stage": "judge",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "gold_source_sha256": source_sha256,
        "inference_protocol_fingerprint": inference_fingerprint,
        "input_mode": "pdf",
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(
                f"{model} Judge manifest {field} mismatch"
            )
    for field in (
        "judge_runner_sha256",
        "source_code_sha256",
        "judge_provider_identity",
        "judge_config",
    ):
        if not metadata.get(field):
            raise ValueError(f"{model} Judge manifest lacks {field}")
    provider = metadata["judge_provider_identity"]
    config = metadata["judge_config"]
    if (
        not isinstance(provider, dict)
        or not isinstance(config, dict)
        or provider.get("provider_name")
        != config.get("judge_provider_name")
    ):
        raise ValueError(f"{model} Judge provider/config mismatch")
    fingerprint = str(metadata.get("judge_protocol_fingerprint", "")).strip()
    fingerprint_payload = dict(metadata)
    fingerprint_payload.pop("judge_protocol_fingerprint", None)
    recomputed = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not fingerprint or fingerprint != recomputed:
        raise ValueError(f"{model} Judge protocol fingerprint is invalid")
    contract = manifest.get("validation_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{model} Judge validation contract is missing")
    if tuple(contract.get("required_qa_fields", [])) != tuple(
        STRICT_JUDGE_REQUIRED_QA_FIELDS
    ):
        raise ValueError(f"{model} Judge required fields mismatch")
    global_values = contract.get("required_qa_field_values")
    expected_global = {
        "judge_protocol_fingerprint": fingerprint,
        "protocol_fingerprint": inference_fingerprint,
        "qa_source_sha256": source_sha256,
        "input_mode": "pdf",
        "pdf_corpus_sha256": pdf_corpus_hash,
    }
    if global_values != expected_global:
        raise ValueError(f"{model} Judge exact-value contract mismatch")
    per_qa = contract.get("required_qa_field_values_by_qa")
    expected_per_qa = {
        str(paper_id): {
            str(qa_id): {
                "inference_binding_sha256": (
                    judge_inference_binding_sha256(qa)
                )
            }
            for qa_id, qa in paper.get("QA", {}).items()
        }
        for paper_id, paper in queue_source.items()
    }
    if per_qa != expected_per_qa:
        raise ValueError(f"{model} Judge per-QA binding contract mismatch")
    return fingerprint


def validate_standard_report_binding(
    standard: dict[str, Any],
    output_dir: Path,
    *,
    source_sha256: str,
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind generic final_report.json to this gold and these Judge bytes."""
    if standard.get("gold_source_sha256") != source_sha256:
        raise ValueError("final_report.json gold_source_sha256 mismatch")
    if standard.get("publication_eligible") is not True:
        raise ValueError("final_report.json is not publication eligible")
    if standard.get("protocol_validation") != "publication_strict":
        raise ValueError("final_report.json protocol validation is not publication_strict")
    if standard.get("missing_files") not in ([], None):
        raise ValueError("final_report.json reports missing Judge files")
    profiles = standard.get("protocol_profiles")
    if not isinstance(profiles, list) or len(profiles) != len(MODELS):
        raise ValueError("final_report.json protocol_profiles are incomplete")
    profile_by_path: dict[Path, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict) or not profile.get("judge_file"):
            raise ValueError("final_report.json has malformed protocol profile")
        resolved = _resolve_recorded_path(profile["judge_file"], output_dir)
        if resolved in profile_by_path:
            raise ValueError("final_report.json repeats a Judge protocol profile")
        profile_by_path[resolved] = profile

    judge_hashes: dict[str, str] = {}
    for model in MODELS:
        judge_path = (output_dir / f"judge_{model}.json").resolve()
        row = next(
            (
                value
                for value in standard.get("models", [])
                if value.get("model_name") == model or value.get("model") == model
            ),
            None,
        )
        if not isinstance(row, dict):
            raise ValueError(f"final_report.json is missing {model}")
        if _resolve_recorded_path(row.get("judge_file"), output_dir) != judge_path:
            raise ValueError(f"final_report.json {model} Judge path mismatch")
        profile = profile_by_path.get(judge_path)
        if profile is None:
            raise ValueError(f"final_report.json lacks {model} Judge profile")
        summary = summaries[model]
        expected_profile = {
            "input_mode": "pdf",
            "total": summary["total"],
            "qa_source_sha256": source_sha256,
            "pdf_corpus_sha256": summary["pdf_corpus_sha256"],
            "protocol_fingerprint": summary["protocol_fingerprint"],
            "judge_protocol_fingerprint": summary[
                "judge_protocol_fingerprint"
            ],
            "evaluated_model": model,
            "inference_file_sha256": sha256(
                output_dir / f"results_{model}.json"
            ),
        }
        for field, expected in expected_profile.items():
            if profile.get(field) != expected:
                raise ValueError(
                    f"final_report.json {model} profile {field} mismatch"
                )
        expected_counts = {
            "total": summary["total"],
            "structured_required_total": summary["total"],
            "evidence_required_total": summary["total"],
            "evidence_required_legal_samples": summary["legal_outputs"],
            "answer_correct": summary["answer_correct"],
            "evidence_pages_correct": summary["evidence_correct"],
            "both_correct": summary["joint_correct"],
        }
        for field, expected in expected_counts.items():
            if int(row.get(field, -1)) != expected:
                raise ValueError(
                    f"final_report.json {model} {field} mismatch"
                )
        judge_hashes[model] = sha256(judge_path)
    return {
        "gold_source_sha256": source_sha256,
        "final_report_sha256": sha256(output_dir / "final_report.json"),
        "judge_file_sha256": judge_hashes,
    }


def summarize_model(
    model: str,
    gold_rows: list[tuple[str, str, dict[str, Any]]],
    inferred: dict[str, Any],
    judged: dict[str, Any],
    source_sha256: str,
    pdf_audit: dict[str, Any],
    manifest_fingerprint: str,
    judge_fingerprint: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inference_index = result_index(inferred)
    judge_index = result_index(judged)
    gold_keys = {(paper_id, qa_id) for paper_id, qa_id, _ in gold_rows}
    for label, index in (("inference", inference_index), ("Judge", judge_index)):
        if set(index) != gold_keys:
            missing = sorted(gold_keys - set(index))
            extra = sorted(set(index) - gold_keys)
            raise ValueError(
                f"{model} {label} keys differ from gold: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    errors: list[str] = []
    answer_correct = 0
    evidence_correct = 0
    joint_correct = 0
    legal_outputs = 0
    normalized_outputs = 0
    normalization_counts: Counter[str] = Counter()
    protocol_fingerprints: set[str] = set()
    pdf_corpus_hashes: set[str] = set()
    pdf_hashes_by_paper: dict[str, set[str]] = defaultdict(set)
    by_type: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for paper_id, qa_id, gold in gold_rows:
        key = (paper_id, qa_id)
        inference = inference_index[key]
        item = judge_index[key]
        item_id = f"{paper_id}/{qa_id}"
        paper_pdf = pdf_audit["papers"].get(paper_id)
        if not isinstance(paper_pdf, dict):
            errors.append(f"{item_id}:missing_independently_audited_pdf")
            continue
        total_pdf_pages = int(paper_pdf["total_pdf_pages"])
        expected_shown_pages = list(range(1, total_pdf_pages + 1))
        gold_answer = canonical_gold_answer(gold, item_id=item_id)
        gold_pages = canonical_gold_pages(
            gold.get("evidence_pages"), item_id=item_id
        )

        # Verify the original inference artifact rather than relying on fields
        # copied into Judge output.  This is the only place prompt style and
        # the complete page list are authoritatively recorded per item.
        inference_contract = {
            "evaluated_model": model,
            "question": gold.get("question"),
            "correct_answer": gold_answer,
            "input_mode": "pdf",
            "require_structured_output": True,
            "require_evidence_pages": True,
            "reference_evidence_pages": gold_pages,
            "page_input_policy": "full",
            "prompt_style": "pdf",
            "shown_pdf_pages": expected_shown_pages,
            "protocol_fingerprint": manifest_fingerprint,
            "qa_source_sha256": source_sha256,
            "pdf_corpus_sha256": pdf_audit["pdf_corpus_sha256"],
            "pdf_sha256": paper_pdf["sha256"],
        }
        mismatch = next(
            (
                field
                for field, expected in inference_contract.items()
                if inference.get(field) != expected
            ),
            None,
        )
        if mismatch:
            errors.append(f"{item_id}:inference_{mismatch}_mismatch")
            continue
        total_field_mismatch = False
        for optional_total_field in ("total_pdf_pages", "pdf_page_count"):
            if (
                optional_total_field in inference
                and inference[optional_total_field] != total_pdf_pages
            ):
                errors.append(
                    f"{item_id}:inference_{optional_total_field}_mismatch"
                )
                total_field_mismatch = True
                break
        if total_field_mismatch:
            continue
        if "max_pdf_pages" in inference and inference["max_pdf_pages"] != 0:
            errors.append(f"{item_id}:inference_max_pdf_pages_not_zero")
            continue
        exact_raw_output = str(inference.get("raw_model_output", ""))
        expected_raw_hash = hashlib.sha256(
            exact_raw_output.encode("utf-8")
        ).hexdigest()
        if inference.get("raw_model_output_sha256") != expected_raw_hash:
            errors.append(f"{item_id}:raw_model_output_hash_mismatch")
            continue
        try:
            canonical_output, actual_normalizations = (
                canonicalize_pdf_output_for_storage(
                    exact_raw_output,
                    allowed_pages=expected_shown_pages,
                )
            )
        except ValueError:
            actual_normalizations = []
            if (
                inference.get("model_output") != exact_raw_output
                or inference.get("deterministic_normalizations") != []
            ):
                errors.append(f"{item_id}:invalid_raw_output_was_altered")
                continue
        else:
            if inference.get("model_output") != canonical_output:
                errors.append(f"{item_id}:canonical_raw_output_mismatch")
                continue
            if (
                inference.get("deterministic_normalizations")
                != actual_normalizations
            ):
                errors.append(f"{item_id}:normalization_audit_mismatch")
                continue
        normalized_outputs += int(bool(actual_normalizations))
        normalization_counts.update(actual_normalizations)

        required_boolean_fields = (
            "answer_is_correct",
            "evidence_pages_is_correct",
            "is_correct",
            "output_is_legal",
        )
        if any(
            not isinstance(item.get(field), bool)
            for field in required_boolean_fields
        ):
            errors.append(f"{item_id}:missing_boolean_score_fields")
            continue
        if item.get("input_mode") != "pdf":
            errors.append(f"{item_id}:input_mode_not_pdf")
            continue
        if item.get("evaluated_model") != model:
            errors.append(f"{item_id}:evaluated_model_mismatch")
            continue
        if item.get("judge_protocol_fingerprint") != judge_fingerprint:
            errors.append(f"{item_id}:Judge_protocol_fingerprint_mismatch")
            continue
        if item.get("publication_eligible") is not True:
            errors.append(f"{item_id}:Judge_not_publication_eligible")
            continue
        if item.get("protocol_validation") != "publication_strict":
            errors.append(f"{item_id}:Judge_protocol_validation_mismatch")
            continue
        expected_binding = judge_inference_binding_sha256(inference)
        if item.get("inference_binding_sha256") != expected_binding:
            errors.append(f"{item_id}:stale_Judge_inference_binding")
            continue
        if item.get("require_structured_output") is not True:
            errors.append(f"{item_id}:structured_output_not_required")
            continue
        if item.get("require_evidence_pages") is not True:
            errors.append(f"{item_id}:evidence_pages_not_required")
            continue
        fingerprint = str(item.get("protocol_fingerprint", "")).strip()
        corpus_hash = str(item.get("pdf_corpus_sha256", "")).strip()
        pdf_hash = str(item.get("pdf_sha256", "")).strip()
        if (
            fingerprint != manifest_fingerprint
            or item.get("qa_source_sha256") != source_sha256
            or corpus_hash != pdf_audit["pdf_corpus_sha256"]
            or pdf_hash != paper_pdf["sha256"]
        ):
            errors.append(f"{item_id}:missing_or_mismatched_hash_lock")
            continue
        protocol_fingerprints.add(fingerprint)
        pdf_corpus_hashes.add(corpus_hash)
        pdf_hashes_by_paper[paper_id].add(pdf_hash)
        reference_pages = item.get("reference_evidence_pages")
        if reference_pages != gold_pages:
            errors.append(f"{item_id}:reference_evidence_mismatch")
            continue
        if str(item.get("question", "")) != str(gold.get("question", "")):
            errors.append(f"{item_id}:question_mismatch")
            continue
        if str(item.get("correct_answer", "")) != gold_answer:
            errors.append(f"{item_id}:reference_answer_mismatch")
            continue
        for field in ("page_input_policy", "shown_pdf_pages"):
            if item.get(field) != inference[field]:
                errors.append(f"{item_id}:Judge_{field}_mismatch")
                break
        else:
            if "prompt_style" in item and item["prompt_style"] != "pdf":
                errors.append(f"{item_id}:Judge_prompt_style_mismatch")
                continue
            if "max_pdf_pages" in item and item["max_pdf_pages"] != 0:
                errors.append(f"{item_id}:Judge_max_pdf_pages_not_zero")
                continue
        if errors and errors[-1].startswith(f"{item_id}:Judge_"):
            continue

        model_output = inference.get("model_output")
        if item.get("model_output") != model_output:
            errors.append(f"{item_id}:Judge_model_output_mismatch")
            continue
        for audit_field in (
            "raw_model_output",
            "raw_model_output_sha256",
            "deterministic_normalizations",
        ):
            if item.get(audit_field) != inference.get(audit_field):
                errors.append(
                    f"{item_id}:Judge_{audit_field}_mismatch"
                )
                break
        if errors and errors[-1].startswith(f"{item_id}:Judge_"):
            continue
        try:
            parsed_answer, predicted_pages = parse_canonical_pdf_output(
                model_output, allowed_pages=expected_shown_pages
            )
            legal = True
        except ValueError:
            parsed_answer, predicted_pages, legal = "", [], False
        independently_evidence_ok = legal and predicted_pages == gold_pages
        if item["output_is_legal"] != legal:
            errors.append(f"{item_id}:independent_legality_mismatch")
            continue
        if str(item.get("parsed_answer", "")) != parsed_answer:
            errors.append(f"{item_id}:independent_parsed_answer_mismatch")
            continue
        if item.get("predicted_evidence_pages", []) != predicted_pages:
            errors.append(f"{item_id}:independent_parsed_evidence_mismatch")
            continue
        if item["evidence_pages_is_correct"] != independently_evidence_ok:
            errors.append(f"{item_id}:independent_evidence_score_mismatch")
            continue
        if not legal and item["answer_is_correct"]:
            errors.append(f"{item_id}:illegal_output_marked_answer_correct")
            continue
        answer_ok = item["answer_is_correct"] and legal
        evidence_ok = independently_evidence_ok
        joint_ok = answer_ok and evidence_ok
        if item["is_correct"] != joint_ok:
            errors.append(f"{item_id}:inconsistent_joint_score")
            continue
        answer_correct += int(answer_ok)
        evidence_correct += int(evidence_ok)
        joint_correct += int(joint_ok)
        legal_outputs += int(legal)
        reasoning_type = str(gold.get("reasoning_type", "unknown"))
        by_type[reasoning_type].append((answer_ok, evidence_ok, joint_ok))
        rows.append(
            {
                "paper_id": paper_id,
                "qa_id": qa_id,
                "question": gold.get("question"),
                "reference_answer": gold_answer,
                "model_answer": parsed_answer if legal else model_output,
                # Reader-facing correctness always uses the publication
                # denominator: a format-illegal output is incorrect even if
                # a lower-level semantic flag happened to be true.
                "answer_is_correct": answer_ok,
                "evidence_pages_is_correct": evidence_ok,
                "is_correct": joint_ok,
                "output_is_legal": legal,
                "raw_model_output": exact_raw_output,
                "deterministic_normalizations": actual_normalizations,
                "raw_answer_judgment": answer_ok,
                "raw_evidence_judgment": evidence_ok,
                "reference_evidence_pages": reference_pages,
                "predicted_evidence_pages": item.get("predicted_evidence_pages"),
                "match_method": item.get("match_method"),
                "judge_verdict": item.get("judge_verdict"),
                "reasoning_type": reasoning_type,
                "shown_pdf_pages": expected_shown_pages,
                "total_pdf_pages": total_pdf_pages,
            }
        )
    if errors or len(rows) != len(gold_rows):
        raise ValueError(
            f"{model} artifacts failed closed: "
            f"invalid={errors[:5]}, valid={len(rows)}/{len(gold_rows)}"
        )
    if len(protocol_fingerprints) != 1 or len(pdf_corpus_hashes) != 1:
        raise ValueError(
            f"{model} mixed protocol/PDF corpus fingerprints"
        )
    if set(pdf_hashes_by_paper) != set(pdf_audit["papers"]) or any(
        len(values) != 1 for values in pdf_hashes_by_paper.values()
    ):
        raise ValueError(f"{model} has inconsistent per-paper PDF hashes")
    total = len(rows)
    return (
        {
            "model": model,
            "total": total,
            "legal_outputs": legal_outputs,
            "illegal_outputs": total - legal_outputs,
            "normalized_outputs": normalized_outputs,
            "normalization_counts": dict(sorted(normalization_counts.items())),
            "protocol_fingerprint": next(iter(protocol_fingerprints)),
            "judge_protocol_fingerprint": judge_fingerprint,
            "pdf_corpus_sha256": next(iter(pdf_corpus_hashes)),
            "per_paper_pdf_sha256": {
                paper_id: next(iter(values))
                for paper_id, values in sorted(pdf_hashes_by_paper.items())
            },
            "legal_output_rate": 100 * legal_outputs / total if total else 0.0,
            "answer_correct": answer_correct,
            "answer_accuracy": 100 * answer_correct / total if total else 0.0,
            "evidence_correct": evidence_correct,
            "evidence_accuracy": 100 * evidence_correct / total if total else 0.0,
            "joint_correct": joint_correct,
            "joint_accuracy": 100 * joint_correct / total if total else 0.0,
            "by_reasoning_type": {
                key: {
                    "total": len(values),
                    "answer_correct": sum(value[0] for value in values),
                    "answer_accuracy": 100 * sum(value[0] for value in values) / len(values),
                    "evidence_correct": sum(value[1] for value in values),
                    "evidence_accuracy": 100 * sum(value[1] for value in values) / len(values),
                    "joint_correct": sum(value[2] for value in values),
                    "joint_accuracy": 100 * sum(value[2] for value in values) / len(values),
                }
                for key, values in sorted(by_type.items())
            },
        },
        rows,
    )


def build_report(
    dataset: dict[str, Any],
    preflight: dict[str, Any],
    output_dir: Path,
    pdf_audit: dict[str, Any],
) -> dict[str, Any]:
    gold_rows = list(iter_qas(dataset))
    summaries: dict[str, Any] = {}
    model_rows: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        fingerprint = validate_inference_manifest(
            output_dir,
            model,
            source_sha256=str(preflight["source_sha256"]),
            pdf_corpus_hash=str(pdf_audit["pdf_corpus_sha256"]),
        )
        inferred = load_json(output_dir / f"results_{model}.json")
        judge_fingerprint = validate_judge_manifest(
            output_dir,
            model,
            source_sha256=str(preflight["source_sha256"]),
            pdf_corpus_hash=str(pdf_audit["pdf_corpus_sha256"]),
            inference_fingerprint=fingerprint,
            inference_results=inferred,
        )
        summary, rows = summarize_model(
            model,
            gold_rows,
            inferred,
            load_json(output_dir / f"judge_{model}.json"),
            str(preflight["source_sha256"]),
            pdf_audit,
            fingerprint,
            judge_fingerprint,
        )
        summaries[model] = summary
        model_rows[model] = rows
    if summaries["4B"]["pdf_corpus_sha256"] != summaries["8B"][
        "pdf_corpus_sha256"
    ]:
        raise ValueError("4B/8B PDF corpus SHA256 mismatch")
    if summaries["4B"]["per_paper_pdf_sha256"] != summaries["8B"][
        "per_paper_pdf_sha256"
    ]:
        raise ValueError("4B/8B per-paper PDF SHA256 mismatch")

    standard_binding = validate_standard_report_binding(
        load_json(output_dir / "final_report.json"),
        output_dir,
        source_sha256=str(preflight["source_sha256"]),
        summaries=summaries,
    )
    artifact_bindings = {
        **standard_binding,
        "pdf_corpus_sha256": pdf_audit["pdf_corpus_sha256"],
        "inference_file_sha256": {
            model: sha256(output_dir / f"results_{model}.json")
            for model in MODELS
        },
        "inference_manifest_sha256": {
            model: sha256(
                output_dir
                / ".adaptive_queue"
                / f"inference_{model}"
                / "manifest.json"
            )
            for model in MODELS
        },
        "judge_manifest_sha256": {
            model: sha256(
                output_dir
                / ".adaptive_queue"
                / f"judge_{model}"
                / "manifest.json"
            )
            for model in MODELS
        },
        "judge_protocol_fingerprint": {
            model: summaries[model]["judge_protocol_fingerprint"]
            for model in MODELS
        },
        "per_paper_pdf_sha256": {
            paper_id: value["sha256"]
            for paper_id, value in sorted(pdf_audit["papers"].items())
        },
    }
    paired_answer = Counter()
    paired_joint = Counter()
    paired_rows = []
    for row_4b, row_8b in zip(model_rows["4B"], model_rows["8B"]):
        if (row_4b["paper_id"], row_4b["qa_id"]) != (
            row_8b["paper_id"],
            row_8b["qa_id"],
        ):
            raise ValueError("4B/8B judged outputs are not aligned")
        answer_key = (
            "both_correct"
            if row_4b["answer_is_correct"] and row_8b["answer_is_correct"]
            else "only_4b_correct"
            if row_4b["answer_is_correct"]
            else "only_8b_correct"
            if row_8b["answer_is_correct"]
            else "both_incorrect"
        )
        joint_key = (
            "both_correct"
            if row_4b["is_correct"] and row_8b["is_correct"]
            else "only_4b_correct"
            if row_4b["is_correct"]
            else "only_8b_correct"
            if row_8b["is_correct"]
            else "both_incorrect"
        )
        paired_answer[answer_key] += 1
        paired_joint[joint_key] += 1
        paired_rows.append(
            {
                "paper_id": row_4b["paper_id"],
                "qa_id": row_4b["qa_id"],
                "4B_answer_correct": row_4b["answer_is_correct"],
                "8B_answer_correct": row_8b["answer_is_correct"],
                "4B_evidence_correct": row_4b["evidence_pages_is_correct"],
                "8B_evidence_correct": row_8b["evidence_pages_is_correct"],
                "4B_joint_correct": row_4b["is_correct"],
                "8B_joint_correct": row_8b["is_correct"],
            }
        )
    return {
        "title": "Strict Reasoning-QA 100: 4B vs 8B Evaluation",
        "publication_eligible": True,
        "protocol_validation": "publication_strict",
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "methodology": {
            "input": "full single-paper PDF plus synthesized reasoning question",
            "scoring": (
                "answer equivalence plus exact evidence-page-set match; "
                "rule-based answer matching is followed by Qwen3.6-27B "
                "semantic judgment only when needed"
            ),
            "denominators": (
                "answer and joint use all 100 items; evidence uses all 100 "
                "answerable evidence-applicable items; illegal output counts "
                "as incorrect"
            ),
            "output_normalization": (
                "raw model output is retained byte-for-byte; only answer "
                "outer whitespace and duplicate-free evidence-page list "
                "order may be deterministically canonicalized and are "
                "reported separately"
            ),
            "models": {
                "4B": "Qwen3-VL-4B-Instruct",
                "8B": "Qwen3-VL-8B-Instruct",
            },
        },
        "dataset_validation": preflight,
        "artifact_bindings": artifact_bindings,
        "models": summaries,
        "paired_outcomes": dict(sorted(paired_answer.items())),
        "paired_answer_outcomes": dict(sorted(paired_answer.items())),
        "paired_joint_outcomes": dict(sorted(paired_joint.items())),
        "incorrect_examples": {
            model: [row for row in rows if not row["is_correct"]][:5]
            for model, rows in model_rows.items()
        },
        "paired_rows": paired_rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    four = report["models"]["4B"]
    eight = report["models"]["8B"]
    paired_answer = report["paired_answer_outcomes"]
    paired_joint = report["paired_joint_outcomes"]
    lines = [
        "# 严格推理 QA 100 条：4B 与 8B 评测报告",
        "",
        "## 结论",
        "",
        f"- 4B：答案 {four['answer_correct']}/{four['total']} ({four['answer_accuracy']:.2f}%)；"
        f"证据页 {four['evidence_correct']}/{four['total']} ({four['evidence_accuracy']:.2f}%)；"
        f"联合 {four['joint_correct']}/{four['total']} ({four['joint_accuracy']:.2f}%)。",
        f"- 8B：答案 {eight['answer_correct']}/{eight['total']} ({eight['answer_accuracy']:.2f}%)；"
        f"证据页 {eight['evidence_correct']}/{eight['total']} ({eight['evidence_accuracy']:.2f}%)；"
        f"联合 {eight['joint_correct']}/{eight['total']} ({eight['joint_accuracy']:.2f}%)。",
        f"- 答案正确率差值（8B−4B）：{eight['answer_accuracy'] - four['answer_accuracy']:.2f} 个百分点。",
        "",
        "## 数据完整性",
        "",
        f"- 最终题目：{report['dataset_validation']['question_count']} 条，"
        f"覆盖 {report['dataset_validation']['paper_count']} 篇论文。",
        "- 每题均要求 Claude 与 Gemini 独立判定 KEEP，且双方置信度均不低于 0.8。",
        "- 题目、答案、双模型审核字段及证据页均通过完整性与唯一性校验。",
        f"- 4B 有 {four['normalized_outputs']} 条、8B 有 {eight['normalized_outputs']} 条发生可审计的表示层归一化；原始输出均保留，未增加或删除任何证据页。",
        "",
        "## 评测方法",
        "",
        "- 输入：完整单篇论文 PDF 与合成推理问题。",
        "- 被测模型：Qwen3-VL-4B-Instruct、Qwen3-VL-8B-Instruct。",
        "- 输出：严格 JSON `answer_pre + evidence_pages`；非法输出按错误计入全量分母。",
        "- 答案评分：先执行确定性匹配；无法直接判定时由 Qwen3.6-27B 语义判卷。",
        "- 证据评分：预测页集合与恢复后的金标页集合精确一致；联合正确要求答案、证据和格式均正确。",
        "",
        "## 两模型联合结果",
        "",
        f"- 答案：两者都正确 {paired_answer.get('both_correct', 0)}；仅4B {paired_answer.get('only_4b_correct', 0)}；仅8B {paired_answer.get('only_8b_correct', 0)}；都错误 {paired_answer.get('both_incorrect', 0)}。",
        f"- 联合：两者都正确 {paired_joint.get('both_correct', 0)}；仅4B {paired_joint.get('only_4b_correct', 0)}；仅8B {paired_joint.get('only_8b_correct', 0)}；都错误 {paired_joint.get('both_incorrect', 0)}。",
        "",
        "## 按推理类型",
        "",
        "| 推理类型 | 4B（答案 / 证据 / 联合） | 8B（答案 / 证据 / 联合） |",
        "|---|---:|---:|",
    ]
    types = sorted(
        set(four["by_reasoning_type"]) | set(eight["by_reasoning_type"])
    )
    for reasoning_type in types:
        left = four["by_reasoning_type"].get(reasoning_type)
        right = eight["by_reasoning_type"].get(reasoning_type)
        left_text = (
            f"{left['answer_accuracy']:.2f}% / {left['evidence_accuracy']:.2f}% / {left['joint_accuracy']:.2f}%"
            if left
            else "—"
        )
        right_text = (
            f"{right['answer_accuracy']:.2f}% / {right['evidence_accuracy']:.2f}% / {right['joint_accuracy']:.2f}%"
            if right
            else "—"
        )
        lines.append(f"| {reasoning_type} | {left_text} | {right_text} |")
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `technical_report.json`：完整结构化指标、逐题配对结果和错误样例。",
            "- `final_report.json` / `final_report.txt`：通用评测框架报告。",
            "- `judge_4B.json` / `judge_8B.json`：逐题判决。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    source_sha256 = sha256(args.qa_json)
    if args.expected_sha256 and source_sha256 != args.expected_sha256:
        raise SystemExit(
            "Reasoning-QA source SHA256 mismatch: "
            f"expected {args.expected_sha256}, found {source_sha256}"
        )
    dataset = load_json(args.qa_json)
    if not isinstance(dataset, dict):
        raise SystemExit("QA dataset must be a JSON object")
    preflight = validate_dataset(
        dataset, args.expected_count, args.min_confidence
    )
    preflight["evidence_provenance_rebuild"] = audit_evidence_provenance(
        dataset, expected_count=args.expected_count
    )
    preflight["source_sha256"] = source_sha256
    pdf_dirs = args.pdf_dir or [ROOT / "data/pdfs"]
    pdf_audit = audit_pdf_corpus(dataset, pdf_dirs)
    if (
        args.expected_pdf_corpus_sha256
        and pdf_audit["pdf_corpus_sha256"]
        != args.expected_pdf_corpus_sha256
    ):
        raise SystemExit(
            "Reasoning-QA PDF corpus SHA256 mismatch: expected "
            f"{args.expected_pdf_corpus_sha256}, found "
            f"{pdf_audit['pdf_corpus_sha256']}"
        )
    preflight["pdf_corpus_validation"] = pdf_audit
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    report = build_report(dataset, preflight, args.output_dir, pdf_audit)
    (args.output_dir / "technical_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "technical_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(report["models"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
