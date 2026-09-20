#!/usr/bin/env python3
"""Summarize closed-book, full-PDF, oracle-page, and ablation evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypdf import PdfReader

from eval_framework import looks_like_error_output
from evaluation_protocol import (
    INFERENCE_PROTOCOL_VERSION,
    JUDGE_INFERENCE_BINDING_FIELDS,
    SCORING_PROTOCOL_VERSION,
    canonical_gold_answer,
    canonical_gold_pages,
    canonicalize_pdf_output_for_storage,
    judge_inference_binding_sha256,
    parse_canonical_pdf_output,
    pdf_corpus_sha256,
    protocol_for_item,
    resolve_pdf_path,
)
from pku_qa.workflows.selection.single_pdf_release_views import (
    COMBINED_RUNTIME_INPUT,
    DEFAULT_RUNTIME_DIR,
    ORDINARY_VIEW,
)

DEFAULT_QA = COMBINED_RUNTIME_INPUT
DEFAULT_QUESTION_ONLY_QA = ORDINARY_VIEW
DEFAULT_ABLATION_QA = (
    DEFAULT_RUNTIME_DIR
    / "work__single_pdf__evidence_ablation__batch01__n676.json"
)
DEFAULT_OUTPUT = ROOT / "data/results/reports/single_pdf_1200"
DEFAULT_PDF_DIR = ROOT / "data/pdfs"
EVALUATION_MODES = {"question_only", "full", "oracle", "ablation"}
FIXED_DENOMINATORS = {
    "question_only": 1000,
    "full": 1200,
    "oracle": 1200,
    "ablation": 676,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument(
        "--question-only-qa-json",
        type=Path,
        default=DEFAULT_QUESTION_ONLY_QA,
        help="Independent 1000-row source used by question-only inference.",
    )
    parser.add_argument(
        "--ablation-qa-json",
        type=Path,
        default=DEFAULT_ABLATION_QA,
        help=(
            "Independent ablation source generated from the final ordinary QA."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-dir",
        action="append",
        type=Path,
        default=None,
        help=(
            "Directory containing <paper_id>.pdf; may be repeated. Full-PDF "
            "and empty-page Oracle rows are checked against the physical page "
            "count of these files."
        ),
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Repeat for each judged result JSON.",
    )
    parser.add_argument(
        "--max-illegal-rate",
        type=float,
        default=None,
        help="Fail if any structured evaluation exceeds this illegal-output fraction.",
    )
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
    """Bind page-set validation to the current physical PDF files."""
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
            raise ValueError(f"PDF has no physical pages: {path}")
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


def source_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return path.name


def iter_qas(dataset: dict[str, Any]):
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Malformed paper record: {paper_id}")
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(f"Malformed QA record: {paper_id}/{qa_id}")
            yield str(paper_id), str(qa_id), qa


def normalize(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_pages(raw: Any) -> set[int]:
    if not isinstance(raw, list):
        return set()
    result = set()
    for value in raw:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page >= 1:
            result.add(page)
    return result


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def parse_eval_specs(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--eval must be LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        parsed[label.strip()] = path
    return parsed


def gold_index(dataset: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (paper_id, qa_id): qa
        for paper_id, qa_id, qa in iter_qas(dataset)
    }


def normalized_queue_source_sha256(dataset: dict[str, Any]) -> str:
    source = {
        str(paper_id): paper
        for paper_id, paper in dataset.items()
        if not str(paper_id).startswith("__") and isinstance(paper, dict)
    }
    payload = json.dumps(
        source,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_current_code_hashes(
    metadata: dict[str, Any], *, artifact: str
) -> None:
    hashes = metadata.get("source_code_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{artifact} manifest lacks source-code hashes")
    for name, expected in hashes.items():
        path = ROOT / "src/pku_qa/evaluation" / str(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(
                f"{artifact} source-code hash is not current: {name}"
            )


def _load_strict_queue_manifest(
    path: Path,
    *,
    source_dataset: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    manifest = load_json(path)
    if not isinstance(manifest, dict) or int(manifest.get("version", 0)) < 2:
        raise ValueError(f"{stage} queue manifest is missing or pre-v2: {path}")
    expected_papers = {
        str(paper_id)
        for paper_id in source_dataset
        if not str(paper_id).startswith("__")
    }
    recorded_papers = manifest.get("papers")
    if not isinstance(recorded_papers, dict) or set(
        map(str, recorded_papers.values())
    ) != expected_papers:
        raise ValueError(f"{stage} queue manifest paper set mismatch: {path}")
    if manifest.get("source_sha256") != normalized_queue_source_sha256(
        source_dataset
    ):
        raise ValueError(f"{stage} queue manifest source hash mismatch: {path}")
    contract = manifest.get("validation_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{stage} queue manifest lacks validation contract: {path}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("stage") != stage:
        raise ValueError(f"{stage} queue manifest metadata mismatch: {path}")
    return manifest


def _validate_source_against_report_gold(
    source_dataset: dict[str, Any],
    report_gold: dict[tuple[str, str], dict[str, Any]],
    *,
    mode: str,
) -> None:
    source = gold_index(source_dataset)
    expected = (
        set(source)
        if mode == "ablation"
        else expected_non_ablation_keys(report_gold, mode)
    )
    if set(source) != expected:
        raise ValueError(f"{mode} source QA keys differ from report gold")
    if mode == "ablation":
        return
    for key, qa in source.items():
        gold = report_gold[key]
        item_id = f"{key[0]}/{key[1]}"
        if (
            str(qa.get("question", "")) != str(gold.get("question", ""))
            or canonical_gold_answer(qa, item_id=item_id)
            != canonical_gold_answer(gold, item_id=item_id)
            or canonical_gold_pages(qa.get("evidence_pages"), item_id=item_id)
            != canonical_gold_pages(gold.get("evidence_pages"), item_id=item_id)
        ):
            raise ValueError(f"{mode} source content differs from report gold: {item_id}")


def validate_eval_provenance(
    *,
    label: str,
    judge_path: Path,
    source_path: Path,
    source_dataset: dict[str, Any],
    report_gold: dict[tuple[str, str], dict[str, Any]],
    pdf_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed unless Judge, inference, queue and current inputs agree."""
    mode = evaluation_mode(label)
    model = label.rpartition("_")[2]
    _validate_source_against_report_gold(
        source_dataset, report_gold, mode=mode
    )
    output_dir = judge_path.resolve().parent
    inference_path = output_dir / f"results_{model}.json"
    inference_manifest_path = (
        output_dir / ".adaptive_queue" / f"inference_{model}" / "manifest.json"
    )
    judge_manifest_path = (
        output_dir / ".adaptive_queue" / f"judge_{model}" / "manifest.json"
    )
    inference_manifest = _load_strict_queue_manifest(
        inference_manifest_path,
        source_dataset=source_dataset,
        stage="inference",
    )
    judge_manifest = _load_strict_queue_manifest(
        judge_manifest_path,
        source_dataset=load_json(inference_path),
        stage="judge",
    )
    source_hash = sha256(source_path)
    expected_input_mode = "question_only" if mode == "question_only" else "pdf"
    expected_policy = "full" if mode in {"question_only", "full"} else "qa_field"
    expected_page_field = (
        "oracle_pages" if mode == "oracle" else "input_pages"
    )
    expected_prompt = "question_only" if mode == "question_only" else "pdf"
    expected_pdf_corpus = (
        None if expected_input_mode == "question_only" else pdf_audit["pdf_corpus_sha256"]
    )

    inference_metadata = inference_manifest["metadata"]
    if inference_metadata.get("protocol_version") != INFERENCE_PROTOCOL_VERSION:
        raise ValueError(
            f"{label} inference protocol is not "
            f"v{INFERENCE_PROTOCOL_VERSION}"
        )
    inference_fingerprint = str(
        inference_metadata.get("protocol_fingerprint", "")
    ).strip()
    if not inference_fingerprint:
        raise ValueError(f"{label} inference manifest lacks fingerprint")
    inference_payload = {
        key: value
        for key, value in inference_metadata.items()
        if key not in {"stage", "protocol_fingerprint"}
    }
    if _fingerprint_payload_sha256(inference_payload) != inference_fingerprint:
        raise ValueError(f"{label} inference fingerprint is internally invalid")
    _validate_current_code_hashes(inference_metadata, artifact=f"{label} inference")
    config = inference_metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} inference manifest lacks strict config")
    expected_config = {
        "qa_source_sha256": source_hash,
        "pdf_corpus_sha256": expected_pdf_corpus,
        "input_mode": expected_input_mode,
        "page_input_policy": expected_policy,
        "qa_page_field": expected_page_field,
        "model": model,
        "prompt_style": expected_prompt,
        "pdf_mode": mode != "question_only",
        "max_pdf_pages": 0,
    }
    mismatches = {
        field: {"expected": expected, "found": config.get(field)}
        for field, expected in expected_config.items()
        if config.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"{label} inference config mismatch: {mismatches}")
    inference_contract = inference_manifest["validation_contract"]
    required_values = inference_contract.get("required_qa_field_values")
    expected_values = {
        "protocol_fingerprint": inference_fingerprint,
        "qa_source_sha256": source_hash,
        "pdf_corpus_sha256": expected_pdf_corpus,
        "input_mode": expected_input_mode,
        "page_input_policy": expected_policy,
        "prompt_style": expected_prompt,
        "max_pdf_pages": 0,
    }
    if not isinstance(required_values, dict) or any(
        required_values.get(field) != expected
        for field, expected in expected_values.items()
    ):
        raise ValueError(f"{label} inference queue value contract mismatch")

    judge_metadata = judge_manifest["metadata"]
    if judge_metadata.get("scoring_protocol_version") != SCORING_PROTOCOL_VERSION:
        raise ValueError(
            f"{label} Judge scoring protocol is not "
            f"v{SCORING_PROTOCOL_VERSION}"
        )
    judge_fingerprint = str(
        judge_metadata.get("judge_protocol_fingerprint", "")
    ).strip()
    judge_payload = {
        key: value
        for key, value in judge_metadata.items()
        if key != "judge_protocol_fingerprint"
    }
    if (
        not judge_fingerprint
        or _fingerprint_payload_sha256(judge_payload) != judge_fingerprint
    ):
        raise ValueError(f"{label} Judge fingerprint is internally invalid")
    _validate_current_code_hashes(judge_metadata, artifact=f"{label} Judge")
    expected_judge_metadata = {
        "gold_source_sha256": source_hash,
        "inference_protocol_fingerprint": inference_fingerprint,
        "input_mode": expected_input_mode,
    }
    if any(
        judge_metadata.get(field) != expected
        for field, expected in expected_judge_metadata.items()
    ):
        raise ValueError(f"{label} Judge manifest provenance mismatch")
    judge_values = judge_manifest["validation_contract"].get(
        "required_qa_field_values"
    )
    expected_judge_values = {
        "judge_protocol_fingerprint": judge_fingerprint,
        "protocol_fingerprint": inference_fingerprint,
        "qa_source_sha256": source_hash,
        "input_mode": expected_input_mode,
        "pdf_corpus_sha256": expected_pdf_corpus,
    }
    if not isinstance(judge_values, dict) or any(
        judge_values.get(field) != expected
        for field, expected in expected_judge_values.items()
    ):
        raise ValueError(f"{label} Judge queue value contract mismatch")

    inferred = load_json(inference_path)
    judged = load_json(judge_path)
    expected_inference_by_paper = {
        str(paper_id): {
            "pdf_sha256": pdf_hash
        }
        for paper_id, pdf_hash in (
            {
                paper_id: None for paper_id in source_dataset
                if not str(paper_id).startswith("__")
            }
            if expected_input_mode == "question_only"
            else {
                paper_id: record["sha256"]
                for paper_id, record in pdf_audit["papers"].items()
            }
        ).items()
    }
    if inference_contract.get(
        "required_qa_field_values_by_paper"
    ) != expected_inference_by_paper:
        raise ValueError(f"{label} inference per-paper PDF lock mismatch")
    source_index = gold_index(source_dataset)
    inference_index = gold_index(inferred)
    judge_index = gold_index(judged)
    if set(inference_index) != set(source_index) or set(judge_index) != set(
        source_index
    ):
        raise ValueError(f"{label} artifact QA keys differ from source")
    pdf_hash_by_paper = (
        {}
        if expected_input_mode == "question_only"
        else {
            paper_id: str(record["sha256"])
            for paper_id, record in pdf_audit["papers"].items()
        }
    )
    for key, source_qa in source_index.items():
        paper_id, qa_id = key
        item_id = f"{paper_id}/{qa_id}"
        inference = inference_index[key]
        judge = judge_index[key]
        answer = canonical_gold_answer(source_qa, item_id=item_id)
        expected_row = {
            "question": source_qa.get("question"),
            "correct_answer": answer,
            "reference_evidence_pages": canonical_gold_pages(
                source_qa.get("evidence_pages"), item_id=item_id
            ),
            "input_mode": expected_input_mode,
            "require_structured_output": expected_input_mode == "pdf",
            "require_evidence_pages": (
                expected_input_mode == "pdf" and answer != "Unanswerable"
            ),
            "page_input_policy": expected_policy,
            "prompt_style": expected_prompt,
            "protocol_fingerprint": inference_fingerprint,
            "qa_source_sha256": source_hash,
            "pdf_corpus_sha256": expected_pdf_corpus,
            "pdf_sha256": pdf_hash_by_paper.get(paper_id),
            "max_pdf_pages": 0,
        }
        mismatch = next(
            (
                field
                for field, expected in expected_row.items()
                if inference.get(field) != expected
            ),
            None,
        )
        if mismatch:
            raise ValueError(f"{label} inference row {mismatch} mismatch: {item_id}")
        if expected_input_mode == "pdf":
            total_pages = int(pdf_audit["papers"][paper_id]["total_pdf_pages"])
            if inference.get("total_pdf_pages") != total_pages:
                raise ValueError(f"{label} inference physical page count mismatch: {item_id}")
        elif inference.get("total_pdf_pages") is not None:
            raise ValueError(f"{label} question-only row records PDF pages: {item_id}")
        binding_mismatch = next(
            (
                field
                for field in JUDGE_INFERENCE_BINDING_FIELDS
                if judge.get(field) != inference.get(field)
            ),
            None,
        )
        if binding_mismatch:
            raise ValueError(
                f"{label} Judge/inference {binding_mismatch} mismatch: {item_id}"
            )
        if judge.get("inference_binding_sha256") != judge_inference_binding_sha256(
            inference
        ):
            raise ValueError(f"{label} Judge inference binding hash mismatch: {item_id}")
        if judge.get("judge_protocol_fingerprint") != judge_fingerprint:
            raise ValueError(f"{label} Judge row fingerprint mismatch: {item_id}")
    expected_judge_by_qa = {
        paper_id: {
            qa_id: {
                "inference_binding_sha256": judge_inference_binding_sha256(
                    inference_index[(paper_id, qa_id)]
                )
            }
            for qa_id in source_dataset[paper_id]["QA"]
        }
        for paper_id in source_dataset
        if not str(paper_id).startswith("__")
    }
    if judge_manifest["validation_contract"].get(
        "required_qa_field_values_by_qa"
    ) != expected_judge_by_qa:
        raise ValueError(f"{label} Judge per-QA inference lock mismatch")
    return {
        "source_sha256": source_hash,
        "pdf_corpus_sha256": expected_pdf_corpus,
        "inference_protocol_fingerprint": inference_fingerprint,
        "judge_protocol_fingerprint": judge_fingerprint,
        "inference_sha256": sha256(inference_path),
        "judge_sha256": sha256(judge_path),
        "inference_manifest_sha256": sha256(inference_manifest_path),
        "judge_manifest_sha256": sha256(judge_manifest_path),
    }


def evaluation_mode(label: str) -> str:
    mode, separator, model = label.rpartition("_")
    if not separator or mode not in EVALUATION_MODES or model not in {"4B", "8B"}:
        raise ValueError(
            f"Evaluation label must be MODE_4B or MODE_8B with MODE in "
            f"{sorted(EVALUATION_MODES)}, got {label!r}"
        )
    return mode


def expected_non_ablation_keys(
    gold: dict[tuple[str, str], dict[str, Any]], mode: str
) -> set[tuple[str, str]]:
    if mode not in EVALUATION_MODES - {"ablation"}:
        raise ValueError(f"Invalid non-ablation mode: {mode!r}")
    if mode != "question_only":
        return set(gold)
    return {
        key
        for key, qa in gold.items()
        if canonical_gold_answer(qa, item_id=f"{key[0]}/{key[1]}")
        != "Unanswerable"
    }


def build_ablation_expectations(
    dataset: dict[str, Any],
    gold: dict[tuple[str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Bind every ablation variant to one base-gold row and its shown pages."""
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for paper_id, variant_id, variant in iter_qas(dataset):
        item_id = f"{paper_id}/{variant_id}"
        metadata = variant.get("ablation")
        if not isinstance(metadata, dict):
            raise ValueError(f"Ablation row lacks metadata: {item_id}")
        source_qa_id = str(metadata.get("source_qa_id", "")).strip()
        removed_page = metadata.get("removed_physical_pdf_page")
        if (
            not source_qa_id
            or isinstance(removed_page, bool)
            or not isinstance(removed_page, int)
            or removed_page <= 0
        ):
            raise ValueError(f"Invalid ablation source/removal metadata: {item_id}")
        if variant_id != f"{source_qa_id}__drop_p{removed_page}":
            raise ValueError(f"Ablation variant id/metadata mismatch: {item_id}")
        base_key = (paper_id, source_qa_id)
        base = gold.get(base_key)
        if base is None:
            raise ValueError(f"Ablation base gold is missing: {item_id}")
        if str(variant.get("question", "")) != str(base.get("question", "")):
            raise ValueError(f"Ablation/base question mismatch: {item_id}")
        if canonical_gold_answer(variant, item_id=item_id) != canonical_gold_answer(
            base, item_id=f"{paper_id}/{source_qa_id}"
        ):
            raise ValueError(f"Ablation/base answer mismatch: {item_id}")
        base_pages = canonical_gold_pages(
            base.get("evidence_pages"), item_id=f"{paper_id}/{source_qa_id}"
        )
        variant_pages = canonical_gold_pages(
            variant.get("evidence_pages"), item_id=item_id
        )
        original_pages = canonical_gold_pages(
            metadata.get("original_evidence_pages"), item_id=item_id
        )
        shown_pages = canonical_gold_pages(
            variant.get("input_pages"), item_id=item_id
        )
        if variant_pages != base_pages or original_pages != base_pages:
            raise ValueError(f"Ablation/base evidence mismatch: {item_id}")
        if removed_page not in base_pages:
            raise ValueError(f"Ablation removes a non-evidence page: {item_id}")
        expected_shown = [page for page in base_pages if page != removed_page]
        if shown_pages != expected_shown:
            raise ValueError(f"Ablation shown pages mismatch: {item_id}")
        key = (paper_id, variant_id)
        if key in expected:
            raise ValueError(f"Duplicate ablation variant: {item_id}")
        expected[key] = {
            "source": base,
            "variant": variant,
            "shown_pages": shown_pages,
        }
    if not expected:
        raise ValueError("Ablation source contains no variants")
    return expected


def classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        gold_unanswerable = row["gold_unanswerable"]
        predicted_unanswerable = row["predicted_unanswerable"]
        if gold_unanswerable and predicted_unanswerable:
            counts["unanswerable_tp"] += 1
        elif gold_unanswerable:
            counts["unanswerable_fn"] += 1
        elif predicted_unanswerable:
            counts["unanswerable_fp"] += 1
        else:
            counts["unanswerable_tn"] += 1
    return dict(counts)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answerable = [row for row in rows if not row["gold_unanswerable"]]
    unanswerable = [row for row in rows if row["gold_unanswerable"]]
    counts = classification_counts(rows)
    tp = counts.get("unanswerable_tp", 0)
    fn = counts.get("unanswerable_fn", 0)
    fp = counts.get("unanswerable_fp", 0)
    tn = counts.get("unanswerable_tn", 0)
    u_precision = safe_div(tp, tp + fp)
    u_recall = safe_div(tp, tp + fn)
    u_f1 = f1(u_precision, u_recall)
    a_precision = safe_div(tn, tn + fn)
    a_recall = safe_div(tn, tn + fp)
    a_f1 = f1(a_precision, a_recall)

    evidence_rows = [row for row in rows if row["reference_pages"]]
    structured_rows = [row for row in rows if row["structured_required"]]
    illegal_rows = [row for row in structured_rows if not row["legal_output"]]
    evidence_tp = sum(
        len(row["reference_pages"] & row["predicted_pages"])
        for row in evidence_rows
    )
    evidence_fp = sum(
        len(row["predicted_pages"] - row["reference_pages"])
        for row in evidence_rows
    )
    evidence_fn = sum(
        len(row["reference_pages"] - row["predicted_pages"])
        for row in evidence_rows
    )
    evidence_precision = safe_div(evidence_tp, evidence_tp + evidence_fp)
    evidence_recall = safe_div(evidence_tp, evidence_tp + evidence_fn)

    return {
        "total_seen": total,
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "structured_output_count": len(structured_rows),
        "illegal_output_count": len(illegal_rows),
        "illegal_output_rate": safe_div(len(illegal_rows), len(structured_rows)),
        "answer_correct": sum(row["answer_correct"] for row in rows),
        "answer_accuracy": safe_div(
            sum(row["answer_correct"] for row in rows), total
        ),
        "answerable_accuracy": safe_div(
            sum(row["answer_correct"] for row in answerable), len(answerable)
        ),
        "unanswerable_recall": u_recall,
        "false_refusal_rate": safe_div(fp, len(answerable)),
        "unanswerable_precision": u_precision,
        "unanswerable_f1": u_f1,
        "answerability_macro_f1": (u_f1 + a_f1) / 2,
        "evidence_exact_match": safe_div(
            sum(
                row["predicted_pages"] == row["reference_pages"]
                for row in evidence_rows
            ),
            len(evidence_rows),
        ),
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": f1(evidence_precision, evidence_recall),
        "confusion": counts,
    }


def eval_rows(
    result: dict[str, Any],
    gold: dict[tuple[str, str], dict[str, Any]],
    *,
    mode: str,
    ablation_expectations: dict[tuple[str, str], dict[str, Any]] | None = None,
    pdf_audit: dict[str, Any] | None = None,
    expected_count: int | None = None,
    require_publication_provenance: bool = False,
) -> list[dict[str, Any]]:
    """Reconcile one Judge artifact against independent gold, then aggregate.

    Publication metrics never trust Judge parsing fields.  PDF output is
    reparsed with the central canonical parser and constrained to pages that
    were actually shown to the model.  The stored fields must agree with that
    independent parse exactly.
    """
    if mode not in EVALUATION_MODES:
        raise ValueError(f"Unknown evaluation mode: {mode!r}")
    result_index = {
        (paper_id, qa_id): judged
        for paper_id, qa_id, judged in iter_qas(result)
    }
    if mode == "ablation":
        if not ablation_expectations:
            raise ValueError("Ablation evaluation requires independent variants")
        expected_keys = set(ablation_expectations)
    else:
        expected_keys = expected_non_ablation_keys(gold, mode)
    fixed_count = FIXED_DENOMINATORS[mode] if expected_count is None else expected_count
    if len(expected_keys) != fixed_count:
        raise ValueError(
            f"{mode} fixed denominator mismatch: "
            f"expected={fixed_count} gold={len(expected_keys)}"
        )
    if set(result_index) != expected_keys:
        missing = sorted(expected_keys - set(result_index))
        extra = sorted(set(result_index) - expected_keys)
        raise ValueError(
            f"{mode} Judge keys differ from the fixed evaluation set: "
            f"missing={missing[:5]} extra={extra[:5]} "
            f"judged={len(result_index)} expected={len(expected_keys)}"
        )

    rows: list[dict[str, Any]] = []
    for paper_id, qa_id in sorted(expected_keys):
        judged = result_index[(paper_id, qa_id)]
        if mode == "ablation":
            assert ablation_expectations is not None
            expectation = ablation_expectations[(paper_id, qa_id)]
            source = expectation["source"]
            expected_shown_pages = expectation["shown_pages"]
        else:
            source = gold.get((paper_id, qa_id))
            expected_shown_pages = None
        if source is None:
            raise ValueError(f"Missing independent gold: {paper_id}/{qa_id}")
        item_id = f"{paper_id}/{qa_id}"
        question = str(source.get("question", ""))
        answer = canonical_gold_answer(source, item_id=item_id)
        reference_pages_list = canonical_gold_pages(
            source.get("evidence_pages", []), item_id=item_id
        )
        if str(judged.get("question", "")) != question:
            raise ValueError(f"Judge/gold question mismatch: {item_id}")
        if str(judged.get("correct_answer", "")) != answer:
            raise ValueError(f"Judge/gold answer mismatch: {item_id}")
        stored_reference_pages = canonical_gold_pages(
            judged.get("reference_evidence_pages"), item_id=item_id
        )
        if stored_reference_pages != reference_pages_list:
            raise ValueError(f"Judge/gold evidence mismatch: {item_id}")

        expected_input_mode = "question_only" if mode == "question_only" else "pdf"
        if judged.get("input_mode") != expected_input_mode:
            raise ValueError(f"Judge input_mode mismatch: {item_id}")
        expected_policy = "full" if mode in {"question_only", "full"} else "qa_field"
        if judged.get("page_input_policy") != expected_policy:
            raise ValueError(f"Judge page_input_policy mismatch: {item_id}")
        protocol = protocol_for_item(expected_input_mode, answer)
        if judged.get("require_structured_output") is not protocol.require_structured_output:
            raise ValueError(f"Judge structured-output protocol mismatch: {item_id}")
        if judged.get("require_evidence_pages") is not protocol.require_evidence_pages:
            raise ValueError(f"Judge evidence protocol mismatch: {item_id}")

        shown_pages = canonical_gold_pages(
            judged.get("shown_pdf_pages"), item_id=f"{item_id}:shown_pdf_pages"
        )
        paper_pdf = (
            pdf_audit.get("papers", {}).get(paper_id)
            if isinstance(pdf_audit, dict)
            else None
        )
        physical_full_pages: list[int] | None = None
        if mode in {"full", "oracle"}:
            if not isinstance(paper_pdf, dict):
                raise ValueError(
                    f"{mode} lacks an independently audited PDF: {item_id}"
                )
            total_pdf_pages = paper_pdf.get("total_pdf_pages")
            if (
                isinstance(total_pdf_pages, bool)
                or not isinstance(total_pdf_pages, int)
                or total_pdf_pages <= 0
            ):
                raise ValueError(f"Invalid physical PDF page count: {item_id}")
            physical_full_pages = list(range(1, total_pdf_pages + 1))
        if mode == "question_only":
            if shown_pages:
                raise ValueError(f"Question-only row exposes PDF pages: {item_id}")
        elif mode == "full":
            assert physical_full_pages is not None
            if shown_pages != physical_full_pages:
                raise ValueError(
                    f"Full-PDF shown pages do not equal all physical PDF pages: "
                    f"{item_id}; expected=1..{len(physical_full_pages)}"
                )
        elif mode == "oracle":
            assert physical_full_pages is not None
            oracle_pages = canonical_gold_pages(
                source.get("oracle_pages", source.get("evidence_pages", [])),
                item_id=f"{item_id}:oracle_pages",
            )
            if answer == "Unanswerable":
                if reference_pages_list or oracle_pages:
                    raise ValueError(
                        f"Unanswerable Oracle gold must have no evidence pages: {item_id}"
                    )
                expected_oracle_pages = physical_full_pages
            else:
                if not oracle_pages or oracle_pages != reference_pages_list:
                    raise ValueError(
                        f"Answerable Oracle gold oracle/evidence pages differ: {item_id}"
                    )
                expected_oracle_pages = oracle_pages
            if shown_pages != expected_oracle_pages:
                raise ValueError(f"Oracle shown pages mismatch: {item_id}")
        else:
            assert expected_shown_pages is not None
            if shown_pages != expected_shown_pages:
                raise ValueError(f"Ablation shown pages mismatch: {item_id}")

        raw_output = judged.get("model_output")
        if not isinstance(raw_output, str):
            raise ValueError(f"Judge model_output must be a string: {item_id}")
        if require_publication_provenance:
            if judged.get("publication_eligible") is not True:
                raise ValueError(
                    f"Judge row is not publication eligible: {item_id}"
                )
            if judged.get("protocol_validation") != "publication_strict":
                raise ValueError(f"Judge protocol marker mismatch: {item_id}")
            exact_raw_output = judged.get("raw_model_output")
            if not isinstance(exact_raw_output, str):
                raise ValueError(
                    f"Judge raw model output is missing: {item_id}"
                )
            if judged.get("raw_model_output_sha256") != hashlib.sha256(
                exact_raw_output.encode("utf-8")
            ).hexdigest():
                raise ValueError(
                    f"Judge raw model output hash mismatch: {item_id}"
                )
            recorded_normalizations = judged.get(
                "deterministic_normalizations"
            )
        else:
            exact_raw_output = raw_output
            recorded_normalizations = []
        if expected_input_mode == "pdf":
            if require_publication_provenance:
                try:
                    canonical_output, actual_normalizations = (
                        canonicalize_pdf_output_for_storage(
                            exact_raw_output, allowed_pages=shown_pages
                        )
                    )
                except ValueError:
                    if (
                        raw_output != exact_raw_output
                        or recorded_normalizations != []
                    ):
                        raise ValueError(
                            "Invalid raw output was altered before report: "
                            f"{item_id}"
                        )
                else:
                    if canonical_output != raw_output:
                        raise ValueError(
                            f"Canonical/raw output mismatch: {item_id}"
                        )
                    if recorded_normalizations != actual_normalizations:
                        raise ValueError(
                            f"Normalization audit mismatch: {item_id}"
                        )
            try:
                parsed_answer, predicted_pages_list = parse_canonical_pdf_output(
                    raw_output, allowed_pages=shown_pages
                )
                independently_legal = True
            except ValueError:
                parsed_answer, predicted_pages_list = "", []
                independently_legal = False
            independently_evidence_correct = (
                independently_legal
                and predicted_pages_list == reference_pages_list
            )
        else:
            if exact_raw_output != raw_output or recorded_normalizations != []:
                raise ValueError(
                    f"Question-only output was normalized: {item_id}"
                )
            parsed_answer = raw_output
            predicted_pages_list = []
            independently_legal = bool(raw_output.strip()) and not looks_like_error_output(
                raw_output
            )
            independently_evidence_correct = True

        if str(judged.get("parsed_answer", "")) != parsed_answer:
            raise ValueError(f"Judge parsed_answer mismatch: {item_id}")
        stored_predicted_pages = canonical_gold_pages(
            judged.get("predicted_evidence_pages"),
            item_id=f"{item_id}:predicted_evidence_pages",
        )
        if stored_predicted_pages != predicted_pages_list:
            raise ValueError(f"Judge predicted evidence mismatch: {item_id}")
        for field in (
            "output_is_legal",
            "evidence_pages_is_correct",
            "answer_is_correct",
            "is_correct",
        ):
            if not isinstance(judged.get(field), bool):
                raise ValueError(f"Judge lacks boolean {field}: {item_id}")
        if judged["output_is_legal"] != independently_legal:
            raise ValueError(f"Judge legality mismatch: {item_id}")
        if judged["evidence_pages_is_correct"] != independently_evidence_correct:
            raise ValueError(f"Judge evidence score mismatch: {item_id}")
        if not independently_legal and judged["answer_is_correct"]:
            raise ValueError(f"Illegal output has a positive answer score: {item_id}")
        if answer == "Unanswerable" and judged["answer_is_correct"] != (
            independently_legal and parsed_answer == "Unanswerable"
        ):
            raise ValueError(f"Judge Unanswerable score mismatch: {item_id}")
        independently_joint = (
            independently_legal
            and judged["answer_is_correct"]
            and independently_evidence_correct
        )
        if judged["is_correct"] != independently_joint:
            raise ValueError(f"Judge joint score mismatch: {item_id}")

        rows.append(
            {
                "paper_id": paper_id,
                "qa_id": qa_id,
                "answer_correct": judged["answer_is_correct"] and independently_legal,
                "gold_unanswerable": answer == "Unanswerable",
                "predicted_unanswerable": parsed_answer == "Unanswerable",
                "structured_required": protocol.require_structured_output,
                "legal_output": independently_legal,
                "reference_pages": set(reference_pages_list),
                "predicted_pages": set(predicted_pages_list),
                "shown_pages": shown_pages,
                "evidence_hops": len(reference_pages_list),
                "evidence_span": (
                    max(reference_pages_list) - min(reference_pages_list)
                    if reference_pages_list
                    else 0
                ),
                "modal_types": "+".join(
                    sorted(str(value) for value in source.get("modal_types", []))
                ),
            }
        )
    return rows


def stratify(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        value: summarize_rows(group)
        for value, group in sorted(groups.items())
    }


def paired_comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any]:
    left_index = {(row["paper_id"], row["qa_id"]): row for row in left}
    right_index = {(row["paper_id"], row["qa_id"]): row for row in right}
    keys = sorted(left_index.keys() & right_index.keys())
    both = [(left_index[key], right_index[key]) for key in keys]
    return {
        "common_items": len(both),
        "left_answer_accuracy": safe_div(
            sum(left_row["answer_correct"] for left_row, _ in both), len(both)
        ),
        "right_answer_accuracy": safe_div(
            sum(right_row["answer_correct"] for _, right_row in both), len(both)
        ),
        "left_wrong_right_correct": sum(
            not left_row["answer_correct"] and right_row["answer_correct"]
            for left_row, right_row in both
        ),
        "left_correct_right_wrong": sum(
            left_row["answer_correct"] and not right_row["answer_correct"]
            for left_row, right_row in both
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.1%}"


def metric_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for order, (label, entry) in enumerate(
        report.get("evaluations", {}).items(), start=1
    ):
        metrics = entry["metrics"]
        mode, _, model = label.rpartition("_")
        rows.append(
            {
                "order": order,
                "evaluation": label,
                "mode": mode or label,
                "model": model or "unknown",
                "total_seen": metrics["total_seen"],
                "answer_accuracy": metrics["answer_accuracy"],
                "answerable_accuracy": metrics["answerable_accuracy"],
                "unanswerable_recall": metrics["unanswerable_recall"],
                "false_refusal_rate": metrics["false_refusal_rate"],
                "illegal_output_rate": metrics["illegal_output_rate"],
                "evidence_exact_match": metrics["evidence_exact_match"],
            }
        )
    return rows


def build_markdown_report(report: dict[str, Any]) -> str:
    rows = metric_rows(report)
    lines = [
        "# 最终单 PDF 1200：4B / 8B 重评技术报告",
        "",
        "## 技术摘要",
        "",
    ]
    full_rows = [row for row in rows if row["mode"] == "full"]
    if full_rows:
        summary = "；".join(
            f"{row['model']} 普通题正确率 {percent(row['answerable_accuracy'])}、"
            f"不可回答识别率 {percent(row['unanswerable_recall'])}"
            for row in full_rows
        )
        lines.append(f"Full PDF 主结果：{summary}。")
    else:
        lines.append("评测结果已汇总，主结果见下表。")
    lines.extend(
        [
            "",
            "全部输入来自 final_2200 的最终普通题与不可回答题；"
            "200条不可回答题只接受精确标签 `Unanswerable`，且不要求证据页。",
            "",
            "## 各模式结果",
            "",
            "| 模式 | 模型 | 样本 | 总体正确率 | 普通题正确率 | 不可回答识别率 | 普通题错误拒答率 | 非法输出率 | 证据页完全匹配 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {mode} | {model} | {total_seen} | {overall} | {answerable} | "
            "{unanswerable} | {false_refusal} | {illegal} | {evidence} |".format(
                mode=row["mode"],
                model=row["model"],
                total_seen=row["total_seen"],
                overall=percent(row["answer_accuracy"]),
                answerable=percent(row["answerable_accuracy"]),
                unanswerable=percent(row["unanswerable_recall"]),
                false_refusal=percent(row["false_refusal_rate"]),
                illegal=percent(row["illegal_output_rate"]),
                evidence=percent(row["evidence_exact_match"]),
            )
        )
    lines.extend(
        [
            "",
            "## 口径与范围",
            "",
            "- 数据粒度：一条 `(paper_id, qa_id)` QA。",
            "- 普通题分母：1000；不可回答题分母：200；Full PDF/Oracle 总分母：1200。",
            "- 闭卷仅用于1000条普通题的知识泄漏诊断，不混入不可回答题。",
            "- `unanswerable_recall` 只把规范化后精确等于 `Unanswerable` 的输出计为命中。",
            "- `illegal_output_rate` 使用完整结构化输出分母，不能通过跳过坏结果抬高正确率。",
            "",
            "## 方法与质量门禁",
            "",
            "4B和8B使用同一评测协议；思考模式关闭。空输出、非法JSON、普通答案缺证据页、"
            "拒答附带证据页等结果会重试。任一结构化阶段非法输出率超过1%时，报告阶段非零退出，"
            "任务不会被标记完成。",
            "",
            "## 限制与不确定性",
            "",
            "消融变体由最终普通题的当前证据页实时生成；证据页调整后必须重新预检，"
            "不得复用旧 850 条消融输入或旧评测结果。",
            "",
            "## 建议",
            "",
            "1. 优先人工复核4B/8B判决不一致项和拒答混淆项。",
            "2. 只有非法输出率不超过1%且所有阶段分母完整时，才引用本报告中的正确率。",
            "",
            "## 后续问题",
            "",
            "- Full PDF 与 Oracle 的差异主要来自上下文缺失，还是证据页标注不足？",
            "- 200条不可回答题与普通题的错误拒答分布是否集中在特定学科？",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifact(report: dict[str, Any]) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = metric_rows(report)
    markdown = build_markdown_report(report)
    sections = markdown.split("\n## ")
    title = sections[0]
    body_sections = ["## " + value for value in sections[1:]]
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": title, "layout": "full"},
    ]
    if body_sections:
        blocks.append(
            {
                "id": "technical-summary",
                "type": "markdown",
                "body": body_sections[0],
                "layout": "full",
                "sourceId": "single-pdf-results",
            }
        )
    blocks.extend(
        [
            {
                "id": "accuracy-chart-intro",
                "type": "markdown",
                "body": "## 各评测条件下的普通题正确率\n\n柱状图使用每个模式实际完成的固定分母；闭卷只用于泄漏诊断，Full PDF与Oracle用于材料条件对比。",
                "layout": "full",
                "sourceId": "single-pdf-results",
            },
            {
                "id": "accuracy-chart",
                "type": "chart",
                "chartId": "answerable-accuracy-chart",
                "layout": "full",
            },
            {
                "id": "metrics-table-intro",
                "type": "markdown",
                "body": "## 完整指标\n\n表格同时展示答案正确率、拒答、非法输出和证据页指标；非法输出率超过1%的阶段不通过质量门禁。",
                "layout": "full",
                "sourceId": "single-pdf-results",
            },
            {
                "id": "metrics-table",
                "type": "table",
                "tableId": "evaluation-metrics",
                "layout": "full",
            },
        ]
    )
    for index, section in enumerate(body_sections[1:], start=1):
        if section.startswith("## 各模式结果"):
            continue
        blocks.append(
            {
                "id": f"section-{index}",
                "type": "markdown",
                "body": section,
                "layout": "full",
                "sourceId": "single-pdf-results",
            }
        )
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "最终单 PDF 1200：4B / 8B 重评技术报告",
            "description": "最终1000条普通题与200条不可回答题双模型重评。",
            "generatedAt": generated_at,
            "sources": [
                {
                    "id": "single-pdf-results",
                    "label": "最终单 PDF 1200 数据与 Judge 结果",
                    "query": {
                        "description": "按固定分母汇总4B/8B在闭卷、Full PDF与Oracle条件下的答案、拒答、非法输出和证据页指标。",
                        "engine": "artifact_snapshot",
                        "sql": "SELECT * FROM evaluation_metrics ORDER BY order",
                        "language": "python",
                        "filters": [
                            "1000 answerable + 200 unanswerable",
                            "exact canonical Unanswerable",
                            "illegal output gate <= 1%",
                        ],
                        "metric_definitions": list(report["metric_definitions"].values()),
                        "tables_used": [
                            report["qa_dataset"],
                            *[
                                entry["source"]
                                for entry in report["evaluations"].values()
                            ],
                        ],
                    },
                }
            ],
            "charts": [
                {
                    "id": "answerable-accuracy-chart",
                    "title": "普通题答案正确率",
                    "subtitle": "闭卷、Full PDF与Oracle；按模型分组",
                    "type": "bar",
                    "dataset": "evaluation_metrics",
                    "sourceId": "single-pdf-results",
                    "encodings": {
                        "x": {"field": "mode", "type": "nominal", "label": "模式"},
                        "y": {
                            "field": "answerable_accuracy",
                            "type": "quantitative",
                            "aggregate": "none",
                            "format": "percent",
                            "label": "普通题正确率",
                        },
                        "color": {"field": "model", "type": "nominal", "label": "模型"},
                    },
                    "layout": "full",
                    "maxRows": 12,
                }
            ],
            "tables": [
                {
                    "id": "evaluation-metrics",
                    "title": "最终单 PDF 1200 双模型结果",
                    "subtitle": "固定分母；百分比均为0到1比例",
                    "dataset": "evaluation_metrics",
                    "density": "spacious",
                    "sourceId": "single-pdf-results",
                    "layout": "full",
                    "columns": [
                        {"field": "mode", "label": "模式", "type": "text"},
                        {"field": "model", "label": "模型", "type": "text"},
                        {"field": "total_seen", "label": "样本", "format": "number"},
                        {"field": "answer_accuracy", "label": "总体正确率", "format": "percent"},
                        {"field": "answerable_accuracy", "label": "普通题正确率", "format": "percent"},
                        {"field": "unanswerable_recall", "label": "不可回答识别率", "format": "percent"},
                        {"field": "false_refusal_rate", "label": "错误拒答率", "format": "percent"},
                        {"field": "illegal_output_rate", "label": "非法输出率", "format": "percent"},
                        {"field": "evidence_exact_match", "label": "证据页完全匹配", "format": "percent"},
                    ],
                }
            ],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"evaluation_metrics": rows},
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_json(args.qa_json)
    gold = gold_index(dataset)
    pdf_dirs = [path.resolve() for path in (args.pdf_dir or [DEFAULT_PDF_DIR])]
    pdf_audit = audit_pdf_corpus(dataset, pdf_dirs)
    question_only_dataset = load_json(args.question_only_qa_json)
    ablation_dataset = load_json(args.ablation_qa_json)
    ablation_pdf_audit = audit_pdf_corpus(ablation_dataset, pdf_dirs)
    ablation_expectations = build_ablation_expectations(
        ablation_dataset, gold
    )
    specs = parse_eval_specs(args.eval)
    required_labels = {
        f"{mode}_{model}"
        for mode in sorted(EVALUATION_MODES)
        for model in ("4B", "8B")
    }
    if set(specs) != required_labels:
        raise ValueError(
            "A publication report requires exactly four modes x two models: "
            f"missing={sorted(required_labels - set(specs))} "
            f"extra={sorted(set(specs) - required_labels)}"
        )
    all_rows: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {
        "qa_dataset": source_name(args.qa_json),
        "question_only_qa_dataset": source_name(args.question_only_qa_json),
        "ablation_qa_dataset": source_name(args.ablation_qa_json),
        "qa_items": len(gold),
        "pdf_audit": pdf_audit,
        "ablation_pdf_audit": ablation_pdf_audit,
        "evaluations": {},
        "paired_comparisons": {},
        "metric_definitions": {
            "answerable_accuracy": (
                "Semantic/typed answer correctness among answerable gold items."
            ),
            "unanswerable_recall": (
                "Gold-unanswerable items predicted exactly as unanswerable."
            ),
            "false_refusal_rate": (
                "Answerable items predicted exactly as unanswerable."
            ),
            "answerability_macro_f1": (
                "Unweighted mean F1 for answerable vs unanswerable classification."
            ),
            "evidence_exact_match": (
                "Predicted physical PDF page set exactly equals the gold set."
            ),
        },
    }
    flat_rows: list[dict[str, Any]] = []
    for label, path in specs.items():
        mode = evaluation_mode(label)
        source_path = (
            args.question_only_qa_json
            if mode == "question_only"
            else args.ablation_qa_json
            if mode == "ablation"
            else args.qa_json
        )
        source_dataset = (
            question_only_dataset
            if mode == "question_only"
            else ablation_dataset
            if mode == "ablation"
            else dataset
        )
        mode_pdf_audit = (
            None
            if mode == "question_only"
            else ablation_pdf_audit
            if mode == "ablation"
            else pdf_audit
        )
        provenance = validate_eval_provenance(
            label=label,
            judge_path=path,
            source_path=source_path,
            source_dataset=source_dataset,
            report_gold=gold,
            pdf_audit=mode_pdf_audit,
        )
        rows = eval_rows(
            load_json(path),
            gold,
            mode=mode,
            ablation_expectations=(
                ablation_expectations if mode == "ablation" else None
            ),
            pdf_audit=pdf_audit,
            expected_count=(
                len(ablation_expectations) if mode == "ablation" else None
            ),
            require_publication_provenance=True,
        )
        all_rows[label] = rows
        metrics = summarize_rows(rows)
        report["evaluations"][label] = {
            "source": source_name(path),
            "provenance": provenance,
            "metrics": metrics,
            "by_evidence_hops": stratify(rows, "evidence_hops"),
            "by_modal_types": stratify(rows, "modal_types"),
        }
        flat_rows.append({"evaluation": label, **metrics})

    labels = list(all_rows)
    for left_label in labels:
        for right_label in labels:
            if left_label >= right_label:
                continue
            key = f"{left_label}__vs__{right_label}"
            report["paired_comparisons"][key] = paired_comparison(
                all_rows[left_label], all_rows[right_label]
            )

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_csv(args.output_dir / "metrics.csv", flat_rows)
    (args.output_dir / "technical_report.md").write_text(
        build_markdown_report(report), encoding="utf-8"
    )
    with (args.output_dir / "artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(build_artifact(report), handle, ensure_ascii=False, indent=2)
    if args.max_illegal_rate is not None:
        failures = [
            f"{label}={entry['metrics']['illegal_output_rate']:.2%}"
            for label, entry in report["evaluations"].items()
            if entry["metrics"]["structured_output_count"]
            and entry["metrics"]["illegal_output_rate"]
            > args.max_illegal_rate
        ]
        if failures:
            raise RuntimeError(
                "illegal-output quality gate failed: "
                + ", ".join(failures)
                + f"; maximum={args.max_illegal_rate:.2%}"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
