#!/usr/bin/env python3
"""
汇总报告脚本
支持读取任意数量的 judge_*.json，生成文本报告和结构化 JSON 报告。

示例：
  python run_report.py
  python run_report.py --judge-files data/results/evaluations/mixed_pdf_eval/judge_4B.json data/results/evaluations/mixed_pdf_eval/judge_8B.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from eval_framework import iter_paper_items, looks_like_error_output, safe_divide
from evaluation_protocol import (
    JUDGE_INFERENCE_BINDING_FIELDS,
    PDF_INPUT_MODE,
    canonical_gold_answer,
    canonical_gold_pages,
    canonicalize_pdf_output_for_storage,
    is_exact_unanswerable,
    iter_dataset_qas,
    judge_inference_binding_sha256,
    parse_canonical_pdf_output,
    pdf_corpus_sha256_from_manifest,
    pdf_sha256_manifest,
    protocol_for_item,
    sha256_file,
    validate_dataset_protocol,
)


def parse_args():
    parser = argparse.ArgumentParser(description="生成最终正确率报告")
    parser.add_argument(
        "--judge-files",
        nargs="+",
        required=True,
        help="One or more publication-grade Judge result files.",
    )
    parser.add_argument(
        "--inference-files",
        nargs="*",
        required=True,
        help=(
            "Inference files aligned by model name with --judge-files. "
            "Required by the strict publication protocol."
        ),
    )
    parser.add_argument(
        "--expected-judge-fingerprint",
        action="append",
        default=[],
        metavar="MODEL=SHA256",
        help="Expected Judge protocol fingerprint for one model.",
    )
    parser.add_argument(
        "--expected-evaluated-model",
        action="append",
        default=[],
        metavar="FILE_MODEL=MODEL_ID",
        help="Expected evaluated_model value for one result filename model.",
    )
    parser.add_argument(
        "--pdf-dir",
        action="append",
        default=[],
        help="PDF corpus directory; repeat for fallbacks.",
    )
    parser.add_argument(
        "--qa-json",
        required=True,
        help="Independent gold QA file used to lock keys and denominators.",
    )
    parser.add_argument("--output", type=str, default="data/results/evaluations/mixed_pdf_eval/final_report.txt", help="文本报告输出路径")
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/results/evaluations/mixed_pdf_eval/final_report.json",
        help="结构化 JSON 报告输出路径",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "PDF evidence summary CSV. Defaults to final_report.csv next to "
            "--output-json when evidence-page evaluation is present."
        ),
    )
    parser.add_argument(
        "--max-skipped-illegal-rate",
        type=float,
        default=None,
        help=(
            "Fail after writing diagnostics when skipped_illegal_answer / "
            "total_seen exceeds this fraction (for example 0.01)."
        ),
    )
    return parser.parse_args()


def discover_judge_files(args) -> list[Path]:
    return [Path(path) for path in args.judge_files]


def derive_model_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("judge_"):
        return stem[len("judge_") :]
    return stem


def derive_inference_model_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("results_"):
        return stem[len("results_") :]
    return stem


def parse_named_values(
    values: list[str], *, option_name: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        model, separator, fingerprint = value.partition("=")
        if not separator or not model.strip() or not fingerprint.strip():
            raise ValueError(
                f"{option_name} must use NAME=VALUE"
            )
        if model in result:
            raise ValueError(f"{option_name}: duplicate name {model}")
        result[model] = fingerprint
    return result


def display_model_name(model_name: str) -> str:
    return {
        "4B": "Qwen3VL_4B",
        "8B": "Qwen3VL_8B",
    }.get(model_name, model_name)


def is_legal_evidence_answer(qa_data: dict) -> bool:
    """Match the canonical PDF legal/illegal structured-answer accounting."""
    if isinstance(qa_data.get("output_is_legal"), bool):
        return qa_data["output_is_legal"]
    method = str(qa_data.get("output_parse_method", ""))
    if not method or method == "invalid_json":
        return False
    parsed_answer = qa_data.get("parsed_answer")
    if parsed_answer is None or not str(parsed_answer).strip():
        return False
    return isinstance(qa_data.get("predicted_evidence_pages"), list)


def analyze(judge_path: Path) -> dict | None:
    if not judge_path.exists():
        return None

    with open(judge_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mcq_correct = 0
    mcq_total = 0
    mcq_parse_fail = 0
    fill_correct = 0
    fill_total = 0
    fill_judge_errors = 0
    fill_direct_match = 0
    fill_llm_judged = 0
    answer_correct = 0
    evidence_pages_correct = 0
    both_correct = 0
    evidence_required_total = 0
    structured_required_total = 0
    evidence_legal_samples = 0
    total_seen = 0
    legal_samples = 0

    for _, paper_data in iter_paper_items(data):
        if "error" in paper_data:
            continue
        for _, qa_data in paper_data.get("QA", {}).items():
            total_seen += 1
            q_type = qa_data.get("type", "")
            is_correct = qa_data.get("is_correct")
            require_evidence = bool(qa_data.get("require_evidence_pages"))
            require_structured = bool(
                qa_data.get("require_structured_output") or require_evidence
            )
            if require_evidence:
                evidence_required_total += 1
            if require_structured:
                structured_required_total += 1
                if is_legal_evidence_answer(qa_data):
                    legal_samples += 1
                    answer_ok = qa_data.get("answer_is_correct") is True
                    page_ok = qa_data.get("evidence_pages_is_correct") is True
                    if answer_ok:
                        answer_correct += 1
                    if require_evidence:
                        evidence_legal_samples += 1
                        if page_ok:
                            evidence_pages_correct += 1
                    if answer_ok and (page_ok or not require_evidence):
                        both_correct += 1
            else:
                plain_legal = qa_data.get("output_is_legal") is not False
                answer_ok = qa_data.get("answer_is_correct") is True
                if plain_legal and answer_ok:
                    answer_correct += 1
                    both_correct += 1
            scored_correct = is_correct is True and (
                not require_structured or is_legal_evidence_answer(qa_data)
            )
            if q_type == "mcq":
                mcq_total += 1
                if scored_correct:
                    mcq_correct += 1
                elif qa_data.get("extracted_choice") is None:
                    mcq_parse_fail += 1
            elif q_type == "fill":
                fill_total += 1
                if scored_correct:
                    fill_correct += 1
                if qa_data.get("judge_verdict") == "ERROR":
                    fill_judge_errors += 1
                if qa_data.get("match_method") == "llm_judge":
                    fill_llm_judged += 1
                elif qa_data.get("match_method"):
                    fill_direct_match += 1

    total = mcq_total + fill_total
    total_correct = mcq_correct + fill_correct
    answer_accuracy = safe_divide(answer_correct, total_seen)
    evidence_pages_accuracy = safe_divide(
        evidence_pages_correct, evidence_required_total
    )
    both_accuracy = safe_divide(both_correct, total_seen)
    answer_accuracy_legal = safe_divide(answer_correct, legal_samples)
    evidence_pages_accuracy_legal = safe_divide(
        evidence_pages_correct, evidence_legal_samples
    )
    both_accuracy_legal = safe_divide(both_correct, legal_samples)
    total_accuracy = safe_divide(total_correct, total)
    joint_accuracy = both_accuracy if evidence_required_total else total_accuracy
    result = {
        "model_name": derive_model_name(judge_path),
        "judge_file": str(judge_path),
        "mcq_total": mcq_total,
        "mcq_correct": mcq_correct,
        "mcq_accuracy": safe_divide(mcq_correct, mcq_total),
        "mcq_parse_fail": mcq_parse_fail,
        "fill_total": fill_total,
        "fill_correct": fill_correct,
        "fill_accuracy": safe_divide(fill_correct, fill_total),
        "fill_judge_errors": fill_judge_errors,
        "fill_direct_match": fill_direct_match,
        "fill_llm_judged": fill_llm_judged,
        "total": total,
        "total_correct": total_correct,
        "total_accuracy": total_accuracy,
        "evidence_required_total": evidence_required_total,
        "structured_required_total": structured_required_total,
        "evidence_required_legal_samples": evidence_legal_samples,
        "answer_correct": answer_correct,
        "answer_accuracy": answer_accuracy,
        "answer_accuracy_legal": answer_accuracy_legal,
        "evidence_pages_correct": evidence_pages_correct,
        "evidence_pages_accuracy": evidence_pages_accuracy,
        "evidence_pages_accuracy_legal": evidence_pages_accuracy_legal,
        "both_correct": both_correct,
        "both_accuracy": both_accuracy,
        "both_accuracy_legal": both_accuracy_legal,
        "joint_accuracy": joint_accuracy,
        # Publication-facing primary accuracy uses the fixed all-item
        # denominator. Legal-only rates remain diagnostics below.
        "primary_accuracy": (
            answer_accuracy if evidence_required_total else total_accuracy
        ),
    }
    if structured_required_total:
        # Keep these names exactly aligned with the collaborator's table.
        result["pdf_summary"] = {
            "model_name": display_model_name(result["model_name"]),
            "total_seen": total_seen,
            "legal_samples": legal_samples,
            "skipped_illegal_answer": total_seen - legal_samples,
            "skipped_illegal_rate": (
                (total_seen - legal_samples) / total_seen
                if total_seen
                else 0.0
            ),
            "evidence_applicable_total": evidence_required_total,
            "evidence_applicable_legal_samples": evidence_legal_samples,
            "answer_correct": answer_correct,
            "page_correct": evidence_pages_correct,
            "both_correct": both_correct,
            "answer_acc": answer_accuracy,
            "evidence_page_acc": evidence_pages_accuracy,
            "both_acc": both_accuracy,
            "answer_acc_legal": answer_accuracy_legal,
            "evidence_page_acc_legal": evidence_pages_accuracy_legal,
            "both_acc_legal": both_accuracy_legal,
        }
    return result


PDF_SUMMARY_COLUMNS = (
    "model_name",
    "total_seen",
    "legal_samples",
    "skipped_illegal_answer",
    "skipped_illegal_rate",
    "evidence_applicable_total",
    "evidence_applicable_legal_samples",
    "answer_correct",
    "page_correct",
    "both_correct",
    "answer_acc",
    "evidence_page_acc",
    "both_acc",
    "answer_acc_legal",
    "evidence_page_acc_legal",
    "both_acc_legal",
)


def build_pdf_table(results: list[dict]) -> str:
    rows = [
        result["pdf_summary"]
        for result in results
        if "pdf_summary" in result
    ]
    if not rows:
        return ""
    lines = ["\t".join(PDF_SUMMARY_COLUMNS)]
    for row in rows:
        values = []
        for column in PDF_SUMMARY_COLUMNS:
            value = row[column]
            if column.endswith("_rate"):
                values.append(f"{float(value) * 100:.2f}%")
            elif column.endswith("_acc"):
                values.append(f"{float(value):.2f}%")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    return "\n".join(lines)


def write_pdf_csv(path: Path, results: list[dict]) -> None:
    rows = [
        result["pdf_summary"]
        for result in results
        if "pdf_summary" in result
    ]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PDF_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: (
                        f"{float(row[column]) * 100:.2f}%"
                        if column.endswith("_rate")
                        else (
                            f"{float(row[column]):.2f}%"
                            if column.endswith("_acc")
                            else row[column]
                        )
                    )
                    for column in PDF_SUMMARY_COLUMNS
                }
            )


def illegal_rate_failures(
    results: list[dict], maximum_rate: float
) -> list[str]:
    failures = []
    for result in results:
        summary = result.get("pdf_summary")
        if not summary:
            continue
        rate = float(summary["skipped_illegal_rate"])
        if rate > maximum_rate:
            failures.append(
                f"{result['model_name']}={rate:.2%} "
                f"({summary['skipped_illegal_answer']}/"
                f"{summary['total_seen']})"
            )
    return failures


def validate_judge_against_gold(
    judge_path: Path,
    gold_data: dict,
    *,
    inference_path: Path | None = None,
    expected_model: str | None = None,
    expected_judge_fingerprint: str | None = None,
) -> dict[str, object]:
    """Lock the report grain and gold fields before aggregating scores."""
    with judge_path.open("r", encoding="utf-8") as handle:
        judged_data = json.load(handle)
    gold_index = {
        (paper_id, qa_id): qa
        for paper_id, qa_id, qa in iter_dataset_qas(gold_data)
    }
    judged_index = {
        (str(paper_id), str(qa_id)): qa
        for paper_id, paper in iter_paper_items(judged_data)
        if isinstance(paper, dict) and not paper.get("error")
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    }
    inference_index: dict[tuple[str, str], dict] | None = None
    if inference_path is not None:
        with inference_path.open("r", encoding="utf-8") as handle:
            inference_data = json.load(handle)
        inference_index = {
            (str(paper_id), str(qa_id)): qa
            for paper_id, paper in iter_paper_items(inference_data)
            if isinstance(paper, dict) and not paper.get("error")
            for qa_id, qa in paper.get("QA", {}).items()
            if isinstance(qa, dict)
        }
        if set(inference_index) != set(gold_index):
            raise ValueError(
                f"{inference_path}: inference/gold keys differ"
            )
    if set(judged_index) != set(gold_index):
        missing = sorted(set(gold_index) - set(judged_index))
        extra = sorted(set(judged_index) - set(gold_index))
        raise ValueError(
            f"{judge_path}: judged/gold keys differ; "
            f"missing={missing[:5]} extra={extra[:5]} "
            f"judged={len(judged_index)} gold={len(gold_index)}"
        )
    modes: set[str] = set()
    protocol_fingerprints: set[str] = set()
    judge_protocol_fingerprints: set[str] = set()
    qa_source_hashes: set[str] = set()
    pdf_corpus_hashes: set[str] = set()
    fingerprint_missing = 0
    judge_fingerprint_missing = 0
    pdf_hashes_by_paper: dict[str, set[str]] = {}
    evaluated_models: set[str] = set()
    required_score_fields = (
        "is_correct",
        "answer_is_correct",
        "evidence_pages_is_correct",
        "output_is_legal",
    )
    for key, gold in gold_index.items():
        judged = judged_index[key]
        item_id = f"{key[0]}/{key[1]}"
        if inference_index is not None:
            inference = inference_index[key]
            for field in JUDGE_INFERENCE_BINDING_FIELDS:
                if judged.get(field) != inference.get(field):
                    raise ValueError(
                        f"{judge_path}: Judge/inference {field} mismatch "
                        f"at {item_id}"
                    )
            expected_binding = judge_inference_binding_sha256(inference)
            if judged.get("inference_binding_sha256") != expected_binding:
                raise ValueError(
                    f"{judge_path}: stale Judge inference binding at "
                    f"{item_id}"
                )
        if str(judged.get("question", "")) != str(gold.get("question", "")):
            raise ValueError(f"{judge_path}: question mismatch at {item_id}")
        gold_answer = canonical_gold_answer(gold, item_id=item_id)
        if str(judged.get("correct_answer", "")) != gold_answer:
            raise ValueError(f"{judge_path}: answer mismatch at {item_id}")
        mode = str(judged.get("input_mode", ""))
        modes.add(mode)
        evaluated_model = str(judged.get("evaluated_model", "")).strip()
        if evaluated_model:
            evaluated_models.add(evaluated_model)
        fingerprint = str(judged.get("protocol_fingerprint", "")).strip()
        judge_fingerprint = str(
            judged.get("judge_protocol_fingerprint", "")
        ).strip()
        if judge_fingerprint:
            judge_protocol_fingerprints.add(judge_fingerprint)
        else:
            judge_fingerprint_missing += 1
        if fingerprint:
            protocol_fingerprints.add(fingerprint)
            source_hash = str(
                judged.get("qa_source_sha256", "") or ""
            ).strip()
            if not source_hash:
                raise ValueError(
                    f"{judge_path}: missing source hash at {item_id}"
                )
            qa_source_hashes.add(source_hash)
            corpus_hash = str(
                judged.get("pdf_corpus_sha256", "") or ""
            ).strip()
            if corpus_hash:
                pdf_corpus_hashes.add(corpus_hash)
            if mode == PDF_INPUT_MODE:
                pdf_hash = str(
                    judged.get("pdf_sha256", "") or ""
                ).strip()
                if not corpus_hash or not pdf_hash:
                    raise ValueError(
                        f"{judge_path}: missing PDF hash at {item_id}"
                    )
                pdf_hashes_by_paper.setdefault(key[0], set()).add(
                    pdf_hash
                )
        else:
            fingerprint_missing += 1
        expected = protocol_for_item(mode, gold_answer)
        if bool(judged.get("require_structured_output")) != expected.require_structured_output:
            raise ValueError(f"{judge_path}: structured protocol mismatch at {item_id}")
        if bool(judged.get("require_evidence_pages")) != expected.require_evidence_pages:
            raise ValueError(f"{judge_path}: evidence protocol mismatch at {item_id}")
        expected_pages = canonical_gold_pages(
            gold.get("evidence_pages", []), item_id=item_id
        )
        if judged.get("reference_evidence_pages") != expected_pages:
            raise ValueError(f"{judge_path}: evidence gold mismatch at {item_id}")
        for field in required_score_fields:
            if not isinstance(judged.get(field), bool):
                raise ValueError(
                    f"{judge_path}: {item_id} missing boolean {field}"
                )
        raw_output = str(judged.get("model_output", ""))
        if judged.get("publication_eligible") is not True:
            raise ValueError(
                f"{judge_path}: row is not publication eligible at {item_id}"
            )
        if judged.get("protocol_validation") != "publication_strict":
            raise ValueError(
                f"{judge_path}: protocol validation marker mismatch at {item_id}"
            )
        exact_raw_output = str(judged.get("raw_model_output", ""))
        expected_raw_hash = hashlib.sha256(
            exact_raw_output.encode("utf-8")
        ).hexdigest()
        if judged.get("raw_model_output_sha256") != expected_raw_hash:
            raise ValueError(
                f"{judge_path}: raw-output hash mismatch at {item_id}"
            )
        recorded_normalizations = judged.get(
            "deterministic_normalizations"
        )
        if expected.require_structured_output:
            try:
                canonical_output, actual_normalizations = (
                    canonicalize_pdf_output_for_storage(
                        exact_raw_output,
                        allowed_pages=judged.get("shown_pdf_pages"),
                    )
                )
            except ValueError:
                if raw_output != exact_raw_output or recorded_normalizations != []:
                    raise ValueError(
                        f"{judge_path}: invalid raw output was altered at "
                        f"{item_id}"
                    )
            else:
                if canonical_output != raw_output:
                    raise ValueError(
                        f"{judge_path}: canonical/raw output mismatch at "
                        f"{item_id}"
                    )
                if recorded_normalizations != actual_normalizations:
                    raise ValueError(
                        f"{judge_path}: normalization audit mismatch at "
                        f"{item_id}"
                    )
        elif exact_raw_output != raw_output or recorded_normalizations != []:
            raise ValueError(
                f"{judge_path}: question-only raw output was normalized "
                f"at {item_id}"
            )
        if expected.require_structured_output:
            shown_pages = canonical_gold_pages(
                judged.get("shown_pdf_pages"),
                item_id=f"{item_id} shown_pdf_pages",
            )
            if not shown_pages:
                raise ValueError(
                    f"{judge_path}: PDF row has no shown pages at {item_id}"
                )
            try:
                parsed_answer, predicted_pages = parse_canonical_pdf_output(
                    raw_output, allowed_pages=shown_pages
                )
                independently_legal = True
            except ValueError:
                parsed_answer, predicted_pages = "", []
                independently_legal = False
            independently_evidence_correct = (
                independently_legal and predicted_pages == expected_pages
            )
        else:
            if judged.get("shown_pdf_pages") not in (None, []):
                raise ValueError(
                    f"{judge_path}: question-only row exposes PDF pages at "
                    f"{item_id}"
                )
            parsed_answer = raw_output
            predicted_pages = []
            independently_legal = bool(raw_output.strip()) and not looks_like_error_output(
                raw_output
            )
            independently_evidence_correct = True
        if judged["output_is_legal"] != independently_legal:
            raise ValueError(
                f"{judge_path}: independently parsed legality mismatch at {item_id}"
            )
        if str(judged.get("parsed_answer", "")) != parsed_answer:
            raise ValueError(
                f"{judge_path}: independently parsed answer mismatch at {item_id}"
            )
        if judged.get("predicted_evidence_pages", []) != predicted_pages:
            raise ValueError(
                f"{judge_path}: independently parsed evidence mismatch at {item_id}"
            )
        if judged["evidence_pages_is_correct"] != independently_evidence_correct:
            raise ValueError(
                f"{judge_path}: independently recomputed evidence score mismatch at {item_id}"
            )
        if is_exact_unanswerable(gold_answer):
            independently_answer_correct = (
                independently_legal and parsed_answer == "Unanswerable"
            )
            if judged["answer_is_correct"] != independently_answer_correct:
                raise ValueError(
                    f"{judge_path}: independently recomputed exact "
                    f"Unanswerable score mismatch at {item_id}"
                )
        if judged["is_correct"] != (
            judged["answer_is_correct"]
            and judged["evidence_pages_is_correct"]
            and judged["output_is_legal"]
        ):
            raise ValueError(f"{judge_path}: inconsistent joint score at {item_id}")
    if len(modes) != 1:
        raise ValueError(f"{judge_path}: expected one input_mode, found {modes}")
    if fingerprint_missing:
        raise ValueError(f"{judge_path}: rows missing protocol fingerprints")
    if judge_fingerprint_missing:
        raise ValueError(
            f"{judge_path}: rows missing Judge protocol fingerprints"
        )
    if len(protocol_fingerprints) > 1 or len(qa_source_hashes) > 1:
        raise ValueError(f"{judge_path}: mixed inference protocol/source hashes")
    if len(judge_protocol_fingerprints) > 1:
        raise ValueError(f"{judge_path}: mixed Judge protocol fingerprints")
    if len(evaluated_models) > 1:
        raise ValueError(f"{judge_path}: mixed evaluated models")
    if len(pdf_corpus_hashes) > 1:
        raise ValueError(f"{judge_path}: mixed PDF corpus hashes")
    for paper_id, values in pdf_hashes_by_paper.items():
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(
                f"{judge_path}: inconsistent PDF hash for paper {paper_id}"
            )
    mode = next(iter(modes))
    evaluated_model = next(iter(evaluated_models), None)
    if expected_model and evaluated_model != expected_model:
        raise ValueError(
            f"{judge_path}: expected evaluated model {expected_model}, "
            f"found {evaluated_model}"
        )
    judge_protocol_fingerprint = next(
        iter(judge_protocol_fingerprints), None
    )
    if (
        expected_judge_fingerprint
        and judge_protocol_fingerprint != expected_judge_fingerprint
    ):
        raise ValueError(
            f"{judge_path}: Judge fingerprint does not match scheduler"
        )
    profile = validate_dataset_protocol(gold_data, mode)
    profile["protocol_fingerprint"] = next(
        iter(protocol_fingerprints), None
    )
    profile["judge_protocol_fingerprint"] = judge_protocol_fingerprint
    profile["evaluated_model"] = evaluated_model
    profile["qa_source_sha256"] = next(iter(qa_source_hashes), None)
    profile["pdf_corpus_sha256"] = next(iter(pdf_corpus_hashes), None)
    profile["pdf_sha256_by_paper"] = {
        paper_id: next(iter(values))
        for paper_id, values in sorted(pdf_hashes_by_paper.items())
    }
    profile["judge_file_sha256"] = sha256_file(judge_path)
    profile["inference_file_sha256"] = (
        sha256_file(inference_path) if inference_path is not None else None
    )
    return profile


def build_text_report(results: list[dict]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("FINAL EVALUATION REPORT")
    lines.append("=" * 78)
    lines.append("")

    for result in results:
        lines.append(f"Model: {result['model_name']}")
        lines.append("-" * 78)
        lines.append("Multiple Choice Questions (MCQ):")
        lines.append(f"  Total:    {result['mcq_total']:>6d}")
        lines.append(f"  Correct:  {result['mcq_correct']:>6d}")
        lines.append(f"  Accuracy: {result['mcq_accuracy']:>6.2f}%")
        if result["mcq_parse_fail"] > 0:
            lines.append(f"  Parse failures: {result['mcq_parse_fail']}")
        lines.append("")
        lines.append("Short Answer Questions:")
        lines.append(f"  Total:    {result['fill_total']:>6d}")
        lines.append(f"  Correct:  {result['fill_correct']:>6d}")
        lines.append(f"  Accuracy: {result['fill_accuracy']:>6.2f}%")
        lines.append(f"  Direct matches: {result['fill_direct_match']}")
        lines.append(f"  LLM judged: {result['fill_llm_judged']}")
        if result["fill_judge_errors"] > 0:
            lines.append(f"  Judge errors: {result['fill_judge_errors']}")
        lines.append("")
        lines.append("OVERALL:")
        lines.append(f"  Total:    {result['total']:>6d}")
        lines.append(f"  Correct:  {result['total_correct']:>6d}")
        if result["evidence_required_total"]:
            lines.append(f"  Joint accuracy: {result['joint_accuracy']:>6.2f}%")
            lines.append(f"  Answer-only accuracy: {result['answer_accuracy']:>6.2f}%")
            lines.append(f"  Evidence-page accuracy: {result['evidence_pages_accuracy']:>6.2f}%")
            lines.append("  Joint rule: answer and exact evidence-page set must both be correct")
        else:
            lines.append(f"  Accuracy: {result['total_accuracy']:>6.2f}%")
        lines.append("")

    if len(results) >= 2:
        ranked = sorted(results, key=lambda item: item["primary_accuracy"], reverse=True)
        lines.append("RANKING")
        lines.append("-" * 78)
        for idx, item in enumerate(ranked, start=1):
            if item["evidence_required_total"]:
                lines.append(
                    f"{idx:>2d}. {item['model_name']:<30s} "
                    f"answer={item['answer_accuracy']:>6.2f}% | "
                    f"page={item['evidence_pages_accuracy']:>6.2f}% | "
                    f"joint={item['joint_accuracy']:>6.2f}%"
                )
            else:
                lines.append(
                    f"{idx:>2d}. {item['model_name']:<30s} "
                    f"overall={item['total_accuracy']:>6.2f}% | "
                    f"mcq={item['mcq_accuracy']:>6.2f}% | fill={item['fill_accuracy']:>6.2f}%"
                )
        lines.append("")

        if len(results) == 2:
            a, b = results[0], results[1]
            lines.append("PAIRWISE COMPARISON")
            lines.append("-" * 78)
            lines.append(f"Models: {a['model_name']} vs {b['model_name']}")
            if a["evidence_required_total"] and b["evidence_required_total"]:
                lines.append(
                    f"Answer accuracy diff: {b['answer_accuracy'] - a['answer_accuracy']:+.2f}%"
                )
                lines.append(
                    "Evidence-page accuracy diff: "
                    f"{b['evidence_pages_accuracy'] - a['evidence_pages_accuracy']:+.2f}%"
                )
                lines.append(
                    f"Joint accuracy diff: {b['joint_accuracy'] - a['joint_accuracy']:+.2f}%"
                )
            else:
                lines.append(f"MCQ accuracy diff:   {b['mcq_accuracy'] - a['mcq_accuracy']:+.2f}%")
                lines.append(f"Fill accuracy diff:  {b['fill_accuracy'] - a['fill_accuracy']:+.2f}%")
                lines.append(f"Overall accuracy diff: {b['total_accuracy'] - a['total_accuracy']:+.2f}%")
            lines.append("")

    lines.append("=" * 78)
    pdf_table = build_pdf_table(results)
    if pdf_table:
        lines.extend(
            [
                "",
                "PDF ANSWER + EVIDENCE PAGE SUMMARY",
                "-" * 78,
                pdf_table,
            ]
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    gold_path = Path(args.qa_json)
    with gold_path.open("r", encoding="utf-8") as handle:
        gold_data = json.load(handle)
    judge_files = discover_judge_files(args)
    if not judge_files:
        raise FileNotFoundError("No judge result files found.")
    inference_by_model: dict[str, Path] = {}
    for raw_path in args.inference_files or []:
        path = Path(raw_path)
        model = derive_inference_model_name(path)
        if model in inference_by_model:
            raise ValueError(f"Duplicate inference file for model {model}")
        inference_by_model[model] = path
    expected_judge_fingerprints = parse_named_values(
        args.expected_judge_fingerprint,
        option_name="--expected-judge-fingerprint",
    )
    expected_evaluated_models = parse_named_values(
        args.expected_evaluated_model,
        option_name="--expected-evaluated-model",
    )
    judge_models = {derive_model_name(path) for path in judge_files}
    if set(inference_by_model) != judge_models:
        raise ValueError(
            "Report requires exactly one matching inference file per Judge "
            f"model; judge={sorted(judge_models)} "
            f"inference={sorted(inference_by_model)}"
        )
    if set(expected_judge_fingerprints) != judge_models:
        raise ValueError(
            "Report requires one expected Judge fingerprint per model; "
            f"judge={sorted(judge_models)} expected="
            f"{sorted(expected_judge_fingerprints)}"
        )

    results = []
    missing_files = []
    protocol_profiles = []
    gold_sha256 = sha256_file(gold_path)
    for path in judge_files:
        model = derive_model_name(path)
        profile = validate_judge_against_gold(
            path,
            gold_data,
            inference_path=inference_by_model.get(model),
            expected_model=expected_evaluated_models.get(model, model),
            expected_judge_fingerprint=(
                expected_judge_fingerprints.get(model)
            ),
        )
        if not profile.get("protocol_fingerprint"):
            raise ValueError(f"{path}: missing required protocol fingerprint")
        if not profile.get("judge_protocol_fingerprint"):
            raise ValueError(
                f"{path}: missing required Judge protocol fingerprint"
            )
        if (
            profile.get("qa_source_sha256")
            and profile["qa_source_sha256"] != gold_sha256
        ):
            raise ValueError(
                f"{path}: inference source SHA256 does not match --qa-json"
            )
        protocol_profiles.append({"judge_file": str(path), **profile})
        analyzed = analyze(path)
        if analyzed is None:
            missing_files.append(str(path))
            continue
        results.append(analyzed)

    source_hashes = {
        str(profile.get("qa_source_sha256"))
        for profile in protocol_profiles
        if profile.get("qa_source_sha256")
    }
    if len(source_hashes) > 1:
        raise ValueError(
            "Judge files were produced from different QA source hashes"
        )
    corpus_hashes = {
        str(profile.get("pdf_corpus_sha256"))
        for profile in protocol_profiles
        if profile.get("input_mode") == PDF_INPUT_MODE
    }
    if len(corpus_hashes) > 1:
        raise ValueError(
            "Judge files were produced from different PDF corpora"
        )
    pdf_manifests = [
        profile.get("pdf_sha256_by_paper")
        for profile in protocol_profiles
        if profile.get("input_mode") == PDF_INPUT_MODE
    ]
    if pdf_manifests and any(
        manifest != pdf_manifests[0] for manifest in pdf_manifests[1:]
    ):
        raise ValueError(
            "Judge files were produced from different per-paper PDFs"
        )
    for profile in protocol_profiles:
        if profile.get("input_mode") != PDF_INPUT_MODE:
            continue
        manifest = profile.get("pdf_sha256_by_paper") or {}
        recorded_corpus = profile.get("pdf_corpus_sha256")
        recomputed_corpus = pdf_corpus_sha256_from_manifest(manifest)
        if recorded_corpus != recomputed_corpus:
            raise ValueError(
                f"{profile['judge_file']}: PDF corpus hash is inconsistent "
                "with its per-paper hash manifest"
            )
        if not args.pdf_dir:
            raise ValueError(
                "Strict PDF report requires at least one --pdf-dir"
            )
    if args.pdf_dir and any(
        profile.get("input_mode") == PDF_INPUT_MODE
        for profile in protocol_profiles
    ):
        current_pdf_manifest = pdf_sha256_manifest(gold_data, args.pdf_dir)
        current_pdf_corpus = pdf_corpus_sha256_from_manifest(
            current_pdf_manifest
        )
        for profile in protocol_profiles:
            if profile.get("input_mode") != PDF_INPUT_MODE:
                continue
            if profile.get("pdf_sha256_by_paper") != current_pdf_manifest:
                raise ValueError(
                    f"{profile['judge_file']}: evaluated PDFs do not match "
                    "the current PDF files"
                )
            if profile.get("pdf_corpus_sha256") != current_pdf_corpus:
                raise ValueError(
                    f"{profile['judge_file']}: evaluated PDF corpus hash "
                    "does not match current files"
                )

    if not results:
        raise FileNotFoundError("No readable judge result files found.")

    report_text = build_text_report(results)
    # These historical reports contain exact-page/joint diagnostics. They are
    # not the paper's macro-evidence headline report produced by evaluation/.
    publication_eligible = False
    protocol_validation = "publication_strict"
    print(report_text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    report_json = {
        "report_scope": "internal_diagnostics",
        "official_reproduction_verified": False,
        "official_evaluator": "evaluation/evaluate.py",
        "models": results,
        "ranking": sorted(
            [
                {
                    "model_name": item["model_name"],
                    "primary_accuracy": item["primary_accuracy"],
                    "answer_accuracy": item["answer_accuracy"],
                    "evidence_pages_accuracy": item["evidence_pages_accuracy"],
                    "joint_accuracy": item["joint_accuracy"],
                }
                for item in results
            ],
            key=lambda item: item["primary_accuracy"],
            reverse=True,
        ),
        "missing_files": missing_files,
        "gold_source_sha256": gold_sha256,
        "protocol_profiles": protocol_profiles,
        "publication_eligible": publication_eligible,
        "protocol_validation": protocol_validation,
    }
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    if any("pdf_summary" in result for result in results):
        csv_path = (
            Path(args.output_csv)
            if args.output_csv
            else output_json_path.with_name("final_report.csv")
        )
        write_pdf_csv(csv_path, results)
        print(f"Evidence summary CSV saved to: {csv_path}")

    print(f"\nText report saved to: {output_path}")
    print(f"JSON report saved to: {output_json_path}")
    if args.max_skipped_illegal_rate is not None:
        failures = illegal_rate_failures(
            results, args.max_skipped_illegal_rate
        )
        if failures:
            raise RuntimeError(
                "skipped_illegal_answer quality gate failed: "
                + ", ".join(failures)
                + f"; maximum={args.max_skipped_illegal_rate:.2%}"
            )


if __name__ == "__main__":
    main()
