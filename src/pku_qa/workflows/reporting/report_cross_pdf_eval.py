#!/usr/bin/env python3
"""Validate and report a Cross-PDF answer/evidence evaluation.

The preflight mode is intentionally GPU-free.  It verifies that the immutable
QA release and every referenced PDF are present before an expensive evaluation
starts.  The report mode additionally reconciles both judge files with the
dataset and writes a reader-facing Markdown report plus machine-readable data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pku_qa.pdf_assets import resolve_pdf_path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_protocol import (
    pdf_corpus_sha256_from_manifest,
    pdf_sha256_manifest,
)
from run_report import (
    is_legal_evidence_answer,
    validate_judge_against_gold,
)


MODEL_NAMES = {"4B": "Qwen3VL-4B", "8B": "Qwen3VL-8B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and report the Cross-PDF evaluation."
    )
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--judge-4b")
    parser.add_argument("--judge-8b")
    parser.add_argument("--final-report")
    parser.add_argument("--baseline-report")
    parser.add_argument("--output")
    parser.add_argument("--output-json")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_dataset_items(dataset: dict[str, Any]) -> Iterable[tuple[str, str, dict]]:
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict):
            raise ValueError(f"{paper_id}: paper record is not an object")
        qas = paper.get("QA")
        if not isinstance(qas, dict) or not qas:
            raise ValueError(f"{paper_id}: missing non-empty QA object")
        for qa_id, qa in qas.items():
            if not isinstance(qa, dict):
                raise ValueError(f"{paper_id}/{qa_id}: QA record is not an object")
            yield str(paper_id), str(qa_id), qa


def validate_dataset(
    qa_path: Path,
    pdf_dir: Path,
    *,
    expected_count: int,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    actual_sha256 = sha256_file(qa_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "QA SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    dataset = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("QA dataset must be a non-empty object")

    qa_rows = list(iter_dataset_items(dataset))
    if len(qa_rows) != expected_count:
        raise ValueError(
            f"QA count mismatch: expected {expected_count}, got {len(qa_rows)}"
        )

    missing_pdfs: list[str] = []
    invalid_pdfs: list[str] = []
    duplicate_uids: list[str] = []
    seen_uids: set[str] = set()
    hop_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    source_papers: set[str] = set()
    evidence_items_total = 0

    for paper_id in dataset:
        pdf_path = resolve_pdf_path(paper_id, [pdf_dir], required=False)
        if pdf_path is None:
            missing_pdfs.append(str(pdf_dir / f"{paper_id}.pdf"))
        else:
            with pdf_path.open("rb") as handle:
                header = handle.read(5)
            if header != b"%PDF-":
                invalid_pdfs.append(str(pdf_path))

    for paper_id, qa_id, qa in qa_rows:
        uid = str(qa.get("qa_uid") or f"{paper_id}/{qa_id}")
        if uid in seen_uids:
            duplicate_uids.append(uid)
        seen_uids.add(uid)

        evidence_pages = qa.get("evidence_pages")
        if (
            not isinstance(evidence_pages, list)
            or len(evidence_pages) < 2
            or any(not isinstance(page, int) or page < 1 for page in evidence_pages)
        ):
            raise ValueError(f"{paper_id}/{qa_id}: invalid evidence_pages")
        evidence_items = qa.get("evidence_items")
        if not isinstance(evidence_items, list) or len(evidence_items) < 2:
            raise ValueError(f"{paper_id}/{qa_id}: incomplete evidence_items")
        evidence_items_total += len(evidence_items)

        hops = qa.get("evidence_hops")
        if not isinstance(hops, int) or hops < 2:
            raise ValueError(f"{paper_id}/{qa_id}: evidence_hops must be >= 2")
        hop_counts[str(hops)] += 1

        modalities = qa.get("modal_types")
        if not isinstance(modalities, list) or not modalities:
            raise ValueError(f"{paper_id}/{qa_id}: missing modal_types")
        for modality in set(map(str, modalities)):
            modality_counts[modality] += 1

        source_ids = qa.get("source_paper_ids")
        if not isinstance(source_ids, list) or len(set(map(str, source_ids))) < 2:
            raise ValueError(f"{paper_id}/{qa_id}: fewer than two source papers")
        source_papers.update(map(str, source_ids))
        review_counts[str(qa.get("review_status") or "unspecified")] += 1

    if missing_pdfs:
        raise FileNotFoundError(
            f"{len(missing_pdfs)} referenced PDFs are missing; first: {missing_pdfs[0]}"
        )
    if invalid_pdfs:
        raise ValueError(
            f"{len(invalid_pdfs)} referenced PDFs are invalid; first: {invalid_pdfs[0]}"
        )
    if duplicate_uids:
        raise ValueError(f"Duplicate qa_uid: {duplicate_uids[0]}")

    summary = {
        "qa_path": str(qa_path),
        "pdf_dir": str(pdf_dir),
        "sha256": actual_sha256,
        "bundles": len(dataset),
        "qas": len(qa_rows),
        "pdfs": len(dataset),
        "source_papers": len(source_papers),
        "evidence_items": evidence_items_total,
        "evidence_hops": dict(sorted(hop_counts.items())),
        "modalities": dict(sorted(modality_counts.items())),
        "review_status": dict(sorted(review_counts.items())),
    }
    return dataset, summary


def load_judge_rows(
    path: Path,
    expected_keys: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Judge file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Judge file must be an object: {path}")

    rows: list[dict[str, Any]] = []
    observed_keys: set[tuple[str, str]] = set()
    paper_errors: list[str] = []
    for paper_id, paper in data.items():
        if not isinstance(paper, dict):
            raise ValueError(f"{path}: {paper_id} is not an object")
        if "error" in paper:
            paper_errors.append(str(paper_id))
            continue
        qas = paper.get("QA")
        if not isinstance(qas, dict):
            raise ValueError(f"{path}: {paper_id} has no QA object")
        for qa_id, qa in qas.items():
            if not isinstance(qa, dict):
                raise ValueError(f"{path}: {paper_id}/{qa_id} is not an object")
            key = (str(paper_id), str(qa_id))
            observed_keys.add(key)
            rows.append({"paper_id": key[0], "qa_id": key[1], **qa})

    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if paper_errors or missing or extra:
        raise ValueError(
            f"Incomplete judge file {path}: paper_errors={len(paper_errors)}, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return rows


def percent(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * (
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return (center - margin) * 100, (center + margin) * 100


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    legal = [row for row in selected if is_legal_evidence_answer(row)]
    answer_correct = sum(row.get("answer_is_correct") is True for row in legal)
    page_correct = sum(row.get("evidence_pages_is_correct") is True for row in legal)
    both_correct = sum(
        row.get("answer_is_correct") is True
        and row.get("evidence_pages_is_correct") is True
        for row in legal
    )
    low, high = wilson_interval(answer_correct, len(selected))
    return {
        "total_seen": len(selected),
        "legal_samples": len(legal),
        "skipped_illegal_answer": len(selected) - len(legal),
        "legal_coverage": percent(len(legal), len(selected)),
        "answer_correct": answer_correct,
        "page_correct": page_correct,
        "both_correct": both_correct,
        "answer_acc_legal": percent(answer_correct, len(legal)),
        "evidence_page_acc_legal": percent(page_correct, len(legal)),
        "both_acc_legal": percent(both_correct, len(legal)),
        "end_to_end_answer_success": percent(answer_correct, len(selected)),
        "end_to_end_answer_success_ci95": [low, high],
    }


def model_metrics(
    model: str,
    judge_rows: list[dict[str, Any]],
    qa_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    enriched = []
    for row in judge_rows:
        metadata = qa_index[(row["paper_id"], row["qa_id"])]
        enriched.append(
            {
                **row,
                "_evidence_hops": str(metadata.get("evidence_hops", "unknown")),
                "_question_type": str(metadata.get("question_type") or "unspecified"),
                "_review_status": str(metadata.get("review_status") or "unspecified"),
            }
        )

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "evidence_hops": defaultdict(list),
        "question_type": defaultdict(list),
        "review_status": defaultdict(list),
    }
    for row in enriched:
        grouped["evidence_hops"][row["_evidence_hops"]].append(row)
        grouped["question_type"][row["_question_type"]].append(row)
        grouped["review_status"][row["_review_status"]].append(row)

    return {
        "model": model,
        "display_name": MODEL_NAMES.get(model, model),
        "overall": aggregate(enriched),
        "subgroups": {
            dimension: {
                key: aggregate(values)
                for key, values in sorted(groups.items())
            }
            for dimension, groups in grouped.items()
        },
    }


def difficulty_label(max_success: float) -> str:
    if max_success <= 40:
        return "高难（两模型最高端到端答案成功率不超过 40%）"
    if max_success <= 60:
        return "有挑战性（两模型最高端到端答案成功率为 40%–60%）"
    return "难度不足（至少一个模型端到端答案成功率超过 60%）"


def baseline_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Baseline report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["model_name"]): item["pdf_summary"]
        for item in report.get("models", [])
        if isinstance(item, dict) and "pdf_summary" in item
    }


def validate_publication_artifacts(
    *,
    dataset: dict[str, Any],
    qa_path: Path,
    pdf_dir: Path,
    judge_paths: dict[str, Path],
    final_report: Path,
) -> dict[str, Any]:
    """Bind the custom Cross-PDF report to strict generic artifacts."""
    standard = json.loads(final_report.read_text(encoding="utf-8"))
    if standard.get("gold_source_sha256") != sha256_file(qa_path):
        raise ValueError("Standard report gold source hash mismatch")
    profiles = standard.get("protocol_profiles")
    if not isinstance(profiles, list) or len(profiles) != len(MODEL_NAMES):
        raise ValueError("Standard report lacks two strict protocol profiles")
    profile_by_model: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ValueError("Malformed standard protocol profile")
        model = str(profile.get("evaluated_model", ""))
        if model in profile_by_model or model not in MODEL_NAMES:
            raise ValueError("Standard report model/profile set mismatch")
        profile_by_model[model] = profile
    if set(profile_by_model) != set(MODEL_NAMES):
        raise ValueError("Standard report model/profile set mismatch")

    current_pdf_manifest = pdf_sha256_manifest(dataset, [pdf_dir])
    current_pdf_corpus = pdf_corpus_sha256_from_manifest(
        current_pdf_manifest
    )
    validated: dict[str, Any] = {}
    for model, judge_path in judge_paths.items():
        profile = profile_by_model[model]
        inference_path = judge_path.parent / f"results_{model}.json"
        fresh = validate_judge_against_gold(
            judge_path,
            dataset,
            inference_path=inference_path,
            expected_model=model,
            expected_judge_fingerprint=str(
                profile.get("judge_protocol_fingerprint", "")
            ),
        )
        expected_fields = (
            "input_mode",
            "total",
            "protocol_fingerprint",
            "judge_protocol_fingerprint",
            "evaluated_model",
            "qa_source_sha256",
            "pdf_corpus_sha256",
            "pdf_sha256_by_paper",
            "judge_file_sha256",
            "inference_file_sha256",
        )
        for field in expected_fields:
            if fresh.get(field) != profile.get(field):
                raise ValueError(
                    f"{model} standard/custom artifact {field} mismatch"
                )
        if fresh["pdf_sha256_by_paper"] != current_pdf_manifest:
            raise ValueError(f"{model} evaluated PDFs differ from current files")
        if fresh["pdf_corpus_sha256"] != current_pdf_corpus:
            raise ValueError(f"{model} PDF corpus hash differs from current files")
        validated[model] = fresh
    return {
        "status": "passed",
        "qa_source_sha256": sha256_file(qa_path),
        "pdf_corpus_sha256": current_pdf_corpus,
        "models": validated,
    }


def fmt(value: float) -> str:
    return f"{value:.2f}%"


def metrics_table(models: list[dict[str, Any]]) -> str:
    lines = [
        "| 模型 | 总题数 | 合法输出 | 答案正确（合法口径） | 证据页精确命中 | 答案+证据页联合 | 端到端答案成功率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        row = model["overall"]
        lines.append(
            f"| {model['display_name']} | {row['total_seen']} | "
            f"{row['legal_samples']} ({fmt(row['legal_coverage'])}) | "
            f"{row['answer_correct']}/{row['legal_samples']} "
            f"({fmt(row['answer_acc_legal'])}) | "
            f"{row['page_correct']}/{row['legal_samples']} "
            f"({fmt(row['evidence_page_acc_legal'])}) | "
            f"{row['both_correct']}/{row['legal_samples']} "
            f"({fmt(row['both_acc_legal'])}) | "
            f"{row['answer_correct']}/{row['total_seen']} "
            f"({fmt(row['end_to_end_answer_success'])}) |"
        )
    return "\n".join(lines)


def subgroup_table(models: list[dict[str, Any]], dimension: str) -> str:
    keys = sorted(
        {
            key
            for model in models
            for key in model["subgroups"][dimension]
        }
    )
    lines = [
        "| 分组 | 模型 | n | 答案准确率（合法） | 证据页准确率（合法） | 联合准确率（合法） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in keys:
        for model in models:
            row = model["subgroups"][dimension].get(key)
            if row is None:
                continue
            lines.append(
                f"| {key} | {model['display_name']} | {row['total_seen']} | "
                f"{fmt(row['answer_acc_legal'])} | "
                f"{fmt(row['evidence_page_acc_legal'])} | "
                f"{fmt(row['both_acc_legal'])} |"
            )
    return "\n".join(lines)


def build_markdown(
    dataset_summary: dict[str, Any],
    models: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    generated_at: str,
) -> str:
    max_success = max(
        model["overall"]["end_to_end_answer_success"] for model in models
    )
    label = difficulty_label(max_success)
    answer_gap = (
        models[1]["overall"]["end_to_end_answer_success"]
        - models[0]["overall"]["end_to_end_answer_success"]
    )
    lines = [
        "# Cross-PDF QA v3：4B / 8B Full-PDF 评测报告",
        "",
        "## 技术摘要",
        "",
        f"- 数据集：{dataset_summary['qas']} 条 QA、{dataset_summary['bundles']} 个拼接 PDF、"
        f"{dataset_summary['source_papers']} 篇去重源论文；"
        f"SHA-256 `{dataset_summary['sha256']}`。",
        f"- 难度结论：**{label}**。该结论按预设门槛使用两模型中较高的端到端答案成功率判断，"
        "不为了得到低分而改动题目、答案或计分规则。",
        f"- 模型差异：8B 相对 4B 的端到端答案成功率差为 {answer_gap:+.2f} 个百分点。",
        "- 主表同时报告合法结构化输出口径和“非法输出按失败计”的端到端口径，避免因排除解析失败而高估正确率。",
        "",
        "## 关键结果",
        "",
        metrics_table(models),
        "",
    ]
    for model in models:
        row = model["overall"]
        low, high = row["end_to_end_answer_success_ci95"]
        lines.append(
            f"- {model['display_name']} 端到端答案成功率的 Wilson 95% 区间："
            f"{fmt(low)}–{fmt(high)}。"
        )

    if baselines:
        lines.extend(
            [
                "",
                "### 与正式 4211 基线对比",
                "",
                "| 模型 | Cross-PDF 答案准确率（合法） | 4211 答案准确率（合法） | 差值 |",
                "|---|---:|---:|---:|",
            ]
        )
        for model in models:
            baseline = baselines.get(model["model"])
            if baseline is None:
                continue
            current = model["overall"]["answer_acc_legal"]
            previous = float(baseline["answer_acc"])
            lines.append(
                f"| {model['display_name']} | {fmt(current)} | "
                f"{fmt(previous)} | {current - previous:+.2f} pp |"
            )

    lines.extend(
        [
            "",
            "## 分层结果",
            "",
            "### 按必要证据跳数",
            "",
            subgroup_table(models, "evidence_hops"),
            "",
            "### 按问题类型",
            "",
            subgroup_table(models, "question_type"),
            "",
            "## 数据与方法",
            "",
            "- 输入模式：Full PDF；4B 与 8B 均读取完整拼接 PDF。",
            "- 推理输出：`answer_pre` 与 `evidence_pages` 的结构化 JSON。",
            "- 答案判定：规则匹配优先，其余由同一 27B Judge 判定语义正确性。",
            "- 证据页判定：预测物理页集合必须与标准页集合精确一致；页码从 1 开始。",
            "- 联合正确：答案正确且证据页集合精确命中。",
            "- 合法口径：仅包含可解析且同时给出非空答案与页码列表的输出；端到端口径将非法输出计为失败。",
            f"- 数据门禁：评测前重新校验固定 SHA、{dataset_summary['qas']} 条 QA、"
            f"{dataset_summary['pdfs']} 个 PDF、逐题至少两份证据事实及至少两篇源论文。",
            "",
            "## 局限性与稳健性",
            "",
            "- 27B Judge 仍可能存在语义判定偏差；正式引用前应对两个模型的正确、错误和争议样本做分层人工复核。",
            "- 证据页采用精确集合匹配，额外报出一张同样有效的支持页也会被判错，因此证据页准确率比答案准确率更严格。",
            "- v3 是覆盖型挑战集，不保证源论文零复用；若用于训练/测试切分，必须按 `source_paper_ids` 分组。",
            "- 两个被测模型来自同一模型家族，不能代表所有多模态模型的普遍难度。",
            "",
            "## 建议的下一步",
            "",
            "1. 每个模型至少人工复核 5 条答案正确、5 条答案错误，并优先抽查两模型分歧项。",
            "2. 对 Judge 争议样本进行盲审，单独报告改判率和改判后的敏感性区间。",
            "3. 若任一模型端到端答案成功率超过 60%，从现有原始候选中补充新的三文档、跨表格/图像和数值链式推理题，保持最终集版本化不可变。",
            "",
            "## 复现信息",
            "",
            f"- 报告生成时间：{generated_at}",
            f"- QA：`{dataset_summary['qa_path']}`",
            f"- PDF：`{dataset_summary['pdf_dir']}`",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(
    *,
    dataset: dict[str, Any],
    dataset_summary: dict[str, Any],
    judge_4b: Path,
    judge_8b: Path,
    final_report: Path,
    baseline_report: Path | None,
    output: Path,
    output_json: Path,
) -> dict[str, Any]:
    if not final_report.is_file():
        raise FileNotFoundError(f"Standard final report not found: {final_report}")
    standard = json.loads(final_report.read_text(encoding="utf-8"))
    standard_models = {
        str(item.get("model_name"))
        for item in standard.get("models", [])
        if isinstance(item, dict)
    }
    if standard_models != {"4B", "8B"}:
        raise ValueError(
            f"Standard report model set mismatch: {sorted(standard_models)}"
        )

    qa_index = {
        (paper_id, qa_id): qa
        for paper_id, qa_id, qa in iter_dataset_items(dataset)
    }
    expected_keys = set(qa_index)
    models = [
        model_metrics(
            "4B",
            load_judge_rows(judge_4b, expected_keys),
            qa_index,
        ),
        model_metrics(
            "8B",
            load_judge_rows(judge_8b, expected_keys),
            qa_index,
        ),
    ]

    # Reconcile the independent aggregation against run_report.py.
    standard_by_model = {
        str(item["model_name"]): item["pdf_summary"]
        for item in standard["models"]
    }
    for model in models:
        expected = standard_by_model[model["model"]]
        actual = model["overall"]
        comparisons = {
            "total_seen": actual["total_seen"],
            "legal_samples": actual["legal_samples"],
            "answer_correct": actual["answer_correct"],
            "page_correct": actual["page_correct"],
            "both_correct": actual["both_correct"],
        }
        for key, value in comparisons.items():
            if int(expected[key]) != int(value):
                raise ValueError(
                    f"{model['model']} report reconciliation failed for {key}: "
                    f"run_report={expected[key]}, technical_report={value}"
                )

    generated_at = datetime.now(timezone.utc).isoformat()
    baselines = baseline_map(baseline_report)
    payload = {
        "title": "Cross-PDF QA v3 4B/8B Full-PDF evaluation",
        "generated_at": generated_at,
        "dataset": dataset_summary,
        "difficulty": {
            "criterion": (
                "maximum end-to-end answer success across 4B and 8B; "
                "<=40% high, <=60% challenging, >60% insufficient"
            ),
            "classification": difficulty_label(
                max(
                    model["overall"]["end_to_end_answer_success"]
                    for model in models
                )
            ),
        },
        "models": models,
        "baseline_report": str(baseline_report) if baseline_report else None,
        "standard_report": str(final_report),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_markdown(dataset_summary, models, baselines, generated_at),
        encoding="utf-8",
    )
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()
    qa_path = Path(args.qa_json)
    pdf_dir = Path(args.pdf_dir)
    dataset, summary = validate_dataset(
        qa_path,
        pdf_dir,
        expected_count=args.expected_count,
        expected_sha256=args.expected_sha256,
    )
    print(
        f"Cross-PDF preflight passed: {summary['qas']} QA, "
        f"{summary['bundles']} PDFs, sha256={summary['sha256']}"
    )
    if args.preflight_only:
        return

    required = {
        "--judge-4b": args.judge_4b,
        "--judge-8b": args.judge_8b,
        "--final-report": args.final_report,
        "--output": args.output,
        "--output-json": args.output_json,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError("Report mode requires " + ", ".join(missing))

    publication_validation = validate_publication_artifacts(
        dataset=dataset,
        qa_path=qa_path,
        pdf_dir=pdf_dir,
        judge_paths={
            "4B": Path(args.judge_4b),
            "8B": Path(args.judge_8b),
        },
        final_report=Path(args.final_report),
    )

    payload = generate_report(
        dataset=dataset,
        dataset_summary=summary,
        judge_4b=Path(args.judge_4b),
        judge_8b=Path(args.judge_8b),
        final_report=Path(args.final_report),
        baseline_report=(
            Path(args.baseline_report) if args.baseline_report else None
        ),
        output=Path(args.output),
        output_json=Path(args.output_json),
    )
    payload["publication_validation"] = publication_validation
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Technical report saved to: {args.output}")
    print(f"Technical report data saved to: {args.output_json}")


if __name__ == "__main__":
    main()
