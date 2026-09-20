#!/usr/bin/env python3
"""Compute joint answer/evidence accuracy and closed-book-aware weighted accuracy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from evaluation_protocol import (
    PDF_INPUT_MODE,
    QUESTION_ONLY_INPUT_MODE,
    pdf_corpus_sha256_from_manifest,
    sha256_file,
)
from run_report import validate_judge_against_gold


MODELS = ("4B", "8B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--pdf-judge-dir", required=True)
    parser.add_argument("--closedbook-judge-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_qa(data: dict[str, Any]):
    for paper_id, paper_data in sorted(data.items(), key=lambda item: str(item[0])):
        if str(paper_id).startswith("__") or not isinstance(paper_data, dict):
            continue
        for qa_id, qa_data in sorted(paper_data.get("QA", {}).items(), key=lambda item: str(item[0])):
            yield str(paper_id), str(qa_id), qa_data


def lookup(data: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(paper_id, qa_id): qa for paper_id, qa_id, qa in iter_qa(data)}


def _pdf_hash_manifest(judged: dict[str, Any]) -> dict[str, str]:
    """Extract one non-empty PDF hash per paper from a validated Judge file."""
    hashes_by_paper: dict[str, set[str]] = {}
    for paper_id, _qa_id, row in iter_qa(judged):
        value = str(row.get("pdf_sha256", "") or "").strip()
        hashes_by_paper.setdefault(paper_id, set()).add(value)
    manifest: dict[str, str] = {}
    for paper_id, values in hashes_by_paper.items():
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(
                f"PDF Judge rows have no single non-empty pdf_sha256 for "
                f"paper {paper_id}"
            )
        manifest[paper_id] = next(iter(values))
    return manifest


def validate_judge_inputs(
    *,
    qa_path: Path,
    qa_data: dict[str, Any],
    pdf_judge_dir: Path,
    closedbook_judge_dir: Path,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    """Validate all four Judge artifacts before any weighted score is used.

    ``run_report.validate_judge_against_gold`` is the project's publication
    validator.  It locks the exact QA key set and independently reparses every
    raw model output.  This wrapper additionally requires strict source locks
    and verifies that 4B and 8B used the same PDF bytes.
    """
    source_sha256 = sha256_file(qa_path)
    mode_dirs = {
        PDF_INPUT_MODE: pdf_judge_dir,
        QUESTION_ONLY_INPUT_MODE: closedbook_judge_dir,
    }
    judged_by_mode: dict[str, dict[str, dict[str, Any]]] = {}
    profiles: dict[str, dict[str, Any]] = {}
    source_hashes: set[str] = set()
    pdf_corpus_hashes: set[str] = set()
    pdf_manifests: dict[str, dict[str, str]] = {}

    for mode, directory in mode_dirs.items():
        judged_by_mode[mode] = {}
        profiles[mode] = {}
        for model in MODELS:
            judge_path = directory / f"judge_{model}.json"
            inference_path = directory / f"results_{model}.json"
            profile = validate_judge_against_gold(
                judge_path,
                qa_data,
                inference_path=inference_path,
                expected_model=model,
            )
            if profile.get("input_mode") != mode:
                raise ValueError(
                    f"{judge_path}: expected input_mode={mode}, found "
                    f"{profile.get('input_mode')!r}"
                )
            fingerprint = str(
                profile.get("protocol_fingerprint", "") or ""
            ).strip()
            source_hash = str(
                profile.get("qa_source_sha256", "") or ""
            ).strip()
            if not fingerprint:
                raise ValueError(
                    f"{judge_path}: publication report requires a protocol "
                    "fingerprint"
                )
            if not str(
                profile.get("judge_protocol_fingerprint", "") or ""
            ).strip():
                raise ValueError(
                    f"{judge_path}: publication report requires a Judge "
                    "protocol fingerprint"
                )
            if source_hash != source_sha256:
                raise ValueError(
                    f"{judge_path}: qa_source_sha256 does not match --qa-json"
                )
            source_hashes.add(source_hash)

            judged = load_json(judge_path)
            judged_by_mode[mode][model] = judged
            audit_profile = {
                **profile,
                "judge_file": str(judge_path),
                "judge_file_sha256": sha256_file(judge_path),
            }
            if mode == PDF_INPUT_MODE:
                corpus_hash = str(
                    profile.get("pdf_corpus_sha256", "") or ""
                ).strip()
                if not corpus_hash:
                    raise ValueError(
                        f"{judge_path}: PDF evaluation has no corpus hash"
                    )
                pdf_corpus_hashes.add(corpus_hash)
                manifest = _pdf_hash_manifest(judged)
                pdf_manifests[model] = manifest
                audit_profile["pdf_sha256_by_paper"] = manifest
            profiles[mode][model] = audit_profile

    if source_hashes != {source_sha256}:
        raise ValueError("Judge artifacts do not share the current QA source")
    if len(pdf_corpus_hashes) != 1:
        raise ValueError(
            "4B and 8B PDF Judge artifacts use different PDF corpus hashes"
        )
    if pdf_manifests.get("4B") != pdf_manifests.get("8B"):
        raise ValueError(
            "4B and 8B PDF Judge artifacts use different per-paper PDF hashes"
        )
    if next(iter(pdf_corpus_hashes)) != pdf_corpus_sha256_from_manifest(
        pdf_manifests["4B"]
    ):
        raise ValueError(
            "PDF corpus hash is inconsistent with per-paper PDF hashes"
        )

    validation = {
        "status": "passed",
        "qa_source_sha256": source_sha256,
        "pdf_corpus_sha256": next(iter(pdf_corpus_hashes)),
        "pdf_sha256_by_paper": pdf_manifests["4B"],
        "profiles": profiles,
    }
    return judged_by_mode, validation


def score_model(
    model: str,
    qa_data: dict[str, Any],
    pdf_judge: dict[str, Any],
    closedbook_judge: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    pdf = lookup(pdf_judge)
    closedbook = lookup(closedbook_judge)
    rows = []
    counts = {
        "total": 0,
        "pdf_joint_correct": 0,
        "pdf_answer_correct": 0,
        "pdf_evidence_correct": 0,
        "closedbook_joint_correct": 0,
        "downweighted_correct": 0,
        "missing_pdf_judge": 0,
        "missing_closedbook_judge": 0,
    }
    weighted_score = 0.0

    for paper_id, qa_id, source in iter_qa(qa_data):
        counts["total"] += 1
        key = (paper_id, qa_id)
        pdf_item = pdf.get(key)
        closedbook_item = closedbook.get(key)
        if pdf_item is None:
            counts["missing_pdf_judge"] += 1
            pdf_item = {}
        if closedbook_item is None:
            counts["missing_closedbook_judge"] += 1
            closedbook_item = {}

        pdf_joint = pdf_item.get("is_correct") is True
        pdf_answer = pdf_item.get("answer_is_correct") is True
        pdf_evidence = pdf_item.get("evidence_pages_is_correct") is True
        closedbook_joint = closedbook_item.get("is_correct") is True
        if pdf_joint:
            counts["pdf_joint_correct"] += 1
        if pdf_answer:
            counts["pdf_answer_correct"] += 1
        if pdf_evidence:
            counts["pdf_evidence_correct"] += 1
        if closedbook_joint:
            counts["closedbook_joint_correct"] += 1

        score = 0.0
        if pdf_joint:
            if closedbook_joint:
                score = 0.5
                counts["downweighted_correct"] += 1
            else:
                score = 1.0
        weighted_score += score
        rows.append(
            {
                "model": model,
                "paper_id": paper_id,
                "qa_id": qa_id,
                "pdf_answer_correct": pdf_answer,
                "pdf_evidence_pages_correct": pdf_evidence,
                "pdf_joint_correct": pdf_joint,
                "closedbook_joint_correct": closedbook_joint,
                "raw_score": int(pdf_joint),
                "weighted_score": score,
                "question": source.get("question", ""),
                "reference_answer": source.get("answer", ""),
                "reference_evidence_pages": json.dumps(source.get("evidence_pages", []), ensure_ascii=False),
                "pdf_model_output": pdf_item.get("model_output", ""),
                "closedbook_model_output": closedbook_item.get("model_output", ""),
            }
        )

    total = counts["total"]
    result = {
        "model": model,
        **counts,
        "pdf_joint_accuracy_percent": counts["pdf_joint_correct"] / total * 100 if total else 0.0,
        "pdf_answer_accuracy_percent": counts["pdf_answer_correct"] / total * 100 if total else 0.0,
        "pdf_evidence_accuracy_percent": counts["pdf_evidence_correct"] / total * 100 if total else 0.0,
        "closedbook_joint_accuracy_percent": counts["closedbook_joint_correct"] / total * 100 if total else 0.0,
        "weighted_score": weighted_score,
        "weighted_accuracy_percent": weighted_score / total * 100 if total else 0.0,
    }
    result["penalty_points"] = result["pdf_joint_accuracy_percent"] - result["weighted_accuracy_percent"]

    csv_path = output_dir / f"{model}_per_question_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    result["per_question_csv"] = str(csv_path)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = Path(args.qa_json)
    qa_data = load_json(qa_path)
    judged, validation = validate_judge_inputs(
        qa_path=qa_path,
        qa_data=qa_data,
        pdf_judge_dir=Path(args.pdf_judge_dir),
        closedbook_judge_dir=Path(args.closedbook_judge_dir),
    )
    results = []
    for model in MODELS:
        results.append(
            score_model(
                model,
                qa_data,
                judged[PDF_INPUT_MODE][model],
                judged[QUESTION_ONLY_INPUT_MODE][model],
                output_dir,
            )
        )

    summary = {
        "joint_correctness_rule": "answer_is_correct AND exact evidence_pages set is correct",
        "weighted_formula": {
            "pdf_joint_incorrect": 0,
            "pdf_joint_correct_and_closedbook_joint_incorrect": 1,
            "pdf_joint_correct_and_closedbook_joint_correct": 0.5,
        },
        "inputs": {
            "qa_json": args.qa_json,
            "pdf_judge_dir": args.pdf_judge_dir,
            "closedbook_judge_dir": args.closedbook_judge_dir,
        },
        "validation": validation,
        "results": results,
    }
    write_json(output_dir / "summary.json", summary)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model", "total", "pdf_joint_correct", "pdf_joint_accuracy_percent",
            "pdf_answer_accuracy_percent", "pdf_evidence_accuracy_percent",
            "closedbook_joint_correct", "closedbook_joint_accuracy_percent",
            "downweighted_correct", "weighted_score", "weighted_accuracy_percent",
            "penalty_points", "missing_pdf_judge", "missing_closedbook_judge",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in results])

    lines = [
        "# 7509题答案与证据页联合评测",
        "",
        "正确条件：模型答案正确，并且预测证据页集合与标准证据页集合完全一致。",
        "加权规则：带 PDF 联合错误记 0 分；带 PDF 联合正确且闭卷联合错误记 1 分；两边联合正确记 0.5 分。",
        "",
        "| 模型 | 总题数 | 答案正确率 | 证据页正确率 | 带PDF联合正确率 | 闭卷联合正确率 | 降权题数 | 加权正确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['total']} | {row['pdf_answer_accuracy_percent']:.2f}% | "
            f"{row['pdf_evidence_accuracy_percent']:.2f}% | {row['pdf_joint_correct']}/{row['total']} = "
            f"{row['pdf_joint_accuracy_percent']:.2f}% | {row['closedbook_joint_correct']}/{row['total']} = "
            f"{row['closedbook_joint_accuracy_percent']:.2f}% | {row['downweighted_correct']} | "
            f"{row['weighted_score']:.1f}/{row['total']} = {row['weighted_accuracy_percent']:.2f}% |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
