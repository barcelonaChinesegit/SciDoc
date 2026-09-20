#!/usr/bin/env python3
"""Retain reasoning QA that two bound question-only models do not both solve."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--judge",
        action="append",
        required=True,
        help="MODEL=question_only_judge.json; supply at least two models.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(dataset: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(paper_id), str(qa_id)): qa
        for paper_id, paper in dataset.items()
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    }


def flatten_judge(
    judge: dict[str, Any], model_name: str
) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for paper_id, paper in judge.items():
        if not isinstance(paper, dict):
            continue
        for qa_id, row in paper.get("QA", {}).items():
            if not isinstance(row, dict):
                continue
            key = (str(paper_id), str(qa_id))
            if key in rows:
                raise ValueError(f"Duplicate {model_name} Judge key: {key}")
            if row.get("input_mode") != "question_only":
                raise ValueError(
                    f"{model_name} Judge is not question_only: {paper_id}/{qa_id}"
                )
            if not isinstance(row.get("answer_is_correct"), bool):
                raise ValueError(
                    f"{model_name} Judge lacks boolean answer score: {paper_id}/{qa_id}"
                )
            rows[key] = row
    return rows


def filter_dataset(
    dataset: dict[str, Any],
    judge_rows: dict[str, dict[tuple[str, str], dict[str, Any]]],
    judge_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(judge_rows) < 2:
        raise ValueError("At least two independent question-only Judges are required")
    expected = set(flatten(dataset))
    for model_name, rows in judge_rows.items():
        if set(rows) != expected:
            missing = sorted(expected - set(rows))[:5]
            extra = sorted(set(rows) - expected)[:5]
            raise ValueError(
                f"{model_name} Judge key mismatch: missing={missing}, extra={extra}"
            )

    output: dict[str, Any] = {}
    retained = 0
    removed = 0
    model_correct = {model_name: 0 for model_name in judge_rows}
    for paper_id, paper in dataset.items():
        output_qas: dict[str, Any] = {}
        for qa_id, qa in paper.get("QA", {}).items():
            key = (str(paper_id), str(qa_id))
            verdicts: dict[str, dict[str, Any]] = {}
            for model_name, rows in judge_rows.items():
                row = rows[key]
                if row.get("question") != qa.get("question"):
                    raise ValueError(f"{model_name} question mismatch: {paper_id}/{qa_id}")
                if row.get("correct_answer") != qa.get("answer"):
                    raise ValueError(f"{model_name} gold-answer mismatch: {paper_id}/{qa_id}")
                correct = bool(row["answer_is_correct"])
                model_correct[model_name] += int(correct)
                verdicts[model_name] = {
                    "answer_is_correct": correct,
                    "parsed_answer": row.get("parsed_answer"),
                    "evaluated_model": row.get("evaluated_model"),
                    "inference_binding_sha256": row.get(
                        "inference_binding_sha256"
                    ),
                    "judge_protocol_fingerprint": row.get(
                        "judge_protocol_fingerprint"
                    ),
                }
            both_correct = all(
                verdict["answer_is_correct"] for verdict in verdicts.values()
            )
            if both_correct:
                removed += 1
                continue
            enriched = deepcopy(qa)
            enriched["closed_book_pdf_dependency"] = {
                "policy": (
                    "retained because the bound question-only models did not "
                    "all answer correctly"
                ),
                "models": verdicts,
                "both_models_correct": False,
                "judge_sha256": judge_hashes,
            }
            output_qas[str(qa_id)] = enriched
            retained += 1
        if output_qas:
            output[str(paper_id)] = {
                key: deepcopy(value)
                for key, value in paper.items()
                if key != "QA"
            }
            output[str(paper_id)]["QA"] = output_qas
    total = len(expected)
    summary = {
        "status": "complete",
        "policy": "remove only when all bound question-only models are correct",
        "input_questions": total,
        "retained_questions": retained,
        "removed_both_correct": removed,
        "model_answer_accuracy": {
            model_name: model_correct[model_name] / total if total else 0.0
            for model_name in sorted(model_correct)
        },
        "judge_sha256": judge_hashes,
    }
    return output, summary


def parse_judge_specs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        model_name, separator, raw_path = spec.partition("=")
        if not separator or not model_name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --judge value: {spec!r}")
        if model_name in result:
            raise ValueError(f"Duplicate Judge model name: {model_name}")
        result[model_name] = Path(raw_path)
    return result


def main() -> None:
    args = parse_args()
    dataset = read_json(args.input)
    judge_paths = parse_judge_specs(args.judge)
    judges = {name: read_json(path) for name, path in judge_paths.items()}
    hashes = {name: sha256_file(path) for name, path in judge_paths.items()}
    filtered, summary = filter_dataset(
        dataset,
        {name: flatten_judge(value, name) for name, value in judges.items()},
        hashes,
    )
    atomic_json(args.output, filtered)
    summary_path = args.summary_output or args.output.with_name(
        args.output.stem + "_summary.json"
    )
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
