#!/usr/bin/env python3
"""Diagnose answer accuracy for externally produced PDF-mode result JSON.

This is a selection diagnostic, not a publication evaluator.  It preserves the
project's canonical ``pdf`` input/output semantics and uses the same strict
semantic-answer prompt as ``evaluation/run_judge.py``.  Publication reports
must still use the fully bound inference/Judge pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from pku_qa.evaluation.eval_framework import (
    atomic_write_json,
    create_provider,
)
from pku_qa.evaluation.run_judge import direct_fill_match, judge_fill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--judge-provider-name", default="local_qwen3_6_27b_judge"
    )
    parser.add_argument("--provider-config")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--answerable-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_qas(
    dataset: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Invalid paper record: {paper_id}")
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(f"Invalid QA record: {paper_id}/{qa_id}")
            yield str(paper_id), str(qa_id), qa


def prediction_from_result(qa: dict[str, Any]) -> tuple[str, str]:
    """Return prediction and extraction status for the stored API outputs."""
    if qa.get("failed") is True or str(qa.get("status", "")).lower() in {
        "failed",
        "error",
    }:
        return "", "failed"
    for field in ("prediction_answer", "answer_pre"):
        value = qa.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", "missing"


def row_binding_sha256(question: str, gold: str, prediction: str) -> str:
    payload = json.dumps(
        [question, gold, prediction],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_rows(
    gold: dict[str, Any],
    results: dict[str, Any],
    *,
    answerable_only: bool,
) -> list[dict[str, Any]]:
    result_index = {
        (paper_id, qa_id): qa for paper_id, qa_id, qa in iter_qas(results)
    }
    rows: list[dict[str, Any]] = []
    for paper_id, qa_id, gold_qa in iter_qas(gold):
        answer = str(gold_qa.get("answer", "")).strip()
        question = str(gold_qa.get("question", "")).strip()
        if not question or not answer:
            raise ValueError(f"Empty gold question/answer: {paper_id}/{qa_id}")
        if answerable_only and answer == "Unanswerable":
            continue
        result_qa = result_index.get((paper_id, qa_id))
        if result_qa is None:
            prediction, extraction = "", "missing_result_row"
        else:
            result_question = str(result_qa.get("question", "")).strip()
            if result_question and result_question != question:
                raise ValueError(f"Question mismatch: {paper_id}/{qa_id}")
            prediction, extraction = prediction_from_result(result_qa)
        rows.append(
            {
                "item_id": f"{paper_id}/{qa_id}",
                "paper_id": paper_id,
                "qa_id": qa_id,
                "question": question,
                "gold_answer": answer,
                "prediction_answer": prediction,
                "prediction_extraction": extraction,
                "binding_sha256": row_binding_sha256(
                    question, answer, prediction
                ),
            }
        )
    return rows


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    correct = sum(row.get("answer_correct") is True for row in rows)
    completed = sum(isinstance(row.get("answer_correct"), bool) for row in rows)
    return {
        "label": payload["label"],
        "selected": len(rows),
        "completed": completed,
        "answer_correct": correct,
        "answer_accuracy": correct / len(rows) if rows else 0.0,
        "methods": {
            method: sum(row.get("match_method") == method for row in rows)
            for method in sorted(
                {
                    str(row.get("match_method"))
                    for row in rows
                    if row.get("match_method")
                }
            )
        },
    }


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard configuration")

    gold = read_json(args.qa_json)
    results = read_json(args.result_json)
    all_rows = build_rows(
        gold, results, answerable_only=args.answerable_only
    )
    selected = [
        row
        for index, row in enumerate(all_rows)
        if index % args.num_shards == args.shard_id
    ]

    existing_by_binding: dict[str, dict[str, Any]] = {}
    if args.output_json.is_file():
        existing = read_json(args.output_json)
        existing_by_binding = {
            str(row.get("binding_sha256")): row
            for row in existing.get("rows", [])
            if isinstance(row, dict)
            and isinstance(row.get("answer_correct"), bool)
        }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "non_publication_selection_diagnostic",
        "input_mode": "pdf",
        "label": args.label,
        "qa_json": str(args.qa_json.resolve()),
        "qa_json_sha256": sha256_file(args.qa_json),
        "result_json": str(args.result_json.resolve()),
        "result_json_sha256": sha256_file(args.result_json),
        "judge_provider_name": args.judge_provider_name,
        "answerable_only": args.answerable_only,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "rows": [],
    }

    provider = None
    for row in selected:
        cached = existing_by_binding.get(row["binding_sha256"])
        if cached is not None:
            payload["rows"].append(cached)
            continue
        prediction = row["prediction_answer"]
        if not prediction:
            row["answer_correct"] = False
            row["match_method"] = row["prediction_extraction"]
        else:
            direct, method = direct_fill_match(
                row["gold_answer"], prediction
            )
            if direct is None:
                if provider is None:
                    provider = create_provider(
                        args.judge_provider_name,
                        config_path=args.provider_config,
                    )
                verdict, raw = judge_fill(
                    provider,
                    row["question"],
                    row["gold_answer"],
                    prediction,
                    args.max_new_tokens,
                )
                row["answer_correct"] = verdict == "CORRECT"
                row["match_method"] = "llm_semantic_judge"
                row["judge_raw"] = raw
            else:
                row["answer_correct"] = direct
                row["match_method"] = method
        payload["rows"].append(row)
        payload["summary"] = summarize(payload)
        atomic_write_json(args.output_json, payload)

    payload["summary"] = summarize(payload)
    atomic_write_json(args.output_json, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
