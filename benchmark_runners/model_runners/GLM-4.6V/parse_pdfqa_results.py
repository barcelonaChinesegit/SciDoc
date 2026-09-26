#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import run_pdfqa as runner

HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "output" / "raw_result"
PARSED_ROOT = HERE / "output" / "parsed_result"
MANIFEST_ROOT = HERE / "output" / "manifests"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def result_name(dataset_id: str, dpi: int) -> str:
    return (
        f"{dataset_id}__{runner.PROMPT_VERSION}"
        f"__dpi{dpi}__{runner.RUN_VERSION}.json"
    )


def parse_dataset(dataset_id: str, dpi: int) -> None:
    spec = runner.DATASETS[dataset_id]
    raw_path = RAW_ROOT / result_name(dataset_id, dpi)

    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    raw = load_json(raw_path)
    parsed = copy.deepcopy(raw)
    counts = Counter()

    for uid, unit, qid, qa in runner.iter_qa(parsed):
        status = qa.get("status")

        if status == "error":
            counts["error"] += 1
            continue

        raw_text = str(qa.get("answer_pre_raw", "") or "")

        if status not in {
            "completed",
            "normalized_completed",
            "parse_failed",
        } and not raw_text:
            counts["not_run"] += 1
            continue

        page_count = qa.get("pdf_total_pages")
        try:
            page_count = int(page_count) if page_count is not None else None
        except Exception:
            page_count = None

        ok, answer, pages, reason = runner.parse_prediction(
            raw_text, page_count
        )

        qa["parsed_valid"] = bool(ok)
        qa["prediction_parse_reason"] = reason
        qa["prediction_parse_status"] = "ok" if ok else reason

        if ok:
            qa["answer_pre"] = answer
            qa["evidence_pages_pre"] = pages

            if reason == "strict_ok":
                qa["status"] = "completed"
                qa["prediction_parse_mode"] = "strict_json"
                counts["strict"] += 1
            else:
                qa["status"] = "normalized_completed"
                qa["prediction_parse_mode"] = "deterministic_normalization"
                counts["normalized_strict"] += 1
        else:
            qa["status"] = "parse_failed"
            qa["prediction_parse_mode"] = "failed"
            qa["answer_pre"] = ""
            qa["evidence_pages_pre"] = []
            counts["failed"] += 1

        qa["format_repair_used"] = False
        qa["format_repair_raw_output"] = ""

    PARSED_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)

    parsed_path = PARSED_ROOT / result_name(dataset_id, dpi)
    atomic_write(parsed_path, parsed)

    manifest = {
        "dataset_id": dataset_id,
        "source_dataset": str(spec["data"]),
        "source_dataset_sha256": sha256_file(spec["data"]),
        "raw_result": str(raw_path),
        "parsed_result": str(parsed_path),
        "model_name": runner.MODEL_NAME,
        "model_id": runner.DEFAULT_MODEL,
        "api_provider": "zhizengzeng_openai_compatible",
        "prompt_id": spec["prompt_id"],
        "prompt_version": runner.PROMPT_VERSION,
        "run_version": runner.RUN_VERSION,
        "render_dpi": dpi,
        "whole_pdf": True,
        "page_dropping": False,
        "thinking_mode": "disabled",
        "format_repair": False,
        "parse_policy": [
            "strict whole-output JSON",
            "whole-output Markdown JSON fence normalization",
            "exactly one already-valid JSON object extracted with JSONDecoder.raw_decode",
            "no regex extraction",
            "no invalid-JSON repair",
            "no semantic repair",
            "no second model call for formatting",
        ],
        "counts": dict(counts),
        "expected_qa": spec["expected"],
    }

    manifest_path = (
        MANIFEST_ROOT
        / result_name(dataset_id, dpi).replace(".json", ".manifest.json")
    )
    atomic_write(manifest_path, manifest)

    print(f"[Parse] {dataset_id} | {dict(counts)}")
    print(f"[Saved] {parsed_path}")
    print(f"[Manifest] {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-id",
        choices=list(runner.DATASETS),
        required=True,
    )
    p.add_argument("--dpi", type=int, default=144)
    args = p.parse_args()
    parse_dataset(args.dataset_id, args.dpi)


if __name__ == "__main__":
    main()
