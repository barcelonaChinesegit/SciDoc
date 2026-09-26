#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from settings.pdfqa_benchmark_config import (
    DATASETS,
    PDF_DPI,
)

from settings.pdfqa_prompts import (
    PROMPT_VERSION,
)


MODEL_DIR = Path(__file__).resolve().parent

RAW_DIR = (
    MODEL_DIR
    / "output"
    / "raw_result"
)

PARSED_DIR = (
    MODEL_DIR
    / "output"
    / "parsed_result"
)

RUN_VERSION = "run-v1"

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|assistant|>",
    "<|user|>",
)


def basename(
    dataset_id: str,
) -> str:

    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi{PDF_DPI}"
        f"__{RUN_VERSION}.json"
    )


def atomic_write_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(f.fileno())

        tmp_path = Path(f.name)

    os.replace(
        tmp_path,
        path,
    )


def clean_text(
    raw: str,
) -> str:

    text = str(
        raw or ""
    ).strip()

    for token in SPECIAL_TOKENS:
        text = text.replace(
            token,
            "",
        )

    return text.strip()


def normalize_pages(
    raw: Any,
) -> List[int]:

    if (
        raw is None
        or isinstance(raw, bool)
    ):
        return []

    items = (
        raw
        if isinstance(raw, list)
        else [raw]
    )

    pages: List[int] = []

    for item in items:

        if isinstance(item, bool):
            continue

        if isinstance(item, int):
            values = [item]
        else:
            values = re.findall(
                r"\d+",
                str(item),
            )

        for value in values:
            try:
                page = int(value)
            except Exception:
                continue

            if (
                page > 0
                and page not in pages
            ):
                pages.append(page)

    return sorted(pages)


def escape_invalid_json_backslashes(
    text: str,
) -> str:

    return re.sub(
        r'\\(?!["\\/bfnrtu])',
        r'\\\\',
        text,
    )


def parse_prediction(
    raw: str,
) -> tuple[
    Optional[Dict[str, Any]],
    str,
]:

    text = clean_text(raw)

    if not text:
        return None, "empty"

    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.S | re.I,
    )

    if fenced:
        text = fenced.group(1).strip()

    candidates = [text]

    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and end > start
    ):
        candidates.append(
            text[start:end + 1]
        )

    # 1. Standard JSON
    for candidate in candidates:
        try:
            obj = json.loads(candidate)

            if isinstance(obj, dict):
                return obj, "json"

        except Exception:
            pass

    # 2. Repair LaTeX backslashes
    for candidate in candidates:
        try:
            repaired = (
                escape_invalid_json_backslashes(
                    candidate
                )
            )

            obj = json.loads(repaired)

            if isinstance(obj, dict):
                return (
                    obj,
                    "json_backslash_repair",
                )

        except Exception:
            pass

    # 3. Python-style dict fallback
    for candidate in candidates:
        try:
            obj = ast.literal_eval(
                candidate
            )

            if isinstance(obj, dict):
                return (
                    obj,
                    "literal_eval",
                )

        except Exception:
            pass

    return None, "parse_failed"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Parse PDF-QA raw outputs."
        )
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        choices=list(DATASETS),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    raw_path = (
        RAW_DIR
        / basename(
            args.dataset_id
        )
    )

    out_path = (
        PARSED_DIR
        / basename(
            args.dataset_id
        )
    )

    if not raw_path.is_file():
        raise FileNotFoundError(
            raw_path
        )

    with raw_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    out = copy.deepcopy(data)

    counts: Dict[str, int] = {}

    for unit in out.values():

        if (
            not isinstance(unit, dict)
            or not isinstance(
                unit.get("QA"),
                dict,
            )
        ):
            continue

        for qa in unit["QA"].values():

            if not isinstance(qa, dict):
                continue

            raw = qa.get(
                "answer_pre_raw",
                "",
            )

            obj, mode = (
                parse_prediction(raw)
            )

            status = "ok"

            if obj is None:
                answer = ""
                pages: List[int] = []
                status = mode

            else:
                answer = str(
                    obj.get(
                        "answer_pre",
                        "",
                    )
                ).strip()

                pages = normalize_pages(
                    obj.get(
                        "evidence_pages"
                    )
                )

                if (
                    "answer_pre" not in obj
                    or "evidence_pages"
                    not in obj
                ):
                    status = (
                        "missing_required_key"
                    )

                elif not answer:
                    status = "empty_answer"

            pdf_total = qa.get(
                "pdf_total_pages"
            )

            invalid_pages: List[int] = []

            if (
                isinstance(pdf_total, int)
                and pdf_total > 0
            ):
                invalid_pages = [
                    page
                    for page in pages
                    if page > pdf_total
                ]

                if (
                    invalid_pages
                    and status == "ok"
                ):
                    status = (
                        "page_out_of_range"
                    )

            qa["answer_pre"] = answer

            qa[
                "evidence_pages_pre"
            ] = pages

            qa[
                "prediction_parse_mode"
            ] = mode

            qa[
                "prediction_parse_status"
            ] = status

            qa[
                "predicted_page_issues"
            ] = [
                f"out_of_range:{page}"
                for page in invalid_pages
            ]

            qa[
                "parsed_valid"
            ] = (
                status == "ok"
            )

            counts[status] = (
                counts.get(
                    status,
                    0,
                )
                + 1
            )

    atomic_write_json(
        out_path,
        out,
    )

    print(
        f"[Parsed] "
        f"{args.dataset_id}"
        f" -> {out_path}"
    )

    print(
        f"[Status] {counts}"
    )


if __name__ == "__main__":
    main()
