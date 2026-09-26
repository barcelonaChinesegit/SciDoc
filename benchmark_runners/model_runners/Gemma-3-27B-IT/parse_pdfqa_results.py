#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Dict

import gemma3_core as c
from run_pdfqa import (
    atomic_write_json,
    strict_parse_output,
)


TECHNICAL_STATUSES = {
    "oom_single_gpu",
    "context_guard_exceeded",
    "error",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Strict post-hoc parser for Gemma-3-27B-IT PDF-QA"
    )

    p.add_argument(
        "--dataset-id",
        required=True,
        choices=list(c.DATASETS),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    inp = c.raw_output_path(
        args.dataset_id
    )

    out = c.parsed_output_path(
        args.dataset_id
    )

    if not inp.is_file():
        raise FileNotFoundError(
            f"Raw result not found: {inp}"
        )

    with inp.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    parsed_data = copy.deepcopy(
        data
    )

    counts: Dict[str, int] = {
        "strict": 0,
        "normalized_strict": 0,
        "failed": 0,
        "technical": 0,
        "not_run": 0,
    }

    for unit_id, unit in parsed_data.items():

        if not isinstance(
            unit,
            dict,
        ):
            continue

        qa_map = unit.get(
            "QA"
        )

        if not isinstance(
            qa_map,
            dict,
        ):
            continue

        for qa_id, qa in qa_map.items():

            if not isinstance(
                qa,
                dict,
            ):
                continue

            status = str(
                qa.get(
                    "status",
                    "",
                )
            )

            if not status:
                counts[
                    "not_run"
                ] += 1
                continue

            if status in TECHNICAL_STATUSES:
                qa[
                    "posthoc_parse_status"
                ] = "technical_failure"

                counts[
                    "technical"
                ] += 1
                continue

            raw = str(
                qa.get(
                    "answer_pre_raw",
                    "",
                )
            )

            page_count = int(
                qa.get(
                    "pdf_total_pages",
                    0,
                )
                or 0
            )

            if page_count <= 0:
                qa[
                    "posthoc_parse_status"
                ] = "missing_page_count"

                qa[
                    "posthoc_parse_ok"
                ] = False

                counts[
                    "failed"
                ] += 1

                continue

            (
                ok,
                reason,
                parsed,
                normalized,
                violations,
            ) = strict_parse_output(
                raw,
                page_count=page_count,
            )

            qa[
                "posthoc_parse_ok"
            ] = bool(ok)

            qa[
                "posthoc_parse_reason"
            ] = reason

            qa[
                "posthoc_normalized"
            ] = bool(normalized)

            qa[
                "posthoc_schema_violations"
            ] = violations

            if ok:
                answer = (
                    parsed[
                        "answer_pre"
                    ].strip()
                )

                pages = list(
                    parsed[
                        "evidence_pages"
                    ]
                )

                # Verification only:
                # parsed values must agree with the runner values.
                saved_answer = str(
                    qa.get(
                        "answer_pre",
                        "",
                    )
                ).strip()

                saved_pages = qa.get(
                    "evidence_pages_pre",
                    [],
                )

                if (
                    saved_answer
                    and saved_answer != answer
                ):
                    raise RuntimeError(
                        f"{unit_id}/{qa_id}: "
                        "posthoc answer differs from runner parse"
                    )

                if (
                    isinstance(
                        saved_pages,
                        list,
                    )
                    and saved_pages
                    and saved_pages != pages
                ):
                    raise RuntimeError(
                        f"{unit_id}/{qa_id}: "
                        "posthoc evidence differs from runner parse"
                    )

                qa[
                    "answer_pre"
                ] = answer

                qa[
                    "evidence_pages_pre"
                ] = pages

                if normalized:
                    qa[
                        "posthoc_parse_status"
                    ] = "normalized_strict"

                    counts[
                        "normalized_strict"
                    ] += 1

                else:
                    qa[
                        "posthoc_parse_status"
                    ] = "strict"

                    counts[
                        "strict"
                    ] += 1

            else:
                qa[
                    "posthoc_parse_status"
                ] = "failed"

                counts[
                    "failed"
                ] += 1

    atomic_write_json(
        out,
        parsed_data,
    )

    print(
        f"[Parse] {args.dataset_id} | "
        f"{counts}"
    )

    print(
        f"[Saved] {out}"
    )


if __name__ == "__main__":
    main()
