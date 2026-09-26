#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import minicpm26_core as c


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
        tmp = Path(f.name)

    os.replace(tmp, path)


def strip_fence(
    text: str,
) -> tuple[str, bool]:
    s = str(
        text or ""
    ).strip()

    if not s:
        return "", False

    lines = s.splitlines()

    if len(lines) >= 3:
        first = (
            lines[0]
            .strip()
            .lower()
        )
        last = (
            lines[-1]
            .strip()
        )

        if (
            first
            in {
                "```",
                "```json",
            }
            and last
            == "```"
        ):
            return (
                "\n".join(
                    lines[1:-1]
                ).strip(),
                True,
            )

    return s, False


def balanced_objects(
    text: str,
) -> List[str]:
    out = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(
        text
    ):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False

            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i

            depth += 1

        elif (
            ch == "}"
            and depth > 0
        ):
            depth -= 1

            if (
                depth == 0
                and start
                is not None
            ):
                out.append(
                    text[
                        start : i + 1
                    ]
                )

                start = None

    return out


def candidate_dicts(
    raw: str,
):
    stripped, fenced = (
        strip_fence(raw)
    )

    texts = [
        (
            raw.strip(),
            "raw_json",
        ),
        (
            stripped,
            (
                "markdown_fence_normalized"
                if fenced
                else "stripped_json"
            ),
        ),
    ]

    for obj_text in balanced_objects(
        raw
    ):
        texts.append(
            (
                obj_text,
                "embedded_json",
            )
        )

    seen = set()

    for text, source in texts:
        if (
            not text
            or text in seen
        ):
            continue

        seen.add(
            text
        )

        try:
            obj = json.loads(
                text
            )

            if isinstance(
                obj,
                dict,
            ):
                yield source, obj
                continue

        except Exception:
            pass

        try:
            obj = ast.literal_eval(
                text
            )

            if isinstance(
                obj,
                dict,
            ):
                yield (
                    "python_literal",
                    obj,
                )

        except Exception:
            pass


def norm_page(
    value: Any,
):
    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        int,
    ):
        return value

    if (
        isinstance(
            value,
            float,
        )
        and value.is_integer()
    ):
        return int(
            value
        )

    if isinstance(
        value,
        str,
    ):
        s = value.strip()

        if re.fullmatch(
            r"\d+",
            s,
        ):
            return int(
                s
            )

        for pattern in (
            r"\[?Page\s+(\d+)\]?",
            r"PDF_PAGE_(\d+)",
            r"PAGE_(\d+)",
        ):
            m = re.fullmatch(
                pattern,
                s,
                flags=re.I,
            )

            if m:
                return int(
                    m.group(1)
                )

    return None


def parse_record(
    raw: str,
    pdf_pages: int | None,
) -> Dict[str, Any]:
    raw = raw or ""

    answer = None
    answer_status = "missing"

    pages = None
    evidence_status = "missing"

    source = ""

    for src, obj in candidate_dicts(
        raw
    ):
        if (
            "answer_pre"
            in obj
            and isinstance(
                obj["answer_pre"],
                str,
            )
        ):
            answer = obj[
                "answer_pre"
            ]
            answer_status = (
                "exact"
            )

        elif (
            "answer"
            in obj
            and isinstance(
                obj["answer"],
                str,
            )
        ):
            answer = obj[
                "answer"
            ]
            answer_status = (
                "alias_answer"
            )

        if (
            "evidence_pages"
            in obj
        ):
            value = obj[
                "evidence_pages"
            ]

            if isinstance(
                value,
                list,
            ):
                temp = [
                    norm_page(
                        x
                    )
                    for x in value
                ]

                if all(
                    x is not None
                    for x in temp
                ):
                    pages = [
                        int(x)
                        for x in temp
                    ]

                    native = all(
                        isinstance(
                            x,
                            int,
                        )
                        and not isinstance(
                            x,
                            bool,
                        )
                        for x
                        in value
                    )

                    evidence_status = (
                        "exact"
                        if native
                        else "normalized"
                    )

                else:
                    evidence_status = (
                        "invalid_nonphysical_values"
                    )

            else:
                evidence_status = (
                    "invalid_not_list"
                )

        source = src

        if (
            answer is not None
            or evidence_status
            != "missing"
        ):
            break

    if (
        answer is None
        and raw.strip()
        and not any(
            token in raw
            for token in (
                "{",
                "}",
                "<think>",
                "PDF_PAGE_",
                "[Page ",
            )
        )
    ):
        answer = raw.strip()

        answer_status = (
            "natural_language_candidate"
        )

    bad = []

    if (
        pages is not None
        and isinstance(
            pdf_pages,
            int,
        )
        and pdf_pages > 0
    ):
        bad = [
            p
            for p in pages
            if (
                p < 1
                or p > pdf_pages
            )
        ]

    evidence_valid = (
        pages is not None
        and not bad
    )

    schema_valid = (
        answer_status
        == "exact"
        and evidence_status
        in {
            "exact",
            "normalized",
        }
        and evidence_valid
    )

    if schema_valid:
        if source == (
            "markdown_fence_normalized"
        ):
            parse_status = (
                "normalized_strict"
            )
        elif evidence_status == (
            "normalized"
        ):
            parse_status = (
                "normalized_strict"
            )
        else:
            parse_status = (
                "strict"
            )

    elif (
        answer is not None
        or pages is not None
    ):
        parse_status = (
            "partial"
        )

    else:
        parse_status = (
            "failed"
        )

    return {
        "parse_status": (
            parse_status
        ),
        "answer_parse_status": (
            answer_status
        ),
        "answer_pre": answer,
        "evidence_parse_status": (
            evidence_status
        ),
        "evidence_pages_pre": pages,
        "evidence_valid": (
            evidence_valid
        ),
        "out_of_range_pages": (
            bad
        ),
        "parse_source": (
            source
        ),
        "schema_valid": (
            schema_valid
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Parse MiniCPM v4.1 "
            "raw result with "
            "answer/evidence separation"
        )
    )

    ap.add_argument(
        "--dataset-id",
        choices=list(c.DATASETS),
        required=True,
    )

    args = ap.parse_args()

    inp = c.raw_output_path(
        args.dataset_id
    )

    out = c.parsed_output_path(
        args.dataset_id
    )

    if not inp.exists():
        raise FileNotFoundError(
            inp
        )

    data = json.loads(
        inp.read_text(
            encoding="utf-8"
        )
    )

    counts = {
        "strict": 0,
        "normalized_strict": 0,
        "partial": 0,
        "failed": 0,
        "technical": 0,
    }

    for _, unit in data.items():
        if not isinstance(
            unit,
            dict,
        ):
            continue

        for _, item in (
            unit.get(
                "QA",
                {},
            ).items()
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue

            run_status = str(
                item.get(
                    "run_status"
                )
                or item.get(
                    "status"
                )
                or ""
            )

            if run_status not in {
                "completed",
                "parse_failed",
            }:
                item[
                    "parse_status"
                ] = (
                    "technical_failure"
                )

                item[
                    "answer_parse_status"
                ] = "missing"

                item[
                    "evidence_parse_status"
                ] = "missing"

                item[
                    "evidence_valid"
                ] = False

                counts[
                    "technical"
                ] += 1

                continue

            parsed = parse_record(
                str(
                    item.get(
                        "answer_pre_raw",
                        "",
                    )
                    or ""
                ),
                item.get(
                    "pdf_total_pages"
                ),
            )

            item.update(
                parsed
            )

            counts[
                parsed[
                    "parse_status"
                ]
            ] += 1

    atomic_write_json(
        out,
        data,
    )

    print(
        f"[Parse] "
        f"{args.dataset_id} "
        f"| {counts}",
        flush=True,
    )

    print(
        f"[Saved] {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
