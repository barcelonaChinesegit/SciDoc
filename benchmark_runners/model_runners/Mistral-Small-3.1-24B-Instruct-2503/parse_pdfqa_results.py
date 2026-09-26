#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings.pdfqa_prompts import (
    PROMPT_VERSION,
    prompt_sha256,
)

import run_pdfqa as r


BASE = Path(__file__).resolve().parent

RAW_DIR = BASE / "output" / "raw_result"
PARSED_DIR = BASE / "output" / "parsed_result"
MANIFEST_DIR = BASE / "output" / "manifests"

MODEL_NAME = "Mistral-Small-3.1-24B-Instruct-2503"

MODEL_PATH = Path(
    str(Path(__file__).resolve().parents[3] / "models" / "Mistral-Small-3.1-24B-Instruct-2503")
)

MODEL_NATIVE_PROTOCOL = (
    "mistral31-v16-tp2-contextfit-wholepdf-no-repair"
)

RUN_VERSION = "run-v1"


def atomic_write_json(
    path: Path,
    obj: Any,
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
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

        tmp = Path(
            f.name
        )

    os.replace(
        tmp,
        path,
    )


def sha256_file(
    path: Path,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def output_name(
    dataset_id: str,
) -> str:

    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi144"
        f"__{RUN_VERSION}.json"
    )


def iter_qa(
    data: Any,
):

    if not isinstance(
        data,
        dict,
    ):
        return

    for unit_id, unit in data.items():

        if not isinstance(
            unit,
            dict,
        ):
            continue

        qa_map = unit.get(
            "QA",
            {},
        )

        if not isinstance(
            qa_map,
            dict,
        ):
            continue

        for qa_id, qa in qa_map.items():

            if isinstance(
                qa,
                dict,
            ):
                yield (
                    str(unit_id),
                    str(qa_id),
                    qa,
                )


def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset-id",
        required=True,
        choices=list(r.DATASETS),
    )

    args = ap.parse_args()

    dataset_id = (
        args.dataset_id
    )

    spec = r.DATASETS[
        dataset_id
    ]

    inp = (
        RAW_DIR
        / output_name(
            dataset_id
        )
    )

    out = (
        PARSED_DIR
        / output_name(
            dataset_id
        )
    )

    manifest_path = (
        MANIFEST_DIR
        / (
            output_name(
                dataset_id
            ).replace(
                ".json",
                ".manifest.json",
            )
        )
    )

    if not inp.is_file():
        raise FileNotFoundError(
            inp
        )

    data = json.loads(
        inp.read_text(
            encoding="utf-8"
        )
    )

    counts = Counter()

    for (
        unit_id,
        qa_id,
        qa,
    ) in iter_qa(data):

        has_run_marker = any(
            key in qa
            for key in [
                "answer_pre_raw",
                "prediction_parse_status",
                "input_tokens",
                "elapsed_seconds",
                "error_type",
                "error_message",
            ]
        )

        if not has_run_marker:
            counts[
                "not_run"
            ] += 1
            continue

        raw = qa.get(
            "answer_pre_raw"
        )

        if (
            raw is None
            or str(raw).strip() == ""
        ):

            if (
                qa.get(
                    "error_type"
                )
                or qa.get(
                    "error_message"
                )
            ):
                qa[
                    "status"
                ] = "error"

                qa[
                    "posthoc_parse_status"
                ] = "technical_failure"

                counts[
                    "technical"
                ] += 1

            else:
                qa[
                    "status"
                ] = "parse_failed"

                qa[
                    "posthoc_parse_status"
                ] = "failed"

                counts[
                    "failed"
                ] += 1

            continue

        (
            ok,
            answer,
            pages,
            reason,
        ) = r.parse_prediction(
            str(raw)
        )

        # Physical-page range is checked post-hoc.
        if ok:

            page_count = int(
                qa.get(
                    "pdf_total_pages",
                    0,
                )
                or 0
            )

            if page_count > 0:

                bad_pages = [
                    x
                    for x in pages
                    if (
                        x < 1
                        or x > page_count
                    )
                ]

                if bad_pages:

                    ok = False

                    reason = (
                        "evidence_page_out_of_range:"
                        + ",".join(
                            map(
                                str,
                                bad_pages,
                            )
                        )
                    )

        qa[
            "format_repair_used"
        ] = False

        qa[
            "posthoc_parse_ok"
        ] = bool(ok)

        qa[
            "posthoc_parse_reason"
        ] = reason

        if ok:

            qa[
                "answer_pre"
            ] = answer

            qa[
                "evidence_pages_pre"
            ] = pages

            if (
                reason
                == "outer_markdown_fence_normalized"
            ):

                qa[
                    "status"
                ] = "normalized_completed"

                qa[
                    "posthoc_parse_status"
                ] = "normalized_strict"

                counts[
                    "normalized_strict"
                ] += 1

            else:

                qa[
                    "status"
                ] = "completed"

                qa[
                    "posthoc_parse_status"
                ] = "strict"

                counts[
                    "strict"
                ] += 1

        else:

            qa[
                "answer_pre"
            ] = ""

            qa[
                "evidence_pages_pre"
            ] = []

            qa[
                "status"
            ] = "parse_failed"

            qa[
                "posthoc_parse_status"
            ] = "failed"

            counts[
                "failed"
            ] += 1

    atomic_write_json(
        out,
        data,
    )

    prompt_text = (
        r.build_system_prompt(
            dataset_id
        )
    )

    dataset_path = Path(
        spec["data"]
    )

    # Environment versions are audit metadata only.
    try:
        import torch
        torch_version = (
            torch.__version__
        )
    except Exception:
        torch_version = "unknown"

    try:
        import vllm
        vllm_version = getattr(
            vllm,
            "__version__",
            "unknown",
        )
    except Exception:
        vllm_version = "unknown"

    try:
        import openai
        openai_version = getattr(
            openai,
            "__version__",
            "unknown",
        )
    except Exception:
        openai_version = "unknown"

    try:
        import mistral_common
        mistral_common_version = getattr(
            mistral_common,
            "__version__",
            "unknown",
        )
    except Exception:
        mistral_common_version = (
            "unknown"
        )

    manifest = {
        "model":
            MODEL_NAME,

        "model_path":
            str(
                MODEL_PATH
            ),

        "model_native_protocol":
            MODEL_NATIVE_PROTOCOL,

        "inference_backend":
            "vllm_openai_local_tp2",

        "dataset_id":
            dataset_id,

        "dataset_path":
            str(
                dataset_path
            ),

        "dataset_sha256":
            sha256_file(
                dataset_path
            ),

        "expected_qa_count":
            int(
                spec["expected"]
            ),

        "run_version":
            RUN_VERSION,

        "prompt_id":
            spec["prompt_id"],

        "prompt_version":
            PROMPT_VERSION,

        "prompt_sha256":
            prompt_sha256(
                prompt_text
            ),

        "pdf_dpi":
            144,

        "render_format":
            "JPEG",

        "jpeg_quality":
            95,

        "jpeg_subsampling":
            0,

        "whole_pdf":
            True,

        "page_dropping":
            False,

        "context_truncation":
            False,

        "format_repair":
            False,

        "context_limit":
            128000,

        "max_new_tokens":
            512,

        "temperature":
            0.0,

        "tensor_parallel_size":
            2,

        "gpu_pair":
            os.getenv(
                "MISTRAL31_GPU_PAIR",
                "",
            ),

        "gpu_memory_utilization":
            0.90,

        "max_images_per_prompt":
            512,

        "context_fit_enabled":
            True,

        "context_fit_target_prompt_tokens":
            118000,

        "context_fit_min_scale":
            0.35,

        "context_fit_max_attempts":
            6,

        "context_fit_policy":
            (
                "uniform linear resize of all supplied "
                "page images only after context overflow; "
                "page count and order unchanged"
            ),

        "torch_version":
            torch_version,

        "vllm_version":
            vllm_version,

        "openai_version":
            openai_version,

        "mistral_common_version":
            mistral_common_version,

        "result_counts":
            dict(
                counts
            ),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    print(
        f"[Parse] {dataset_id} | "
        f"{dict(counts)}"
    )

    print(
        f"[Saved] {out}"
    )

    print(
        f"[Manifest] "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
