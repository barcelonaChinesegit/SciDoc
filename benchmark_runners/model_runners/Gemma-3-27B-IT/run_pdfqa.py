#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import transformers

import gemma3_core as c


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemma-3-27B-IT standardized PDF-QA runner"
    )

    p.add_argument(
        "--dataset-id",
        required=True,
        choices=list(c.DATASETS),
    )

    p.add_argument(
        "--model-path",
        type=Path,
        default=c.MODEL_PATH,
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=-1,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--retry-technical",
        action="store_true",
        help=(
            "Retry records previously marked OOM/error/context failure. "
            "Without this flag, technical failures are treated as terminal "
            "for resume purposes."
        ),
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    return p.parse_args()


# ============================================================
# Utilities
# ============================================================

def atomic_write_json(
    path: Path,
    obj: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
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

        os.replace(
            tmp_name,
            path,
        )

    finally:
        if os.path.exists(
            tmp_name
        ):
            try:
                os.remove(
                    tmp_name
                )
            except OSError:
                pass


def file_sha256(
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

            h.update(chunk)

    return h.hexdigest()


def get_existing_record(
    results: Dict[str, Any],
    unit_id: str,
    qa_id: str,
) -> Dict[str, Any] | None:

    unit = results.get(
        unit_id
    )

    if not isinstance(
        unit,
        dict,
    ):
        return None

    qa_map = unit.get(
        "QA"
    )

    if not isinstance(
        qa_map,
        dict,
    ):
        return None

    qa = qa_map.get(
        qa_id
    )

    return (
        qa
        if isinstance(qa, dict)
        else None
    )


def should_skip_existing(
    existing: Dict[str, Any] | None,
    *,
    retry_technical: bool,
) -> bool:

    if not isinstance(
        existing,
        dict,
    ):
        return False

    status = str(
        existing.get(
            "status",
            "",
        )
    ).lower()

    # Model response already obtained.
    if status in {
        "completed",
        "normalized_completed",
        "parse_failed",
    }:
        return True

    # Retry technical failures only when explicitly requested.
    if retry_technical:
        return False

    return status in {
        "oom_single_gpu",
        "context_guard_exceeded",
        "error",
    }


# ============================================================
# Strict JSON parser
#
# Allowed mechanical normalization:
#   1. surrounding whitespace
#   2. one outer ```json ... ``` or ``` ... ``` wrapper
#
# Forbidden:
#   - ast.literal_eval
#   - extracting an inner {...} from arbitrary prose
#   - changing field names/types/content
#   - second model generation
# ============================================================

def strip_outer_fence(
    raw: str,
) -> Tuple[str, bool]:

    s = str(
        raw or ""
    ).strip()

    if not s:
        return "", False

    if (
        s.startswith("```")
        and s.endswith("```")
    ):
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
                and last == "```"
            ):
                return (
                    "\n".join(
                        lines[1:-1]
                    ).strip(),
                    True,
                )

    return s, False


def strict_parse_output(
    raw: str,
    *,
    page_count: int,
) -> Tuple[
    bool,
    str,
    Dict[str, Any] | None,
    bool,
    list[str],
]:

    text, fence_normalized = (
        strip_outer_fence(
            raw
        )
    )

    if not text:
        return (
            False,
            "empty",
            None,
            fence_normalized,
            ["empty_output"],
        )

    try:
        obj = json.loads(
            text
        )

    except Exception as exc:
        return (
            False,
            "invalid_json",
            None,
            fence_normalized,
            [
                "json_decode:"
                + type(exc).__name__
            ],
        )

    if not isinstance(
        obj,
        dict,
    ):
        return (
            False,
            "not_object",
            None,
            fence_normalized,
            ["top_level_not_object"],
        )

    expected_keys = {
        "answer_pre",
        "evidence_pages",
    }

    if set(obj) != expected_keys:
        return (
            False,
            "wrong_keys",
            obj,
            fence_normalized,
            [
                "expected_exact_keys:"
                "answer_pre,evidence_pages"
            ],
        )

    answer = obj.get(
        "answer_pre"
    )

    if (
        not isinstance(
            answer,
            str,
        )
        or not answer.strip()
    ):
        return (
            False,
            "invalid_answer_pre",
            obj,
            fence_normalized,
            [
                "answer_pre_must_be_"
                "nonempty_string"
            ],
        )

    pages = obj.get(
        "evidence_pages"
    )

    if not isinstance(
        pages,
        list,
    ):
        return (
            False,
            "evidence_not_list",
            obj,
            fence_normalized,
            [
                "evidence_pages_must_"
                "be_list"
            ],
        )

    if not all(
        isinstance(x, int)
        and not isinstance(x, bool)
        for x in pages
    ):
        return (
            False,
            "evidence_not_integer_list",
            obj,
            fence_normalized,
            [
                "evidence_pages_must_"
                "contain_only_integers"
            ],
        )

    out_of_range = [
        x
        for x in pages
        if x < 1
        or x > page_count
    ]

    if out_of_range:
        return (
            False,
            "evidence_page_out_of_range",
            obj,
            fence_normalized,
            [
                "out_of_range_pages:"
                + ",".join(
                    map(
                        str,
                        out_of_range,
                    )
                )
            ],
        )

    if (
        answer.strip()
        == "Unanswerable"
        and pages
    ):
        return (
            False,
            "unanswerable_with_evidence",
            obj,
            fence_normalized,
            [
                "Unanswerable_requires_"
                "empty_evidence_pages"
            ],
        )

    reason = (
        "outer_markdown_fence_normalized"
        if fence_normalized
        else "strict_ok"
    )

    return (
        True,
        reason,
        obj,
        fence_normalized,
        [],
    )


# ============================================================
# Records
# ============================================================

def prediction_record(
    *,
    raw: str,
    parse_ok: bool,
    parse_reason: str,
    parsed: Dict[str, Any] | None,
    normalized: bool,
    violations: list[str],
    stats: Dict[str, Any],
    pdf_path: Path,
    page_count: int,
    original_sizes: list,
    prompt_id: str,
    prompt_text: str,
) -> Dict[str, Any]:

    if parse_ok:
        status = (
            "normalized_completed"
            if normalized
            else "completed"
        )

        answer_pre = (
            parsed["answer_pre"]
            .strip()
        )

        evidence_pages = list(
            parsed[
                "evidence_pages"
            ]
        )

    else:
        status = "parse_failed"
        answer_pre = ""
        evidence_pages = []

    return {
        "status":
            status,

        "model_name":
            c.MODEL_NAME,

        "model_path":
            str(c.MODEL_PATH),

        "model_native_protocol":
            c.MODEL_NATIVE_PROTOCOL,

        "answer_pre_raw":
            raw,

        "answer_pre":
            answer_pre,

        "evidence_pages_pre":
            evidence_pages,

        "strict_format_reason":
            parse_reason,

        "normalized_by_parser":
            normalized,

        "schema_violations":
            violations,

        "prompt_id":
            prompt_id,

        "prompt_version":
            c.PROMPT_VERSION,

        "prompt_sha256":
            c.prompt_sha256(
                prompt_text
            ),

        "gold_answer_sent":
            False,

        "gold_evidence_pages_sent":
            False,

        "pdf_path_resolved":
            str(pdf_path),

        "pdf_total_pages":
            page_count,

        "pdf_pages_supplied":
            list(
                range(
                    1,
                    page_count + 1,
                )
            ),

        "pdf_pages_supplied_count":
            page_count,

        "pdf_dpi":
            c.PDF_DPI,

        "render_format":
            "JPEG",

        "jpeg_quality":
            c.JPEG_QUALITY,

        "jpeg_subsampling":
            c.JPEG_SUBSAMPLING,

        "render_original_pixel_sizes":
            [
                list(x)
                for x
                in original_sizes
            ],

        "processor_image_policy":
            c.PROCESSOR_IMAGE_POLICY,

        "page_dropping":
            False,

        "context_truncation":
            False,

        "format_repair_used":
            False,

        "max_new_tokens":
            c.MAX_NEW_TOKENS,

        "do_sample":
            c.DO_SAMPLE,

        "use_cache":
            c.USE_CACHE,

        "attn_implementation":
            c.ATTN_IMPLEMENTATION,

        "dtype":
            "bfloat16",

        **stats,
    }


def technical_error_record(
    *,
    status: str,
    exc: Exception,
    stage: str,
    pdf_path: Path | None,
    prompt_id: str,
    prompt_text: str,
    elapsed_seconds: float,
) -> Dict[str, Any]:

    return {
        "status":
            status,

        "model_name":
            c.MODEL_NAME,

        "model_path":
            str(c.MODEL_PATH),

        "model_native_protocol":
            c.MODEL_NATIVE_PROTOCOL,

        "error_stage":
            stage,

        "error_type":
            type(exc).__name__,

        "error_message":
            str(exc),

        "traceback_tail":
            traceback.format_exc()[
                -6000:
            ],

        "answer_pre_raw":
            "",

        "answer_pre":
            "",

        "evidence_pages_pre":
            [],

        "strict_format_reason":
            "technical_failure",

        "normalized_by_parser":
            False,

        "schema_violations":
            [],

        "prompt_id":
            prompt_id,

        "prompt_version":
            c.PROMPT_VERSION,

        "prompt_sha256":
            c.prompt_sha256(
                prompt_text
            ),

        "gold_answer_sent":
            False,

        "gold_evidence_pages_sent":
            False,

        "pdf_path_resolved":
            (
                str(pdf_path)
                if pdf_path
                else ""
            ),

        "pdf_dpi":
            c.PDF_DPI,

        "render_format":
            "JPEG",

        "jpeg_quality":
            c.JPEG_QUALITY,

        "jpeg_subsampling":
            c.JPEG_SUBSAMPLING,

        "processor_image_policy":
            c.PROCESSOR_IMAGE_POLICY,

        "page_dropping":
            False,

        "context_truncation":
            False,

        "format_repair_used":
            False,

        "max_new_tokens":
            c.MAX_NEW_TOKENS,

        "do_sample":
            c.DO_SAMPLE,

        "use_cache":
            c.USE_CACHE,

        "attn_implementation":
            c.ATTN_IMPLEMENTATION,

        "dtype":
            "bfloat16",

        "elapsed_seconds":
            round(
                elapsed_seconds,
                4,
            ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    data, samples = (
        c.load_dataset(
            args.dataset_id
        )
    )

    spec = c.DATASETS[
        args.dataset_id
    ]

    prompt_id, prompt_text = (
        c.task_prompt(
            args.dataset_id
        )
    )

    out_path = (
        c.raw_output_path(
            args.dataset_id
        )
    )

    manifest_path = (
        c.manifest_path(
            args.dataset_id
        )
    )

    # --------------------------------------------------------
    # Preflight: resolve every unique PDF before CUDA/model load
    # --------------------------------------------------------

    resolved_pdfs: Dict[
        str,
        Path,
    ] = {}

    for sample in samples:
        unit_id = sample[
            "unit_id"
        ]

        if unit_id in resolved_pdfs:
            continue

        resolved_pdfs[
            unit_id
        ] = c.resolve_pdf_path(
            args.dataset_id,
            unit_id,
            sample["unit"],
        )

    print(
        f"[Preflight] "
        f"dataset={args.dataset_id} "
        f"| QA={len(samples)} "
        f"| PDFs={len(resolved_pdfs)}",
        flush=True,
    )

    if args.dry_run:

        print(
            f"[DryRun] data="
            f"{spec['json_path']}"
        )

        print(
            f"[DryRun] pdf_root="
            f"{spec['pdf_root']}"
        )

        print(
            f"[DryRun] output="
            f"{out_path}"
        )

        print(
            f"[DryRun] prompt_id="
            f"{prompt_id}"
        )

        print(
            f"[DryRun] prompt_version="
            f"{c.PROMPT_VERSION}"
        )

        print(
            f"[DryRun] prompt_sha256="
            f"{c.prompt_sha256(prompt_text)}"
        )

        print(
            f"[DryRun] protocol="
            f"{c.MODEL_NATIVE_PROTOCOL}"
        )

        print(
            f"[DryRun] model_path="
            f"{args.model_path}"
        )

        print(
            f"[DryRun] pdf_dpi="
            f"{c.PDF_DPI}"
        )

        print(
            f"[DryRun] render="
            f"JPEG quality="
            f"{c.JPEG_QUALITY} "
            f"subsampling="
            f"{c.JPEG_SUBSAMPLING}"
        )

        print(
            f"[DryRun] max_new_tokens="
            f"{c.MAX_NEW_TOKENS}"
        )

        print(
            f"[DryRun] context_limit="
            f"{c.CONTEXT_LIMIT}"
        )

        print(
            f"[DryRun] do_sample="
            f"{c.DO_SAMPLE}"
        )

        print(
            f"[DryRun] use_cache="
            f"{c.USE_CACHE}"
        )

        print(
            "[DryRun] page_dropping=False"
        )

        print(
            "[DryRun] context_truncation=False"
        )

        print(
            "[DryRun] format_repair=False"
        )

        return

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        out_path.is_file()
        and not args.overwrite
    ):
        with out_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            results = json.load(f)

    else:
        results = copy.deepcopy(
            data
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("=" * 90)
    print(
        "Gemma-3-27B-IT PDF-QA"
    )
    print(
        "Python             :",
        os.sys.executable,
    )
    print(
        "Torch              :",
        torch.__version__,
    )
    print(
        "Transformers       :",
        transformers.__version__,
    )
    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "",
        ),
    )
    print(
        "Model path         :",
        args.model_path,
    )
    print(
        "Protocol           :",
        c.MODEL_NATIVE_PROTOCOL,
    )
    print(
        "Prompt             :",
        c.PROMPT_VERSION,
    )
    print("=" * 90)

    engine = c.GemmaEngine(
        args.model_path
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    dataset_path = Path(
        spec["json_path"]
    )

    manifest = {
        "model":
            c.MODEL_NAME,

        "model_path":
            str(
                args.model_path
            ),

        "dataset_id":
            args.dataset_id,

        "dataset_path":
            str(
                dataset_path
            ),

        "dataset_sha256":
            file_sha256(
                dataset_path
            ),

        "qa_count":
            len(samples),

        "pdf_count":
            len(
                resolved_pdfs
            ),

        "run_version":
            c.RUN_VERSION,

        "model_native_protocol":
            c.MODEL_NATIVE_PROTOCOL,

        "prompt_id":
            prompt_id,

        "prompt_version":
            c.PROMPT_VERSION,

        "prompt_sha256":
            c.prompt_sha256(
                prompt_text
            ),

        "pdf_dpi":
            c.PDF_DPI,

        "render_format":
            "JPEG",

        "jpeg_quality":
            c.JPEG_QUALITY,

        "jpeg_subsampling":
            c.JPEG_SUBSAMPLING,

        "processor_image_policy":
            c.PROCESSOR_IMAGE_POLICY,

        "page_numbering":
            "physical_pdf_page_1_based",

        "whole_pdf":
            True,

        "page_dropping":
            False,

        "context_truncation":
            False,

        "format_repair":
            False,

        "max_new_tokens":
            c.MAX_NEW_TOKENS,

        "context_limit":
            engine.context_limit,

        "do_sample":
            c.DO_SAMPLE,

        "use_cache":
            c.USE_CACHE,

        "attn_implementation":
            c.ATTN_IMPLEMENTATION,

        "dtype":
            "bfloat16",

        "torch_version":
            torch.__version__,

        "transformers_version":
            transformers.__version__,

        "cuda_visible_devices":
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "",
            ),

        "gpu_name_logical_cuda0":
            torch.cuda.get_device_name(
                0
            ),

        "runner_path":
            str(
                Path(
                    __file__
                ).resolve()
            ),

        "runner_sha256":
            file_sha256(
                Path(
                    __file__
                ).resolve()
            ),

        "started_at_unix":
            time.time(),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    attempted = 0
    skipped = 0

    try:

        for index, sample in enumerate(
            samples,
            start=1,
        ):

            unit_id = sample[
                "unit_id"
            ]

            qa_id = sample[
                "qa_id"
            ]

            uid = sample[
                "item_uid"
            ]

            qa = sample[
                "qa"
            ]

            existing = (
                get_existing_record(
                    results,
                    unit_id,
                    qa_id,
                )
            )

            if (
                not args.overwrite
                and should_skip_existing(
                    existing,
                    retry_technical=(
                        args.retry_technical
                    ),
                )
            ):
                skipped += 1
                continue

            if (
                args.max_samples > 0
                and attempted
                >= args.max_samples
            ):
                break

            attempted += 1

            question = str(
                qa.get(
                    "question",
                    "",
                )
            ).strip()

            if not question:
                raise RuntimeError(
                    f"{uid}: empty question"
                )

            pdf_path = (
                resolved_pdfs[
                    unit_id
                ]
            )

            start = time.time()
            stage = "render"

            try:

                page_files, original_sizes = (
                    c.render_pdf_cache(
                        pdf_path
                    )
                )

                page_count = len(
                    page_files
                )

                if page_count <= 0:
                    raise RuntimeError(
                        "PDF rendered zero pages"
                    )

                print(
                    f"[QA] "
                    f"{index}/{len(samples)} "
                    f"| {uid} "
                    f"| pages={page_count}",
                    flush=True,
                )

                stage = "inference"

                raw, stats = (
                    c.infer_one(
                        engine=engine,
                        dataset_id=(
                            args.dataset_id
                        ),
                        question=question,
                        page_files=(
                            page_files
                        ),
                    )
                )

                stage = "strict_parse"

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

                record = prediction_record(
                    raw=raw,
                    parse_ok=ok,
                    parse_reason=reason,
                    parsed=parsed,
                    normalized=normalized,
                    violations=violations,
                    stats=stats,
                    pdf_path=pdf_path,
                    page_count=page_count,
                    original_sizes=(
                        original_sizes
                    ),
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                )

            except torch.cuda.OutOfMemoryError as exc:

                record = technical_error_record(
                    status="oom_single_gpu",
                    exc=exc,
                    stage=stage,
                    pdf_path=pdf_path,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    elapsed_seconds=(
                        time.time()
                        - start
                    ),
                )

            except RuntimeError as exc:

                if str(exc).startswith(
                    "context_overflow:"
                ):
                    status = (
                        "context_guard_exceeded"
                    )
                else:
                    status = "error"

                record = technical_error_record(
                    status=status,
                    exc=exc,
                    stage=stage,
                    pdf_path=pdf_path,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    elapsed_seconds=(
                        time.time()
                        - start
                    ),
                )

            except Exception as exc:

                record = technical_error_record(
                    status="error",
                    exc=exc,
                    stage=stage,
                    pdf_path=pdf_path,
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    elapsed_seconds=(
                        time.time()
                        - start
                    ),
                )

            finally:
                engine.cleanup()

            results[
                unit_id
            ][
                "QA"
            ][
                qa_id
            ].update(
                record
            )

            atomic_write_json(
                out_path,
                results,
            )

            print(
                f"[RESULT] {uid} "
                f"| status="
                f"{record['status']} "
                f"| parse="
                f"{record.get('strict_format_reason')} "
                f"| peak="
                f"{record.get('gpu_peak_allocated_gib', 0)} GiB",
                flush=True,
            )

    finally:
        engine.cleanup()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    counts: Dict[
        str,
        int,
    ] = {}

    for sample in samples:

        rec = get_existing_record(
            results,
            sample["unit_id"],
            sample["qa_id"],
        )

        status = (
            str(
                rec.get(
                    "status",
                    "",
                )
            )
            if rec
            else ""
        )

        if not status:
            status = "not_run"

        counts[
            status
        ] = (
            counts.get(
                status,
                0,
            )
            + 1
        )

    print()
    print(
        "[Summary]",
        counts,
    )
    print(
        "[Attempted this run]",
        attempted,
    )
    print(
        "[Skipped existing]",
        skipped,
    )
    print(
        "[Raw output]",
        out_path,
    )
    print(
        "[Manifest]",
        manifest_path,
    )


if __name__ == "__main__":
    main()
