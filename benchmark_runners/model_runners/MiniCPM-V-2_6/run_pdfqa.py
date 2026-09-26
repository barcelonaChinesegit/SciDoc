#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    os.environ.get("PDFQA_GPU", "2"),
)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from transformers import AutoModel, AutoTokenizer

import minicpm26_core as c


# ============================================================
# IO
# ============================================================

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


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# CLI / resume
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MiniCPM-V-2.6 PDF-QA "
            "reference-faithful runner"
        )
    )

    p.add_argument(
        "--dataset-id",
        choices=list(c.DATASETS),
        required=True,
    )

    p.add_argument(
        "--model-path",
        default=str(c.MODEL_PATH),
    )

    p.add_argument(
        "--retry-technical",
        action="store_true",
        help=(
            "Retry prior OOM/error entries. "
            "Parse failures are model behavior and are not retried."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all remaining samples",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    return p.parse_args()


def get_existing(
    results: Dict[str, Any],
    unit_id: str,
    qa_id: str,
):
    return (
        results
        .get(unit_id, {})
        .get("QA", {})
        .get(qa_id)
    )


def should_skip(
    existing: Any,
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

    # Completed model behaviours:
    # don't silently re-run them.
    if status in {
        "completed",
        "normalized_completed",
        "parse_failed",
    }:
        return True

    if retry_technical:
        return False

    return status in {
        "oom_single_gpu",
        "error",
    }


# ============================================================
# Result structure
# ============================================================

def build_unit_shell(
    unit_id: str,
    unit: Dict[str, Any],
    pdf_path: Path,
    page_count: int,
) -> Dict[str, Any]:

    # Preserve benchmark unit metadata while excluding
    # the source QA collection itself.
    out = {
        k: v
        for k, v in unit.items()
        if k != "QA"
    }

    out.setdefault(
        "paper",
        unit.get(
            "paper",
            unit_id,
        ),
    )

    out["pdf_path_resolved"] = str(
        pdf_path
    )
    out["pdf_total_pages"] = int(
        page_count
    )
    out["QA"] = {}

    return out


def count_output_tokens(
    tokenizer,
    text: str,
) -> int:
    try:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        return len(ids)
    except Exception:
        return -1


# ============================================================
# Strict raw-output validation
#
# Allowed normalization:
#   1. boundary <|endoftext|>
#   2. one outer ```json ... ``` fence
#
# No semantic repair.
# ============================================================

_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.I | re.S,
)


def strict_parse_output(
    raw: str,
    pdf_pages: int,
):
    text = str(
        raw or ""
    ).strip()

    normalized_end_tokens = 0
    normalized_fence = False

    end_token = "<|endoftext|>"

    def strip_end_tokens(
        value: str,
    ):
        nonlocal normalized_end_tokens

        changed = True

        while changed:
            changed = False

            if value.startswith(
                end_token
            ):
                value = value[
                    len(end_token):
                ].lstrip()

                normalized_end_tokens += 1
                changed = True

            if value.endswith(
                end_token
            ):
                value = value[
                    :-len(end_token)
                ].rstrip()

                normalized_end_tokens += 1
                changed = True

        return value

    text = strip_end_tokens(
        text
    )

    m = _FENCE_RE.match(
        text
    )

    if m:
        text = m.group(1).strip()
        normalized_fence = True
        text = strip_end_tokens(
            text
        )

    try:
        obj = json.loads(
            text
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": "invalid_json",
            "answer_pre": "",
            "evidence_pages": [],
            "normalized_fence": normalized_fence,
            "normalized_endoftext_tokens":
                normalized_end_tokens,
            "parse_error": repr(exc),
            "violations": [
                "invalid_json"
            ],
        }

    violations: List[str] = []

    if not isinstance(
        obj,
        dict,
    ):
        violations.append(
            "not_object"
        )
        obj = {}

    allowed = {
        "answer_pre",
        "evidence_pages",
    }

    extra = sorted(
        set(obj) - allowed
    )

    if extra:
        violations.append(
            "unexpected_keys:"
            + ",".join(extra)
        )

    answer = obj.get(
        "answer_pre"
    )
    pages = obj.get(
        "evidence_pages"
    )

    if (
        not isinstance(answer, str)
        or not answer.strip()
    ):
        violations.append(
            "invalid_answer_pre"
        )

        if not isinstance(
            answer,
            str,
        ):
            answer = ""

    if not isinstance(
        pages,
        list,
    ):
        violations.append(
            "evidence_pages_not_list"
        )
        pages = []

    else:
        for p in pages:
            if (
                isinstance(p, bool)
                or not isinstance(p, int)
            ):
                violations.append(
                    "non_integer_evidence_page"
                )
                break

        if all(
            isinstance(p, int)
            and not isinstance(p, bool)
            for p in pages
        ):
            if any(
                p < 1
                or p > pdf_pages
                for p in pages
            ):
                violations.append(
                    "evidence_page_out_of_bounds"
                )

    if (
        isinstance(answer, str)
        and answer.strip().lower()
        == "unanswerable"
        and pages != []
    ):
        violations.append(
            "unanswerable_must_have_empty_evidence"
        )

    normalized = (
        normalized_fence
        or normalized_end_tokens > 0
    )

    return {
        "ok": len(
            violations
        ) == 0,
        "reason": (
            "normalized_format"
            if (
                not violations
                and normalized
            )
            else (
                "strict_json"
                if not violations
                else "schema_invalid"
            )
        ),
        "answer_pre": (
            answer
            if isinstance(
                answer,
                str,
            )
            else ""
        ),
        "evidence_pages": (
            pages
            if isinstance(
                pages,
                list,
            )
            else []
        ),
        "normalized_fence":
            normalized_fence,
        "normalized_endoftext_tokens":
            normalized_end_tokens,
        "parse_error": "",
        "violations":
            violations,
    }


# ============================================================
# PDF cache
#
# Caching does not alter model-facing input; it only avoids
# rendering the same PDF again for adjacent QA pairs.
# ============================================================

class SinglePdfImageCache:

    def __init__(self) -> None:
        self.path: str | None = None
        self.images = []

    def get(
        self,
        path: Path,
    ):
        resolved = str(
            path.resolve()
        )

        if (
            self.path == resolved
            and self.images
        ):
            return self.images

        self.clear()

        self.images = (
            c.pdf_to_images(
                path
            )
        )

        self.path = resolved

        return self.images

    def clear(self) -> None:
        c.close_images(
            self.images
        )

        self.images = []
        self.path = None

        gc.collect()

    def __del__(self):
        try:
            self.clear()
        except Exception:
            pass


# ============================================================
# Error classification
# ============================================================

def is_oom_error(
    exc: BaseException,
) -> bool:

    if isinstance(
        exc,
        torch.OutOfMemoryError,
    ):
        return True

    text = str(
        exc
    ).lower()

    return (
        "cuda out of memory"
        in text
        or "out of memory"
        in text
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    _, samples = c.load_dataset(
        args.dataset_id
    )

    prompt_id, instruct = (
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
    # Resolve every unique PDF before loading CUDA/model.
    # --------------------------------------------------------

    resolved: Dict[
        str,
        Path,
    ] = {}

    for sample in samples:
        unit_id = str(
            sample["unit_id"]
        )

        if unit_id not in resolved:
            resolved[
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
        f"| PDFs={len(resolved)}",
        flush=True,
    )

    if args.dry_run:
        spec = (
            c.DATASETS[
                args.dataset_id
            ]
        )

        print(
            f"[DryRun] "
            f"data={spec['json_path']}"
        )
        print(
            f"[DryRun] "
            f"pdf_root={spec['pdf_root']}"
        )
        print(
            f"[DryRun] "
            f"output={out_path}"
        )
        print(
            f"[DryRun] "
            f"model={args.model_path}"
        )
        print(
            f"[DryRun] "
            f"prompt_id={prompt_id}"
        )
        print(
            f"[DryRun] "
            f"prompt_version="
            f"{c.PROMPT_VERSION}"
        )
        print(
            f"[DryRun] "
            f"prompt_sha256="
            f"{c.prompt_sha256(instruct)}"
        )
        print(
            f"[DryRun] "
            f"model_native_protocol="
            f"{c.MODEL_NATIVE_PROTOCOL}"
        )
        print(
            f"[DryRun] "
            f"pdf_dpi={c.PDF_DPI}"
        )
        print(
            f"[DryRun] "
            f"page_size="
            f"{c.PAGE_W}x{c.PAGE_H}"
        )
        print(
            f"[DryRun] "
            f"page_label_height="
            f"{c.PAGE_LABEL_H}"
        )
        print(
            f"[DryRun] "
            f"max_slice_nums="
            f"{c.MAX_SLICE_NUMS}"
        )
        print(
            f"[DryRun] "
            f"max_new_tokens="
            f"{c.MAX_NEW_TOKENS}"
        )
        print(
            f"[DryRun] "
            f"sampling={c.SAMPLING}"
        )
        print(
            f"[DryRun] "
            f"use_image_id="
            f"{c.USE_IMAGE_ID}"
        )
        print(
            "[DryRun] "
            "num_beams_override=None"
        )
        print(
            "[DryRun] "
            "max_inp_length_override=None"
        )
        print(
            "[DryRun] "
            "system_prompt=False"
        )
        print(
            "[DryRun] "
            "runner_AutoProcessor=False"
        )

        return

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one "
            "visible GPU, got "
            f"{torch.cuda.device_count()}"
        )

    if (
        args.overwrite
        or not out_path.exists()
    ):
        results: Dict[
            str,
            Any,
        ] = {}

    else:
        results = load_json(
            out_path
        )

    visible_gpu = (
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "",
        )
    )

    print(
        "=" * 112,
        flush=True,
    )
    print(
        "MiniCPM-V-2.6 PDF-QA "
        "reference-faithful",
        flush=True,
    )
    print(
        f"Dataset               : "
        f"{args.dataset_id}"
    )
    print(
        f"Dataset JSON          : "
        f"{c.DATASETS[args.dataset_id]['json_path']}"
    )
    print(
        f"Expected QA           : "
        f"{c.DATASETS[args.dataset_id]['expected']}"
    )
    print(
        f"Model                 : "
        f"{args.model_path}"
    )
    print(
        f"Physical CUDA_VISIBLE : "
        f"{visible_gpu}"
    )
    print(
        f"Logical GPU           : "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"PDF DPI               : "
        f"{c.PDF_DPI}"
    )
    print(
        f"Page canvas           : "
        f"{c.PAGE_W}x{c.PAGE_H}"
    )
    print(
        "Page label            : "
        "drawn PDF_PAGE_N"
    )
    print(
        "Message               : "
        "[all page images, one final user text]"
    )
    print(
        f"max_slice_nums        : "
        f"{c.MAX_SLICE_NUMS}"
    )
    print(
        f"max_new_tokens        : "
        f"{c.MAX_NEW_TOKENS}"
    )
    print(
        f"sampling              : "
        f"{c.SAMPLING}"
    )
    print(
        "num_beams override    : NONE"
    )
    print(
        "max_inp_length        : NONE"
    )
    print(
        "system_prompt         : NONE"
    )
    print(
        "runner AutoProcessor  : NONE"
    )
    print(
        "Gold answer/pages     : NEVER SENT"
    )
    print(
        "=" * 112,
        flush=True,
    )

    # --------------------------------------------------------
    # Load exactly like referencefaithful-v3.
    # --------------------------------------------------------

    print(
        "[Load] model...",
        flush=True,
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = (
        AutoModel
        .from_pretrained(
            args.model_path,
            trust_remote_code=True,
            attn_implementation=(
                c.ATTN_IMPLEMENTATION
            ),
            torch_dtype=dtype,
        )
        .eval()
        .cuda()
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
    )

    attn_class = "unknown"

    try:
        attn_class = type(
            model
            .llm
            .model
            .layers[0]
            .self_attn
        ).__name__
    except Exception:
        pass

    print(
        f"[Load] GPU="
        f"{torch.cuda.get_device_name(0)} "
        f"| dtype={dtype} "
        f"| attention={attn_class}",
        flush=True,
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "model":
            c.MODEL_NAME,
        "dataset_id":
            args.dataset_id,
        "run_version":
            c.RUN_VERSION,
        "prompt_id":
            prompt_id,
        "prompt_version":
            c.PROMPT_VERSION,
        "prompt_sha256":
            c.prompt_sha256(
                instruct
            ),
        "model_native_protocol":
            c.MODEL_NATIVE_PROTOCOL,
        "dataset_json":
            str(
                c.DATASETS[
                    args.dataset_id
                ]["json_path"]
            ),
        "pdf_root":
            str(
                c.DATASETS[
                    args.dataset_id
                ]["pdf_root"]
            ),
        "model_path":
            str(
                args.model_path
            ),
        "pdf_dpi":
            c.PDF_DPI,
        "page_width":
            c.PAGE_W,
        "page_height":
            c.PAGE_H,
        "page_label_height":
            c.PAGE_LABEL_H,
        "page_label_policy":
            "drawn_PDF_PAGE_N",
        "message_structure":
            "[all_PIL_page_images, one_combined_user_text]",
        "page_dropping":
            False,
        "gold_answer_sent":
            False,
        "gold_evidence_pages_sent":
            False,
        "max_slice_nums":
            c.MAX_SLICE_NUMS,
        "max_new_tokens":
            c.MAX_NEW_TOKENS,
        "sampling":
            c.SAMPLING,
        "use_image_id":
            c.USE_IMAGE_ID,
        "num_beams_override":
            None,
        "max_inp_length_override":
            None,
        "system_prompt_used":
            False,
        "runner_auto_processor":
            False,
        "attn_implementation":
            c.ATTN_IMPLEMENTATION,
        "attention_module_class":
            attn_class,
        "torch_dtype":
            str(dtype),
        "visible_cuda":
            visible_gpu,
        "gpu_name":
            torch.cuda.get_device_name(
                0
            ),
        "gpu_total_vram_gib":
            round(
                torch.cuda
                .get_device_properties(
                    0
                )
                .total_memory
                / 1024**3,
                3,
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

    cache = (
        SinglePdfImageCache()
    )

    attempted_this_run = 0

    try:
        for idx, sample in enumerate(
            samples,
            start=1,
        ):
            unit_id = str(
                sample["unit_id"]
            )
            qa_id = str(
                sample["qa_id"]
            )
            unit = sample["unit"]
            qa = sample["qa"]

            existing = get_existing(
                results,
                unit_id,
                qa_id,
            )

            if should_skip(
                existing,
                args.retry_technical,
            ):
                print(
                    f"[{idx}/{len(samples)}] "
                    f"SKIP "
                    f"{sample['item_uid']} "
                    f"| status="
                    f"{existing.get('status')}",
                    flush=True,
                )
                continue

            if (
                args.max_samples > 0
                and attempted_this_run
                >= args.max_samples
            ):
                print(
                    "[STOP] "
                    f"--max-samples="
                    f"{args.max_samples}",
                    flush=True,
                )
                break

            attempted_this_run += 1

            item_start = time.time()

            pdf_path = (
                resolved[
                    unit_id
                ]
            )

            render_start = (
                time.time()
            )

            try:
                images = cache.get(
                    pdf_path
                )

                render_seconds = (
                    time.time()
                    - render_start
                )

                page_count = len(
                    images
                )

                if page_count <= 0:
                    raise RuntimeError(
                        "PDF rendered zero pages"
                    )

                print(
                    f"[{idx}/{len(samples)}] "
                    f"{sample['item_uid']} "
                    f"| pages={page_count} "
                    f"| slice="
                    f"{c.MAX_SLICE_NUMS}",
                    flush=True,
                )

                torch.cuda.reset_peak_memory_stats()

                raw_obj, infer_seconds = (
                    c.model_chat(
                        model=model,
                        tokenizer=tokenizer,
                        pdf_images=images,
                        question=str(
                            qa.get(
                                "question",
                                "",
                            )
                        ),
                        instruct=instruct,
                    )
                )

                torch.cuda.synchronize()

                peak_gib = (
                    torch.cuda
                    .max_memory_allocated()
                    / 1024**3
                )

                raw = (
                    ""
                    if raw_obj is None
                    else str(
                        raw_obj
                    ).strip()
                )

                parse_start = (
                    time.time()
                )

                parsed = (
                    strict_parse_output(
                        raw,
                        page_count,
                    )
                )

                parse_seconds = (
                    time.time()
                    - parse_start
                )

                if parsed["ok"]:
                    if (
                        parsed[
                            "normalized_fence"
                        ]
                        or parsed[
                            "normalized_endoftext_tokens"
                        ] > 0
                    ):
                        status = (
                            "normalized_completed"
                        )
                    else:
                        status = (
                            "completed"
                        )
                else:
                    status = (
                        "parse_failed"
                    )

                if unit_id not in results:
                    results[
                        unit_id
                    ] = build_unit_shell(
                        unit_id,
                        unit,
                        pdf_path,
                        page_count,
                    )

                results[
                    unit_id
                ].setdefault(
                    "QA",
                    {},
                )

                entry = {
                    "item_uid":
                        sample["item_uid"],
                    "question":
                        qa.get(
                            "question",
                            "",
                        ),
                    "answer":
                        qa.get(
                            "answer",
                            "",
                        ),
                    "evidence_pages":
                        qa.get(
                            "evidence_pages",
                            [],
                        ),
                    "modal_types":
                        qa.get(
                            "modal_types",
                            [],
                        ),
                    "question_type":
                        qa.get(
                            "question_type",
                            "",
                        ),
                    "question_category":
                        qa.get(
                            "question_category",
                            "",
                        ),

                    "status":
                        status,
                    "run_status":
                        "completed",

                    "answer_pre_raw":
                        raw,
                    "answer_pre":
                        (
                            parsed[
                                "answer_pre"
                            ]
                            if parsed["ok"]
                            else ""
                        ),
                    "evidence_pages_pre":
                        (
                            parsed[
                                "evidence_pages"
                            ]
                            if parsed["ok"]
                            else None
                        ),

                    "strict_format_reason":
                        parsed["reason"],
                    "schema_violations":
                        parsed[
                            "violations"
                        ],
                    "normalized_by_parser":
                        (
                            parsed[
                                "normalized_fence"
                            ]
                            or parsed[
                                "normalized_endoftext_tokens"
                            ] > 0
                        ),
                    "normalized_markdown_fence":
                        parsed[
                            "normalized_fence"
                        ],
                    "normalized_endoftext_tokens":
                        parsed[
                            "normalized_endoftext_tokens"
                        ],
                    "parse_error":
                        parsed[
                            "parse_error"
                        ],

                    "output_tokens":
                        count_output_tokens(
                            tokenizer,
                            raw,
                        ),
                    "output_chars":
                        len(raw),

                    "pdf_path_resolved":
                        str(
                            pdf_path
                        ),
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

                    "render_dpi":
                        c.PDF_DPI,
                    "page_size":
                        [
                            c.PAGE_W,
                            c.PAGE_H,
                        ],
                    "page_label_policy":
                        "drawn_PDF_PAGE_N",

                    "max_slice_nums_used":
                        c.MAX_SLICE_NUMS,
                    "max_new_tokens":
                        c.MAX_NEW_TOKENS,
                    "sampling":
                        c.SAMPLING,
                    "use_image_id":
                        c.USE_IMAGE_ID,
                    "num_beams_override":
                        None,
                    "max_inp_length_override":
                        None,

                    "model_name":
                        c.MODEL_NAME,
                    "model_native_protocol":
                        c.MODEL_NATIVE_PROTOCOL,
                    "prompt_id":
                        prompt_id,
                    "prompt_version":
                        c.PROMPT_VERSION,
                    "prompt_sha256":
                        c.prompt_sha256(
                            instruct
                        ),
                    "prompt_delivery":
                        "inside_final_user_text",
                    "message_structure":
                        "[all_PIL_page_images, one_combined_user_text]",

                    "gold_answer_sent":
                        False,
                    "gold_evidence_pages_sent":
                        False,

                    "peak_allocated_gib":
                        round(
                            peak_gib,
                            3,
                        ),

                    "timing": {
                        "render_or_cache_seconds":
                            round(
                                render_seconds,
                                4,
                            ),
                        "inference_seconds":
                            round(
                                infer_seconds,
                                4,
                            ),
                        "parse_seconds":
                            round(
                                parse_seconds,
                                4,
                            ),
                        "total_seconds":
                            round(
                                time.time()
                                - item_start,
                                4,
                            ),
                    },
                }

                results[
                    unit_id
                ]["QA"][
                    qa_id
                ] = entry

                atomic_write_json(
                    out_path,
                    results,
                )

                print(
                    f"[RESULT] "
                    f"{sample['item_uid']} "
                    f"| status={status} "
                    f"| parse="
                    f"{parsed['reason']} "
                    f"| peak="
                    f"{round(peak_gib,3)} GiB",
                    flush=True,
                )

            except Exception as exc:
                oom = is_oom_error(
                    exc
                )

                status = (
                    "oom_single_gpu"
                    if oom
                    else "error"
                )

                if unit_id not in results:
                    results[
                        unit_id
                    ] = build_unit_shell(
                        unit_id,
                        unit,
                        pdf_path,
                        0,
                    )

                results[
                    unit_id
                ].setdefault(
                    "QA",
                    {},
                )

                results[
                    unit_id
                ]["QA"][
                    qa_id
                ] = {
                    "item_uid":
                        sample["item_uid"],
                    "question":
                        qa.get(
                            "question",
                            "",
                        ),
                    "answer":
                        qa.get(
                            "answer",
                            "",
                        ),
                    "evidence_pages":
                        qa.get(
                            "evidence_pages",
                            [],
                        ),
                    "modal_types":
                        qa.get(
                            "modal_types",
                            [],
                        ),
                    "question_type":
                        qa.get(
                            "question_type",
                            "",
                        ),
                    "question_category":
                        qa.get(
                            "question_category",
                            "",
                        ),

                    "status":
                        status,
                    "run_status":
                        status,
                    "answer_pre_raw":
                        "",
                    "answer_pre":
                        "",
                    "evidence_pages_pre":
                        None,

                    "pdf_path_resolved":
                        str(
                            pdf_path
                        ),
                    "render_dpi":
                        c.PDF_DPI,

                    "max_slice_nums_used":
                        c.MAX_SLICE_NUMS,
                    "max_new_tokens":
                        c.MAX_NEW_TOKENS,
                    "sampling":
                        c.SAMPLING,
                    "use_image_id":
                        c.USE_IMAGE_ID,
                    "num_beams_override":
                        None,
                    "max_inp_length_override":
                        None,

                    "model_name":
                        c.MODEL_NAME,
                    "model_native_protocol":
                        c.MODEL_NATIVE_PROTOCOL,
                    "prompt_id":
                        prompt_id,
                    "prompt_version":
                        c.PROMPT_VERSION,
                    "prompt_sha256":
                        c.prompt_sha256(
                            instruct
                        ),

                    "gold_answer_sent":
                        False,
                    "gold_evidence_pages_sent":
                        False,

                    "error_type":
                        type(
                            exc
                        ).__name__,
                    "error_message":
                        str(
                            exc
                        ),
                    "error_traceback":
                        traceback.format_exc(),
                    "timing": {
                        "total_seconds":
                            round(
                                time.time()
                                - item_start,
                                4,
                            )
                    },
                }

                atomic_write_json(
                    out_path,
                    results,
                )

                print(
                    f"[{status.upper()}] "
                    f"{sample['item_uid']} "
                    f"| {type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

                gc.collect()
                torch.cuda.empty_cache()

    finally:
        cache.clear()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Final status summary
    # --------------------------------------------------------

    status_counts: Dict[
        str,
        int,
    ] = {}

    for sample in samples:
        row = (
            results
            .get(
                str(
                    sample["unit_id"]
                ),
                {},
            )
            .get("QA", {})
            .get(
                str(
                    sample["qa_id"]
                ),
                {},
            )
        )

        status = str(
            row.get(
                "status",
                "not_run",
            )
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    manifest[
        "completed_at_unix"
    ] = time.time()

    manifest[
        "status_counts"
    ] = status_counts

    manifest[
        "attempted_this_run"
    ] = attempted_this_run

    atomic_write_json(
        manifest_path,
        manifest,
    )

    atomic_write_json(
        out_path,
        results,
    )

    print(
        "=" * 112,
        flush=True,
    )
    print(
        f"[Summary] "
        f"{args.dataset_id} "
        f"| {status_counts}",
        flush=True,
    )
    print(
        f"[Saved] {out_path}",
        flush=True,
    )
    print(
        f"[Manifest] "
        f"{manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
