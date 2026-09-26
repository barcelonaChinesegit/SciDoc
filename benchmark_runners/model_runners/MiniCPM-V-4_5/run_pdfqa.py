#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    os.environ.get("PDFQA_GPU", "3"),
)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)

import minicpm_legacy_base as legacy
import minicpm_v41_core as c


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MiniCPM-V-4.5 PDF-QA "
            "context-safe runner"
        )
    )

    p.add_argument(
        "--dataset-id",
        choices=list(c.DATASETS),
        required=True,
    )

    p.add_argument(
        "--model-path",
        default=c.DEFAULT_MODEL_PATH,
    )

    p.add_argument(
        "--retry-technical",
        action="store_true",
        help=(
            "Retry prior OOM/error/context_guard entries."
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
        results.get(unit_id, {})
        .get("QA", {})
        .get(qa_id)
    )


def should_skip(
    existing: Any,
    retry_technical: bool,
) -> bool:
    if not isinstance(existing, dict):
        return False

    status = str(
        existing.get("status", "")
    ).lower()

    # These are completed model behaviors.
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
        "context_guard_exceeded",
    }


def build_unit_shell(
    unit_id: str,
    unit: Dict[str, Any],
    pdf_path: Path,
    page_count: int,
) -> Dict[str, Any]:
    return {
        "paper": unit.get(
            "paper",
            unit_id,
        ),
        "primary_category": unit.get(
            "primary_category",
            "",
        ),
        "secondary_category": unit.get(
            "secondary_category",
            "",
        ),
        "pdf_path": str(pdf_path),
        "pdf_page_count": page_count,
        "QA": {},
    }


def format_eta(
    seconds: float,
) -> str:
    if seconds < 0:
        seconds = 0

    total = int(round(seconds))

    h, rem = divmod(
        total,
        3600,
    )
    m, s = divmod(
        rem,
        60,
    )

    if h:
        return f"{h}h{m:02d}m"

    if m:
        return f"{m}m{s:02d}s"

    return f"{s}s"


def count_output_tokens(
    tokenizer: Any,
    raw: str,
) -> int:
    if not raw:
        return 0

    try:
        ids = tokenizer(
            raw,
            add_special_tokens=False,
        ).input_ids

        if isinstance(ids, list):
            return len(ids)

    except Exception:
        pass

    return -1


def main() -> None:
    args = parse_args()

    _, samples = c.load_dataset(
        args.dataset_id
    )

    prompt_id, system_prompt = (
        c.task_prompt(
            args.dataset_id
        )
    )

    out_path = c.raw_output_path(
        args.dataset_id
    )

    manifest_path = c.manifest_path(
        args.dataset_id
    )

    # Resolve unique PDFs during preflight.
    resolved = {}

    for sample in samples:
        unit_id = str(sample["unit_id"])

        if unit_id not in resolved:
            resolved[unit_id] = c.resolve_pdf_path(
                args.dataset_id,
                unit_id,
                sample["unit"],
            )

    print(
        f"[Preflight] dataset={args.dataset_id} "
        f"| QA={len(samples)} "
        f"| PDFs={len(resolved)}",
        flush=True,
    )

    if args.dry_run:
        spec = c.DATASETS[args.dataset_id]

        print(
            f"[DryRun] data={spec['json_path']}",
            flush=True,
        )
        print(
            f"[DryRun] pdf_root={spec['pdf_root']}",
            flush=True,
        )
        print(
            f"[DryRun] output={out_path}",
            flush=True,
        )
        print(
            f"[DryRun] prompt_id={prompt_id}",
            flush=True,
        )
        print(
            f"[DryRun] prompt_version={c.PROMPT_FAMILY_VERSION}",
            flush=True,
        )
        print(
            f"[DryRun] prompt_sha256={c.prompt_sha256(system_prompt)}",
            flush=True,
        )
        print(
            f"[DryRun] pdf_dpi={c.PDF_DPI}",
            flush=True,
        )
        print(
            f"[DryRun] slice_policy=contextfit-v4.1",
            flush=True,
        )
        print(
            f"[DryRun] max_slice_nums={c.MAX_SLICE_NUMS}",
            flush=True,
        )
        print(
            f"[DryRun] preferred_input_tokens={c.PREFERRED_INPUT_TOKENS}",
            flush=True,
        )
        print(
            f"[DryRun] hard_max_input_tokens={c.HARD_MAX_INPUT_TOKENS}",
            flush=True,
        )
        print(
            f"[DryRun] max_new_tokens={c.MAX_NEW_TOKENS}",
            flush=True,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU, "
            f"got {torch.cuda.device_count()}"
        )

    if (
        args.overwrite
        or not out_path.exists()
    ):
        results: Dict[str, Any] = {}
    else:
        results = legacy.load_json(
            out_path
        )

    visible_gpu = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        "",
    )

    print("=" * 122, flush=True)
    print(
        "MiniCPM-V-4.5 PDF-QA",
        flush=True,
    )
    print(
        f"Dataset               : "
        f"{args.dataset_id}",
        flush=True,
    )
    print(
        f"Dataset JSON          : "
        f"{c.DATASETS[args.dataset_id]['json_path']}",
        flush=True,
    )
    print(
        f"Expected              : "
        f"{c.DATASETS[args.dataset_id]['expected']}",
        flush=True,
    )
    print(
        f"Model                 : "
        f"{args.model_path}",
        flush=True,
    )
    print(
        f"Physical CUDA_VISIBLE : "
        f"{visible_gpu}",
        flush=True,
    )
    print(
        "Logical device        : cuda:0",
        flush=True,
    )
    print(
        f"GPU                    : "
        f"{torch.cuda.get_device_name(0)}",
        flush=True,
    )
    print(
        f"GPU total VRAM         : "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB",
        flush=True,
    )
    print(
        f"Protocol               : "
        f"{c.MODEL_NATIVE_PROTOCOL}",
        flush=True,
    )
    print(
        f"PDF DPI                : "
        f"{c.PDF_DPI}",
        flush=True,
    )
    print(
        "Page dropping          : NEVER",
        flush=True,
    )
    print(
        "Document map           : NEVER for Cross",
        flush=True,
    )
    print(
        "Gold answer/evidence   : NEVER SENT",
        flush=True,
    )
    print(
        f"Preferred input        : "
        f"<= {c.PREFERRED_INPUT_TOKENS}",
        flush=True,
    )
    print(
        f"Hard input cap         : "
        f"{c.HARD_MAX_INPUT_TOKENS}",
        flush=True,
    )
    print(
        f"Max new tokens         : "
        f"{c.MAX_NEW_TOKENS}",
        flush=True,
    )
    print(
        f"Max slice/page         : "
        f"{c.MAX_SLICE_NUMS}",
        flush=True,
    )
    print(
        "Page labels            : one [Page N] before each page image",
        flush=True,
    )
    print(
        "Question/output rules  : appended after all images",
        flush=True,
    )
    print("=" * 122, flush=True)

    print(
        "[Load] model...",
        flush=True,
    )

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation=(
            c.ATTN_IMPLEMENTATION
        ),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().cuda()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
    )

    processor = (
        AutoProcessor.from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
    )

    model.processor = processor

    llm_context_limit = (
        legacy.get_llm_context_limit(
            model
        )
    )

    print(
        f"[Load] GPU={torch.cuda.get_device_name(0)} "
        f"| llm_context={llm_context_limit}",
        flush=True,
    )

    cache = legacy.SinglePdfImageCache()

    counts = {
        "completed": 0,
        "normalized_completed": 0,
        "parse_failed": 0,
        "oom_single_gpu": 0,
        "context_guard_exceeded": 0,
        "error": 0,
        "skipped": 0,
    }

    run_wall_start = time.time()
    processed_now = 0
    attempted_this_run = 0
    cumulative_total_seconds = 0.0

    manifest = {
        "model": c.MODEL_NAME,
        "dataset_id": args.dataset_id,
        "run_version": c.RUN_VERSION,
        "prompt_id": prompt_id,
        "prompt_version": c.PROMPT_FAMILY_VERSION,
        "prompt_sha256": c.prompt_sha256(system_prompt),
        "semantic_prompt_family": (
            c.PROMPT_FAMILY_VERSION
        ),
        "model_native_protocol": (
            c.MODEL_NATIVE_PROTOCOL
        ),
        "dataset_json": (
            c.DATASETS[
                args.dataset_id
            ]["json_path"]
        ),
        "pdf_root": (
            c.DATASETS[
                args.dataset_id
            ]["pdf_root"]
        ),
        "pdf_dpi": c.PDF_DPI,
        "page_dropping": False,
        "document_manifest_mode": "none",
        "gold_answer_sent": False,
        "gold_evidence_sent": False,
        "preferred_input_tokens": (
            c.PREFERRED_INPUT_TOKENS
        ),
        "hard_max_input_tokens": (
            c.HARD_MAX_INPUT_TOKENS
        ),
        "max_new_tokens": (
            c.MAX_NEW_TOKENS
        ),
        "max_slice_nums": (
            c.MAX_SLICE_NUMS
        ),
        "page_label_policy": (
            "single_[Page_N]_before_each_image"
        ),
        "question_and_output_rules_at_end": True,
        "concrete_json_example_in_wrapper": False,
        "doc_id_not_page_id_rule": True,
        "enable_thinking": False,
        "sampling": False,
        "stream": False,
        "use_image_id": False,
        "visible_cuda": visible_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_vram_gib": round(
            torch.cuda.get_device_properties(
                0
            ).total_memory
            / 1024**3,
            3,
        ),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    try:
        for idx, sample in enumerate(
            samples,
            1,
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
                counts["skipped"] += 1

                print(
                    f"[{idx}/{len(samples)}] "
                    f"SKIP {unit_id}/{qa_id} "
                    f"| status="
                    f"{existing.get('status')}",
                    flush=True,
                )

                continue

            if (
                args.max_samples > 0
                and attempted_this_run >= args.max_samples
            ):
                break

            attempted_this_run += 1
            item_wall_start = time.time()
            render_start = time.time()

            try:
                pdf_path = (
                    c.resolve_pdf_path(
                        args.dataset_id,
                        unit_id,
                        unit,
                    )
                )

                images = cache.get(
                    pdf_path,
                    dpi=c.PDF_DPI,
                )

                page_count = len(images)
                render_seconds = (
                    time.time()
                    - render_start
                )

                msgs = c.build_messages(
                    images,
                    str(
                        qa.get(
                            "question",
                            "",
                        )
                    ),
                )

                context_start = time.time()

                slice_num, budget = (
                    c.choose_slice_budget(
                        msgs=msgs,
                        system_prompt=(
                            system_prompt
                        ),
                        processor=processor,
                        llm_context_limit=(
                            llm_context_limit
                        ),
                    )
                )

                context_seconds = (
                    time.time()
                    - context_start
                )

                selected_ctx = (
                    budget["selected"]
                )

                pre = int(
                    selected_ctx[
                        "input_tokens_pre_truncation"
                    ]
                )

                print(
                    f"[{idx}/{len(samples)}] "
                    f"{sample['item_uid']} "
                    f"| pages={page_count} "
                    f"| slice={slice_num} "
                    f"| pre={pre}/"
                    f"{c.HARD_MAX_INPUT_TOKENS} "
                    f"| soft_exceeded="
                    f"{budget['soft_budget_exceeded']}",
                    flush=True,
                )

                infer_seconds = 0.0
                parse_seconds = 0.0
                raw = ""
                parsed = None
                output_tokens = 0
                normalized_by_parser = False

                if slice_num is None:
                    status = (
                        "context_guard_exceeded"
                    )
                    strict_reason = (
                        "context_guard_exceeded_even_at_slice1"
                    )

                else:
                    infer_start = time.time()

                    raw_obj = model.chat(
                        msgs=msgs,
                        tokenizer=tokenizer,
                        processor=processor,
                        system_prompt=(
                            system_prompt
                        ),
                        enable_thinking=False,
                        stream=False,
                        sampling=False,
                        max_new_tokens=(
                            c.MAX_NEW_TOKENS
                        ),
                        max_inp_length=(
                            c.HARD_MAX_INPUT_TOKENS
                        ),
                        max_slice_nums=(
                            slice_num
                        ),
                        use_image_id=False,
                    )

                    infer_seconds = (
                        time.time()
                        - infer_start
                    )

                    raw = (
                        ""
                        if raw_obj is None
                        else str(
                            raw_obj
                        ).strip()
                    )

                    output_tokens = (
                        count_output_tokens(
                            tokenizer,
                            raw,
                        )
                    )

                    parse_start = time.time()

                    ok, strict_reason, parsed = (
                        c.strict_schema_parse(
                            raw,
                            allow_fence_normalization=True,
                        )
                    )

                    parse_seconds = (
                        time.time()
                        - parse_start
                    )

                    if ok:
                        if (
                            strict_reason
                            == "normalized_markdown_fence"
                        ):
                            status = (
                                "normalized_completed"
                            )
                            normalized_by_parser = True
                        else:
                            status = "completed"
                    else:
                        status = "parse_failed"

                if unit_id not in results:
                    results[unit_id] = (
                        build_unit_shell(
                            unit_id,
                            unit,
                            pdf_path,
                            page_count,
                        )
                    )

                results[
                    unit_id
                ].setdefault(
                    "QA",
                    {},
                )

                pre_save_elapsed = (
                    time.time()
                    - item_wall_start
                )

                entry = {
                    "question": qa.get(
                        "question",
                        "",
                    ),
                    # Gold fields stored for evaluation only.
                    "answer": qa.get(
                        "answer",
                        "",
                    ),
                    "evidence_pages": qa.get(
                        "evidence_pages",
                        [],
                    ),
                    "modal_types": qa.get(
                        "modal_types",
                        [],
                    ),
                    "question_type": qa.get(
                        "question_type",
                        "",
                    ),
                    "question_category": qa.get(
                        "question_category",
                        "",
                    ),
                    "prompt_id": prompt_id,
                    "prompt_version": c.PROMPT_FAMILY_VERSION,
                    "prompt_sha256": c.prompt_sha256(system_prompt),
                    "status": status,
                    "run_status": (
                        "completed"
                        if status
                        in {
                            "completed",
                            "normalized_completed",
                            "parse_failed",
                        }
                        else status
                    ),
                    "answer_pre_raw": raw,
                    "answer_pre": (
                        parsed.get(
                            "answer_pre"
                        )
                        if isinstance(
                            parsed,
                            dict,
                        )
                        and status
                        in {
                            "completed",
                            "normalized_completed",
                        }
                        else ""
                    ),
                    "evidence_pages_pre": (
                        parsed.get(
                            "evidence_pages"
                        )
                        if isinstance(
                            parsed,
                            dict,
                        )
                        and status
                        in {
                            "completed",
                            "normalized_completed",
                        }
                        else None
                    ),
                    "strict_format_reason": strict_reason,
                    "normalized_by_parser": normalized_by_parser,
                    "output_tokens": output_tokens,
                    "output_chars": len(raw),
                    "pdf_path_resolved": str(
                        pdf_path
                    ),
                    "pdf_total_pages": (
                        page_count
                    ),
                    "pdf_pages_supplied": list(
                        range(
                            1,
                            page_count + 1,
                        )
                    ),
                    "pdf_pages_supplied_count": (
                        page_count
                    ),
                    "render_dpi": c.PDF_DPI,
                    "max_slice_nums_used": (
                        slice_num
                    ),
                    "input_tokens_pre_truncation": (
                        pre
                    ),
                    "preferred_input_tokens": (
                        c.PREFERRED_INPUT_TOKENS
                    ),
                    "max_input_length": (
                        c.HARD_MAX_INPUT_TOKENS
                    ),
                    "max_new_tokens": (
                        c.MAX_NEW_TOKENS
                    ),
                    "llm_context_limit": (
                        llm_context_limit
                    ),
                    "soft_budget_exceeded": (
                        budget[
                            "soft_budget_exceeded"
                        ]
                    ),
                    "slice_selection_reason": (
                        budget[
                            "selection_reason"
                        ]
                    ),
                    "slice_trials": (
                        budget["trials"]
                    ),
                    "model_native_protocol": (
                        c.MODEL_NATIVE_PROTOCOL
                    ),
                    "page_label_policy": (
                        "single_[Page_N]_before_each_image"
                    ),
                    "question_and_rules_at_end": True,
                    "gold_fields_sent": False,
                    "timing": {
                        "render_seconds": round(
                            render_seconds,
                            4,
                        ),
                        "context_select_seconds": round(
                            context_seconds,
                            4,
                        ),
                        "inference_seconds": round(
                            infer_seconds,
                            4,
                        ),
                        "parse_seconds": round(
                            parse_seconds,
                            4,
                        ),
                        # save_seconds filled after first write.
                        "save_seconds": None,
                        "total_seconds": None,
                    },
                }

                results[
                    unit_id
                ]["QA"][
                    qa_id
                ] = entry

                save_start = time.time()

                atomic_write_json(
                    out_path,
                    results,
                )

                save_seconds = (
                    time.time()
                    - save_start
                )

                total_seconds = (
                    time.time()
                    - item_wall_start
                )

                entry["timing"][
                    "save_seconds"
                ] = round(
                    save_seconds,
                    4,
                )

                entry["timing"][
                    "total_seconds"
                ] = round(
                    total_seconds,
                    4,
                )

                # Persist final timing once more.
                atomic_write_json(
                    out_path,
                    results,
                )

                counts[status] += 1
                processed_now += 1
                cumulative_total_seconds += (
                    total_seconds
                )

                avg_total = (
                    cumulative_total_seconds
                    / processed_now
                )

                remaining = (
                    len(samples) - idx
                )

                eta = format_eta(
                    avg_total
                    * remaining
                )

                print(
                    "[Timing] "
                    f"render={render_seconds:.2f}s "
                    f"| context={context_seconds:.2f}s "
                    f"| infer={infer_seconds:.2f}s "
                    f"| parse={parse_seconds:.3f}s "
                    f"| save={save_seconds:.3f}s "
                    f"| total={total_seconds:.2f}s",
                    flush=True,
                )

                print(
                    "[Output] "
                    f"tokens={output_tokens} "
                    f"| chars={len(raw)} "
                    f"| status={status}",
                    flush=True,
                )

                print(
                    "[Progress] "
                    f"processed_now={processed_now} "
                    f"| avg_total={avg_total:.2f}s "
                    f"| ETA≈{eta}",
                    flush=True,
                )

                print(
                    f"[Raw] {raw[:2000]}",
                    flush=True,
                )

            except Exception as exc:
                total_seconds = (
                    time.time()
                    - item_wall_start
                )

                status = (
                    "oom_single_gpu"
                    if c.is_cuda_oom_error(
                        exc
                    )
                    else "error"
                )

                counts[status] += 1
                processed_now += 1
                cumulative_total_seconds += (
                    total_seconds
                )

                try:
                    pdf_path = (
                        c.resolve_pdf_path(
                            args.dataset_id,
                            unit_id,
                            unit,
                        )
                    )
                except Exception:
                    pdf_path = Path("")

                if unit_id not in results:
                    results[unit_id] = (
                        build_unit_shell(
                            unit_id,
                            unit,
                            pdf_path,
                            0,
                        )
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
                    "question": qa.get(
                        "question",
                        "",
                    ),
                    "answer": qa.get(
                        "answer",
                        "",
                    ),
                    "evidence_pages": qa.get(
                        "evidence_pages",
                        [],
                    ),
                    "status": status,
                    "run_status": status,
                    "answer_pre_raw": "",
                    "answer_pre": "",
                    "evidence_pages_pre": None,
                    "strict_format_reason": (
                        type(exc).__name__
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    "model_native_protocol": (
                        c.MODEL_NATIVE_PROTOCOL
                    ),
                    "gold_fields_sent": False,
                    "timing": {
                        "total_seconds": round(
                            total_seconds,
                            4,
                        ),
                    },
                }

                atomic_write_json(
                    out_path,
                    results,
                )

                avg_total = (
                    cumulative_total_seconds
                    / processed_now
                )

                remaining = (
                    len(samples)
                    - idx
                )

                print(
                    f"[Error] "
                    f"{sample['item_uid']} "
                    f"| {type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

                print(
                    "[Timing] "
                    f"total={total_seconds:.2f}s "
                    f"| avg_total={avg_total:.2f}s "
                    f"| ETA≈"
                    f"{format_eta(avg_total * remaining)}",
                    flush=True,
                )

                traceback.print_exc()

                gc.collect()
                torch.cuda.empty_cache()

    finally:
        cache.clear()

        atomic_write_json(
            out_path,
            results,
        )

        gc.collect()
        torch.cuda.empty_cache()

    run_total = (
        time.time()
        - run_wall_start
    )

    print("=" * 122, flush=True)
    print(
        f"[Summary] {counts}",
        flush=True,
    )
    print(
        f"[Run total] "
        f"{format_eta(run_total)}",
        flush=True,
    )
    print(
        f"[Saved] {out_path}",
        flush=True,
    )
    print(
        f"[Manifest] {manifest_path}",
        flush=True,
    )
    print("=" * 122, flush=True)


if __name__ == "__main__":
    main()
