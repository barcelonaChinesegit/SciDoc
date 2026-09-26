#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ============================================================
# Environment
# ============================================================

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


import pypdfium2 as pdfium
import torch
from PIL import Image

try:
    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
except Exception:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.pdfqa_dataset import (
    build_document_manifest,
    flatten_dataset,
    load_json,
    resolve_pdf_path,
)

from settings.pdfqa_benchmark_config import (
    DATASETS,
    MODEL_DEFAULTS,
    PDF_DPI,
    get_dataset_spec,
)

from settings.pdfqa_prompts import (
    PROMPT_VERSION,
    get_prompt,
    prompt_sha256,
)


# ============================================================
# Model / output config
# ============================================================

MODEL_NAME = "Qwen3-VL-8B"
MODEL_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = MODEL_DIR / "output"
RAW_DIR = OUTPUT_ROOT / "raw_result"
PARSED_DIR = OUTPUT_ROOT / "parsed_result"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

RUN_VERSION = "run-v1"

SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|assistant|>",
    "<|user|>",
)


# ============================================================
# Output helpers
# ============================================================

def ensure_output_dirs() -> None:
    for path in (
        RAW_DIR,
        PARSED_DIR,
        LOG_DIR,
        MANIFEST_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def output_basename(dataset_id: str, dpi: int) -> str:
    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi{dpi}"
        f"__{RUN_VERSION}.json"
    )


def raw_output_path(dataset_id: str, dpi: int) -> Path:
    return RAW_DIR / output_basename(dataset_id, dpi)


def manifest_output_path(dataset_id: str, dpi: int) -> Path:
    name = output_basename(dataset_id, dpi).replace(
        ".json",
        ".manifest.json",
    )
    return MANIFEST_DIR / name


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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

    os.replace(tmp_path, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# Output parsing for resume detection
# ============================================================

def normalize_pages(raw: Any) -> List[int]:
    if raw is None or isinstance(raw, bool):
        return []

    items = raw if isinstance(raw, list) else [raw]

    pages: List[int] = []

    for item in items:
        if isinstance(item, bool):
            continue

        if isinstance(item, int):
            values = [item]
        else:
            values = [
                int(x)
                for x in re.findall(r"\d+", str(item))
            ]

        for value in values:
            if value > 0 and value not in pages:
                pages.append(value)

    return sorted(pages)


def strip_special_tokens(text: str) -> str:
    text = str(text or "").strip()

    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")

    return text.strip()


def quick_parse_prediction(
    raw_text: str,
) -> Optional[Dict[str, Any]]:

    text = strip_special_tokens(raw_text)

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

    if start >= 0 and end > start:
        candidates.append(
            text[start:end + 1]
        )

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
            except Exception:
                continue

        if not isinstance(obj, dict):
            continue

        if (
            "answer_pre" not in obj
            or "evidence_pages" not in obj
        ):
            continue

        return {
            "answer_pre":
                str(obj.get("answer_pre", "")).strip(),
            "evidence_pages":
                normalize_pages(
                    obj.get("evidence_pages")
                ),
        }

    return None


def completed_entry(
    qa_result: Dict[str, Any],
) -> bool:

    return (
        qa_result.get("status") == "completed"
        and isinstance(
            qa_result.get("answer_pre_raw"),
            str,
        )
        and bool(
            qa_result.get(
                "answer_pre_raw",
                "",
            ).strip()
        )
    )


# ============================================================
# PDF rendering
# ============================================================

def pdf_to_images(
    pdf_path: Path,
    dpi: int,
) -> List[Image.Image]:

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    images: List[Image.Image] = []

    pdf = pdfium.PdfDocument(str(pdf_path))

    scale = dpi / 72.0

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]

            bitmap = page.render(scale=scale)

            try:
                image = (
                    bitmap
                    .to_pil()
                    .convert("RGB")
                )
                images.append(image)

            finally:
                try:
                    bitmap.close()
                except Exception:
                    pass

                try:
                    page.close()
                except Exception:
                    pass

    finally:
        try:
            pdf.close()
        except Exception:
            pass

    if not images:
        raise RuntimeError(
            f"No pages rendered: {pdf_path}"
        )

    return images


def close_images(
    images: Sequence[Image.Image],
) -> None:

    for image in images:
        try:
            image.close()
        except Exception:
            pass


class SinglePdfImageCache:

    def __init__(self) -> None:
        self.pdf_path: Optional[Path] = None
        self.images: Optional[List[Image.Image]] = None

    def get(
        self,
        pdf_path: Path,
        dpi: int,
    ) -> List[Image.Image]:

        if (
            self.pdf_path == pdf_path
            and self.images is not None
        ):
            return self.images

        self.clear()

        self.images = pdf_to_images(
            pdf_path,
            dpi,
        )

        self.pdf_path = pdf_path

        return self.images

    def clear(self) -> None:

        if self.images is not None:
            close_images(self.images)

        self.images = None
        self.pdf_path = None

        gc.collect()

    def __del__(self) -> None:
        self.clear()


# ============================================================
# Qwen3-VL inference
# ============================================================

def build_messages(
    images: Sequence[Image.Image],
    question: str,
    system_prompt: str,
    document_manifest: str,
) -> List[Dict[str, Any]]:

    content: List[Dict[str, Any]] = []

    if document_manifest:
        content.append(
            {
                "type": "text",
                "text": document_manifest,
            }
        )

    for page_idx, image in enumerate(
        images,
        start=1,
    ):
        content.append(
            {
                "type": "text",
                "text":
                    f"PDF_PAGE_{page_idx}_START",
            }
        )

        content.append(
            {
                "type": "image",
                "image": image,
            }
        )

        content.append(
            {
                "type": "text",
                "text":
                    f"PDF_PAGE_{page_idx}_END",
            }
        )

    content.append(
        {
            "type": "text",
            "text": f"Question:\n{question}",
        }
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": content,
        },
    ]


@torch.inference_mode()
def infer_one(
    images: Sequence[Image.Image],
    question: str,
    system_prompt: str,
    document_manifest: str,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> Tuple[str, float]:

    messages = build_messages(
        images=images,
        question=question,
        system_prompt=system_prompt,
        document_manifest=document_manifest,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = (
        process_vision_info(messages)
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    start = time.time()

    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated_ids = [
            output[len(inp):]
            for inp, output
            in zip(
                inputs.input_ids,
                output_ids,
            )
        ]

        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        elapsed = time.time() - start

        return (
            output_text[0].strip()
            if output_text
            else "",
            elapsed,
        )

    finally:
        try:
            del inputs
        except Exception:
            pass

        try:
            del output_ids
        except Exception:
            pass


def get_processor_policy(
    processor: Any,
) -> Dict[str, Any]:

    image_processor = getattr(
        processor,
        "image_processor",
        None,
    )

    if image_processor is None:
        return {
            "policy": "native_default",
        }

    result: Dict[str, Any] = {
        "policy": "native_default",
    }

    for name in (
        "min_pixels",
        "max_pixels",
    ):
        if hasattr(image_processor, name):
            value = getattr(
                image_processor,
                name,
            )

            if isinstance(
                value,
                (int, float, str, type(None)),
            ):
                result[name] = value

    size = getattr(
        image_processor,
        "size",
        None,
    )

    if isinstance(size, dict):
        result["size"] = copy.deepcopy(size)

    return result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-VL-8B whole-PDF "
            "PDF-QA runner."
        )
    )

    parser.add_argument(
        "--dataset-id",
        required=True,
        choices=list(DATASETS),
    )

    parser.add_argument(
        "--model-path",
        default=(
            MODEL_DEFAULTS[MODEL_NAME]
            ["model_path"]
        ),
    )

    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=PDF_DPI,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=(
            MODEL_DEFAULTS[MODEL_NAME]
            ["max_new_tokens"]
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 = all remaining",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset(
    dataset_id: str,
    samples: List[Dict[str, Any]],
) -> None:

    spec = get_dataset_spec(dataset_id)

    expected = spec.get("expected_qa_count")
    actual = len(samples)

    if expected is not None:
        expected = int(expected)

        if actual != expected:
            raise RuntimeError(
                f"{dataset_id}: "
                f"expected {expected}, "
                f"found {actual}"
            )

    seen_uid = set()
    resolved: Dict[str, Path] = {}

    for sample in samples:
        uid = sample["item_uid"]

        if uid in seen_uid:
            raise RuntimeError(
                f"Duplicate item_uid: {uid}"
            )

        seen_uid.add(uid)

        unit_id = sample["unit_id"]

        if unit_id not in resolved:
            resolved[unit_id] = resolve_pdf_path(
                dataset_id,
                unit_id,
                sample["unit"],
            )

    print(
        f"[Preflight] "
        f"dataset={dataset_id} "
        f"| QA={actual} "
        f"| PDFs={len(resolved)}",
        flush=True,
    )


def main() -> None:

    args = parse_args()

    ensure_output_dirs()

    if args.pdf_dpi != PDF_DPI:
        raise ValueError(
            f"Protocol requires "
            f"DPI={PDF_DPI}, "
            f"got {args.pdf_dpi}"
        )

    spec = get_dataset_spec(
        args.dataset_id
    )

    data_path = Path(
        spec["json_path"]
    )

    data = load_json(data_path)

    samples = flatten_dataset(
        args.dataset_id,
        data,
    )

    validate_dataset(
        args.dataset_id,
        samples,
    )

    prompt_id, system_prompt = (
        get_prompt(
            spec["prompt_key"]
        )
    )

    save_path = raw_output_path(
        args.dataset_id,
        args.pdf_dpi,
    )

    manifest_path = (
        manifest_output_path(
            args.dataset_id,
            args.pdf_dpi,
        )
    )

    if args.dry_run:
        print(
            f"[DryRun] dataset={args.dataset_id}"
        )
        print(
            f"[DryRun] data={data_path}"
        )
        print(
            f"[DryRun] pdf_root={spec['pdf_root']}"
        )
        print(
            f"[DryRun] output={save_path}"
        )
        print(
            f"[DryRun] prompt_id={prompt_id}"
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable."
        )

    model_cfg = (
        MODEL_DEFAULTS[MODEL_NAME]
    )

    print("=" * 80)
    print(f"Model        : {MODEL_NAME}")
    print(f"Dataset      : {args.dataset_id}")
    print(f"GPU          : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DPI          : {args.pdf_dpi}")
    print(f"Output       : {save_path}")
    print(f"Max samples  : {args.max_samples or 'ALL'}")
    print("=" * 80)

    model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            args.model_path,
            dtype=model_cfg["dtype"],
            device_map=model_cfg["device_map"],
        )
    )

    processor = (
        AutoProcessor.from_pretrained(
            args.model_path
        )
    )

    processor_policy = (
        get_processor_policy(processor)
    )

    if (
        save_path.exists()
        and not args.overwrite
    ):
        results = load_json(
            save_path
        )
    else:
        results = copy.deepcopy(data)

    for unit_id, unit in data.items():

        if unit_id not in results:
            results[unit_id] = copy.deepcopy(unit)

        elif (
            isinstance(unit, dict)
            and isinstance(
                unit.get("QA"),
                dict,
            )
        ):
            results[unit_id].setdefault(
                "QA",
                {},
            )

            for qa_id, qa in (
                unit["QA"].items()
            ):
                results[unit_id]["QA"].setdefault(
                    qa_id,
                    copy.deepcopy(qa),
                )

    manifest = {
        "model_name": MODEL_NAME,
        "model_path": args.model_path,
        "dataset_id": args.dataset_id,
        "dataset_path": str(data_path),
        "dataset_sha256":
            file_sha256(data_path),
        "prompt_id": prompt_id,
        "prompt_version":
            PROMPT_VERSION,
        "prompt_sha256":
            prompt_sha256(system_prompt),
        "pdf_dpi": args.pdf_dpi,
        "max_new_tokens":
            args.max_new_tokens,
        "do_sample": False,
        "cuda_visible_devices":
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "",
            ),
        "torch_version":
            torch.__version__,
        "runner_sha256":
            file_sha256(
                Path(__file__).resolve()
            ),
        "started_at_unix":
            time.time(),
    }

    atomic_write_json(
        manifest_path,
        manifest,
    )

    cache = SinglePdfImageCache()

    attempted_this_run = 0

    try:
        for sample in samples:

            unit_id = sample["unit_id"]
            qa_id = sample["qa_id"]

            qa_result = (
                results[unit_id]
                ["QA"][qa_id]
            )

            if (
                not args.overwrite
                and completed_entry(
                    qa_result
                )
            ):
                continue

            if (
                args.max_samples > 0
                and attempted_this_run
                >= args.max_samples
            ):
                break

            attempted_this_run += 1

            pdf_path = resolve_pdf_path(
                args.dataset_id,
                unit_id,
                sample["unit"],
            )

            document_manifest = (
                build_document_manifest(
                    args.dataset_id,
                    sample["unit"],
                )
            )

            try:
                images = cache.get(
                    pdf_path,
                    args.pdf_dpi,
                )

                page_count = len(images)

                print(
                    f"[{sample['flat_index']}"
                    f"/{len(samples)}] "
                    f"{sample['item_uid']} "
                    f"| pages={page_count}",
                    flush=True,
                )

                raw_text, elapsed = infer_one(
                    images=images,
                    question=str(
                        sample["qa"].get(
                            "question",
                            "",
                        )
                    ),
                    system_prompt=
                        system_prompt,
                    document_manifest=
                        document_manifest,
                    model=model,
                    processor=processor,
                    max_new_tokens=
                        args.max_new_tokens,
                )

                parsed = (
                    quick_parse_prediction(
                        raw_text
                    )
                )

                status = (
                    "completed"
                    if parsed is not None
                    else "parse_failed"
                )

                qa_result.update(
                    {
                        "item_uid":
                            sample["item_uid"],
                        "model_name":
                            MODEL_NAME,
                        "status":
                            status,
                        "answer_pre_raw":
                            raw_text,
                        "elapsed_seconds":
                            round(elapsed, 4),
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
                        "render_dpi":
                            args.pdf_dpi,
                        "prompt_id":
                            prompt_id,
                        "prompt_version":
                            PROMPT_VERSION,
                        "gold_answer_sent":
                            False,
                        "gold_evidence_pages_sent":
                            False,
                        "processor_pixel_policy":
                            processor_policy,
                    }
                )

                print(
                    f"[Done] "
                    f"status={status} "
                    f"elapsed={elapsed:.2f}s",
                    flush=True,
                )

            except Exception as exc:

                qa_result.update(
                    {
                        "item_uid":
                            sample["item_uid"],
                        "model_name":
                            MODEL_NAME,
                        "status":
                            "error",
                        "answer_pre_raw":
                            "",
                        "error_type":
                            type(exc).__name__,
                        "error_message":
                            str(exc),
                        "error_traceback":
                            traceback.format_exc(),
                    }
                )

                print(
                    f"[Error] "
                    f"{sample['item_uid']} "
                    f"| {type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            atomic_write_json(
                save_path,
                results,
            )

    finally:
        cache.clear()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest[
        "completed_at_unix"
    ] = time.time()

    manifest[
        "attempted_this_run"
    ] = attempted_this_run

    atomic_write_json(
        manifest_path,
        manifest,
    )

    atomic_write_json(
        save_path,
        results,
    )

    print("=" * 80)
    print(
        f"[Finished] "
        f"attempted={attempted_this_run}"
    )
    print(
        f"[Saved] {save_path}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
