#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pypdfium2 as pdfium
import torch
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)

# ============================================================
# Shared benchmark imports
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings.pdfqa_benchmark_config import (  # noqa: E402
    DATASETS,
    MODEL_DEFAULTS,
)

from settings.pdfqa_prompts import (  # noqa: E402
    PROMPT_VERSION,
    PROMPTS,
    prompt_sha256,
)


# ============================================================
# Fixed model / protocol constants
# ============================================================

MODEL_NAME = "Gemma-3-27B-IT"

MODEL_NATIVE_PROTOCOL = (
    "gemma3-v2-wholepdf-jpeg95-no-repair"
)

RUN_VERSION = "run-v1"

MODEL_CONFIG = MODEL_DEFAULTS[
    MODEL_NAME
]

MODEL_PATH = Path(
    MODEL_CONFIG["model_path"]
)

PDF_DPI = int(
    MODEL_CONFIG["pdf_dpi"]
)

JPEG_QUALITY = int(
    MODEL_CONFIG["jpeg_quality"]
)

JPEG_SUBSAMPLING = int(
    MODEL_CONFIG["jpeg_subsampling"]
)

MAX_NEW_TOKENS = int(
    MODEL_CONFIG["max_new_tokens"]
)

CONTEXT_LIMIT = int(
    MODEL_CONFIG["context_limit"]
)

ATTN_IMPLEMENTATION = str(
    MODEL_CONFIG["attn_implementation"]
)

DO_SAMPLE = bool(
    MODEL_CONFIG["do_sample"]
)

USE_CACHE = bool(
    MODEL_CONFIG["use_cache"]
)

TRUST_REMOTE_CODE = bool(
    MODEL_CONFIG["trust_remote_code"]
)

PROCESSOR_IMAGE_POLICY = str(
    MODEL_CONFIG["processor_image_policy"]
)


EXPECTED_QA_COUNTS = {
    "ordinary": 1000,
    "unanswerable": 200,
    "reasoning": 200,
    "cross_pdf": 800,
}


# ============================================================
# Paths
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = (
    MODEL_DIR / "output"
)

RAW_DIR = (
    OUTPUT_ROOT / "raw_result"
)

PARSED_DIR = (
    OUTPUT_ROOT / "parsed_result"
)

MANIFEST_DIR = (
    OUTPUT_ROOT / "manifests"
)

LOG_DIR = (
    OUTPUT_ROOT / "logs"
)

RENDER_CACHE_DIR = (
    OUTPUT_ROOT / "render_cache"
)


def output_basename(
    dataset_id: str,
) -> str:
    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi{PDF_DPI}"
        f"__{RUN_VERSION}.json"
    )


def raw_output_path(
    dataset_id: str,
) -> Path:
    return (
        RAW_DIR
        / output_basename(
            dataset_id
        )
    )


def parsed_output_path(
    dataset_id: str,
) -> Path:
    return (
        PARSED_DIR
        / output_basename(
            dataset_id
        )
    )


def manifest_path(
    dataset_id: str,
) -> Path:
    return (
        MANIFEST_DIR
        / output_basename(
            dataset_id
        ).replace(
            ".json",
            ".manifest.json",
        )
    )


# ============================================================
# JSON / dataset
# ============================================================

def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def flatten_dataset(
    dataset_id: str,
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:

    samples: List[
        Dict[str, Any]
    ] = []

    for unit_id, unit in data.items():

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

            item_uid = str(
                qa.get(
                    "item_uid",
                    "",
                )
                or (
                    f"{dataset_id}"
                    f"::{unit_id}"
                    f"::{qa_id}"
                )
            )

            samples.append(
                {
                    "dataset_id":
                        dataset_id,
                    "unit_id":
                        str(unit_id),
                    "qa_id":
                        str(qa_id),
                    "item_uid":
                        item_uid,
                    "unit":
                        unit,
                    "qa":
                        qa,
                }
            )

    return samples


def load_dataset(
    dataset_id: str,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:

    if dataset_id not in DATASETS:
        raise KeyError(
            f"Unknown dataset: "
            f"{dataset_id}"
        )

    spec = DATASETS[
        dataset_id
    ]

    path = Path(
        spec["json_path"]
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    data = load_json(
        path
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"{dataset_id}: "
            "top-level dataset "
            "must be dict"
        )

    samples = flatten_dataset(
        dataset_id,
        data,
    )

    expected = (
        EXPECTED_QA_COUNTS[
            dataset_id
        ]
    )

    if len(samples) != expected:
        raise RuntimeError(
            f"{dataset_id}: "
            f"got {len(samples)} QA, "
            f"expected {expected}"
        )

    return data, samples


# ============================================================
# Prompt
# ============================================================

def task_prompt(
    dataset_id: str,
) -> Tuple[str, str]:

    if dataset_id not in DATASETS:
        raise KeyError(
            dataset_id
        )

    key = DATASETS[
        dataset_id
    ]["prompt_key"]

    if key not in PROMPTS:
        raise KeyError(
            f"Unknown prompt key: "
            f"{key}"
        )

    return PROMPTS[key]


# ============================================================
# PDF resolution
# ============================================================

def resolve_pdf_path(
    dataset_id: str,
    unit_id: str,
    unit: Dict[str, Any],
) -> Path:

    root = Path(
        DATASETS[
            dataset_id
        ]["pdf_root"]
    )

    for key in (
        "pdf_path",
        "merged_pdf_path",
        "paper_path",
    ):
        value = (
            unit.get(key)
            if isinstance(
                unit,
                dict,
            )
            else None
        )

        if not value:
            continue

        p = Path(
            str(value)
        )

        if (
            p.is_absolute()
            and p.is_file()
        ):
            return p

        candidate = (
            root / p
        )

        if candidate.is_file():
            return candidate

    direct = (
        root
        / f"{unit_id}.pdf"
    )

    if direct.is_file():
        return direct

    matches = list(
        root.glob(
            f"{unit_id}*.pdf"
        )
    )

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Cannot resolve PDF "
        f"for dataset={dataset_id} "
        f"unit={unit_id} "
        f"under {root}"
    )


# ============================================================
# JPEG95 PDF render cache
#
# Preserve old Gemma v2 semantics:
#   PDF -> RGB -> JPEG quality=95, subsampling=0
#   then Gemma AutoProcessor performs native normalization.
# ============================================================

def _render_cache_key(
    pdf_path: Path,
) -> str:

    st = pdf_path.stat()

    raw = (
        f"{pdf_path.resolve()}"
        f"|{st.st_size}"
        f"|{st.st_mtime_ns}"
        f"|dpi={PDF_DPI}"
        f"|q={JPEG_QUALITY}"
        f"|sub={JPEG_SUBSAMPLING}"
    )

    return hashlib.sha1(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:24]


def render_pdf_cache(
    pdf_path: Path,
) -> Tuple[
    List[Path],
    List[Tuple[int, int]],
]:

    pdf_path = (
        pdf_path.resolve()
    )

    key = _render_cache_key(
        pdf_path
    )

    cache_dir = (
        RENDER_CACHE_DIR
        / key
    )

    manifest_path_ = (
        cache_dir
        / "manifest.json"
    )

    if manifest_path_.is_file():

        try:
            manifest = load_json(
                manifest_path_
            )

            files = [
                cache_dir / name
                for name
                in manifest.get(
                    "files",
                    [],
                )
            ]

            sizes = [
                tuple(x)
                for x
                in manifest.get(
                    "sizes",
                    [],
                )
            ]

            valid = (
                manifest.get(
                    "pdf_path"
                )
                == str(
                    pdf_path
                )
                and manifest.get(
                    "dpi"
                )
                == PDF_DPI
                and manifest.get(
                    "jpeg_quality"
                )
                == JPEG_QUALITY
                and manifest.get(
                    "jpeg_subsampling"
                )
                == JPEG_SUBSAMPLING
                and files
                and len(files)
                == len(sizes)
                and all(
                    p.is_file()
                    for p in files
                )
            )

            if valid:
                return (
                    files,
                    sizes,
                )

        except Exception:
            pass

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    doc = pdfium.PdfDocument(
        str(pdf_path)
    )

    scale = (
        PDF_DPI
        / 72.0
    )

    files: List[Path] = []
    sizes: List[
        Tuple[int, int]
    ] = []

    try:
        for index in range(
            len(doc)
        ):
            page = doc[index]

            image = (
                page
                .render(
                    scale=scale
                )
                .to_pil()
                .convert("RGB")
            )

            out = (
                cache_dir
                / (
                    f"page_"
                    f"{index + 1:04d}"
                    ".jpg"
                )
            )

            tmp = out.with_suffix(
                ".jpg.tmp"
            )

            image.save(
                tmp,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=(
                    JPEG_SUBSAMPLING
                ),
                optimize=True,
            )

            os.replace(
                tmp,
                out,
            )

            files.append(
                out
            )

            sizes.append(
                tuple(
                    image.size
                )
            )

            image.close()

    finally:
        try:
            doc.close()
        except Exception:
            pass

    manifest = {
        "pdf_path":
            str(pdf_path),
        "dpi":
            PDF_DPI,
        "jpeg_quality":
            JPEG_QUALITY,
        "jpeg_subsampling":
            JPEG_SUBSAMPLING,
        "files":
            [
                p.name
                for p in files
            ],
        "sizes":
            [
                list(x)
                for x in sizes
            ],
        "processor_image_policy":
            PROCESSOR_IMAGE_POLICY,
    }

    with manifest_path_.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return (
        files,
        sizes,
    )


# ============================================================
# Multimodal message
#
# Preserve Gemma v2 structural protocol:
#   system = benchmark prompt
#   user starts with question
#   PDF_PAGE_N_START
#   image
#   PDF_PAGE_N_END
#   ...
#   final JSON-only reminder
#
# Gold answer/evidence can never enter here because this function
# accepts only prompt, question and page image paths.
# ============================================================

def build_multimodal_messages(
    system_prompt: str,
    question: str,
    page_files: List[Path],
) -> List[
    Dict[str, Any]
]:

    if not page_files:
        raise ValueError(
            "No page images supplied"
        )

    user_content: List[
        Dict[str, Any]
    ] = [
        {
            "type": "text",
            "text": (
                "Question: "
                + str(
                    question
                ).strip()
                + "\n"
                "Inspect all supplied pages "
                "before answering. "
                "Physical PDF page labels "
                "are external and 1-based."
            ),
        }
    ]

    for page_no, path in enumerate(
        page_files,
        start=1,
    ):

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"PDF_PAGE_"
                    f"{page_no}_START"
                ),
            }
        )

        user_content.append(
            {
                "type": "image",
                "path": str(
                    path.resolve()
                ),
            }
        )

        user_content.append(
            {
                "type": "text",
                "text": (
                    f"PDF_PAGE_"
                    f"{page_no}_END"
                ),
            }
        )

    user_content.append(
        {
            "type": "text",
            "text": (
                "Now answer the Question above. "
                "Return ONLY the required JSON "
                "object with keys answer_pre "
                "and evidence_pages."
            ),
        }
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text":
                        system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content":
                user_content,
        },
    ]


# ============================================================
# Gemma engine
#
# No:
#   - page dropping
#   - input truncation
#   - second-pass format repair
#   - sampling
# ============================================================

class GemmaEngine:

    def __init__(
        self,
        model_path: Path
        | str = MODEL_PATH,
    ) -> None:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required"
            )

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expected exactly one "
                "visible CUDA GPU, got "
                f"{torch.cuda.device_count()}"
            )

        self.model_path = Path(
            model_path
        )

        if not self.model_path.is_dir():
            raise FileNotFoundError(
                self.model_path
            )

        torch.set_grad_enabled(
            False
        )

        self.device = torch.device(
            "cuda:0"
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=(
                    TRUST_REMOTE_CODE
                ),
            )
        )

        self.model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                self.model_path,
                torch_dtype=(
                    torch.bfloat16
                ),
                device_map={
                    "": 0
                },
                attn_implementation=(
                    ATTN_IMPLEMENTATION
                ),
                local_files_only=True,
                low_cpu_mem_usage=True,
                trust_remote_code=(
                    TRUST_REMOTE_CODE
                ),
            )
            .eval()
        )

        text_cfg = getattr(
            self.model.config,
            "text_config",
            self.model.config,
        )

        self.context_limit = int(
            getattr(
                text_cfg,
                "max_position_embeddings",
                CONTEXT_LIMIT,
            )
        )

        if (
            self.context_limit
            != CONTEXT_LIMIT
        ):
            raise RuntimeError(
                "Gemma context limit "
                f"mismatch: model="
                f"{self.context_limit}, "
                f"config={CONTEXT_LIMIT}"
            )

    def _move_inputs(
        self,
        batch: Any,
    ) -> Dict[
        str,
        Any,
    ]:

        moved: Dict[
            str,
            Any,
        ] = {}

        for key, value in batch.items():

            if not torch.is_tensor(
                value
            ):
                moved[key] = value

            elif value.is_floating_point():
                moved[key] = value.to(
                    self.device,
                    dtype=torch.bfloat16,
                    non_blocking=True,
                )

            else:
                moved[key] = value.to(
                    self.device,
                    non_blocking=True,
                )

        return moved

    def generate(
        self,
        messages: List[
            Dict[str, Any]
        ],
    ) -> Tuple[
        str,
        Dict[str, Any],
    ]:

        torch.cuda.reset_peak_memory_stats(
            0
        )

        prep_start = (
            time.time()
        )

        inputs = (
            self.processor
            .apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        input_tokens = int(
            inputs[
                "input_ids"
            ].shape[-1]
        )

        if (
            input_tokens
            + MAX_NEW_TOKENS
            > self.context_limit
        ):
            del inputs

            gc.collect()

            raise RuntimeError(
                "context_overflow: "
                f"input_tokens="
                f"{input_tokens} "
                f"+ max_new_tokens="
                f"{MAX_NEW_TOKENS} "
                f"> context_limit="
                f"{self.context_limit}; "
                "no truncation is permitted"
            )

        inputs = (
            self._move_inputs(
                inputs
            )
        )

        preprocess_seconds = (
            time.time()
            - prep_start
        )

        input_len = int(
            inputs[
                "input_ids"
            ].shape[-1]
        )

        generation_start = (
            time.time()
        )

        with torch.inference_mode():

            output = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=(
                        MAX_NEW_TOKENS
                    ),
                    do_sample=(
                        DO_SAMPLE
                    ),
                    use_cache=(
                        USE_CACHE
                    ),
                )
            )

        torch.cuda.synchronize()

        generation_seconds = (
            time.time()
            - generation_start
        )

        new_ids = (
            output[
                0,
                input_len:
            ]
        )

        raw = (
            self.processor
            .decode(
                new_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        generated_tokens = int(
            new_ids.numel()
        )

        peak_gib = (
            torch.cuda
            .max_memory_allocated(
                0
            )
            / 1024**3
        )

        stats = {
            "input_tokens":
                input_tokens,
            "generated_tokens":
                generated_tokens,
            "context_limit":
                self.context_limit,
            "context_usage_pct":
                round(
                    100.0
                    * (
                        input_tokens
                        + MAX_NEW_TOKENS
                    )
                    / self.context_limit,
                    3,
                ),
            "preprocess_seconds":
                round(
                    preprocess_seconds,
                    4,
                ),
            "generation_seconds":
                round(
                    generation_seconds,
                    4,
                ),
            "gpu_peak_allocated_gib":
                round(
                    peak_gib,
                    3,
                ),
        }

        del (
            inputs,
            output,
            new_ids,
        )

        return (
            raw,
            stats,
        )

    def cleanup(
        self,
    ) -> None:

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# Convenience single-call wrapper
# ============================================================

def infer_one(
    engine: GemmaEngine,
    dataset_id: str,
    question: str,
    page_files: List[Path],
) -> Tuple[
    str,
    Dict[str, Any],
]:

    prompt_id, system_prompt = (
        task_prompt(
            dataset_id
        )
    )

    messages = (
        build_multimodal_messages(
            system_prompt=(
                system_prompt
            ),
            question=question,
            page_files=(
                page_files
            ),
        )
    )

    raw, stats = (
        engine.generate(
            messages
        )
    )

    stats[
        "prompt_id"
    ] = prompt_id

    stats[
        "prompt_version"
    ] = PROMPT_VERSION

    stats[
        "prompt_sha256"
    ] = prompt_sha256(
        system_prompt
    )

    return (
        raw,
        stats,
    )
