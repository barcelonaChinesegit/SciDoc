from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


# ----------------------------------------------------------------------
# Shared benchmark imports
# ----------------------------------------------------------------------

import sys

SXZ_ROOT = Path(__file__).resolve().parents[2]

if str(SXZ_ROOT) not in sys.path:
    sys.path.insert(0, str(SXZ_ROOT))

from common.pdfqa_dataset import (
    flatten_dataset as shared_flatten_dataset,
    load_json as shared_load_json,
    resolve_pdf_path as shared_resolve_pdf_path,
)

from settings.pdfqa_benchmark_config import (
    DATASETS as SHARED_DATASETS,
    MODEL_DEFAULTS,
)

from settings.pdfqa_prompts import (
    PROMPT_VERSION,
    get_prompt,
    prompt_sha256,
)


# ----------------------------------------------------------------------
# MiniCPM-V-2.6 reference-faithful contract
# ----------------------------------------------------------------------

MODEL_NAME = "MiniCPM-V-2_6"

MODEL_NATIVE_PROTOCOL = (
    "minicpm-v2.6-referencefaithful-v3"
)

RUN_VERSION = "run-v1"

CFG = MODEL_DEFAULTS[MODEL_NAME]

MODEL_PATH = Path(CFG["model_path"])

PDF_DPI = int(CFG["pdf_dpi"])

PAGE_W = int(CFG["page_width"])
PAGE_H = int(CFG["page_height"])
PAGE_LABEL_H = int(CFG["page_label_height"])

MAX_SLICE_NUMS = int(
    CFG["max_slice_nums"]
)

MAX_NEW_TOKENS = int(
    CFG["max_new_tokens"]
)

USE_IMAGE_ID = bool(
    CFG["use_image_id"]
)

SAMPLING = bool(
    CFG["sampling"]
)

ATTN_IMPLEMENTATION = str(
    CFG["attn_implementation"]
)


# ----------------------------------------------------------------------
# Output paths
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_ROOT = BASE_DIR / "output"
RAW_DIR = OUTPUT_ROOT / "raw_result"
PARSED_DIR = OUTPUT_ROOT / "parsed_result"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"


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
        / output_basename(dataset_id)
    )


def parsed_output_path(
    dataset_id: str,
) -> Path:

    return (
        PARSED_DIR
        / output_basename(dataset_id)
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


# ----------------------------------------------------------------------
# Shared final_2200 datasets
# ----------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, Any]] = {
    dataset_id: {
        "json_path": spec["json_path"],
        "pdf_root": spec["pdf_root"],
        "expected": spec.get(
            "expected_qa_count"
        ),
        "prompt_key": spec["prompt_key"],
    }
    for dataset_id, spec
    in SHARED_DATASETS.items()
}


def load_dataset(
    dataset_id: str,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:

    if dataset_id not in DATASETS:
        raise KeyError(
            f"Unknown dataset: {dataset_id}"
        )

    spec = DATASETS[dataset_id]

    path = Path(
        spec["json_path"]
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset JSON not found: {path}"
        )

    data = shared_load_json(path)

    samples = shared_flatten_dataset(
        dataset_id,
        data,
    )

    expected = spec.get(
        "expected"
    )

    if expected is not None:
        expected = int(expected)

        if len(samples) != expected:
            raise RuntimeError(
                f"Dataset count mismatch "
                f"for {dataset_id}: "
                f"got {len(samples)}, "
                f"expected {expected}"
            )

    return data, samples


def resolve_pdf_path(
    dataset_id: str,
    unit_id: str,
    unit: Dict[str, Any],
) -> Path:

    return shared_resolve_pdf_path(
        dataset_id,
        unit_id,
        unit,
    )


def task_prompt(
    dataset_id: str,
) -> Tuple[str, str]:

    if dataset_id not in DATASETS:
        raise KeyError(
            f"Unknown dataset: {dataset_id}"
        )

    return get_prompt(
        DATASETS[dataset_id][
            "prompt_key"
        ]
    )


# ----------------------------------------------------------------------
# Reference-faithful PDF rendering
#
# Preserved from referencefaithful-v3:
#
# - DPI 144
# - page canvas 1008 x 1344
# - 64 px label area
# - visible PDF_PAGE_N label
# ----------------------------------------------------------------------

def add_page_label_and_normalize(
    image: Image.Image,
    page_idx: int,
) -> Image.Image:

    image = image.convert("RGB")

    canvas = Image.new(
        "RGB",
        (PAGE_W, PAGE_H),
        (255, 255, 255),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    label = (
        f"PDF_PAGE_{page_idx}"
    )

    draw.rectangle(
        [
            0,
            0,
            PAGE_W,
            PAGE_LABEL_H,
        ],
        fill=(
            245,
            245,
            245,
        ),
    )

    draw.text(
        (24, 16),
        label,
        fill=(0, 0, 0),
    )

    content_w = PAGE_W
    content_h = (
        PAGE_H
        - PAGE_LABEL_H
    )

    page = image.copy()

    page.thumbnail(
        (
            content_w,
            content_h,
        ),
        Image.Resampling.LANCZOS,
    )

    paste_x = (
        content_w
        - page.width
    ) // 2

    paste_y = (
        PAGE_LABEL_H
        + (
            content_h
            - page.height
        ) // 2
    )

    canvas.paste(
        page,
        (
            paste_x,
            paste_y,
        ),
    )

    try:
        page.close()
    except Exception:
        pass

    return canvas


def pdf_to_images(
    pdf_path: str | Path,
) -> List[Image.Image]:

    pdf_path = str(
        pdf_path
    )

    if not os.path.isfile(
        pdf_path
    ):
        raise FileNotFoundError(
            f"PDF not found: "
            f"{pdf_path}"
        )

    images: List[
        Image.Image
    ] = []

    pdf = pdfium.PdfDocument(
        pdf_path
    )

    scale = (
        PDF_DPI
        / 72.0
    )

    try:
        for page_idx, page in enumerate(
            pdf,
            start=1,
        ):

            source = (
                page
                .render(
                    scale=scale
                )
                .to_pil()
                .convert("RGB")
            )

            normalized = (
                add_page_label_and_normalize(
                    source,
                    page_idx,
                )
            )

            try:
                source.close()
            except Exception:
                pass

            images.append(
                normalized
            )

    finally:
        try:
            pdf.close()
        except Exception:
            pass

    return images


def close_images(
    images: List[
        Image.Image
    ],
) -> None:

    for image in images:
        try:
            image.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Reference-faithful message structure
#
# Exact structure:
#
# [
#   page1 PIL,
#   page2 PIL,
#   ...,
#   one combined prompt + question string
# ]
#
# No:
# - system_prompt
# - per-page textual START/END
# - AutoProcessor created by runner
# ----------------------------------------------------------------------

def build_messages(
    pdf_images: List[
        Image.Image
    ],
    question: str,
    instruct: str,
) -> List[
    Dict[str, Any]
]:

    if not pdf_images:
        raise ValueError(
            "No PDF pages supplied"
        )

    user_text = (
        instruct
        + "\n\n"
        + (
            "The PDF page images are "
            "provided above in order. "
        )
        + (
            "Each image has an external "
            "page label drawn at the top, "
            "such as PDF_PAGE_1. "
        )
        + (
            "Use this drawn external label "
            "as the evidence page number. "
        )
        + (
            "Do not use page numbers printed "
            "inside the paper body, header, "
            "or footer.\n\n"
        )
        + (
            "Now answer the following question "
            "strictly in the required JSON "
            "format.\n"
        )
        + f"Question: {question}"
    )

    content: List[Any] = []

    content.extend(
        pdf_images
    )

    content.append(
        user_text
    )

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


# ----------------------------------------------------------------------
# Reference-faithful model.chat
#
# Deliberately NO:
# - processor=
# - system_prompt=
# - num_beams=
# - max_inp_length=
#
# These omissions are part of the v3 reference contract.
# ----------------------------------------------------------------------

def model_chat(
    model,
    tokenizer,
    pdf_images: List[
        Image.Image
    ],
    question: str,
    instruct: str,
):

    msgs = build_messages(
        pdf_images,
        question,
        instruct,
    )

    start = time.time()

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        use_image_id=USE_IMAGE_ID,
        max_slice_nums=MAX_SLICE_NUMS,
        sampling=SAMPLING,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    elapsed = (
        time.time()
        - start
    )

    return answer, elapsed
