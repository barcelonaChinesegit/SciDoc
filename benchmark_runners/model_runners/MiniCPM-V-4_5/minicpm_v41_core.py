#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Reuse the already-working MiniCPM PDF cache and exact context inspector
# from the existing run_pdfqa.py beside this package.
import minicpm_legacy_base as legacy

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

MODEL_NAME = "MiniCPM-V-4_5"
MODEL_NATIVE_PROTOCOL = "minicpm-v4.1-contextsafe-singlelabel"
PROMPT_FAMILY_VERSION = PROMPT_VERSION
RUN_VERSION = "run-v1"

PDF_DPI = 144

_MODEL_CFG = MODEL_DEFAULTS[MODEL_NAME]

MAX_NEW_TOKENS = int(_MODEL_CFG["max_new_tokens"])
MAX_SLICE_NUMS = int(_MODEL_CFG["max_slice_nums"])
PREFERRED_INPUT_TOKENS = int(_MODEL_CFG["preferred_input_tokens"])
HARD_MAX_INPUT_TOKENS = int(_MODEL_CFG["max_input_length"])
EXPECTED_LLM_CONTEXT = int(_MODEL_CFG["expected_llm_context"])
ATTN_IMPLEMENTATION = str(_MODEL_CFG["attn_implementation"])

DEFAULT_GPU = "2"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "output"
RAW_DIR = OUTPUT_ROOT / "raw_result"
PARSED_DIR = OUTPUT_ROOT / "parsed_result"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

DEFAULT_MODEL_PATH = str(_MODEL_CFG["model_path"])

DATASETS: Dict[str, Dict[str, Any]] = {
    dataset_id: {
        "json_path": spec["json_path"],
        "pdf_root": spec["pdf_root"],
        "expected": spec.get("expected_qa_count"),
        "prompt_key": spec["prompt_key"],
    }
    for dataset_id, spec in SHARED_DATASETS.items()
}


# MiniCPM-specific output wrapper.
# Important: no concrete answer/page-number example is given.
COMMON_OUTPUT_RULES = """Important output rules:
- Return exactly one JSON object and no text outside the JSON object.
- The JSON object must contain exactly two fields: answer_pre and evidence_pages.
- answer_pre must contain your actual concise answer as a JSON string.
- evidence_pages must be a JSON array containing only the actual supporting 1-based physical PDF page numbers as JSON integers.
- Use the external [Page N] labels supplied before the PDF page images.
- Do not use page numbers printed inside the paper body, header, or footer.
- "Doc 1", "Doc 2", and "Doc 3" are document identities, NOT page numbers.
- Never output page 1, 2, or 3 merely because the question mentions Doc 1, Doc 2, or Doc 3.
- Use the smallest set of physical PDF pages that directly supports the answer.
- Do not output document names, section names, headings, page-label strings, Markdown fences, <think>, or any explanation outside the JSON object.
- Do not continue or imitate the page-label sequence."""


def task_prompt(dataset_id: str) -> Tuple[str, str]:
    if dataset_id not in DATASETS:
        raise KeyError(dataset_id)

    return get_prompt(
        DATASETS[dataset_id]["prompt_key"]
    )


def task_prompt_id(dataset_id: str) -> str:
    return task_prompt(dataset_id)[0]


def task_system_prompt(dataset_id: str) -> str:
    return task_prompt(dataset_id)[1]


def build_messages(
    images: Sequence[Any],
    question: str,
) -> List[Dict[str, Any]]:
    # v4.1 keeps the successful v4 structural changes:
    # 1) one compact [Page N] label per page;
    # 2) question and output rules appear AFTER all images;
    # 3) no concrete page-number example that MiniCPM can copy.
    content: List[Any] = []

    for page_no, image in enumerate(images, start=1):
        content.append(f"[Page {page_no}]")
        content.append(image)

    content.append(
        "Question: "
        + str(question).strip()
        + "\n\n"
        + COMMON_OUTPUT_RULES
    )

    return [{"role": "user", "content": content}]


def load_dataset(
    dataset_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_id}")

    spec = DATASETS[dataset_id]
    path = Path(spec["json_path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found: {path}"
        )

    data = shared_load_json(path)
    samples = shared_flatten_dataset(dataset_id, data)

    expected = spec.get("expected")

    if expected is not None:
        expected = int(expected)

        if len(samples) != expected:
            raise RuntimeError(
                f"Dataset count mismatch for {dataset_id}: "
                f"got {len(samples)}, expected {expected}"
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


def output_basename(dataset_id: str) -> str:
    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi{PDF_DPI}"
        f"__{RUN_VERSION}.json"
    )


def raw_output_path(dataset_id: str) -> Path:
    return RAW_DIR / output_basename(dataset_id)


def parsed_output_path(dataset_id: str) -> Path:
    return PARSED_DIR / output_basename(dataset_id)


def manifest_path(dataset_id: str) -> Path:
    return MANIFEST_DIR / output_basename(
        dataset_id
    ).replace(".json", ".manifest.json")


def strip_markdown_fence(text: str) -> Tuple[str, bool]:
    s = str(text or "").strip()

    if not s:
        return "", False

    if s.startswith("```") and s.endswith("```"):
        lines = s.splitlines()

        if len(lines) >= 3:
            first = lines[0].strip().lower()
            last = lines[-1].strip()

            if first in {"```", "```json"} and last == "```":
                return "\n".join(lines[1:-1]).strip(), True

    return s, False


def strict_schema_parse(
    raw: str,
    *,
    allow_fence_normalization: bool = True,
) -> Tuple[bool, str, Dict[str, Any] | None]:
    text = legacy.strip_special_tokens(raw)
    normalized_by_fence = False

    if allow_fence_normalization:
        text, normalized_by_fence = strip_markdown_fence(text)

    if not text:
        return False, "empty", None

    try:
        obj = json.loads(text)
    except Exception as exc:
        return (
            False,
            f"json_decode:{type(exc).__name__}",
            None,
        )

    if not isinstance(obj, dict):
        return False, "not_object", None

    if set(obj.keys()) != {"answer_pre", "evidence_pages"}:
        return False, "wrong_keys", obj

    if (
        not isinstance(obj.get("answer_pre"), str)
        or not obj["answer_pre"].strip()
    ):
        return False, "invalid_answer_pre", obj

    pages = obj.get("evidence_pages")

    if not isinstance(pages, list):
        return False, "evidence_not_list", obj

    if not all(
        isinstance(x, int) and not isinstance(x, bool)
        for x in pages
    ):
        return False, "evidence_not_integer_list", obj

    reason = (
        "normalized_markdown_fence"
        if normalized_by_fence
        else "strict_ok"
    )

    return True, reason, obj


def choose_slice_budget(
    msgs: List[Dict[str, Any]],
    system_prompt: str,
    processor: Any,
    llm_context_limit: int,
) -> Tuple[int | None, Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    slice1_ctx: Dict[str, Any] | None = None

    for slice_num in range(MAX_SLICE_NUMS, 0, -1):
        ctx = legacy.inspect_minicpm_context(
            msgs=msgs,
            system_prompt=system_prompt,
            processor=processor,
            max_slice_nums=slice_num,
            max_input_length=HARD_MAX_INPUT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            llm_context_limit=llm_context_limit,
        )

        trial = dict(ctx)
        trial["max_slice_nums"] = slice_num
        trials.append(trial)

        if slice_num == 1:
            slice1_ctx = ctx

        pre = int(ctx["input_tokens_pre_truncation"])
        hard_over = bool(
            ctx.get("hard_context_overflow", False)
        )
        would_truncate = bool(
            ctx.get(
                "would_truncate_at_max_input_length",
                False,
            )
        )

        if (
            pre <= PREFERRED_INPUT_TOKENS
            and not hard_over
            and not would_truncate
        ):
            return slice_num, {
                "selected": dict(ctx),
                "trials": trials,
                "soft_budget_exceeded": False,
                "selection_reason": (
                    "largest_slice_within_preferred_budget"
                ),
            }

    if slice1_ctx is None:
        raise RuntimeError(
            "slice=1 context inspection was not executed"
        )

    pre = int(slice1_ctx["input_tokens_pre_truncation"])
    hard_over = bool(
        slice1_ctx.get("hard_context_overflow", False)
    )
    would_truncate = bool(
        slice1_ctx.get(
            "would_truncate_at_max_input_length",
            False,
        )
    )

    if (
        pre <= HARD_MAX_INPUT_TOKENS
        and not hard_over
        and not would_truncate
    ):
        return 1, {
            "selected": dict(slice1_ctx),
            "trials": trials,
            "soft_budget_exceeded": True,
            "selection_reason": (
                "slice1_above_preferred_but_within_hard_cap"
            ),
        }

    return None, {
        "selected": dict(slice1_ctx),
        "trials": trials,
        "soft_budget_exceeded": True,
        "selection_reason": (
            "context_guard_exceeded_even_at_slice1"
        ),
    }


def is_cuda_oom_error(exc: BaseException) -> bool:
    return legacy.is_cuda_oom_error(exc)
