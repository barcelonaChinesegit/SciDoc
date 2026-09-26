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

# -----------------------------------------------------------------------------
# CUDA must be selected before importing torch/model libraries.
# Formal default for this runner: physical GPU 3.
# The shell launcher may override it with GPU=... / PDFQA_GPU=....
# -----------------------------------------------------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PDFQA_GPU", "3"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pypdfium2 as pdfium
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# models/MiniCPM-V-4_5/run_pdfqa.py -> repository root
SXZ_ROOT = Path(__file__).resolve().parents[2]
if str(SXZ_ROOT) not in sys.path:
    sys.path.insert(0, str(SXZ_ROOT))

from common.pdfqa_dataset import (  # noqa: E402
    build_document_manifest,
    flatten_dataset,
    load_json,
    resolve_pdf_path,
)
from settings.pdfqa_benchmark_config import (  # noqa: E402
    DATASETS,
    MODEL_DEFAULTS,
    PDF_DPI,
    get_dataset_spec,
)
from settings.pdfqa_prompts import (  # noqa: E402
    PROMPT_VERSION,
    get_prompt,
    prompt_sha256,
)

MODEL_NAME = "MiniCPM-V-4_5"
MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = MODEL_DIR / "output"
RAW_DIR = OUTPUT_ROOT / "raw_result"
PARSED_DIR = OUTPUT_ROOT / "parsed_result"
REVIEWED_DIR = OUTPUT_ROOT / "reviewed_result"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"

# run-v3 is intentional: visual slicing is now selected by exact native
# pre-truncation token usage rather than coarse page-count thresholds.
# This prevents mixing run-v2 pagecount-v1 results with the new protocol.
RUN_VERSION = "run-v3"
SLICE_POLICY_ID = "tokenbudget-v1"
SLICE_SEARCH_ID = "native-max-valid-slice-v1"
CONTEXT_INSPECTION_ID = "native-placeholder-pretruncation-v1"
DEFAULT_MAX_INPUT_LENGTH = 39424
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_AUTO_MAX_SLICE_NUMS = 9
EXPECTED_LLM_CONTEXT = 40960
INPUT_SAFETY_MARGIN = 512
IMAGE_TAG = "(<image>./</image>)"
SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|assistant|>",
    "<|user|>",
)


# -----------------------------------------------------------------------------
# Formal MiniCPM visual policy (run-v3).
#
# The runner tests MiniCPM native max_slice_nums values from 9 down to 1 and
# chooses the largest value whose EXACT expanded multimodal prompt fits both:
#   1) max_input_length (39424 by default), and
#   2) llm_context_limit after reserving max_new_tokens.
#
# The selection uses only the actual model input (PDF pages + question + prompt)
# and MiniCPM's native image placeholder expansion. It never uses gold answers,
# gold evidence, model predictions, or correctness. GPU memory does NOT affect
# slice selection: a context-valid sample that OOMs on one GPU is kept as OOM
# and can later be rerun on more GPUs with the exact same selected slice value.
# -----------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    for p in (RAW_DIR, PARSED_DIR, REVIEWED_DIR, LOG_DIR, MANIFEST_DIR):
        p.mkdir(parents=True, exist_ok=True)


def output_basename(dataset_id: str) -> str:
    return f"{dataset_id}__prompt-v1__dpi144__{RUN_VERSION}.json"


def raw_output_path(dataset_id: str) -> Path:
    return RAW_DIR / output_basename(dataset_id)


def manifest_output_path(dataset_id: str) -> Path:
    return MANIFEST_DIR / output_basename(dataset_id).replace(
        ".json", ".manifest.json"
    )


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
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        tmp = Path(f.name)
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_pages(raw: Any) -> List[int]:
    if raw is None or isinstance(raw, bool):
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: List[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            vals = [item]
        else:
            vals = [int(x) for x in re.findall(r"\d+", str(item))]
        for v in vals:
            if v > 0 and v not in out:
                out.append(v)
    return sorted(out)


def strip_special_tokens(text: str) -> str:
    out = str(text or "").strip()
    for token in SPECIAL_TOKENS:
        out = out.replace(token, "")
    return out.strip()


def quick_parse_prediction(raw_text: str) -> Optional[Dict[str, Any]]:
    """Conservative parser used only for status/resume.

    The original model text is always preserved unchanged as answer_pre_raw.
    """
    text = strip_special_tokens(raw_text)
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()

    candidates = [text]
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        candidates.append(text[s : e + 1])

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
        if "answer_pre" not in obj or "evidence_pages" not in obj:
            continue

        return {
            "answer_pre": str(obj.get("answer_pre", "")).strip(),
            "evidence_pages": normalize_pages(obj.get("evidence_pages")),
        }

    return None


def completed_entry(qa_result: Dict[str, Any]) -> bool:
    return (
        qa_result.get("status") == "completed"
        and isinstance(qa_result.get("answer_pre_raw"), str)
        and bool(qa_result.get("answer_pre_raw", "").strip())
    )


def is_cuda_oom_error(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "outofmemoryerror" in msg


def pdf_to_images(pdf_path: Path, dpi: int = PDF_DPI) -> List[Image.Image]:
    if not pdf_path.is_file():
        raise FileNotFoundError(str(pdf_path))

    images: List[Image.Image] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0

    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            bitmap = page.render(scale=scale)
            try:
                images.append(bitmap.to_pil().convert("RGB"))
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
        raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")

    return images


def close_images(images: Sequence[Image.Image]) -> None:
    for image in images:
        try:
            image.close()
        except Exception:
            pass


class SinglePdfImageCache:
    """Cache only the currently used complete PDF."""

    def __init__(self) -> None:
        self.pdf_path: Optional[Path] = None
        self.images: Optional[List[Image.Image]] = None

    def get(self, pdf_path: Path, dpi: int) -> List[Image.Image]:
        if self.pdf_path == pdf_path and self.images is not None:
            return self.images

        self.clear()
        self.images = pdf_to_images(pdf_path, dpi=dpi)
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


def build_minicpm_messages(
    images: Sequence[Image.Image],
    question: str,
    document_manifest: str,
) -> List[Dict[str, Any]]:
    """Build MiniCPM native multi-image messages with physical page labels."""
    content: List[Any] = []

    if document_manifest:
        content.append(document_manifest)

    for page_idx, image in enumerate(images, start=1):
        content.append(f"PDF_PAGE_{page_idx}_START")
        content.append(image)
        content.append(f"PDF_PAGE_{page_idx}_END")

    content.append(f"Question:\n{question}")
    return [{"role": "user", "content": content}]


def build_native_chat_prompt(
    msgs: List[Dict[str, Any]],
    system_prompt: str,
    processor: Any,
) -> Tuple[str, List[Image.Image]]:
    """Mirror MiniCPM-V-4.5 model.chat prompt construction exactly.

    This intentionally avoids deepcopying PIL images.  Token-budget search may
    inspect up to nine slice candidates for one QA; copying all rendered PDF
    pages for every candidate would waste host memory without changing prompt
    semantics.
    """
    prompt_msgs: List[Dict[str, str]] = []
    images: List[Image.Image] = []

    for msg in msgs:
        role = msg["role"]
        content = msg["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported chat role: {role}")
        if isinstance(content, str):
            content = [content]

        cur_msgs: List[str] = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append(IMAGE_TAG)
            elif isinstance(c, str):
                cur_msgs.append(c)
        prompt_msgs.append({"role": role, "content": "\n".join(cur_msgs)})

    if system_prompt:
        prompt_msgs = [{"role": "system", "content": system_prompt}] + prompt_msgs

    prompt = processor.tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt, images


def inspect_minicpm_context(
    msgs: List[Dict[str, Any]],
    system_prompt: str,
    processor: Any,
    max_slice_nums: int,
    max_input_length: int,
    max_new_tokens: int,
    llm_context_limit: int,
) -> Dict[str, Any]:
    """Measure MiniCPM pre-truncation input length without building GPU tensors.

    MiniCPM-V-4.5's processor expands every image to a native image placeholder
    derived from get_sliced_grid()/get_slice_image_placeholder(), tokenizes the
    resulting final_text, and then applies input_ids[:max_length].  This helper
    reproduces that expansion using the loaded processor but intentionally does
    NOT apply the truncation, so the result is the exact pre-truncation token
    length for this prompt/image geometry.
    """
    prompt, prompt_images = build_native_chat_prompt(
        msgs=msgs,
        system_prompt=system_prompt,
        processor=processor,
    )

    text_chunks = prompt.split(IMAGE_TAG)
    if len(text_chunks) != len(prompt_images) + 1:
        raise RuntimeError(
            "MiniCPM prompt/image mismatch during context inspection: "
            f"tags={len(text_chunks)-1}, images={len(prompt_images)}"
        )

    final_text_parts: List[str] = []
    page_visual_units: List[int] = []
    page_slice_patch_counts: List[int] = []
    page_slice_grids: List[Optional[List[int]]] = []

    image_processor = processor.image_processor
    for idx, image in enumerate(prompt_images):
        final_text_parts.append(text_chunks[idx])
        grid = image_processor.get_sliced_grid(image.size, max_slice_nums)
        if grid is None:
            grid_list: Optional[List[int]] = None
            patch_count = 0
        else:
            grid_list = [int(grid[0]), int(grid[1])]
            patch_count = int(grid[0]) * int(grid[1])

        # MiniCPM always includes one source-image unit plus optional grid patches.
        visual_units = 1 + patch_count
        page_visual_units.append(visual_units)
        page_slice_patch_counts.append(patch_count)
        page_slice_grids.append(grid_list)

        final_text_parts.append(
            image_processor.get_slice_image_placeholder(
                image.size,
                idx,
                max_slice_nums,
                False,  # use_image_id=False in the formal protocol
            )
        )

    final_text_parts.append(text_chunks[-1])
    final_text = "".join(final_text_parts)

    input_ids = processor.tokenizer.encode(final_text)
    version = getattr(processor, "version", 999.0)
    if not (version > 2.5 or not getattr(processor.tokenizer, "add_bos_token", False)):
        input_ids = [processor.tokenizer.bos_id] + input_ids

    pre_tokens = int(len(input_ids))
    actual_visual_units = int(sum(page_visual_units))
    image_feature_size = int(
        getattr(image_processor, "image_feature_size", 0) or 0
    )
    visual_feature_tokens_est = (
        actual_visual_units * image_feature_size
        if image_feature_size > 0
        else None
    )

    would_truncate = pre_tokens > max_input_length
    hard_context_overflow = (pre_tokens + max_new_tokens) > llm_context_limit

    return {
        "context_inspection_id": CONTEXT_INSPECTION_ID,
        "input_tokens_pre_truncation": pre_tokens,
        "max_input_length": int(max_input_length),
        "would_truncate_at_max_input_length": bool(would_truncate),
        "overflow_tokens_at_max_input_length": max(0, pre_tokens - max_input_length),
        "input_budget_usage_pct": round(100.0 * pre_tokens / max_input_length, 2),
        "llm_context_limit": int(llm_context_limit),
        "max_new_tokens": int(max_new_tokens),
        "total_context_if_max_generation": int(pre_tokens + max_new_tokens),
        "total_context_usage_pct": round(
            100.0 * (pre_tokens + max_new_tokens) / llm_context_limit, 2
        ),
        "hard_context_overflow": bool(hard_context_overflow),
        "input_safety_margin_tokens": int(
            llm_context_limit - max_input_length - max_new_tokens
        ),
        "actual_visual_units": actual_visual_units,
        "page_visual_units": page_visual_units,
        "page_slice_patch_counts": page_slice_patch_counts,
        "page_slice_grids": page_slice_grids,
        "image_feature_size": image_feature_size or None,
        "visual_feature_tokens_est": visual_feature_tokens_est,
    }


def select_max_slice_nums_by_context(
    msgs: List[Dict[str, Any]],
    system_prompt: str,
    processor: Any,
    max_input_length: int,
    max_new_tokens: int,
    llm_context_limit: int,
    auto_max_slice_nums: int,
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]], bool]:
    """Select the largest native max_slice_nums that fits the context budget.

    Search is deterministic and descending.  The first candidate that does not
    trigger either MiniCPM's real input truncation boundary or the LLM context
    boundary is selected.  If even max_slice_nums=1 does not fit, 1 is returned
    together with fits_context=False so the caller can record context overflow
    without invoking model.chat().
    """
    if auto_max_slice_nums < 1:
        raise ValueError(f"auto_max_slice_nums must be >=1, got {auto_max_slice_nums}")

    trace: List[Dict[str, Any]] = []
    selected_info: Optional[Dict[str, Any]] = None
    selected_slice = 1
    fits_context = False

    for candidate in range(auto_max_slice_nums, 0, -1):
        info = inspect_minicpm_context(
            msgs=msgs,
            system_prompt=system_prompt,
            processor=processor,
            max_slice_nums=candidate,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            llm_context_limit=llm_context_limit,
        )
        valid = (
            not info["would_truncate_at_max_input_length"]
            and not info["hard_context_overflow"]
        )
        trace.append(
            {
                "max_slice_nums": candidate,
                "actual_visual_units": info["actual_visual_units"],
                "input_tokens_pre_truncation": info["input_tokens_pre_truncation"],
                "input_budget_usage_pct": info["input_budget_usage_pct"],
                "total_context_if_max_generation": info["total_context_if_max_generation"],
                "total_context_usage_pct": info["total_context_usage_pct"],
                "fits_context": bool(valid),
            }
        )

        selected_slice = candidate
        selected_info = info
        if valid:
            fits_context = True
            break

    if selected_info is None:
        raise RuntimeError("Slice search produced no candidate")

    selected_info = dict(selected_info)
    selected_info.update(
        {
            "slice_search_id": SLICE_SEARCH_ID,
            "slice_search_auto_max": int(auto_max_slice_nums),
            "slice_search_selected": int(selected_slice),
            "slice_search_selected_fits_context": bool(fits_context),
            "slice_search_candidates_tested": len(trace),
            "slice_search_trace": trace,
        }
    )
    return selected_slice, selected_info, trace, fits_context


@torch.inference_mode()
def infer_one(
    images: Sequence[Image.Image],
    question: str,
    system_prompt: str,
    document_manifest: str,
    model: Any,
    tokenizer: Any,
    processor: Any,
    max_new_tokens: int,
    max_input_length: int,
    max_slice_nums: int,
) -> Tuple[str, float]:
    msgs = build_minicpm_messages(
        images=images,
        question=question,
        document_manifest=document_manifest,
    )

    start = time.time()
    answer = model.chat(
        msgs=msgs,
        tokenizer=tokenizer,
        processor=processor,
        system_prompt=system_prompt,
        enable_thinking=False,
        stream=False,
        sampling=False,
        max_new_tokens=max_new_tokens,
        max_inp_length=max_input_length,
        max_slice_nums=max_slice_nums,
        use_image_id=False,
    )
    elapsed = time.time() - start

    return ("" if answer is None else str(answer)).strip(), elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "MiniCPM-V-4.5 whole-PDF runner with exact native pre-truncation "
            "context diagnostics and no silent truncation."
        )
    )
    p.add_argument("--dataset-id", required=True, choices=list(DATASETS))
    p.add_argument(
        "--model-path",
        default=MODEL_DEFAULTS[MODEL_NAME]["model_path"],
    )
    p.add_argument("--pdf-dpi", type=int, default=PDF_DPI)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-input-length", type=int, default=DEFAULT_MAX_INPUT_LENGTH)
    p.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2"],
        default=MODEL_DEFAULTS[MODEL_NAME].get("attn_implementation", "sdpa"),
    )
    p.add_argument(
        "--slice-policy",
        choices=[SLICE_POLICY_ID, "fixed"],
        default=SLICE_POLICY_ID,
    )
    p.add_argument(
        "--auto-max-slice-nums",
        type=int,
        default=DEFAULT_AUTO_MAX_SLICE_NUMS,
        help=(
            "Maximum native max_slice_nums considered by tokenbudget-v1. "
            "Formal default is 9 (MiniCPM-V-4.5 native maximum)."
        ),
    )
    p.add_argument(
        "--max-slice-nums",
        type=int,
        default=1,
        help="Used only with --slice-policy fixed.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0 = all remaining")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate_dataset(
    dataset_id: str,
    data: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> None:
    spec = get_dataset_spec(dataset_id)
    expected = int(spec["expected_qa_count"])
    actual = len(samples)

    if actual != expected:
        raise RuntimeError(f"{dataset_id}: expected {expected} QA, found {actual}")

    seen_uid = set()
    resolved: Dict[str, Path] = {}

    for sample in samples:
        uid = sample["item_uid"]
        if uid in seen_uid:
            raise RuntimeError(f"Duplicate item_uid: {uid}")
        seen_uid.add(uid)

        unit_id = sample["unit_id"]
        if unit_id not in resolved:
            resolved[unit_id] = resolve_pdf_path(
                dataset_id,
                unit_id,
                sample["unit"],
            )

    print(f"[Preflight] dataset={dataset_id} | QA={actual} | PDFs={len(resolved)}", flush=True)
    for unit_id, path in list(resolved.items())[:5]:
        print(f"  {unit_id}: {path}", flush=True)
    if len(resolved) > 5:
        print(f"  ... {len(resolved)-5} more PDFs", flush=True)


def get_llm_context_limit(model: Any) -> int:
    cfg = model.config
    candidates = [
        getattr(cfg, "max_position_embeddings", None),
        getattr(getattr(cfg, "llm_config", None), "max_position_embeddings", None),
        getattr(getattr(cfg, "language_config", None), "max_position_embeddings", None),
        getattr(getattr(cfg, "text_config", None), "max_position_embeddings", None),
    ]
    vals = [int(x) for x in candidates if isinstance(x, int) and x > 0]
    if not vals:
        raise RuntimeError("Cannot determine MiniCPM LLM max_position_embeddings")
    return vals[0]


def base_result_metadata(
    sample: Dict[str, Any],
    pdf_path: Path,
    page_count: int,
    args: argparse.Namespace,
    prompt_id: str,
    max_slice_nums: int,
    page_slice_cap_proxy: int,
    context_info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "item_uid": sample["item_uid"],
        "model_name": MODEL_NAME,
        "pdf_path_resolved": str(pdf_path),
        "pdf_total_pages": page_count,
        "pdf_pages_supplied": list(range(1, page_count + 1)),
        "pdf_pages_supplied_count": page_count,
        "render_dpi": args.pdf_dpi,
        "prompt_id": prompt_id,
        "prompt_version": PROMPT_VERSION,
        "gold_answer_sent": False,
        "gold_evidence_pages_sent": False,
        "slice_policy": args.slice_policy,
        "max_slice_nums_used": max_slice_nums,
        "page_slice_cap_proxy": page_slice_cap_proxy,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
        "stream": False,
        "sampling": False,
        "use_image_id": False,
        "attn_implementation": args.attn_implementation,
        **context_info,
    }


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if args.pdf_dpi != 144:
        raise ValueError(
            "Formal protocol requires --pdf-dpi 144. Create a new protocol "
            "version for any other DPI."
        )
    if args.slice_policy == "fixed" and args.max_slice_nums < 1:
        raise ValueError("--max-slice-nums must be >= 1")
    if args.auto_max_slice_nums < 1:
        raise ValueError("--auto-max-slice-nums must be >= 1")

    spec = get_dataset_spec(args.dataset_id)
    data_path = Path(spec["json_path"])
    data = load_json(data_path)
    samples = flatten_dataset(args.dataset_id, data)
    validate_dataset(args.dataset_id, data, samples)

    prompt_id, system_prompt = get_prompt(spec["prompt_key"])
    save_path = raw_output_path(args.dataset_id)
    manifest_path = manifest_output_path(args.dataset_id)

    if args.dry_run:
        print(f"[DryRun] data={data_path}", flush=True)
        print(f"[DryRun] pdf_root={spec['pdf_root']}", flush=True)
        print(f"[DryRun] output={save_path}", flush=True)
        print(f"[DryRun] prompt_id={prompt_id}", flush=True)
        print(f"[DryRun] prompt_sha256={prompt_sha256(system_prompt)}", flush=True)
        print(f"[DryRun] slice_policy={args.slice_policy}", flush=True)
        print(f"[DryRun] auto_max_slice_nums={args.auto_max_slice_nums}", flush=True)
        print(f"[DryRun] max_input_length={args.max_input_length}", flush=True)
        print(f"[DryRun] max_new_tokens={args.max_new_tokens}", flush=True)
        print(f"[DryRun] expected_llm_context={EXPECTED_LLM_CONTEXT}", flush=True)
        print(f"[DryRun] run_version={RUN_VERSION}", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is disabled.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU for this first-pass runner. "
            f"Detected {torch.cuda.device_count()}. OOM samples can be rerun "
            "later with a dedicated multi-GPU runner."
        )

    print("=" * 110, flush=True)
    print(f"Model              : {MODEL_NAME}", flush=True)
    print(f"Model path         : {args.model_path}", flush=True)
    print(f"Dataset            : {args.dataset_id}", flush=True)
    print(f"Dataset path       : {data_path}", flush=True)
    print(f"PDF root           : {spec['pdf_root']}", flush=True)
    print(f"PDF DPI            : {args.pdf_dpi}", flush=True)
    print(f"Prompt ID          : {prompt_id}", flush=True)
    print(f"Prompt version     : {PROMPT_VERSION}", flush=True)
    print(f"Run version        : {RUN_VERSION}", flush=True)
    print(f"Raw result         : {save_path}", flush=True)
    print(f"Visible GPU(s)     : {os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"Logical CUDA       : cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)
    print("Page dropping      : NEVER", flush=True)
    print("Gold answer sent   : NEVER", flush=True)
    print("Gold evidence sent : NEVER", flush=True)
    print("Thinking           : OFF", flush=True)
    print("Streaming          : OFF", flush=True)
    print("Sampling           : OFF", flush=True)
    print("use_image_id       : False", flush=True)
    print(f"Slice policy       : {args.slice_policy}", flush=True)
    print(f"Auto max slice     : {args.auto_max_slice_nums}", flush=True)
    print(f"Max input length   : {args.max_input_length}", flush=True)
    print(f"Max new tokens     : {args.max_new_tokens}", flush=True)
    print(f"Attention          : {args.attn_implementation}", flush=True)
    print("Silent truncation  : FORBIDDEN", flush=True)
    print("=" * 110, flush=True)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"MiniCPM model path does not exist: {args.model_path}")

    print("[Load] Loading MiniCPM-V-4.5 ...", flush=True)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.processor = processor

    native_max_slice_nums = int(
        getattr(processor.image_processor, "max_slice_nums", DEFAULT_AUTO_MAX_SLICE_NUMS)
    )
    if args.auto_max_slice_nums > native_max_slice_nums:
        raise ValueError(
            f"--auto-max-slice-nums={args.auto_max_slice_nums} exceeds MiniCPM native "
            f"processor max_slice_nums={native_max_slice_nums}."
        )
    if args.slice_policy == "fixed" and args.max_slice_nums > native_max_slice_nums:
        raise ValueError(
            f"--max-slice-nums={args.max_slice_nums} exceeds MiniCPM native "
            f"processor max_slice_nums={native_max_slice_nums}."
        )

    llm_context_limit = get_llm_context_limit(model)
    tokenizer_model_max_length = getattr(tokenizer, "model_max_length", None)
    safety_margin = llm_context_limit - args.max_input_length - args.max_new_tokens

    if safety_margin < 0:
        raise ValueError(
            "Invalid context configuration: max_input_length + max_new_tokens "
            f"= {args.max_input_length + args.max_new_tokens} exceeds "
            f"LLM context limit {llm_context_limit}."
        )

    print("[Load] Model, tokenizer and processor loaded.", flush=True)
    print(f"[ContextConfig] tokenizer.model_max_length={tokenizer_model_max_length}", flush=True)
    print(f"[ContextConfig] llm.max_position_embeddings={llm_context_limit}", flush=True)
    print(f"[ContextConfig] max_input_length={args.max_input_length}", flush=True)
    print(f"[ContextConfig] max_new_tokens={args.max_new_tokens}", flush=True)
    print(f"[ContextConfig] safety_margin={safety_margin}", flush=True)
    print(f"[SliceConfig] native_max_slice_nums={native_max_slice_nums}", flush=True)
    print(f"[SliceConfig] auto_max_slice_nums={args.auto_max_slice_nums}", flush=True)

    if save_path.exists() and not args.overwrite:
        results = load_json(save_path)
    else:
        results = copy.deepcopy(data)

    for unit_id, unit in data.items():
        if unit_id not in results:
            results[unit_id] = copy.deepcopy(unit)
        elif isinstance(unit, dict) and isinstance(unit.get("QA"), dict):
            results[unit_id].setdefault("QA", {})
            for qa_id, qa in unit["QA"].items():
                results[unit_id]["QA"].setdefault(qa_id, copy.deepcopy(qa))

    manifest = {
        "model_name": MODEL_NAME,
        "model_path": args.model_path,
        "dataset_id": args.dataset_id,
        "dataset_path": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "expected_qa_count": spec["expected_qa_count"],
        "prompt_id": prompt_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(system_prompt),
        "run_version": RUN_VERSION,
        "pdf_dpi": args.pdf_dpi,
        "page_numbering": "physical_pdf_page_1_based",
        "page_dropping": False,
        "slice_policy": args.slice_policy,
        "slice_policy_id": SLICE_POLICY_ID,
        "slice_search_id": SLICE_SEARCH_ID,
        "slice_policy_rule": (
            "Test native max_slice_nums from auto_max down to 1; select the largest "
            "candidate whose exact pre-truncation input fits max_input_length and whose "
            "input+max_new_tokens fits llm_context_limit."
        ),
        "auto_max_slice_nums": args.auto_max_slice_nums,
        "native_max_slice_nums": native_max_slice_nums,
        "fixed_max_slice_nums": args.max_slice_nums if args.slice_policy == "fixed" else None,
        "context_inspection_id": CONTEXT_INSPECTION_ID,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "llm_context_limit": llm_context_limit,
        "tokenizer_model_max_length": tokenizer_model_max_length,
        "input_safety_margin_tokens": safety_margin,
        "silent_input_truncation_allowed": False,
        "enable_thinking": False,
        "stream": False,
        "sampling": False,
        "use_image_id": False,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": "bfloat16",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name_logical_cuda0": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "started_at_unix": time.time(),
    }
    atomic_write_json(manifest_path, manifest)

    cache = SinglePdfImageCache()
    attempted_this_run = 0
    completed_before = 0

    try:
        for sample in samples:
            unit_id = sample["unit_id"]
            qa_id = sample["qa_id"]
            qa_result = results[unit_id]["QA"][qa_id]

            if not args.overwrite and completed_entry(qa_result):
                completed_before += 1
                continue

            if args.max_samples > 0 and attempted_this_run >= args.max_samples:
                break

            attempted_this_run += 1
            pdf_path = resolve_pdf_path(args.dataset_id, unit_id, sample["unit"])
            document_manifest = build_document_manifest(args.dataset_id, sample["unit"])

            try:
                images = cache.get(pdf_path, dpi=args.pdf_dpi)
                page_count = len(images)

                msgs = build_minicpm_messages(
                    images=images,
                    question=str(sample["qa"].get("question", "")),
                    document_manifest=document_manifest,
                )

                if args.slice_policy == SLICE_POLICY_ID:
                    max_slice_nums, context_info, slice_trace, slice_fits = (
                        select_max_slice_nums_by_context(
                            msgs=msgs,
                            system_prompt=system_prompt,
                            processor=processor,
                            max_input_length=args.max_input_length,
                            max_new_tokens=args.max_new_tokens,
                            llm_context_limit=llm_context_limit,
                            auto_max_slice_nums=args.auto_max_slice_nums,
                        )
                    )
                else:
                    max_slice_nums = args.max_slice_nums
                    context_info = inspect_minicpm_context(
                        msgs=msgs,
                        system_prompt=system_prompt,
                        processor=processor,
                        max_slice_nums=max_slice_nums,
                        max_input_length=args.max_input_length,
                        max_new_tokens=args.max_new_tokens,
                        llm_context_limit=llm_context_limit,
                    )
                    slice_fits = (
                        not context_info["would_truncate_at_max_input_length"]
                        and not context_info["hard_context_overflow"]
                    )
                    slice_trace = [
                        {
                            "max_slice_nums": max_slice_nums,
                            "actual_visual_units": context_info["actual_visual_units"],
                            "input_tokens_pre_truncation": context_info["input_tokens_pre_truncation"],
                            "input_budget_usage_pct": context_info["input_budget_usage_pct"],
                            "total_context_if_max_generation": context_info["total_context_if_max_generation"],
                            "total_context_usage_pct": context_info["total_context_usage_pct"],
                            "fits_context": bool(slice_fits),
                        }
                    ]
                    context_info = dict(context_info)
                    context_info.update(
                        {
                            "slice_search_id": "fixed",
                            "slice_search_auto_max": None,
                            "slice_search_selected": int(max_slice_nums),
                            "slice_search_selected_fits_context": bool(slice_fits),
                            "slice_search_candidates_tested": 1,
                            "slice_search_trace": slice_trace,
                        }
                    )

                # Legacy theoretical proxy retained for comparison only.
                page_slice_cap_proxy = page_count * max_slice_nums

                for step in slice_trace:
                    verdict = "FIT" if step["fits_context"] else "OVERFLOW"
                    print(
                        f"[SliceSearch] max_slice_nums={step['max_slice_nums']} | "
                        f"visual_units={step['actual_visual_units']} | "
                        f"pre_tokens={step['input_tokens_pre_truncation']}/"
                        f"{args.max_input_length} | total_ctx="
                        f"{step['total_context_if_max_generation']}/{llm_context_limit} | "
                        f"{verdict}",
                        flush=True,
                    )

                print(
                    f"[{sample['flat_index']}/{len(samples)}] {sample['item_uid']} | "
                    f"pages={page_count} | max_slice_nums={max_slice_nums} | "
                    f"actual_visual_units={context_info['actual_visual_units']} | "
                    f"pre_tokens={context_info['input_tokens_pre_truncation']}/"
                    f"{args.max_input_length} | "
                    f"total_ctx={context_info['total_context_if_max_generation']}/"
                    f"{llm_context_limit} | pdf={pdf_path}",
                    flush=True,
                )
                print(
                    f"[Slices] page_visual_units={context_info['page_visual_units']}",
                    flush=True,
                )
                print(
                    f"[Context] input_usage={context_info['input_budget_usage_pct']:.2f}% | "
                    f"total_usage={context_info['total_context_usage_pct']:.2f}% | "
                    f"would_truncate={context_info['would_truncate_at_max_input_length']} | "
                    f"hard_overflow={context_info['hard_context_overflow']}",
                    flush=True,
                )

                metadata = base_result_metadata(
                    sample=sample,
                    pdf_path=pdf_path,
                    page_count=page_count,
                    args=args,
                    prompt_id=prompt_id,
                    max_slice_nums=max_slice_nums,
                    page_slice_cap_proxy=page_slice_cap_proxy,
                    context_info=context_info,
                )
                metadata["document_manifest_sent"] = document_manifest

                if not slice_fits:
                    qa_result.update(
                        {
                            **metadata,
                            "status": "context_guard_exceeded",
                            "answer_pre_raw": "",
                            "error_type": "WouldTruncateInput",
                            "error_message": (
                                "No MiniCPM native max_slice_nums candidate in the formal search range "
                                f"1..{args.auto_max_slice_nums} fits the context budget. At the lowest "
                                f"tested slice value, pre-truncation input length is "
                                f"{context_info['input_tokens_pre_truncation']} with "
                                f"max_input_length={args.max_input_length}; inference skipped to prevent "
                                "silent input_ids[:max_length] truncation or LLM context overflow."
                            ),
                        }
                    )
                    print(
                        f"[ContextGuard] SKIP {sample['item_uid']} | "
                        f"selected_min_slice={max_slice_nums} | "
                        f"pre_tokens={context_info['input_tokens_pre_truncation']} | "
                        f"max_input_length={args.max_input_length} | "
                        f"total_ctx={context_info['total_context_if_max_generation']}/"
                        f"{llm_context_limit}",
                        flush=True,
                    )
                    atomic_write_json(save_path, results)
                    continue

                raw_text, elapsed = infer_one(
                    images=images,
                    question=str(sample["qa"].get("question", "")),
                    system_prompt=system_prompt,
                    document_manifest=document_manifest,
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                    max_input_length=args.max_input_length,
                    max_slice_nums=max_slice_nums,
                )

                parsed = quick_parse_prediction(raw_text)
                status = "completed" if parsed is not None else "parse_failed"

                qa_result.update(
                    {
                        **metadata,
                        "status": status,
                        "answer_pre_raw": raw_text,
                        "elapsed_seconds": round(elapsed, 4),
                    }
                )

                print(f"[Done] status={status} elapsed={elapsed:.2f}s", flush=True)
                print(f"[Raw] {raw_text[:2000]}", flush=True)

            except Exception as exc:
                status = "oom_single_gpu" if is_cuda_oom_error(exc) else "error"
                qa_result.update(
                    {
                        "item_uid": sample["item_uid"],
                        "model_name": MODEL_NAME,
                        "status": status,
                        "answer_pre_raw": "",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "error_traceback": traceback.format_exc(),
                        "pdf_path_resolved": str(pdf_path),
                        "render_dpi": args.pdf_dpi,
                        "prompt_id": prompt_id,
                        "prompt_version": PROMPT_VERSION,
                        "gold_answer_sent": False,
                        "gold_evidence_pages_sent": False,
                    }
                )

                prefix = "[OOM]" if status == "oom_single_gpu" else "[Error]"
                print(
                    f"{prefix} {sample['item_uid']} | {type(exc).__name__}: {exc}",
                    flush=True,
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            atomic_write_json(save_path, results)

    finally:
        cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status_counts: Dict[str, int] = {}
    for sample in samples:
        qa = results.get(sample["unit_id"], {}).get("QA", {}).get(sample["qa_id"], {})
        status = str(qa.get("status", "not_run"))
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest["completed_at_unix"] = time.time()
    manifest["status_counts"] = status_counts
    manifest["attempted_this_run"] = attempted_this_run
    manifest["completed_skipped_at_start"] = completed_before
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(save_path, results)

    print("=" * 110, flush=True)
    print(f"[Summary] {args.dataset_id} | {status_counts}", flush=True)
    print(f"[Saved] {save_path}", flush=True)
    print(f"[Manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
