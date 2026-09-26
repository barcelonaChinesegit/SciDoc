#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Physical GPU 2 is the default for this package.
# CUDA must be selected before importing torch.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("PDFQA_GPU", "2"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pypdfium2 as pdfium
import torch
import torchvision.transforms as T
import transformers
from packaging.version import Version
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

MODEL_NAME = "InternVL3_5-8B"
MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = MODEL_DIR / "output"
RAW_DIR = OUTPUT_ROOT / "raw_result"
PARSED_DIR = OUTPUT_ROOT / "parsed_result"
REVIEWED_DIR = OUTPUT_ROOT / "reviewed_result"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
RUN_VERSION = "run-v1"
TILE_POLICY_ID = "docbudget96-a40safe-v2"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|assistant|>",
    "<|user|>",
)


def ensure_output_dirs() -> None:
    for p in (RAW_DIR, PARSED_DIR, REVIEWED_DIR, LOG_DIR, MANIFEST_DIR):
        p.mkdir(parents=True, exist_ok=True)


def output_basename(dataset_id: str) -> str:
    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi{PDF_DPI}"
        f"__{RUN_VERSION}.json"
    )


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
    if "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    return out.strip()


def quick_parse_prediction(raw_text: str) -> Optional[Dict[str, Any]]:
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
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return (
        "cuda out of memory" in msg
        or "outofmemoryerror" in msg
        or "cublas_status_alloc_failed" in msg
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_environment() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is disabled.")

    current = Version(transformers.__version__)
    required = Version("4.52.1")
    if current < required:
        raise RuntimeError(
            f"InternVL3.5-8B requires transformers>={required}, "
            f"but current version is {current}. Use the dedicated InternVL env."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expected exactly one visible GPU. "
            f"Detected {torch.cuda.device_count()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )


def build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: Sequence[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            threshold = 0.5 * image_size * image_size * ratio[0] * ratio[1]
            if area > threshold:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> List[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    sorted_ratios = sorted(target_ratios, key=lambda r: r[0] * r[1])
    target_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        sorted_ratios,
        width,
        height,
        image_size,
    )

    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resized = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BICUBIC,
    )

    processed: List[Image.Image] = []
    tiles_per_row = target_width // image_size
    for idx in range(blocks):
        box = (
            (idx % tiles_per_row) * image_size,
            (idx // tiles_per_row) * image_size,
            ((idx % tiles_per_row) + 1) * image_size,
            ((idx // tiles_per_row) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    resized.close()

    if use_thumbnail and len(processed) != 1:
        processed.append(
            image.resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )
        )

    return processed


def get_pdf_page_count(pdf_path: Path) -> int:
    if not pdf_path.is_file():
        raise FileNotFoundError(str(pdf_path))
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        count = int(len(pdf))
    finally:
        pdf.close()
    if count <= 0:
        raise RuntimeError(f"Empty PDF: {pdf_path}")
    return count


def select_max_num_per_page(
    page_count: int,
    tile_budget: int,
    native_max: int = 12,
) -> int:
    """Question-independent document-wide tile policy.

    The first cap is a nominal document budget (default 96 tiles):
        budget_cap = floor(tile_budget / page_count)

    A second page-count-only A40 safety cap is applied because historical
    whole-PDF tests showed that max_num=4 around 15 pages could already OOM.
    We therefore do NOT use a pure B=96 rule, which would assign 6 tiles/page
    at 15 pages.

    Safety cap:
      <= 4 pages  -> at most 12 tiles/page
       5-8 pages  -> at most  8 tiles/page
       9-14 pages -> at most  4 tiles/page
      15-24 pages -> at most  3 tiles/page
      25-48 pages -> at most  2 tiles/page
       >48 pages  -> exactly  1 tile/page at most

    Examples with tile_budget=96:
       4 pages -> 12
       8 pages -> 8
      12 pages -> 4
      15 pages -> 3
      24 pages -> 3
      45 pages -> 2
      60 pages -> 1

    The policy depends ONLY on total physical PDF page count. It never depends
    on the question, gold answer/evidence, model output, or correctness.
    """
    if page_count <= 0:
        raise ValueError(page_count)

    budget_cap = min(native_max, max(1, tile_budget // page_count))

    if page_count <= 4:
        a40_cap = 12
    elif page_count <= 8:
        a40_cap = 8
    elif page_count <= 14:
        a40_cap = 4
    elif page_count <= 24:
        a40_cap = 3
    elif page_count <= 48:
        a40_cap = 2
    else:
        a40_cap = 1

    return min(native_max, budget_cap, a40_cap)


def pdf_to_pixel_values(
    pdf_path: Path,
    dpi: int,
    input_size: int,
    max_num_per_page: int,
    use_thumbnail: bool,
) -> Tuple[torch.Tensor, List[int], int]:
    transform = build_transform(input_size)
    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []

    pdf = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    try:
        total_pages = len(pdf)
        for page_index in range(total_pages):
            page = pdf[page_index]
            bitmap = page.render(scale=scale)
            page_image: Optional[Image.Image] = None
            tiles: List[Image.Image] = []
            try:
                page_image = bitmap.to_pil().convert("RGB")
                tiles = dynamic_preprocess(
                    image=page_image,
                    min_num=1,
                    max_num=max_num_per_page,
                    image_size=input_size,
                    use_thumbnail=use_thumbnail,
                )
                page_tensor = torch.stack([transform(tile) for tile in tiles])
                pixel_values_list.append(page_tensor)
                num_patches_list.append(int(page_tensor.size(0)))
            finally:
                for tile in tiles:
                    try:
                        tile.close()
                    except Exception:
                        pass
                if page_image is not None:
                    try:
                        page_image.close()
                    except Exception:
                        pass
                try:
                    bitmap.close()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass
    finally:
        pdf.close()

    if not pixel_values_list:
        raise RuntimeError(f"No pages rendered from PDF: {pdf_path}")

    pixel_values = torch.cat(pixel_values_list, dim=0)
    if len(num_patches_list) != total_pages:
        raise RuntimeError(
            f"Page/tile mapping mismatch: pages={total_pages}, "
            f"num_patches_list={len(num_patches_list)}"
        )
    if int(pixel_values.shape[0]) != int(sum(num_patches_list)):
        raise RuntimeError(
            "Tile tensor count does not match sum(num_patches_list)."
        )

    return pixel_values, num_patches_list, total_pages


class SinglePdfTensorCache:
    """Cache one PDF's CPU-preprocessed tiles and optional GPU tensor."""

    def __init__(self) -> None:
        self.pdf_path: Optional[Path] = None
        self.policy_key: Optional[Tuple[int, int, int, bool]] = None
        self.pixel_values_cpu: Optional[torch.Tensor] = None
        self.pixel_values_device: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None
        self.num_patches_list: Optional[List[int]] = None
        self.page_count = 0

    def get_cpu(
        self,
        pdf_path: Path,
        dpi: int,
        input_size: int,
        max_num_per_page: int,
        use_thumbnail: bool,
    ) -> Tuple[torch.Tensor, List[int], int]:
        policy_key = (dpi, input_size, max_num_per_page, use_thumbnail)
        if (
            self.pdf_path == pdf_path
            and self.policy_key == policy_key
            and self.pixel_values_cpu is not None
            and self.num_patches_list is not None
        ):
            return self.pixel_values_cpu, self.num_patches_list, self.page_count

        self.clear()
        pixel_values, num_patches_list, page_count = pdf_to_pixel_values(
            pdf_path=pdf_path,
            dpi=dpi,
            input_size=input_size,
            max_num_per_page=max_num_per_page,
            use_thumbnail=use_thumbnail,
        )
        self.pdf_path = pdf_path
        self.policy_key = policy_key
        self.pixel_values_cpu = pixel_values
        self.num_patches_list = num_patches_list
        self.page_count = page_count
        return pixel_values, num_patches_list, page_count

    def get_device(self, device: torch.device) -> torch.Tensor:
        if self.pixel_values_cpu is None:
            raise RuntimeError("CPU pixel tensor is not prepared.")
        if self.pixel_values_device is None or self.device != device:
            self.pixel_values_device = self.pixel_values_cpu.to(
                device=device,
                dtype=torch.bfloat16,
                non_blocking=True,
            )
            self.device = device
        return self.pixel_values_device

    def clear_device(self) -> None:
        self.pixel_values_device = None
        self.device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear(self) -> None:
        self.pixel_values_device = None
        self.pixel_values_cpu = None
        self.device = None
        self.num_patches_list = None
        self.pdf_path = None
        self.policy_key = None
        self.page_count = 0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.clear()


def get_module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Module has no parameters.") from exc


def get_vision_device(model: Any) -> torch.device:
    vision_model = getattr(model, "vision_model", None)
    if vision_model is None:
        raise AttributeError("InternVL model has no vision_model attribute.")
    return get_module_device(vision_model)


def get_context_limit(model: Any, fallback: int = 40960) -> int:
    llm_cfg = getattr(getattr(model, "config", None), "llm_config", None)
    value = getattr(llm_cfg, "max_position_embeddings", None)
    try:
        ivalue = int(value)
        if ivalue > 0:
            return ivalue
    except Exception:
        pass
    return fallback


def get_num_image_token(model: Any, fallback: int = 256) -> int:
    value = getattr(model, "num_image_token", None)
    try:
        ivalue = int(value)
        if ivalue > 0:
            return ivalue
    except Exception:
        pass
    return fallback


def build_question(
    page_count: int,
    question: str,
    document_manifest: str,
) -> str:
    blocks: List[str] = []
    if document_manifest:
        blocks.append(document_manifest.strip())

    for page in range(1, page_count + 1):
        blocks.append(f"PDF_PAGE_{page}_START\n<image>\nPDF_PAGE_{page}_END")

    blocks.append(f"Question:\n{question}")
    return "\n\n".join(blocks) + "\n\nReturn only the required JSON object."


def estimate_context_tokens(
    tokenizer: Any,
    model_question: str,
    system_prompt: str,
    total_tiles: int,
    num_image_token: int,
    max_new_tokens: int,
    safety_margin: int,
) -> Dict[str, int]:
    # The text-side estimate intentionally excludes the repeated IMG_CONTEXT
    # expansion; those visual tokens are counted separately below.
    text_tokens = len(
        tokenizer(model_question, add_special_tokens=True)["input_ids"]
    )
    system_tokens = len(
        tokenizer(system_prompt, add_special_tokens=False)["input_ids"]
    )
    visual_tokens = int(total_tiles * num_image_token)
    estimated_input = text_tokens + system_tokens + visual_tokens
    estimated_total = estimated_input + max_new_tokens + safety_margin
    return {
        "estimated_text_tokens": int(text_tokens + system_tokens),
        "estimated_visual_tokens": visual_tokens,
        "estimated_input_tokens": estimated_input,
        "estimated_total_with_output_margin": estimated_total,
    }


@torch.inference_mode()
def infer_one(
    pixel_values: torch.Tensor,
    num_patches_list: List[int],
    model_question: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> Tuple[str, float]:
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }
    start = time.time()
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=model_question,
        generation_config=generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )
    return strip_special_tokens(str(response)), time.time() - start


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="InternVL3.5-8B whole-PDF runner for the four PDF-QA datasets."
    )
    p.add_argument("--dataset-id", required=True, choices=list(DATASETS))
    p.add_argument("--model-path", default=MODEL_DEFAULTS[MODEL_NAME]["model_path"])
    p.add_argument("--pdf-dpi", type=int, default=PDF_DPI)
    p.add_argument("--input-size", type=int, default=MODEL_DEFAULTS[MODEL_NAME]["input_size"])
    p.add_argument("--tile-budget", type=int, default=MODEL_DEFAULTS[MODEL_NAME]["tile_budget"])
    p.add_argument("--native-max-num", type=int, default=12)
    p.add_argument("--use-thumbnail", action="store_true", default=False)
    p.add_argument("--max-new-tokens", type=int, default=MODEL_DEFAULTS[MODEL_NAME]["max_new_tokens"])
    p.add_argument("--context-safety-margin", type=int, default=MODEL_DEFAULTS[MODEL_NAME]["context_safety_margin"])
    p.add_argument(
        "--device-map",
        choices=["balanced", "auto", "balanced_low_0", "sequential"],
        default=MODEL_DEFAULTS[MODEL_NAME]["device_map"],
    )
    p.add_argument("--use-flash-attn", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all remaining")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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
                f"{dataset_id}: expected {expected} QA, found {actual}"
            )

    seen = set()
    resolved: Dict[str, Path] = {}

    for sample in samples:
        uid = sample["item_uid"]

        if uid in seen:
            raise RuntimeError(f"Duplicate item_uid: {uid}")

        seen.add(uid)

        unit_id = sample["unit_id"]

        if unit_id not in resolved:
            resolved[unit_id] = resolve_pdf_path(
                dataset_id,
                unit_id,
                sample["unit"],
            )

    print(
        f"[Preflight] dataset={dataset_id} "
        f"| QA={actual} "
        f"| PDFs={len(resolved)}",
        flush=True,
    )

def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    if args.pdf_dpi != 144:
        raise ValueError("Formal protocol v1 requires 144 DPI.")
    if args.input_size != 448:
        raise ValueError("Formal InternVL protocol v1 requires input_size=448.")
    if args.tile_budget < 1:
        raise ValueError("tile_budget must be positive.")
    if args.native_max_num < 1 or args.native_max_num > 12:
        raise ValueError("native_max_num must be in [1, 12].")
    if args.use_thumbnail:
        raise ValueError(
            "Formal protocol v1 keeps use_thumbnail=False. "
            "Create a new protocol version to enable thumbnails."
        )

    spec = get_dataset_spec(args.dataset_id)
    data_path = Path(spec["json_path"])
    data = load_json(data_path)
    samples = flatten_dataset(args.dataset_id, data)
    validate_dataset(args.dataset_id, samples)

    prompt_id, system_prompt = get_prompt(spec["prompt_key"])
    save_path = raw_output_path(args.dataset_id)
    manifest_path = manifest_output_path(args.dataset_id)

    if args.dry_run:
        print(f"[DryRun] data={data_path}", flush=True)
        print(f"[DryRun] pdf_root={spec['pdf_root']}", flush=True)
        print(f"[DryRun] output={save_path}", flush=True)
        print(f"[DryRun] prompt_id={prompt_id}", flush=True)
        print(f"[DryRun] prompt_sha256={prompt_sha256(system_prompt)}", flush=True)
        print(f"[DryRun] tile_policy={TILE_POLICY_ID}", flush=True)
        print(f"[DryRun] tile_budget={args.tile_budget}", flush=True)
        print(f"[DryRun] input_size={args.input_size}", flush=True)
        print(f"[DryRun] use_thumbnail=False", flush=True)
        return

    validate_environment()
    seed_everything(args.seed)

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")

    print("=" * 112, flush=True)
    print(f"Model              : {MODEL_NAME}", flush=True)
    print(f"Python             : {sys.executable}", flush=True)
    print(f"Transformers       : {transformers.__version__}", flush=True)
    print(f"Model path         : {model_path}", flush=True)
    print(f"Dataset            : {args.dataset_id}", flush=True)
    print(f"Dataset path       : {data_path}", flush=True)
    print(f"PDF root           : {spec['pdf_root']}", flush=True)
    print(f"PDF DPI            : {args.pdf_dpi}", flush=True)
    print(f"Input size         : {args.input_size}", flush=True)
    print(f"Prompt ID          : {prompt_id}", flush=True)
    print(f"Prompt version     : {PROMPT_VERSION}", flush=True)
    print(f"Raw result         : {save_path}", flush=True)
    print(f"Visible GPU(s)     : {os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"Logical CUDA       : cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)
    print("Page dropping      : NEVER", flush=True)
    print("Gold answer sent   : NEVER", flush=True)
    print("Gold evidence sent : NEVER", flush=True)
    print("Thinking           : OFF", flush=True)
    print("Sampling           : OFF", flush=True)
    print("Use thumbnail      : False", flush=True)
    print(f"Tile policy        : {TILE_POLICY_ID}", flush=True)
    print(f"Tile budget        : {args.tile_budget}", flush=True)
    print(f"Native max/page    : {args.native_max_num}", flush=True)
    print(f"Max new tokens     : {args.max_new_tokens}", flush=True)
    print(f"Context margin     : {args.context_safety_margin}", flush=True)
    print(f"Device map         : {args.device_map}", flush=True)
    print(f"Use flash-attn     : {args.use_flash_attn}", flush=True)
    print("=" * 112, flush=True)

    if args.use_flash_attn:
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("flash_attn requested but not installed.") from exc

    print("[Load] Loading InternVL3.5-8B ...", flush=True)
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
        device_map=args.device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        use_fast=False,
    )
    model.system_message = system_prompt

    vision_device = get_vision_device(model)
    context_limit = get_context_limit(model)
    num_image_token = get_num_image_token(model)

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        used_devices = sorted(
            {
                str(device)
                for device in device_map.values()
                if str(device) not in {"cpu", "disk"}
            }
        )
        print(f"[Load] Model logical devices: {used_devices}", flush=True)
    print(f"[Load] Vision input device : {vision_device}", flush=True)
    print(f"[Load] Context limit       : {context_limit}", flush=True)
    print(f"[Load] Tokens / image tile : {num_image_token}", flush=True)
    print("[Load] Model and tokenizer loaded.", flush=True)

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
        "model_path": str(model_path),
        "dataset_id": args.dataset_id,
        "dataset_path": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "expected_qa_count": spec["expected_qa_count"],
        "prompt_id": prompt_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(system_prompt),
        "pdf_dpi": args.pdf_dpi,
        "input_size": args.input_size,
        "page_numbering": "physical_pdf_page_1_based",
        "page_dropping": False,
        "tile_policy": TILE_POLICY_ID,
        "tile_budget": args.tile_budget,
        "native_max_num_per_page": args.native_max_num,
        "use_thumbnail": False,
        "max_new_tokens": args.max_new_tokens,
        "context_safety_margin": args.context_safety_margin,
        "context_limit_from_model": context_limit,
        "num_image_token_from_model": num_image_token,
        "device_map": args.device_map,
        "use_flash_attn": args.use_flash_attn,
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

    cache = SinglePdfTensorCache()
    attempted_this_run = 0
    skipped_completed = 0

    try:
        for sample in samples:
            unit_id = sample["unit_id"]
            qa_id = sample["qa_id"]
            qa_result = results[unit_id]["QA"][qa_id]

            if not args.overwrite and completed_entry(qa_result):
                skipped_completed += 1
                continue
            if args.max_samples > 0 and attempted_this_run >= args.max_samples:
                break
            attempted_this_run += 1

            pdf_path = resolve_pdf_path(args.dataset_id, unit_id, sample["unit"])
            document_manifest = build_document_manifest(args.dataset_id, sample["unit"])

            current_stage = "page_count"
            try:
                page_count = get_pdf_page_count(pdf_path)
                max_num = select_max_num_per_page(
                    page_count=page_count,
                    tile_budget=args.tile_budget,
                    native_max=args.native_max_num,
                )

                current_stage = "pdf_preprocess"
                _, num_patches_list, processed_page_count = cache.get_cpu(
                    pdf_path=pdf_path,
                    dpi=args.pdf_dpi,
                    input_size=args.input_size,
                    max_num_per_page=max_num,
                    use_thumbnail=False,
                )
                if processed_page_count != page_count:
                    raise RuntimeError(
                        f"PDF page count changed: counted={page_count}, "
                        f"processed={processed_page_count}"
                    )

                total_tiles = int(sum(num_patches_list))
                model_question = build_question(
                    page_count=page_count,
                    question=str(sample["qa"].get("question", "")),
                    document_manifest=document_manifest,
                )
                estimate = estimate_context_tokens(
                    tokenizer=tokenizer,
                    model_question=model_question,
                    system_prompt=system_prompt,
                    total_tiles=total_tiles,
                    num_image_token=num_image_token,
                    max_new_tokens=args.max_new_tokens,
                    safety_margin=args.context_safety_margin,
                )

                print(
                    f"[{sample['flat_index']}/{len(samples)}] "
                    f"{sample['item_uid']} | pages={page_count} | "
                    f"max_num/page={max_num} | tiles={total_tiles} | "
                    f"ctx_est={estimate['estimated_total_with_output_margin']}/{context_limit} | "
                    f"pdf={pdf_path}",
                    flush=True,
                )
                print(
                    f"[Tiles] num_patches_list={num_patches_list}",
                    flush=True,
                )

                common_meta = {
                    "item_uid": sample["item_uid"],
                    "model_name": MODEL_NAME,
                    "pdf_path_resolved": str(pdf_path),
                    "pdf_total_pages": page_count,
                    "pdf_pages_supplied": list(range(1, page_count + 1)),
                    "pdf_pages_supplied_count": page_count,
                    "render_dpi": args.pdf_dpi,
                    "input_size": args.input_size,
                    "prompt_id": prompt_id,
                    "prompt_version": PROMPT_VERSION,
                    "document_manifest_sent": document_manifest,
                    "gold_answer_sent": False,
                    "gold_evidence_pages_sent": False,
                    "tile_policy": TILE_POLICY_ID,
                    "tile_budget": args.tile_budget,
                    "max_num_per_page_used": max_num,
                    "use_thumbnail": False,
                    "num_patches_list": num_patches_list,
                    "total_tiles": total_tiles,
                    "num_image_token": num_image_token,
                    "context_limit": context_limit,
                    "context_safety_margin": args.context_safety_margin,
                    **estimate,
                }

                if estimate["estimated_total_with_output_margin"] > context_limit:
                    qa_result.update(common_meta)
                    qa_result.update(
                        {
                            "status": "context_overflow",
                            "answer_pre_raw": "",
                            "error_type": "ContextOverflow",
                            "error_message": (
                                "Estimated InternVL multimodal context exceeds "
                                f"model max_position_embeddings: "
                                f"{estimate['estimated_total_with_output_margin']} > {context_limit}"
                            ),
                        }
                    )
                    print(
                        "[ContextOverflow] generation skipped; result retained as technical failure.",
                        flush=True,
                    )
                    atomic_write_json(save_path, results)
                    continue

                current_stage = "device_transfer"
                pixel_values = cache.get_device(vision_device)

                current_stage = "generation"
                raw_text, elapsed = infer_one(
                    pixel_values=pixel_values,
                    num_patches_list=num_patches_list,
                    model_question=model_question,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
                parsed = quick_parse_prediction(raw_text)
                status = "completed" if parsed is not None else "parse_failed"

                qa_result.update(common_meta)
                qa_result.update(
                    {
                        "status": status,
                        "answer_pre_raw": raw_text,
                        "elapsed_seconds": round(elapsed, 4),
                    }
                )
                print(f"[Done] status={status} elapsed={elapsed:.2f}s", flush=True)
                print(f"[Raw] {raw_text[:2000]}", flush=True)

            except Exception as exc:
                status = "oom" if is_cuda_oom_error(exc) else "error"
                qa_result.update(
                    {
                        "item_uid": sample["item_uid"],
                        "model_name": MODEL_NAME,
                        "status": status,
                        "answer_pre_raw": "",
                        "error_stage": current_stage,
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
                print(
                    f"[{status.upper()}] {sample['item_uid']} | "
                    f"stage={current_stage} | {type(exc).__name__}: {exc}",
                    flush=True,
                )
                cache.clear_device()
                gc.collect()
                torch.cuda.empty_cache()

            atomic_write_json(save_path, results)

    finally:
        cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status_counts: Dict[str, int] = {}
    for sample in samples:
        qa = (
            results.get(sample["unit_id"], {})
            .get("QA", {})
            .get(sample["qa_id"], {})
        )
        status = str(qa.get("status", "not_run"))
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest["completed_at_unix"] = time.time()
    manifest["status_counts"] = status_counts
    manifest["attempted_this_run"] = attempted_this_run
    manifest["completed_skipped_at_start"] = skipped_completed
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(save_path, results)

    print("=" * 112, flush=True)
    print(f"[Summary] {args.dataset_id} | {status_counts}", flush=True)
    print(f"[Saved] {save_path}", flush=True)
    print(f"[Manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
