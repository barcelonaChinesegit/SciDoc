#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import copy
import gc
import hashlib
import math
import json
import os
import re
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pypdfium2 as pdfium
from PIL import Image
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings.paths import REPO_ROOT, DATA_ROOT, PDF_ROOT, MODEL_ROOT
from settings.pdfqa_prompts import (
    PROMPT_VERSION,
    PROMPTS,
    prompt_sha256,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "models" / "Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output/raw_result"
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / "output/render_cache"

DATASETS: Dict[str, Dict[str, Any]] = {
    "ordinary": {
        "expected": 1000,
        "data": ROOT / "data" / "qa" / "7.final_2200" / "ordinary_qa.json",
        "pdf_root": PDF_ROOT,
        "prompt_key": "ordinary",
        "prompt_id": "ordinary-v2-strict-min",
    },
    "unanswerable": {
        "expected": 200,
        "data": ROOT / "data" / "qa" / "7.final_2200" / "unanswerable_qa.json",
        "pdf_root": PDF_ROOT,
        "prompt_key": "ordinary",
        "prompt_id": "ordinary-v2-strict-min",
    },
    "reasoning": {
        "expected": 200,
        "data": ROOT / "data" / "qa" / "7.final_2200" / "reasoning_qa.json",
        "pdf_root": PDF_ROOT,
        "prompt_key": "reasoning",
        "prompt_id": "reasoning-v2-strict-min",
    },
    "cross_pdf": {
        "expected": 800,
        "data": ROOT / "data" / "qa" / "7.final_2200" / "cross_pdf_qa.json",
        "pdf_root": PDF_ROOT,
        "prompt_key": "cross_document",
        "prompt_id": "cross-document-v2-strict-min",
    },
}


def build_system_prompt(dataset_id: str) -> str:
    key = DATASETS[dataset_id]["prompt_key"]
    return PROMPTS[key][1]


def evidence_mode(dataset_id: str) -> str:
    return "strict-minimum"


PREDICTION_FIELDS = {
    "model_name", "model_path", "model_family", "inference_backend", "dataset_id",
    "status", "error_type", "error_message", "traceback_tail",
    "answer_pre_raw", "answer_pre", "evidence_pages_pre",
    "prediction_parse_mode", "prediction_parse_status", "parsed_valid", "predicted_page_issues",
    "pdf_path_resolved", "pdf_total_pages", "pdf_pages_supplied", "pdf_pages_supplied_count",
    "render_dpi", "render_format", "render_jpeg_quality", "render_original_pixel_sizes",
    "prompt_id", "prompt_version", "evidence_minimality_mode",
    "gold_answer_sent", "gold_evidence_pages_sent", "document_manifest_sent",
    "format_repair_used", "format_repair_raw_output",
    "elapsed_seconds", "generation_seconds", "input_tokens", "generated_tokens",
    "context_limit", "context_usage_pct", "nominal_visual_tokens_hint", "image_count",
    "max_new_tokens", "do_sample", "dtype", "attn_implementation",
    "reasoning_mode", "reasoning_content_saved", "gpu_peak_allocated_gib",
    "processor_image_policy", "source_count_at_run", "source_expected_count",
    "context_fit_enabled", "context_fit_scale", "context_fit_effective_dpi",
    "context_fit_attempts", "context_fit_retry_history", "context_fit_target_prompt_tokens",
    "context_fit_min_scale", "render_fitted_pixel_sizes",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mistral Small 3.1 24B Instruct 2503 vLLM Scheme-B context-fit runner for the five formal PDF-QA datasets. "
            "Gold answers/evidence are never serialized into model input."
        )
    )
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--model-path", type=Path, default=Path(os.getenv("MISTRAL31_MODEL_PATH", str(DEFAULT_MODEL_PATH))))
    p.add_argument("--output-root", type=Path, default=Path(os.getenv("MISTRAL31_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))))
    p.add_argument("--cache-root", type=Path, default=Path(os.getenv("MISTRAL31_CACHE_ROOT", str(DEFAULT_CACHE_ROOT))))
    p.add_argument("--dpi", type=int, default=144)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--source-qa-limit", type=int, default=-1,
                   help="Restrict each selected dataset to its deterministic first N source QA; -1 = full dataset.")
    p.add_argument("--max-new-qa", type=int, default=-1,
                   help="Stop after N newly attempted QA per dataset; -1 = unlimited. Mostly for debugging.")
    p.add_argument("--allow-limited-source-mismatch", action="store_true",
                   help="Legacy compatibility flag. The current formal source counts are 1200/398/400/100/100, so this is normally unnecessary.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-format-repair", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--base-url", default=os.getenv("MISTRAL31_VLLM_BASE_URL", "http://127.0.0.1:8101/v1"))
    p.add_argument("--served-model", default=os.getenv("MISTRAL31_SERVED_MODEL", "mistral31"))
    p.add_argument("--request-timeout", type=float, default=float(os.getenv("MISTRAL31_REQUEST_TIMEOUT", "1800")))
    p.add_argument("--context-fit-target-prompt-tokens", type=int, default=int(os.getenv("MISTRAL31_CONTEXT_FIT_TARGET", "118000")),
                   help="When a full-page request overflows 128k, uniformly resize ALL pages until the prompt is near/below this target. Default 118000 leaves more output/runtime headroom.")
    p.add_argument("--context-fit-min-scale", type=float, default=float(os.getenv("MISTRAL31_CONTEXT_FIT_MIN_SCALE", "0.35")),
                   help="Minimum linear resize scale relative to the 144-DPI render. No page is ever removed.")
    p.add_argument("--context-fit-max-attempts", type=int, default=int(os.getenv("MISTRAL31_CONTEXT_FIT_MAX_ATTEMPTS", "6")),
                   help="Maximum context-fit inference attempts per QA, including the first attempt.")
    return p.parse_args()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_qa(data: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any], str, Dict[str, Any]]]:
    for unit_id, unit in data.items():
        if not isinstance(unit, dict):
            continue
        qa_map = unit.get("QA")
        if not isinstance(qa_map, dict):
            continue
        for qa_id, qa in qa_map.items():
            if isinstance(qa, dict):
                yield str(unit_id), unit, str(qa_id), qa


def count_qa(data: Dict[str, Any]) -> int:
    return sum(1 for _ in iter_qa(data))


def validate_dataset_counts(args: argparse.Namespace) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for dataset_id in args.datasets:
        spec = DATASETS[dataset_id]
        path: Path = spec["data"]
        if not path.exists():
            raise FileNotFoundError(path)
        data = load_json(path)
        if not isinstance(data, dict):
            raise TypeError(f"{dataset_id}: top-level JSON must be dict: {path}")
        found = count_qa(data)
        expected = int(spec["expected"])
        counts[dataset_id] = found
        if found != expected:
            limited_ok = (
                args.allow_limited_source_mismatch
                and args.source_qa_limit > 0
                and found >= args.source_qa_limit
            )
            if limited_ok:
                print(
                    f"[WARN] {dataset_id}: expected {expected}, found {found}; "
                    f"allowing ONLY limited prefix run because --source-qa-limit={args.source_qa_limit}.",
                    flush=True,
                )
            else:
                raise ValueError(
                    f"{dataset_id}: expected {expected} QA, found {found} in {path}. "
                    "The runner uses the current formal dataset counts: 1200/398/400/100/100."
                )
    return counts


def merge_existing_predictions(source: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(source)
    if not isinstance(existing, dict):
        return out
    for unit_id, _, qa_id, qa in iter_qa(out):
        old_unit = existing.get(unit_id)
        if not isinstance(old_unit, dict):
            continue
        old_map = old_unit.get("QA")
        if not isinstance(old_map, dict):
            continue
        old = old_map.get(qa_id)
        if not isinstance(old, dict):
            continue
        for k in PREDICTION_FIELDS:
            if k in old:
                qa[k] = copy.deepcopy(old[k])
    return out


def is_success(qa: Dict[str, Any]) -> bool:
    return (
        qa.get("status") in {"completed", "normalized_completed"}
        and qa.get("parsed_valid") is True
    )


def is_attempted(qa: Dict[str, Any]) -> bool:
    """
    Resume semantics:
    skip every QA that already received one formal model answer,
    including strict-format failures.

    Technical/runtime failures remain retryable.
    """
    return qa.get("status") in {
        "completed",
        "normalized_completed",
        "parse_failed",
    }


def find_pdf_hint(unit: Dict[str, Any]) -> Optional[str]:
    candidates = [
        "pdf_path", "pdf_path_resolved", "merged_pdf_path", "source_pdf_path",
        "pdf_file", "pdf_filename", "file_name", "filename",
    ]
    for key in candidates:
        v = unit.get(key)
        if isinstance(v, str) and v.strip().lower().endswith(".pdf"):
            return v.strip()
    meta = unit.get("pdf_metadata")
    if isinstance(meta, dict):
        for key in candidates:
            v = meta.get(key)
            if isinstance(v, str) and v.strip().lower().endswith(".pdf"):
                return v.strip()
    return None


def resolve_pdf_path(unit_id: str, unit: Dict[str, Any], pdf_root: Path) -> Path:
    hint = find_pdf_hint(unit)
    if hint:
        p = Path(hint)
        if not p.is_absolute():
            p = pdf_root / p
        if p.exists():
            return p
    direct = pdf_root / f"{unit_id}.pdf"
    if direct.exists():
        return direct
    matches = list(pdf_root.glob(f"{unit_id}*.pdf"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot resolve PDF for unit={unit_id} under {pdf_root}")


def safe_cache_key(pdf_path: Path, dpi: int, jpeg_quality: int) -> str:
    st = pdf_path.stat()
    s = f"{pdf_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|dpi={dpi}|q={jpeg_quality}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def render_pdf_cache(pdf_path: Path, cache_root: Path, dpi: int, jpeg_quality: int) -> Tuple[List[Path], List[Tuple[int, int]]]:
    key = safe_cache_key(pdf_path, dpi, jpeg_quality)
    cache_dir = cache_root / key
    manifest = cache_dir / "manifest.json"
    if manifest.exists():
        try:
            m = load_json(manifest)
            if (
                m.get("pdf_path") == str(pdf_path.resolve())
                and m.get("dpi") == dpi
                and m.get("jpeg_quality") == jpeg_quality
            ):
                files = [cache_dir / x for x in m.get("files", [])]
                sizes = [tuple(x) for x in m.get("sizes", [])]
                if files and len(files) == len(sizes) and all(x.exists() for x in files):
                    return files, sizes
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    files: List[Path] = []
    sizes: List[Tuple[int, int]] = []
    try:
        for idx in range(len(doc)):
            page = doc[idx]
            pil = page.render(scale=scale).to_pil().convert("RGB")
            out = cache_dir / f"page_{idx + 1:04d}.jpg"
            tmp = out.with_suffix(".jpg.tmp")
            pil.save(tmp, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
            os.replace(tmp, out)
            files.append(out)
            sizes.append(tuple(pil.size))
            pil.close()
    finally:
        try:
            doc.close()
        except Exception:
            pass

    atomic_write_json(manifest, {
        "pdf_path": str(pdf_path.resolve()),
        "dpi": dpi,
        "jpeg_quality": jpeg_quality,
        "files": [p.name for p in files],
        "sizes": [list(s) for s in sizes],
        "note": "External render is 144 DPI by default; Mistral/Pixtral processor applies its own model-native image normalization afterwards.",
    })
    return files, sizes



def _scale_tag(scale: float) -> str:
    return f"s{int(round(scale * 1000)):04d}"


def resize_page_cache(
    page_files: List[Path],
    original_sizes: List[Tuple[int, int]],
    scale: float,
    jpeg_quality: int,
) -> Tuple[List[Path], List[Tuple[int, int]]]:
    """Uniformly resize every page by the same linear scale. No page is dropped."""
    scale = float(scale)
    if scale >= 0.9995:
        return page_files, original_sizes
    if not page_files:
        raise ValueError("page_files is empty")

    variant_dir = page_files[0].parent / f"context_fit_{_scale_tag(scale)}"
    manifest = variant_dir / "manifest.json"
    expected_sizes = [
        (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        for (w, h) in original_sizes
    ]
    if manifest.exists():
        try:
            m = load_json(manifest)
            files = [variant_dir / x for x in m.get("files", [])]
            sizes = [tuple(x) for x in m.get("sizes", [])]
            if (
                abs(float(m.get("scale", -1.0)) - scale) < 1e-6
                and len(files) == len(page_files)
                and len(sizes) == len(expected_sizes)
                and all(x.exists() for x in files)
            ):
                return files, sizes
        except Exception:
            pass

    variant_dir.mkdir(parents=True, exist_ok=True)
    out_files: List[Path] = []
    out_sizes: List[Tuple[int, int]] = []
    for idx, src in enumerate(page_files):
        dst = variant_dir / f"page_{idx + 1:04d}.jpg"
        target_size = expected_sizes[idx]
        if not dst.exists():
            with Image.open(src) as im:
                im = im.convert("RGB")
                if im.size != target_size:
                    im = im.resize(target_size, Image.Resampling.LANCZOS)
                tmp = dst.with_suffix(".jpg.tmp")
                im.save(tmp, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
                os.replace(tmp, dst)
        out_files.append(dst)
        out_sizes.append(target_size)

    atomic_write_json(manifest, {
        "scale": scale,
        "source_files": [str(x.resolve()) for x in page_files],
        "files": [x.name for x in out_files],
        "sizes": [list(x) for x in out_sizes],
        "policy": "Uniform linear resize of every 144-DPI rendered page; page count/order unchanged.",
    })
    return out_files, out_sizes


def parse_context_overflow(exc: Exception, context_limit: int) -> Tuple[bool, Optional[int], str]:
    """
    Return (is_context_overflow, estimated_prompt_tokens, diagnostic).
    vLLM/OpenAI can report e.g.:
      max_tokens must be at least 1, got -52419.
    In that case available = context_limit - prompt_tokens = -52419.
    """
    text = str(exc)
    low = text.lower()

    m = re.search(r"max_tokens\s+must\s+be\s+at\s+least\s+1,\s*got\s*(-\d+)", text, flags=re.I)
    if m:
        deficit = abs(int(m.group(1)))
        return True, int(context_limit + deficit), f"negative_available_tokens={-deficit}"

    patterns = [
        r"(?:request|prompt|input)\s+(?:has|contains|is)\s*([0-9][0-9,]*)\s*(?:input\s*)?tokens",
        r"([0-9][0-9,]*)\s+tokens.*(?:maximum|max).*?([0-9][0-9,]*)",
    ]
    for pat in patterns:
        mm = re.search(pat, text, flags=re.I | re.S)
        if mm:
            try:
                first = int(mm.group(1).replace(",", ""))
                if first > context_limit:
                    return True, first, "explicit_prompt_tokens"
            except Exception:
                pass

    context_markers = (
        "maximum context length",
        "max model len",
        "max_model_len",
        "context length",
        "too many tokens",
        "prompt is too long",
        "input is too long",
    )
    if any(x in low for x in context_markers):
        return True, None, "generic_context_overflow"
    return False, None, ""


def next_context_fit_scale(
    current_scale: float,
    estimated_prompt_tokens: Optional[int],
    target_prompt_tokens: int,
    min_scale: float,
) -> float:
    if estimated_prompt_tokens and estimated_prompt_tokens > 0:
        # Pixtral visual tokens are approximately proportional to image area,
        # therefore the linear image scale follows sqrt(target/current).
        factor = 0.96 * math.sqrt(float(target_prompt_tokens) / float(estimated_prompt_tokens))
        # Always shrink by at least 10% after a confirmed overflow.
        factor = min(0.90, factor)
        factor = max(0.50, factor)
    else:
        factor = 0.80
    new_scale = current_scale * factor
    # Quantize downward to make cache keys stable and avoid rounding back upward.
    new_scale = math.floor(new_scale * 1000.0) / 1000.0
    new_scale = max(float(min_scale), new_scale)
    if new_scale >= current_scale:
        return current_scale
    return new_scale


def strip_fence(text: str) -> str:
    s = text.strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.I | re.S)
    return m.group(1).strip() if m else s


def extract_json_candidate(text: str) -> str:
    s = strip_fence(text)
    for parser in (json.loads,):
        try:
            obj = parser(s)
            if isinstance(obj, dict):
                return s
        except Exception:
            pass
    first = s.find("{")
    if first < 0:
        return s
    depth = 0
    in_string = False
    quote = ""
    esc = False
    for i in range(first, len(s)):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_string = False
            continue
        if ch in ('"', "'"):
            in_string = True
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[first:i + 1]
    return s


def _flatten_singletons(v: Any) -> Any:
    while isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
        v = v[0]
    return v


def normalize_pages(v: Any) -> Optional[List[int]]:
    v = _flatten_singletons(v)
    if not isinstance(v, list):
        return None
    out: List[int] = []
    for x in v:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            n = x
        elif isinstance(x, float) and x.is_integer():
            n = int(x)
        elif isinstance(x, str):
            t = x.strip()
            m = re.fullmatch(
                r"(?:(?:PDF[_\s-]*PAGE|PAGE)[_\s-]*)?(\d+)(?:[_\s-]*(?:START|END))?",
                t,
                flags=re.I,
            )
            if not m:
                return None
            n = int(m.group(1))
        else:
            return None
        out.append(n)
    dedup: List[int] = []
    seen = set()
    for n in out:
        if n not in seen:
            seen.add(n)
            dedup.append(n)
    return dedup


def parse_prediction(raw: str):
    text = str(raw or "").strip()
    normalized = False

    # Only deterministic outer Markdown-fence removal.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()

        if len(lines) >= 3:
            first = lines[0].strip().lower()
            last = lines[-1].strip()

            if first in {"```", "```json"} and last == "```":
                text = "\n".join(lines[1:-1]).strip()
                normalized = True

    if not text:
        return False, "", [], "empty"

    try:
        obj = json.loads(text)

    except Exception as exc:
        return (
            False,
            "",
            [],
            "invalid_json:" + type(exc).__name__,
        )

    if not isinstance(obj, dict):
        return False, "", [], "not_object"

    if set(obj.keys()) != {"answer_pre", "evidence_pages"}:
        return False, "", [], "wrong_keys"

    answer = obj.get("answer_pre")
    pages = obj.get("evidence_pages")

    if not isinstance(answer, str) or not answer.strip():
        return False, "", [], "invalid_answer_pre"

    if not isinstance(pages, list):
        return False, "", [], "evidence_not_list"

    if not all(
        isinstance(x, int) and not isinstance(x, bool)
        for x in pages
    ):
        return False, "", [], "evidence_not_integer_list"

    if answer.strip() == "Unanswerable" and pages:
        return False, "", [], "unanswerable_with_evidence"

    return (
        True,
        answer.strip(),
        list(pages),
        (
            "outer_markdown_fence_normalized"
            if normalized
            else "strict_ok"
        ),
    )


def build_multimodal_messages(dataset_id: str, question: str, page_files: List[Path]) -> List[Dict[str, Any]]:
    # GOLD-SAFETY: this function accepts only the question string + rendered page paths.
    # It has no access to qa['answer'], qa['evidence_pages'], evidence_items, review ledgers, etc.
    user_content: List[Dict[str, Any]] = [{
        "type": "text",
        "text": (
            "Question: " + question.strip() +
            "\nInspect all supplied pages before answering. Physical page labels are external and 1-based."
        ),
    }]
    for i, path in enumerate(page_files, start=1):
        user_content.append({"type": "text", "text": f"PDF_PAGE_{i}_START"})
        user_content.append({"type": "image_url", "image_url": {"url": "file://" + str(path.resolve())}})
        user_content.append({"type": "text", "text": f"PDF_PAGE_{i}_END"})
    user_content.append({
        "type": "text",
        "text": "Now answer the Question above. Return ONLY the required JSON object with keys answer_pre and evidence_pages.",
    })
    return [
        {"role": "system", "content": build_system_prompt(dataset_id)},
        {"role": "user", "content": user_content},
    ]


def is_fatal_server_exception(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"APIConnectionError", "APITimeoutError"}:
        return True
    status = getattr(exc, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except Exception:
        status = None
    if status in {500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    hard_markers = (
        "connection refused", "connection reset", "server disconnected",
        "engine is dead", "enginedead", "failed to connect",
    )
    return any(m in text for m in hard_markers)


class Mistral31Engine:
    def __init__(self, args: argparse.Namespace):
        self.client = OpenAI(
            base_url=args.base_url,
            api_key="EMPTY",
            timeout=args.request_timeout,
            max_retries=0,
        )
        self.served_model = args.served_model
        self.context_limit = 128000
        print("=" * 110)
        print("Using Mistral Small 3.1 24B via local vLLM OpenAI server")
        print(f"Base URL            : {args.base_url}")
        print(f"Served model        : {args.served_model}")
        print("Tensor-parallel GPU : managed by vLLM server")
        print("Gold answer sent    : NEVER")
        print("Gold evidence sent  : NEVER")
        print("=" * 110, flush=True)
        # Fail fast if server is not reachable.
        models = self.client.models.list()
        ids = [m.id for m in models.data]
        if self.served_model not in ids and ids:
            print(f"[WARN] served-model={self.served_model!r} not in /models={ids}; request will still use configured name.", flush=True)
        print(f"[SERVER READY] context_limit={self.context_limit}", flush=True)

    def healthcheck(self) -> None:
        models = self.client.models.list()
        ids = [m.id for m in models.data]
        if self.served_model not in ids and ids:
            raise FatalServerUnavailable(
                f"served-model={self.served_model!r} not present in /models={ids}"
            )

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.served_model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stream=False,
        )
        elapsed = time.time() - t0
        raw = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        stats = {
            "input_tokens": prompt_tokens,
            "generated_tokens": completion_tokens,
            "context_limit": self.context_limit,
            "context_usage_pct": round(100.0 * (prompt_tokens + max_new_tokens) / self.context_limit, 3) if prompt_tokens else 0.0,
            "generation_seconds": round(elapsed, 4),
            "gpu_peak_allocated_gib": 0.0,
        }
        return raw, stats

    def cleanup(self) -> None:
        return

def infer_one(
    args: argparse.Namespace,
    engine: Mistral31Engine,
    dataset_id: str,
    prompt_id: str,
    question: str,
    page_files: List[Path],
    original_sizes: List[Tuple[int, int]],
    source_count: int,
    expected_count: int,
    initial_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Scheme B: preserve every physical page, but if the native 144-DPI request exceeds
    Mistral's 128000-token context, uniformly shrink ALL page images and retry.
    No page selection/truncation is ever performed.
    """
    start = time.time()
    scale = min(1.0, max(float(args.context_fit_min_scale), float(initial_scale)))
    retry_history: List[Dict[str, Any]] = []
    raw = ""
    stats: Dict[str, Any] = {}
    fitted_files = page_files
    fitted_sizes = original_sizes

    for attempt in range(1, int(args.context_fit_max_attempts) + 1):
        fitted_files, fitted_sizes = resize_page_cache(
            page_files=page_files,
            original_sizes=original_sizes,
            scale=scale,
            jpeg_quality=args.jpeg_quality,
        )
        messages = build_multimodal_messages(dataset_id, question, fitted_files)
        try:
            raw, stats = engine.generate(messages, args.max_new_tokens)
            retry_history.append({
                "attempt": attempt,
                "scale": round(scale, 3),
                "effective_dpi": round(args.dpi * scale, 2),
                "result": "success",
                "input_tokens": int(stats.get("input_tokens", 0) or 0),
            })
            break
        except Exception as exc:
            is_overflow, est_prompt_tokens, diag = parse_context_overflow(exc, engine.context_limit)
            retry_history.append({
                "attempt": attempt,
                "scale": round(scale, 3),
                "effective_dpi": round(args.dpi * scale, 2),
                "result": "context_overflow" if is_overflow else "error",
                "estimated_prompt_tokens": est_prompt_tokens,
                "diagnostic": diag,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1200],
            })
            if not is_overflow:
                raise
            if attempt >= int(args.context_fit_max_attempts):
                raise RuntimeError(
                    f"context_fit_failed_after_{attempt}_attempts: last_scale={scale:.3f}; "
                    f"estimated_prompt_tokens={est_prompt_tokens}; error={exc}"
                ) from exc
            new_scale = next_context_fit_scale(
                current_scale=scale,
                estimated_prompt_tokens=est_prompt_tokens,
                target_prompt_tokens=int(args.context_fit_target_prompt_tokens),
                min_scale=float(args.context_fit_min_scale),
            )
            if new_scale >= scale - 1e-6:
                raise RuntimeError(
                    f"context_fit_cannot_shrink_further: scale={scale:.3f}; "
                    f"min_scale={args.context_fit_min_scale}; error={exc}"
                ) from exc
            if new_scale <= float(args.context_fit_min_scale) + 1e-9 and scale <= float(args.context_fit_min_scale) + 1e-9:
                raise RuntimeError(
                    f"context_fit_min_scale_exhausted: scale={scale:.3f}; error={exc}"
                ) from exc
            print(
                f"[CONTEXT FIT] overflow at scale={scale:.3f}"
                + (f" est_prompt={est_prompt_tokens}" if est_prompt_tokens else "")
                + f" -> retry scale={new_scale:.3f} effective_dpi={args.dpi * new_scale:.1f}",
                flush=True,
            )
            scale = new_scale
    else:
        raise RuntimeError("context_fit_loop_exhausted")

    ok, answer, pages, parse_status = parse_prediction(raw)
    repair_used = False
    repair_raw = ""
    # Formal benchmark: no model-based format repair.
    # Keep the original first-pass parse result unchanged.

    page_count = len(fitted_files)
    issues = sorted({p for p in pages if p < 1 or p > page_count}) if ok else []
    parsed_valid = bool(ok and not issues)
    return {
        "status": "completed" if parsed_valid else "parse_failed",
        "model_name": "Mistral Small 3.1 24B Instruct 2503",
        "model_path": str(args.model_path),
        "model_family": "Mistral-3.1",
        "inference_backend": "vllm_openai_local_tp2_context_fit",
        "dataset_id": dataset_id,
        "prompt_id": prompt_id,
        "prompt_version": PROMPT_VERSION,
        "evidence_minimality_mode": evidence_mode(dataset_id),
        "answer_pre_raw": raw,
        "answer_pre": answer if ok else "",
        "evidence_pages_pre": pages if ok else [],
        "prediction_parse_mode": "local_format_repair" if repair_used else "strict_json_or_mechanical_normalization",
        "prediction_parse_status": "ok" if parsed_valid else parse_status,
        "parsed_valid": parsed_valid,
        "predicted_page_issues": issues,
        "gold_answer_sent": False,
        "gold_evidence_pages_sent": False,
        "document_manifest_sent": False,
        "format_repair_used": repair_used,
        "format_repair_raw_output": repair_raw,
        "elapsed_seconds": round(time.time() - start, 4),
        "generation_seconds": stats.get("generation_seconds", 0.0),
        "input_tokens": stats.get("input_tokens", 0),
        "generated_tokens": stats.get("generated_tokens", 0),
        "context_limit": stats.get("context_limit", engine.context_limit),
        "context_usage_pct": stats.get("context_usage_pct", 0.0),
        "nominal_visual_tokens_hint": None,
        "image_count": page_count,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "dtype": "bfloat16",
        "attn_implementation": "vllm",
        "reasoning_mode": "no_explicit_reasoning_interface; prompt_requests_final_JSON_only",
        "reasoning_content_saved": False,
        "gpu_peak_allocated_gib": stats.get("gpu_peak_allocated_gib", 0.0),
        "processor_image_policy": (
            "Scheme-B context fit: source pages rendered at 144 DPI JPEG95; every physical page retained in original order; "
            "only if the 128000-token context overflows, ALL page images are uniformly downscaled by one shared linear scale; "
            "no page selection, cropping, truncation, or gold-guided processing."
        ),
        "render_dpi": args.dpi,
        "render_format": "JPEG",
        "render_jpeg_quality": args.jpeg_quality,
        "render_original_pixel_sizes": [list(x) for x in original_sizes],
        "render_fitted_pixel_sizes": [list(x) for x in fitted_sizes],
        "context_fit_enabled": True,
        "context_fit_scale": round(scale, 3),
        "context_fit_effective_dpi": round(args.dpi * scale, 2),
        "context_fit_attempts": len(retry_history),
        "context_fit_retry_history": retry_history,
        "context_fit_target_prompt_tokens": int(args.context_fit_target_prompt_tokens),
        "context_fit_min_scale": float(args.context_fit_min_scale),
        "source_count_at_run": source_count,
        "source_expected_count": expected_count,
    }




def make_error_record(
    args: argparse.Namespace,
    dataset_id: str,
    prompt_id: str,
    exc: Exception,
    elapsed: float,
    source_count: int,
    expected_count: int,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "model_name": "Mistral Small 3.1 24B Instruct 2503",
        "model_path": str(args.model_path),
        "model_family": "Mistral-3.1",
        "inference_backend": "vllm_openai_local_tp2_context_fit",
        "dataset_id": dataset_id,
        "prompt_id": prompt_id,
        "prompt_version": PROMPT_VERSION,
        "evidence_minimality_mode": evidence_mode(dataset_id),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_tail": traceback.format_exc()[-6000:],
        "answer_pre_raw": "",
        "answer_pre": "",
        "evidence_pages_pre": [],
        "prediction_parse_mode": "empty",
        "prediction_parse_status": "error",
        "parsed_valid": False,
        "predicted_page_issues": [],
        "gold_answer_sent": False,
        "gold_evidence_pages_sent": False,
        "document_manifest_sent": False,
        "format_repair_used": False,
        "format_repair_raw_output": "",
        "elapsed_seconds": round(elapsed, 4),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "dtype": "bfloat16",
        "attn_implementation": "vllm",
        "reasoning_mode": "no_explicit_reasoning_interface; prompt_requests_final_JSON_only",
        "reasoning_content_saved": False,
        "render_dpi": args.dpi,
        "render_format": "JPEG",
        "render_jpeg_quality": args.jpeg_quality,
        "context_limit": 128000,
        "context_fit_enabled": True,
        "context_fit_scale": None,
        "context_fit_effective_dpi": None,
        "context_fit_attempts": 0,
        "context_fit_retry_history": [],
        "context_fit_target_prompt_tokens": int(args.context_fit_target_prompt_tokens),
        "context_fit_min_scale": float(args.context_fit_min_scale),
        "processor_image_policy": (
            "Scheme-B context fit: source pages rendered at 144 DPI JPEG95; every physical page retained; "
            "uniform whole-page downscale only on context overflow; no page selection/truncation."
        ),
        "source_count_at_run": source_count,
        "source_expected_count": expected_count,
    }


def patch_pdf_metadata(record: Dict[str, Any], pdf_path: Path, page_count: int) -> None:
    record["pdf_path_resolved"] = str(pdf_path)
    record["pdf_total_pages"] = page_count
    record["pdf_pages_supplied"] = list(range(1, page_count + 1))
    record["pdf_pages_supplied_count"] = page_count


def run_dataset(args: argparse.Namespace, engine: Mistral31Engine, dataset_id: str, validated_count: int) -> None:
    spec = DATASETS[dataset_id]
    data_path: Path = spec["data"]
    pdf_root: Path = spec["pdf_root"]
    prompt_id: str = spec["prompt_id"]
    expected_count = int(spec["expected"])
    out_path = args.output_root / f"{dataset_id}__{PROMPT_VERSION}__dpi{args.dpi}__run-v1.json"

    source = load_json(data_path)
    existing = load_json(out_path) if out_path.exists() else None
    result = merge_existing_predictions(source, existing)

    allowed_keys = None
    if args.source_qa_limit >= 0:
        allowed_keys = set()
        for idx, (uid, _unit, qid, _qa) in enumerate(iter_qa(result)):
            if idx >= args.source_qa_limit:
                break
            allowed_keys.add((uid, qid))

    def in_scope(uid: str, qid: str) -> bool:
        return allowed_keys is None or (str(uid), str(qid)) in allowed_keys

    scope_total = validated_count if allowed_keys is None else len(allowed_keys)
    pending_total = sum(
        1 for uid, _, qid, qa in iter_qa(result)
        if in_scope(uid, qid) and (args.force or not is_attempted(qa))
    )

    print("=" * 110)
    print(f"Dataset            : {dataset_id}")
    print(f"Source QA          : {validated_count} (expected {expected_count})")
    print(f"Inference scope    : {scope_total} QA" + (" (fixed source prefix)" if allowed_keys is not None else " (full dataset)"))
    print(f"Pending in scope   : {pending_total}")
    print(f"Data               : {data_path}")
    print(f"PDF root           : {pdf_root}")
    print(f"Output             : {out_path}")
    print(f"DPI                : {args.dpi}")
    print(f"Evidence mode      : {evidence_mode(dataset_id)}")
    print("Model              : Mistral Small 3.1 24B Instruct 2503")
    print(f"Model path         : {args.model_path}")
    print("Whole PDF          : YES; every physical page supplied")
    print("Context fit        : YES; uniform ALL-page downscale only if 128k overflows")
    print(f"Fit target/min     : {args.context_fit_target_prompt_tokens} prompt tokens / scale {args.context_fit_min_scale}")
    print("Gold answer sent   : NEVER")
    print("Gold evidence sent : NEVER")
    print("=" * 110, flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, result)

    processed_new = 0
    completed_new = 0
    failed_new = 0

    for unit_id, unit in list(result.items()):
        if not isinstance(unit, dict) or not isinstance(unit.get("QA"), dict):
            continue
        qa_map: Dict[str, Dict[str, Any]] = unit["QA"]
        pending_ids = [
            str(qid) for qid, qa in qa_map.items()
            if isinstance(qa, dict) and in_scope(str(unit_id), str(qid)) and (args.force or not is_attempted(qa))
        ]
        if not pending_ids:
            continue
        if args.max_new_qa >= 0:
            remain = args.max_new_qa - processed_new
            if remain <= 0:
                break
            pending_ids = pending_ids[:remain]

        try:
            pdf_path = resolve_pdf_path(str(unit_id), unit, pdf_root)
            page_files, original_sizes = render_pdf_cache(pdf_path, args.cache_root, args.dpi, args.jpeg_quality)
            page_count = len(page_files)
            print(
                f"\n[Paper] {unit_id} pages={page_count} pending={len(pending_ids)} pdf={pdf_path}\n"
                f"[Vision hint] Mistral/Pixtral uses resolution-dependent visual tokens; actual processor input_tokens are logged per QA.",
                flush=True,
            )
        except Exception as exc:
            print(f"[PDF ERROR] unit={unit_id}: {type(exc).__name__}: {exc}", flush=True)
            for qid in pending_ids:
                rec = make_error_record(args, dataset_id, prompt_id, exc, 0.0, validated_count, expected_count)
                qa_map[qid].update(rec)
                atomic_write_json(out_path, result)
                processed_new += 1
                failed_new += 1
            continue

        paper_scale = 1.0

        for qid in pending_ids:
            qa = qa_map[qid]
            q_start = time.time()
            question = str(qa.get("question", "")).strip()
            try:
                if not question:
                    raise ValueError("empty question")
                rec = infer_one(
                    args=args,
                    engine=engine,
                    dataset_id=dataset_id,
                    prompt_id=prompt_id,
                    question=question,
                    page_files=page_files,
                    original_sizes=original_sizes,
                    source_count=validated_count,
                    expected_count=expected_count,
                    initial_scale=paper_scale,
                )
                if isinstance(rec.get("context_fit_scale"), (int, float)):
                    paper_scale = min(paper_scale, float(rec["context_fit_scale"]))
                if rec.get("status") == "completed":
                    completed_new += 1
                else:
                    failed_new += 1
            except Exception as exc:
                if is_fatal_server_exception(exc):
                    print(
                        f"[FATAL SERVER] {dataset_id}::{unit_id}::{qid} "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    print(
                        "[FATAL SERVER] Stopping immediately. Current QA is left pending; "
                        "the managed launcher will tear down the entire vLLM process group. "
                        "No automatic restart is attempted.",
                        flush=True,
                    )
                    raise FatalServerUnavailable(str(exc)) from exc
                rec = make_error_record(
                    args, dataset_id, prompt_id, exc, time.time() - q_start, validated_count, expected_count
                )
                failed_new += 1
                print(f"[QA ERROR] {dataset_id}::{unit_id}::{qid} {type(exc).__name__}: {exc}", flush=True)
            patch_pdf_metadata(rec, pdf_path, page_count)
            qa.update(rec)
            atomic_write_json(out_path, result)
            processed_new += 1
            print(
                f"[Done] {dataset_id}::{unit_id}::{qid} status={rec.get('status')} "
                f"parse={rec.get('prediction_parse_status')} evidence_n={len(rec.get('evidence_pages_pre', []))} "
                f"input_tokens={rec.get('input_tokens', 0)} ctx={rec.get('context_usage_pct', 0)}% "
                f"fit_scale={rec.get('context_fit_scale')} eff_dpi={rec.get('context_fit_effective_dpi')} "
                f"fit_attempts={rec.get('context_fit_attempts', 0)} "
                f"peak={rec.get('gpu_peak_allocated_gib', 0)}GiB sec={rec.get('elapsed_seconds', 0):.2f}",
                flush=True,
            )
            if args.max_new_qa >= 0 and processed_new >= args.max_new_qa:
                break
        engine.cleanup()
        if args.max_new_qa >= 0 and processed_new >= args.max_new_qa:
            break

    scope_success = sum(1 for uid, _, qid, qa in iter_qa(result) if in_scope(uid, qid) and is_success(qa))
    scope_fail = sum(
        1 for uid, _, qid, qa in iter_qa(result)
        if in_scope(uid, qid) and qa.get("status") in {"error", "parse_failed"}
    )
    all_success = sum(1 for _, _, _, qa in iter_qa(result) if is_success(qa))
    print(
        f"\n[Dataset Finished] {dataset_id} new={processed_new} completed_new={completed_new} failed_new={failed_new} "
        f"scope_success={scope_success}/{scope_total} scope_fail={scope_fail} all_success={all_success}/{validated_count}\n"
        f"Saved: {out_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    counts = validate_dataset_counts(args)
    if args.dry_run:
        print("[DRY RUN] Dataset validation passed for selected scope; model was not loaded.")
        for ds in args.datasets:
            print(f"  {ds}: {counts[ds]}/{DATASETS[ds]['expected']}")
        return
    engine = Mistral31Engine(args)
    for ds in args.datasets:
        run_dataset(args, engine, ds, counts[ds])
        engine.healthcheck()
        print(f"[SERVER HEALTH] after {ds}: READY", flush=True)


if __name__ == "__main__":
    main()
