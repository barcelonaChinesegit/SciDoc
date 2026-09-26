#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pypdfium2 as pdfium
from openai import OpenAI
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SCIDOC_ROOT = REPO_ROOT.parent
DATA_ROOT = SCIDOC_ROOT / "data"
PDF_ROOT = DATA_ROOT / "pdfs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from settings.pdfqa_prompts import PROMPT_VERSION, PROMPTS  # noqa: E402

MODEL_NAME = "GLM-4.6V"
DEFAULT_MODEL = "glm-4.6v"
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_API_KEY_ENV = "ZHIZENGZENG_API_KEY"
RUN_VERSION = "run-v1"

DATASETS: Dict[str, Dict[str, Any]] = {
    "ordinary": {
        "data": DATA_ROOT / "qa" / "7.final_2200" / "ordinary_qa.json",
        "expected": 1000,
        "prompt_id": "ordinary-v2-strict-min",
        "prompt_key": "ordinary",
    },
    "unanswerable": {
        "data": DATA_ROOT / "qa" / "7.final_2200" / "unanswerable_qa.json",
        "expected": 200,
        "prompt_id": "ordinary-v2-strict-min",
        "prompt_key": "ordinary",
    },
    "reasoning": {
        "data": DATA_ROOT / "qa" / "7.final_2200" / "reasoning_qa.json",
        "expected": 200,
        "prompt_id": "reasoning-v2-strict-min",
        "prompt_key": "reasoning",
    },
    "cross_pdf": {
        "data": DATA_ROOT / "qa" / "7.final_2200" / "cross_pdf_qa.json",
        "expected": 800,
        "prompt_id": "cross-document-v2-strict-min",
        "prompt_key": "cross_document",
    },
}

def system_prompt(dataset_id: str) -> str:
    return PROMPTS[DATASETS[dataset_id]["prompt_key"]][1]


INFERENCE_KEYS = {
    "status", "parsed_valid", "prediction_parse_status", "prediction_parse_reason",
    "prediction_parse_mode", "answer_pre_raw", "answer_pre", "evidence_pages_pre",
    "model_name", "model_id", "api_provider", "api_base_url", "api_request_id",
    "api_finish_reason", "dataset_id", "prompt_id", "prompt_version",
    "evidence_minimality_mode", "gold_answer_sent", "gold_evidence_pages_sent",
    "document_manifest_sent", "format_repair_used", "format_repair_raw_output",
    "reasoning_content_chars", "reasoning_content_discarded", "thinking_mode",
    "thinking_request", "pdf_path_resolved", "pdf_total_pages",
    "pdf_pages_supplied", "pdf_pages_supplied_count", "render_dpi",
    "render_format", "image_encoding", "image_jpeg_quality",
    "image_final_sizes", "image_payload_bytes_total", "image_base64_bytes_total",
    "max_base64_bytes_per_page", "max_output_tokens", "api_attempts", "usage",
    "input_tokens", "output_tokens", "elapsed_seconds", "error_type",
    "error_message", "traceback_tail",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def iter_qa(data: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any], str, Dict[str, Any]]]:
    for uid, unit in data.items():
        if not isinstance(unit, dict):
            continue
        qmap = unit.get("QA")
        if not isinstance(qmap, dict):
            continue
        for qid, qa in qmap.items():
            if isinstance(qa, dict):
                yield str(uid), unit, str(qid), qa


def merge_saved(source: Dict[str, Any], saved: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = copy.deepcopy(source)
    if not isinstance(saved, dict):
        return result

    for uid, unit in result.items():
        if not isinstance(unit, dict) or not isinstance(unit.get("QA"), dict):
            continue
        old_unit = saved.get(uid)
        if not isinstance(old_unit, dict) or not isinstance(old_unit.get("QA"), dict):
            continue
        for qid, qa in unit["QA"].items():
            old = old_unit["QA"].get(qid)
            if not isinstance(qa, dict) or not isinstance(old, dict):
                continue
            for k in INFERENCE_KEYS:
                if k in old:
                    qa[k] = copy.deepcopy(old[k])
    return result


def is_success(qa: Dict[str, Any]) -> bool:
    return (
        qa.get("status") in {"completed", "normalized_completed"}
        and qa.get("parsed_valid") is True
    )


def is_attempted(qa: Dict[str, Any]) -> bool:
    return qa.get("status") in {
        "completed",
        "normalized_completed",
        "parse_failed",
    }


def validate_dataset(dataset_id: str) -> Tuple[Dict[str, Any], int]:
    spec = DATASETS[dataset_id]
    data = load_json(spec["data"])
    if not isinstance(data, dict):
        raise TypeError(f"{dataset_id}: top-level dataset must be dict: {spec['data']}")
    n = sum(1 for _ in iter_qa(data))
    if n != spec["expected"]:
        raise RuntimeError(
            f"{dataset_id}: QA={n}, expected={spec['expected']} in {spec['data']}"
        )
    return data, n


def resolve_pdf(unit_id: str, unit: Dict[str, Any]) -> Path:
    direct = PDF_ROOT / f"{unit_id}.pdf"
    if direct.is_file():
        return direct

    for key in (
        "pdf_path", "merged_pdf_path", "source_pdf_path",
        "paper_id", "pdf_id", "source_pdf_id",
    ):
        value = unit.get(key)
        if not value:
            continue
        s = str(value)
        p = Path(s)
        candidates = [
            p,
            PDF_ROOT / s,
            PDF_ROOT / (s if s.lower().endswith(".pdf") else f"{s}.pdf"),
        ]
        for c in candidates:
            if c.is_file():
                return c

    matches = list(PDF_ROOT.rglob(f"{unit_id}.pdf"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous PDF for unit={unit_id}: {[str(x) for x in matches[:8]]}"
        )
    raise FileNotFoundError(f"Cannot resolve PDF for unit={unit_id} under {PDF_ROOT}")


def page_count(pdf_path: Path) -> int:
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return int(len(doc))
    finally:
        try:
            doc.close()
        except Exception:
            pass


def cache_key(
    pdf_path: Path,
    dpi: int,
    jpeg_quality: int,
    max_b64_bytes: int,
) -> str:
    stat = pdf_path.stat()
    s = (
        f"{pdf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"dpi={dpi}|q={jpeg_quality}|maxb64={max_b64_bytes}"
    )
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def encode_jpeg_with_limit(
    image: Image.Image,
    start_quality: int,
    max_b64_bytes: int,
) -> Tuple[bytes, int, Tuple[int, int]]:
    img = image.convert("RGB")
    if img is not image:
        try:
            image.close()
        except Exception:
            pass

    quality_candidates: List[int] = []
    for q in [start_quality, 85, 78, 70, 60, 50, 40]:
        q = max(25, min(95, int(q)))
        if q not in quality_candidates:
            quality_candidates.append(q)

    try:
        for _resize_round in range(6):
            for q in quality_candidates:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q, optimize=True)
                raw = buf.getvalue()
                estimated_b64 = ((len(raw) + 2) // 3) * 4
                if estimated_b64 <= max_b64_bytes:
                    return raw, q, img.size

            new_w = max(32, int(img.width * 0.85))
            new_h = max(32, int(img.height * 0.85))
            if (new_w, new_h) == img.size:
                break
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            img.close()
            img = resized
    finally:
        try:
            img.close()
        except Exception:
            pass

    raise ValueError(
        "Could not compress one rendered PDF page below Base64 safety limit "
        f"{max_b64_bytes} bytes"
    )


def render_pdf_cache(
    pdf_path: Path,
    cache_root: Path,
    dpi: int,
    jpeg_quality: int,
    max_b64_bytes: int,
) -> Tuple[List[Path], List[int], List[Tuple[int, int]]]:
    key = cache_key(pdf_path, dpi, jpeg_quality, max_b64_bytes)
    cache_dir = cache_root / key
    manifest_path = cache_dir / "manifest.json"

    if manifest_path.is_file():
        try:
            m = load_json(manifest_path)
            if (
                m.get("pdf_path") == str(pdf_path.resolve())
                and m.get("dpi") == dpi
                and m.get("jpeg_quality") == jpeg_quality
                and m.get("max_b64_bytes") == max_b64_bytes
            ):
                files = [cache_dir / x for x in m.get("files", [])]
                if files and all(p.is_file() for p in files):
                    qualities = [int(x) for x in m.get("qualities", [])]
                    sizes = [tuple(x) for x in m.get("sizes", [])]
                    if len(qualities) == len(files) and len(sizes) == len(files):
                        return files, qualities, sizes
        except Exception:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    files: List[Path] = []
    qualities: List[int] = []
    sizes: List[Tuple[int, int]] = []

    try:
        for idx in range(len(doc)):
            page = doc[idx]
            bitmap = None
            pil = None
            try:
                bitmap = page.render(scale=scale)
                pil = bitmap.to_pil().convert("RGB")
                raw, q, final_size = encode_jpeg_with_limit(
                    pil,
                    start_quality=jpeg_quality,
                    max_b64_bytes=max_b64_bytes,
                )
                pil = None  # closed by encode_jpeg_with_limit
            finally:
                try:
                    if pil is not None:
                        pil.close()
                except Exception:
                    pass
                try:
                    if bitmap is not None:
                        bitmap.close()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass

            out = cache_dir / f"page_{idx + 1:04d}.jpg"
            tmp = out.with_suffix(".jpg.tmp")
            with tmp.open("wb") as f:
                f.write(raw)
            os.replace(tmp, out)
            files.append(out)
            qualities.append(q)
            sizes.append(final_size)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    atomic_write_json(
        manifest_path,
        {
            "pdf_path": str(pdf_path.resolve()),
            "dpi": dpi,
            "jpeg_quality": jpeg_quality,
            "max_b64_bytes": max_b64_bytes,
            "files": [p.name for p in files],
            "qualities": qualities,
            "sizes": [list(s) for s in sizes],
        },
    )
    return files, qualities, sizes


def page_files_to_payloads(page_files: List[Path]) -> Tuple[List[str], int, int]:
    urls: List[str] = []
    raw_total = 0
    b64_total = 0
    for p in page_files:
        raw = p.read_bytes()
        raw_total += len(raw)
        b64 = base64.b64encode(raw).decode("ascii")
        url = "data:image/jpeg;base64," + b64
        b64_total += len(url)
        urls.append(url)
    return urls, raw_total, b64_total


def build_user_content(question: str, page_urls: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Question: " + question.strip()
                + "\nInspect all supplied pages before answering. "
                "Physical page labels are external and 1-based."
            ),
        }
    ]
    for i, url in enumerate(page_urls, start=1):
        content.append({"type": "text", "text": f"PDF_PAGE_{i}_START"})
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"PDF_PAGE_{i}_END"})
    content.append(
        {
            "type": "text",
            "text": (
                "Now answer the Question above. Return ONLY the required JSON object "
                "with keys answer_pre and evidence_pages."
            ),
        }
    )
    return content


def validate_prediction_object(
    obj: Any,
    page_count_value: Optional[int],
) -> Tuple[bool, str, List[int], str]:
    if not isinstance(obj, dict):
        return False, "", [], "not_object"
    if set(obj) != {"answer_pre", "evidence_pages"}:
        return False, "", [], "wrong_keys"

    answer = obj["answer_pre"]
    pages = obj["evidence_pages"]

    if not isinstance(answer, str) or not answer.strip():
        return False, "", [], "invalid_answer_pre"
    if not isinstance(pages, list):
        return False, "", [], "evidence_not_list"
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in pages):
        return False, "", [], "evidence_not_integer_list"

    if page_count_value is not None:
        bad = [x for x in pages if x < 1 or x > page_count_value]
        if bad:
            return (
                False,
                "",
                [],
                "evidence_page_out_of_range:" + ",".join(map(str, bad)),
            )

    if answer.strip() == "Unanswerable" and pages:
        return False, "", [], "unanswerable_with_evidence"

    return True, answer.strip(), pages, "ok"


def parse_prediction(
    raw: str,
    page_count_value: Optional[int] = None,
) -> Tuple[bool, str, List[int], str]:
    original = str(raw or "").strip()
    if not original:
        return False, "", [], "empty"

    # 1) Strict whole-output JSON.
    try:
        obj = json.loads(original)
    except Exception:
        obj = None

    if obj is not None:
        ok, answer, pages, reason = validate_prediction_object(
            obj, page_count_value
        )
        if ok:
            return True, answer, pages, "strict_ok"
        return False, "", [], reason

    # 2) Whole-output Markdown JSON fence only.
    if original.startswith("```") and original.endswith("```"):
        lines = original.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip().lower() in {"```", "```json"}
            and lines[-1].strip() == "```"
        ):
            candidate = "\n".join(lines[1:-1]).strip()
            try:
                obj = json.loads(candidate)
            except Exception:
                obj = None
            if obj is not None:
                ok, answer, pages, reason = validate_prediction_object(
                    obj, page_count_value
                )
                if ok:
                    return (
                        True,
                        answer,
                        pages,
                        "outer_markdown_fence_normalized",
                    )

    # 3) Deterministically locate exactly one already-valid JSON object.
    # No regex extraction, no backslash repair, no content mutation.
    decoder = json.JSONDecoder()
    candidates: List[Tuple[int, int, str, List[int]]] = []

    for i, ch in enumerate(original):
        if ch != "{":
            continue
        try:
            obj, consumed = decoder.raw_decode(original[i:])
        except json.JSONDecodeError:
            continue

        ok, answer, pages, _ = validate_prediction_object(
            obj, page_count_value
        )
        if ok:
            candidates.append((i, i + consumed, answer, pages))

    unique: List[Tuple[int, int, str, List[int]]] = []
    seen = set()
    for item in candidates:
        span = (item[0], item[1])
        if span not in seen:
            seen.add(span)
            unique.append(item)

    if len(unique) == 1:
        _, _, answer, pages = unique[0]
        return True, answer, pages, "single_json_object_extracted"

    if len(unique) > 1:
        return (
            False,
            "",
            [],
            f"ambiguous_multiple_json_objects:{len(unique)}",
        )

    return False, "", [], "invalid_json:JSONDecodeError"


def retryable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        x in text
        for x in (
            "429", "rate limit", "timeout", "timed out",
            "connection", "temporar", "502", "503", "504",
        )
    )


def call_api(
    client: OpenAI,
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
) -> Tuple[str, int, str, str, int, Dict[str, Any]]:
    last_exc: Optional[Exception] = None

    for attempt in range(1, args.max_api_attempts + 1):
        try:
            print(
                f"[API SEND] provider=zhizengzeng model={args.model} attempt={attempt}",
                flush=True,
            )
            t0 = time.time()
            completion = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=False,
                max_tokens=args.max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            print(
                f"[API DONE] elapsed={time.time() - t0:.2f}s",
                flush=True,
            )

            if not getattr(completion, "choices", None):
                raise RuntimeError("API returned no choices")

            choice = completion.choices[0]
            message = choice.message
            content = getattr(message, "content", None) or ""
            reasoning = getattr(message, "reasoning_content", None) or ""
            request_id = str(getattr(completion, "id", "") or "")
            finish_reason = str(getattr(choice, "finish_reason", "") or "")

            usage_obj = getattr(completion, "usage", None)
            usage: Dict[str, Any] = {}
            if usage_obj is not None:
                try:
                    usage = usage_obj.model_dump()
                except Exception:
                    usage = {}

            return (
                str(content).strip(),
                len(str(reasoning)),
                request_id,
                finish_reason,
                attempt,
                usage,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= args.max_api_attempts or not retryable_error(exc):
                raise
            sleep_s = min(60.0, float(2 ** (attempt - 1)))
            print(
                f"[API RETRY] attempt={attempt} sleep={sleep_s:.1f}s "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def infer_one(
    client: OpenAI,
    args: argparse.Namespace,
    dataset_id: str,
    question: str,
    page_urls: List[str],
    page_count_value: int,
    raw_image_bytes_total: int,
    b64_total: int,
    qualities: List[int],
    sizes: List[Tuple[int, int]],
    pdf_path: Path,
) -> Dict[str, Any]:
    start = time.time()
    spec = DATASETS[dataset_id]

    messages = [
        {"role": "system", "content": system_prompt(dataset_id)},
        {"role": "user", "content": build_user_content(question, page_urls)},
    ]

    raw, reasoning_chars, request_id, finish_reason, attempts, usage = call_api(
        client, args, messages
    )
    ok, answer, pages, reason = parse_prediction(raw, page_count_value)

    if ok:
        status = "completed" if reason == "strict_ok" else "normalized_completed"
        parse_mode = "strict_json" if reason == "strict_ok" else "deterministic_normalization"
    else:
        status = "parse_failed"
        parse_mode = "failed"

    input_tokens = int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    )
    output_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    )

    return {
        "status": status,
        "parsed_valid": bool(ok),
        "prediction_parse_status": "ok" if ok else reason,
        "prediction_parse_reason": reason,
        "prediction_parse_mode": parse_mode,
        "answer_pre_raw": raw,
        "answer_pre": answer if ok else "",
        "evidence_pages_pre": pages if ok else [],
        "model_name": MODEL_NAME,
        "model_id": args.model,
        "api_provider": "zhizengzeng_openai_compatible",
        "api_base_url": args.base_url,
        "api_request_id": request_id,
        "api_finish_reason": finish_reason,
        "dataset_id": dataset_id,
        "prompt_id": spec["prompt_id"],
        "prompt_version": PROMPT_VERSION,
        "evidence_minimality_mode": "strict-minimum",
        "gold_answer_sent": False,
        "gold_evidence_pages_sent": False,
        "document_manifest_sent": False,
        "format_repair_used": False,
        "format_repair_raw_output": "",
        "reasoning_content_chars": reasoning_chars,
        "reasoning_content_discarded": True,
        "thinking_mode": "disabled",
        "thinking_request": {"type": "disabled"},
        "pdf_path_resolved": str(pdf_path),
        "pdf_total_pages": page_count_value,
        "pdf_pages_supplied": list(range(1, page_count_value + 1)),
        "pdf_pages_supplied_count": page_count_value,
        "render_dpi": args.pdf_dpi,
        "render_format": "JPEG",
        "image_encoding": "data_url_base64",
        "image_jpeg_quality": qualities,
        "image_final_sizes": [list(x) for x in sizes],
        "image_payload_bytes_total": raw_image_bytes_total,
        "image_base64_bytes_total": b64_total,
        "max_base64_bytes_per_page": args.max_base64_bytes_per_page,
        "max_output_tokens": args.max_tokens,
        "api_attempts": attempts,
        "usage": usage,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": round(time.time() - start, 4),
        "error_type": "",
        "error_message": "",
        "traceback_tail": "",
    }


def error_record(
    args: argparse.Namespace,
    dataset_id: str,
    pdf_path: Optional[Path],
    page_count_value: Optional[int],
    exc: Exception,
    elapsed: float,
) -> Dict[str, Any]:
    spec = DATASETS[dataset_id]
    tb = traceback.format_exc()
    return {
        "status": "error",
        "parsed_valid": False,
        "prediction_parse_status": "error",
        "prediction_parse_reason": "",
        "prediction_parse_mode": "empty",
        "answer_pre_raw": "",
        "answer_pre": "",
        "evidence_pages_pre": [],
        "model_name": MODEL_NAME,
        "model_id": args.model,
        "api_provider": "zhizengzeng_openai_compatible",
        "api_base_url": args.base_url,
        "api_request_id": "",
        "api_finish_reason": "",
        "dataset_id": dataset_id,
        "prompt_id": spec["prompt_id"],
        "prompt_version": PROMPT_VERSION,
        "evidence_minimality_mode": "strict-minimum",
        "gold_answer_sent": False,
        "gold_evidence_pages_sent": False,
        "document_manifest_sent": False,
        "format_repair_used": False,
        "format_repair_raw_output": "",
        "reasoning_content_chars": 0,
        "reasoning_content_discarded": True,
        "thinking_mode": "disabled",
        "thinking_request": {"type": "disabled"},
        "pdf_path_resolved": str(pdf_path) if pdf_path else "",
        "pdf_total_pages": page_count_value,
        "pdf_pages_supplied": [],
        "pdf_pages_supplied_count": 0,
        "render_dpi": args.pdf_dpi,
        "render_format": "JPEG",
        "image_encoding": "data_url_base64",
        "image_jpeg_quality": [],
        "image_final_sizes": [],
        "image_payload_bytes_total": 0,
        "image_base64_bytes_total": 0,
        "max_base64_bytes_per_page": args.max_base64_bytes_per_page,
        "max_output_tokens": args.max_tokens,
        "api_attempts": 0,
        "usage": {},
        "input_tokens": 0,
        "output_tokens": 0,
        "elapsed_seconds": round(elapsed, 4),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback_tail": "\n".join(tb.splitlines()[-12:]) if tb else "",
    }


def output_path(output_root: Path, dataset_id: str, dpi: int) -> Path:
    return (
        output_root
        / f"{dataset_id}__{PROMPT_VERSION}__dpi{dpi}__{RUN_VERSION}.json"
    )


def run_dataset(
    args: argparse.Namespace,
    client: Optional[OpenAI],
    dataset_id: str,
) -> None:
    source, total = validate_dataset(dataset_id)
    spec = DATASETS[dataset_id]
    out = output_path(args.output_root, dataset_id, args.pdf_dpi)
    saved = load_json(out) if out.is_file() else None
    result = merge_saved(source, saved)

    pending = sum(
        1
        for *_, qa in iter_qa(result)
        if args.force or not is_attempted(qa)
    )

    print("=" * 100)
    print(
        f"Dataset={dataset_id} QA={total} pending={pending} "
        f"model={args.model} DPI={args.pdf_dpi} max_tokens={args.max_tokens}"
    )
    print(f"Data={spec['data']}")
    print(f"PDF root={PDF_ROOT}")
    print(f"Output={out}")
    print("Thinking=disabled")
    print("Whole PDF=YES; gold answer/evidence=NEVER; model format repair=OFF")
    print("=" * 100)

    if args.dry_run:
        missing: List[str] = []
        checked = 0
        for uid, unit in result.items():
            if not isinstance(unit, dict) or not isinstance(unit.get("QA"), dict):
                continue
            try:
                p = resolve_pdf(str(uid), unit)
                _ = page_count(p)
                checked += 1
            except Exception as exc:
                missing.append(f"{uid}: {type(exc).__name__}: {exc}")
        print(
            f"[Dry Run] dataset={dataset_id} units_checked={checked} "
            f"missing_or_invalid={len(missing)}"
        )
        for x in missing[:20]:
            print("[Dry Run Missing]", x)
        if missing:
            raise RuntimeError(
                f"{dataset_id}: {len(missing)} PDF units missing/invalid"
            )
        return

    assert client is not None

    processed_new = 0
    success_new = 0
    parse_failed_new = 0
    error_new = 0

    for uid, unit in result.items():
        if not isinstance(unit, dict) or not isinstance(unit.get("QA"), dict):
            continue

        qmap = unit["QA"]
        pending_ids = [
            str(qid)
            for qid, qa in qmap.items()
            if isinstance(qa, dict) and (args.force or not is_attempted(qa))
        ]

        if not pending_ids:
            continue

        if args.max_new_qa >= 0:
            remain = args.max_new_qa - processed_new
            if remain <= 0:
                break
            pending_ids = pending_ids[:remain]

        pdf_path: Optional[Path] = None
        n_pages: Optional[int] = None
        page_urls: List[str] = []

        try:
            pdf_path = resolve_pdf(str(uid), unit)
            page_files, qualities, sizes = render_pdf_cache(
                pdf_path=pdf_path,
                cache_root=args.cache_root,
                dpi=args.pdf_dpi,
                jpeg_quality=args.jpeg_quality,
                max_b64_bytes=args.max_base64_bytes_per_page,
            )
            n_pages = len(page_files)
            page_urls, raw_total, b64_total = page_files_to_payloads(page_files)

            print(
                f"\n[Paper] {uid} pages={n_pages} pending={len(pending_ids)} "
                f"pdf={pdf_path}",
                flush=True,
            )
            print(
                f"[Base64] raw_jpeg={raw_total / 1024 / 1024:.3f}MiB "
                f"data_urls={b64_total / 1024 / 1024:.3f}MiB "
                f"q_min={min(qualities)} q_max={max(qualities)}",
                flush=True,
            )

            if b64_total > 44 * 1024 * 1024:
                print(
                    "[WARN] Base64 image payload exceeds 44 MiB before JSON overhead. "
                    "All pages are still supplied; gateway may enforce its own limit.",
                    flush=True,
                )
        except Exception as exc:
            for qid in pending_ids:
                qmap[qid].update(
                    error_record(
                        args, dataset_id, pdf_path, n_pages, exc, 0.0
                    )
                )
                processed_new += 1
                error_new += 1
                atomic_write_json(out, result)
            continue

        for qid in pending_ids:
            qa = qmap[qid]
            t0 = time.time()
            try:
                question = str(qa.get("question", "")).strip()
                if not question:
                    raise ValueError("empty question")

                rec = infer_one(
                    client=client,
                    args=args,
                    dataset_id=dataset_id,
                    question=question,
                    page_urls=page_urls,
                    page_count_value=int(n_pages),
                    raw_image_bytes_total=raw_total,
                    b64_total=b64_total,
                    qualities=qualities,
                    sizes=sizes,
                    pdf_path=pdf_path,
                )
                qa.update(rec)

                if is_success(qa):
                    success_new += 1
                elif qa.get("status") == "parse_failed":
                    parse_failed_new += 1
            except Exception as exc:
                qa.update(
                    error_record(
                        args,
                        dataset_id,
                        pdf_path,
                        n_pages,
                        exc,
                        time.time() - t0,
                    )
                )
                error_new += 1
                print(
                    f"[QA ERROR] {dataset_id}::{uid}::{qid} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            processed_new += 1
            atomic_write_json(out, result)
            print(
                f"[Done] {dataset_id}::{uid}::{qid} "
                f"status={qa.get('status')} "
                f"parse={qa.get('prediction_parse_status')} "
                f"reason={qa.get('prediction_parse_reason')} "
                f"evidence_n={len(qa.get('evidence_pages_pre', []))} "
                f"input_tokens={qa.get('input_tokens', 0)} "
                f"output_tokens={qa.get('output_tokens', 0)} "
                f"sec={qa.get('elapsed_seconds', 0)}",
                flush=True,
            )

            if args.max_new_qa >= 0 and processed_new >= args.max_new_qa:
                break

        del page_urls

        if args.max_new_qa >= 0 and processed_new >= args.max_new_qa:
            break

    atomic_write_json(out, result)

    attempted = sum(
        1 for *_, qa in iter_qa(result) if is_attempted(qa)
    )
    success = sum(
        1 for *_, qa in iter_qa(result) if is_success(qa)
    )

    print(
        f"\n[Dataset Finished] {dataset_id} "
        f"new={processed_new} success_new={success_new} "
        f"parse_failed_new={parse_failed_new} error_new={error_new} "
        f"all_success={success}/{total} all_attempted={attempted}/{total}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GLM-4.6V formal PDF-QA runner via "
            "Zhizengzeng OpenAI-compatible API."
        )
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    p.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("GLM_BASE_URL", DEFAULT_BASE_URL),
    )
    p.add_argument(
        "--model",
        default=os.getenv("GLM_MODEL", DEFAULT_MODEL),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=HERE / "output" / "raw_result",
    )
    p.add_argument(
        "--cache-root",
        type=Path,
        default=HERE / "output" / "render_cache",
    )
    p.add_argument("--pdf-dpi", type=int, default=144)
    p.add_argument("--jpeg-quality", type=int, default=82)
    p.add_argument(
        "--max-base64-bytes-per-page",
        type=int,
        default=9_000_000,
    )
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--request-timeout", type=float, default=1800.0)
    p.add_argument("--max-api-attempts", type=int, default=5)
    p.add_argument("--max-new-qa", type=int, default=-1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for ds in args.datasets:
            run_dataset(args, None, ds)
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not set"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.request_timeout,
        max_retries=0,
    )

    for ds in args.datasets:
        run_dataset(args, client, ds)


if __name__ == "__main__":
    main()
