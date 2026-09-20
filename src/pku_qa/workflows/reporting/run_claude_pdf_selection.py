#!/usr/bin/env python3
"""Run resumable gold-blind Claude PDF inference for dataset selection audits."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from pku_qa.pdf_assets import resolve_pdf_path

from pku_qa.workflows.review.review_cross_pdf_qa_api import (
    CLAUDE_BASE_URL,
    load_key,
    parse_json_response,
    post_with_retries,
    response_text_claude,
)


ROOT = Path(__file__).resolve().parents[4]
_WRITE_LOCK = threading.Lock()
PDF_SYSTEM_PROMPT = """You receive a question and PDF page images labeled [Page N].
Answer only from those pages and return JSON only:
{"answer_pre":"","evidence_pages":[]}
Use the external 1-based [Page N] labels, not page numbers printed in the paper.
Keep the answer concise. Include all and only directly supporting pages as a
sorted unique integer list. If the question cannot be answered from the PDF,
answer_pre must be exactly "Unanswerable" and evidence_pages must be [].
Do not emit any text outside the JSON object."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_prediction(
    value: dict[str, Any], page_count: int
) -> tuple[str, list[int], list[str]]:
    answer = value.get("answer_pre")
    pages = value.get("evidence_pages")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer_pre must be a non-empty string")
    if not isinstance(pages, list) or any(type(page) is not int for page in pages):
        raise ValueError("evidence_pages must be an integer list")
    if any(page < 1 or page > page_count for page in pages):
        raise ValueError("evidence_pages must be within the PDF")
    normalized_pages = sorted(set(pages))
    normalizations = []
    if pages != normalized_pages:
        normalizations.append("sorted_unique_evidence_pages")
    answer = answer.strip()
    if answer == "Unanswerable" and normalized_pages:
        raise ValueError("Unanswerable requires empty evidence_pages")
    if answer != "Unanswerable" and not normalized_pages:
        raise ValueError("An answer requires at least one evidence page")
    return answer, normalized_pages, normalizations


def render_pdf(path: Path) -> tuple[list[tuple[int, str]], dict[str, Any]]:
    trials = ((1584, 68), (1400, 60), (1200, 55), (1056, 50))
    document = pdfium.PdfDocument(path)
    try:
        for max_side, quality in trials:
            rows: list[tuple[int, str]] = []
            binary_bytes = 0
            for index in range(len(document)):
                image = document[index].render(scale=2).to_pil().convert("RGB")
                image.thumbnail((max_side, max_side))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                binary = buffer.getvalue()
                binary_bytes += len(binary)
                rows.append((index + 1, base64.b64encode(binary).decode("ascii")))
            if binary_bytes <= 18 * 1024 * 1024 or (max_side, quality) == trials[-1]:
                return rows, {
                    "render_dpi": 144,
                    "compression_policy": "dynamic_overall_by_page_count",
                    "selected_max_long_side": max_side,
                    "selected_jpeg_quality": quality,
                    "binary_payload_mb": round(binary_bytes / 1024 / 1024, 4),
                    "pdf_total_pages": len(document),
                    "pdf_pages_supplied": list(range(1, len(document) + 1)),
                }
    finally:
        document.close()
    raise AssertionError("unreachable")


def call_claude(
    *, key: str, model: str, question: str, images: list[tuple[int, str]],
    max_tokens: int, timeout: int, retries: int,
) -> tuple[dict[str, Any], str]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Question: {question}"}
    ]
    for page, encoded in images:
        content.extend(
            (
                {"type": "text", "text": f"[Page {page}]"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded,
                    },
                },
            )
        )
    response = post_with_retries(
        f"{CLAUDE_BASE_URL}/v1/messages",
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": PDF_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
        },
        timeout,
        retries,
    )
    payload = response.json()
    return payload, response_text_claude(payload)


def completed(existing: dict[str, Any], source_qa: dict[str, Any]) -> bool:
    return (
        existing.get("status") == "completed"
        and existing.get("question") == source_qa.get("question")
        and existing.get("answer") == source_qa.get("answer")
        and isinstance(existing.get("prediction_answer"), str)
    )


def evaluate_paper(
    paper_id: str,
    paper: dict[str, Any],
    existing_paper: dict[str, Any],
    args: argparse.Namespace,
    key: str,
) -> dict[str, Any]:
    output = copy.deepcopy(paper)
    existing_qas = existing_paper.get("QA", {}) if isinstance(existing_paper, dict) else {}
    pending = [
        (str(qa_id), qa)
        for qa_id, qa in paper.get("QA", {}).items()
        if not completed(existing_qas.get(str(qa_id), {}), qa)
    ]
    for qa_id, qa in paper.get("QA", {}).items():
        if completed(existing_qas.get(str(qa_id), {}), qa):
            output["QA"][str(qa_id)] = copy.deepcopy(existing_qas[str(qa_id)])
    if not pending:
        return output
    pdf_path = resolve_pdf_path(paper_id, [args.pdf_dir])
    images, render_meta = render_pdf(pdf_path)
    for qa_id, qa in pending:
        started = time.time()
        row = copy.deepcopy(qa)
        try:
            raw, text = call_claude(
                key=key,
                model=args.model,
                question=str(qa["question"]),
                images=images,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            parsed = parse_json_response(text)
            answer, pages, normalizations = validate_prediction(
                parsed, render_meta["pdf_total_pages"]
            )
            row.update(
                {
                    "status": "completed",
                    "failed": False,
                    "answer_pre": text,
                    "answer_pre_raw": text,
                    "prediction_answer": answer,
                    "evidence_pages_pre": pages,
                    "prediction_parse_status": "ok",
                    "deterministic_normalizations": normalizations,
                    "api_provider": "anthropic_messages",
                    "api_model_requested": args.model,
                    "api_model_returned": raw.get("model"),
                    "api_request_id": raw.get("id"),
                    "api_stop_reason": raw.get("stop_reason"),
                    "api_usage": raw.get("usage"),
                    "gold_fields_sent": False,
                    "elapsed_seconds": round(time.time() - started, 4),
                    "pdf_path_resolved": str(pdf_path.resolve()),
                    **render_meta,
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "failed": True,
                    "prediction_answer": "",
                    "evidence_pages_pre": [],
                    "gold_fields_sent": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(time.time() - started, 4),
                    "pdf_path_resolved": str(pdf_path.resolve()),
                    **render_meta,
                }
            )
        output["QA"][qa_id] = row
    return output


def main() -> int:
    args = parse_args()
    source = read_json(args.qa_json)
    output = read_json(args.output_json) if args.output_json.is_file() else {}
    key = load_key(args.api_key_file)
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                evaluate_paper,
                str(paper_id),
                paper,
                output.get(str(paper_id), {}),
                args,
                key,
            ): str(paper_id)
            for paper_id, paper in source.items()
        }
        completed_count = 0
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                paper_output = future.result()
            except Exception as exc:
                failures.append((paper_id, f"{type(exc).__name__}: {exc}"))
                continue
            with _WRITE_LOCK:
                output[paper_id] = paper_output
                atomic_json(args.output_json, output)
            completed_count += len(paper_output.get("QA", {}))
            print(json.dumps({"paper_id": paper_id, "completed_total": completed_count}), flush=True)
    if failures:
        print(json.dumps({"paper_failures": failures}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
