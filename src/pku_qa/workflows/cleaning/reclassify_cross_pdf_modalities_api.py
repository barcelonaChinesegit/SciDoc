#!/usr/bin/env python3
"""Reclassify Cross-PDF QA modalities from evidence-page images via API."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import threading
from collections import Counter
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
from pku_qa.workflows.selection.final_2200_contract import CANONICAL_MODALITIES
from pku_qa.workflows.selection.normalize_final_2200_release import atomic_json
from pku_qa.workflows.selection.sync_final_2200_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QA = ROOT / "data/qa/7.final_2200/cross_pdf_qa.json"
DEFAULT_PDF_DIR = ROOT / "data/pdfs"
DEFAULT_AUDIT = ROOT / "data/qa/6.review/final_2200_modalities/work__cross_pdf__modality_api_audit__n800__v1.json"
DEFAULT_MANIFEST = ROOT / "data/qa/7.final_2200/rel__collection__final_2200__manifest.json"
REVIEW_VERSION = "2026-09-11.1"
WRITE_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()

SYSTEM_PROMPT = """You audit modality labels for a scientific PDF QA benchmark.
Inspect the question, gold answer, and only the cited evidence-page images for
each item. Ignore the existing modal_types value.

Return every modality that is actually necessary to derive or verify the gold
answer from the cited pages:
- text: prose, headings, captions, legends, or written labels are necessary.
- image: a plot, chart, diagram, photograph, or other spatial visual encoding is
  necessary; a caption merely mentioning a figure is not enough.
- table: row/column/cell structure or values read from a table are necessary.
- formula: a displayed or inline mathematical expression, symbolic relation, or
  derivation is necessary; a number repeated in prose is not enough.

A page may contain a visual element without that modality being necessary for
the answer. Include multiple modalities when the answer depends on all of them.
Use only text, image, table, formula, in that order. Return JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--pdf-dir", type=Path, action="append")
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, default=0, help="Limit bundle count for an API smoke test.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def qa_fingerprint(qa: dict[str, Any]) -> str:
    payload = {
        "question": qa.get("question"),
        "answer": qa.get("answer"),
        "evidence_pages": qa.get("evidence_pages"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def expected_output(bundle_id: str, qas: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundle_id": bundle_id,
        "items": [
            {
                "qa_id": qa_id,
                "modal_types": ["text", "image", "table", "formula"],
                "confidence": "number from 0 to 1",
                "rationale": "concise reason these modalities are necessary",
                "evidence": [
                    {
                        "page": "one cited merged-PDF page number",
                        "modalities": ["text", "table"],
                        "reason": "what answer-critical information is read here",
                    }
                ],
            }
            for qa_id in qas
        ],
    }


def build_prompt(
    bundle_id: str,
    qas: dict[str, Any],
    review_pages: dict[str, list[int]] | None = None,
) -> str:
    payload = []
    for qa_id, qa in qas.items():
        payload.append(
            {
                "qa_id": qa_id,
                "question": qa["question"],
                "gold_answer": qa["answer"],
                "evidence_pages": qa["evidence_pages"],
                "pages_supplied_for_modality_review": (
                    review_pages[qa_id] if review_pages is not None else qa["evidence_pages"]
                ),
            }
        )
    return (
        f"BUNDLE: {bundle_id}\n\nQA ITEMS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nREQUIRED OUTPUT SHAPE:\n"
        + json.dumps(expected_output(bundle_id, qas), ensure_ascii=False, indent=2)
        + "\n\nEvery QA ID must appear exactly once. A QA may use only its own "
        "pages_supplied_for_modality_review. For an Unanswerable item, inspect the "
        "supplied whole PDF and classify the information form that the question asks "
        "the model to search or verify; do not default it to text merely because no "
        "answer or gold evidence page exists."
    )


def render_pages(pdf_path: Path, pages: list[int]) -> list[tuple[int, str]]:
    rendered: list[tuple[int, str]] = []
    # PDFium's document/page loader is not thread-safe across these merged PDFs.
    with RENDER_LOCK:
        document = pdfium.PdfDocument(pdf_path)
        try:
            page_count = len(document)
            for page in pages:
                if page < 1 or page > page_count:
                    raise ValueError(f"Evidence page {page} is outside {pdf_path} ({page_count} pages)")
                image = document[page - 1].render(scale=1.6).to_pil().convert("RGB")
                image.thumbnail((1400, 1400))
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=72, optimize=True)
                rendered.append((page, base64.b64encode(buffer.getvalue()).decode("ascii")))
        finally:
            document.close()
    return rendered


def resolve_pdf(bundle_id: str, pdf_dirs: list[Path]) -> Path:
    return resolve_pdf_path(bundle_id, pdf_dirs)


def page_count(pdf_path: Path) -> int:
    with RENDER_LOCK:
        document = pdfium.PdfDocument(pdf_path)
        try:
            return len(document)
        finally:
            document.close()


def call_api(
    *, key: str, model: str, prompt: str, images: list[tuple[int, str]],
    max_tokens: int, timeout: int, retries: int,
) -> tuple[dict[str, Any], str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for page, encoded in images:
        content.extend(
            [
                {"type": "text", "text": f"[Merged PDF Page {page}]"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encoded,
                    },
                },
            ]
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
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
        },
        timeout,
        retries,
    )
    raw = response.json()
    return raw, response_text_claude(raw)


def validate_response(
    bundle_id: str,
    qas: dict[str, Any],
    response: dict[str, Any],
    review_pages: dict[str, list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    if response.get("bundle_id") != bundle_id:
        raise ValueError("response bundle_id mismatch")
    raw_items = response.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("response items must be an array")
    items: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("response item must be an object")
        qa_id = str(item.get("qa_id", ""))
        if qa_id not in qas or qa_id in items:
            raise ValueError(f"unexpected or duplicate qa_id: {qa_id}")
        modalities = item.get("modal_types")
        if not isinstance(modalities, list) or not modalities:
            raise ValueError(f"{qa_id}: modal_types must be a non-empty array")
        if any(value not in CANONICAL_MODALITIES for value in modalities):
            raise ValueError(f"{qa_id}: invalid modal_types")
        normalized = [value for value in CANONICAL_MODALITIES if value in modalities]
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"{qa_id}: invalid confidence")
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{qa_id}: missing rationale")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"{qa_id}: missing modality evidence")
        allowed_pages = set(
            review_pages[qa_id] if review_pages is not None else qas[qa_id]["evidence_pages"]
        )
        supported_modalities: set[str] = set()
        cited_evidence = []
        for row in evidence:
            if not isinstance(row, dict):
                continue
            raw_page = row.get("page")
            page = int(raw_page) if isinstance(raw_page, str) and raw_page.isdigit() else raw_page
            if page not in allowed_pages:
                continue
            row_modalities = row.get("modalities")
            if not isinstance(row_modalities, list):
                continue
            normalized_row_modalities = [
                value for value in normalized if value in row_modalities
            ]
            if not normalized_row_modalities:
                continue
            normalized_row = copy.deepcopy(row)
            normalized_row["page"] = page
            normalized_row["modalities"] = normalized_row_modalities
            supported_modalities.update(normalized_row_modalities)
            cited_evidence.append(normalized_row)
        if not cited_evidence:
            raise ValueError(f"{qa_id}: no cited page-level modality evidence")
        normalized_item = copy.deepcopy(item)
        normalized_item["modal_types"] = normalized
        normalized_item["evidence"] = cited_evidence
        normalized_item["page_support_complete"] = supported_modalities == set(normalized)
        normalized_item["page_support_missing_modalities"] = [
            value for value in normalized if value not in supported_modalities
        ]
        if not normalized_item["page_support_complete"]:
            raise ValueError(f"{qa_id}: not every modality has page-level support")
        items[qa_id] = normalized_item
    if set(items) != set(qas):
        raise ValueError(f"response omitted QA IDs: {sorted(set(qas) - set(items))}")
    return items


def completed(entry: dict[str, Any], qa: dict[str, Any]) -> bool:
    return (
        entry.get("status") == "completed"
        and entry.get("review_version") == REVIEW_VERSION
        and entry.get("input_fingerprint") == qa_fingerprint(qa)
        and entry.get("page_support_complete") is True
    )


def review_bundle(
    bundle_id: str, paper: dict[str, Any], args: argparse.Namespace, key: str
) -> dict[str, Any]:
    qas = paper["QA"]
    pdf_path = resolve_pdf(bundle_id, args.pdf_dir)
    total_pages = page_count(pdf_path)
    review_pages = {
        qa_id: (
            list(qa["evidence_pages"])
            if qa["evidence_pages"]
            else list(range(1, total_pages + 1))
        )
        for qa_id, qa in qas.items()
    }
    pages = sorted({page for values in review_pages.values() for page in values})
    images = render_pages(pdf_path, pages)
    prompt = build_prompt(bundle_id, qas, review_pages)
    last_error: Exception | None = None
    for _ in range(2):
        try:
            raw, response_text = call_api(
                key=key,
                model=args.model,
                prompt=prompt,
                images=images,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            items = validate_response(
                bundle_id, qas, parse_json_response(response_text), review_pages
            )
            return {
                "status": "completed",
                "review_version": REVIEW_VERSION,
                "model_requested": args.model,
                "model_returned": raw.get("model"),
                "api_request_id": raw.get("id"),
                "api_usage": raw.get("usage"),
                "page_count_sent": len(images),
                "items": {
                    qa_id: {
                        **item,
                        "input_fingerprint": qa_fingerprint(qas[qa_id]),
                        "status": "completed",
                        "review_version": REVIEW_VERSION,
                    }
                    for qa_id, item in items.items()
                },
            }
        except Exception as exc:  # retain the bundle for a resumable retry
            last_error = exc

    # Some bundle responses accidentally attach one QA's explanation to another
    # QA's page. Isolate each item and send only its own cited pages as a strict
    # fallback while preserving one bundle-level checkpoint.
    isolated_items: dict[str, dict[str, Any]] = {}
    request_ids: list[str] = []
    model_returned: list[str] = []
    usages: list[dict[str, Any]] = []
    for qa_id, qa in qas.items():
        isolated_qas = {qa_id: qa}
        isolated_images = [row for row in images if row[0] in set(review_pages[qa_id])]
        isolated_error: Exception | None = None
        for _ in range(2):
            try:
                raw, response_text = call_api(
                    key=key,
                    model=args.model,
                    prompt=build_prompt(
                        bundle_id,
                        isolated_qas,
                        {qa_id: review_pages[qa_id]},
                    ),
                    images=isolated_images,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                item = validate_response(
                    bundle_id,
                    isolated_qas,
                    parse_json_response(response_text),
                    {qa_id: review_pages[qa_id]},
                )[qa_id]
                isolated_items[qa_id] = item
                if raw.get("id"):
                    request_ids.append(str(raw["id"]))
                if raw.get("model"):
                    model_returned.append(str(raw["model"]))
                if isinstance(raw.get("usage"), dict):
                    usages.append(raw["usage"])
                break
            except Exception as exc:
                isolated_error = exc
        else:
            raise RuntimeError(
                f"{bundle_id}/{qa_id}: {type(isolated_error).__name__}: {isolated_error}; "
                f"bundle_error={type(last_error).__name__}: {last_error}"
            )
    return {
        "status": "completed",
        "review_version": REVIEW_VERSION,
        "model_requested": args.model,
        "model_returned": sorted(set(model_returned)),
        "api_request_id": request_ids,
        "api_usage": usages,
        "page_count_sent": sum(len(review_pages[qa_id]) for qa_id in qas),
        "isolated_qa_fallback": True,
        "items": {
            qa_id: {
                **item,
                "input_fingerprint": qa_fingerprint(qas[qa_id]),
                "status": "completed",
                "review_version": REVIEW_VERSION,
            }
            for qa_id, item in isolated_items.items()
        },
    }


def audit_items(audit: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for bundle_id, bundle in audit.get("bundles", {}).items():
        if not isinstance(bundle, dict):
            continue
        for qa_id, item in bundle.get("items", {}).items():
            if isinstance(item, dict):
                result[(str(bundle_id), str(qa_id))] = item
    return result


def refresh_page_support_metadata(audit: dict[str, Any]) -> bool:
    changed = False
    for item in audit_items(audit).values():
        modalities = set(item.get("modal_types", []))
        supported = {
            modality
            for row in item.get("evidence", [])
            if isinstance(row, dict)
            for modality in row.get("modalities", [])
            if modality in CANONICAL_MODALITIES
        }
        complete = modalities <= supported
        missing = [
            value for value in CANONICAL_MODALITIES
            if value in modalities and value not in supported
        ]
        if item.get("page_support_complete") != complete:
            item["page_support_complete"] = complete
            changed = True
        if item.get("page_support_missing_modalities") != missing:
            item["page_support_missing_modalities"] = missing
            changed = True
    return changed


def apply_decisions(
    dataset: dict[str, Any],
    audit: dict[str, Any],
    *,
    audit_path: Path = DEFAULT_AUDIT,
) -> Counter[str]:
    decisions = audit_items(audit)
    counts: Counter[str] = Counter()
    for bundle_id, paper in dataset.items():
        for qa_id, qa in paper["QA"].items():
            item = decisions.get((str(bundle_id), str(qa_id)))
            if item is None or not completed(item, qa):
                raise ValueError(f"Missing current modality decision for {bundle_id}/{qa_id}")
            previous = list(qa["modal_types"])
            modalities = list(item["modal_types"])
            qa["modal_types"] = modalities
            provenance = qa.setdefault("annotation_provenance", {})
            existing_review = provenance.get("modality_api_review", {})
            original_modalities = (
                existing_review.get("previous_modal_types", previous)
                if isinstance(existing_review, dict)
                else previous
            )
            provenance["modality_api_review"] = {
                "version": REVIEW_VERSION,
                "model": audit["model"],
                "previous_modal_types": original_modalities,
                "modal_types": modalities,
                "confidence": item["confidence"],
                "rationale": item["rationale"],
                "evidence": item["evidence"],
                "page_support_complete": item.get("page_support_complete", False),
                "page_support_missing_modalities": item.get(
                    "page_support_missing_modalities", []
                ),
                "audit_file": str(audit_path.relative_to(ROOT)),
            }
            normalization = provenance.get("final_2200_core_normalization")
            if isinstance(normalization, dict):
                normalization["modal_types_method"] = "vision_api_evidence_page_review"
                normalization.setdefault("original_modal_types", previous)
            counts["changed" if previous != modalities else "unchanged"] += 1
            counts.update(modalities)
    return counts


def main() -> int:
    args = parse_args()
    if not args.pdf_dir:
        args.pdf_dir = [DEFAULT_PDF_DIR]
    dataset = read_json(args.qa_json)
    audit = read_json(args.audit_json) if args.audit_json.is_file() else {
        "schema_version": 1,
        "review_version": REVIEW_VERSION,
        "source_path": str(args.qa_json.resolve()),
        "model": args.model,
        "bundles": {},
    }
    if audit.get("review_version") != REVIEW_VERSION or audit.get("model") != args.model:
        if not args.force:
            raise ValueError("Existing audit uses a different review contract or model; pass --force")
        audit = {
            "schema_version": 1,
            "review_version": REVIEW_VERSION,
            "source_path": str(args.qa_json.resolve()),
            "model": args.model,
            "bundles": {},
        }
    if refresh_page_support_metadata(audit):
        atomic_json(args.audit_json, audit)

    bundle_ids = list(dataset)
    if args.sample:
        bundle_ids = bundle_ids[: args.sample]
    pending = []
    for bundle_id in bundle_ids:
        existing = audit.get("bundles", {}).get(bundle_id, {})
        existing_items = existing.get("items", {}) if isinstance(existing, dict) else {}
        if args.force or any(
            not completed(existing_items.get(qa_id, {}), qa)
            for qa_id, qa in dataset[bundle_id]["QA"].items()
        ):
            pending.append(bundle_id)

    key = load_key(args.api_key_file) if pending else ""
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(review_bundle, bundle_id, dataset[bundle_id], args, key): bundle_id
            for bundle_id in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            bundle_id = futures[future]
            try:
                result = future.result()
                with WRITE_LOCK:
                    audit.setdefault("bundles", {})[bundle_id] = result
                    atomic_json(args.audit_json, audit)
                print(f"[modality-review] {index}/{len(pending)} {bundle_id} completed", flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(f"[modality-review] {index}/{len(pending)} {bundle_id} failed: {exc}", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} bundle reviews failed; rerun to resume")
    reviewed = audit_items(audit)
    expected = sum(len(paper["QA"]) for paper in dataset.values())
    current = sum(
        completed(reviewed.get((str(bundle_id), str(qa_id)), {}), qa)
        for bundle_id, paper in dataset.items()
        for qa_id, qa in paper["QA"].items()
    )
    if args.apply:
        if current != expected:
            raise ValueError(f"Cannot apply an incomplete audit: {current}/{expected}")
        updated = copy.deepcopy(dataset)
        counts = apply_decisions(updated, audit)
        atomic_json(args.qa_json, updated)
        atomic_json(args.manifest, build_manifest(args.qa_json.parent))
        print(json.dumps({"status": "applied", "qa_count": expected, "modalities": counts}, ensure_ascii=False))
    else:
        print(json.dumps({"status": "audited", "qa_count": current, "expected": expected}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
