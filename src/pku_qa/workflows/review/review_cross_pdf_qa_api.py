#!/usr/bin/env python3
"""Review cross-paper QA bundles with a remote Gemini or Claude API.

The script never writes or prints an API key. Put the school key in the
Git-ignored workspace `.env` as `AICODEMIRROR_API_KEY=...`; an optional
`AICODEMIRROR_API_KEY_CLASSMATE=...` can be selected with
`AICODEMIRROR_API_KEY_PROFILE=school|classmate|auto`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import requests
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QA = ROOT / "data/qa/qa_cross_pdf_manual_verified_500_20260724.json"
DEFAULT_ALL_QA = ROOT / "data/qa/qa_expansion_cross_pdf_cleaned_20260711.json"
DEFAULT_MANIFEST = (
    ROOT / "data/qa_generation/expansion_20260711/cross_pdf_bundle_manifest.json"
)
DEFAULT_KEY_FILE = ROOT / ".env"
DEFAULT_OUTPUT = ROOT / "data/qa/4.cross_pdf/semantic_reaudit/api_reviews"

CLAUDE_BASE_URL = "https://api.aicodemirror.ai/api/claudecode"
GEMINI_BASE_URL = "https://api.aicodemirror.ai/api/gemini"


class ApiQuotaExhaustedError(RuntimeError):
    """The account needs a manual balance/quota renewal before more calls."""


API_QUOTA_PATTERNS = (
    re.compile(r"insufficient[_ -]?quota", re.IGNORECASE),
    re.compile(r"insufficient (?:account )?(?:balance|credit)", re.IGNORECASE),
    re.compile(r"(?:account|credit) balance (?:is )?(?:insufficient|exhausted)", re.IGNORECASE),
    re.compile(r"(?:monthly|billing) (?:quota|limit).*(?:exhausted|exceeded|reached)", re.IGNORECASE),
    re.compile(r"current quota.*(?:plan|billing)", re.IGNORECASE),
    re.compile(r"quota (?:is |has been )?exhausted", re.IGNORECASE),
    re.compile(r"(?:余额不足|额度不足|额度(?:已)?(?:用完|耗尽)|欠费)"),
)

API_KEY_PROFILE_ENV = "AICODEMIRROR_API_KEY_PROFILE"
API_KEY_ENV_BY_PROFILE = {
    "school": "AICODEMIRROR_API_KEY",
    "classmate": "AICODEMIRROR_API_KEY_CLASSMATE",
}
API_KEY_PROFILES = ("school", "classmate", "auto")
_ACTIVE_API_KEYS: tuple[tuple[str, str], ...] = ()
_EXHAUSTED_API_KEY_PROFILES: set[str] = set()
_API_KEY_STATE_LOCK = threading.Lock()


def api_quota_exhausted(status_code: int, detail: str) -> bool:
    """Distinguish manual account renewal from transient 429 rate limits."""
    if status_code == 402:
        return True
    return any(pattern.search(detail or "") for pattern in API_QUOTA_PATTERNS)

ALLOWED_DECISIONS = {"KEEP", "FIX", "REJECT"}
CRITERIA = (
    "cross_document_required",
    "question_coherent",
    "question_specific",
    "answer_correct",
    "answer_complete",
    "evidence_sufficient",
    "no_unsupported_claim",
)

SYSTEM_PROMPT = """\
You are the senior reviewer of a multi-paper scientific QA benchmark.
Read every supplied source-paper page, then inspect every QA item separately.
Be skeptical: fluent wording is not evidence of correctness.

The benchmark is intended to test meaningful synthesis across multiple papers.
An item is acceptable only when all of the following are true:
1. A complete answer necessarily uses substantive facts from at least two
   distinct source papers. Reading any single paper alone is insufficient.
2. The question is coherent, specific, scientifically useful, and not an
   arbitrary juxtaposition of unrelated terms, abstracts, or definitions.
3. The answer directly answers the question and is factually correct,
   complete, and precise according to the supplied papers.
4. Every material answer claim is supported by the cited evidence pages.
5. Evidence pages cover every source paper actually needed for the answer.
6. The item contains no invented causal relation, unsupported negative claim,
   exaggerated contrast, false equivalence, wrong number, wrong mechanism, or
   misleading terminology.

Decision policy:
- KEEP: all seven criteria are true without changing the item.
- FIX: the question is genuinely valuable and cross-document, and only a
  small, unambiguous correction to wording, answer, or evidence pages is
  needed. Supply each changed field and use null explicitly for unchanged
  fields. At least one corrected field must be non-null.
- REJECT: any core premise is wrong; the comparison is arbitrary or trivial;
  the item is not truly cross-document; the answer needs substantial
  reconstruction; or the evidence cannot establish the answer.

Do not rescue a weak item merely to increase the retained count. Do not infer
facts from outside the supplied papers. Use merged PDF page numbers when
reporting support. Return JSON only, with no markdown or commentary.
The corrected_evidence_pages field must be either null or a JSON array of
positive integers such as [5, 24, 36]. Never return page numbers as a string
or as nested objects.
"""

HARD_REVIEW_PROMPT = """\

HARD-EXPANSION ADDITIONAL POLICY:
- Reject one-sentence-per-document stitching, parallel subquestions, and an
  answer that merely places source facts side by side.
- A scientifically related comparison is not automatically stitching. It may
  be KEEP when facts from every cited paper are needed to derive one new
  taxonomy, selection rule, invariant, incompatibility, or conflict resolution
  that is not stated in any paper alone. Do not reject merely because each
  premise is individually documented; verify whether the unified conclusion
  itself depends on all premises.
- A Cross-PDF benchmark question may explicitly impose a target requirement,
  selection criterion, or transfer setting; the source papers do not need to
  have proposed a joint system or cited one another. Such an item may be KEEP
  when the question clearly states the external constraint and facts from every
  cited paper deterministically yield one compatibility decision, filtered
  choice, or conflict resolution. Do not call this "fabricated" merely because
  the benchmark performs the cross-paper application. Reject it only when the
  constraint is absent from the question, the answer invents an unstated source
  property, or the result remains parallel per-paper mapping rather than one
  decision.
- For `compatibility_judgment`, `method_transfer`, and `conflict_resolution`,
  treat an explicit constraint in the question as a legitimate benchmark
  premise, not as a factual claim that the papers built a joint system. If all
  method properties are verified and they deterministically produce one
  compatible/incompatible choice, transfer verdict, or resolution, mark the
  coherence and cross-document criteria true even when the papers study
  different application domains. Do not reject such an item merely as an
  "arbitrary application scenario" or because the scenario itself is absent
  from the papers; the cross-paper inference is the object being tested.
- For an item declaring three source documents, independently verify that all
  three contribute a necessary fact to one inference. Two-paper reasoning plus
  an incidental third-paper citation is REJECT.
- Prefer genuine adjacent-module dependencies, metric reasoning, component
  hierarchy, method transfer, compatibility judgment, or conflict resolution.
- Verify the intermediate-fact ledger against the full papers and require the
  final answer to state the answer-critical bridge facts.
"""


@dataclass(frozen=True)
class Source:
    doc_number: int
    title: str
    paper_id: str
    path: Path
    merged_start: int
    merged_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=("gemini", "claude"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--bundle-id", action="append", default=[])
    parser.add_argument("--bundle-list", type=Path)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260724)
    parser.add_argument("--max-output-tokens", type=int, default=14000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--api-workers", type=int, default=1)
    parser.add_argument("--worker-id")
    parser.add_argument("--quota-retries", type=int, default=0)
    parser.add_argument("--quota-retry-wait-seconds", type=int, default=45)
    parser.add_argument("--include-evidence-images", action="store_true")
    parser.add_argument(
        "--evidence-context-only",
        action="store_true",
        help=(
            "Send cited evidence pages plus nearby context instead of all source "
            "pages; useful for a second reviewer with request-token limits."
        ),
    )
    parser.add_argument("--context-pages", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hard-mode",
        action="store_true",
        help="Apply hard-expansion anti-stitching and all-document-necessity gates.",
    )
    parser.add_argument(
        "--revalidate-existing",
        action="store_true",
        help="Re-run local schema validation on saved reviews without an API call.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def single_api_worker_argv(
    argv: list[str], worker_index: int, bundle_list: Path
) -> list[str]:
    """Build one child argv after resolving all parent-side bundle selection."""
    value_options = {
        "--api-workers",
        "--worker-id",
        "--bundle-id",
        "--bundle-list",
        "--sample",
        "--sample-seed",
    }
    result: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in value_options:
            index += 2
            continue
        if any(value.startswith(option + "=") for option in value_options):
            index += 1
            continue
        result.append(value)
        index += 1
    result.extend(
        (
            "--api-workers",
            "1",
            "--worker-id",
            f"cross-review-api-{worker_index}",
            "--bundle-list",
            str(bundle_list),
        )
    )
    return result


def run_parallel_api_workers(args: argparse.Namespace) -> int:
    qa = load_json(args.qa_json)
    if not isinstance(qa, dict):
        raise SystemExit("QA JSON must be an object keyed by bundle id.")
    bundle_ids = select_bundles(args, qa)
    worker_count = min(args.api_workers, len(bundle_ids))
    if worker_count <= 0:
        return 0
    shard_dir = (
        args.output_dir
        / ".review_shards"
        / safe_model_dir(args.provider, args.model)
    ).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for worker_index in range(worker_count):
        shard = bundle_ids[worker_index::worker_count]
        path = shard_dir / f"worker_{worker_index:02d}.txt"
        path.write_text("".join(f"{bundle_id}\n" for bundle_id in shard), encoding="utf-8")
        shard_paths.append(path)
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *single_api_worker_argv(sys.argv[1:], worker_index, shard_path),
            ]
        )
        for worker_index, shard_path in enumerate(shard_paths)
    ]
    stopping = False

    def stop_children() -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def handle_signal(signum: int, _frame: Any) -> None:
        stop_children()
        raise SystemExit(128 + signum)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        exit_codes = [process.wait() for process in processes]
    finally:
        stop_children()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    failures = [
        {"worker_index": index, "exit_code": code}
        for index, code in enumerate(exit_codes)
        if code != 0
    ]
    if failures:
        print(
            "Cross-PDF review workers failed: "
            + json.dumps(failures, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    return 0


def _key_file_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    values: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        values[name.strip()] = value.strip().strip("\"'")
    if not values and "\n" not in raw and "=" not in raw:
        values["AICODEMIRROR_API_KEY"] = raw.strip()
    return values


def load_key(path: Path) -> str:
    """Load the selected key and configure quota-only failover for API calls."""
    values = _key_file_values(path)
    for name in (*API_KEY_ENV_BY_PROFILE.values(), API_KEY_PROFILE_ENV):
        environment_value = os.environ.get(name, "").strip()
        if environment_value:
            values[name] = environment_value

    profile = values.get(API_KEY_PROFILE_ENV, "school").strip().casefold()
    if profile not in API_KEY_PROFILES:
        raise SystemExit(
            f"Invalid {API_KEY_PROFILE_ENV}={profile!r}; choose school, "
            "classmate, or auto."
        )
    requested_profiles = (
        ("school", "classmate") if profile == "auto" else (profile,)
    )
    configured: list[tuple[str, str]] = []
    for profile_name in requested_profiles:
        environment_name = API_KEY_ENV_BY_PROFILE[profile_name]
        key = values.get(environment_name, "").strip()
        if not key:
            if profile == "auto":
                continue
            raise SystemExit(
                f"API key profile {profile_name!r} is unavailable. Set "
                f"{environment_name} or add it to {path}."
            )
        if any(ch.isspace() for ch in key):
            raise SystemExit(
                f"API key profile {profile_name!r} contains whitespace; "
                "refusing to use it."
            )
        if all(existing_key != key for _, existing_key in configured):
            configured.append((profile_name, key))
    if not configured:
        expected = " or ".join(API_KEY_ENV_BY_PROFILE.values())
        raise SystemExit(
            f"API key unavailable. Set {expected}, or create {path} with mode 0600."
        )

    global _ACTIVE_API_KEYS
    with _API_KEY_STATE_LOCK:
        _ACTIVE_API_KEYS = tuple(configured)
        _EXHAUSTED_API_KEY_PROFILES.clear()
    if profile == "auto" and len(configured) > 1:
        print("API key profile: auto (school, then classmate on quota exhaustion)")
    else:
        print(f"API key profile: {configured[0][0]}")
    return configured[0][1]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def valid_unicode(text: str) -> str:
    """Replace lone UTF-16 surrogates sometimes emitted by PDF extractors."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def manifest_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError("Bundle manifest must be a JSON list.")
    return {str(row["id"]): row for row in rows}


def source_rows(bundle: dict[str, Any]) -> list[Source]:
    rows: list[Source] = []
    for number, raw in enumerate(bundle["sources"], start=1):
        path = Path(raw["source_pdf_path"])
        if not path.is_absolute():
            path = ROOT / path
        rows.append(
            Source(
                doc_number=number,
                title=str(raw["title"]),
                paper_id=str(raw["paper_id"]),
                path=path,
                merged_start=int(raw["merged_start_page"]),
                merged_end=int(raw["merged_end_page"]),
            )
        )
    return rows


def extract_source_text(source: Source) -> tuple[str, int, list[str]]:
    reader = PdfReader(source.path)
    pages: list[str] = []
    errors: list[str] = []
    for offset, page in enumerate(reader.pages):
        merged_page = source.merged_start + offset
        try:
            text = valid_unicode(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - depends on malformed PDFs
            text = ""
            errors.append(f"page {merged_page}: {type(exc).__name__}")
        pages.append(
            f"\n===== Doc {source.doc_number}, merged PDF page "
            f"{merged_page}, source page {offset + 1} =====\n{text.strip()}\n"
        )
    return "".join(pages), len(reader.pages), errors


def selected_context_pages(
    bundle_qa: dict[str, Any],
    sources: list[Source],
    context_pages: int,
) -> set[int]:
    """Return cited merged pages expanded within each source boundary."""
    cited = {
        int(page)
        for qa in bundle_qa.get("QA", {}).values()
        for page in qa.get("evidence_pages", [])
    }
    selected: set[int] = set()
    for page in cited:
        matches = [
            source
            for source in sources
            if source.merged_start <= page <= source.merged_end
        ]
        if len(matches) != 1:
            raise ValueError(f"Merged evidence page {page} maps to {len(matches)} sources")
        source = matches[0]
        selected.update(
            range(
                max(source.merged_start, page - context_pages),
                min(source.merged_end, page + context_pages) + 1,
            )
        )
    return selected


def extract_source_context(
    source: Source, selected_merged_pages: set[int]
) -> tuple[str, int, list[str]]:
    reader = PdfReader(source.path)
    pages: list[str] = []
    errors: list[str] = []
    selected = sorted(
        page
        for page in selected_merged_pages
        if source.merged_start <= page <= source.merged_end
    )
    for merged_page in selected:
        offset = merged_page - source.merged_start
        try:
            text = valid_unicode(reader.pages[offset].extract_text() or "")
        except Exception as exc:  # pragma: no cover - malformed PDFs
            text = ""
            errors.append(f"page {merged_page}: {type(exc).__name__}")
        pages.append(
            f"\n===== Doc {source.doc_number}, merged PDF page {merged_page}, "
            f"source page {offset + 1} =====\n{text.strip()}\n"
        )
    return "".join(pages), len(selected), errors


def qa_payload(bundle_id: str, bundle_qa: dict[str, Any]) -> list[dict[str, Any]]:
    payload = []
    for qa_id, qa in bundle_qa.get("QA", {}).items():
        payload.append(
            {
                "qa_id": qa_id,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence_pages": qa.get("evidence_pages", []),
                "modal_types": qa.get("modal_types", []),
                "question_type": qa.get("question_type"),
                "question_category": qa.get("question_category"),
                "source_qa_ids": qa.get("source_qa_ids"),
                "source_doc_numbers": qa.get("source_doc_numbers"),
                "source_document_count": qa.get("source_document_count"),
                "reasoning_type": qa.get("reasoning_type"),
                "relation": qa.get("relation"),
                "derived_conclusion": qa.get("derived_conclusion"),
                "intermediate_facts": qa.get("intermediate_facts"),
            }
        )
    if not payload:
        raise ValueError(f"{bundle_id} has no QA items")
    return payload


def expected_schema(bundle_id: str, qa_ids: list[str]) -> dict[str, Any]:
    return {
        "bundle_id": bundle_id,
        "bundle_summary": (
            "One short sentence describing whether the bundle topics form a "
            "reasonable basis for cross-paper questions."
        ),
        "items": [
            {
                "qa_id": qa_id,
                "decision": "KEEP | FIX | REJECT",
                "criteria": {name: "boolean" for name in CRITERIA},
                "confidence": "number from 0 to 1",
                "severity": "none | low | medium | high | critical",
                "reason": "Specific explanation grounded in the supplied papers.",
                "support": [
                    {
                        "doc_number": "integer",
                        "merged_page": "integer",
                        "supported_fact": "Concise paraphrase of the supporting fact.",
                    }
                ],
                "corrected_question": (
                    "For FIX: complete corrected question, or null if unchanged."
                ),
                "corrected_answer": (
                    "For FIX: complete corrected answer, or null if unchanged."
                ),
                "corrected_evidence_pages": (
                    "For FIX: complete corrected page list, or null if unchanged."
                ),
            }
            for qa_id in qa_ids
        ],
    }


def build_text_prompt(
    bundle_id: str,
    bundle_qa: dict[str, Any],
    bundle_manifest: dict[str, Any],
    hard_mode: bool = False,
    evidence_context_only: bool = False,
    context_pages: int = 1,
) -> tuple[str, dict[str, Any]]:
    sources = source_rows(bundle_manifest)
    context_selection = (
        selected_context_pages(bundle_qa, sources, max(0, context_pages))
        if evidence_context_only
        else set()
    )
    source_blocks = []
    source_meta = []
    extraction_errors: list[str] = []
    total_pages = 0
    for source in sources:
        if evidence_context_only:
            text, pages, errors = extract_source_context(
                source, context_selection
            )
        else:
            text, pages, errors = extract_source_text(source)
        total_pages += pages
        extraction_errors.extend(
            f"Doc {source.doc_number} {error}" for error in errors
        )
        source_meta.append(
            {
                "doc_number": source.doc_number,
                "paper_id": source.paper_id,
                "title": source.title,
                "merged_start_page": source.merged_start,
                "merged_end_page": source.merged_end,
                "source_page_count": pages,
                "extraction_errors": errors,
            }
        )
        source_blocks.append(
            f"\n\n######## DOC {source.doc_number}: {source.title} ########\n"
            f"paper_id={source.paper_id}; merged pages "
            f"{source.merged_start}-{source.merged_end}\n{text}"
        )

    qas = qa_payload(bundle_id, bundle_qa)
    schema = expected_schema(bundle_id, [row["qa_id"] for row in qas])
    prompt = (
        f"BUNDLE: {bundle_id}\n\n"
        "SOURCE MAP:\n"
        + json.dumps(source_meta, ensure_ascii=False, indent=2)
        + "\n\nQA ITEMS TO REVIEW:\n"
        + json.dumps(qas, ensure_ascii=False, indent=2)
        + "\n\nREQUIRED OUTPUT SHAPE:\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + (
            "\n\nSOURCE EVIDENCE CONTEXT FOLLOWS. Read every supplied page. "
            "The independent other reviewer receives full papers:"
            if evidence_context_only
            else "\n\nFULL SOURCE PAPERS FOLLOW. Read all pages before deciding:"
        )
        + "".join(source_blocks)
        + (HARD_REVIEW_PROMPT if hard_mode else "")
    )
    metadata = {
        "bundle_id": bundle_id,
        "source_count": len(sources),
        "source_pages": total_pages,
        "prompt_characters": len(prompt),
        "qa_count": len(qas),
        "qa_ids": [row["qa_id"] for row in qas],
        "source_meta": source_meta,
        "extraction_errors": extraction_errors,
        "evidence_context_only": evidence_context_only,
        "context_pages": context_pages if evidence_context_only else None,
    }
    return prompt, metadata


def map_merged_page(sources: list[Source], page: int) -> tuple[Source, int]:
    matches = [
        source for source in sources if source.merged_start <= page <= source.merged_end
    ]
    if len(matches) != 1:
        raise ValueError(f"Merged page {page} maps to {len(matches)} sources")
    source = matches[0]
    return source, page - source.merged_start


def evidence_images(
    bundle_qa: dict[str, Any],
    bundle_manifest: dict[str, Any],
) -> list[tuple[str, str]]:
    pages: set[int] = set()
    for qa in bundle_qa.get("QA", {}).values():
        modalities = set(qa.get("modal_types", []))
        if modalities - {"text"}:
            pages.update(int(page) for page in qa.get("evidence_pages", []))
    if not pages:
        return []

    sources = source_rows(bundle_manifest)
    open_pdfs: dict[Path, Any] = {}
    rendered: list[tuple[str, str]] = []
    try:
        for merged_page in sorted(pages):
            source, zero_based_page = map_merged_page(sources, merged_page)
            document = open_pdfs.setdefault(source.path, pdfium.PdfDocument(source.path))
            bitmap = document[zero_based_page].render(scale=1.5)
            image = bitmap.to_pil()
            if image.mode != "RGB":
                image = image.convert("RGB")
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=82, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            rendered.append(
                (
                    (
                        f"Evidence image for merged PDF page {merged_page} "
                        f"(Doc {source.doc_number}, {source.title})"
                    ),
                    encoded,
                )
            )
    finally:
        for document in open_pdfs.values():
            document.close()
    return rendered


def response_text_claude(payload: dict[str, Any]) -> str:
    blocks = payload.get("content", [])
    return "\n".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )


def response_text_gemini(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(
        str(part.get("text", "")) for part in parts if isinstance(part, dict)
    )


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Model response is not a JSON object")
    return value


def normalize_review_types(review: dict[str, Any]) -> list[str]:
    """Apply narrow, auditable normalizations to common API JSON type slips."""
    normalizations: list[str] = []
    items = review.get("items")
    if not isinstance(items, list):
        return normalizations
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("corrected_evidence_pages")
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        pages: list[int] | None = None
        if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", stripped):
            pages = [int(token.strip()) for token in stripped.split(",")]
        elif stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if (
                isinstance(parsed, list)
                and parsed
                and all(
                    isinstance(page, int)
                    and not isinstance(page, bool)
                    and page >= 1
                    for page in parsed
                )
            ):
                pages = parsed
        if pages and all(page >= 1 for page in pages):
            item["corrected_evidence_pages"] = pages
            normalizations.append(
                f"{item.get('qa_id', 'unknown')}:corrected_evidence_pages_string_to_list"
            )
    return normalizations


def validate_review(
    review: dict[str, Any], bundle_id: str, expected_qa_ids: list[str]
) -> list[str]:
    errors: list[str] = []
    if review.get("bundle_id") != bundle_id:
        errors.append("bundle_id mismatch")
    items = review.get("items")
    if not isinstance(items, list):
        return errors + ["items is not a list"]
    actual_ids = [str(item.get("qa_id")) for item in items if isinstance(item, dict)]
    if len(actual_ids) != len(set(actual_ids)):
        errors.append(f"duplicate qa_ids: {actual_ids}")
    if set(actual_ids) != set(expected_qa_ids):
        errors.append(
            f"qa_id set mismatch: expected {expected_qa_ids}, got {actual_ids}"
        )
    for item in items:
        if not isinstance(item, dict):
            errors.append("non-object item")
            continue
        qa_id = item.get("qa_id")
        if item.get("decision") not in ALLOWED_DECISIONS:
            errors.append(f"{qa_id}: invalid decision")
        criteria = item.get("criteria")
        if not isinstance(criteria, dict):
            errors.append(f"{qa_id}: missing criteria")
        else:
            for name in CRITERIA:
                if not isinstance(criteria.get(name), bool):
                    errors.append(f"{qa_id}: criterion {name} is not boolean")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"{qa_id}: invalid confidence")
        if not str(item.get("reason", "")).strip():
            errors.append(f"{qa_id}: empty reason")
        support = item.get("support")
        if not isinstance(support, list):
            errors.append(f"{qa_id}: support is not a list")
        else:
            for support_index, row in enumerate(support):
                prefix = f"{qa_id}: support[{support_index}]"
                if not isinstance(row, dict):
                    errors.append(f"{prefix} is not an object")
                    continue
                doc_number = row.get("doc_number")
                if (
                    not isinstance(doc_number, int)
                    or isinstance(doc_number, bool)
                    or doc_number < 1
                ):
                    errors.append(f"{prefix} has invalid doc_number")
                merged_page = row.get("merged_page")
                if (
                    not isinstance(merged_page, int)
                    or isinstance(merged_page, bool)
                    or merged_page < 1
                ):
                    errors.append(f"{prefix} has invalid merged_page")
                if not isinstance(row.get("supported_fact"), str) or not row[
                    "supported_fact"
                ].strip():
                    errors.append(f"{prefix} has empty supported_fact")

        corrected_question = item.get("corrected_question")
        if corrected_question is not None and (
            not isinstance(corrected_question, str)
            or not corrected_question.strip()
        ):
            errors.append(f"{qa_id}: invalid corrected_question")
        corrected_answer = item.get("corrected_answer")
        if corrected_answer is not None and (
            not isinstance(corrected_answer, str)
            or not corrected_answer.strip()
        ):
            errors.append(f"{qa_id}: invalid corrected_answer")
        corrected_pages = item.get("corrected_evidence_pages")
        if corrected_pages is not None and (
            not isinstance(corrected_pages, list)
            or not corrected_pages
            or not all(
                isinstance(page, int)
                and not isinstance(page, bool)
                and page >= 1
                for page in corrected_pages
            )
        ):
            errors.append(f"{qa_id}: invalid corrected_evidence_pages")
        if item.get("decision") == "FIX":
            corrected_values = [
                corrected_question,
                corrected_answer,
                corrected_pages,
            ]
            if all(value in (None, "", []) for value in corrected_values):
                errors.append(f"{qa_id}: FIX has no corrected field")
    return errors


def post_with_retries(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
    retries: int,
) -> requests.Response:
    credential_header = next(
        (
            name
            for name in ("x-api-key", "x-goog-api-key")
            if name in headers
        ),
        None,
    )
    request_headers = dict(headers)
    configured_profiles: list[tuple[str, str]] = []
    matching_index: int | None = None
    if credential_header is not None:
        supplied_key = request_headers[credential_header]
        with _API_KEY_STATE_LOCK:
            matching_index = next(
                (
                    index
                    for index, (_, key) in enumerate(_ACTIVE_API_KEYS)
                    if key == supplied_key
                ),
                None,
            )
            if matching_index is not None:
                configured_profiles = [
                    pair
                    for pair in _ACTIVE_API_KEYS[matching_index:]
                    if pair[0] not in _EXHAUSTED_API_KEY_PROFILES
                ]
        if matching_index is not None and not configured_profiles:
            raise ApiQuotaExhaustedError(
                "All configured API key profiles are exhausted."
            )
        if configured_profiles:
            request_headers[credential_header] = configured_profiles[0][1]

    active_profile_index = 0
    last_error: Exception | None = None
    attempt = 0
    while attempt < retries:
        try:
            response = requests.post(
                url,
                headers=dict(request_headers),
                json=body,
                timeout=timeout,
            )
            if response.status_code < 400:
                return response
            safe_detail = response.text[:2000]
            if api_quota_exhausted(response.status_code, safe_detail):
                if configured_profiles:
                    exhausted_profile = configured_profiles[active_profile_index][0]
                    with _API_KEY_STATE_LOCK:
                        _EXHAUSTED_API_KEY_PROFILES.add(exhausted_profile)
                    active_profile_index += 1
                    if active_profile_index < len(configured_profiles):
                        next_profile, next_key = configured_profiles[active_profile_index]
                        request_headers[credential_header] = next_key
                        print(
                            "API quota exhausted for key profile "
                            f"{exhausted_profile}; switching to {next_profile}.",
                            file=sys.stderr,
                        )
                        continue
                raise ApiQuotaExhaustedError(
                    f"HTTP {response.status_code}: {safe_detail}"
                )
            if response.status_code not in {
                408,
                409,
                429,
                500,
                502,
                503,
                504,
                520,
                522,
                523,
                524,
                525,
                526,
            }:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {safe_detail}"
                )
            last_error = RuntimeError(
                f"retryable HTTP {response.status_code}: {safe_detail}"
            )
        except requests.RequestException as exc:
            last_error = exc
        attempt += 1
        if attempt < retries:
            time.sleep(min(30, 2 ** (attempt - 1) + random.random()))
    raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")


def call_claude(
    key: str,
    model: str,
    prompt: str,
    images: list[tuple[str, str]],
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], str]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, encoded in images:
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encoded,
                },
            }
        )
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": content}],
    }
    response = post_with_retries(
        f"{CLAUDE_BASE_URL}/v1/messages",
        {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        body,
        timeout,
        retries,
    )
    raw = response.json()
    return raw, response_text_claude(raw)


def call_gemini(
    key: str,
    model: str,
    prompt: str,
    images: list[tuple[str, str]],
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[dict[str, Any], str]:
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for label, encoded in images:
        parts.append({"text": label})
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": encoded,
                }
            }
        )
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    response = post_with_retries(
        f"{GEMINI_BASE_URL}/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key, "content-type": "application/json"},
        body,
        timeout,
        retries,
    )
    raw = response.json()
    return raw, response_text_gemini(raw)


def select_bundles(args: argparse.Namespace, qa: dict[str, Any]) -> list[str]:
    requested = list(args.bundle_id)
    if args.bundle_list:
        requested.extend(
            line.strip()
            for line in args.bundle_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if requested:
        missing = [bundle_id for bundle_id in requested if bundle_id not in qa]
        if missing:
            raise SystemExit(f"Unknown bundle ids: {missing}")
        bundle_ids = list(dict.fromkeys(requested))
    else:
        bundle_ids = list(qa)
    if args.sample:
        rng = random.Random(args.sample_seed)
        bundle_ids = sorted(rng.sample(bundle_ids, min(args.sample, len(bundle_ids))))
    return bundle_ids


def safe_model_dir(provider: str, model: str) -> str:
    return f"{provider}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', model)}"


def main() -> int:
    args = parse_args()
    if args.api_workers <= 0:
        raise SystemExit("--api-workers must be positive")
    if args.api_workers > 1:
        return run_parallel_api_workers(args)
    qa = load_json(args.qa_json)
    if not isinstance(qa, dict):
        raise SystemExit("QA JSON must be an object keyed by bundle id.")
    manifests = manifest_index(args.manifest_json)
    bundle_ids = select_bundles(args, qa)
    missing_manifest = [bundle_id for bundle_id in bundle_ids if bundle_id not in manifests]
    if missing_manifest:
        raise SystemExit(f"Missing bundle manifest rows: {missing_manifest}")

    key = "" if args.dry_run else load_key(args.api_key_file)
    model_dir = args.output_dir / safe_model_dir(args.provider, args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_summary: list[dict[str, Any]] = []
    run_errors: list[dict[str, Any]] = []
    had_validation_errors = False

    for index, bundle_id in enumerate(bundle_ids, start=1):
        output_path = model_dir / f"{bundle_id}.json"
        if output_path.exists() and not args.force:
            if args.revalidate_existing:
                record = load_json(output_path)
                expected_ids = [
                    row["qa_id"] for row in qa_payload(bundle_id, qa[bundle_id])
                ]
                errors = validate_review(
                    record.get("review", {}), bundle_id, expected_ids
                )
                record["validation_errors"] = errors
                if errors:
                    had_validation_errors = True
                record["revalidated_at_utc"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                output_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[{index}/{len(bundle_ids)}] {bundle_id}: "
                    f"revalidated with {len(errors)} errors"
                )
                run_summary.append(
                    {
                        "bundle_id": bundle_id,
                        "output_path": str(output_path),
                        "revalidated": True,
                        "validation_error_count": len(errors),
                    }
                )
            else:
                print(f"[{index}/{len(bundle_ids)}] {bundle_id}: already reviewed")
            continue

        prompt, metadata = build_text_prompt(
            bundle_id,
            qa[bundle_id],
            manifests[bundle_id],
            hard_mode=args.hard_mode,
            evidence_context_only=args.evidence_context_only,
            context_pages=args.context_pages,
        )
        fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        images = (
            evidence_images(qa[bundle_id], manifests[bundle_id])
            if args.include_evidence_images
            else []
        )
        metadata["evidence_image_count"] = len(images)
        metadata["input_fingerprint_sha256"] = fingerprint

        if args.dry_run:
            print(
                f"[{index}/{len(bundle_ids)}] {bundle_id}: "
                f"{metadata['source_pages']} pages, "
                f"{metadata['prompt_characters']} chars, "
                f"{metadata['qa_count']} QA, {len(images)} images"
            )
            run_summary.append(metadata)
            continue

        started = time.time()
        raw: dict[str, Any] | None = None
        text = ""
        deterministic_normalizations: list[str] = []
        quota_retries = 0
        try:
            while True:
                try:
                    if args.provider == "claude":
                        raw, text = call_claude(
                            key,
                            args.model,
                            prompt,
                            images,
                            args.max_output_tokens,
                            args.timeout,
                            args.max_retries,
                        )
                    else:
                        raw, text = call_gemini(
                            key,
                            args.model,
                            prompt,
                            images,
                            args.max_output_tokens,
                            args.timeout,
                            args.max_retries,
                        )
                    break
                except ApiQuotaExhaustedError:
                    if quota_retries >= max(0, args.quota_retries):
                        raise
                    quota_retries += 1
                    wait_seconds = max(1, args.quota_retry_wait_seconds)
                    print(
                        f"[{index}/{len(bundle_ids)}] {bundle_id}: "
                        f"quota retry {quota_retries} after {wait_seconds}s",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
            review = parse_json_response(text)
            deterministic_normalizations = normalize_review_types(review)
            errors = validate_review(review, bundle_id, metadata["qa_ids"])
        except Exception as exc:
            error_path = model_dir / f"{bundle_id}.error.json"
            error_record = {
                "created_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
                "provider": args.provider,
                "model": args.model,
                "qa_source": str(args.qa_json),
                "manifest_source": str(args.manifest_json),
                "metadata": metadata,
                "elapsed_seconds": round(time.time() - started, 3),
                "error_type": type(exc).__name__,
                "error": valid_unicode(str(exc))[:8000],
                "response_text": valid_unicode(text)[:100000] if text else None,
                "raw_response": raw,
            }
            error_path.write_text(
                json.dumps(error_record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            run_errors.append(error_record)
            run_summary.append(
                {
                    **metadata,
                    "error_path": str(error_path),
                    "error_type": type(exc).__name__,
                }
            )
            print(
                f"[{index}/{len(bundle_ids)}] {bundle_id}: "
                f"failed with {type(exc).__name__}; saved {error_path}",
                file=sys.stderr,
            )
            continue

        error_path = model_dir / f"{bundle_id}.error.json"
        if error_path.exists():
            error_path.unlink()
        record = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": args.provider,
            "model": args.model,
            "qa_source": str(args.qa_json),
            "manifest_source": str(args.manifest_json),
            "metadata": metadata,
            "elapsed_seconds": round(time.time() - started, 3),
            "quota_retries": quota_retries,
            "validation_errors": errors,
            "deterministic_normalizations": deterministic_normalizations,
            "review": review,
            "usage": raw.get("usage") or raw.get("usageMetadata"),
            "stop_reason": raw.get("stop_reason"),
            "raw_response": raw,
        }
        output_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if errors:
            had_validation_errors = True
            print(
                f"[{index}/{len(bundle_ids)}] {bundle_id}: "
                f"saved with {len(errors)} validation errors",
                file=sys.stderr,
            )
        else:
            counts: dict[str, int] = {}
            for item in review["items"]:
                decision = item["decision"]
                counts[decision] = counts.get(decision, 0) + 1
            print(
                f"[{index}/{len(bundle_ids)}] {bundle_id}: "
                f"{counts}, {record['elapsed_seconds']}s"
            )
        run_summary.append(
            {
                **metadata,
                "output_path": str(output_path),
                "validation_error_count": len(errors),
            }
        )

    summary_suffix = f"_{args.worker_id}" if args.worker_id else ""
    summary_path = model_dir / f"_last_run_summary{summary_suffix}.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if run_errors or had_validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
