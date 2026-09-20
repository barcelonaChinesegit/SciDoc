#!/usr/bin/env python3
"""通用批量推理脚本，支持带 PDF 和仅问题两种输入模式。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
import pypdfium2 as pdfium

from pku_qa.pdf_assets import resolve_pdf_path as resolve_asset_pdf_path

from durable_work_queue import Claim, DurablePaperQueue
from evaluation_protocol import (
    PDF_INPUT_MODE,
    INFERENCE_REQUIRED_QA_FIELDS,
    STRICT_INFERENCE_REQUIRED_QA_FIELDS,
    UNANSWERABLE_LABEL,
    configure_protocol,
    build_inference_queue_value_contract,
    build_inference_protocol_metadata,
    canonicalize_pdf_output_for_storage,
    parse_canonical_pdf_output,
    pdf_corpus_sha256,
    pdf_corpus_sha256_from_manifest,
    pdf_sha256_manifest,
    provider_runtime_identity,
    protocol_for_item,
    sha256_file,
    validate_dataset_protocol,
)
from eval_framework import (
    append_progress_line, atomic_write_json, configure_hf_environment,
    create_provider, load_json, load_provider_specs, looks_like_error_output,
    resolve_provider_name, sanitize_name,
)
from gpu_reservation import managed_gpu_reservation
from progress_logging import progress_fields


RUNAWAY_EVIDENCE_PAGE_COUNT = 128


class InvalidModelOutputError(ValueError):
    """The model returned text, but it violated the output contract."""


PDF_SYSTEM_PROMPT = """You will be provided with a question and multiple PDF page images. Each image is preceded by an external page label such as [Page 1].

Please answer the question based only on the content of these images and clearly indicate the source page number(s).

Important:
- The evidence_pages field must use the external [Page N] numbers provided before each image. It must be a sorted ascending list of unique positive integers.
- Do not use page numbers printed inside the paper body, header, or footer.
- Include all and only the pages that directly support the answer; do not add unrelated pages.
- A consecutive page range is valid only when every listed page directly supports the answer.
- The answer should be as concise as possible while preserving correctness.
- If the requested information cannot be answered from the provided pages, the
  answer_pre value must be exactly "Unanswerable" and evidence_pages must be [].
- Never use variants such as "not mentioned", "not provided", "unknown", "N/A",
  or "None". Use only the exact label "Unanswerable" for a refusal.
- If answer_pre is not "Unanswerable", evidence_pages must contain at least one
  directly supporting external [Page N] number.

You must strictly output in the following JSON format:

{
  "answer_pre": "",
  "evidence_pages": []
}

The field "answer_pre" should contain a concise answer to the question, using as few words as possible while preserving correctness.
The field "evidence_pages" should contain the external [Page N] number(s) where the answer is supported.
There may be multiple page numbers.

Do not provide any extra explanation, comments, or text outside the JSON object.

Example:

{
  "answer_pre": "In Contrastive Order Learning, it is used for ordinal regression via LConOrd; in BACH, it is used in the contrastive softmaxes pr to fit head concentrations via KL divergence minimization.",
  "evidence_pages": [12, 13, 17]
}"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["4B", "8B"])
    parser.add_argument("--provider-name", type=str)
    parser.add_argument("--provider-config", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--qa-json", type=str, default="data/qa/1.base/rel__single_pdf__mixed__n4211__v1.json")
    parser.add_argument("--pdf-dir", type=str, default="data/pdfs")
    parser.add_argument(
        "--extra-pdf-dir",
        action="append",
        default=[],
        help="Additional PDF directory to search after --pdf-dir. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/evaluations/mixed_pdf_eval",
    )
    parser.add_argument(
        "--input-mode",
        choices=["pdf", "question_only"],
        default="pdf",
        help="pdf: 把 PDF 页面图像和问题一起输入；question_only: 只输入问题，不读取 PDF。",
    )
    parser.add_argument(
        "--page-input-policy",
        choices=["full", "oracle", "qa_field"],
        default="full",
        help=(
            "full: show the selected/full document; oracle: show only each "
            "QA's gold evidence pages; qa_field: read pages from "
            "--qa-page-field for evidence-ablation variants."
        ),
    )
    parser.add_argument(
        "--qa-page-field",
        default="input_pages",
        help="QA field used when --page-input-policy qa_field.",
    )
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument("--progress-file", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--max-total-pdf-megapixels",
        type=float,
        default=0,
        help=(
            "For full-PDF input, automatically lower the render DPI when all "
            "selected pages would exceed this aggregate pixel budget. 0 "
            "disables the budget. Every selected page is still included."
        ),
    )
    parser.add_argument(
        "--min-auto-dpi",
        type=int,
        default=72,
        help="Lowest DPI allowed by --max-total-pdf-megapixels.",
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=0,
        help="Maximum pages shown per PDF; 0 keeps every page. Ground-truth evidence pages are always retained.",
    )
    parser.add_argument("--page-selection-seed", type=int, default=20260713)
    parser.add_argument(
        "--disable-pdf-vision-cache",
        action="store_true",
        help=(
            "Recompute Qwen3-VL image features for every question. By default, "
            "features are cached while multiple questions reuse the same PDF."
        ),
    )
    parser.add_argument("--disable-hf-mirror", action="store_true")
    # 🌟 新增：分片参数
    parser.add_argument("--num-shards", type=int, default=1, help="总分片数")
    parser.add_argument("--shard-id", type=int, default=0, help="当前分片编号(0~N-1)")
    parser.add_argument(
        "--work-queue-dir",
        default=None,
        help=(
            "Use a shared crash-safe paper queue instead of fixed modulo "
            "shards. Multiple workers/GPU processes may share this directory."
        ),
    )
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--max-qa-retries", type=int, default=3)
    parser.add_argument(
        "--max-paper-failures",
        type=int,
        default=20,
        help="Quarantine a paper only after this many worker-level failures.",
    )
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--worker-idle-seconds", type=float, default=3.0)
    parser.add_argument("--protocol-fingerprint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--qa-source-sha256", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pdf-corpus-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def protocol_metadata_for_run(
    args: argparse.Namespace,
    *,
    model_name: str,
    provider_name: str,
    qa_source_sha256: str,
    pdf_corpus_hash: str | None,
) -> dict[str, object]:
    provider_config_sha256 = (
        sha256_file(args.provider_config) if args.provider_config else None
    )
    provider_spec = load_provider_specs(args.provider_config)[provider_name]
    return build_inference_protocol_metadata(
        {
            "qa_source_sha256": qa_source_sha256,
            "pdf_corpus_sha256": pdf_corpus_hash,
            "input_mode": args.input_mode,
            "page_input_policy": args.page_input_policy,
            "qa_page_field": args.qa_page_field,
            "model": model_name,
            "prompt_style": (
                "pdf" if args.pdf_mode else "question_only"
            ),
            "pdf_mode": bool(args.pdf_mode),
            "max_new_tokens": args.max_new_tokens,
            "max_qa_retries": args.max_qa_retries,
            "dpi": args.dpi,
            "max_total_pdf_megapixels": args.max_total_pdf_megapixels,
            "min_auto_dpi": args.min_auto_dpi,
            "max_pdf_pages": args.max_pdf_pages,
            "page_selection_seed": args.page_selection_seed,
            "provider_config_sha256": provider_config_sha256,
            "provider_runtime_identity": provider_runtime_identity(
                provider_name, provider_spec
            ),
        }
    )

def resolve_pdf_path(paper_id: str, pdf_dirs: list[Path]) -> Path | None:
    return resolve_asset_pdf_path(paper_id, pdf_dirs, required=False)


def select_pdf_pages(
    paper_id: str,
    paper_data: dict,
    page_count: int,
    max_pdf_pages: int,
    seed: int,
) -> list[int]:
    if max_pdf_pages <= 0 or page_count <= max_pdf_pages:
        return list(range(1, page_count + 1))

    evidence_pages = {
        int(page)
        for qa in paper_data.get("QA", {}).values()
        for page in (qa.get("evidence_pages") or [])
        if str(page).isdigit() and 1 <= int(page) <= page_count
    }
    target_count = max(max_pdf_pages, len(evidence_pages))
    candidates = [page for page in range(1, page_count + 1) if page not in evidence_pages]
    stable_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{paper_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    random.Random(stable_seed).shuffle(candidates)
    selected = evidence_pages | set(candidates[: max(0, target_count - len(evidence_pages))])
    return sorted(selected)


def build_per_qa_page_plan(
    qa_items: dict,
    page_field: str,
    page_count: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Plan rendered and per-QA pages for Oracle/ablation PDF modes.

    Empty page targets (the canonical case for Unanswerable) fall back to the
    full PDF so that PDF mode never silently becomes question-only mode.
    """
    item_pages = {
        str(qa_id): sorted(set(qa.get(page_field, []) or []))
        for qa_id, qa in qa_items.items()
    }
    full_pages = list(range(1, page_count + 1))
    rendered_pages = (
        full_pages
        if any(not pages for pages in item_pages.values())
        else sorted({page for pages in item_pages.values() for page in pages})
    )
    return rendered_pages, {
        qa_id: pages if pages else full_pages
        for qa_id, pages in item_pages.items()
    }


def pdf_to_images(pdf_path: str, page_numbers: list[int], dpi: int = 144) -> list[tuple[int, object]]:
    if not Path(pdf_path).exists(): return []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        scale = dpi / 72.0
        images = []
        for page_number in page_numbers:
            if 1 <= page_number <= len(pdf):
                images.append((page_number, pdf[page_number - 1].render(scale=scale).to_pil()))
        return images
    except Exception as exc:
        print(f"  Warning: Failed to convert PDF {pdf_path}: {exc}")
        return []


def choose_pdf_render_dpi(
    pdf,
    page_numbers: list[int],
    requested_dpi: int,
    max_total_megapixels: float,
    min_auto_dpi: int = 72,
) -> int:
    """Bound aggregate visual tokens while retaining every selected page."""
    requested_dpi = max(1, int(requested_dpi))
    if max_total_megapixels <= 0 or not page_numbers:
        return requested_dpi
    total_area_points = 0.0
    for page_number in page_numbers:
        if 1 <= page_number <= len(pdf):
            width, height = pdf[page_number - 1].get_size()
            total_area_points += float(width) * float(height)
    if total_area_points <= 0:
        return requested_dpi
    target_pixels = float(max_total_megapixels) * 1_000_000.0
    budget_dpi = math.floor(
        72.0 * math.sqrt(target_pixels / total_area_points)
    )
    return max(
        min(requested_dpi, max(1, budget_dpi)),
        min(requested_dpi, max(1, int(min_auto_dpi))),
    )


def structured_output_instruction(input_mode: str) -> str:
    if input_mode != "pdf":
        raise ValueError("question_only never requests evidence pages")
    source_note = "Use the original PDF page numbers shown before the page images."
    return (
        f"{source_note} Return exactly one valid JSON object with this schema: "
        '{"answer_pre":"<final answer>","evidence_pages":[<integer page numbers>]}. '
        "Do not output markdown, reasoning, or any other keys. Include all and "
        "only the pages that directly support the answer. If the requested "
        "information cannot be answered from the provided material, return "
        'exactly {"answer_pre":"Unanswerable","evidence_pages":[]}. Do not use '
        "any other refusal phrase; do not use any synonym: answer_pre must be "
        "exactly 'Unanswerable'. "
        "A non-Unanswerable answer must include at "
        "least one supporting page."
    )


def build_mcq_prompt(
    question_text: str,
    options: list,
    input_mode: str = "pdf",
    require_evidence_pages: bool = False,
) -> str:
    if input_mode == "pdf":
        require_evidence_pages = True
    elif require_evidence_pages:
        raise ValueError("question_only never requests evidence pages")
    lines = [f"Question: {question_text}", "", "Options:"]
    for opt in options: lines.append(f"{opt['id']}. {opt['text']}")
    if require_evidence_pages:
        lines.append("\n" + structured_output_instruction(input_mode))
    elif input_mode == "question_only":
        lines.append(
            "\nNo paper/PDF is provided. Answer from your own knowledge only. "
            "Please answer with ONLY the choice letter (A, B, C, or D). "
            "Output exactly one letter, nothing else."
        )
    else:
        lines.append("\nBased on the paper content above, please answer with ONLY the choice letter (A, B, C, or D). Output exactly one letter, nothing else.")
    return "\n".join(lines)

def build_fill_prompt(
    question_text: str,
    input_mode: str = "pdf",
    require_evidence_pages: bool = False,
) -> str:
    if input_mode == "pdf":
        require_evidence_pages = True
    elif require_evidence_pages:
        raise ValueError("question_only never requests evidence pages")
    if require_evidence_pages:
        context_instruction = (
            "Use the provided PDF pages to answer the question."
            if input_mode == "pdf"
            else "Use your internal knowledge to make a best-effort answer even though no PDF is provided."
        )
        return (
            f"Question: {question_text}\n\n"
            f"{context_instruction} Do not use 'unanswerable' as a default fallback. "
            + structured_output_instruction(input_mode)
        )
    if input_mode == "question_only":
        return (
            f"Question: {question_text}\n\n"
            "This is a closed-book evaluation: no paper/PDF is provided. "
            "Make a concise best-effort answer from internal knowledge. Do not "
            "output Unanswerable merely because the PDF is absent. If the "
            "question itself is intentionally unanswerable, output exactly "
            f"'{UNANSWERABLE_LABEL}'; never use a synonym or case variant. "
            "Output only the final answer, with no reasoning or explanation."
        )
    return (
        f"Question: {question_text}\n\n"
        "Based on the paper content above, please provide a concise answer. "
        "Do not answer 'unanswerable' as a default fallback. "
        "If the information cannot be found, output exactly 'Unanswerable'; "
        "do not use any synonym such as 'not mentioned', 'unknown', or 'None'. "
        "Output only the final answer, no explanation."
    )

def repair_json_string_backslashes(raw: str) -> str:
    """Escape model-emitted LaTeX slashes without altering JSON structure."""
    repaired: list[str] = []
    in_string = False
    index = 0
    while index < len(raw):
        character = raw[index]
        if character == '"':
            in_string = not in_string
            repaired.append(character)
            index += 1
            continue
        if character != "\\" or not in_string:
            repaired.append(character)
            index += 1
            continue
        if index + 1 >= len(raw):
            repaired.append("\\\\")
            index += 1
            continue
        following = raw[index + 1]
        if following in {'"', "\\", "/"}:
            repaired.extend((character, following))
            index += 2
            continue
        if (
            following == "u"
            and index + 5 < len(raw)
            and all(
                digit in "0123456789abcdefABCDEF"
                for digit in raw[index + 2 : index + 6]
            )
        ):
            repaired.append(raw[index : index + 6])
            index += 6
            continue
        # Qwen frequently emits LaTeX such as \leq, \tau or \Omega in an
        # otherwise valid JSON string. Preserve the literal slash by escaping
        # it for JSON. This also prevents \t/\n from becoming control chars.
        repaired.append("\\\\")
        index += 1
    return "".join(repaired)


def canonical_structured_output(
    output: str, *, strict_container: bool = False
) -> str | None:
    raw = str(output or "").strip()
    if not raw or looks_like_error_output(raw):
        return None
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    if strict_container:
        try:
            answer, pages = parse_canonical_pdf_output(raw)
        except ValueError:
            return None
        return json.dumps(
            {"answer_pre": answer, "evidence_pages": pages},
            ensure_ascii=False,
        )
    else:
        for index, character in enumerate(raw):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except (json.JSONDecodeError, TypeError):
                try:
                    value, _ = decoder.raw_decode(
                        repair_json_string_backslashes(raw[index:])
                    )
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(value, dict):
                candidates.append(value)
    for value in candidates:
        answer_field = next(
            (
                field
                for field in ("answer_pre", "answer", "model_answer")
                if field in value
            ),
            None,
        )
        raw_pages = value.get("evidence_pages")
        if answer_field is None or not isinstance(raw_pages, list):
            continue
        answer = str(value.get(answer_field, "")).strip()
        if not answer:
            continue
        pages = []
        for page in raw_pages:
            if isinstance(page, bool):
                continue
            if isinstance(page, int):
                pages.append(page)
                continue
            match = re.fullmatch(
                r"\s*(?:PDF_PAGE_|PAGE\s*)?(\d+)\s*",
                str(page),
                flags=re.IGNORECASE,
            )
            if match:
                pages.append(int(match.group(1)))
        pages = sorted(set(pages))
        if answer == "Unanswerable":
            if pages:
                continue
        elif answer.lower() == "unanswerable":
            # The abstention label is intentionally case-sensitive so the
            # tested model must follow the controlled output protocol.
            continue
        elif not pages:
            continue
        return json.dumps(
            {
                "answer_pre": answer,
                "evidence_pages": pages,
            },
            ensure_ascii=False,
        )
    return None


def structured_output_is_valid(
    output: str,
    required: bool,
    allowed_pages: list[int] | None = None,
) -> bool:
    raw = str(output or "").strip()
    if not raw or looks_like_error_output(raw):
        return False
    if not required:
        return True
    try:
        parse_canonical_pdf_output(raw, allowed_pages=allowed_pages)
    except ValueError:
        return False
    return True


def prepare_pdf_output_for_storage(
    raw_output: str,
    *,
    allowed_pages: list[int] | None,
) -> tuple[str, list[str], str | None]:
    """Preserve an exhausted illegal prediction for publication scoring.

    Legal PDF predictions receive only the representation-level canonical
    normalizations defined by ``canonicalize_pdf_output_for_storage``.  If a
    model still violates the PDF output contract after its finite correction
    attempts, retain the exact raw response instead of changing the prediction
    or aborting the full benchmark.  Judge will independently mark that row
    illegal and score it incorrect.
    """
    try:
        model_output, normalizations = canonicalize_pdf_output_for_storage(
            raw_output,
            allowed_pages=allowed_pages,
        )
    except ValueError:
        return str(raw_output), [], "invalid_model_output_after_retries"
    return model_output, normalizations, None


def clear_cuda_after_failure(provider) -> None:
    clear_cache = getattr(provider, "clear_vision_cache", None)
    if callable(clear_cache):
        clear_cache()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def has_runaway_repetition(output: str, minimum_repeats: int = 16) -> bool:
    """Detect the repeated-token degeneration that can consume the token cap."""
    raw = str(output or "")
    if not raw:
        return False
    escaped_unicode = re.compile(
        rf"(\\u[0-9a-fA-F]{{4}})(?:\1){{{minimum_repeats - 1},}}"
    )
    if escaped_unicode.search(raw):
        return True
    repeated_character = re.compile(
        rf"(.)(?:\1){{{max(32, minimum_repeats) - 1},}}",
        re.DOTALL,
    )
    return repeated_character.search(raw) is not None


def has_runaway_evidence_pages(
    output: str, maximum_pages: int = RUNAWAY_EVIDENCE_PAGE_COUNT
) -> bool:
    """Detect obvious repetition even when generation ends before valid JSON.

    This is only a degeneration guard.  It is deliberately much larger than
    any expected gold evidence set and is not part of the scoring protocol.
    A syntactically valid list is always preserved for the Judge to score.
    """
    raw = str(output or "")
    match = re.search(
        r'["\']evidence_pages["\']\s*:\s*\[(.*)',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False
    page_numbers = re.findall(r"\b\d+\b", match.group(1))
    return len(set(page_numbers)) > maximum_pages


def generate_with_retries(
    provider,
    messages,
    args,
    *,
    allowed_evidence_pages: list[int] | None = None,
) -> str:
    last_error: BaseException | None = None
    last_invalid_response: str | None = None
    attempt_messages = list(messages)
    fallback_generation = False
    set_generation_overrides = getattr(
        provider, "set_generation_overrides", None
    )
    try:
        for attempt in range(1, args.max_qa_retries + 1):
            invalid_response: str | None = None
            if callable(set_generation_overrides):
                set_generation_overrides(
                    {
                        "repetition_penalty": 1.12,
                    }
                    if fallback_generation
                    else {}
                )
            try:
                generation_token_limit = (
                    min(int(args.max_new_tokens), 256)
                    if fallback_generation
                    else int(args.max_new_tokens)
                )
                output = provider.generate(
                    attempt_messages, generation_token_limit
                )
                if not structured_output_is_valid(
                    output,
                    args.require_evidence_pages,
                    allowed_pages=allowed_evidence_pages,
                ):
                    invalid_response = str(output)
                    last_invalid_response = invalid_response
                    preview = " ".join(str(output).split())[:500]
                    print(f" invalid_output_preview={preview!r}", flush=True)
                    runaway_repetition = has_runaway_repetition(output)
                    # Token-level n-gram blocking is useful for true repeated
                    # character degeneration, but is prohibitively expensive
                    # for very long full-PDF visual prompts. Overlong evidence
                    # lists are corrected by the explicit retry instruction.
                    if runaway_repetition:
                        fallback_generation = True
                    raise InvalidModelOutputError(
                        "empty/error/malformed model output"
                    )
                if args.require_evidence_pages:
                    canonical = canonical_structured_output(
                        output, strict_container=True
                    )
                    assert canonical is not None
                    # Preserve the exact legal model response for downstream
                    # audit. Judge applies the same strict parser again.
                    return str(output).strip()
                return output
            except Exception as exc:
                last_error = exc
                print(
                    f" retry {attempt}/{args.max_qa_retries} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                # A format violation means generation itself succeeded. Keep
                # the cached full-PDF vision features for the correction retry;
                # rebuilding dozens of pages here can turn seconds into many
                # minutes. Runtime/model failures still clear CUDA state.
                if not isinstance(exc, InvalidModelOutputError):
                    clear_cuda_after_failure(provider)
                if attempt < args.max_qa_retries:
                    evidence_guidance = (
                        "all and only pages that directly support the answer. "
                        "A consecutive range is valid when every listed page "
                        "directly supports the answer. Use a sorted ascending "
                        "list of unique positive integer external page labels. "
                    )
                    correction = (
                        f"Correction attempt {attempt + 1}: your previous "
                        "response was invalid. "
                        "Return exactly one JSON object with only "
                        '"answer_pre" and "evidence_pages". '
                        "Do not include reasoning or markdown. Include "
                        f"{evidence_guidance}"
                        "If the answer is unavailable, use exactly "
                        '"answer_pre":"Unanswerable" with '
                        '"evidence_pages":[]; otherwise answer_pre must be '
                        "non-empty and evidence_pages must be non-empty."
                    )
                    if fallback_generation:
                        correction += (
                            " The previous answer degenerated into repeated "
                            "symbols. Keep answer_pre under 60 words, avoid "
                            "LaTeX and backslashes, express formulas in plain "
                            "ASCII, and close the JSON object immediately."
                        )
                    attempt_messages = [*messages]
                    if invalid_response is not None and not fallback_generation:
                        # Make the retry a real conversational correction.
                        # Previously two consecutive user messages were sent,
                        # so Qwen often repeated the identical invalid list.
                        attempt_messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": invalid_response[:2000],
                                    }
                                ],
                            }
                        )
                    attempt_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": correction,
                                }
                            ],
                        }
                    )
                    time.sleep(
                        args.retry_backoff_seconds * (2 ** (attempt - 1))
                    )
        assert last_error is not None
        if (
            isinstance(last_error, InvalidModelOutputError)
            and last_invalid_response is not None
        ):
            print(
                " exhausted format-correction retries; preserving exact "
                "model output for Judge scoring",
                flush=True,
            )
            return last_invalid_response
        raise RuntimeError(
            f"model generation failed after {args.max_qa_retries} attempts"
        ) from last_error
    finally:
        if callable(set_generation_overrides):
            set_generation_overrides({})


def process_paper(
    *,
    paper_id: str,
    paper_data: dict,
    provider,
    provider_name: str,
    args,
    pdf_dirs: list[Path],
    existing_result: dict | None = None,
    checkpoint=None,
) -> dict:
    qa_items = paper_data.get("QA", {})
    num_q = len(qa_items)

    pdf_images: list[tuple[int, object]] = []
    shown_pages: list[int] = []
    per_qa_page_plan: dict[str, list[int]] = {}
    paper_pdf_sha256: str | None = None
    pdf_page_count: int | None = None
    render_dpi = int(args.dpi)
    if args.input_mode == "pdf":
        pdf_path = resolve_pdf_path(paper_id, pdf_dirs)
        if pdf_path is not None:
            try:
                paper_pdf_sha256 = sha256_file(pdf_path)
                expected_pdf_hash = getattr(
                    args, "expected_pdf_sha256_by_paper", {}
                ).get(str(paper_id))
                if (
                    expected_pdf_hash
                    and paper_pdf_sha256 != expected_pdf_hash
                ):
                    raise ValueError(
                        f"{paper_id}: PDF changed after worker startup; "
                        f"expected {expected_pdf_hash}, found "
                        f"{paper_pdf_sha256}"
                    )
                pdf_doc = pdfium.PdfDocument(str(pdf_path))
                page_count = len(pdf_doc)
                pdf_page_count = page_count
                for qa_id, qa in qa_items.items():
                    fields = ["evidence_pages"]
                    if args.page_input_policy == "qa_field":
                        fields.append(args.qa_page_field)
                    for field in fields:
                        for page in qa.get(field, []) or []:
                            if (
                                isinstance(page, bool)
                                or not isinstance(page, int)
                                or not 1 <= page <= page_count
                            ):
                                raise ValueError(
                                    f"{paper_id}/{qa_id}: {field} contains "
                                    f"page {page!r} outside PDF bounds "
                                    f"1..{page_count}"
                                )
                if args.page_input_policy == "full":
                    shown_pages = select_pdf_pages(
                        paper_id,
                        paper_data,
                        page_count,
                        args.max_pdf_pages,
                        args.page_selection_seed,
                    )
                else:
                    page_field = (
                        "evidence_pages"
                        if args.page_input_policy == "oracle"
                        else args.qa_page_field
                    )
                    shown_pages, per_qa_page_plan = build_per_qa_page_plan(
                        qa_items,
                        page_field,
                        page_count,
                    )
                render_dpi = choose_pdf_render_dpi(
                    pdf_doc,
                    shown_pages,
                    args.dpi,
                    args.max_total_pdf_megapixels,
                    args.min_auto_dpi,
                )
                del pdf_doc
                if render_dpi < args.dpi:
                    print(
                        f"[pdf-render] paper={paper_id} pages={len(shown_pages)} "
                        f"requested_dpi={args.dpi} effective_dpi={render_dpi} "
                        f"pixel_budget_mp={args.max_total_pdf_megapixels:g}",
                        flush=True,
                    )
                pdf_images = pdf_to_images(
                    str(pdf_path), shown_pages, dpi=render_dpi
                )
                if sha256_file(pdf_path) != paper_pdf_sha256:
                    raise ValueError(
                        f"{paper_id}: PDF changed while it was being rendered"
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to inspect/render PDF {pdf_path}: {exc}"
                ) from exc
        if not pdf_images:
            searched = [str(pdf_dir) for pdf_dir in pdf_dirs]
            raise FileNotFoundError(
                f"PDF not found or unreadable; searched={searched}"
            )

    set_vision_cache_key = getattr(provider, "set_vision_cache_key", None)
    if callable(set_vision_cache_key):
        set_vision_cache_key(
            None
            if (
                args.disable_pdf_vision_cache
                or args.input_mode != "pdf"
                or args.page_input_policy != "full"
            )
            else f"{paper_id}:{render_dpi}:{','.join(map(str, shown_pages))}"
        )

    paper_result = (
        existing_result
        if isinstance(existing_result, dict)
        else {
            "paper": paper_id,
            "provider": provider_name,
            "input_mode": args.input_mode,
            "QA": {},
        }
    )
    paper_result["paper"] = paper_id
    paper_result["provider"] = provider_name
    paper_result["input_mode"] = args.input_mode
    if args.protocol_metadata is not None:
        paper_result["protocol_fingerprint"] = args.protocol_metadata[
            "protocol_fingerprint"
        ]
        paper_result["qa_source_sha256"] = args.qa_source_sha256
        paper_result["pdf_corpus_sha256"] = args.pdf_corpus_sha256
        paper_result["pdf_sha256"] = paper_pdf_sha256
    paper_result.setdefault("QA", {})

    for q_idx, (qa_id, qa_data_item) in enumerate(qa_items.items(), start=1):
        previous = paper_result["QA"].get(qa_id)
        previous_matches_protocol = (
            args.protocol_metadata is None
            or (
                previous.get("protocol_fingerprint")
                == args.protocol_metadata["protocol_fingerprint"]
                and previous.get("qa_source_sha256")
                == args.qa_source_sha256
                and previous.get("pdf_sha256") == paper_pdf_sha256
            )
        ) if isinstance(previous, dict) else False
        if (
            isinstance(previous, dict)
            and previous_matches_protocol
            and structured_output_is_valid(
                previous.get("model_output", ""),
                args.require_evidence_pages,
                allowed_pages=previous.get("shown_pdf_pages"),
            )
        ):
            continue
        qa_started_at = time.monotonic()
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"event=qa_started worker={args.worker_id or args.shard_id} "
            f"paper={paper_id} qa={qa_id} "
            f"{progress_fields(q_idx, num_q)} progress_scope=paper_qa",
            flush=True,
        )
        question = qa_data_item["question"]
        is_mcq = "options" in qa_data_item

        if args.input_mode == "pdf" and args.page_input_policy != "full":
            page_field = (
                "evidence_pages"
                if args.page_input_policy == "oracle"
                else args.qa_page_field
            )
            qa_shown_pages = per_qa_page_plan[str(qa_id)]
            qa_page_set = set(qa_shown_pages)
            qa_pdf_images = [
                (page, image)
                for page, image in pdf_images
                if page in qa_page_set
            ]
        else:
            qa_shown_pages = shown_pages
            qa_pdf_images = pdf_images

        content = []
        for page_number, image in qa_pdf_images:
            page_label = (
                f"[Page {page_number}]"
                if args.pdf_mode
                else f"[Original PDF page {page_number}]"
            )
            content.append({"type": "text", "text": page_label})
            content.append({"type": "image", "image": image})
        if args.pdf_mode:
            if is_mcq:
                option_lines = "\n".join(
                    f"{option['id']}. {option['text']}"
                    for option in qa_data_item["options"]
                )
                prompt_text = f"Question: {question}\n\nOptions:\n{option_lines}"
            else:
                prompt_text = f"Question: {question}"
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": args.pdf_system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": content
                    + [{"type": "text", "text": prompt_text}],
                },
            ]
        else:
            prompt_text = (
                build_mcq_prompt(
                    question,
                    qa_data_item["options"],
                    args.input_mode,
                    args.require_evidence_pages,
                )
                if is_mcq
                else build_fill_prompt(
                    question, args.input_mode, args.require_evidence_pages
                )
            )
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]

        raw_model_output = generate_with_retries(
            provider,
            messages,
            args,
            allowed_evidence_pages=(
                qa_shown_pages if args.input_mode == PDF_INPUT_MODE else None
            ),
        )
        if args.input_mode == PDF_INPUT_MODE:
            (
                model_output,
                deterministic_normalizations,
                generation_status,
            ) = prepare_pdf_output_for_storage(
                raw_model_output,
                allowed_pages=qa_shown_pages,
            )
        else:
            model_output = str(raw_model_output)
            deterministic_normalizations = []
            generation_status = None
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"event=qa_completed worker={args.worker_id or args.shard_id} "
            f"paper={paper_id} qa={qa_id} "
            f"{progress_fields(q_idx, num_q)} progress_scope=paper_qa "
            f"duration_seconds={time.monotonic() - qa_started_at:.2f}",
            flush=True,
        )
        reference_answer = qa_data_item.get("answer", "")
        item_protocol = protocol_for_item(args.input_mode, reference_answer)
        qa_result = {
            "evaluated_model": str(args.model or provider_name),
            "question": question,
            "correct_answer": reference_answer,
            "model_output": model_output,
            "raw_model_output": str(raw_model_output),
            "raw_model_output_sha256": hashlib.sha256(
                str(raw_model_output).encode("utf-8")
            ).hexdigest(),
            "deterministic_normalizations": deterministic_normalizations,
            "type": "mcq" if is_mcq else "fill",
            "input_mode": args.input_mode,
            "reference_evidence_pages": qa_data_item.get("evidence_pages", []),
            "require_structured_output": (
                item_protocol.require_structured_output
            ),
            "require_evidence_pages": item_protocol.require_evidence_pages,
            "shown_pdf_pages": qa_shown_pages,
            "total_pdf_pages": pdf_page_count,
            "max_pdf_pages": int(args.max_pdf_pages),
            "requested_pdf_dpi": int(args.dpi),
            "rendered_pdf_dpi": render_dpi,
            "pdf_render_downscaled": render_dpi < int(args.dpi),
            "page_input_policy": args.page_input_policy,
            "prompt_style": (
                "pdf" if args.pdf_mode else "question_only"
            ),
        }
        if args.protocol_metadata is not None:
            qa_result.update(
                {
                    "protocol_fingerprint": args.protocol_metadata[
                        "protocol_fingerprint"
                    ],
                    "qa_source_sha256": args.qa_source_sha256,
                    "pdf_corpus_sha256": args.pdf_corpus_sha256,
                    "pdf_sha256": paper_pdf_sha256,
                }
            )
        if generation_status is not None:
            qa_result["generation_status"] = generation_status
        for metadata_field in (
            "answer_format",
            "answer_aliases",
            "answer_unit",
            "numeric_tolerance",
            "string_metric",
            "evidence_items",
            "evidence_hops",
            "evidence_span",
            "evidence_span_ratio",
            "modal_types",
            "question_type",
            "question_category",
            "annotation_provenance",
            "review_status",
        ):
            if metadata_field in qa_data_item:
                qa_result[metadata_field] = qa_data_item[metadata_field]
        if is_mcq:
            qa_result["options"] = qa_data_item["options"]
        paper_result["QA"][qa_id] = qa_result
        if checkpoint is not None:
            checkpoint(paper_result)
    return paper_result


def run_queue_worker(
    args,
    provider,
    provider_name: str,
    qa_data: dict,
    pdf_dirs: list[Path],
) -> None:
    required_fields = list(
        STRICT_INFERENCE_REQUIRED_QA_FIELDS
        if args.protocol_metadata is not None
        else INFERENCE_REQUIRED_QA_FIELDS
    )
    manifest_metadata = (
        {"stage": "inference", **args.protocol_metadata}
        if args.protocol_metadata is not None
        else {
            "stage": "inference",
            "protocol_version": 1,
            "input_mode": args.input_mode,
            "page_input_policy": args.page_input_policy,
            "qa_page_field": args.qa_page_field,
            "model": args.model or provider_name,
            "prompt_style": (
                "pdf" if args.pdf_mode else "question_only"
            ),
        }
    )
    if args.protocol_metadata is not None:
        required_values, required_values_by_paper = (
            build_inference_queue_value_contract(
                qa_data,
                protocol_metadata=args.protocol_metadata,
                qa_source_sha256=args.qa_source_sha256,
                pdf_corpus_hash=args.pdf_corpus_sha256,
                pdf_sha256_by_paper=args.expected_pdf_sha256_by_paper,
                input_mode=args.input_mode,
                page_input_policy=args.page_input_policy,
                prompt_style=(
                    "pdf"
                    if args.pdf_mode
                    else "question_only"
                ),
                max_pdf_pages=args.max_pdf_pages,
                evaluated_model=str(args.model or provider_name),
            )
        )
    else:
        required_values, required_values_by_paper = {}, {}
    queue = DurablePaperQueue(
        args.work_queue_dir,
        qa_data,
        max_paper_failures=args.max_paper_failures,
        require_structured_output=args.require_evidence_pages,
        required_qa_fields=tuple(required_fields),
        required_qa_field_values=required_values,
        required_qa_field_values_by_paper=required_values_by_paper,
        manifest_metadata=manifest_metadata,
    )
    worker_id = args.worker_id or f"gpu{args.gpu}-pid{os.getpid()}"
    initial_status = queue.status()
    print(
        f"[queue-worker] id={worker_id} gpu={args.gpu} "
        f"unfinished={initial_status['unfinished_papers']} "
        f"{progress_fields(initial_status['completed_papers'], initial_status['total_papers'])} "
        "progress_scope=inference_papers",
        flush=True,
    )
    while not queue.is_complete():
        claim = queue.claim_next(worker_id)
        if claim is None:
            status = queue.status()
            quarantined = [
                item for item in status["failures"] if item.get("quarantined")
            ]
            if (
                status["unfinished_papers"]
                and not status["active_claims"]
                and quarantined
                and not status.get("retryable_papers", 0)
            ):
                raise RuntimeError(
                    f"{len(quarantined)} paper(s) quarantined after repeated failures"
                )
            time.sleep(args.worker_idle_seconds)
            continue
        try:
            existing = queue.load_result(claim.paper_id)

            def checkpoint(result: dict) -> None:
                queue.save_partial(claim.paper_id, result)
                queue.heartbeat(claim)

            result = process_paper(
                paper_id=claim.paper_id,
                paper_data=qa_data[claim.paper_id],
                provider=provider,
                provider_name=provider_name,
                args=args,
                pdf_dirs=pdf_dirs,
                existing_result=existing,
                checkpoint=checkpoint,
            )
            queue.finish(claim, result)
            status = queue.status()
            print(
                f"[queue-worker] event=paper_completed worker={worker_id} "
                f"paper={claim.paper_id} "
                f"{progress_fields(status['completed_papers'], status['total_papers'])} "
                "progress_scope=inference_papers",
                flush=True,
            )
        except KeyboardInterrupt:
            queue.release(claim)
            raise
        except BaseException as exc:
            queue.fail(claim, exc)
            failure_status = queue.status()
            clear_cuda_after_failure(provider)
            chain = []
            current: BaseException | None = exc
            while current is not None and len(chain) < 8:
                chain.append(f"{type(current).__name__}: {current}")
                current = current.__cause__ or current.__context__
            failure_text = "\n".join(chain).lower()
            if "empty/error/malformed model output" in failure_text:
                print(
                    f"[queue-worker] paper={claim.paper_id} format failure; "
                    "released for a later retry without reloading the model. "
                    f"{progress_fields(failure_status['completed_papers'], failure_status['total_papers'])} "
                    "progress_scope=inference_papers",
                    flush=True,
                )
                continue
            # OOM and unknown runtime faults exit the worker so the adaptive
            # pool can fully unload the model and perform a clean restart.
            raise


def main():
    args = parse_args()
    configure_protocol(args)

    if args.pdf_mode and args.provider_name is None:
        provider_name = f"local_qwen3_vl_{args.model.lower()}_pdf"
    else:
        provider_name = resolve_provider_name(
            model=args.model,
            provider_name=args.provider_name,
        )
    provider_key = sanitize_name(args.model if args.model else provider_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名加上 shard_id 防止多进程写入冲突
    res_name = args.result_file if args.result_file else f"results_{provider_key}_shard{args.shard_id}.json"
    prog_name = args.progress_file if args.progress_file else f"progress_{provider_key}_shard{args.shard_id}.txt"
    result_file = Path(res_name)
    progress_file = Path(prog_name)
    if not result_file.is_absolute() and args.result_file is None:
        result_file = output_dir / result_file
    if not progress_file.is_absolute() and args.progress_file is None:
        progress_file = output_dir / progress_file

    qa_data = load_json(args.qa_json, default={})
    dataset_profile = validate_dataset_protocol(qa_data, args.input_mode)
    actual_qa_sha256 = sha256_file(args.qa_json)
    if args.qa_source_sha256 and args.qa_source_sha256 != actual_qa_sha256:
        raise ValueError(
            "QA source changed after scheduling: "
            f"expected {args.qa_source_sha256}, found {actual_qa_sha256}"
        )
    args.qa_source_sha256 = args.qa_source_sha256 or actual_qa_sha256
    pdf_dirs = [Path(args.pdf_dir), *[Path(path) for path in args.extra_pdf_dir]]
    strict_pdf_lock = bool(
        args.input_mode == PDF_INPUT_MODE
        and (args.protocol_fingerprint or not args.work_queue_dir)
    )
    args.expected_pdf_sha256_by_paper = {}
    if strict_pdf_lock:
        args.expected_pdf_sha256_by_paper = pdf_sha256_manifest(
            qa_data, pdf_dirs
        )
        actual_pdf_corpus_sha256 = pdf_corpus_sha256_from_manifest(
            args.expected_pdf_sha256_by_paper
        )
        if (
            args.pdf_corpus_sha256
            and args.pdf_corpus_sha256 != actual_pdf_corpus_sha256
        ):
            raise ValueError(
                "PDF corpus changed after scheduling: expected "
                f"{args.pdf_corpus_sha256}, found "
                f"{actual_pdf_corpus_sha256}"
            )
        args.pdf_corpus_sha256 = actual_pdf_corpus_sha256
    elif args.pdf_corpus_sha256 is None and not args.work_queue_dir:
        args.pdf_corpus_sha256 = (
            pdf_corpus_sha256(qa_data, pdf_dirs)
            if args.input_mode == PDF_INPUT_MODE
            else None
        )
    if args.work_queue_dir and not args.protocol_fingerprint:
        raise ValueError(
            "Durable inference queues require a scheduler-issued protocol "
            "fingerprint. Pre-v4 queues are audit-only and cannot be resumed "
            "by the publication runner."
        )
    args.protocol_metadata = protocol_metadata_for_run(
        args,
        model_name=args.model or provider_name,
        provider_name=provider_name,
        qa_source_sha256=args.qa_source_sha256,
        pdf_corpus_hash=args.pdf_corpus_sha256,
    )
    if (
        args.protocol_fingerprint
        and args.protocol_fingerprint
        != args.protocol_metadata["protocol_fingerprint"]
    ):
        raise ValueError(
            "Inference protocol fingerprint does not match scheduler"
        )
    args.protocol_fingerprint = str(
        args.protocol_metadata["protocol_fingerprint"]
    )
    args.pdf_system_prompt = PDF_SYSTEM_PROMPT
    print(
        "[protocol] "
        f"input_mode={args.input_mode} "
        f"structured={args.require_evidence_pages} "
        f"evidence={args.require_evidence_pages} "
        f"items={dataset_profile['total']} "
        f"answerable={dataset_profile['answerable']} "
        f"unanswerable={dataset_profile['unanswerable']}",
        flush=True,
    )

    use_hf_mirror = not args.disable_hf_mirror
    configure_hf_environment(use_mirror=use_hf_mirror, cuda_visible_devices=args.gpu)

    if args.shard_id == 0:
        print(f"{'=' * 60}\nProvider: {provider_name} | Shards: {args.num_shards}\n{'=' * 60}")

    provider = create_provider(provider_name, config_path=args.provider_config)
    
    all_papers = sorted(list(qa_data.items()), key=lambda x: x[0])

    if args.work_queue_dir:
        run_queue_worker(args, provider, provider_name, qa_data, pdf_dirs)
        return
    
    # 🌟 核心：根据 shard_id 切分任务
    my_papers = [p for i, p in enumerate(all_papers) if i % args.num_shards == args.shard_id]
    
    results = load_json(result_file, default={}) or {}
    if args.protocol_metadata is not None:
        fingerprint = args.protocol_metadata["protocol_fingerprint"]
        completed_papers = {
            str(paper_id)
            for paper_id, paper in results.items()
            if isinstance(paper, dict)
            and paper.get("protocol_fingerprint") == fingerprint
            and paper.get("qa_source_sha256") == args.qa_source_sha256
            and set(map(str, paper.get("QA", {})))
            == set(map(str, qa_data.get(str(paper_id), {}).get("QA", {})))
            and all(
                isinstance(item, dict)
                and item.get("protocol_fingerprint") == fingerprint
                and item.get("qa_source_sha256") == args.qa_source_sha256
                and structured_output_is_valid(
                    item.get("model_output", ""),
                    args.require_evidence_pages,
                    allowed_pages=item.get("shown_pdf_pages"),
                )
                for item in paper.get("QA", {}).values()
            )
        }
    else:
        completed_papers = set(results.keys())
    if args.protocol_metadata is None and progress_file.exists():
        for line in progress_file.read_text(encoding="utf-8").splitlines():
            if line.strip(): completed_papers.add(line.strip())

    total_my_papers = len(my_papers)
    
    print(
        f"[Shard {args.shard_id}/{args.num_shards}] 输入模式: {args.input_mode} | "
        f"分配到 {total_my_papers} 篇论文，已完成 {len(completed_papers)} 篇。 "
        f"{progress_fields(len(completed_papers), total_my_papers)} "
        "progress_scope=shard_papers"
    )

    for idx, (paper_id, paper_data) in enumerate(my_papers):
        if paper_id in completed_papers: continue
        
        paper_result = process_paper(
            paper_id=paper_id,
            paper_data=paper_data,
            provider=provider,
            provider_name=provider_name,
            args=args,
            pdf_dirs=pdf_dirs,
            existing_result=results.get(paper_id),
        )
        results[paper_id] = paper_result
        atomic_write_json(result_file, results)
        append_progress_line(progress_file, paper_id)
        completed_papers.add(paper_id)
        print(
            f"[Shard {args.shard_id}/{args.num_shards}] "
            f"event=paper_completed paper={paper_id} "
            f"{progress_fields(len(completed_papers), total_my_papers)} "
            "progress_scope=shard_papers",
            flush=True,
        )

if __name__ == "__main__":
    with managed_gpu_reservation("run_inference.py"):
        main()
