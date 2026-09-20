#!/usr/bin/env python3
"""Compose hard Cross-PDF QA from previously audited QA facts.

The generator sees QA text and the three-document boundary map, but no PDF
pages. Full PDFs are reserved for the independent Claude/Gemini review. Raw
responses are checkpointed per bundle and generation round so several workers
can safely share one output directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from pku_qa.evaluation.model_paths import model_directory
from pku_qa.pdf_assets import resolve_source_pdf_path
from pku_qa.workflows.generation.build_single_pdf_reasoning_qa import (
    GeminiGenerator,
    LocalQwenGenerator,
    atomic_json,
    normalized_text,
    read_json,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "data/qa/4.cross_pdf/expansion/work__cross_pdf__raw_expansion__batch00__n2584.json"
DEFAULT_VERIFIED = ROOT / "data/qa/4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json"
DEFAULT_BASE_SOURCE = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
DEFAULT_REVIEW_DIR = (
    ROOT
    / "data/qa/4.cross_pdf/semantic_reaudit/remaining_1941_api_reviews"
    / "gemini_gemini-3-flash-preview"
)
DEFAULT_SOURCE_PDF_DIR = ROOT / "data/pdfs"
DEFAULT_OUTPUT = ROOT / "data/qa/4.cross_pdf/hard_expansion"
DEFAULT_MODEL = model_directory("Qwen3.6-27B")
DEFAULT_API_KEY_FILE = ROOT / ".env"
AGGREGATE_CHECKPOINT_INTERVAL = 100

REASONING_TYPES = {
    "adjacent_module_dependency",
    "metric_reasoning",
    "component_hierarchy",
    "method_transfer",
    "compatibility_judgment",
    "conflict_resolution",
}
QUALITY_FLAGS = (
    "requires_all_source_qas",
    "requires_all_source_documents",
    "not_simple_concatenation",
    "scientifically_meaningful",
    "intermediate_facts_verified",
    "answer_contains_reasoning_bridge",
)

PAIR_STOPWORDS = {
    "about",
    "across",
    "approach",
    "approaches",
    "between",
    "because",
    "compare",
    "contrast",
    "different",
    "document",
    "documents",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "performance",
    "primary",
    "question",
    "regarding",
    "result",
    "results",
    "shows",
    "table",
    "specific",
    "system",
    "systems",
    "their",
    "using",
    "which",
}

RELATION_ROLE_PATTERNS = {
    "limitation": (
        r"\b(?:cannot|can't|does not|do not|fails? to|lack(?:s|ing)?|unavailable)\b",
        r"\b(?:incompatib|limitation|poorly suited|not suited|unsuitable|bottleneck)\w*\b",
    ),
    "requirement": (
        r"\b(?:require(?:s|d)?|prerequisite|depends? on|relies? on|needs?)\b",
        r"\b(?:only when|provided that|conditioned on)\b",
    ),
    "metric_label": (
        r"\b(?:metric|score|agreement|correlation|kappa|accuracy|precision|recall|f1|rate|convergence|bound|error|variance|loss|auroc|dice)\b",
        r"\b(?:categorical|ordinal|continuous|binary|labels?|rubric|annotation|spectral|distributional)\b",
    ),
    "method": (
        r"\b(?:algorithm|method|optimizer|descent|sampling|solver|estimator|framework|procedure)\b",
        r"\b(?:monte carlo|langevin|gradient|bayesian|gibbs|regression|stochastic)\b",
    ),
    "producer": (
        r"\b(?:output|produce(?:s|d)?|emit(?:s|ted)?|generate(?:s|d)?|returns?)\b",
    ),
    "consumer": (
        r"\b(?:input|consume(?:s|d)?|accept(?:s|ed)?|fed into|passed to)\b",
    ),
    "hierarchy": (
        r"\b(?:component|module|stage|layer|subsystem|pipeline|architecture)\b",
        r"\b(?:consists? of|comprises?|contains?|part of|within)\b",
    ),
    "transfer": (
        r"\b(?:transfer(?:red)?|adapt(?:s|ed)?|extend(?:s|ed)?|appl(?:y|ied))\b",
        r"\b(?:target domain|source domain|cross-domain)\b",
    ),
}

SYSTEM_PROMPT = """\
You construct hard, reliable multi-paper scientific QA from audited QA facts.
You do not receive PDFs. Use only the supplied source QA and document map.

Each new item must require one coherent inference across 2-3 papers. It must
not ask for one independent sentence from each paper, list document summaries,
or concatenate old answers. Prefer these relations, especially the first three:
adjacent modules in a pipeline, metric-to-behavior reasoning, component
hierarchy, method transfer, compatibility judgment, and conflict resolution.

The strongest pattern is a grounded precondition test: one paper states that a
method requires property X, another establishes that the target system lacks X
or instead uses Y, and the answer derives incompatibility or selects a supported
alternative. An adjacent-module item is valid only when the supplied facts state
the actual input/output or prerequisite relation; never invent a unified pipeline.
For compatibility, transfer, and conflict items, put one explicit benchmark
constraint in the question and ask for exactly one decision. The papers need not
have proposed the combined setting, but every method property used in the verdict
must be stated by a source fact. Do not ask a generic "how do these contrast" or
"what do they share" question.

For a three-paper item, every paper must be indispensable to the same inference;
three parallel clauses do not qualify. For a two-paper item, the result must
still be a derived decision, dependency, constraint, or explanation rather than
a side-by-side comparison. Do not expose source QA IDs or the answer in the
question. Include all answer-critical bridge facts in both intermediate_facts
and the final answer, but keep the final answer at or below 900 characters.
Use no outside knowledge. Return JSON only.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--prepared-contexts",
        type=Path,
        help=(
            "Optional QA-only bundle contexts created from the 4211 single-paper "
            "corpus. When set, bypass the legacy fixed-bundle source lookup."
        ),
    )
    parser.add_argument("--verified-source", type=Path, default=DEFAULT_VERIFIED)
    parser.add_argument("--base-source", type=Path, default=DEFAULT_BASE_SOURCE)
    parser.add_argument("--review-metadata-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--source-pdf-dir", type=Path, default=DEFAULT_SOURCE_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--generation-backend",
        choices=("local", "gemini", "claude"),
        default="local",
    )
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--api-model", default="gemini-2.5-flash")
    parser.add_argument("--api-timeout", type=int, default=900)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--api-workers", type=int, default=1)
    parser.add_argument("--gpus", default="2")
    parser.add_argument("--max-memory-gib", default="")
    parser.add_argument("--target", type=int, default=800)
    parser.add_argument("--candidates-per-bundle", type=int, default=5)
    parser.add_argument("--max-accepted-per-bundle", type=int, default=8)
    parser.add_argument("--generation-rounds", type=int, default=2)
    parser.add_argument("--max-bundles", type=int, default=0)
    parser.add_argument(
        "--three-doc-only",
        action="store_true",
        help=(
            "Restrict the prepared-context run to bundles with a semantic "
            "three-document source group and reject every two-document item."
        ),
    )
    parser.add_argument("--min-source-qas", type=int, default=2)
    parser.add_argument("--max-source-qas", type=int, default=5)
    parser.add_argument("--min-three-doc-ratio", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--local-self-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a second local Qwen pass to remove contrived relationships. "
            "Disable only when deliberately building a broad pool that will "
            "still receive strict full-PDF Claude/Gemini review."
        ),
    )
    parser.add_argument(
        "--source-policy",
        choices=(
            "base_single_qa",
            "evidence_facts",
            "audited_evidence_facts",
            "verified_only",
            "audited_keep_fix",
            "fact_valid",
        ),
        default="audited_keep_fix",
        help="Choose which previously reviewed QA may serve as construction facts.",
    )
    parser.add_argument(
        "--include-fact-valid-rejects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse a rejected old QA only when the prior audit still marked its "
            "answer correct, evidence sufficient, and claims supported."
        ),
    )
    parser.add_argument(
        "--require-source-overlap",
        action="store_true",
        help=(
            "Restrict generation to an exact pair of audited QA facts that "
            "share scientific anchor terms and have overlapping but "
            "complementary document coverage."
        ),
    )
    parser.add_argument(
        "--min-shared-terms",
        type=int,
        default=1,
        help=(
            "Minimum number of non-generic scientific anchor terms connecting "
            "each edge of a required source group. Use 2 for precision-first "
            "hard-expansion runs."
        ),
    )
    parser.add_argument(
        "--require-relation-template",
        action="store_true",
        help=(
            "Require complementary semantic roles (for example metric/label "
            "limitations or producer/consumer dependencies) in addition to "
            "shared scientific terms."
        ),
    )
    return parser.parse_args()


def single_api_worker_argv(argv: list[str], worker_index: int) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"--api-workers", "--worker-id"}:
            index += 2
            continue
        if value.startswith("--api-workers=") or value.startswith("--worker-id="):
            index += 1
            continue
        result.append(value)
        index += 1
    result.extend(("--api-workers", "1", "--worker-id", f"hard-cross-api-{worker_index}"))
    return result


def run_parallel_api_workers(worker_count: int) -> None:
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *single_api_worker_argv(sys.argv[1:], worker_index),
            ]
        )
        for worker_index in range(worker_count)
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
        raise SystemExit(
            "Gemini Cross-PDF generation workers failed: "
            + json.dumps(failures, ensure_ascii=False)
        )


def is_aggregate_coordinator(generation_backend: str, worker_id: str | None) -> bool:
    return generation_backend == "local" or worker_id in {None, "hard-cross-api-0"}


@dataclass
class ClaudeGenerator:
    """QA-only Claude adapter with the same checkpoint contract as Gemini."""

    api_key_file: Path
    model: str
    max_new_tokens: int
    temperature: float
    timeout: int
    retries: int
    api_key: str | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        from pku_qa.workflows.review.review_cross_pdf_qa_api import (
            CLAUDE_BASE_URL,
            load_key,
            post_with_retries,
            response_text_claude,
        )

        if self.api_key is None:
            self.api_key = load_key(self.api_key_file)
        system = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        content = [
            {
                "role": "assistant" if message.get("role") == "assistant" else "user",
                "content": str(message.get("content", "")),
            }
            for message in messages
            if message.get("role") != "system"
        ]
        body = {
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": content,
        }
        response = post_with_retries(
            f"{CLAUDE_BASE_URL}/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body,
            self.timeout,
            self.retries,
        )
        return response_text_claude(response.json()).strip()


def accepted_progress(output_dir: Path, fallback: int = 0) -> int:
    try:
        progress = read_json(output_dir / "progress.json")
        return max(0, int(progress.get("completed", fallback)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def source_pdf_path(pdf_dir: Path, paper_id: str) -> Path:
    return resolve_source_pdf_path(paper_id, pdf_dir)


def load_prepared_contexts(
    path: Path, source: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load deterministic QA-only contexts planned independently of old bundles."""
    payload = read_json(path)
    raw_contexts = payload.get("contexts") if "contexts" in payload else payload
    if not isinstance(raw_contexts, dict):
        raise ValueError("Prepared contexts must be an object keyed by bundle ID")
    contexts: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for raw_bundle_id, raw_context in raw_contexts.items():
        bundle_id = str(raw_bundle_id)
        if bundle_id not in source:
            raise ValueError(f"Prepared bundle missing from --source: {bundle_id}")
        if not isinstance(raw_context, dict):
            raise ValueError(f"Prepared context is not an object: {bundle_id}")
        row = raw_context.get("manifest")
        source_qas = raw_context.get("source_qas")
        if not isinstance(row, dict) or str(row.get("id")) != bundle_id:
            raise ValueError(f"Prepared manifest ID mismatch: {bundle_id}")
        if not isinstance(row.get("sources"), list) or len(row["sources"]) not in {2, 3}:
            raise ValueError(f"Prepared bundle must contain 2 or 3 papers: {bundle_id}")
        if not isinstance(source_qas, list) or len(source_qas) < 2:
            raise ValueError(f"Prepared bundle has insufficient source QA: {bundle_id}")
        contexts[bundle_id] = {
            "bundle": source[bundle_id],
            "source_qas": source_qas,
            "manifest": row,
            "prior_bundle_summary": str(
                raw_context.get(
                    "prior_bundle_summary",
                    "QA-only bundle selected by shared high-IDF scientific anchors.",
                )
            ).strip(),
        }
        manifest.append(row)
    manifest.sort(key=lambda row: str(row["id"]))
    return contexts, manifest


def audited_evidence_fact_qas(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Split KEEP/FIX full-PDF audit support into per-document source facts."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for review_item in record.get("review", {}).get("items", []):
        if not isinstance(review_item, dict):
            continue
        decision = str(review_item.get("decision", ""))
        if decision not in {"KEEP", "FIX"}:
            continue
        qa_id = str(review_item.get("qa_id", "unknown"))
        for fact_index, support in enumerate(review_item.get("support", []), start=1):
            if not isinstance(support, dict):
                continue
            fact = str(support.get("supported_fact", "")).strip()
            try:
                merged_page = int(support.get("merged_page"))
                source_doc_number = int(support.get("doc_number"))
            except (TypeError, ValueError):
                continue
            key = normalized_text(fact)
            if (
                not fact
                or merged_page <= 0
                or source_doc_number not in {1, 2, 3}
                or key in seen
            ):
                continue
            seen.add(key)
            result.append(
                {
                    "source_qa_id": f"audited_fact/{qa_id}/{fact_index}",
                    "question": "Which audited fact is stated by this source paper?",
                    "answer": fact,
                    "semantic_text": fact,
                    "evidence_pages": [merged_page],
                    "audit_decision": f"{decision}_EVIDENCE_FACT",
                    "source_doc_number": source_doc_number,
                }
            )
    return result


def load_bundle_contexts(
    source: dict[str, Any],
    verified: dict[str, Any],
    base_source: dict[str, Any],
    review_dir: Path,
    pdf_dir: Path,
    include_fact_valid_rejects: bool = True,
    source_policy: str = "audited_keep_fix",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Recover source maps and retain only previously audited KEEP/FIX QA."""
    contexts: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    for bundle_id in sorted(source):
        review_path = review_dir / f"{bundle_id}.json"
        if not review_path.is_file():
            continue
        record = read_json(review_path)
        source_meta = record.get("metadata", {}).get("source_meta", [])
        if len(source_meta) != 3:
            continue
        review_items = {
            str(item.get("qa_id")): item
            for item in record.get("review", {}).get("items", [])
            if isinstance(item, dict)
        }
        source_qas: list[dict[str, Any]] = []
        seen_content: set[str] = set()
        audited_qas = (
            source[bundle_id].get("QA", {})
            if source_policy in {"audited_keep_fix", "fact_valid"}
            else {}
        )
        for qa_id, qa in audited_qas.items():
            audit = review_items.get(str(qa_id), {})
            criteria = audit.get("criteria", {})
            fact_valid_reject = (
                source_policy == "fact_valid"
                and include_fact_valid_rejects
                and audit.get("decision") == "REJECT"
                and isinstance(criteria, dict)
                and criteria.get("answer_correct") is True
                and criteria.get("evidence_sufficient") is True
                and criteria.get("no_unsupported_claim") is True
            )
            if audit.get("decision") not in {"KEEP", "FIX"} and not fact_valid_reject:
                continue
            question = audit.get("corrected_question") or qa.get("question")
            answer = audit.get("corrected_answer") or qa.get("answer")
            pages = audit.get("corrected_evidence_pages")
            if not pages:
                pages = [
                    row.get("merged_page")
                    for row in audit.get("support", [])
                    if isinstance(row, dict) and row.get("merged_page") is not None
                ]
            if not pages:
                pages = qa.get("evidence_pages", [])
            item = {
                "source_qa_id": f"audited/{qa_id}",
                "question": str(question or "").strip(),
                "answer": str(answer or "").strip(),
                "evidence_pages": sorted({int(page) for page in pages}),
                "audit_decision": (
                    "REJECT_FACTS_VALID" if fact_valid_reject else audit.get("decision")
                ),
            }
            key = normalized_text(item["question"] + "\0" + item["answer"])
            if item["question"] and item["answer"] and key not in seen_content:
                seen_content.add(key)
                source_qas.append(item)

        for qa_id, qa in verified.get(bundle_id, {}).get("QA", {}).items():
            item = {
                "source_qa_id": f"verified/{qa_id}",
                "question": str(qa.get("question", "")).strip(),
                "answer": str(qa.get("answer", "")).strip(),
                "evidence_pages": sorted(
                    {int(page) for page in qa.get("evidence_pages", [])}
                ),
                "audit_decision": "VERIFIED_RELEASE",
            }
            key = normalized_text(item["question"] + "\0" + item["answer"])
            if item["question"] and item["answer"] and key not in seen_content:
                seen_content.add(key)
                source_qas.append(item)

        sources = []
        for meta in source_meta:
            paper_id = str(meta["paper_id"])
            sources.append(
                {
                    "paper_id": paper_id,
                    "title": str(meta["title"]),
                    "source_pdf_path": str(source_pdf_path(pdf_dir, paper_id)),
                    "merged_start_page": int(meta["merged_start_page"]),
                    "merged_end_page": int(meta["merged_end_page"]),
                }
            )
        row = {"id": bundle_id, "sources": sources}
        manifest.append(row)
        if source_policy == "audited_evidence_facts":
            source_qas = audited_evidence_fact_qas(record)
        elif source_policy == "evidence_facts":
            source_qas = []
            seen_content = set()
            for qa_id, qa in verified.get(bundle_id, {}).get("QA", {}).items():
                for fact_index, evidence in enumerate(
                    qa.get("evidence_items", []), start=1
                ):
                    if not isinstance(evidence, dict):
                        continue
                    fact = str(evidence.get("supported_fact", "")).strip()
                    try:
                        merged_page = int(evidence.get("physical_pdf_page"))
                        source_doc_number = int(
                            evidence.get("source_doc_number")
                        )
                    except (TypeError, ValueError):
                        continue
                    item = {
                        "source_qa_id": (
                            f"verified_fact/{qa_id}/{fact_index}"
                        ),
                        "question": (
                            "Which audited fact is stated by this source paper?"
                        ),
                        "answer": fact,
                        "semantic_text": fact,
                        "evidence_pages": [merged_page],
                        "audit_decision": "VERIFIED_EVIDENCE_FACT",
                        "source_doc_number": source_doc_number,
                    }
                    key = normalized_text(fact)
                    if fact and key not in seen_content:
                        seen_content.add(key)
                        source_qas.append(item)
        elif source_policy == "base_single_qa":
            source_qas = []
            seen_content = set()
            for source_number, source_row in enumerate(sources, start=1):
                paper_id = source_row["paper_id"]
                base_paper = base_source.get(paper_id, {})
                for qa_id, qa in base_paper.get("QA", {}).items():
                    merged_pages = []
                    for page in qa.get("evidence_pages", []):
                        try:
                            local_page = int(page)
                        except (TypeError, ValueError):
                            continue
                        merged_page = (
                            int(source_row["merged_start_page"])
                            + local_page
                            - 1
                        )
                        if merged_page <= int(source_row["merged_end_page"]):
                            merged_pages.append(merged_page)
                    item = {
                        "source_qa_id": f"base/{paper_id}/{qa_id}",
                        "question": str(qa.get("question", "")).strip(),
                        "answer": str(qa.get("answer", "")).strip(),
                        "evidence_pages": sorted(set(merged_pages)),
                        "audit_decision": "BASE_4211",
                        "source_doc_number": source_number,
                    }
                    key = normalized_text(
                        item["question"] + "\0" + item["answer"]
                    )
                    if (
                        item["question"]
                        and item["answer"]
                        and item["evidence_pages"]
                        and key not in seen_content
                    ):
                        seen_content.add(key)
                        source_qas.append(item)
        if len(source_qas) >= 2:
            contexts[bundle_id] = {
                "bundle": source[bundle_id],
                "source_qas": source_qas,
                "manifest": row,
                "prior_bundle_summary": str(
                    record.get("review", {}).get("bundle_summary", "")
                ).strip(),
            }
    return contexts, manifest


def docs_for_pages(pages: list[int], manifest: dict[str, Any]) -> list[int]:
    docs: set[int] = set()
    for page in pages:
        matches = [
            index
            for index, source in enumerate(manifest["sources"], start=1)
            if int(source["merged_start_page"])
            <= int(page)
            <= int(source["merged_end_page"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Page {page} maps to {len(matches)} source documents")
        docs.add(matches[0])
    return sorted(docs)


def enrich_source_qas(context: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for qa in context["source_qas"]:
        item = dict(qa)
        item["source_doc_numbers"] = docs_for_pages(
            item["evidence_pages"], context["manifest"]
        )
        result.append(item)
    return result


def prompt_source_qas(
    context: dict[str, Any], source_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    """Expose audited facts and document membership, never evidence pages."""
    return [
        {
            "source_qa_id": item["source_qa_id"],
            "question": item["question"],
            "answer": item["answer"],
            "source_doc_numbers": item["source_doc_numbers"],
        }
        for item in enrich_source_qas(context)
        if source_ids is None or item["source_qa_id"] in source_ids
    ]


def pair_terms(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized_text(value)
        )
        if len(token) >= 5
        and token not in PAIR_STOPWORDS
        and not token.startswith("doc")
    }


def relation_roles(value: Any) -> set[str]:
    """Return conservative semantic roles used to reject mere term overlap."""
    text = normalized_text(value)
    return {
        role
        for role, patterns in RELATION_ROLE_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def relation_template_suggestions(
    group: tuple[dict[str, Any], ...], shared_terms: set[str]
) -> list[str]:
    """Suggest only relations supported by complementary source-fact roles."""
    if not shared_terms:
        return []
    roles = [
        relation_roles(
            item.get("semantic_text")
            or f"{item.get('question', '')} {item.get('answer', '')}"
        )
        for item in group
    ]
    suggestions: set[str] = set()
    flattened = set().union(*roles)
    if "limitation" in flattened and "requirement" in flattened:
        if any("limitation" in row for row in roles) and any(
            "requirement" in row for row in roles
        ):
            suggestions.update({"compatibility_judgment", "conflict_resolution"})
    if "limitation" in flattened and "metric_label" in flattened:
        if sum("metric_label" in row for row in roles) >= 2:
            suggestions.update({"metric_reasoning", "compatibility_judgment"})
    if "metric_label" in flattened and (
        "method" in flattened or "requirement" in flattened
    ):
        suggestions.update({"metric_reasoning", "compatibility_judgment"})
    if "producer" in flattened and "consumer" in flattened:
        if any("producer" in row for row in roles) and any(
            "consumer" in row for row in roles
        ):
            suggestions.add("adjacent_module_dependency")
    if "hierarchy" in flattened and sum("hierarchy" in row for row in roles) >= 2:
        suggestions.add("component_hierarchy")
    if "transfer" in flattened and (
        "limitation" in flattened or "requirement" in flattened
    ):
        suggestions.update({"method_transfer", "compatibility_judgment"})
    return sorted(suggestions)


def semantic_source_pair_hints(
    context: dict[str, Any],
    *,
    min_shared_terms: int = 1,
    limit: int = 24,
    require_relation_template: bool = False,
) -> list[dict[str, Any]]:
    """Find connected 2/3-fact groups spanning distinct source papers."""
    candidates: list[tuple[tuple[int, int, int, int, str], dict[str, Any]]] = []
    sources = enrich_source_qas(context)
    for size in (2, 3, 4):
        for group in combinations(sources, size):
            doc_sets = [set(item["source_doc_numbers"]) for item in group]
            union_docs = sorted(set().union(*doc_sets))
            if len(union_docs) < 2 or (size >= 3 and len(union_docs) < 3):
                continue
            if size == 2 and all(len(docs) > 1 for docs in doc_sets):
                if not doc_sets[0] & doc_sets[1] or doc_sets[0] == doc_sets[1]:
                    continue

            term_sets = [
                pair_terms(
                    item.get("semantic_text")
                    or f"{item['question']} {item['answer']}"
                )
                for item in group
            ]
            adjacency = {index: set() for index in range(size)}
            shared_terms: set[str] = set()
            for left_index, right_index in combinations(range(size), 2):
                overlap = term_sets[left_index] & term_sets[right_index]
                if len(overlap) < min_shared_terms:
                    continue
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
                shared_terms.update(overlap)
            reached = {0}
            frontier = [0]
            while frontier:
                node = frontier.pop()
                for neighbor in adjacency[node] - reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
            if len(reached) != size:
                continue
            source_ids = [item["source_qa_id"] for item in group]
            suggested_relations = relation_template_suggestions(group, shared_terms)
            if require_relation_template and not suggested_relations:
                continue
            row = {
                "source_qa_ids": source_ids,
                "shared_terms": sorted(shared_terms)[:12],
                "suggested_reasoning_types": suggested_relations,
                "source_doc_numbers_by_qa": {
                    item["source_qa_id"]: sorted(doc_sets[index])
                    for index, item in enumerate(group)
                },
                "union_doc_numbers": union_docs,
            }
            rank = (
                int(bool(suggested_relations)),
                len(union_docs),
                size,
                len(shared_terms),
                "\0".join(source_ids),
            )
            candidates.append((rank, row))
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return [row for _, row in candidates[:limit]]


def prior_bundle_is_coherent(context: dict[str, Any]) -> bool:
    """Reject bundles the prior full-PDF audit called thematically invalid."""
    summary = normalized_text(context.get("prior_bundle_summary"))
    negative_patterns = (
        r"\black(?:s|ing)? (?:a |any )?(?:common|coherent|shared)",
        r"\bno (?:clear |meaningful )?(?:common|scientific|thematic)",
        r"\bunrelated\b",
        r"\bdisparate\b",
        r"\blimited (?:common|thematic|scientific)",
        r"\bdo not share\b",
        r"\bacross different domains\b",
        r"\bdifferent domains\b",
    )
    return bool(summary) and not any(
        re.search(pattern, summary) for pattern in negative_patterns
    )


def expected_item(source_ids: list[str] | None = None) -> dict[str, Any]:
    example_ids = source_ids or ["audited/QA1", "audited/QA4"]
    return {
        "question": "One focused question requiring a derived judgment",
        "answer": "Concise answer containing the bridge facts and conclusion",
        "derived_conclusion": "One newly inferred decision or dependency",
        "source_qa_ids": example_ids,
        "reasoning_type": "component_hierarchy",
        "relation": "Why the source facts form one necessary inference",
        "intermediate_facts": [
            {
                "source_qa_id": source_id,
                "doc_numbers": [min(index + 1, 3)],
                "fact": "supported fact from its source paper",
            }
            for index, source_id in enumerate(example_ids)
        ]
        + [{"inference": "bridge step that is explicit in the answer"}],
        "source_necessity_tests": [
            {
                "source_qa_id": source_id,
                "missing_fact_if_removed": "exact missing fact",
                "why_answer_is_impossible": "counterfactual failure",
            }
            for source_id in example_ids
        ],
        "quality_check": {flag: True for flag in QUALITY_FLAGS},
    }


REASONING_TYPE_CYCLE = (
    "adjacent_module_dependency",
    "metric_reasoning",
    "component_hierarchy",
    "adjacent_module_dependency",
    "metric_reasoning",
    "component_hierarchy",
    "method_transfer",
    "conflict_resolution",
    "compatibility_judgment",
)


def choose_target_reasoning_type(
    bundle_id: str,
    generation_round: int,
    pair_hints: list[dict[str, Any]],
) -> str:
    """Rotate relation targets while respecting roles supported by source facts."""
    supported = {
        str(reasoning_type)
        for hint in pair_hints
        for reasoning_type in hint.get("suggested_reasoning_types", [])
    }
    try:
        bundle_offset = int(bundle_id.rsplit("_", 1)[-1])
    except ValueError:
        bundle_offset = sum(ord(char) for char in bundle_id)
    start = (bundle_offset + generation_round) % len(REASONING_TYPE_CYCLE)
    for offset in range(len(REASONING_TYPE_CYCLE)):
        candidate = REASONING_TYPE_CYCLE[
            (start + offset) % len(REASONING_TYPE_CYCLE)
        ]
        if not supported or candidate in supported:
            return candidate
    return REASONING_TYPE_CYCLE[start]


def build_messages(
    bundle_id: str,
    context: dict[str, Any],
    candidate_count: int,
    generation_round: int,
    three_doc_target: bool,
    require_source_overlap: bool = False,
    min_shared_terms: int = 1,
    require_relation_template: bool = False,
) -> list[dict[str, str]]:
    all_pair_hints = (
        semantic_source_pair_hints(
            context,
            min_shared_terms=min_shared_terms,
            require_relation_template=require_relation_template,
        )
        if require_source_overlap
        else []
    )
    three_document_hints = [
        hint
        for hint in all_pair_hints
        if len(hint["union_doc_numbers"]) == 3
    ]
    pair_hints = (
        three_document_hints
        if three_doc_target and three_document_hints
        else all_pair_hints
    )
    target_reasoning_type = choose_target_reasoning_type(
        bundle_id, generation_round, pair_hints
    )
    target_hints = [
        hint
        for hint in pair_hints
        if target_reasoning_type in hint.get("suggested_reasoning_types", [])
    ]
    if target_hints:
        pair_hints = target_hints
    hinted_source_ids = {
        source_id
        for hint in pair_hints
        for source_id in hint["source_qa_ids"]
    }
    source_qas = prompt_source_qas(
        context, hinted_source_ids if require_source_overlap else None
    )
    source_map = [
        {
            "doc_number": index,
            "paper_id": source["paper_id"],
            "title": source["title"],
        }
        for index, source in enumerate(context["manifest"]["sources"], start=1)
    ]
    target_text = (
        "At least the first three viable items must require all three documents."
        if three_doc_target
        else "Generate a mix of indispensable two-document and three-document items."
    )
    prompt = (
        f"bundle_id={bundle_id}\ngeneration_round={generation_round}\n"
        f"Generate {candidate_count} distinct candidates whenever that many can "
        "be grounded without violating the rules; do not return fewer merely "
        f"for brevity. {target_text}\n"
        f"TARGET reasoning_type: {target_reasoning_type}. Every returned item "
        "must use exactly this reasoning_type and one source group whose stated "
        "semantic roles support it. Do not fall back to compatibility_judgment. "
        "Use each reasoning_type at most once in this response when possible. "
        "The exact allowed values are: adjacent_module_dependency, "
        "metric_reasoning, component_hierarchy, method_transfer, "
        "compatibility_judgment, conflict_resolution.\n\n"
        "Return this exact top-level shape:\n"
        + json.dumps(
            {
                "bundle_id": bundle_id,
                "items": [
                    expected_item(
                        pair_hints[0]["source_qa_ids"]
                        if pair_hints
                        else [
                            item["source_qa_id"] for item in source_qas[:2]
                        ]
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nDOCUMENT MAP (titles only; no PDF text):\n"
        + json.dumps(source_map, ensure_ascii=False, indent=2)
        + "\n\nAUDITED SOURCE QA FACTS:\n"
        + json.dumps(source_qas, ensure_ascii=False, indent=2)
        + (
            "\n\nREQUIRED SEMANTIC SOURCE GROUPS:\n"
            + json.dumps(pair_hints, ensure_ascii=False, indent=2)
            + "\nEvery candidate must use exactly one listed source_qa_ids group. "
            "Use one of that group's suggested_reasoning_types when the field is "
            "present. "
            "The shared_terms are a scientific anchor, not permission to repeat "
            "separate descriptions. Use that anchor to make the source-paper "
            "facts complementary constraints in one "
            "decision, dependency, or conflict resolution. Do not refer to papers "
            "as Doc 1, Doc 2, or Doc 3 in the question or answer; name the actual "
            "method, component, metric, or constraint.\n"
            if require_source_overlap
            else ""
        )
        + "\n\nBefore returning an item, remove it if one source can be dropped, "
        "if its answer is old answers placed side by side, or if it asks parallel "
        "subquestions. Every source-fact row must declare doc_numbers, those numbers "
        "must be supported by that source QA, and their union must equal every paper "
        "the item claims to require. The answer must be a newly derived conclusion, "
        "not the answer of any one source QA. Put that single inference in "
        "derived_conclusion and include it inside the final answer. For every source QA, add a "
        "source_necessity_tests counterfactual naming the exact missing fact and why "
        "the answer becomes impossible when that source is removed. Reject an "
        "invented unified pipeline that the papers never define, a claim that a "
        "method solves a limitation from an unrelated domain, and a common-claim "
        "question whose answer is independently stated in every source. A valid "
        "comparison must derive a new taxonomy, selection rule, invariant, "
        "incompatibility, or conflict resolution from related premises. For "
        "compatibility, transfer, or conflict questions, state one target requirement "
        "inside the question and derive one verdict under that requirement; do not "
        "claim the papers actually built the combined system. Reject speculative "
        "'how might' or 'would likely' questions whose answer is not explicitly "
        "checkable from the supplied facts. Later "
        "rounds must use different source "
        "combinations and relations. FINAL FORMAT CHECK: search both question and "
        "answer for Doc 1, Doc 2, Doc 3, document 1, document 2, and document 3; "
        "rewrite every occurrence using the actual named method, component, metric, "
        "rubric, or constraint before returning JSON."
        " FINAL LENGTH CHECK: the answer of every item must be at most 900 "
        "characters while still stating all answer-critical intermediate facts."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text.strip()):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text.strip()[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_response_shape(
    response: dict[str, Any] | None, bundle_id: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Deterministically wrap a lone item while preserving the raw response."""
    if response is None:
        return None, []
    if "items" in response or "bundle_id" in response:
        return response, []
    if "question" in response and "answer" in response:
        return {"bundle_id": bundle_id, "items": [response]}, [
            "wrapped_single_item_response"
        ]
    return response, []


def repair_document_labels(
    generator: Any,
    response: dict[str, Any] | None,
    context: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Rewrite useful candidates that expose internal Doc 1/2/3 labels."""
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        return response, []
    label_pattern = re.compile(r"\bdoc(?:ument)?\s*[123]\b", re.IGNORECASE)
    logs: list[dict[str, Any]] = []
    source_index = {
        row["source_qa_id"]: row for row in prompt_source_qas(context)
    }
    source_map = [
        {"paper_id": source["paper_id"], "title": source["title"]}
        for source in context["manifest"]["sources"]
    ]
    for item_index, item in enumerate(response["items"]):
        if not isinstance(item, dict):
            continue
        visible = " ".join(
            str(item.get(field, ""))
            for field in ("question", "answer", "derived_conclusion")
        )
        if not label_pattern.search(visible):
            continue
        source_rows = [
            source_index[source_id]
            for source_id in item.get("source_qa_ids", [])
            if source_id in source_index
        ]
        prompt = (
            "Rewrite only question, answer, and derived_conclusion so they name "
            "the actual method, component, metric, rubric, or constraint instead "
            "of Doc 1, Doc 2, Doc 3, document 1, document 2, or document 3. "
            "Preserve the exact facts, conclusion, source IDs, and scientific "
            "meaning; add no new claim. Return one JSON object with exactly those "
            "three string fields.\n\nSOURCE TITLES:\n"
            + json.dumps(source_map, ensure_ascii=False, indent=2)
            + "\n\nSOURCE FACTS:\n"
            + json.dumps(source_rows, ensure_ascii=False, indent=2)
            + "\n\nCANDIDATE:\n"
            + json.dumps(
                {
                    field: item.get(field)
                    for field in ("question", "answer", "derived_conclusion")
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raw_text = generator.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You remove internal document-number labels from a "
                        "scientific benchmark item. Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        repaired = extract_json_object(raw_text)
        accepted = False
        if isinstance(repaired, dict):
            values = {
                field: str(repaired.get(field, "")).strip()
                for field in ("question", "answer", "derived_conclusion")
            }
            accepted = all(values.values()) and not label_pattern.search(
                " ".join(values.values())
            )
            if accepted:
                item.update(values)
        logs.append(
            {
                "item_index": item_index,
                "accepted": accepted,
                "raw_response": raw_text,
            }
        )
    return response, logs


def leaked_answer_phrases(
    item: dict[str, Any], context: dict[str, Any]
) -> list[str]:
    """Return final/source answers copied verbatim into a candidate question."""
    question = content_text(item.get("question"))
    phrases = []
    final_answer = str(item.get("answer", "")).strip()
    if answer_phrase_is_leaked(question, final_answer, minimum_length=5):
        phrases.append(final_answer)
    source_index = {
        row["source_qa_id"]: row for row in enrich_source_qas(context)
    }
    for source_id in item.get("source_qa_ids", []):
        source_answer = str(source_index.get(str(source_id), {}).get("answer", "")).strip()
        if answer_phrase_is_leaked(question, source_answer, minimum_length=8):
            phrases.append(source_answer)
    return list(dict.fromkeys(phrase for phrase in phrases if phrase))


def lightly_stemmed_tokens(value: Any) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", normalized_text(value))
    result = []
    for token in tokens:
        if len(token) > 6 and token.endswith("ing"):
            token = token[:-3]
        elif len(token) > 5 and token.endswith("ed"):
            token = token[:-2]
        elif len(token) > 5 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        result.append(token)
    return result


def answer_phrase_is_leaked(
    normalized_question: str,
    answer: Any,
    *,
    minimum_length: int,
) -> bool:
    answer_content = content_text(answer)
    if len(answer_content) < minimum_length:
        return False
    if answer_content in normalized_question:
        return True
    answer_tokens = lightly_stemmed_tokens(answer_content)
    if len(answer_tokens) < 3:
        return False
    question_tokens = set(lightly_stemmed_tokens(normalized_question))
    covered = sum(token in question_tokens for token in answer_tokens)
    return covered / len(answer_tokens) >= 0.7


def question_has_parallel_structure(question: Any) -> bool:
    value = normalized_text(question)
    return bool(
        re.search(r"\b(respectively|first,? .*second|for each document)\b", value)
        or re.search(
            r"\bwhat\b[^?]{0,260}\band\s+(?:how|what|which|determine|does|is)\b",
            value,
        )
    )


def repair_leaked_cross_questions(
    generator: Any,
    response: dict[str, Any] | None,
    context: dict[str, Any],
    *,
    max_repairs: int = 3,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Hide source answers in retained questions without changing their logic."""
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        return response, []
    source_index = {
        row["source_qa_id"]: row for row in prompt_source_qas(context)
    }
    logs = []
    repairs = 0
    for item_index, item in enumerate(response["items"]):
        if not isinstance(item, dict) or repairs >= max_repairs:
            continue
        banned = leaked_answer_phrases(item, context)
        if not banned:
            continue
        source_rows = [
            source_index[source_id]
            for source_id in item.get("source_qa_ids", [])
            if source_id in source_index
        ]
        prompt = (
            "Rewrite only the benchmark question so it still asks for the exact "
            "same derived conclusion and requires every listed source, but does "
            "not state or paraphrase any banned answer phrase. Do not change the "
            "answer, source IDs, conditions, or scientific claim. Do not use "
            "Doc 1/2/3 labels. Return JSON exactly as {\"question\": \"...\"}.\n\n"
            "BANNED ANSWER PHRASES:\n"
            + json.dumps(banned, ensure_ascii=False)
            + "\n\nSOURCE QA (meaning only; do not copy answers):\n"
            + json.dumps(source_rows, ensure_ascii=False, indent=2)
            + "\n\nCANDIDATE:\n"
            + json.dumps(item, ensure_ascii=False, indent=2)
        )
        raw_responses = []
        repaired_question = ""
        accepted = False
        old_question = str(item.get("question", ""))
        for attempt in range(2):
            retry_note = (
                "\n\nThe previous rewrite still paraphrased a banned answer. "
                "Refer only to the named method and the existence of its reported "
                "issue or condition; make the reader retrieve the missing fact."
                if attempt
                else ""
            )
            raw_text = generator.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "You repair answer leakage in cross-paper scientific "
                            "benchmark questions. Return one JSON object only."
                        ),
                    },
                    {"role": "user", "content": prompt + retry_note},
                ]
            )
            raw_responses.append(raw_text)
            parsed = extract_json_object(raw_text)
            repaired_question = (
                str(parsed.get("question", "")).strip()
                if isinstance(parsed, dict)
                else ""
            )
            if not repaired_question:
                continue
            item["question"] = repaired_question
            accepted = (
                not leaked_answer_phrases(item, context)
                and not question_has_parallel_structure(repaired_question)
                and not re.search(
                    r"\bdoc(?:ument)?\s*[123]\b", repaired_question, re.IGNORECASE
                )
            )
            if accepted:
                break
            item["question"] = old_question
        logs.append(
            {
                "item_index": item_index,
                "banned_phrases": banned,
                "accepted": accepted,
                "raw_responses": raw_responses,
            }
        )
        repairs += 1
    return response, logs


def self_review_cross_relations(
    generator: Any,
    response: dict[str, Any] | None,
    bundle_id: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Use local Qwen to reject contrived cross-paper unifications."""
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        return response, None
    source_ids = {
        str(source_id)
        for item in response["items"]
        if isinstance(item, dict)
        for source_id in item.get("source_qa_ids", [])
    }
    source_rows = prompt_source_qas(context, source_ids)
    prompt = (
        f"bundle_id={bundle_id}\n\nSOURCE FACTS:\n"
        + json.dumps(source_rows, ensure_ascii=False, indent=2)
        + "\n\nCANDIDATES:\n"
        + json.dumps(response, ensure_ascii=False, indent=2)
        + "\n\nReturn the same {bundle_id, items} JSON shape. Delete every "
        "candidate that invents a unified pipeline, evaluation framework, audit, "
        "or deployment claim as if it were stated by the papers; merely aggregates "
        "one metric or sentence per paper; creates an arbitrary compatibility "
        "constraint; or compares unrelated "
        "domains just because they share a generic term. A valid item must use "
        "every source fact to answer one natural scientific question through an "
        "explicit shared object, metric, precondition, component dependency, or "
        "real conflict. Three-paper items must have one connected inference, not "
        "three parallel mappings. Do not reject solely because the papers do not "
        "propose one joint system: a benchmark comparison is valid when it derives "
        "a direct taxonomy, ordering, compatibility decision, or shared failure "
        "mechanism from the explicitly named method, metric, component, and "
        "conditions in the source facts. The bridge must follow by standard "
        "technical definitions or one explicit benchmark constraint. A grounded "
        "compatibility/transfer question may impose such a constraint and ask for "
        "one decision even if the papers did not propose the combined setting; "
        "reject it only when a needed method property is unstated. You may "
        "repair wording only if the same source "
        "facts exactly support the repaired conclusion. Keep derived_conclusion "
        "inside the answer and remove Doc 1/2/3 labels. Prefer an empty list over "
        "a plausible but contrived item."
    )
    raw_text = generator.generate(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict cross-paper scientific entailment auditor. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    stripped = raw_text.strip()
    try:
        raw_parsed = json.loads(stripped)
    except json.JSONDecodeError:
        raw_parsed = None
    if isinstance(raw_parsed, list):
        parsed = {"bundle_id": bundle_id, "items": raw_parsed}
    else:
        parsed = extract_json_object(raw_text)
    normalized_empty = parsed == {}
    if normalized_empty:
        parsed = {"bundle_id": bundle_id, "items": []}
    accepted = (
        isinstance(parsed, dict)
        and str(parsed.get("bundle_id")) == bundle_id
        and isinstance(parsed.get("items"), list)
    )
    return (
        parsed if accepted else response,
        {
            "accepted_response": accepted,
            "normalized_empty_object_to_empty_items": normalized_empty,
            "input_item_count": len(response["items"]),
            "output_item_count": (
                len(parsed["items"]) if accepted else len(response["items"])
            ),
            "raw_response": raw_text,
        },
    )


def answer_is_concatenation(answer: str, sources: list[dict[str, Any]]) -> bool:
    normalized_answer = re.sub(r"[^\w]+", " ", normalized_text(answer)).strip()
    source_answers = [
        re.sub(r"[^\w]+", " ", normalized_text(source.get("answer"))).strip()
        for source in sources
        if len(normalized_text(source.get("answer"))) >= 5
    ]
    if len(source_answers) < 2 or not all(
        source_answer in normalized_answer for source_answer in source_answers
    ):
        return False
    residual = normalized_answer
    for source_answer in sorted(source_answers, key=len, reverse=True):
        residual = residual.replace(source_answer, " ", 1)
    residual_tokens = re.findall(r"\w+", residual)
    return len(residual_tokens) < 8


def content_text(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", normalized_text(value)).strip()


def validate_item(
    raw: Any,
    context: dict[str, Any],
    min_source_qas: int,
    max_source_qas: int,
    seen_questions: set[str],
    require_source_overlap: bool = False,
    min_shared_terms: int = 1,
    require_relation_template: bool = False,
    require_three_documents: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, ["item_not_object"]
    errors: list[str] = []
    source_index = {
        item["source_qa_id"]: item for item in enrich_source_qas(context)
    }
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    derived_conclusion = str(raw.get("derived_conclusion", "")).strip()
    raw_ids = raw.get("source_qa_ids")
    source_ids = (
        list(dict.fromkeys(str(value) for value in raw_ids))
        if isinstance(raw_ids, list)
        else []
    )
    if not question:
        errors.append("empty_question")
    if not answer:
        errors.append("empty_answer")
    if not derived_conclusion:
        errors.append("empty_derived_conclusion")
    if len(answer) > 900:
        errors.append("answer_too_long")
    if not min_source_qas <= len(source_ids) <= max_source_qas:
        errors.append("invalid_source_qa_count")
    if any(source_id not in source_index for source_id in source_ids):
        errors.append("unknown_source_qa_id")
    if require_source_overlap:
        allowed_groups = {
            frozenset(row["source_qa_ids"])
            for row in semantic_source_pair_hints(
                context,
                min_shared_terms=min_shared_terms,
                require_relation_template=require_relation_template,
            )
        }
        if (
            len(source_ids) not in {2, 3, 4}
            or frozenset(source_ids) not in allowed_groups
        ):
            errors.append("source_qas_must_match_semantic_group")
    if re.search(r"(?:audited|verified)?/?QA\s*\d+", question, re.IGNORECASE):
        errors.append("question_mentions_source_qa_id")
    if re.search(r"\bdoc(?:ument)?\s*[123]\b", question, re.IGNORECASE):
        errors.append("question_uses_document_labels")
    if re.search(r"\bdoc(?:ument)?\s*[123]\b", answer, re.IGNORECASE):
        errors.append("answer_uses_document_labels")
    question_key = normalized_text(question)
    if question_key in seen_questions:
        errors.append("duplicate_question")
    if answer_phrase_is_leaked(content_text(question), answer, minimum_length=5):
        errors.append("answer_leaked_in_question")
    for source_id in source_ids:
        source_answer = source_index.get(source_id, {}).get("answer")
        if answer_phrase_is_leaked(
            content_text(question), source_answer, minimum_length=8
        ):
            errors.append(f"source_answer_leaked_in_question:{source_id}")
    if question_has_parallel_structure(question):
        errors.append("parallel_subquestions")
    if re.search(
        r"\b(how might|would likely|could potentially)\b",
        question_key,
    ):
        errors.append("speculative_or_hypothetical_question")
    if re.search(
        r"\bhow does\b.{0,220}\b(?:contrast with|relate to)\b",
        question_key,
    ) or re.search(
        r"\b(?:what|which).{0,120}\b(?:shared|common)\b.{0,120}\b"
        r"(?:benchmark|evaluation suite|finding|claim)s?\b",
        question_key,
    ):
        errors.append("generic_cross_document_juxtaposition")
    if re.search(
        r"\b(?:likely|susceptib\w*|risk\w*|optimized for|more conservative|"
        r"could accelerate|may accelerate)\b",
        normalized_text(answer),
    ):
        errors.append("unsupported_speculation_in_answer")

    reasoning_type = str(raw.get("reasoning_type", "")).strip()
    if reasoning_type not in REASONING_TYPES:
        errors.append("invalid_reasoning_type")
    relation = str(raw.get("relation", "")).strip()
    if not relation:
        errors.append("empty_relation")
    intermediate = raw.get("intermediate_facts")
    if not isinstance(intermediate, list) or len(intermediate) < len(source_ids) + 1:
        errors.append("intermediate_facts_too_short")
    else:
        blob = json.dumps(intermediate, ensure_ascii=False)
        for source_id in source_ids:
            if source_id not in blob:
                errors.append(f"intermediate_facts_missing:{source_id}")
    necessity_tests = raw.get("source_necessity_tests")
    necessity_by_id = (
        {
            str(row.get("source_qa_id")): row
            for row in necessity_tests
            if isinstance(row, dict)
        }
        if isinstance(necessity_tests, list)
        else {}
    )
    for source_id in source_ids:
        row = necessity_by_id.get(source_id, {})
        if not str(row.get("missing_fact_if_removed", "")).strip():
            errors.append(f"necessity_test_missing_fact:{source_id}")
        if not str(row.get("why_answer_is_impossible", "")).strip():
            errors.append(f"necessity_test_missing_counterfactual:{source_id}")
    quality = raw.get("quality_check")
    if not isinstance(quality, dict):
        errors.append("quality_check_not_object")
    else:
        for flag in QUALITY_FLAGS:
            if quality.get(flag) is not True:
                errors.append(f"quality_flag_false:{flag}")

    valid_sources = [source_index[source_id] for source_id in source_ids if source_id in source_index]
    if any(
        re.search(
            r"\b(full name|stands for|expand(?:s|ed)? the acronym|acronym)\b",
            normalized_text(source.get("question")),
        )
        for source in valid_sources
    ) and re.search(
        r"\b(full name|formally defined|stands for|identity|acronym)\b",
        normalized_text(f"{question} {relation} {derived_conclusion}"),
    ):
        errors.append("acronym_expansion_stitch")
    evidence_pages = sorted(
        {
            page
            for source in valid_sources
            for page in source.get("evidence_pages", [])
        }
    )
    try:
        source_docs = docs_for_pages(evidence_pages, context["manifest"])
    except ValueError:
        source_docs = []
        errors.append("invalid_evidence_page_mapping")
    if len(source_docs) < 2:
        errors.append("not_cross_document")
    if require_three_documents and len(source_docs) != 3:
        errors.append("requires_three_source_documents")
    fact_docs: set[int] = set()
    if isinstance(intermediate, list):
        for fact in intermediate:
            if not isinstance(fact, dict) or "source_qa_id" not in fact:
                continue
            source_id = str(fact.get("source_qa_id"))
            raw_fact_docs = fact.get("doc_numbers")
            if not isinstance(raw_fact_docs, list) or not raw_fact_docs:
                errors.append(f"intermediate_fact_missing_doc_numbers:{source_id}")
                continue
            try:
                declared_docs = {int(value) for value in raw_fact_docs}
            except (TypeError, ValueError):
                errors.append(f"intermediate_fact_invalid_doc_numbers:{source_id}")
                continue
            supported_docs = set(
                source_index.get(source_id, {}).get("source_doc_numbers", [])
            )
            if not declared_docs <= supported_docs:
                errors.append(f"intermediate_fact_unsupported_doc:{source_id}")
            fact_docs.update(declared_docs)
    if source_docs and fact_docs != set(source_docs):
        errors.append("intermediate_facts_do_not_cover_all_source_documents")
    if valid_sources and answer_is_concatenation(answer, valid_sources):
        errors.append("answer_is_source_answer_concatenation")
    answer_content = content_text(answer)
    conclusion_content = content_text(derived_conclusion)
    if conclusion_content:
        conclusion_tokens = conclusion_content.split()
        covered_tokens = sum(
            token in answer_content.split() for token in conclusion_tokens
        )
        if (
            conclusion_content not in answer_content
            and covered_tokens / len(conclusion_tokens) < 0.35
        ):
            errors.append("derived_conclusion_missing_from_answer")
    for source in valid_sources:
        source_content = content_text(source.get("answer"))
        if not source_content:
            continue
        similarity = SequenceMatcher(
            None, answer_content, source_content
        ).ratio()
        dominance = (
            len(source_content) / len(answer_content)
            if source_content in answer_content and answer_content
            else 0.0
        )
        if similarity >= 0.82 or dominance >= 0.75:
            errors.append(
                f"answer_too_similar_to_source_answer:{source['source_qa_id']}"
            )
    for source in valid_sources:
        if SequenceMatcher(
            None, question_key, normalized_text(source.get("question"))
        ).ratio() >= 0.9:
            errors.append("too_similar_to_source_question")
            break
    if errors:
        return None, errors
    return {
        "question": question,
        "answer": answer,
        "derived_conclusion": derived_conclusion,
        "source_qa_ids": source_ids,
        "source_doc_numbers": source_docs,
        "source_document_count": len(source_docs),
        "reasoning_type": reasoning_type,
        "relation": relation,
        "intermediate_facts": intermediate,
        "source_necessity_tests": necessity_tests,
        "evidence_pages": evidence_pages,
        "evidence_span": max(evidence_pages) - min(evidence_pages),
        "modal_types": ["text"],
        "question_type": "Inferential",
        "question_category": reasoning_type,
        "construction_quality_check": {flag: True for flag in QUALITY_FLAGS},
    }, []


def validate_response(
    response: dict[str, Any],
    bundle_id: str,
    context: dict[str, Any],
    min_source_qas: int,
    max_source_qas: int,
    seen_questions: set[str],
    require_source_overlap: bool = False,
    min_shared_terms: int = 1,
    require_relation_template: bool = False,
    require_three_documents: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if str(response.get("bundle_id")) != bundle_id:
        issues.append({"item_index": None, "errors": ["bundle_id_mismatch"]})
    items = response.get("items")
    if not isinstance(items, list):
        return [], issues + [{"item_index": None, "errors": ["items_not_list"]}]
    accepted = []
    for index, raw in enumerate(items):
        item, errors = validate_item(
            raw,
            context,
            min_source_qas,
            max_source_qas,
            seen_questions,
            require_source_overlap,
            min_shared_terms,
            require_relation_template,
            require_three_documents,
        )
        if errors:
            issues.append({"item_index": index, "errors": errors})
            continue
        assert item is not None
        seen_questions.add(normalized_text(item["question"]))
        accepted.append(item)
    return accepted, issues


def try_lock(lock_dir: Path, name: str):
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{name}.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def release_lock(handle: Any) -> None:
    if handle is not None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def collect_existing(
    raw_dir: Path,
    contexts: dict[str, dict[str, Any]],
    min_source_qas: int,
    max_source_qas: int,
    max_per_bundle: int,
    require_source_overlap: bool = False,
    min_shared_terms: int = 1,
    require_relation_template: bool = False,
    require_three_documents: bool = False,
    progress_heartbeat: Callable[[], None] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    accepted: dict[str, list[dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for path_index, path in enumerate(sorted(raw_dir.glob("*.json"))):
        if progress_heartbeat is not None and path_index % 25 == 0:
            progress_heartbeat()
        payload = read_json(path)
        bundle_id = str(payload.get("bundle_id", path.stem.split("__", 1)[0]))
        normalized_response, _ = normalize_response_shape(
            payload.get("parsed_response"), bundle_id
        )
        if bundle_id not in contexts or not isinstance(normalized_response, dict):
            continue
        items, item_issues = validate_response(
            normalized_response,
            bundle_id,
            contexts[bundle_id],
            min_source_qas,
            max_source_qas,
            seen_questions,
            require_source_overlap,
            min_shared_terms,
            require_relation_template,
            require_three_documents,
        )
        bucket = accepted.setdefault(bundle_id, [])
        bucket.extend(items[: max(0, max_per_bundle - len(bucket))])
        issues.extend({"bundle_id": bundle_id, **issue} for issue in item_issues)
    return accepted, issues


def ordered_items(
    accepted: dict[str, list[dict[str, Any]]],
    target: int,
    min_three_doc_ratio: float,
) -> list[tuple[str, dict[str, Any]]]:
    rows = [(bundle_id, item) for bundle_id, items in accepted.items() for item in items]
    three = [row for row in rows if row[1]["source_document_count"] == 3]
    two = [row for row in rows if row[1]["source_document_count"] == 2]
    wanted_three = min(len(three), int(round(target * min_three_doc_ratio)))
    chosen = three[:wanted_three]
    remaining = target - len(chosen)
    chosen.extend(two[:remaining])
    remaining = target - len(chosen)
    if remaining:
        chosen.extend(three[wanted_three : wanted_three + remaining])
    return chosen[:target]


def build_dataset(
    source: dict[str, Any],
    accepted: dict[str, list[dict[str, Any]]],
    target: int,
    model_name: str,
    min_three_doc_ratio: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bundle_id, item in ordered_items(accepted, target, min_three_doc_ratio):
        bucket = output.setdefault(
            bundle_id,
            {
                "paper": bundle_id,
                "primary_category": source[bundle_id].get("primary_category", ""),
                "secondary_category": source[bundle_id].get("secondary_category", ""),
                "QA": {},
            },
        )
        digest = hashlib.sha256(
            (bundle_id + "\0" + normalized_text(item["question"])).encode("utf-8")
        ).hexdigest()[:16]
        qa_id = f"HQA_{digest}"
        enriched = dict(item)
        enriched["construction_model"] = model_name
        enriched["construction_protocol"] = "qa_only_no_pdf"
        bucket["QA"][qa_id] = enriched
    return output


def write_outputs(
    args: argparse.Namespace,
    source: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    selected: list[str],
    raw_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    def progress_heartbeat() -> None:
        completed = accepted_progress(args.output_dir)
        atomic_json(
            args.output_dir / "progress.json",
            {
                "status": "in_progress",
                "completed": completed,
                "total": args.target,
                "percent": round(100 * completed / args.target, 4),
                "stage": "hard_cross_pdf_generation",
                "work_state": "validating_checkpoints",
                "updated_at": time.time(),
            },
        )

    accepted, issues = collect_existing(
        raw_dir,
        contexts,
        args.min_source_qas,
        args.max_source_qas,
        args.max_accepted_per_bundle,
        args.require_source_overlap,
        args.min_shared_terms,
        args.require_relation_template,
        args.three_doc_only,
        progress_heartbeat,
    )
    construction_model = (
        args.model_path.name
        if args.generation_backend == "local"
        else args.api_model
    )
    dataset = build_dataset(
        source,
        accepted,
        args.target,
        construction_model,
        args.min_three_doc_ratio,
    )
    count = sum(len(bundle["QA"]) for bundle in dataset.values())
    items = [qa for bundle in dataset.values() for qa in bundle["QA"].values()]
    atomic_json(args.output_dir / f"hard_cross_pdf_candidates_{args.target}.json", dataset)
    atomic_json(args.output_dir / "validation_issues.json", issues)
    atomic_json(
        args.output_dir / "summary.json",
        {
            "status": "complete" if count >= args.target else "in_progress",
            "target": args.target,
            "accepted_questions": count,
            "accepted_before_quota_selection": sum(len(rows) for rows in accepted.values()),
            "bundles_used": len(dataset),
            "eligible_bundles": len(selected),
            "three_document_questions": sum(
                item["source_document_count"] == 3 for item in items
            ),
            "three_document_ratio": (
                sum(item["source_document_count"] == 3 for item in items) / count
                if count
                else 0.0
            ),
            "reasoning_type_distribution": dict(
                sorted(Counter(item["reasoning_type"] for item in items).items())
            ),
            "raw_responses": len(list(raw_dir.glob("*.json"))),
            "model": (
                args.model_path.name
                if args.generation_backend == "local"
                else args.api_model
            ),
            "pdf_uploaded_to_generator": False,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    atomic_json(
        args.output_dir / "progress.json",
        {
            "status": "complete" if count >= args.target else "in_progress",
            "completed": count,
            "total": args.target,
            "percent": round(100 * count / args.target, 4),
            "stage": "hard_cross_pdf_generation",
            "updated_at": time.time(),
        },
    )
    return accepted, dataset


def write_readme(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(
        """# Hard Cross-PDF expansion

This directory contains resumable Cross-PDF candidates. The configured local
Qwen, Gemini, or Claude generator receives audited QA text and the document
title map, not PDF content.
Candidates emphasize adjacent modules, metrics, component hierarchy, method
transfer, compatibility, and conflict resolution. Static validation rejects
answer leakage, source-answer concatenation, single-document items, and missing
intermediate facts. Full source PDFs are used only by later Claude/Gemini review.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.api_workers <= 0:
        raise SystemExit("--api-workers must be positive")
    if args.api_workers > 1:
        if args.generation_backend == "local":
            raise SystemExit(
                "--api-workers > 1 requires an API generation backend"
            )
        run_parallel_api_workers(args.api_workers)
        return
    if not 0 <= args.min_three_doc_ratio <= 1:
        raise SystemExit("--min-three-doc-ratio must be between 0 and 1")
    if args.min_shared_terms < 1:
        raise SystemExit("--min-shared-terms must be at least 1")
    source = read_json(args.source)
    if args.prepared_contexts:
        contexts, manifest = load_prepared_contexts(args.prepared_contexts, source)
    else:
        verified = (
            read_json(args.verified_source) if args.verified_source.is_file() else {}
        )
        base_source = read_json(args.base_source)
        contexts, manifest = load_bundle_contexts(
            source,
            verified,
            base_source,
            args.review_metadata_dir,
            args.source_pdf_dir,
            include_fact_valid_rejects=args.include_fact_valid_rejects,
            source_policy=args.source_policy,
        )
    if args.require_source_overlap:
        contexts = {
            bundle_id: context
            for bundle_id, context in contexts.items()
            if prior_bundle_is_coherent(context)
            and semantic_source_pair_hints(
                context,
                min_shared_terms=args.min_shared_terms,
                require_relation_template=args.require_relation_template,
            )
        }
    if args.three_doc_only:
        contexts = {
            bundle_id: context
            for bundle_id, context in contexts.items()
            if any(
                len(hint["union_doc_numbers"]) == 3
                for hint in semantic_source_pair_hints(
                    context,
                    min_shared_terms=args.min_shared_terms,
                    require_relation_template=args.require_relation_template,
                )
            )
        }
    rng = random.Random(args.seed)
    selected = list(contexts)
    rng.shuffle(selected)
    selected.sort(
        key=lambda bundle_id: (
            max(
                (
                    len(hint["union_doc_numbers"]),
                    len(hint["shared_terms"]),
                )
                for hint in semantic_source_pair_hints(
                    contexts[bundle_id],
                    min_shared_terms=args.min_shared_terms,
                    require_relation_template=args.require_relation_template,
                )
            )
            if args.require_source_overlap
            else (0, 0),
            len({doc for qa in enrich_source_qas(contexts[bundle_id]) for doc in qa["source_doc_numbers"]}),
            len(contexts[bundle_id]["source_qas"]),
        ),
        reverse=True,
    )
    if args.max_bundles:
        selected = selected[: args.max_bundles]
    write_readme(args.output_dir)
    atomic_json(args.output_dir / "selected_bundles.json", selected)
    atomic_json(args.output_dir / "bundle_manifest.json", manifest)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = args.output_dir / ".bundle_locks"
    aggregate_coordinator = is_aggregate_coordinator(
        args.generation_backend, args.worker_id
    )
    if aggregate_coordinator:
        _, dataset = write_outputs(args, source, contexts, selected, raw_dir)
        count = sum(len(bundle["QA"]) for bundle in dataset.values())
        last_aggregate_raw_count = len(list(raw_dir.glob("*.json")))
    else:
        dataset = {}
        count = accepted_progress(args.output_dir)
        last_aggregate_raw_count = 0
    if args.prepare_only:
        print(json.dumps({"event": "prepared", "eligible_bundles": len(selected)}))
        return

    if args.generation_backend == "local":
        generator: Any = LocalQwenGenerator(
            model_path=args.model_path,
            physical_gpus=args.gpus,
            max_memory_gib=args.max_memory_gib,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        generator_name = args.model_path.name
    elif args.generation_backend == "gemini":
        generator = GeminiGenerator(
            api_key_file=args.api_key_file,
            model=args.api_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            retries=args.api_retries,
        )
        generator_name = args.api_model
    else:
        generator = ClaudeGenerator(
            api_key_file=args.api_key_file,
            model=args.api_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            retries=args.api_retries,
        )
        generator_name = args.api_model
    while count < args.target:
        processed = False
        pending_locked = False
        for generation_round in range(args.generation_rounds):
            for bundle_id in selected:
                suffix = "" if generation_round == 0 else f"__round{generation_round}"
                raw_path = raw_dir / f"{bundle_id}{suffix}.json"
                if raw_path.exists() and not args.force:
                    continue
                claim = try_lock(lock_dir, raw_path.stem)
                if claim is None:
                    pending_locked = True
                    continue
                try:
                    if raw_path.exists() and not args.force:
                        continue
                    three_target = args.three_doc_only or (
                        (generation_round + int(bundle_id.split("_")[-1])) % 5 != 0
                    )
                    messages = build_messages(
                        bundle_id,
                        contexts[bundle_id],
                        args.candidates_per_bundle,
                        generation_round,
                        three_target,
                        args.require_source_overlap,
                        args.min_shared_terms,
                        args.require_relation_template,
                    )
                    started = time.time()
                    raw_text = generator.generate(messages)
                    raw_parsed_response = extract_json_object(raw_text)
                    parsed_response, shape_normalizations = normalize_response_shape(
                        raw_parsed_response, bundle_id
                    )
                    if args.local_self_review:
                        parsed_response, self_review_log = self_review_cross_relations(
                            generator,
                            parsed_response,
                            bundle_id,
                            contexts[bundle_id],
                        )
                    else:
                        self_review_log = {"skipped": True}
                    parsed_response, document_label_repairs = repair_document_labels(
                        generator, parsed_response, contexts[bundle_id]
                    )
                    parsed_response, leakage_repairs = repair_leaked_cross_questions(
                        generator, parsed_response, contexts[bundle_id]
                    )
                    atomic_json(
                        raw_path,
                        {
                            "bundle_id": bundle_id,
                            "generation_round": generation_round,
                            "worker_id": args.worker_id,
                            "model": generator_name,
                            "elapsed_seconds": round(time.time() - started, 3),
                            "raw_response": raw_text,
                            "raw_parsed_response": raw_parsed_response,
                            "parsed_response": parsed_response,
                            "deterministic_normalizations": shape_normalizations,
                            "self_review": self_review_log,
                            "document_label_repairs": document_label_repairs,
                            "leakage_repairs": leakage_repairs,
                        },
                    )
                finally:
                    release_lock(claim)
                current_raw_count = len(list(raw_dir.glob("*.json")))
                if (
                    aggregate_coordinator
                    and current_raw_count - last_aggregate_raw_count
                    >= AGGREGATE_CHECKPOINT_INTERVAL
                ):
                    _, dataset = write_outputs(
                        args, source, contexts, selected, raw_dir
                    )
                    last_aggregate_raw_count = current_raw_count
                    count = sum(len(bundle["QA"]) for bundle in dataset.values())
                else:
                    count = accepted_progress(args.output_dir, count)
                print(
                    json.dumps(
                        {
                            "event": "bundle_complete",
                            "worker_id": args.worker_id,
                            "bundle_id": bundle_id,
                            "generation_round": generation_round,
                            "accepted_total": count,
                            "target": args.target,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                processed = True
                break
            if processed:
                break
        if processed:
            continue
        if pending_locked:
            time.sleep(3)
            count = accepted_progress(args.output_dir, count)
            continue
        break
    if not aggregate_coordinator:
        print(
            json.dumps(
                {
                    "event": "worker_complete",
                    "worker_id": args.worker_id,
                    "accepted_questions_at_last_aggregate": accepted_progress(
                        args.output_dir, count
                    ),
                    "target": args.target,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return
    _, dataset = write_outputs(args, source, contexts, selected, raw_dir)
    count = sum(len(bundle["QA"]) for bundle in dataset.values())
    if count < args.target:
        raise SystemExit(
            f"search_space_exhausted: accepted={count}/{args.target}; "
            "increase rounds, bundles, or per-bundle limit"
        )


if __name__ == "__main__":
    main()
