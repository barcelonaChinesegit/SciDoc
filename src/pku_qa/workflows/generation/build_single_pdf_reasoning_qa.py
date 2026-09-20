#!/usr/bin/env python3
"""Build reasoning QA by composing related QA from the same paper.

The generator deliberately does not read PDFs or evidence pages.  Each model
call receives every original QA from one paper and may create new questions
only from facts present in those QA pairs.  Raw responses are stored per paper
so a long local-model run is resumable.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Any, Callable

from pku_qa.evaluation.model_paths import (
    configure_model_cache_environment,
    model_directory,
    resolve_local_model_path,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/qa/2.reasoning"
DEFAULT_MODEL = model_directory("Qwen3.6-27B")
DEFAULT_API_KEY_FILE = ROOT / ".env"
REASONING_TYPES = {
    "causal_chain",
    "conditional_inference",
    "comparison",
    "constraint_intersection",
    "multi_step_calculation",
}
REASONING_TYPE_ALIASES = {
    # These aliases differ only in taxonomy wording. They do not change the
    # generated question, answer, sources, derivation, or quality gates.
    "calculation": "multi_step_calculation",
    "intersection_of_constraints": "constraint_intersection",
    "causal_inference": "causal_chain",
    "comparative_inference": "comparison",
    "comparison_inference": "comparison",
}
QUALITY_FLAGS = (
    "requires_all_sources",
    "not_simple_concatenation",
    "answerable_from_source_qas",
    "answer_unique_and_concise",
)
HARD_REASONING_FOCI = {
    "sequential_reasoning",
    "conditional_filtering",
    "similar_concept_discrimination",
    "model_relationship",
    "metric_reasoning",
}
AGGREGATE_CHECKPOINT_INTERVAL = 100
HARD_QUALITY_FLAGS = (
    "answer_not_leaked_in_question",
    "evidence_pages_non_adjacent",
    "intermediate_facts_verified",
    "answer_contains_intermediate_facts",
)


def aggregate_checkpoint_interval(target: int) -> int:
    """Limit repeated full scans while preserving periodic durable progress."""
    return max(AGGREGATE_CHECKPOINT_INTERVAL, target // 10)

SYSTEM_PROMPT = """\
You construct difficult, reliable scientific reasoning QA from existing QA
pairs belonging to one paper. You do not have the PDF. Use only facts explicitly
present in the supplied QA pairs.

Find genuinely related QA pairs and compose a new, self-contained question whose
answer requires a logical inference across 2-4 source QA pairs. Good operations
include a causal or conditional chain, applying a stated method to a stated
result, intersecting constraints, a meaningful comparison that yields a new
conclusion, or a multi-step calculation.

Prefer deterministic entailment: intersect two stated constraints to select one
candidate, combine two stated quantities in a calculation, apply an explicitly
stated prerequisite to a stated system property, or distinguish concepts using
criteria stated in separate QA. Never transfer a conclusion from an infinite to
a finite setting, assign an item to a category without a stated mapping, or infer
causation from two correlated descriptions.

When two or three source QA report comparable numeric values for the same
metric, population, component family, or experimental contrast, prefer an exact
derived quantity: a difference, percentage-point gap, ratio, ordering, range, or
threshold decision. State every operand and the operation in the final answer.
Never reinterpret a numeric difference as causal contribution or scientific
cost unless a source QA explicitly states that causal meaning.

The `reasoning_type` field MUST be exactly one of these five strings:
`causal_chain`, `conditional_inference`, `comparison`,
`constraint_intersection`, `multi_step_calculation`. Never invent a synonym,
combined label, or additional taxonomy value.

Reject mere conjunction: do not ask for two independent facts, do not join old
questions with "and", and do not make the new answer a list/concatenation of old
answers. Do not copy an original question, refer to QA IDs in the new question or answer,
add outside knowledge, or invent missing assumptions. The answer must be short,
unique, and directly judgeable. Return JSON only, without analysis outside JSON.

Do not state every premise in the new question: the supporting paper must remain
necessary. Every cited source QA must contribute materially to the inference.
Prefer one focused conclusion over broad explanatory or multi-part questions.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--generation-backend",
        choices=("local", "gemini"),
        default="local",
        help="Local Qwen or the resumable Gemini JSON generation backend.",
    )
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--api-model", default="gemini-2.5-flash")
    parser.add_argument("--api-timeout", type=int, default=900)
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument(
        "--api-workers",
        type=int,
        default=1,
        help="Parallel checkpoint workers for the Gemini backend.",
    )
    parser.add_argument(
        "--gpus",
        default="2",
        help="Comma-separated physical GPU IDs exposed to the model.",
    )
    parser.add_argument(
        "--max-memory-gib",
        default="",
        help=(
            "Optional comma-separated per-visible-GPU limits, for example "
            "'26,42'. By default limits are derived from currently free memory."
        ),
    )
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--candidates-per-paper", type=int, default=3)
    parser.add_argument("--max-accepted-per-paper", type=int, default=2)
    parser.add_argument(
        "--generation-rounds",
        type=int,
        default=1,
        help=(
            "Independent resumable passes over eligible papers. Later rounds "
            "use separate checkpoints and request alternative reasoning."
        ),
    )
    parser.add_argument(
        "--stable-qa-ids",
        action="store_true",
        help="Use content-derived IDs so expanding the pool cannot renumber items.",
    )
    parser.add_argument("--min-source-qas", type=int, default=2)
    parser.add_argument("--max-source-qas", type=int, default=4)
    parser.add_argument("--min-paper-qas", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2200)
    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Stop after this many selected papers; 0 means no extra limit.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--hard-mode",
        action="store_true",
        help=(
            "Build the harder expansion: require non-adjacent source evidence, "
            "block source-answer leakage, and record a targeted reasoning focus."
        ),
    )
    parser.add_argument(
        "--min-evidence-span",
        type=int,
        default=3,
        help="Minimum physical-page distance between source evidence in hard mode.",
    )
    parser.add_argument(
        "--repair-leakage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In hard mode, ask the same local model once to rewrite leaked questions.",
    )
    parser.add_argument(
        "--require-semantic-chain",
        action="store_true",
        help=(
            "Require an exact semantically connected 2/3-QA group with "
            "non-adjacent evidence and no direct answer-entity substitution."
        ),
    )
    parser.add_argument(
        "--local-self-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use a second local Qwen semantic-entailment pass. Disable only "
            "for a broad pool that still receives strict Claude/Gemini review."
        ),
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Optional identifier used by concurrent dynamic-GPU workers.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate input and write the selected-paper manifest without loading the model.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def single_api_worker_argv(argv: list[str], worker_index: int) -> list[str]:
    """Return child arguments with parent-only worker flags normalized."""
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
    result.extend(("--api-workers", "1", "--worker-id", f"reasoning-api-{worker_index}"))
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
            "Gemini generation workers failed: "
            + json.dumps(failures, ensure_ascii=False)
        )


def is_aggregate_coordinator(generation_backend: str, worker_id: str | None) -> bool:
    """Keep expensive checkpoint aggregation on one Gemini worker."""
    return generation_backend != "gemini" or worker_id in {
        None,
        "reasoning-api-0",
    }


def accepted_progress(output_dir: Path, fallback: int = 0) -> int:
    """Read the coordinator's last durable accepted count."""
    try:
        progress = read_json(output_dir / "progress.json")
        return max(0, int(progress.get("completed", fallback)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def write_generation_progress(
    output_dir: Path,
    target: int,
    completed: int,
    work_state: str,
) -> None:
    """Refresh scheduler liveness without changing the durable QA count."""
    atomic_json(
        output_dir / "progress.json",
        {
            "status": "in_progress",
            "completed": completed,
            "total": target,
            "percent": round(100 * completed / target, 4) if target else 100.0,
            "stage": "reasoning_qa_generation",
            "work_state": work_state,
            "updated_at": time.time(),
        },
    )


def try_paper_lock(lock_dir: Path, paper_id: str):
    """Claim one paper with a process-scoped lock released automatically."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{paper_id}.lock").open(
        "a+", encoding="utf-8"
    )
    try:
        fcntl.flock(
            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
        )
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "paper_id": paper_id,
                "pid": os.getpid(),
                "updated_at": time.time(),
            }
        )
    )
    handle.flush()
    return handle


def release_paper_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def normalized_content(value: Any) -> str:
    return re.sub(r"[^\w]+", " ", normalized_text(value)).strip()


def sequence_ratio_at_least(left: str, right: str, threshold: float) -> bool:
    """Return the exact SequenceMatcher decision after a cheap upper bound."""
    matcher = SequenceMatcher(None, left, right)
    return matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold


class QuestionSimilarityIndex:
    """Index exact SequenceMatcher decisions by safe length and count bounds."""

    def __init__(self) -> None:
        self._exact: set[str] = set()
        self._by_length: dict[int, list[tuple[str, Counter[str]]]] = {}

    def __contains__(self, question: str) -> bool:
        return question in self._exact

    def add(self, question: str) -> None:
        if question in self._exact:
            return
        self._exact.add(question)
        self._by_length.setdefault(len(question), []).append(
            (question, Counter(question))
        )

    def is_near_duplicate(self, question: str, threshold: float) -> bool:
        if question in self._exact:
            return True
        question_length = len(question)
        if not question_length:
            return False
        # real_quick_ratio is 2*min(a,b)/(a+b). Widen the integer range
        # by one on each side so floating-point rounding cannot exclude a pair.
        minimum_length = max(
            0,
            math.floor(threshold * question_length / (2 - threshold)) - 1,
        )
        maximum_length = (
            math.ceil((2 - threshold) * question_length / threshold) + 1
        )
        question_counts = Counter(question)
        for prior_length in range(minimum_length, maximum_length + 1):
            for prior, prior_counts in self._by_length.get(prior_length, []):
                matches = sum(
                    min(count, prior_counts.get(character, 0))
                    for character, count in question_counts.items()
                )
                quick_ratio = 2 * matches / (question_length + prior_length)
                if quick_ratio < threshold:
                    continue
                if SequenceMatcher(None, question, prior).ratio() >= threshold:
                    return True
        return False


def generated_question_is_duplicate(
    question: str,
    seen_questions: set[str] | QuestionSimilarityIndex,
    threshold: float = 0.88,
) -> bool:
    if isinstance(seen_questions, QuestionSimilarityIndex):
        return seen_questions.is_near_duplicate(question, threshold)
    return question in seen_questions or any(
        sequence_ratio_at_least(question, prior, threshold)
        for prior in seen_questions
    )


def normalize_reasoning_type(value: Any) -> tuple[str, str | None]:
    """Normalize only explicitly approved taxonomy synonyms."""
    original = str(value or "").strip()
    normalized = REASONING_TYPE_ALIASES.get(original, original)
    return normalized, original if normalized != original else None


def compact_original_qas(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all QA content while intentionally excluding evidence metadata."""
    result = []
    for qa_id, qa in paper.get("QA", {}).items():
        if not isinstance(qa, dict):
            continue
        result.append(
            {
                "qa_id": str(qa_id),
                "question": str(qa.get("question", "")).strip(),
                "answer": str(qa.get("answer", "")).strip(),
                "question_type": qa.get("question_type"),
                "question_category": qa.get("question_category"),
            }
        )
    return result


def expected_item(
    hard_mode: bool = False,
    example_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    source_ids = example_source_ids or ["QA1", "QA4"]
    item = {
        "question": "A self-contained new reasoning question",
        "answer": "one concise answer",
        "source_qa_ids": source_ids,
        "reasoning_type": "conditional_inference",
        "relation": "Why these source facts form one inference chain",
        "derivation": [
            {"source_qa_id": source_id, "fact": "material source fact"}
            for source_id in source_ids
        ]
        + [{"inference": "the logical step producing the new answer"}],
        "quality_check": {
            "requires_all_sources": True,
            "not_simple_concatenation": True,
            "answerable_from_source_qas": True,
            "answer_unique_and_concise": True,
        },
    }
    if hard_mode:
        item["derived_conclusion"] = (
            "One newly inferred conclusion stated inside the final answer"
        )
        item["reasoning_focus"] = "conditional_filtering"
        item["intermediate_facts"] = [
            "The necessary bridge fact that must also be explicit in the answer"
        ]
        item["source_necessity_tests"] = [
            {
                "source_qa_id": source_id,
                "missing_fact_if_removed": "fact unavailable without this QA",
                "why_answer_is_impossible": "specific counterfactual failure",
            }
            for source_id in source_ids
        ]
        item["quality_check"].update(
            {flag: True for flag in HARD_QUALITY_FLAGS}
        )
    return item


def build_messages(
    paper_id: str,
    paper: dict[str, Any],
    candidates_per_paper: int,
    min_source_qas: int,
    max_source_qas: int,
    generation_round: int = 0,
    hard_mode: bool = False,
    require_semantic_chain: bool = False,
    min_evidence_span: int = 3,
) -> list[dict[str, str]]:
    qas = compact_original_qas(paper)
    hard_instruction = ""
    if hard_mode:
        focus_order = [
            "sequential_reasoning",
            "conditional_filtering",
            "similar_concept_discrimination",
            "model_relationship",
            "metric_reasoning",
        ]
        offset = (
            int(hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:8], 16)
            + generation_round * candidates_per_paper
        ) % len(focus_order)
        requested_foci = [
            focus_order[(offset + index) % len(focus_order)]
            for index in range(min(candidates_per_paper, len(focus_order)))
        ]
        allowed_combinations = hard_source_combinations(
            paper,
            min_source_qas,
            max_source_qas,
            min_span=min_evidence_span,
            limit=80,
        )
        all_chain_hints = semantic_chain_hints(
            paper, min_span=min_evidence_span
        ) if require_semantic_chain else []
        chain_hints = all_chain_hints
        if all_chain_hints:
            batch_size = min(
                len(all_chain_hints), max(6, candidates_per_paper * 3)
            )
            start = (generation_round * batch_size) % len(all_chain_hints)
            chain_hints = [
                all_chain_hints[(start + index) % len(all_chain_hints)]
                for index in range(batch_size)
            ]
        if require_semantic_chain:
            allowed_combinations = [
                row["source_qa_ids"] for row in chain_hints
            ]
            chain_ids = {
                qa_id
                for row in chain_hints
                for qa_id in row["source_qa_ids"]
            }
            qas = [qa for qa in qas if qa["qa_id"] in chain_ids]
        hard_instruction = (
            "\nHARD EXPANSION RULES:\n"
            "- Use exactly one reasoning_focus from: sequential_reasoning, "
            "conditional_filtering, similar_concept_discrimination, "
            "model_relationship, metric_reasoning.\n"
            "- Cover these requested foci in order when the source facts support "
            "them; do not emit several candidates with the same focus: "
            + json.dumps(requested_foci)
            + ".\n"
            "- Do not reveal any cited source QA answer in the question. The "
            "question must remain impossible to answer from its wording alone.\n"
            "- State the bridge facts in intermediate_facts and repeat every "
            "answer-critical bridge fact in the final answer; do not jump from "
            "premises to a bare conclusion.\n"
            "- Prefer sequential application, filtering by multiple conditions, "
            "disambiguating similar concepts/models, or relating metrics to model "
            "behavior. Avoid an ordinary comparison.\n"
            "- Evidence-page distance is checked after generation without exposing "
            "page metadata to you. Set the four hard quality flags true only after "
            "checking the content requirements you can observe.\n"
            "- Use one of the source-QA combinations in the allow-list below. The "
            "list was precomputed for structural separation, but contains no page "
            "numbers or evidence content.\n"
            "- Never wrap a direct source question in decorative context. If the "
            "final answer equals a terminal source answer, the new question must "
            "hide the terminal question's identifying entity so the preceding "
            "source is indispensable.\n"
            "- For every source QA, provide one source_necessity_tests entry with "
            "the exact missing fact and why the answer becomes impossible if that "
            "source is removed. A source used only as a name, definition, "
            "inspiration, or background preamble is not necessary; drop it.\n"
            "- Ask one focused inferential question, not two independent clauses "
            "joined by 'and'.\n"
            + (
                "- SEMANTIC RELATION MODE: use exactly one 2/3-QA group below. "
                "Use a different group for every emitted candidate. "
                "The group is linked by shared scientific terms, excludes direct "
                "answer-entity substitution, and has distant evidence. Derive a "
                "single new decision, dependency, filtered result, calculation, "
                "or concept distinction that no source QA answers alone. Put the "
                "new inference in derived_conclusion and include it in the final "
                "answer. For a comparable_numeric_derivation hint, use only the "
                "listed exact values and an allowed arithmetic operation; the "
                "answer must show the operands, operation, and result, without "
                "claiming that a gap is a cause, contribution, or cost.\n"
                "For an identity_hidden_sequential_chain hint, the question must "
                "not name or paraphrase bridge_entity or terminal_fact. Refer to "
                "the bridge entity only through the identifying condition in the "
                "first source question; ask for the downstream property supplied "
                "by the second source. The answer must state the bridge entity, "
                "the terminal fact, and their logical link in one explanation.\n"
                "SEMANTIC RELATION HINTS:\n"
                + json.dumps(chain_hints, ensure_ascii=False)
                + "\n"
                if require_semantic_chain
                else ""
            )
            +
            "ALLOWED SOURCE QA COMBINATIONS:\n"
            + json.dumps(allowed_combinations, ensure_ascii=False)
            + "\n"
        )
    user = (
        f"paper_id={paper_id}\n"
        f"generation_round={generation_round}\n"
        f"Generate up to {candidates_per_paper} high-quality candidates. "
        f"Each candidate must use {min_source_qas}-{max_source_qas} source QA pairs. "
        "For every item, reasoning_type must be exactly one of: "
        "causal_chain, conditional_inference, comparison, "
        "constraint_intersection, multi_step_calculation. "
        "It is acceptable to return fewer candidates or an empty list when the "
        "source QA do not support a genuine inference.\n\n"
        "Return this exact top-level shape:\n"
        + json.dumps(
            {
                "paper_id": paper_id,
                "items": [
                    expected_item(
                        hard_mode,
                        (
                            chain_hints[0]["source_qa_ids"]
                            if hard_mode and chain_hints
                            else None
                        ),
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n\nAll original QA from this paper (no PDF or evidence pages):\n"
        + json.dumps(qas, ensure_ascii=False, indent=2)
        + hard_instruction
        + "\n\nBefore including an item, verify inside your reasoning that every "
        "source fact is necessary and that the answer is not just the old answers "
        "placed side by side. Do not repeat an obvious question from an earlier "
        "round; seek a different relation or source-QA combination. Expose only "
        "the concise derivation JSON."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def strip_parenthetical_source_citations(
    response: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Remove generator-only ``(QA5)`` labels without changing QA meaning."""
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        return response, []
    changes: list[dict[str, Any]] = []
    pattern = re.compile(r"\s*\(\s*(?:from\s+)?QA\s*\d+\s*\)", re.IGNORECASE)
    for item_index, item in enumerate(response["items"]):
        if not isinstance(item, dict):
            continue
        for field in ("question", "answer", "derived_conclusion"):
            original = item.get(field)
            if not isinstance(original, str):
                continue
            normalized = pattern.sub("", original).strip()
            if normalized != original:
                item[field] = normalized
                changes.append(
                    {
                        "item_index": item_index,
                        "field": field,
                        "normalization": "removed_parenthetical_source_qa_citation",
                    }
                )
    return response, changes


def leaked_phrases(
    item: dict[str, Any], paper: dict[str, Any]
) -> list[str]:
    """Return final/source answers copied into a generated question."""
    question = normalized_content(item.get("question"))
    phrases: list[str] = []
    answer = str(item.get("answer", "")).strip()
    if len(normalized_content(answer)) >= 4 and normalized_content(answer) in question:
        phrases.append(answer)
    for source_id, source_qa in paper.get("QA", {}).items():
        source_answer = str(source_qa.get("answer", "")).strip()
        if (
            len(normalized_content(source_answer)) >= 5
            and normalized_content(source_answer) in question
        ):
            phrases.append(source_answer)
    return list(dict.fromkeys(phrase for phrase in phrases if phrase))


def repair_leaked_questions(
    generator: Any,
    response: dict[str, Any] | None,
    paper: dict[str, Any],
    *,
    max_repairs: int = 3,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Repair leakage or a missing answer bridge with the same local model."""
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        return response, []
    logs: list[dict[str, Any]] = []
    repairs = 0
    for index, item in enumerate(response["items"]):
        if not isinstance(item, dict) or repairs >= max_repairs:
            continue
        banned = leaked_phrases(item, paper)
        needs_answer_bridge = (
            len(re.findall(r"\w+", str(item.get("answer", "")))) < 8
            or item.get("quality_check", {}).get(
                "answer_contains_intermediate_facts"
            )
            is not True
        )
        if not banned and not needs_answer_bridge:
            continue
        source_qas = []
        for source_id in item.get("source_qa_ids", []):
            qa = paper.get("QA", {}).get(str(source_id))
            if isinstance(qa, dict):
                source_qas.append(
                    {
                        "qa_id": str(source_id),
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                    }
                )
        prompt = (
            "Repair the candidate while preserving its source IDs, conclusion, "
            "inference, and scientific meaning. Rewrite the question only when "
            "needed so it does not contain or paraphrase a banned answer phrase "
            "and still requires paper evidence. Expand the answer when needed so "
            "it contains at least eight words and explicitly states the key "
            "intermediate bridge fact; do not concatenate independent source "
            "answers. Return JSON exactly as "
            "{\"question\": \"...\", \"answer\": \"...\"}.\n\n"
            "BANNED ANSWER PHRASES:\n"
            + json.dumps(banned, ensure_ascii=False)
            + "\n\nCANDIDATE:\n"
            + json.dumps(
                {
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "intermediate_facts": item.get("intermediate_facts"),
                    "derivation": item.get("derivation"),
                    "needs_answer_bridge": needs_answer_bridge,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n\nSOURCE QA (for meaning only; do not copy their answers):\n"
            + json.dumps(source_qas, ensure_ascii=False, indent=2)
        )
        raw_text = generator.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You repair answer leakage in scientific benchmark "
                        "questions. Output one JSON object only."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )
        parsed = extract_json_object(raw_text)
        repaired_question = (
            str(parsed.get("question", "")).strip()
            if isinstance(parsed, dict)
            else ""
        )
        repaired_answer = (
            str(parsed.get("answer", "")).strip()
            if isinstance(parsed, dict)
            else ""
        )
        accepted = bool(repaired_question and repaired_answer)
        if accepted:
            old_question = str(item.get("question", ""))
            old_answer = str(item.get("answer", ""))
            item["question"] = repaired_question
            item["answer"] = repaired_answer
            accepted = (
                not leaked_phrases(item, paper)
                and len(re.findall(r"\w+", repaired_answer)) >= 8
            )
            if not accepted:
                item["question"] = old_question
                item["answer"] = old_answer
            else:
                item.setdefault("quality_check", {})[
                    "answer_contains_intermediate_facts"
                ] = True
        logs.append(
            {
                "item_index": index,
                "banned_phrases": banned,
                "accepted": accepted,
                "raw_response": raw_text,
            }
        )
        repairs += 1
    return response, logs


def answer_is_simple_concatenation(
    answer: str,
    source_ids: list[str],
    original_by_id: dict[str, dict[str, Any]],
) -> bool:
    normalized_answer = normalized_content(answer)
    source_answers = [
        normalized_content(original_by_id[source_id].get("answer"))
        for source_id in source_ids
    ]
    source_answers = [value for value in source_answers if len(value) >= 3]
    if len(source_answers) < 2:
        return False
    contained = sum(value in normalized_answer for value in source_answers)
    return contained == len(source_answers)


def self_review_semantic_relations(
    generator: Any,
    response: dict[str, Any] | None,
    paper_id: str,
    paper: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Use the same local model to remove logically unsupported candidates."""
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        return response, None
    source_ids = {
        str(source_id)
        for item in response["items"]
        if isinstance(item, dict)
        for source_id in item.get("source_qa_ids", [])
    }
    source_qas = [
        {
            "qa_id": str(qa_id),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
        }
        for qa_id, qa in paper.get("QA", {}).items()
        if str(qa_id) in source_ids and isinstance(qa, dict)
    ]
    prompt = (
        f"paper_id={paper_id}\n\nSOURCE QA:\n"
        + json.dumps(source_qas, ensure_ascii=False, indent=2)
        + "\n\nCANDIDATES:\n"
        + json.dumps(response, ensure_ascii=False, indent=2)
        + "\n\nReturn the same top-level {paper_id, items} JSON shape. "
        "Delete every item unless its conclusion is logically entailed by all "
        "cited source QA and no one source QA answers the candidate's exact "
        "wording alone. In a valid identity-hidden chain, a terminal value may "
        "equal the second source answer because the first source is still needed "
        "to resolve the unnamed bridge entity; retain it only when the question "
        "hides that entity and the answer states both bridge and terminal facts. "
        "Reject changed-"
        "condition extrapolation (for example infinity to a fixed finite value), "
        "invented mappings (for example assigning a component to a phase without "
        "a source stating that mapping), decorative name resolution, ordinary "
        "side-by-side comparison, and plausible outside knowledge. A valid item "
        "must derive a new calculation, constraint intersection, necessary order, "
        "selection rule, incompatibility, or concept distinction. You may repair "
        "wording or the answer only when the corrected conclusion follows exactly "
        "from the same source IDs. Exact arithmetic over comparable values is "
        "valid when the answer shows all operands and the operation, but a numeric "
        "gap must not be relabeled as a cause, contribution, or cost. Keep "
        "derived_conclusion explicit in the answer."
    )
    raw_text = generator.generate(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict logical entailment auditor for scientific "
                    "reasoning QA. Prefer returning an empty items list to "
                    "retaining a merely plausible or trivial item. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    )
    try:
        raw_parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        raw_parsed = None
    if isinstance(raw_parsed, list):
        parsed = {"paper_id": paper_id, "items": raw_parsed}
    else:
        parsed = extract_json_object(raw_text)
    normalized_empty = parsed == {}
    if normalized_empty:
        parsed = {"paper_id": paper_id, "items": []}
    accepted = (
        isinstance(parsed, dict)
        and str(parsed.get("paper_id")) == paper_id
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


def validate_item(
    raw: Any,
    original_qas: list[dict[str, Any]],
    min_source_qas: int,
    max_source_qas: int,
    seen_questions: set[str] | QuestionSimilarityIndex,
    *,
    hard_mode: bool = False,
    paper: dict[str, Any] | None = None,
    min_evidence_span: int = 3,
    require_semantic_chain: bool = False,
    semantic_hints: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, ["item_not_object"]

    errors: list[str] = []
    original_by_id = {str(qa["qa_id"]): qa for qa in original_qas}
    question = str(raw.get("question", "")).strip()
    answer = str(raw.get("answer", "")).strip()
    derived_conclusion = str(raw.get("derived_conclusion", "")).strip()
    source_ids_raw = raw.get("source_qa_ids")
    if not isinstance(source_ids_raw, list):
        source_ids: list[str] = []
        errors.append("source_qa_ids_not_list")
    else:
        source_ids = list(dict.fromkeys(str(value) for value in source_ids_raw))

    if not question:
        errors.append("empty_question")
    if not answer:
        errors.append("empty_answer")
    if len(answer) > 300:
        errors.append("answer_too_long")
    if not min_source_qas <= len(source_ids) <= max_source_qas:
        errors.append("invalid_source_qa_count")
    if any(source_id not in original_by_id for source_id in source_ids):
        errors.append("unknown_source_qa_id")
    if re.search(r"\bQA\s*\d+\b", question, flags=re.IGNORECASE):
        errors.append("question_mentions_qa_id")
    if re.search(r"\bQA\s*\d+\b", answer, flags=re.IGNORECASE):
        errors.append("answer_mentions_qa_id")

    if hard_mode:
        if len(re.findall(r"\w+", answer, flags=re.UNICODE)) < 8:
            errors.append("answer_too_short_for_intermediate_facts")
        if normalized_content(answer) and len(normalized_content(answer)) >= 4:
            if normalized_content(answer) in normalized_content(question):
                errors.append("answer_leaked_in_question")
        for source_id, source_qa in original_by_id.items():
            source_answer = normalized_content(source_qa.get("answer"))
            if len(source_answer) >= 5 and source_answer in normalized_content(question):
                errors.append(f"paper_answer_leaked_in_question:{source_id}")

    question_key = normalized_text(question)
    if generated_question_is_duplicate(question_key, seen_questions):
        errors.append("duplicate_generated_question")
    for qa in original_qas:
        old_question = str(qa.get("question", ""))
        if sequence_ratio_at_least(
            question_key, normalized_text(old_question), 0.9
        ):
            errors.append("too_similar_to_original_question")
            break

    reasoning_type, original_reasoning_type = normalize_reasoning_type(
        raw.get("reasoning_type", "")
    )
    if reasoning_type not in REASONING_TYPES:
        errors.append("invalid_reasoning_type")
    reasoning_focus = str(raw.get("reasoning_focus", "")).strip()
    if hard_mode and reasoning_focus not in HARD_REASONING_FOCI:
        errors.append("invalid_reasoning_focus")
    if require_semantic_chain and paper is not None:
        available_semantic_hints = (
            semantic_hints
            if semantic_hints is not None
            else semantic_chain_hints(paper, min_evidence_span)
        )
        semantic_groups = {
            frozenset(row["source_qa_ids"])
            for row in available_semantic_hints
        }
        if (
            len(source_ids) not in {2, 3}
            or frozenset(source_ids) not in semantic_groups
        ):
            errors.append("semantic_relation_must_use_exact_group")
        if not derived_conclusion:
            errors.append("empty_derived_conclusion")
        else:
            answer_content = normalized_content(answer)
            conclusion_content = normalized_content(derived_conclusion)
            conclusion_tokens = conclusion_content.split()
            coverage = (
                sum(token in answer_content.split() for token in conclusion_tokens)
                / len(conclusion_tokens)
                if conclusion_tokens
                else 0.0
            )
            if conclusion_content not in answer_content and coverage < 0.5:
                errors.append("derived_conclusion_missing_from_answer")
            for source_id in source_ids:
                source_answer = normalized_content(
                    original_by_id.get(source_id, {}).get("answer")
                )
                if (
                    source_answer
                    and sequence_ratio_at_least(
                        conclusion_content, source_answer, 0.85
                    )
                ):
                    errors.append(
                        f"derived_conclusion_copies_source_answer:{source_id}"
                    )
    relation = str(raw.get("relation", "")).strip()
    if not relation:
        errors.append("empty_relation")

    derivation = raw.get("derivation")
    if not isinstance(derivation, list) or len(derivation) < len(source_ids) + 1:
        errors.append("derivation_too_short")
    else:
        derivation_blob = json.dumps(derivation, ensure_ascii=False)
        for source_id in source_ids:
            if source_id not in derivation_blob:
                errors.append(f"derivation_missing_{source_id}")

    necessity_tests = raw.get("source_necessity_tests")
    if hard_mode:
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
                errors.append(
                    f"necessity_test_missing_counterfactual:{source_id}"
                )

    quality = raw.get("quality_check")
    if not isinstance(quality, dict):
        errors.append("quality_check_not_object")
    else:
        for flag in QUALITY_FLAGS:
            if quality.get(flag) is not True:
                errors.append(f"quality_flag_false:{flag}")
        if hard_mode:
            for flag in HARD_QUALITY_FLAGS:
                if quality.get(flag) is not True:
                    errors.append(f"quality_flag_false:{flag}")

    intermediate_facts = raw.get("intermediate_facts")
    if hard_mode and (
        not isinstance(intermediate_facts, list)
        or not intermediate_facts
        or any(not str(fact).strip() for fact in intermediate_facts)
    ):
        errors.append("invalid_intermediate_facts")

    evidence_pages: list[int] = []
    if hard_mode and paper is not None:
        for source_id in source_ids:
            source_qa = paper.get("QA", {}).get(source_id, {})
            for page in source_qa.get("evidence_pages", []):
                try:
                    evidence_pages.append(int(page))
                except (TypeError, ValueError):
                    errors.append(f"invalid_source_evidence_page:{source_id}")
        unique_pages = sorted(set(evidence_pages))
        if len(unique_pages) < 2:
            errors.append("source_evidence_not_on_distinct_pages")
        elif not select_separated_pages(
            source_ids, paper, min_evidence_span
        ):
            errors.append("source_key_evidence_pages_too_close")

    matching_semantic_hint = None
    if require_semantic_chain and paper is not None:
        matching_semantic_hint = next(
            (
                row
                for row in available_semantic_hints
                if row["source_qa_ids"] == source_ids
            ),
            None,
        )
    if (
        source_ids
        and all(source_id in original_by_id for source_id in source_ids)
        and answer_is_simple_concatenation(answer, source_ids, original_by_id)
        and (
            not matching_semantic_hint
            or matching_semantic_hint.get("relation_hint")
            != "identity_hidden_sequential_chain"
        )
    ):
        errors.append("answer_is_source_answer_concatenation")

    if (
        matching_semantic_hint
        and matching_semantic_hint.get("relation_hint")
        == "identity_hidden_sequential_chain"
    ):
        bridge = normalized_content(matching_semantic_hint.get("bridge_entity"))
        terminal = normalized_content(matching_semantic_hint.get("terminal_fact"))
        if bridge not in normalized_content(answer):
            errors.append("identity_chain_answer_missing_bridge_entity")
        terminal_tokens = terminal.split()
        terminal_coverage = (
            sum(token in normalized_content(answer).split() for token in terminal_tokens)
            / len(terminal_tokens)
            if terminal_tokens
            else 0.0
        )
        if terminal not in normalized_content(answer) and terminal_coverage < 0.75:
            errors.append("identity_chain_answer_missing_terminal_fact")

    if errors:
        return None, errors

    normalized = {
        "question": question,
        "answer": answer,
        "source_qa_ids": source_ids,
        "reasoning_type": reasoning_type,
        "relation": relation,
        "derivation": derivation,
        "construction_quality_check": {
            flag: True for flag in QUALITY_FLAGS
        },
    }
    if hard_mode:
        normalized["reasoning_focus"] = reasoning_focus
        if require_semantic_chain:
            normalized["derived_conclusion"] = derived_conclusion
        normalized["intermediate_facts"] = [
            str(fact).strip() for fact in intermediate_facts
        ]
        normalized["source_necessity_tests"] = necessity_tests
        normalized["source_evidence_pages"] = sorted(set(evidence_pages))
        normalized["construction_key_evidence_pages"] = (
            select_separated_pages(
                source_ids, paper or {}, min_evidence_span
            )
            or []
        )
        normalized["evidence_span"] = (
            max(evidence_pages) - min(evidence_pages) if evidence_pages else 0
        )
        normalized["hard_construction_quality_check"] = {
            flag: True for flag in HARD_QUALITY_FLAGS
        }
    if original_reasoning_type is not None:
        normalized["reasoning_type_normalization"] = {
            "original": original_reasoning_type,
            "canonical": reasoning_type,
            "rule": "approved_taxonomy_alias",
        }
    return normalized, []


def validate_response(
    response: dict[str, Any],
    paper_id: str,
    paper: dict[str, Any],
    min_source_qas: int,
    max_source_qas: int,
    seen_questions: set[str] | QuestionSimilarityIndex,
    *,
    hard_mode: bool = False,
    min_evidence_span: int = 3,
    require_semantic_chain: bool = False,
    original_qas: list[dict[str, Any]] | None = None,
    semantic_hints: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if str(response.get("paper_id")) != paper_id:
        issues.append({"item_index": None, "errors": ["paper_id_mismatch"]})
    items = response.get("items")
    if not isinstance(items, list):
        return [], issues + [{"item_index": None, "errors": ["items_not_list"]}]

    accepted = []
    original_qas = original_qas or compact_original_qas(paper)
    for index, item in enumerate(items):
        normalized, errors = validate_item(
            item,
            original_qas,
            min_source_qas,
            max_source_qas,
            seen_questions,
            hard_mode=hard_mode,
            paper=paper,
            min_evidence_span=min_evidence_span,
            require_semantic_chain=require_semantic_chain,
            semantic_hints=semantic_hints,
        )
        if errors:
            issues.append({"item_index": index, "errors": errors})
            continue
        assert normalized is not None
        seen_questions.add(normalized_text(normalized["question"]))
        accepted.append(normalized)
    return accepted, issues


def select_papers(
    dataset: dict[str, Any],
    min_paper_qas: int,
    seed: int,
    *,
    hard_mode: bool = False,
    min_evidence_span: int = 3,
    require_semantic_chain: bool = False,
) -> list[str]:
    eligible = [
        str(paper_id)
        for paper_id, paper in dataset.items()
        if isinstance(paper, dict)
        and len(compact_original_qas(paper)) >= min_paper_qas
    ]
    if hard_mode:
        eligible = [
            paper_id
            for paper_id in eligible
            if paper_has_evidence_span(dataset[paper_id], min_evidence_span)
        ]
    if require_semantic_chain:
        eligible = [
            paper_id
            for paper_id in eligible
            if semantic_chain_hints(dataset[paper_id], min_evidence_span)
        ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    # Comparable numeric relations admit exact, auditable derivations, so place
    # them first in semantic-relation mode.  QA count remains the secondary
    # signal, while shuffle provides deterministic tie-breaking and diversity.
    eligible.sort(
        key=lambda paper_id: (
            (
                max(
                    (
                        2
                        if hint.get("relation_hint")
                        == "identity_hidden_sequential_chain"
                        else 1
                        if hint.get("relation_hint")
                        == "comparable_numeric_derivation"
                        else 0
                    )
                    for hint in semantic_chain_hints(
                        dataset[paper_id], min_evidence_span
                    )
                )
                if require_semantic_chain
                else 0
            ),
            len(compact_original_qas(dataset[paper_id])),
        ),
        reverse=True,
    )
    return eligible


def paper_has_evidence_span(paper: dict[str, Any], min_span: int) -> bool:
    """Return whether two source QA expose sufficiently separated gold pages."""
    page_groups: list[list[int]] = []
    for qa in paper.get("QA", {}).values():
        pages: list[int] = []
        for page in qa.get("evidence_pages", []):
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                continue
        if pages:
            page_groups.append(pages)
    return any(
        abs(left_page - right_page) >= min_span
        for left_index, left in enumerate(page_groups)
        for right in page_groups[left_index + 1 :]
        for left_page in left
        for right_page in right
    )


def hard_source_combinations(
    paper: dict[str, Any],
    min_source_qas: int,
    max_source_qas: int,
    *,
    min_span: int,
    limit: int,
) -> list[list[str]]:
    """Select structurally separated source-ID sets without exposing pages."""
    qa_pages: dict[str, list[int]] = {}
    for qa_id, qa in paper.get("QA", {}).items():
        pages: list[int] = []
        for page in qa.get("evidence_pages", []):
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                continue
        if pages:
            qa_pages[str(qa_id)] = pages
    ranked: list[tuple[int, int, list[str]]] = []
    ids = list(qa_pages)
    for size in range(min_source_qas, min(max_source_qas, len(ids)) + 1):
        for source_ids in combinations(ids, size):
            selected_pages = select_separated_pages(
                list(source_ids), paper, min_span
            )
            if not selected_pages:
                continue
            ranked.append(
                (
                    max(selected_pages) - min(selected_pages),
                    size,
                    list(source_ids),
                )
            )
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [source_ids for _, _, source_ids in ranked[:limit]]


def select_separated_pages(
    source_ids: list[str], paper: dict[str, Any], min_span: int
) -> list[int] | None:
    """Choose maximally separated key pages, one from every source QA."""
    groups: list[list[int]] = []
    for source_id in source_ids:
        pages: list[int] = []
        for page in paper.get("QA", {}).get(source_id, {}).get(
            "evidence_pages", []
        ):
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                continue
        if not pages:
            return None
        groups.append(sorted(set(pages)))
    viable: list[tuple[int, int, tuple[int, ...]]] = []
    for selected in product(*groups):
        if len(set(selected)) != len(selected):
            continue
        distances = [
            abs(left - right)
            for index, left in enumerate(selected)
            for right in selected[index + 1 :]
        ]
        if distances and min(distances) >= min_span:
            viable.append(
                (min(distances), max(selected) - min(selected), selected)
            )
    if not viable:
        return None
    _, _, best = max(viable, key=lambda row: (row[0], row[1], row[2]))
    return list(best)


def semantic_chain_hints(
    paper: dict[str, Any], min_span: int
) -> list[dict[str, Any]]:
    """Find semantically connected, non-substitutional 2/3-QA groups."""
    stopwords = {
        "about",
        "according",
        "approach",
        "based",
        "between",
        "does",
        "from",
        "method",
        "many",
        "model",
        "paper",
        "result",
        "specific",
        "study",
        "their",
        "using",
        "which",
    }

    def terms(qa: dict[str, Any]) -> set[str]:
        return {
            token
            for token in re.findall(
                r"[a-z0-9][a-z0-9.+-]*",
                normalized_text(
                    f"{qa.get('question', '')} {qa.get('answer', '')}"
                ),
            )
            if len(token) >= 5 and token not in stopwords
        }

    def simple_numeric_answer(qa: dict[str, Any]) -> float | None:
        value = str(qa.get("answer", "")).strip().replace(",", "")
        match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*(%)?", value)
        if not match:
            return None
        return float(match.group(1))

    def measurement_signatures(qa: dict[str, Any]) -> set[str]:
        question = normalized_text(qa.get("question"))
        patterns = {
            "percentage": r"\b(?:percentage|percent)\b",
            "accuracy": r"\baccuracy\b",
            "score": r"\b(?:score|auc|f1)\b",
            "rate": r"\brate\b",
            "time": r"\b(?:time|latency|duration)\b",
            "memory": r"\bmemory\b",
            "temperature": r"\btemperature\b",
            "resolution": r"\bresolution\b",
            "exponent": r"\bexponent\b",
            "coefficient": r"\bcoefficient\b",
            "probability": r"\b(?:probability|proportion|fraction)\b",
            "count": r"\b(?:how many|number of)\b",
            "value": r"\bvalue\b",
        }
        return {
            name for name, pattern in patterns.items() if re.search(pattern, question)
        }

    def measurement_subject_terms(qa: dict[str, Any]) -> set[str]:
        generic = stopwords | {
            "accuracy",
            "coefficient",
            "decrease",
            "difference",
            "does",
            "from",
            "how",
            "increase",
            "into",
            "many",
            "metric",
            "number",
            "that",
            "the",
            "percent",
            "percentage",
            "rate",
            "reduction",
            "score",
            "what",
            "when",
            "where",
            "which",
            "value",
        }
        return {
            token
            for token in re.findall(
                r"[a-z0-9][a-z0-9.+-]*", normalized_text(qa.get("question"))
            )
            if len(token) >= 3 and token not in generic
        }

    def specific_identity_condition(qa: dict[str, Any]) -> bool:
        """Reject generic name lookups that cannot form a necessary first hop."""
        question = normalized_text(qa.get("question"))
        generic_patterns = (
            r"\b(?:new|novel|proposed|introduced)\s+(?:method|model|framework|"
            r"approach|algorithm|solution|concept|architecture)\b",
            r"\b(?:method|model|framework|approach|algorithm|solution|concept|"
            r"architecture)\s+(?:proposed|introduced|used)\s+in\s+the\s+paper\b",
            r"\b(?:name|title)\s+of\s+the\b",
        )
        return not any(re.search(pattern, question) for pattern in generic_patterns)

    qas = [
        (str(qa_id), qa)
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    ]
    hints: list[tuple[tuple[int, int, int, str], dict[str, Any]]] = []

    # A genuine hidden-entity chain has fact A identify entity X from a
    # description and fact B state an outcome/property for X.  The new question
    # hides X and asks for the downstream result; both facts remain necessary
    # even when B's answer is the terminal value.
    for (left_id, left_qa), (right_id, right_qa) in permutations(qas, 2):
        bridge_raw = str(left_qa.get("answer", "")).strip()
        bridge = normalized_content(bridge_raw)
        if (
            len(bridge) < 5
            or len(bridge_raw) > 100
            or re.fullmatch(r"(?:yes|no|true|false|[-+]?\d+(?:\.\d+)?)", bridge)
            or not specific_identity_condition(left_qa)
            or bridge not in normalized_content(right_qa.get("question"))
            or not select_separated_pages([left_id, right_id], paper, min_span)
        ):
            continue
        terminal = str(right_qa.get("answer", "")).strip()
        if not terminal or normalized_content(terminal) == bridge:
            continue
        row = {
            "source_qa_ids": [left_id, right_id],
            "shared_terms": sorted(terms(left_qa) & terms(right_qa))[:12],
            "relation_hint": "identity_hidden_sequential_chain",
            "bridge_entity": bridge_raw,
            "terminal_fact": terminal,
            "construction_rule": (
                "Hide the bridge entity from the question; describe it using "
                "the first source fact, then ask for the terminal property from "
                "the second source fact. State both bridge and terminal facts "
                "in one explanatory final answer."
            ),
        }
        hints.append(((2, 2, len(row["shared_terms"]), left_id + "\0" + right_id), row))

    for size in (2, 3):
        for group in combinations(qas, size):
            source_ids = [qa_id for qa_id, _ in group]
            if not select_separated_pages(source_ids, paper, min_span):
                continue
            substitution = False
            for left_index, (_, left_qa) in enumerate(group):
                answer = normalized_content(left_qa.get("answer"))
                if len(answer) < 5:
                    continue
                if any(
                    answer in normalized_content(right_qa.get("question"))
                    for right_index, (_, right_qa) in enumerate(group)
                    if right_index != left_index
                ):
                    substitution = True
                    break
            if substitution:
                continue
            term_sets = [terms(qa) for _, qa in group]
            adjacency = {index: set() for index in range(size)}
            shared_terms: set[str] = set()
            for left_index, right_index in combinations(range(size), 2):
                overlap = term_sets[left_index] & term_sets[right_index]
                if len(overlap) < 2:
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
            row = {
                "source_qa_ids": source_ids,
                "shared_terms": sorted(shared_terms)[:12],
            }
            numeric_values = [simple_numeric_answer(qa) for _, qa in group]
            signatures = [measurement_signatures(qa) for _, qa in group]
            common_signatures = set.intersection(*signatures) if signatures else set()
            subject_sets = [measurement_subject_terms(qa) for _, qa in group]
            shared_subject_terms = (
                set.intersection(*subject_sets) if subject_sets else set()
            )
            comparable_signature = bool(
                (
                    common_signatures - {"count", "value"}
                    and shared_subject_terms
                )
                or (
                    common_signatures & {"count", "value"}
                    and len(shared_subject_terms) >= 3
                )
            )
            numeric_relation = (
                all(value is not None for value in numeric_values)
                and comparable_signature
            )
            if numeric_relation:
                row["relation_hint"] = "comparable_numeric_derivation"
                row["measurement_signatures"] = sorted(common_signatures)
                row["measurement_subject_terms"] = sorted(shared_subject_terms)
                row["numeric_values"] = {
                    qa_id: value
                    for (qa_id, _), value in zip(group, numeric_values)
                }
                row["allowed_operations"] = [
                    "difference_or_percentage_point_gap",
                    "ratio",
                    "ordering_or_range",
                    "threshold_decision_when_explicitly_supported",
                ]
            else:
                row["relation_hint"] = "semantic_constraint_or_dependency"
            key = (
                1 if numeric_relation else 0,
                size,
                len(shared_terms),
                "\0".join(source_ids),
            )
            hints.append((key, row))
    hints.sort(key=lambda entry: entry[0], reverse=True)
    deduplicated: list[dict[str, Any]] = []
    seen_hint_keys: set[tuple[str, ...]] = set()
    for _, row in hints:
        key = tuple(row["source_qa_ids"])
        if key in seen_hint_keys:
            continue
        seen_hint_keys.add(key)
        deduplicated.append(row)
    return deduplicated[:80]


@dataclass
class LocalQwenGenerator:
    model_path: Path
    physical_gpus: str
    max_memory_gib: str
    max_new_tokens: int
    temperature: float
    seed: int
    model: Any = None
    tokenizer: Any = None

    def load(self) -> None:
        configure_model_cache_environment()
        self.model_path = resolve_local_model_path(
            self.model_path, require_config=True
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.physical_gpus

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        visible_count = torch.cuda.device_count()
        if visible_count == 0:
            raise RuntimeError("No CUDA GPU is visible")
        max_memory = self._max_memory(torch)
        print(
            json.dumps(
                {
                    "event": "model_load",
                    "model_path": str(self.model_path),
                    "physical_gpus": self.physical_gpus,
                    "max_memory": max_memory,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        offloaded = {
            name: device
            for name, device in (getattr(self.model, "hf_device_map", {}) or {}).items()
            if str(device).lower() in {"cpu", "disk"}
        }
        if offloaded:
            raise RuntimeError(
                "Model was offloaded to CPU/disk; provide more GPU memory: "
                + json.dumps(offloaded, ensure_ascii=False)
            )
        self.model.eval()

    def _max_memory(self, torch: Any) -> dict[int, str]:
        count = torch.cuda.device_count()
        if self.max_memory_gib.strip():
            values = [
                int(value.strip())
                for value in self.max_memory_gib.split(",")
                if value.strip()
            ]
            if len(values) != count:
                raise ValueError(
                    "--max-memory-gib must have one value per visible GPU "
                    f"({count} expected, {len(values)} supplied)"
                )
            return {index: f"{value}GiB" for index, value in enumerate(values)}

        limits: dict[int, str] = {}
        for index in range(count):
            free_bytes, _ = torch.cuda.mem_get_info(index)
            free_gib = free_bytes / (1024**3)
            usable_gib = max(1, int(free_gib - 3))
            limits[index] = f"{usable_gib}GiB"
        return limits

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        new_tokens = generated[0][inputs.input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@dataclass
class GeminiGenerator:
    """Small adapter that keeps API generation compatible with local checkpoints."""

    api_key_file: Path
    model: str
    max_new_tokens: int
    temperature: float
    timeout: int
    retries: int
    api_key: str | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        from pku_qa.workflows.review.review_cross_pdf_qa_api import (
            GEMINI_BASE_URL,
            load_key,
            post_with_retries,
            response_text_gemini,
        )

        if self.api_key is None:
            self.api_key = load_key(self.api_key_file)
        system = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        contents = [
            {
                "role": "model" if message.get("role") == "assistant" else "user",
                "parts": [{"text": str(message.get("content", ""))}],
            }
            for message in messages
            if message.get("role") != "system"
        ]
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_new_tokens,
                "responseMimeType": "application/json",
            },
        }
        response = post_with_retries(
            f"{GEMINI_BASE_URL}/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.api_key, "content-type": "application/json"},
            body,
            self.timeout,
            self.retries,
        )
        return response_text_gemini(response.json()).strip()


def collect_existing(
    raw_dir: Path,
    dataset: dict[str, Any],
    min_source_qas: int,
    max_source_qas: int,
    max_accepted_per_paper: int,
    *,
    hard_mode: bool = False,
    min_evidence_span: int = 3,
    require_semantic_chain: bool = False,
    local_self_review: bool = True,
    progress_heartbeat: Callable[[], None] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    accepted_by_paper: dict[str, list[dict[str, Any]]] = {}
    all_issues: list[dict[str, Any]] = []
    seen_questions = QuestionSimilarityIndex()
    original_qas_by_paper: dict[str, list[dict[str, Any]]] = {}
    semantic_hints_by_paper: dict[str, list[dict[str, Any]]] = {}
    for path_index, path in enumerate(sorted(raw_dir.glob("*.json"))):
        if progress_heartbeat is not None and path_index % 25 == 0:
            progress_heartbeat()
        payload = read_json(path)
        paper_id = str(payload.get("paper_id", path.stem))
        if paper_id not in dataset:
            continue
        response = payload.get(
            "parsed_response" if local_self_review else "pre_self_review_response"
        )
        if not isinstance(response, dict):
            continue
        response, _ = strip_parenthetical_source_citations(response)
        if paper_id not in original_qas_by_paper:
            original_qas_by_paper[paper_id] = compact_original_qas(
                dataset[paper_id]
            )
        original_qas = original_qas_by_paper[paper_id]
        if require_semantic_chain and paper_id not in semantic_hints_by_paper:
            semantic_hints_by_paper[paper_id] = semantic_chain_hints(
                dataset[paper_id], min_evidence_span
            )
        semantic_hints = (
            semantic_hints_by_paper[paper_id]
            if require_semantic_chain
            else None
        )
        accepted, issues = validate_response(
            response,
            paper_id,
            dataset[paper_id],
            min_source_qas,
            max_source_qas,
            seen_questions,
            hard_mode=hard_mode,
            min_evidence_span=min_evidence_span,
            require_semantic_chain=require_semantic_chain,
            original_qas=original_qas,
            semantic_hints=semantic_hints,
        )
        bucket = accepted_by_paper.setdefault(paper_id, [])
        remaining = max(0, max_accepted_per_paper - len(bucket))
        bucket.extend(accepted[:remaining])
        all_issues.extend(
            {"paper_id": paper_id, **issue} for issue in issues
        )
    return accepted_by_paper, all_issues


def build_dataset(
    source: dict[str, Any],
    accepted_by_paper: dict[str, list[dict[str, Any]]],
    target: int,
    model_name: str,
    stable_qa_ids: bool = False,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    count = 0
    for paper_id, items in accepted_by_paper.items():
        if count >= target:
            break
        paper = source[paper_id]
        qas: dict[str, Any] = {}
        for item in items:
            if count >= target:
                break
            count += 1
            enriched = dict(item)
            enriched["construction_model"] = model_name
            enriched["source_pdf_id"] = paper_id
            if stable_qa_ids:
                digest = hashlib.sha256(
                    (
                        paper_id
                        + "\0"
                        + normalized_text(enriched.get("question"))
                    ).encode("utf-8")
                ).hexdigest()[:16]
                qa_id = f"RQA_{digest}"
            else:
                qa_id = f"RQA{count}"
            if qa_id in qas:
                raise RuntimeError(
                    f"Stable QA ID collision in paper {paper_id}: {qa_id}"
                )
            qas[qa_id] = enriched
        if qas:
            output[paper_id] = {
                "paper": str(paper.get("paper", paper_id)),
                "primary_category": paper.get("primary_category", ""),
                "secondary_category": paper.get("secondary_category", ""),
                "QA": qas,
            }
    return output


def write_summary(
    output_dir: Path,
    dataset: dict[str, Any],
    source_path: Path,
    model_name: str,
    issues: list[dict[str, Any]],
    selected_papers: list[str],
    raw_count: int,
    target: int,
) -> None:
    items = [
        qa
        for paper in dataset.values()
        for qa in paper.get("QA", {}).values()
    ]
    summary = {
        "status": "complete" if len(items) >= target else "in_progress",
        "target": target,
        "accepted_questions": len(items),
        "papers_used": len(dataset),
        "source_dataset": str(source_path),
        "source_question_count": 4211,
        "model": model_name,
        "pdf_uploaded": False,
        "evidence_pages_used": False,
        "raw_papers_processed": raw_count,
        "selected_papers_available": len(selected_papers),
        "rejected_candidate_count": len(
            [issue for issue in issues if issue.get("item_index") is not None]
        ),
        "source_qa_count_distribution": dict(
            sorted(Counter(len(item["source_qa_ids"]) for item in items).items())
        ),
        "reasoning_type_distribution": dict(
            sorted(Counter(item["reasoning_type"] for item in items).items())
        ),
        "reasoning_focus_distribution": dict(
            sorted(
                Counter(
                    str(item.get("reasoning_focus", "unspecified"))
                    for item in items
                ).items()
            )
        ),
        "reasoning_type_normalization_distribution": dict(
            sorted(
                Counter(
                    (
                        item["reasoning_type_normalization"]["original"]
                        + " -> "
                        + item["reasoning_type_normalization"]["canonical"]
                    )
                    for item in items
                    if "reasoning_type_normalization" in item
                ).items()
            )
        ),
        "primary_category_distribution": dict(
            sorted(
                Counter(
                    paper.get("primary_category", "") for paper in dataset.values()
                ).items()
            )
        ),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(output_dir / "validation_issues.json", issues)
    atomic_json(
        output_dir / "progress.json",
        {
            "status": summary["status"],
            "completed": len(items),
            "total": target,
            "percent": round(100 * len(items) / target, 4) if target else 100.0,
            "stage": "reasoning_qa_generation",
            "updated_at": time.time(),
        },
    )


def mark_search_space_exhausted(
    output_dir: Path,
    accepted_questions: int,
    target: int,
    generation_rounds: int,
    selected_papers: int,
) -> None:
    """Publish a durable, actionable terminal state for this round budget."""
    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    summary.update(
        {
            "status": "search_space_exhausted",
            "accepted_questions": accepted_questions,
            "target": target,
            "exhausted_generation_rounds": generation_rounds,
            "selected_papers_available": selected_papers,
            "remaining_to_target": max(0, target - accepted_questions),
            "generated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
    )
    atomic_json(summary_path, summary)
    atomic_json(
        output_dir / "progress.json",
        {
            "status": "search_space_exhausted",
            "completed": accepted_questions,
            "total": target,
            "percent": (
                round(100 * accepted_questions / target, 4)
                if target
                else 100.0
            ),
            "stage": "reasoning_qa_generation",
            "exhausted_generation_rounds": generation_rounds,
            "selected_papers": selected_papers,
            "remaining_to_target": max(0, target - accepted_questions),
            "updated_at": time.time(),
        },
    )


def write_readme(output_dir: Path) -> None:
    text = """# Single-PDF QA-composition pilot

This pilot constructs new reasoning QA from related original QA pairs belonging
to the same paper.

- Source: `data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json` (4211 QA).
- Generator: local `Qwen3.6-27B`.
- The generator sees all QA from one paper per request.
- No PDF content is supplied.
- Evidence pages are excluded from generator and blind-review prompts. After
  both reviewers accept the final item, its gold pages are deterministically
  restored as the union of the cited source QA pages and checked against the
  local PDF.
- Each accepted question links 2-4 source QA and must pass structural checks
  against copied questions and concatenated source answers.

Files:

- `reasoning_qa_<N>.json`: resumable candidate pool plus construction
  provenance (`source_qa_ids` and `derivation`). Large strict-review runs use
  multiple generation rounds and content-derived stable QA IDs.
- `work__reasoning__historical_clean__batch00__n100.json`: final strict subset retained
  only when Claude and Gemini both independently return KEEP with all criteria
  true, with auditable restored `evidence_pages` and provenance.
- `selected_papers.json`: deterministic generation order.
- `raw/`: resumable per-paper model outputs.
- `summary.json`: counts and composition profile.
- `validation_issues.json`: rejected-output diagnostics.
- `dual_review/`: resumable provider responses, rejection ledger, API errors,
  review progress, and the final cleaning summary.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.api_workers <= 0:
        raise SystemExit("--api-workers must be positive")
    if args.api_workers > 1:
        if args.generation_backend != "gemini":
            raise SystemExit("--api-workers > 1 requires --generation-backend gemini")
        run_parallel_api_workers(args.api_workers)
        return
    source = read_json(args.source)
    if not isinstance(source, dict):
        raise SystemExit("Source dataset must be a JSON object keyed by paper ID")
    question_count = sum(
        len(paper.get("QA", {}))
        for paper in source.values()
        if isinstance(paper, dict)
    )
    if question_count != 4211:
        raise SystemExit(
            f"Expected the 4211-question source dataset, found {question_count}"
        )
    if args.target <= 0:
        raise SystemExit("--target must be positive")
    if args.generation_rounds <= 0:
        raise SystemExit("--generation-rounds must be positive")
    if args.max_accepted_per_paper <= 0:
        raise SystemExit("--max-accepted-per-paper must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = args.output_dir / ".paper_locks"
    write_readme(args.output_dir)
    selected_papers = select_papers(
        source,
        args.min_paper_qas,
        args.seed,
        hard_mode=args.hard_mode,
        min_evidence_span=args.min_evidence_span,
        require_semantic_chain=args.require_semantic_chain,
    )
    if args.max_papers:
        selected_papers = selected_papers[: args.max_papers]
    atomic_json(args.output_dir / "selected_papers.json", selected_papers)

    output_path = args.output_dir / f"reasoning_qa_{args.target}.json"
    generator_name = (
        args.model_path.name
        if args.generation_backend == "local"
        else args.api_model
    )
    aggregate_coordinator = is_aggregate_coordinator(
        args.generation_backend, args.worker_id
    )

    def refresh_outputs():
        def progress_heartbeat() -> None:
            completed = accepted_progress(args.output_dir)
            write_generation_progress(
                args.output_dir,
                args.target,
                completed,
                "validating_checkpoints",
            )

        refreshed, refreshed_issues = collect_existing(
            raw_dir,
            source,
            args.min_source_qas,
            args.max_source_qas,
            args.max_accepted_per_paper,
            hard_mode=args.hard_mode,
            min_evidence_span=args.min_evidence_span,
            require_semantic_chain=args.require_semantic_chain,
            local_self_review=args.local_self_review,
            progress_heartbeat=progress_heartbeat,
        )
        refreshed_dataset = build_dataset(
            source,
            refreshed,
            args.target,
            generator_name,
            stable_qa_ids=args.stable_qa_ids,
        )
        atomic_json(output_path, refreshed_dataset)
        write_summary(
            args.output_dir,
            refreshed_dataset,
            args.source,
            generator_name,
            refreshed_issues,
            selected_papers,
            len(list(raw_dir.glob("*.json"))),
            args.target,
        )
        return refreshed, refreshed_issues, refreshed_dataset

    if aggregate_coordinator:
        accepted_by_paper, issues, dataset = refresh_outputs()
        last_aggregate_raw_count = len(list(raw_dir.glob("*.json")))
    else:
        accepted_by_paper, issues, dataset = {}, [], {}
        last_aggregate_raw_count = 0
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "event": "prepared",
                    "source_questions": question_count,
                    "selected_papers": len(selected_papers),
                    "already_accepted": sum(
                        len(items) for items in accepted_by_paper.values()
                    ),
                    "output_dir": str(args.output_dir),
                },
                ensure_ascii=False,
            )
        )
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
    else:
        generator = GeminiGenerator(
            api_key_file=args.api_key_file,
            model=args.api_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            retries=args.api_retries,
        )
    accepted_total = (
        sum(len(paper.get("QA", {})) for paper in dataset.values())
        if aggregate_coordinator
        else accepted_progress(args.output_dir)
    )
    last_generation_heartbeat = 0.0

    def heartbeat_generation() -> None:
        nonlocal last_generation_heartbeat
        if not aggregate_coordinator:
            return
        timestamp = time.time()
        if timestamp - last_generation_heartbeat < 30:
            return
        write_generation_progress(
            args.output_dir,
            args.target,
            accepted_progress(args.output_dir, accepted_total),
            "generating_checkpoints",
        )
        last_generation_heartbeat = timestamp

    while True:
        heartbeat_generation()
        if accepted_total >= args.target:
            break
        remaining_unprocessed = False
        processed_one = False
        for generation_round in range(args.generation_rounds):
            for paper_id in selected_papers:
                raw_name = (
                    f"{paper_id}.json"
                    if generation_round == 0
                    else f"{paper_id}__round{generation_round}.json"
                )
                raw_path = raw_dir / raw_name
                if raw_path.exists() and not args.force:
                    continue
                remaining_unprocessed = True
                claim = try_paper_lock(lock_dir, raw_path.stem)
                if claim is None:
                    continue
                try:
                    if raw_path.exists() and not args.force:
                        continue
                    messages = build_messages(
                        paper_id,
                        source[paper_id],
                        args.candidates_per_paper,
                        args.min_source_qas,
                        args.max_source_qas,
                        generation_round,
                        hard_mode=args.hard_mode,
                        require_semantic_chain=args.require_semantic_chain,
                        min_evidence_span=args.min_evidence_span,
                    )
                    started = time.time()
                    raw_text = generator.generate(messages)
                    parsed = extract_json_object(raw_text)
                    repair_logs: list[dict[str, Any]] = []
                    if args.hard_mode and args.repair_leakage:
                        parsed, repair_logs = repair_leaked_questions(
                            generator, parsed, source[paper_id]
                        )
                    pre_self_review_response = parsed
                    self_review_log = None
                    if (
                        args.local_self_review
                        and args.hard_mode
                        and args.require_semantic_chain
                    ):
                        parsed, self_review_log = self_review_semantic_relations(
                            generator,
                            parsed,
                            paper_id,
                            source[paper_id],
                        )
                    parsed, citation_normalizations = (
                        strip_parenthetical_source_citations(parsed)
                    )
                    payload = {
                        "paper_id": paper_id,
                        "generation_round": generation_round,
                        "model": generator_name,
                        "worker_id": args.worker_id,
                        "elapsed_seconds": round(
                            time.time() - started, 3
                        ),
                        "raw_response": raw_text,
                        "pre_self_review_response": pre_self_review_response,
                        "parsed_response": parsed,
                        "self_review": self_review_log,
                        "deterministic_normalizations": citation_normalizations,
                        "leakage_repairs": repair_logs,
                    }
                    atomic_json(raw_path, payload)
                finally:
                    release_paper_lock(claim)
                current_raw_count = len(list(raw_dir.glob("*.json")))
                if (
                    aggregate_coordinator
                    and current_raw_count - last_aggregate_raw_count
                    >= aggregate_checkpoint_interval(args.target)
                ):
                    accepted_by_paper, issues, dataset = refresh_outputs()
                    last_aggregate_raw_count = current_raw_count
                    accepted_total = sum(
                        len(paper.get("QA", {}))
                        for paper in dataset.values()
                    )
                    accepted_from_paper: int | None = len(
                        accepted_by_paper.get(paper_id, [])
                    )
                else:
                    accepted_total = accepted_progress(
                        args.output_dir, accepted_total
                    )
                    accepted_from_paper = None
                    heartbeat_generation()
                print(
                    json.dumps(
                        {
                            "event": "paper_complete",
                            "worker_id": args.worker_id,
                            "paper_id": paper_id,
                            "generation_round": generation_round,
                            "accepted_from_paper": accepted_from_paper,
                            "accepted_total": accepted_total,
                            "target": args.target,
                            "elapsed_seconds": payload["elapsed_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                processed_one = True
                break
            if processed_one:
                break
        if processed_one:
            continue
        if remaining_unprocessed:
            # Other workers currently own the remaining papers. Kernel file
            # locks are released automatically if a worker dies.
            heartbeat_generation()
            time.sleep(5)
            continue
        break

    if not aggregate_coordinator:
        print(
            json.dumps(
                {
                    "event": "worker_complete",
                    "worker_id": args.worker_id,
                    "accepted_questions_at_last_aggregate": accepted_progress(
                        args.output_dir, accepted_total
                    ),
                    "target": args.target,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    accepted_by_paper, issues, dataset = refresh_outputs()
    final_count = sum(
        len(paper.get("QA", {})) for paper in dataset.values()
    )
    if final_count < args.target:
        mark_search_space_exhausted(
            args.output_dir,
            final_count,
            args.target,
            args.generation_rounds,
            len(selected_papers),
        )
        print(
            json.dumps(
                {
                    "event": "search_space_exhausted",
                    "accepted_questions": final_count,
                    "target": args.target,
                    "generation_rounds": args.generation_rounds,
                    "selected_papers": len(selected_papers),
                    "remaining_to_target": args.target - final_count,
                    "next_action": "increase_generation_rounds_or_paper_pool",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(
            "search_space_exhausted: "
            f"accepted={final_count}/{args.target}, "
            f"generation_rounds={args.generation_rounds}, "
            f"selected_papers={len(selected_papers)}; "
            "increase --generation-rounds or expand the eligible paper pool"
        )
    print(
        json.dumps(
            {
                "event": "complete",
                "accepted_questions": final_count,
                "papers_used": len(dataset),
                "output": str(output_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
