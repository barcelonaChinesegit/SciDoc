#!/usr/bin/env python3
"""Audit and refresh Reasoning QA that must depend on non-adjacent PDF evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXISTING = (
    ROOT / "data/qa/3.reasoning/work__reasoning__historical_clean__batch00__n100.json"
)
DEFAULT_SOURCE = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
DEFAULT_AUDIT = ROOT / "data/qa/3.reasoning/pdf_dependency_audit.json"
DEFAULT_GPT_DETAIL = (
    ROOT
    / "data/qa/3.reasoning/output/gpt56_qwen36_judge/"
    "GPT-5_6_reasoning_qa100_eval_detail.csv"
)
DEFAULT_QWEN_DETAIL = (
    ROOT
    / "data/qa/3.reasoning/output/qwen35_qwen36_judge/"
    "reasoning_qa_qwen35_qwen36_eval_detail.csv"
)
REQUIRED_CLOSED_BOOK_MODELS = ("GPT-5_6-Reasoning", "Qwen3_5")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = {
    "about",
    "after",
    "also",
    "answer",
    "because",
    "from",
    "into",
    "paper",
    "that",
    "their",
    "then",
    "these",
    "this",
    "using",
    "which",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--closed-book-detail",
        type=Path,
        action="append",
        default=[],
        help="Judge detail CSV; may be repeated.",
    )
    parser.add_argument(
        "--required-closed-book-model",
        action="append",
        default=[],
        help="Model that must be correct before the closed-book failure flag is set.",
    )
    parser.add_argument("--replacement-pool", type=Path)
    parser.add_argument("--refreshed-output", type=Path)
    parser.add_argument("--incremental-output", type=Path)
    parser.add_argument("--incremental-target", type=int, default=100)
    parser.add_argument("--min-evidence-gap", type=int, default=3)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_content(value: Any) -> str:
    return " ".join(TOKEN_RE.findall(str(value or "").casefold()))


def substantive_tokens(value: Any) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(str(value or "").casefold())
        if len(token) >= 4 and token not in STOPWORDS
    }


def flatten(dataset: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (str(paper_id), str(qa_id), qa)
        for paper_id, paper in dataset.items()
        for qa_id, qa in paper.get("QA", {}).items()
        if isinstance(qa, dict)
    ]


def load_closed_book_scores(
    paths: list[Path],
) -> dict[tuple[str, str], dict[str, bool]]:
    scores: dict[tuple[str, str], dict[str, bool]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (str(row.get("paper_id", "")), str(row.get("qa_id", "")))
                model = str(row.get("model_name", ""))
                if not all((*key, model)):
                    continue
                scores.setdefault(key, {})[model] = str(
                    row.get("answer_correct", "")
                ).strip().casefold() in {"1", "true", "yes"}
    return scores


def source_answer_leaks(
    paper_id: str,
    qa: dict[str, Any],
    source: dict[str, Any],
) -> list[dict[str, str]]:
    question = normalized_content(qa.get("question"))
    leaks = []
    source_qas = source.get(paper_id, {}).get("QA", {})
    for source_id in qa.get("source_qa_ids", []):
        answer = str(source_qas.get(str(source_id), {}).get("answer", "")).strip()
        normalized = normalized_content(answer)
        if len(normalized) >= 4 and normalized in question:
            leaks.append({"source_qa_id": str(source_id), "answer": answer})
    final_answer = str(qa.get("answer", "")).strip()
    normalized_final = normalized_content(final_answer)
    if len(normalized_final) >= 4 and normalized_final in question:
        leaks.append({"source_qa_id": "__final_answer__", "answer": final_answer})
    return leaks


def contribution_page_options(qa: dict[str, Any]) -> list[list[int]]:
    provenance = qa.get("evidence_provenance")
    contributions = (
        provenance.get("source_qa_contributions", [])
        if isinstance(provenance, dict)
        else []
    )
    result = []
    for row in contributions:
        if not isinstance(row, dict):
            continue
        pages = sorted(
            {
                int(page)
                for page in row.get("evidence_pages", [])
                if isinstance(page, int) and not isinstance(page, bool) and page > 0
            }
        )
        if pages:
            result.append(pages)
    return result


def separated_evidence_selection(
    qa: dict[str, Any], min_gap: int
) -> list[int] | None:
    options = contribution_page_options(qa)
    if len(options) >= 2:
        for pages in itertools.product(*options):
            if len(set(pages)) != len(pages):
                continue
            if all(
                abs(left - right) >= min_gap
                for left, right in itertools.combinations(pages, 2)
            ):
                return list(pages)
        return None
    pages = sorted(
        {
            int(page)
            for page in qa.get("evidence_pages", [])
            if isinstance(page, int) and not isinstance(page, bool) and page > 0
        }
    )
    if len(pages) >= 2 and max(pages) - min(pages) >= min_gap:
        return [min(pages), max(pages)]
    return None


def intermediate_fact_diagnostics(qa: dict[str, Any]) -> dict[str, Any]:
    derivation = qa.get("derivation", [])
    fact_rows = [
        row
        for row in derivation
        if isinstance(row, dict) and str(row.get("fact", "")).strip()
    ]
    inference_rows = [
        row
        for row in derivation
        if isinstance(row, dict) and str(row.get("inference", "")).strip()
    ]
    declared = qa.get("intermediate_facts", [])
    bridge_values = [str(row.get("fact")) for row in fact_rows]
    if isinstance(declared, list):
        bridge_values.extend(str(value) for value in declared if str(value).strip())
    answer_tokens = substantive_tokens(qa.get("answer"))
    overlap_counts = [
        len(answer_tokens & substantive_tokens(value)) for value in bridge_values
    ]
    return {
        "fact_steps": len(fact_rows),
        "inference_steps": len(inference_rows),
        "answer_word_count": len(str(qa.get("answer", "")).split()),
        "max_bridge_token_overlap": max(overlap_counts, default=0),
        "valid": (
            len(fact_rows) >= 2
            and len(inference_rows) >= 1
            and len(str(qa.get("answer", "")).split()) >= 8
            and max(overlap_counts, default=0) >= 2
        ),
    }


def audit_dataset(
    existing: dict[str, Any],
    source: dict[str, Any],
    closed_book_scores: dict[tuple[str, str], dict[str, bool]],
    required_models: list[str],
    min_evidence_gap: int,
) -> dict[str, Any]:
    rows = []
    reason_counts: Counter[str] = Counter()
    for paper_id, qa_id, qa in flatten(existing):
        key = (paper_id, qa_id)
        model_scores = closed_book_scores.get(key, {})
        closed_book_correct = bool(required_models) and all(
            model_scores.get(model) is True for model in required_models
        )
        pages = sorted(set(qa.get("evidence_pages", [])))
        evidence_selection = separated_evidence_selection(qa, min_evidence_gap)
        leaks = source_answer_leaks(paper_id, qa, source)
        intermediate = intermediate_fact_diagnostics(qa)
        reasons = []
        if len(pages) < 2:
            reasons.append("fewer_than_two_evidence_pages")
        if evidence_selection is None:
            reasons.append("no_non_adjacent_source_evidence_selection")
        if leaks:
            reasons.append("source_or_final_answer_leaked_in_question")
        if not intermediate["valid"]:
            reasons.append("missing_answer_visible_intermediate_fact_chain")
        if closed_book_correct:
            reasons.append("all_required_models_answered_correctly_without_pdf")
        reason_counts.update(reasons)
        rows.append(
            {
                "paper_id": paper_id,
                "qa_id": qa_id,
                "replace": bool(reasons),
                "reasons": reasons,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence_pages": pages,
                "separated_evidence_selection": evidence_selection,
                "answer_leaks": leaks,
                "intermediate_fact_diagnostics": intermediate,
                "closed_book_answer_correct": model_scores,
            }
        )
    replace_count = sum(row["replace"] for row in rows)
    return {
        "status": "complete",
        "total_questions": len(rows),
        "replacement_required": replace_count,
        "retained_without_change": len(rows) - replace_count,
        "min_evidence_gap": min_evidence_gap,
        "required_closed_book_models": required_models,
        "reason_counts": dict(sorted(reason_counts.items())),
        "items": rows,
    }


def candidate_sort_key(row: tuple[str, str, dict[str, Any]]) -> tuple[Any, ...]:
    paper_id, qa_id, qa = row
    return (paper_id, qa_id)


def balanced_candidate_rows(
    rows: list[tuple[str, str, dict[str, Any]]],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Interleave hard-reasoning foci so one type cannot consume the pool."""
    focus_order = (
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "model_relationship",
        "metric_reasoning",
    )
    pattern = (
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "model_relationship",
        "metric_reasoning",
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "model_relationship",
        "sequential_reasoning",
        "conditional_filtering",
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "model_relationship",
        "sequential_reasoning",
        "metric_reasoning",
    )
    groups: dict[str, list[tuple[str, str, dict[str, Any]]]] = {
        focus: [] for focus in focus_order
    }
    other: list[tuple[str, str, dict[str, Any]]] = []
    for row in sorted(rows, key=candidate_sort_key):
        focus = str(row[2].get("reasoning_focus", ""))
        (groups[focus] if focus in groups else other).append(row)
    result: list[tuple[str, str, dict[str, Any]]] = []
    while any(groups.values()):
        made_progress = False
        for focus in pattern:
            if groups[focus]:
                result.append(groups[focus].pop(0))
                made_progress = True
        if not made_progress:
            break
    result.extend(other)
    return result


def validate_replacement_pool(
    replacement_pool: dict[str, Any],
    source: dict[str, Any],
    min_evidence_gap: int,
) -> dict[str, Any]:
    """Reject candidates lacking the publication gates needed by refresh."""
    errors: list[str] = []
    focus_counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    allowed_foci = {
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "model_relationship",
        "metric_reasoning",
    }
    for paper_id, qa_id, qa in flatten(replacement_pool):
        key = f"{paper_id}/{qa_id}"
        question_key = normalized_content(qa.get("question"))
        if not question_key or question_key in seen_questions:
            errors.append(f"{key}:empty_or_duplicate_question")
        seen_questions.add(question_key)
        validation = qa.get("dual_model_validation")
        if not isinstance(validation, dict) or any(
            not isinstance(validation.get(provider), dict)
            or validation[provider].get("decision") != "KEEP"
            for provider in ("claude", "gemini")
        ):
            errors.append(f"{key}:missing_dual_keep")
        dependency = qa.get("closed_book_pdf_dependency")
        if not isinstance(dependency, dict) or dependency.get(
            "both_models_correct"
        ) is not False:
            errors.append(f"{key}:missing_closed_book_failure")
        if separated_evidence_selection(qa, min_evidence_gap) is None:
            errors.append(f"{key}:evidence_not_non_adjacent")
        if not intermediate_fact_diagnostics(qa)["valid"]:
            errors.append(f"{key}:missing_intermediate_fact_chain")
        focus = str(qa.get("reasoning_focus", ""))
        if focus not in allowed_foci:
            errors.append(f"{key}:invalid_reasoning_focus")
        focus_counts[focus] += 1
        if source_answer_leaks(paper_id, qa, source):
            errors.append(f"{key}:answer_leaked_in_question")
    if errors:
        preview = "; ".join(errors[:12])
        suffix = "" if len(errors) <= 12 else f"; ... ({len(errors)} total)"
        raise ValueError("Invalid replacement pool: " + preview + suffix)
    return {
        "status": "passed",
        "question_count": len(flatten(replacement_pool)),
        "reasoning_focus_distribution": dict(sorted(focus_counts.items())),
    }


def add_item(
    dataset: dict[str, Any],
    source_dataset: dict[str, Any],
    paper_id: str,
    qa_id: str,
    qa: dict[str, Any],
) -> None:
    source_paper = source_dataset.get(paper_id, {})
    bucket = dataset.setdefault(
        paper_id,
        {
            "paper": paper_id,
            "primary_category": source_paper.get("primary_category", ""),
            "secondary_category": source_paper.get("secondary_category", ""),
            "QA": {},
        },
    )
    bucket["QA"][qa_id] = deepcopy(qa)


def build_refreshed_and_incremental(
    existing: dict[str, Any],
    source: dict[str, Any],
    replacement_pool: dict[str, Any],
    audit: dict[str, Any],
    incremental_target: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    replace_keys = {
        (str(row["paper_id"]), str(row["qa_id"]))
        for row in audit["items"]
        if row["replace"]
    }
    pool_validation = validate_replacement_pool(
        replacement_pool,
        source,
        int(audit.get("min_evidence_gap", 3)),
    )
    candidates = balanced_candidate_rows(flatten(replacement_pool))
    needed = len(replace_keys) + incremental_target
    if len(candidates) < needed:
        raise ValueError(
            f"replacement pool has {len(candidates)} questions, requires {needed}"
        )
    used_questions: set[str] = set()
    existing_questions = {
        normalized_content(qa.get("question"))
        for _, _, qa in flatten(existing)
        if normalized_content(qa.get("question"))
    }

    # Reserve the incremental set first so replacement consumption cannot
    # exhaust a focus that the new batch is expected to contain.  The quotas
    # mirror the documented 30/25/20/15/10 focus target and gracefully fall
    # back to the balanced order when a focus has too few candidates.
    focus_quotas = {
        "sequential_reasoning": 30,
        "conditional_filtering": 25,
        "similar_concept_discrimination": 20,
        "model_relationship": 15,
        "metric_reasoning": 10,
    }
    reserved: list[tuple[str, str, dict[str, Any]]] = []
    reserved_ids: set[tuple[str, str]] = set()
    for focus, quota in focus_quotas.items():
        if len(reserved) >= incremental_target:
            break
        for row in candidates:
            if len(reserved) >= incremental_target or sum(
                1 for item in reserved if item[2].get("reasoning_focus") == focus
            ) >= quota:
                break
            paper_id, qa_id, qa = row
            if (paper_id, qa_id) in reserved_ids:
                continue
            question_key = normalized_content(qa.get("question"))
            if not question_key or question_key in existing_questions:
                continue
            if str(qa.get("reasoning_focus", "")) != focus:
                continue
            reserved.append(row)
            reserved_ids.add((paper_id, qa_id))
    if len(reserved) < incremental_target:
        for row in candidates:
            if len(reserved) >= incremental_target:
                break
            paper_id, qa_id, qa = row
            if (paper_id, qa_id) in reserved_ids:
                continue
            question_key = normalized_content(qa.get("question"))
            if not question_key or question_key in existing_questions:
                continue
            reserved.append(row)
            reserved_ids.add((paper_id, qa_id))

    replacement_candidates = [
        row for row in candidates if (row[0], row[1]) not in reserved_ids
    ]
    refreshed: dict[str, Any] = {}
    replacement_ledger = []
    candidate_index = 0
    for paper_id, qa_id, qa in flatten(existing):
        if (paper_id, qa_id) not in replace_keys:
            add_item(refreshed, existing, paper_id, qa_id, qa)
            used_questions.add(normalized_content(qa.get("question")))
            continue
        while candidate_index < len(replacement_candidates):
            new_paper_id, candidate_id, candidate = replacement_candidates[candidate_index]
            candidate_index += 1
            question_key = normalized_content(candidate.get("question"))
            if question_key and question_key not in used_questions:
                break
        else:
            raise ValueError("replacement pool exhausted while refreshing existing QA")
        digest = hashlib.sha256(
            (new_paper_id + "\0" + question_key).encode("utf-8")
        ).hexdigest()[:16]
        refreshed_id = f"RQA_REFRESH_{digest}"
        refreshed_qa = deepcopy(candidate)
        provenance = deepcopy(refreshed_qa.get("annotation_provenance", {}))
        provenance["replaces_existing_reasoning_qa"] = f"{paper_id}/{qa_id}"
        refreshed_qa["annotation_provenance"] = provenance
        add_item(
            refreshed,
            replacement_pool,
            new_paper_id,
            refreshed_id,
            refreshed_qa,
        )
        used_questions.add(question_key)
        replacement_ledger.append(
            {
                "removed": f"{paper_id}/{qa_id}",
                "replacement_source": f"{new_paper_id}/{candidate_id}",
                "released_as": f"{new_paper_id}/{refreshed_id}",
            }
        )

    incremental: dict[str, Any] = {}
    incremental_ledger = []
    for paper_id, candidate_id, candidate in reserved:
        if len(incremental_ledger) >= incremental_target:
            break
        question_key = normalized_content(candidate.get("question"))
        if not question_key or question_key in used_questions:
            continue
        add_item(incremental, replacement_pool, paper_id, candidate_id, candidate)
        used_questions.add(question_key)
        incremental_ledger.append(
            {"source": f"{paper_id}/{candidate_id}"}
        )

    summary = {
        "refreshed_total": len(flatten(refreshed)),
        "replaced_existing": len(replacement_ledger),
        "incremental_total": len(flatten(incremental)),
        "incremental_reasoning_focus_distribution": dict(
            sorted(
                Counter(
                    str(qa.get("reasoning_focus", ""))
                    for _, _, qa in flatten(incremental)
                ).items()
            )
        ),
        "replacement_ledger": replacement_ledger,
        "incremental_ledger": incremental_ledger,
        "replacement_pool_validation": pool_validation,
    }
    return refreshed, incremental, summary


def main() -> None:
    args = parse_args()
    detail_paths = args.closed_book_detail or [
        DEFAULT_GPT_DETAIL,
        DEFAULT_QWEN_DETAIL,
    ]
    required_models = (
        args.required_closed_book_model or list(REQUIRED_CLOSED_BOOK_MODELS)
    )
    existing = read_json(args.existing)
    source = read_json(args.source)
    audit = audit_dataset(
        existing,
        source,
        load_closed_book_scores(detail_paths),
        required_models,
        args.min_evidence_gap,
    )
    atomic_json(args.audit_output, audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "items"}))

    output_args = (
        args.replacement_pool,
        args.refreshed_output,
        args.incremental_output,
    )
    if not any(output_args):
        return
    if not all(output_args):
        raise SystemExit(
            "--replacement-pool, --refreshed-output, and --incremental-output "
            "must be supplied together"
        )
    replacement_pool = read_json(args.replacement_pool)
    refreshed, incremental, summary = build_refreshed_and_incremental(
        existing,
        source,
        replacement_pool,
        audit,
        args.incremental_target,
    )
    atomic_json(args.refreshed_output, refreshed)
    atomic_json(args.incremental_output, incremental)
    atomic_json(args.audit_output.with_name("reasoning_refresh_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
