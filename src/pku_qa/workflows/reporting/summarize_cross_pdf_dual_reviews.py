#!/usr/bin/env python3
"""Summarize complete paired API reviews for an arbitrary cross-PDF QA file."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
CRITERIA = (
    "cross_document_required",
    "question_coherent",
    "question_specific",
    "answer_correct",
    "answer_complete",
    "evidence_sufficient",
    "no_unsupported_claim",
)
PROVIDERS = {
    "gemini": {
        "directory": "gemini_gemini-3-flash-preview",
        "model": "gemini-3-flash-preview",
    },
    "claude": {
        "directory": "claude_claude-sonnet-5",
        "model": "claude-sonnet-5",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expected_keys(qa: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (bundle_id, qa_id)
        for bundle_id, bundle in qa.items()
        for qa_id in bundle.get("QA", {})
    }


def normalized_usage(provider: str, usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    if provider == "gemini":
        visible = int(usage.get("candidatesTokenCount", 0))
        thinking = int(usage.get("thoughtsTokenCount", 0))
        return {
            "ordinary_input_tokens": int(usage.get("promptTokenCount", 0)),
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "visible_output_tokens": visible,
            "thinking_tokens": thinking,
            "billed_output_assumption": visible + thinking,
        }
    return {
        "ordinary_input_tokens": int(usage.get("input_tokens", 0)),
        "cache_creation_tokens": int(usage.get("cache_creation_input_tokens", 0)),
        "cache_read_tokens": int(usage.get("cache_read_input_tokens", 0)),
        "visible_output_tokens": int(usage.get("output_tokens", 0)),
        "thinking_tokens": 0,
        "billed_output_assumption": int(usage.get("output_tokens", 0)),
    }


def estimated_cost(provider: str, usage: dict[str, int]) -> dict[str, float]:
    if provider == "gemini":
        usd = (
            usage["ordinary_input_tokens"] * 0.5
            + usage["billed_output_assumption"] * 3.0
        ) / 1_000_000
        return {
            "nominal_usd": usd,
            "platform_cny_low": usd * 0.7,
            "platform_cny_high": usd * 0.7,
        }
    usd = (
        usage["ordinary_input_tokens"] * 2.0
        + usage["cache_creation_tokens"] * 2.5
        + usage["cache_read_tokens"] * 0.2
        + usage["visible_output_tokens"] * 10.0
    ) / 1_000_000
    return {
        "nominal_usd": usd,
        "platform_cny_low": usd * 1.4,
        "platform_cny_high": usd * 4.3,
    }


def read_reviews(
    provider: str,
    qa: dict[str, Any],
    qa_source: Path,
    reviews_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    config = PROVIDERS[provider]
    model_dir = reviews_dir / config["directory"]
    expected = expected_keys(qa)
    items: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_files: dict[str, list[str]] = {}
    missing_bundles: list[str] = []
    usage: Counter[str] = Counter()
    elapsed_seconds = 0.0

    for bundle_id in qa:
        path = model_dir / f"{bundle_id}.json"
        if not path.exists():
            missing_bundles.append(bundle_id)
            continue
        record = load_json(path)
        if record.get("validation_errors"):
            invalid_files[bundle_id] = list(record["validation_errors"])
            continue
        if Path(record["qa_source"]).resolve() != qa_source.resolve():
            invalid_files[bundle_id] = ["qa_source does not match requested QA file"]
            continue
        elapsed_seconds += float(record.get("elapsed_seconds", 0))
        for key, value in normalized_usage(provider, record.get("usage")).items():
            usage[key] += value
        for item in record["review"]["items"]:
            key = (bundle_id, str(item["qa_id"]))
            if key in items:
                raise RuntimeError(f"Duplicate review item: {key}")
            items[key] = item

    extra = sorted(set(items) - expected)
    missing_items = sorted(expected - set(items))
    metadata = {
        "provider": provider,
        "model": config["model"],
        "model_dir": str(model_dir),
        "review_file_count": sum(
            (model_dir / f"{bundle_id}.json").exists() for bundle_id in qa
        ),
        "error_file_count": sum(
            (model_dir / f"{bundle_id}.error.json").exists() for bundle_id in qa
        ),
        "missing_bundles": missing_bundles,
        "invalid_files": invalid_files,
        "missing_item_count": len(missing_items),
        "missing_item_examples": missing_items[:20],
        "extra_item_count": len(extra),
        "extra_item_examples": extra[:20],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": dict(usage),
        "estimated_cost": estimated_cost(provider, dict(usage)),
    }
    return items, metadata


def flatten_row(
    bundle_id: str,
    qa_id: str,
    qa_item: dict[str, Any],
    gemini: dict[str, Any],
    claude: dict[str, Any],
) -> dict[str, Any]:
    gemini_accept = gemini["decision"] in {"KEEP", "FIX"}
    claude_accept = claude["decision"] in {"KEEP", "FIX"}
    if gemini_accept and claude_accept:
        tier = "both_accept"
    elif claude_accept:
        tier = "claude_only_accept"
    elif gemini_accept:
        tier = "gemini_only_accept"
    else:
        tier = "both_reject"
    row: dict[str, Any] = {
        "bundle_id": bundle_id,
        "qa_id": qa_id,
        "question": qa_item.get("question"),
        "answer": qa_item.get("answer"),
        "evidence_pages": qa_item.get("evidence_pages", []),
        "modal_types": qa_item.get("modal_types", []),
        "review_tier": tier,
        "gemini_decision": gemini["decision"],
        "gemini_confidence": gemini["confidence"],
        "gemini_reason": gemini["reason"],
        "claude_decision": claude["decision"],
        "claude_confidence": claude["confidence"],
        "claude_reason": claude["reason"],
    }
    for provider, item in (("gemini", gemini), ("claude", claude)):
        for criterion in CRITERIA:
            row[f"{provider}_{criterion}"] = item["criteria"][criterion]
        row[f"{provider}_corrected_question"] = item.get("corrected_question")
        row[f"{provider}_corrected_answer"] = item.get("corrected_answer")
        row[f"{provider}_corrected_evidence_pages"] = item.get(
            "corrected_evidence_pages"
        )
    return row


def main() -> int:
    args = parse_args()
    qa = load_json(args.qa_json)
    if not isinstance(qa, dict):
        raise SystemExit("QA JSON must be an object keyed by bundle id.")
    expected = expected_keys(qa)
    gemini, gemini_meta = read_reviews(
        "gemini", qa, args.qa_json, args.reviews_dir
    )
    claude, claude_meta = read_reviews(
        "claude", qa, args.qa_json, args.reviews_dir
    )
    for metadata, items in ((gemini_meta, gemini), (claude_meta, claude)):
        if (
            metadata["missing_bundles"]
            or metadata["invalid_files"]
            or metadata["missing_item_count"]
            or metadata["extra_item_count"]
            or set(items) != expected
        ):
            raise SystemExit(
                f"Incomplete {metadata['provider']} reviews: "
                + json.dumps(metadata, ensure_ascii=False)
            )

    rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    tier_counts: Counter[str] = Counter()
    exact_agree = 0
    binary_agree = 0
    criterion_agreement: Counter[str] = Counter()
    for bundle_id, bundle in qa.items():
        for qa_id, qa_item in bundle.get("QA", {}).items():
            key = (bundle_id, qa_id)
            left = gemini[key]
            right = claude[key]
            confusion[(left["decision"], right["decision"])] += 1
            exact_agree += left["decision"] == right["decision"]
            binary_agree += (
                left["decision"] in {"KEEP", "FIX"}
            ) == (right["decision"] in {"KEEP", "FIX"})
            for criterion in CRITERIA:
                criterion_agreement[criterion] += (
                    left["criteria"][criterion] == right["criteria"][criterion]
                )
            row = flatten_row(bundle_id, qa_id, qa_item, left, right)
            tier_counts[row["review_tier"]] += 1
            rows.append(row)

    total = len(expected)
    summary = {
        "qa_source": str(args.qa_json),
        "reviews_source": str(args.reviews_dir),
        "bundle_count": len(qa),
        "qa_count": total,
        "models": {
            "gemini": {
                **gemini_meta,
                "decision_counts": dict(
                    sorted(Counter(item["decision"] for item in gemini.values()).items())
                ),
            },
            "claude": {
                **claude_meta,
                "decision_counts": dict(
                    sorted(Counter(item["decision"] for item in claude.values()).items())
                ),
            },
        },
        "agreement": {
            "exact_decision_count": exact_agree,
            "exact_decision_rate": exact_agree / total,
            "binary_accept_reject_count": binary_agree,
            "binary_accept_reject_rate": binary_agree / total,
            "criterion_agreement_rates": {
                criterion: criterion_agreement[criterion] / total
                for criterion in CRITERIA
            },
            "confusion_gemini_rows_claude_columns": {
                f"{left}->{right}": count
                for (left, right), count in sorted(confusion.items())
            },
        },
        "review_tier_counts": dict(sorted(tier_counts.items())),
        "cost_notes": [
            "Price assumptions match the supplied platform price pages.",
            "Gemini thinking tokens are conservatively counted as billed output.",
            "Claude CNY estimate is a range because multiple channel factors are listed.",
            "Cost estimates use API-reported tokens and are not wallet ledger charges.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "review_ledger.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_rows = []
    for row in rows:
        copied = dict(row)
        for field in (
            "evidence_pages",
            "modal_types",
            "gemini_corrected_evidence_pages",
            "claude_corrected_evidence_pages",
        ):
            copied[field] = json.dumps(copied[field], ensure_ascii=False)
        csv_rows.append(copied)
    with (args.output_dir / "review_ledger.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"Summarized {len(qa)} bundles / {total} QA; "
        f"tiers={dict(sorted(tier_counts.items()))}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
