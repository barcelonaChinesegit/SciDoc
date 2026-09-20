#!/usr/bin/env python3
"""Summarize a paired Gemini/Claude calibration run reproducibly."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
AUDIT_ROOT = ROOT / "data/qa/4.cross_pdf/semantic_reaudit"
DEFAULT_CALIBRATION = AUDIT_ROOT / "calibration_manifest.json"
DEFAULT_REVIEWS = AUDIT_ROOT / "api_reviews"
DEFAULT_OUTPUT = AUDIT_ROOT / "calibration_comparison"

MODELS = {
    "gemini": {
        "directory": "gemini_gemini-3-flash-preview",
        "display": "gemini-3-flash-preview",
    },
    "claude": {
        "directory": "claude_claude-sonnet-5",
        "display": "claude-sonnet-5",
    },
}

CRITERIA = (
    "cross_document_required",
    "question_coherent",
    "question_specific",
    "answer_correct",
    "answer_complete",
    "evidence_sufficient",
    "no_unsupported_claim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--reviews-dir", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manual-adjudication-json",
        type=Path,
        default=DEFAULT_OUTPUT / "manual_adjudication.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_usage(provider: str, usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    if provider == "gemini":
        prompt = int(usage.get("promptTokenCount", 0))
        visible_output = int(usage.get("candidatesTokenCount", 0))
        thinking = int(usage.get("thoughtsTokenCount", 0))
        return {
            "input_tokens": prompt,
            "visible_output_tokens": visible_output,
            "thinking_tokens": thinking,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "total_billed_output_assumption": visible_output + thinking,
        }
    cache_creation = int(usage.get("cache_creation_input_tokens", 0))
    cache_read = int(usage.get("cache_read_input_tokens", 0))
    ordinary_input = int(usage.get("input_tokens", 0))
    output = int(usage.get("output_tokens", 0))
    return {
        "input_tokens": ordinary_input,
        "visible_output_tokens": output,
        "thinking_tokens": 0,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "total_billed_output_assumption": output,
    }


def add_usage(total: Counter[str], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total[key] += value


def nominal_cost(provider: str, usage: dict[str, int]) -> dict[str, float]:
    if provider == "gemini":
        usd = (
            usage["input_tokens"] * 0.5
            + usage["total_billed_output_assumption"] * 3.0
        ) / 1_000_000
        return {
            "nominal_usd": usd,
            "platform_cny_low": usd * 0.7,
            "platform_cny_high": usd * 0.7,
        }
    usd = (
        usage["input_tokens"] * 2.0
        + usage["cache_creation_tokens"] * 2.5
        + usage["cache_read_tokens"] * 0.2
        + usage["visible_output_tokens"] * 10.0
    ) / 1_000_000
    return {
        "nominal_usd": usd,
        "platform_cny_low": usd * 1.4,
        "platform_cny_high": usd * 4.3,
    }


def load_model_reviews(
    provider: str,
    bundle_ids: list[str],
    reviews_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    config = MODELS[provider]
    model_dir = reviews_dir / config["directory"]
    items: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    invalid: dict[str, list[str]] = {}
    usage_total: Counter[str] = Counter()
    elapsed = 0.0
    for bundle_id in bundle_ids:
        path = model_dir / f"{bundle_id}.json"
        if not path.exists():
            missing.append(bundle_id)
            continue
        record = load_json(path)
        validation_errors = record.get("validation_errors") or []
        if validation_errors:
            invalid[bundle_id] = validation_errors
            continue
        elapsed += float(record.get("elapsed_seconds", 0))
        add_usage(usage_total, normalize_usage(provider, record.get("usage")))
        for item in record["review"]["items"]:
            key = (bundle_id, str(item["qa_id"]))
            if key in items:
                raise RuntimeError(f"Duplicate review item: {key}")
            items[key] = item
    usage = dict(usage_total)
    metadata = {
        "provider": provider,
        "model": config["display"],
        "model_dir": str(model_dir),
        "missing_bundles": missing,
        "invalid_bundles": invalid,
        "elapsed_seconds": round(elapsed, 3),
        "usage": usage,
        "estimated_cost": nominal_cost(provider, usage),
    }
    return items, metadata


def main() -> int:
    args = parse_args()
    calibration = load_json(args.calibration_json)
    bundle_ids = [row["bundle_id"] for row in calibration["bundles"]]
    gemini, gemini_meta = load_model_reviews("gemini", bundle_ids, args.reviews_dir)
    claude, claude_meta = load_model_reviews("claude", bundle_ids, args.reviews_dir)
    expected_count = int(calibration["sample_size_qas"])
    if len(gemini) != expected_count or len(claude) != expected_count:
        raise SystemExit(
            "Calibration is incomplete: "
            f"expected {expected_count} items per model, "
            f"got Gemini={len(gemini)}, Claude={len(claude)}. "
            f"Gemini missing={gemini_meta['missing_bundles']}, "
            f"Claude missing={claude_meta['missing_bundles']}."
        )
    if set(gemini) != set(claude):
        raise RuntimeError("Gemini and Claude reviewed different QA keys.")

    gemini_decisions = Counter(item["decision"] for item in gemini.values())
    claude_decisions = Counter(item["decision"] for item in claude.values())
    confusion: Counter[tuple[str, str]] = Counter()
    criterion_agreement: Counter[str] = Counter()
    disagreements: list[dict[str, Any]] = []
    exact_agree = 0
    binary_agree = 0
    for bundle_id, qa_id in sorted(gemini):
        left = gemini[(bundle_id, qa_id)]
        right = claude[(bundle_id, qa_id)]
        left_decision = left["decision"]
        right_decision = right["decision"]
        confusion[(left_decision, right_decision)] += 1
        if left_decision == right_decision:
            exact_agree += 1
        left_accept = left_decision in {"KEEP", "FIX"}
        right_accept = right_decision in {"KEEP", "FIX"}
        if left_accept == right_accept:
            binary_agree += 1
        for criterion in CRITERIA:
            if left["criteria"][criterion] == right["criteria"][criterion]:
                criterion_agreement[criterion] += 1
        if left_decision != right_decision:
            disagreements.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "gemini_decision": left_decision,
                    "gemini_confidence": left["confidence"],
                    "gemini_reason": left["reason"],
                    "claude_decision": right_decision,
                    "claude_confidence": right["confidence"],
                    "claude_reason": right["reason"],
                }
            )

    summary = {
        "calibration_source": str(args.calibration_json),
        "reviewed_qas": expected_count,
        "models": {
            "gemini": {
                **gemini_meta,
                "decision_counts": dict(sorted(gemini_decisions.items())),
            },
            "claude": {
                **claude_meta,
                "decision_counts": dict(sorted(claude_decisions.items())),
            },
        },
        "agreement": {
            "exact_decision_count": exact_agree,
            "exact_decision_rate": exact_agree / expected_count,
            "binary_accept_reject_count": binary_agree,
            "binary_accept_reject_rate": binary_agree / expected_count,
            "criterion_rates": {
                criterion: criterion_agreement[criterion] / expected_count
                for criterion in CRITERIA
            },
            "confusion_gemini_rows_claude_columns": {
                f"{left}->{right}": count
                for (left, right), count in sorted(confusion.items())
            },
            "decision_disagreement_count": len(disagreements),
        },
        "cost_notes": [
            "Gemini estimate uses $0.5/M input and $3/M output from the supplied price page.",
            "Gemini thinking tokens are conservatively counted as billed output tokens.",
            "Claude estimate uses $2/M input, $10/M output, $2.5/M cache creation, and $0.2/M cache read.",
            "Platform CNY conversion uses 0.7 CNY/USD-usage for Gemini and a 1.4-4.3 range for Claude because the supplied page lists multiple Claude channels.",
            "These are estimates from reported usage, not wallet ledger charges.",
        ],
        "manual_adjudication_status": "pending",
    }

    if args.manual_adjudication_json.exists():
        adjudication = load_json(args.manual_adjudication_json)
        human = {
            (str(item["bundle_id"]), str(item["qa_id"])): str(item["decision"])
            for item in adjudication["items"]
        }
        disagreement_keys = {
            (item["bundle_id"], item["qa_id"]) for item in disagreements
        }
        if set(human) != disagreement_keys:
            raise RuntimeError(
                "Manual adjudication keys do not exactly match model disagreements."
            )
        human_counts = Counter(human.values())
        model_metrics: dict[str, Any] = {}
        for provider, reviews in (("gemini", gemini), ("claude", claude)):
            exact = 0
            binary = 0
            for key, human_decision in human.items():
                model_decision = reviews[key]["decision"]
                exact += model_decision == human_decision
                binary += (
                    model_decision in {"KEEP", "FIX"}
                ) == (human_decision in {"KEEP", "FIX"})
            model_metrics[provider] = {
                "exact_decision_count": exact,
                "exact_decision_rate": exact / len(human),
                "binary_accept_reject_count": binary,
                "binary_accept_reject_rate": binary / len(human),
            }
        summary["manual_adjudication_status"] = "complete"
        summary["manual_adjudication"] = {
            "source": str(args.manual_adjudication_json),
            "adjudicated_items": len(human),
            "human_decision_counts": dict(sorted(human_counts.items())),
            "model_agreement_on_disagreements": model_metrics,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "decision_disagreements.json").write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "decision_disagreements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(disagreements[0]) if disagreements else [
            "bundle_id",
            "qa_id",
            "gemini_decision",
            "gemini_confidence",
            "gemini_reason",
            "claude_decision",
            "claude_confidence",
            "claude_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)

    print(
        f"Compared {expected_count} QA: exact={exact_agree}, "
        f"binary={binary_agree}, disagreements={len(disagreements)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
