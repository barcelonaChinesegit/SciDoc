#!/usr/bin/env python3
"""Merge several independently dual-reviewed hard Cross-PDF candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from pku_qa.workflows.review.finalize_hard_cross_pdf_qa import (
    atomic_json,
    collect,
    normalized,
    read_json,
    select_items,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-three-doc-ratio", type=float, default=0.4)
    parser.add_argument("--max-per-bundle", type=int, default=6)
    return parser.parse_args()


def load_pools(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("pools") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("pool config must contain a non-empty pools list")
    required = {"name", "candidates", "claude_dir", "gemini_dir"}
    names: set[str] = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"pool row must contain {sorted(required)}")
        name = str(row["name"]).strip()
        if not name or name in names:
            raise ValueError(f"invalid or duplicate pool name: {name!r}")
        names.add(name)
        result.append(
            {
                "name": name,
                **{key: Path(str(row[key])) for key in required - {"name"}},
            }
        )
    return result


def remove_cross_pool_duplicates(
    items: list[dict[str, Any]], threshold: float = 0.92
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Prefer higher-confidence items when questions repeat across pools."""
    ordered = sorted(
        items,
        key=lambda item: (
            -min(
                float(row.get("confidence", 0))
                for row in item.get("dual_model_validation", {}).values()
            ),
            item["_pool_name"],
            item["_bundle_id"],
            item["_source_qa_id"],
        ),
    )
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    normalized_kept: list[str] = []
    for item in ordered:
        question = normalized(item.get("question"))
        duplicate_index = next(
            (
                index
                for index, prior in enumerate(normalized_kept)
                if SequenceMatcher(None, question, prior).ratio() >= threshold
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(item)
            normalized_kept.append(question)
            continue
        rejected.append(
            {
                "pool": item["_pool_name"],
                "bundle_id": item["_bundle_id"],
                "qa_id": item["_source_qa_id"],
                "reason": "near_duplicate_across_pools",
                "kept_pool": kept[duplicate_index]["_pool_name"],
                "kept_qa_id": kept[duplicate_index]["_source_qa_id"],
            }
        )
    return kept, rejected


def build_release(
    candidates_by_pool: dict[str, dict[str, Any]], chosen: list[dict[str, Any]]
) -> dict[str, Any]:
    release: dict[str, Any] = {}
    for index, source_item in enumerate(chosen, start=1):
        item = dict(source_item)
        pool_name = item.pop("_pool_name")
        bundle_id = item.pop("_bundle_id")
        source_qa_id = item.pop("_source_qa_id")
        source_bundle = candidates_by_pool[pool_name][bundle_id]
        bucket = release.setdefault(
            bundle_id,
            {
                "paper": bundle_id,
                "primary_category": source_bundle.get("primary_category", ""),
                "secondary_category": source_bundle.get("secondary_category", ""),
                "QA": {},
            },
        )
        digest = hashlib.sha256(
            (pool_name + "\0" + bundle_id + "\0" + normalized(item["question"])).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        item["candidate_pool"] = pool_name
        item["candidate_qa_id"] = source_qa_id
        item["release_rank"] = index
        bucket["QA"][f"CQA_{digest}"] = item
    return release


def main() -> None:
    args = parse_args()
    pools = load_pools(args.pool_config)
    candidates_by_pool: dict[str, dict[str, Any]] = {}
    all_eligible: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    eligible_counts: Counter[str] = Counter()
    for pool in pools:
        candidates = read_json(pool["candidates"])
        candidates_by_pool[pool["name"]] = candidates
        eligible, pool_ledger = collect(
            candidates,
            pool["claude_dir"],
            pool["gemini_dir"],
            args.min_confidence,
        )
        for item in eligible:
            item["_pool_name"] = pool["name"]
        for row in pool_ledger:
            row["pool"] = pool["name"]
        all_eligible.extend(eligible)
        ledger.extend(pool_ledger)
        eligible_counts[pool["name"]] = len(eligible)

    deduplicated, duplicate_ledger = remove_cross_pool_duplicates(all_eligible)
    chosen = select_items(
        deduplicated,
        args.target,
        args.min_three_doc_ratio,
        args.max_per_bundle,
    )
    if len(chosen) != args.target:
        raise SystemExit(
            f"insufficient dual-KEEP candidates across pools: "
            f"selected={len(chosen)}/{args.target}, "
            f"eligible={len(all_eligible)}, deduplicated={len(deduplicated)}"
        )
    selected_keys = {
        (item["_pool_name"], item["_bundle_id"], item["_source_qa_id"])
        for item in chosen
    }
    for row in ledger:
        row["selected"] = (
            row["pool"],
            row["bundle_id"],
            row["qa_id"],
        ) in selected_keys
    release = build_release(candidates_by_pool, chosen)
    items = [qa for bundle in release.values() for qa in bundle["QA"].values()]
    three_count = sum(qa["source_document_count"] == 3 for qa in items)
    summary = {
        "status": "complete",
        "target": args.target,
        "released_questions": len(items),
        "released_bundles": len(release),
        "eligible_by_pool": dict(sorted(eligible_counts.items())),
        "dual_keep_eligible_before_cross_pool_dedup": len(all_eligible),
        "cross_pool_duplicates_removed": len(duplicate_ledger),
        "three_document_questions": three_count,
        "three_document_ratio": three_count / len(items),
        "reasoning_type_distribution": dict(
            sorted(Counter(qa["reasoning_type"] for qa in items).items())
        ),
        "policy": (
            "global near-duplicate removal + Claude/Gemini strict KEEP or "
            "criteria-true no-op FIX + all hard static gates"
        ),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(args.output, release)
    atomic_json(
        args.ledger_output or args.output.with_name(args.output.stem + "_ledger.json"),
        {"candidate_ledger": ledger, "cross_pool_duplicates": duplicate_ledger},
    )
    atomic_json(
        args.summary_output or args.output.with_name(args.output.stem + "_summary.json"),
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
