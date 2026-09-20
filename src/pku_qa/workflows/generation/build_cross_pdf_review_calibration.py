#!/usr/bin/env python3
"""Build a deterministic, pre-model calibration sample for API reviewers."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QA = ROOT / "data/qa/qa_cross_pdf_manual_verified_500_20260724.json"
DEFAULT_MANIFEST = (
    ROOT / "data/qa_generation/expansion_20260711/cross_pdf_bundle_manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "data/qa/4.cross_pdf/semantic_reaudit/calibration_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--known-risk-bundle",
        default="xb_0017",
        help="Selected before model runs because manual inspection found a false contrast.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def profile(
    bundle_id: str,
    bundle_qa: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    qas = list(bundle_qa["QA"].values())
    modalities = Counter(
        modality for qa in qas for modality in qa.get("modal_types", [])
    )
    non_text = sum(
        bool(set(qa.get("modal_types", [])) - {"text"}) for qa in qas
    )
    source_pages = sum(int(source["source_page_count"]) for source in manifest["sources"])
    return {
        "bundle_id": bundle_id,
        "qa_count": len(qas),
        "source_count": len(manifest["sources"]),
        "source_pages": source_pages,
        "non_text_qa_count": non_text,
        "modalities": dict(sorted(modalities.items())),
        "question_categories": dict(
            sorted(Counter(qa.get("question_category") for qa in qas).items())
        ),
    }


def add_reason(
    selected: dict[str, dict[str, Any]],
    row: dict[str, Any],
    reason: str,
) -> None:
    bundle_id = row["bundle_id"]
    if bundle_id not in selected:
        selected[bundle_id] = {**row, "selection_reasons": []}
    selected[bundle_id]["selection_reasons"].append(reason)


def main() -> int:
    args = parse_args()
    if args.sample_size < 5:
        raise SystemExit("--sample-size must be at least 5")
    qa = load_json(args.qa_json)
    manifests = {
        row["id"]: row for row in load_json(args.manifest_json)
    }
    rows = [profile(bundle_id, bundle, manifests[bundle_id]) for bundle_id, bundle in qa.items()]
    by_id = {row["bundle_id"]: row for row in rows}
    if args.known_risk_bundle not in by_id:
        raise SystemExit(f"Unknown risk bundle: {args.known_risk_bundle}")

    selected: dict[str, dict[str, Any]] = {}
    add_reason(
        selected,
        by_id[args.known_risk_bundle],
        "predeclared_known_risk_false_contrast",
    )
    add_reason(
        selected,
        max(rows, key=lambda row: (row["source_pages"], row["bundle_id"])),
        "maximum_source_page_count",
    )
    add_reason(
        selected,
        min(rows, key=lambda row: (row["source_pages"], row["bundle_id"])),
        "minimum_source_page_count",
    )
    for rank, row in enumerate(
        sorted(
            rows,
            key=lambda row: (
                -row["non_text_qa_count"],
                -row["source_pages"],
                row["bundle_id"],
            ),
        )[:2],
        start=1,
    ):
        add_reason(selected, row, f"non_text_coverage_rank_{rank}")

    remaining = [row for row in rows if row["bundle_id"] not in selected]
    rng = random.Random(args.seed)
    random_count = min(args.sample_size - len(selected), len(remaining))
    for row in sorted(rng.sample(remaining, random_count), key=lambda item: item["bundle_id"]):
        add_reason(selected, row, f"seeded_random_{args.seed}")

    ordered = list(selected.values())
    if len(ordered) != args.sample_size:
        raise RuntimeError(
            f"Expected {args.sample_size} unique bundles, selected {len(ordered)}"
        )
    output = {
        "created_before_api_review": True,
        "qa_source": str(args.qa_json),
        "manifest_source": str(args.manifest_json),
        "sample_size_bundles": len(ordered),
        "sample_size_qas": sum(row["qa_count"] for row in ordered),
        "seed": args.seed,
        "selection_method": [
            "one predeclared known-risk bundle",
            "maximum source-page bundle",
            "minimum source-page bundle",
            "two bundles ranked by non-text QA coverage",
            "remaining slots sampled without replacement using the fixed seed",
        ],
        "bundles": ordered,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    list_path = args.output_json.with_suffix(".txt")
    list_path.write_text(
        "".join(f"{row['bundle_id']}\n" for row in ordered),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(ordered)} bundles / {output['sample_size_qas']} QA to "
        f"{args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
