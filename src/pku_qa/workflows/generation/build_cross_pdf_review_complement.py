#!/usr/bin/env python3
"""Build the exact QA complement of an already-reviewed cross-PDF subset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ALL = ROOT / "data/qa/qa_expansion_cross_pdf_cleaned_20260711.json"
DEFAULT_REVIEWED = ROOT / "data/qa/qa_cross_pdf_manual_verified_500_20260724.json"
DEFAULT_OUTPUT = (
    ROOT
    / "data/qa/4.cross_pdf/semantic_reaudit"
    / "qa_cross_pdf_remaining_1941.json"
)
DEFAULT_SUMMARY = (
    ROOT / "data/qa/4.cross_pdf/semantic_reaudit/complement_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-qa-json", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--reviewed-qa-json", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def qa_count(data: dict[str, Any]) -> int:
    return sum(len(bundle.get("QA", {})) for bundle in data.values())


def main() -> int:
    args = parse_args()
    all_qa = load_json(args.all_qa_json)
    reviewed = load_json(args.reviewed_qa_json)
    if not isinstance(all_qa, dict) or not isinstance(reviewed, dict):
        raise SystemExit("Both QA files must be JSON objects keyed by bundle id.")

    mismatches: list[dict[str, str]] = []
    reviewed_keys: set[tuple[str, str]] = set()
    for bundle_id, bundle in reviewed.items():
        if bundle_id not in all_qa:
            mismatches.append(
                {"bundle_id": bundle_id, "qa_id": "*", "reason": "missing bundle"}
            )
            continue
        for qa_id, item in bundle.get("QA", {}).items():
            reviewed_keys.add((bundle_id, qa_id))
            source = all_qa[bundle_id].get("QA", {}).get(qa_id)
            if source is None:
                mismatches.append(
                    {"bundle_id": bundle_id, "qa_id": qa_id, "reason": "missing QA"}
                )
            elif source != item:
                mismatches.append(
                    {
                        "bundle_id": bundle_id,
                        "qa_id": qa_id,
                        "reason": "reviewed item differs from cleaned source",
                    }
                )
    if mismatches:
        raise RuntimeError(
            "Reviewed QA is not an exact subset of the cleaned source: "
            + json.dumps(mismatches[:20], ensure_ascii=False)
        )

    complement: dict[str, Any] = {}
    complement_keys: set[tuple[str, str]] = set()
    for bundle_id, bundle in all_qa.items():
        remaining = {
            qa_id: copy.deepcopy(item)
            for qa_id, item in bundle.get("QA", {}).items()
            if (bundle_id, qa_id) not in reviewed_keys
        }
        if not remaining:
            continue
        copied = copy.deepcopy(bundle)
        copied["QA"] = remaining
        complement[bundle_id] = copied
        complement_keys.update((bundle_id, qa_id) for qa_id in remaining)

    all_keys = {
        (bundle_id, qa_id)
        for bundle_id, bundle in all_qa.items()
        for qa_id in bundle.get("QA", {})
    }
    if reviewed_keys & complement_keys:
        raise RuntimeError("Reviewed set and complement overlap.")
    if reviewed_keys | complement_keys != all_keys:
        raise RuntimeError("Reviewed set plus complement does not reconstruct source.")

    all_count = qa_count(all_qa)
    reviewed_count = qa_count(reviewed)
    complement_count = qa_count(complement)
    if all_count != reviewed_count + complement_count:
        raise RuntimeError("QA count identity failed.")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(complement, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "all_qa_source": str(args.all_qa_json),
        "reviewed_qa_source": str(args.reviewed_qa_json),
        "complement_output": str(args.output_json),
        "all_bundle_count": len(all_qa),
        "all_qa_count": all_count,
        "reviewed_bundle_count": len(reviewed),
        "reviewed_qa_count": reviewed_count,
        "complement_bundle_count": len(complement),
        "complement_qa_count": complement_count,
        "subset_field_equality_verified": True,
        "reviewed_complement_disjoint": True,
        "union_reconstructs_all_cleaned_qa": True,
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
