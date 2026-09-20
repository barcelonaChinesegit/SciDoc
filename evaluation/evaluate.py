#!/usr/bin/env python3
"""Evaluate a ScienceDoc PDF-mode submission against immutable release gold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import __version__
from evaluation.judge import Judge, api_generator
from evaluation.metrics import score
from evaluation.prompts import PROMPT_HASH
from evaluation.validation import ROOT, file_hash, load_gold, load_submission


def display(report: dict) -> None:
    def number(value):
        return "UNRESOLVED" if value is None else f"{value:.2f}"
    print("ScienceDoc Official Evaluation")
    print("Release status: " + report["release_status"])
    print("\nDataset:")
    for key in ("total", "received", "missing", "legal", "illegal", "technical_failures"):
        print(f"  {key.replace('_', ' ').title()}: {report[key]}")
    print("\nAnswer Accuracy (%):")
    for task, value in report["answer_metrics"].items():
        print(f"  {task}: {number(value['accuracy'])}")
    print("\nEvidence Localization:")
    for metric, value in report["evidence_metrics"].items():
        print(f"  {metric}: {number(value)}")
    print("\nAccuracy by Discipline:")
    for discipline, value in report["discipline_breakdown"].items():
        print(f"  {discipline}: {number(value['accuracy'])}")
    print("\nAudit Diagnostics:")
    for key, value in report["audit_diagnostics"].items():
        print(f"  {key}: {number(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--judge-config", type=Path, help="Explicit run configuration; never inferred")
    parser.add_argument("--judge-cache", type=Path)
    parser.add_argument("--offline", action="store_true", help="Only reuse exactly matching decisions; cache misses are technical failures")
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()
    try:
        gold, metadata = load_gold(pdf_dir=args.pdf_dir)
        predictions = load_submission(args.predictions, gold)
        config = json.loads(args.judge_config.read_text(encoding="utf-8")) if args.judge_config else {}
        if not args.offline and not config:
            raise ValueError("Supply --judge-config or --offline; paper binary-judge settings remain unresolved")
        judge = Judge(config, args.judge_cache, None if args.offline else api_generator(config))
        report = score(gold, predictions, judge, detailed=args.detailed)
        # A new fully specified paper-contract run can finish even though the old
        # paper numbers have not been reproduced. These are different claims.
        complete_run = not (report["technical_failures"] or report["illegal"] or report["missing"])
        report.update(metadata)
        report.update({"evaluator_version": __version__, "timestamp": datetime.now(timezone.utc).isoformat(),
                       "prediction_file_sha256": file_hash(args.predictions), "prompt_sha256": PROMPT_HASH,
                       "judge_identity": config.get("identity"), "judge_config_sha256": judge.config_hash,
                       "release_status": ("paper_contract_scored_historical_reproduction_unverified" if complete_run
                                          else "provisional_pending_paper_history_reconciliation"),
                       "official_reproduction_verified": False,
                       "unresolved_items": json.loads((Path(__file__).parent / "reconciliation.json").read_text())["unresolved_items"]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        display(report)
        return 0 if complete_run else 2
    except (ValueError, OSError, KeyError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
