#!/usr/bin/env python3
"""Validate IDs, PDF bounds, and exact output contracts without a judge."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.validation import load_gold, load_submission, validation_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path)
    args = parser.parse_args()
    try:
        gold, _ = load_gold(pdf_dir=args.pdf_dir)
        report = validation_report(load_submission(args.predictions, gold))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["illegal"] or report["missing"] or report["technical_failures"]))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
