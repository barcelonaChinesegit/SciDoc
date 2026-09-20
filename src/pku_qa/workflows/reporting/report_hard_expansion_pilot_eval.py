#!/usr/bin/env python3
"""Report local-Qwen accuracy by old/new pilot cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cohort_map(path: Path) -> dict[tuple[str, str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for paper_id, paper in data.items():
        for qa_id, qa in paper.get("QA", {}).items():
            cohort = str(qa.get("benchmark_cohort", "")).strip()
            if not cohort:
                raise ValueError(f"missing benchmark_cohort for {paper_id}/{qa_id}")
            mapping[(str(paper_id), str(qa_id))] = cohort
    return mapping


def summarize(path: Path, cohorts: dict[tuple[str, str], str]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for paper_id, paper in data.items():
        for qa_id, qa in paper.get("QA", {}).items():
            cohort = cohorts.get((str(paper_id), str(qa_id)), "missing")
            counts[cohort]["total"] += 1
            if qa.get("answer_is_correct") is True:
                counts[cohort]["correct"] += 1
            if qa.get("publication_eligible") is False:
                counts[cohort]["ineligible"] += 1
    return {
        cohort: {
            **counter,
            "accuracy": counter["correct"] / counter["total"] if counter["total"] else None,
        }
        for cohort, counter in sorted(counts.items())
    }


def main() -> None:
    args = parse_args()
    cohorts = cohort_map(args.qa_json)
    report = {}
    for model in ("4B", "8B"):
        path = args.eval_dir / f"judge_{model}.json"
        if not path.is_file():
            raise SystemExit(f"Missing judge output: {path}")
        report[model] = summarize(path, cohorts)
        old = report[model].get("old_baseline", {}).get("accuracy")
        new = report[model].get("new_hard_expansion", {}).get("accuracy")
        report[model]["accuracy_change_new_minus_old"] = (
            new - old if isinstance(old, float) and isinstance(new, float) else None
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
