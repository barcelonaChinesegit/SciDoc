#!/usr/bin/env python3
"""Fast, exhaustive feasibility checks before spending GPU time on reproduction.

Bounds are mathematical limits, never simulated Judge decisions. Missing,
illegal and version-mismatched inputs stay in the fixed gold denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.judge import Judge
from evaluation.metrics import score
from evaluation.prompts import JUDGE_PROMPT, PROMPT_HASH
from evaluation.reproduce import MODELS, load_baseline, write_csv
from evaluation.validation import ROOT, file_hash, load_gold


def answer_bounds(rows: list[dict]) -> dict:
    """Only a legal answerable prediction awaiting its Judge can add credit."""
    pending = sum(r["prediction_status"] == "legal" and r["task"] != "Unanswerable"
                  and r["judge"].get("decision") is None for r in rows)
    correct = sum(r["answer_correct"] for r in rows)
    n = len(rows)
    return {"denominator": n, "correct_known": correct, "pending_judge": pending,
            "minimum": 100 * correct / n, "maximum": 100 * (correct + pending) / n,
            "exact": 100 * correct / n if pending == 0 else None}


def compare_bound(model, metric, paper, bound):
    # Match the evaluator's actual display rounding, including halfway cases.
    impossible = float(f"{bound['maximum']:.2f}") < paper or float(f"{bound['minimum']:.2f}") > paper
    return {"model": model, "metric": metric, "paper": paper, **bound,
            "difference": None if bound["exact"] is None else bound["exact"] - paper,
            "status": "PROVEN_MISMATCH" if impossible else ("DISPLAY_MATCH" if bound["exact"] is not None else "UNSCORED_BOUND")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data/results")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-judge-items", type=int, default=4,
                        help="Prepare complete-model local jobs only when pending items fit this budget")
    args = parser.parse_args()
    if args.max_judge_items < 0:
        parser.error("--max-judge-items must be nonnegative")
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT / "sxz") or out.is_relative_to(ROOT / "data/qa"):
        parser.error("Read-only input directory cannot be an output destination")
    out.mkdir(parents=True, exist_ok=False)
    gold, metadata = load_gold()
    paper = json.loads((ROOT / "evaluation/paper_reference.json").read_text())
    manuscript = ROOT / paper["source"]
    if file_hash(manuscript) != paper["paper_sha256"]:
        raise ValueError("Paper changed; recover current tables before comparing")
    rows2, rows3, summary, jobs, sources = [], [], {}, [], {}
    remaining = args.max_judge_items
    for directory, model in MODELS.items():
        predictions, counts, details, hashes = load_baseline(args.results_dir, gold, directory)
        sources.update(hashes)
        report = score(gold, predictions, Judge({}))
        pending = [r for r in report["rows"] if r["prediction_status"] == "legal" and
                   r["task"] != "Unanswerable" and r["judge"].get("decision") is None]
        complete_local = len(pending) <= remaining
        if complete_local:
            remaining -= len(pending)
            for row in pending:
                g, p = gold[row["qa_id"]], predictions[row["qa_id"]]
                jobs.append({"model": model, "gold": asdict(g), "prediction": asdict(p),
                             "prompt": JUDGE_PROMPT.format(question=g.question, correct=g.answer, model_answer=p.answer_pre)})
        for metric, value in paper["table2"][model].items():
            if metric in report["answer_metrics"]:
                subset = report["rows"] if metric == "All" else [r for r in report["rows"] if r["task"] == metric]
                rows2.append(compare_bound(model, metric, value, answer_bounds(subset)))
            else:
                rows2.append({"model": model, "metric": metric, "paper": value,
                    **dict.fromkeys(("denominator", "correct_known", "pending_judge", "minimum", "maximum", "exact", "difference")),
                    "status": "UNRESOLVED_ILLEGAL_EVIDENCE"})
        for metric, value in paper["table3"][model].items():
            subset = report["rows"] if metric == "All" else [r for r in report["rows"] if r["discipline"] == metric]
            rows3.append(compare_bound(model, metric, value, answer_bounds(subset)))
        errors = Counter(e["type"] for p in predictions.values() for e in p.errors)
        witnesses = [{"qa_id": qid, "errors": p.errors, "raw": p.audit.get("original_raw_output"),
                      "raw_sha256": p.audit.get("raw_output_sha256"), "source": p.audit.get("historical_source")}
                     for qid, p in predictions.items() if p.status == "illegal"][:3]
        summary[model] = {"historical_counts": dict(counts), "received": report["received"],
                          "missing": report["missing"], "illegal": report["illegal"],
                          "pending_judge": len(pending), "error_counts": dict(errors),
                          "complete_model_selected_for_local_judge": complete_local,
                          "witnesses": witnesses}
        if complete_local:
            (out / (directory + '.json')).write_text(json.dumps(report, ensure_ascii=False) + '\n')
        print(f"{model}: pending={len(pending)}, illegal={report['illegal']}, missing={report['missing']}", flush=True)
    write_csv(out / "table2_bounds.csv", rows2)
    write_csv(out / "table3_bounds.csv", rows3)
    (out / "local_jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + '\n')
    output = {"claim": "Current-release strict diagnostics differ from recorded paper cells; this does not invalidate the historical experiments or their reconstructable aggregates",
              "official_reproduction_verified": False, "models": summary, "dataset_hashes": metadata["dataset_hashes"],
              "paper_sha256": file_hash(manuscript), "prompt_sha256": PROMPT_HASH, "source_hashes": sources,
              "table2_proven_mismatches": sum(r["status"] == "PROVEN_MISMATCH" for r in rows2),
              "table3_proven_mismatches": sum(r["status"] == "PROVEN_MISMATCH" for r in rows3),
              "local_jobs": len(jobs), "semantic_jobs_not_run": sum(s["pending_judge"] for s in summary.values()) - len(jobs),
              "limitations": "Bounds are conditional on the current-release adapter and raw-output contract, not a same-run historical reproduction test. Input-version mismatches receive diagnostic technical-failure slots. Historical table provenance, Judge configuration differences, and illegal-evidence ambiguity must be reconciled separately."}
    (out / "feasibility.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in output.items() if k not in {"models", "source_hashes"}}, indent=2))
    return 2 if output["table2_proven_mismatches"] or output["table3_proven_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
