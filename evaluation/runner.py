"""Portable orchestration around the unchanged sxz v4 scoring kernel.

The scorer consumes the same five self-contained result files per model.
Source inputs/caches are read-only. New local runs use a separate, identity-bound
cache; recorded-cache replay never pretends to rerun model generation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from evaluation import __version__, sxz_v4 as rules
from evaluation.judge import object_hash
from evaluation.reproduce import DATASETS, MODELS, read_csv, write_csv
from evaluation.validation import ROOT, file_hash, strict_json


def writable_output(path: Path) -> Path:
    path = path.resolve()
    for protected in (ROOT / "sxz", ROOT / "data/qa", ROOT / "data/pdfs", ROOT / "models"):
        if path == protected.resolve() or path.is_relative_to(protected.resolve()):
            raise ValueError(f"Read-only input location cannot contain outputs: {path}")
    return path


def cache_key(dataset: str, question: str, reference: str, answer: str) -> str:
    return rules.stable_key("qwen36_calibrated_semantic_triclass_v4", rules.JUDGE_RULES_SHA256,
                            dataset, question, reference, answer)


def read_recorded_cache(path: Path) -> dict:
    cache = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(cache, dict):
        raise ValueError("Expected the sxz v4 JSON object cache, not a binary/JSONL cache")
    for key, row in cache.items():
        if not isinstance(row, dict) or row.get("judge_rules_sha256") != rules.JUDGE_RULES_SHA256:
            raise ValueError(f"Cache prompt mismatch: {key}")
        try:
            expected = cache_key(*(row[k] for k in ("dataset_id", "question", "gold_answer", "pred_answer")))
        except KeyError as exc:
            raise ValueError(f"Incomplete cache binding: {key}") from exc
        if key != expected or row.get("answer_verdict") not in {"correct", "partial", "wrong"}:
            raise ValueError(f"Invalid cache input binding or verdict: {key}")
        parsed, _ = rules.parse_judge_label(row.get("judge_raw", ""))
        if parsed != row["answer_verdict"]:
            raise ValueError(f"Cached verdict disagrees with saved Judge reply: {key}")
    return cache


def required_decisions(path: Path, dataset: str) -> set[str]:
    keys = set()
    for _, _, _, qa in rules.iter_qa_records(rules.load_json(path)):
        question = rules.normalize_space(qa.get("question", ""))
        reference, source = rules.get_gold_answer_from_qa(qa)
        answer, _, _, raw = rules.get_pred_answer_from_qa(qa)
        if not rules.is_illegal_answer(answer, raw) and question and source != "missing":
            keys.add(cache_key(dataset, question, reference, answer))
    return keys


def table2_metrics(rows: list[dict]) -> dict:
    by_dataset = {row["Dataset"]: row for row in rows}
    full = [by_dataset[d] for d in DATASETS]
    result = {"All": 100 * sum(r["Correct"] for r in full) / 2200,
              "General": 100 * by_dataset["ordinary1190_answerable"]["Correct"] / 1000,
              "Unanswerable": 100 * by_dataset["ordinary1190_unanswerable"]["Correct"] / 200,
              "Reasoning": 100 * sum(by_dataset[d]["Correct"] for d in DATASETS if d.startswith("reasoning")) / 200,
              "Multi-Document": 100 * sum(by_dataset[d]["Correct"] for d in DATASETS if d.startswith("cross")) / 800}
    for field, name, scale in (("E_Precision", "E-Precision", 100), ("E_Recall", "E-Recall", 100),
                               ("E_F1", "E-F1", 100), ("Avg Pred Pages", "A-Pages", 1)):
        result[name] = scale * sum(r[field] * r["Samples"] for r in full) / 2200
    return result


def table3_metrics(details: list[dict], subjects: dict) -> tuple[dict, dict]:
    from collections import Counter
    index = {(r["dataset"], r["unit_id"], r["qa_id"]): r for r in details}
    totals = Counter(s["discipline"] for s in subjects.values())
    correct = Counter()
    missing = []
    for key, subject in subjects.items():
        if key not in index:
            missing.append(list(key))
        correct[subject["discipline"]] += index.get(key, {}).get("answer_correct", 0)
    result = {"All": 100 * sum(correct.values()) / len(subjects),
              **{s: 100 * correct[s] / n for s, n in totals.items()}}
    outside = [r for key, r in index.items() if key not in subjects]
    return result, {"subject_counts": dict(totals), "missing_subject_predictions": missing,
                    "outside_subject_predictions": len(outside),
                    "correct_outside_subject_cohort": sum(r["answer_correct"] for r in outside)}


def compare_paper(table: str, calculated: dict, references: dict) -> list[dict]:
    return [{"table": table, "model": name, "metric": metric, "paper": reference,
             "calculated": calculated[name][metric], "display": f"{calculated[name][metric]:.2f}",
             "difference_pp": float(f"{calculated[name][metric]:.2f}") - reference,
             "matches": f"{calculated[name][metric]:.2f}" == f"{reference:.2f}"}
            for name in calculated for metric, reference in references[name].items()]


def compare_summary(rows: list[dict], path: Path) -> dict:
    old = {(r["Model"], r["Dataset"]): r for r in read_csv(path)}
    compared, differences = 0, []
    for row in rows:
        key = row["Model"], row["Dataset"]
        if key not in old:
            raise ValueError(f"Missing reference summary row: {key}")
        for field, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            compared += 1
            if abs(value - float(old[key][field])) > 1e-12:
                differences.append({"model": key[0], "dataset": key[1], "field": field,
                                    "calculated": value, "recorded": old[key][field]})
    return {"numeric_cells": compared, "differences": differences, "sha256": file_hash(path)}


def run(args) -> dict:
    output = writable_output(args.output_dir)
    if args.judge_cache and (output == args.judge_cache.resolve() or output in args.judge_cache.resolve().parents):
        raise ValueError("Input Judge cache must be outside the output directory")
    for source in (args.subject_xlsx, args.reference_summary):
        if source and (source.resolve() == output or output in source.resolve().parents):
            raise ValueError("Reference inputs must be outside the output directory")
    models = args.models or list(MODELS)
    if len(set(models)) != len(models):
        raise ValueError("Each model may be selected only once")
    jobs = [(m, d, args.results_dir / m / (d + ".json")) for m in models for d in DATASETS]
    # All five components are required, even when they contain missing predictions.
    sources = {str(p.resolve()): file_hash(p) for _, _, p in jobs}
    if any(p.resolve() == output or output in p.resolve().parents for _, _, p in jobs):
        raise ValueError("Result inputs must be outside the output directory")
    if output.exists() and any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Output directory must not contain symlinked files or directories")
    source_cache = read_recorded_cache(args.judge_cache) if args.judge_cache else {}
    if args.offline:
        if not args.judge_cache:
            raise ValueError("Recorded replay requires --judge-cache")
        needed = set().union(*(required_decisions(p, d) for _, d, p in jobs))
        missing = needed - source_cache.keys()
        if missing:
            raise ValueError(f"Recorded cache has {len(missing)} missing decisions; replay aborted before scoring")
    elif args.judge_cache:
        raise ValueError("New Judge runs cannot import historical decisions; omit --judge-cache")
    subject_hash = file_hash(args.subject_xlsx) if args.subject_xlsx else None
    binding = {"scoring_protocol": "sxz_v4", "version": __version__, "input_hashes": sources,
               "prompt_sha256": rules.JUDGE_RULES_SHA256, "kernel_sha256": file_hash(Path(rules.__file__)),
               "runner_sha256": file_hash(Path(__file__)), "subject_sha256": subject_hash,
               "execution": "recorded_cache_replay" if args.offline else "new_local_judge",
               "recorded_cache_sha256": file_hash(args.judge_cache) if args.judge_cache else None}
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "run_binding.json"
    if binding_path.exists():
        if strict_json(binding_path.read_text()) != binding:
            raise ValueError("Run inputs/code changed; use a new output directory")
    elif any(output.iterdir()):
        raise ValueError("Output directory is not empty and has no matching run binding")
    rules.atomic_write_json(binding_path, binding)
    cache_path = output / "judge_cache.json"
    summaries, table2, table3, membership = [], {}, {}, {}
    subjects = None
    if args.subject_xlsx:
        from evaluation.audit_score_provenance import read_subjects
        subjects = read_subjects(args.subject_xlsx)
        if len(subjects) != 2200:
            raise ValueError("Historical subject cohort must have 2200 distinct items")
    with ExitStack() as stack:
        model = processor = None
        if not args.offline:
            # Hash exact current weights and runtime; do not claim the old missing revision.
            import subprocess
            gpu_name = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
            if "A800" not in gpu_name:
                raise ValueError("The local v4 runner requires an A800")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            sys.path.insert(0, str(ROOT / "src"))
            from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation
            stack.enter_context(managed_gpu_reservation("sxz v4 semantic evaluation", gpu_ids=[str(args.gpu)]))
            import torch
            import transformers
            artifacts = {str(p.relative_to(args.model_dir)): file_hash(p)
                         for p in sorted(args.model_dir.rglob("*")) if p.is_file()}
            if not artifacts:
                raise ValueError("Empty model checkpoint directory")
            identity = {"model": "Qwen/Qwen3.6-27B", "artifacts": artifacts, "torch": torch.__version__,
                        "transformers": transformers.__version__, "seed": 42, "do_sample": False,
                        "max_new_tokens": 32, "enable_thinking": False, "max_attempts": 1,
                        "historical_weight_identity_verified": False}
            identity_path = output / "judge_identity.json"
            if identity_path.exists() and strict_json(identity_path.read_text()) != identity:
                raise ValueError("Judge identity changed; use a new output directory")
            if cache_path.exists() and not identity_path.exists():
                raise ValueError("Cannot resume a cache without its Judge identity")
            rules.atomic_write_json(identity_path, identity)
            source_cache = read_recorded_cache(cache_path) if cache_path.exists() else {}
            rules.QWEN36_JUDGE_MODEL_PATH = str(args.model_dir.resolve())
            model, processor = rules.load_qwen36_judge()
        starting_cache_size = len(source_cache)
        for name in models:
            all_details, model_summaries = [], []
            for m, dataset, path in (j for j in jobs if j[0] == name):
                details, _ = rules.evaluate_one_result(m, dataset, path, model, processor, source_cache, cache_path)
                all_details.extend(details)
                model_summaries.extend(rules.build_summary_rows_for_job(m, dataset, path, details))
                if not args.offline:
                    rules.save_cache(cache_path, source_cache)
            model_out = writable_output(output / name)
            model_out.mkdir(exist_ok=True)
            write_csv(model_out / "details.csv", all_details)
            write_csv(model_out / "summary.csv", model_summaries)
            summaries.extend(model_summaries)
            table2[MODELS[name]] = table2_metrics(model_summaries)
            if subjects:
                table3[MODELS[name]], membership[MODELS[name]] = table3_metrics(all_details, subjects)
    # Detect changes to result/cache inputs during scoring.
    if any(file_hash(Path(p)) != expected for p, expected in sources.items()):
        raise ValueError("Result inputs changed during evaluation")
    if args.judge_cache and file_hash(args.judge_cache) != binding["recorded_cache_sha256"]:
        raise ValueError("Recorded Judge cache changed during evaluation")
    if args.subject_xlsx and file_hash(args.subject_xlsx) != subject_hash:
        raise ValueError("Subject cohort changed during evaluation")
    reference_path = Path(__file__).with_name("paper_reference.json")
    paper = strict_json(reference_path.read_text())
    comparisons = compare_paper("table2", table2, paper["table2"])
    comparisons += compare_paper("table3", table3, paper["table3"])
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paper_comparison.csv", comparisons)
    report = {"scoring_protocol": "sxz_v4", "evaluator_version": __version__,
              "execution": binding["execution"], "timestamp": datetime.now(timezone.utc).isoformat(),
              "models": models, "table2": table2, "table3": table3, "membership": membership,
              "table2_display_matches": sum(r["matches"] for r in comparisons if r["table"] == "table2"),
              "table3_display_matches": sum(r["matches"] for r in comparisons if r["table"] == "table3"),
              "paper_differences": [r for r in comparisons if not r["matches"]],
              "paper_reference_sha256": file_hash(reference_path), "binding": binding,
              "historical_weight_identity_verified": False,
              "new_judge_calls": len(source_cache) - starting_cache_size}
    if args.reference_summary:
        report["historical_summary_check"] = compare_summary(summaries, args.reference_summary)
    rules.atomic_write_json(output / "report.json", report)
    return report


def main(argv=None, *, require_paper_match=False) -> int:
    parser = argparse.ArgumentParser(description="ScienceDoc evaluation: exact sxz v4 scoring rules")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data/results")
    parser.add_argument("--model", dest="models", action="append", choices=list(MODELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-cache", type=Path, help="Read-only original v4 JSON cache, for --offline replay")
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument("--offline", action="store_true", help="Replay only bound recorded decisions; never load a model")
    execution.add_argument("--gpu", type=int, help="New local Judge run on this A800, with a new output cache")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3.6-27B")
    parser.add_argument("--subject-xlsx", type=Path, help="Original 2200-item subject cohort, for Table 3")
    parser.add_argument("--reference-summary", type=Path, help="Compare every numeric cell against the original summary")
    args = parser.parse_args(argv)
    try:
        report = run(args)
        print(json.dumps({k: report[k] for k in ("execution", "table2_display_matches", "table3_display_matches", "paper_differences")}, indent=2))
        if report.get("historical_summary_check", {}).get("differences"):
            return 2
        return 2 if require_paper_match and report["paper_differences"] else 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
