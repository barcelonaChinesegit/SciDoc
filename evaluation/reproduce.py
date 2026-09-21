#!/usr/bin/env python3
"""Reproduce sxz v4 scores and compare paper cells.

Older current-release inspection helpers remain for reading archived diagnostics.
The CLI delegates to the single current sxz v4 scorer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.judge import Judge
from evaluation.metrics import DISCIPLINES, score
from evaluation.validation import ROOT, Prediction, digest, file_hash, load_gold, validate_raw

MODELS = {
    "MiniCPM-V-2.6": "MiniCPM-V 2.6", "InternVL25-8B": "InternVL2.5-8B",
    "MiniCPM-V-4.5": "MiniCPM-V 4.5", "InternVL35-8B": "InternVL3.5-8B",
    "Gemma3-27B-IT": "Gemma 3 27B", "Mistral-Small3.1-24B": "Mistral-Small-3.1-24B",
    "Qwen3-VL-4B": "Qwen3-VL-4B", "Qwen3-VL-8B": "Qwen3-VL-8B",
    "Claude-sonncet-5": "Claude Sonnet 5", "DeepSeek-V4-flash-Vision-Exp": "DeepSeek-V4-Flash",
    "GLM-4.6V": "GLM-4.6V"}
DATASETS = ("ordinary1190", "cross_old400", "cross_hard400", "reasoning_old100", "reasoning_hard100")
V4 = "evaluation_11models_5datasets_qwen36_calibrated_fixeddenom_v4"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def reference_revision_is_rescorable(qa: dict, gold, asset: dict | None) -> bool:
    """A changed scoring reference is not itself a changed model input.

    Require an unchanged question, explicit no-gold-input provenance, and the
    release manifest's PDF identity. This permits a *new* current-gold rescore;
    it does not establish historical checkpoint/prompt/PDF-byte provenance.
    """
    if (qa.get("question") != gold.question or qa.get("gold_answer_sent") is not False
            or qa.get("gold_evidence_pages_sent") is not False or not asset):
        return False
    path = qa.get("pdf_path_resolved")
    if not isinstance(path, str) or ".." in Path(path).parts:
        return False
    aliases = [*asset.get("legacy_paths", []), "data/pdfs/" + asset["filename"]]
    if not any(path == alias or path.endswith("/" + alias) for alias in aliases):
        return False
    if "pdf_total_pages" in qa and (type(qa["pdf_total_pages"]) is not int
                                    or qa["pdf_total_pages"] != gold.page_count):
        return False
    if "pdf_sha256" in qa and qa["pdf_sha256"] != asset["sha256"]:
        return False
    return True


def load_baseline(results: Path, gold, directory: str, *, pdf_assets: dict | None = None):
    """Read one historical run without repairing raw outputs or changing gold."""
    index = {(g.legacy_paper_id, g.legacy_qa_id): qid for qid, g in gold.items()}
    if pdf_assets is None:
        pdf_assets = {a["pdf_id"]: a for a in json.loads(
            (ROOT / "data/pdf_assets_manifest.json").read_text(encoding="utf-8"))["assets"]}
    model = MODELS[directory]
    predictions, counts, details, source_hashes = {}, Counter(), [], {}
    for dataset in DATASETS:
        path = results / directory / (dataset + ".json")
        if not path.exists():
            counts["missing_result_files"] += 1
            continue
        source_hashes[str(path.relative_to(results))] = file_hash(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for paper_id, paper in data.items():
            for legacy_qa, qa in paper.get("QA", {}).items():
                counts["historical_records"] += 1
                qid = index.get((str(paper_id), str(legacy_qa)))
                if qid is None:
                    counts["unmapped_legacy_ids"] += 1
                    details.append({"model": model, "dataset": dataset, "legacy_paper": paper_id,
                        "legacy_qa": legacy_qa, "qa_id": "", "status": "unmapped", "differences": "identity", "raw_sha256": ""})
                    continue
                if qid in predictions:
                    raise ValueError(f"{model}: duplicate baseline binding for {qid}")
                g = gold[qid]
                differences = [field for field, old, new in (
                    ("question", qa.get("question"), g.question), ("answer", qa.get("answer"), g.answer),
                    ("evidence_pages", qa.get("evidence_pages"), list(g.evidence_pages))) if old != new]
                counts.update("changed_" + field for field in differences)
                # Claude's original ordinary file stores the raw JSON in answer_pre.
                # Do not reconstruct absent raw JSON from already-parsed answer/pages.
                raw = qa.get("answer_pre_raw")
                source_field = "answer_pre_raw"
                if raw is None and directory == "Claude-sonncet-5" and dataset == "ordinary1190":
                    raw, source_field = qa.get("answer_pre"), "answer_pre"
                if isinstance(raw, str):
                    p = validate_raw(qid, raw, g.page_count)
                else:
                    p = Prediction(qid, "technical_failure", errors=[{"type": "raw_output_unavailable", "message": "No original model bytes available"}])
                p.audit["historical_source"] = {"file": str(path.relative_to(results)), "field": source_field,
                    "legacy_paper": str(paper_id), "legacy_qa": str(legacy_qa), "dataset_differences": differences}
                counts["strict_raw_" + p.status] += 1
                if differences:
                    counts["dataset_version_mismatch"] += 1
                    rescorable = reference_revision_is_rescorable(qa, g, pdf_assets.get(g.pdf_id))
                    p.audit["historical_source"]["reference_revision_rescorable"] = rescorable
                    if rescorable:
                        # Never import the old answer/decision. Judge against
                        # the independent current reference; raw validity stays.
                        counts["reference_only_revision_rescored"] += 1
                        p.audit["historical_source"]["historical_reproduction_verified"] = False
                    else:
                        counts["unresolved_input_version_mismatch"] += 1
                        p.audit["raw_validation_status"] = p.status
                        p.status = "technical_failure"
                        p.errors.append({"type": "dataset_version_mismatch", "message": ", ".join(differences)})
                predictions[qid] = p
                details.append({"model": model, "dataset": dataset, "legacy_paper": paper_id,
                    "legacy_qa": legacy_qa, "qa_id": qid, "status": p.status,
                    "differences": ";".join(differences), "raw_sha256": digest(raw) if isinstance(raw, str) else ""})
    return predictions, counts, details, source_hashes


def audit_baselines(results: Path, gold, output: Path) -> dict:
    summaries, details, source_hashes = {}, [], {}
    for directory, model in MODELS.items():
        predictions, counts, model_details, hashes = load_baseline(results, gold, directory)
        details.extend(model_details)
        source_hashes.update(hashes)
        result = score(gold, predictions, Judge({}), detailed=True)
        counts["mapped_predictions"] = len(predictions)
        counts["missing_current_ids"] = len(gold) - len(predictions)
        counts["unavailable_binary_judge_decisions"] = sum(r["judge"] is not None and r["judge"].get("decision") is None for r in result["rows"])
        # No raw outputs are published by this audit; hashes and field differences suffice.
        summaries[model] = {"counts": dict(counts), "current_gold_offline_diagnostic": {k: v for k, v in result.items() if k != "rows"},
                            "official_rescore_status": "not_reproduced_no_bound_binary_judge_decisions"}
        print(model + ": " + json.dumps(dict(counts)), flush=True)
    write_csv(output / "baseline_item_audit.csv", details)
    return {"models": summaries, "source_hashes": source_hashes}


def historical_table2(history: Path, results: Path) -> tuple[dict, dict]:
    path = results / "evaluations/summary_11models_77rows_calibrated_fixed_denominator.csv"
    rows = read_csv(path)
    result = {}
    for raw_model, model in MODELS.items():
        by_dataset = {r["Dataset"]: r for r in rows if r["Model"] == raw_model}
        full = [by_dataset[d] for d in DATASETS]
        if sum(int(r["Samples"]) for r in full) != 2200:
            raise ValueError(f"{model}: historical denominator is not 2200")
        result[model] = {"All": 100 * sum(int(r["Correct"]) for r in full) / 2200,
            "General": 100 * int(by_dataset["ordinary1190_answerable"]["Correct"]) / 1000,
            "Unanswerable": 100 * int(by_dataset["ordinary1190_unanswerable"]["Correct"]) / 200,
            "Reasoning": 100 * sum(int(by_dataset[d]["Correct"]) for d in DATASETS if d.startswith("reasoning")) / 200,
            "Multi-Document": 100 * sum(int(by_dataset[d]["Correct"]) for d in DATASETS if d.startswith("cross")) / 800}
        for field, output, scale in (("E_Precision", "E-Precision", 100), ("E_Recall", "E-Recall", 100),
                                     ("E_F1", "E-F1", 100), ("Avg Pred Pages", "A-Pages", 1)):
            result[model][output] = scale * sum(float(r[field]) * int(r["Samples"]) for r in full) / 2200
    return result, {"data/results/evaluations/" + path.name: file_hash(path)}


def historical_table3(history: Path, results: Path) -> tuple[dict, dict]:
    import openpyxl

    subject_path = history / "final_2200_classification_statistics.xlsx"
    detail_path = results / "evaluations/detail_11models_5datasets.xlsx"
    subject_book = openpyxl.load_workbook(subject_path, read_only=True, data_only=True)
    subjects = list(subject_book["2200逐题明细"].iter_rows(min_row=2, values_only=True))
    subject_book.close()
    # These are explicit historical component IDs, not normalization of model outputs.
    component_map = {"ordinary_and_unanswerable_1200": "ordinary1190", "reasoning_refreshed_100": "reasoning_old100",
                     "reasoning_incremental_100": "reasoning_hard100", "cross_pdf_first_400": "cross_old400", "cross_pdf_second_400": "cross_hard400"}
    subject_counts = Counter(r[6] for r in subjects)
    book = openpyxl.load_workbook(detail_path, read_only=True, data_only=True)
    decisions = {}
    for sheet in book:
        rows = sheet.iter_rows(values_only=True)
        first, second = next(rows), next(rows)
        positions = {}
        current = None
        for i, (group, name) in enumerate(zip(first, second)):
            if group in MODELS:
                current = group
            if current and name == "Answer Correct":
                positions[current] = i
        unit, qa = second.index("Unit ID"), second.index("QA ID")
        for row in rows:
            key = (sheet.title, str(row[unit]), str(row[qa]))
            if key in decisions:
                raise ValueError(f"Duplicate historical workbook key: {key}")
            decisions[key] = {model: row[pos] == 1 for model, pos in positions.items()}
    book.close()
    correct = {m: Counter() for m in MODELS}
    unmatched = 0
    for row in subjects:
        # Read mapping from the historical script to detect unanticipated identifiers.
        component = row[2]
        dataset = component_map.get(component)
        if dataset is None:
            raise ValueError(f"Unrecognized historical component: {component}")
        key = (dataset, str(row[4]), str(row[5]))
        if key not in decisions:
            unmatched += 1
        for model in MODELS:
            correct[model][row[6]] += int(decisions.get(key, {}).get(model, False))
    result = {display: {"All": 100 * sum(correct[m].values()) / len(subjects),
                        **{d: 100 * correct[m][d] / subject_counts[d] for d in DISCIPLINES}}
              for m, display in MODELS.items()}
    return result, {"subject_counts": dict(subject_counts), "unmatched_historical_items": unmatched,
                    "source_hashes": {"sxz/" + subject_path.name: file_hash(subject_path),
                                      "data/results/evaluations/" + detail_path.name: file_hash(detail_path)}}


def comparisons(paper, historical, baselines, table):
    rows = []
    for model, metrics in paper.items():
        run = baselines["models"][model]
        for metric, reference in metrics.items():
            old = historical[model][metric]
            diagnostic = run["current_gold_offline_diagnostic"]
            new = diagnostic["evidence_metrics"].get(metric) if table == "table2" and metric in diagnostic["evidence_metrics"] else None
            # Missing semantic decisions cannot be presented as a completed answer rescore.
            rows.append({"model": model, "metric": metric, "paper": reference,
                "historical_reaggregated_full_precision": old, "historical_display": f"{old:.2f}",
                "historical_display_difference": float(f"{old:.2f}") - reference,
                "new_official_evaluator": new, "new_difference": None if new is None else new - reference,
                "status": "UNRESOLVED: see baseline audit and paper/history protocol differences"})
    return rows


def main() -> int:
    # The reproduction entry point uses the same scorer as evaluate.py.
    from evaluation.runner import main as evaluate_main
    return evaluate_main(require_paper_match=True)


if __name__ == "__main__":
    raise SystemExit(main())
