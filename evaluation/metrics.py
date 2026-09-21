"""sxz v4 scoring on explicitly bound gold and prediction records."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from math import fsum

from evaluation.judge import Judge
from evaluation.validation import Gold, Prediction
from evaluation.sxz_v4 import (calc_evidence_metrics, get_pred_answer_from_qa,
    get_pred_pages_from_qa, is_illegal_answer, normalize_space)

DISCIPLINES = ("Computer Science", "Physics", "Statistics", "Mathematics",
               "Electrical Engineering and Systems Science", "Economics",
               "Quantitative Biology", "Quantitative Finance")


def evidence_metrics(predicted, gold) -> tuple[float, float, float]:
    result = calc_evidence_metrics(list(gold), list(predicted))
    return result["e_precision"], result["e_recall"], result["e_f1"]


def recover_prediction(prediction: Prediction) -> Prediction:
    """Recover using the source rules; preserve raw bytes and strict diagnostics."""
    if prediction.status in {"missing", "technical_failure"}:
        return prediction
    record = prediction.audit.get("sxz_record")
    if record is None:
        raw = prediction.audit.get("original_raw_output")
        record = ({"answer_pre": raw} if raw is not None else
                  {"answer_pre": prediction.answer_pre, "evidence_pages_pre": prediction.evidence_pages})
    answer, parse_status, source, raw_answer = get_pred_answer_from_qa(record)
    pages, page_source = get_pred_pages_from_qa(record)
    return replace(prediction, answer_pre=answer, evidence_pages=pages,
                   status="technical_failure" if is_illegal_answer(answer, raw_answer) else "legal",
                   audit={**prediction.audit, "producer_format_status": prediction.status,
                          "sxz_parse_status": parse_status, "sxz_answer_source": source,
                          "sxz_pages_source": page_source})


def score(gold: dict[str, Gold], predictions: dict[str, Prediction], judge: Judge,
          *, detailed: bool = False) -> dict:
    if not gold:
        raise ValueError("Gold denominator cannot be empty")
    if set(predictions) - set(gold):
        raise ValueError("Unknown prediction IDs")
    rows = []
    for qa_id, g in gold.items():
        p = recover_prediction(predictions.get(qa_id, Prediction(qa_id, "missing")))
        answer_correct, decision = False, None
        e, page_count, exact = (0.0, 0.0, 0.0), 0, False
        status = p.status
        if status != "missing":
            # The experiment computes evidence BEFORE testing answer recovery.
            e = evidence_metrics(p.evidence_pages or [], g.evidence_pages)
            page_count = len(set(p.evidence_pages or []))
            exact = set(p.evidence_pages or []) == set(g.evidence_pages)
            if status == "legal" and normalize_space(g.question):
                decision = judge.decide(g, p)  # Includes Unanswerable, as in sxz.
                answer_correct = decision["decision"] == "CORRECT"
            elif status == "legal":
                status = "technical_failure"
        rows.append({"qa_id": qa_id, "task": g.task, "discipline": g.discipline, "field": g.field,
                     "status": status, "prediction_status": p.status, "answer_correct": answer_correct,
                     "evidence": e, "predicted_unique_page_count": page_count, "exact_page_set": exact,
                     "joint_correct": bool(answer_correct and exact), "judge": decision,
                     "prediction": asdict(p)})
    n = len(gold)

    def accuracy(subset):
        return {"count": len(subset), "correct": sum(r["answer_correct"] for r in subset),
                "accuracy": 100 * sum(r["answer_correct"] for r in subset) / len(subset) if subset else None}

    unresolved_evidence = sum(r["evidence"] is None for r in rows)
    evidence = {name: None if unresolved_evidence else 100 * fsum(r["evidence"][i] for r in rows) / n
                for i, name in enumerate(("E-Precision", "E-Recall", "E-F1"))}
    evidence["A-Pages"] = None if unresolved_evidence else fsum(r["predicted_unique_page_count"] for r in rows) / n
    counts = Counter(r["status"] for r in rows)
    result = {"total": n, "received": n - counts["missing"],
              "missing": counts["missing"], "legal": sum(r["prediction_status"] == "legal" for r in rows),
              "illegal": counts["illegal"], "technical_failures": counts["technical_failure"],
              "answer_metrics": {"All": accuracy(rows), **{task: accuracy([r for r in rows if r["task"] == task])
                  for task in ("General", "Unanswerable", "Reasoning", "Multi-Document")}},
              "evidence_metrics": evidence,
              "discipline_breakdown": {d: accuracy([r for r in rows if r["discipline"] == d]) for d in DISCIPLINES},
              "audit_diagnostics": {"illegal_output_rate": 100 * counts["illegal"] / n,
                  "exact_page_set_match": None if unresolved_evidence else 100 * sum(r["exact_page_set"] for r in rows) / n,
                  "joint_answer_evidence_correctness": 100 * sum(r["joint_correct"] for r in rows) / n,
                  "unresolved_illegal_evidence_items": unresolved_evidence,
                  "judge_failures_counted_zero": sum(r["judge"] is not None and r["judge"].get("decision") is None for r in rows)},
              "rows": rows}
    if detailed:
        result["field_breakdown"] = {f: accuracy([r for r in rows if r["field"] == f]) for f in sorted({g.field for g in gold.values()})}
    return result
