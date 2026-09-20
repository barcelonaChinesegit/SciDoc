"""Appendix B.12 macro metrics; never filter the gold denominator."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from math import fsum

from evaluation.judge import Judge
from evaluation.validation import Gold, Prediction

DISCIPLINES = ("Computer Science", "Physics", "Statistics", "Mathematics",
               "Electrical Engineering and Systems Science", "Economics",
               "Quantitative Biology", "Quantitative Finance")


def evidence_metrics(predicted, gold) -> tuple[float, float, float]:
    p, g = set(predicted), set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    precision, recall = len(p & g) / len(p), len(p & g) / len(g)
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(gold: dict[str, Gold], predictions: dict[str, Prediction], judge: Judge,
          *, detailed: bool = False) -> dict:
    if not gold:
        raise ValueError("Gold denominator cannot be empty")
    if set(predictions) - set(gold):
        raise ValueError("Unknown prediction IDs")
    rows = []
    for qa_id, g in gold.items():
        p = predictions.get(qa_id, Prediction(qa_id, "missing"))
        answer_correct, decision = False, None
        e, page_count, exact = (0.0, 0.0, 0.0), 0, False
        status = p.status
        if status == "legal":
            e = evidence_metrics(p.evidence_pages, g.evidence_pages)
            page_count = len(set(p.evidence_pages))
            exact = set(p.evidence_pages) == set(g.evidence_pages)
            if g.task == "Unanswerable":
                answer_correct = p.answer_pre == "Unanswerable" and p.evidence_pages == []
                decision = {"method": "exact_canonical_refusal", "decision": "CORRECT" if answer_correct else "INCORRECT"}
            else:
                decision = judge.decide(g, p)
                answer_correct = decision["decision"] == "CORRECT"
                if decision["decision"] is None:
                    status = "technical_failure"
        elif status == "illegal":
            # Historical code scores recovered pages even on invalid outputs;
            # the paper does not settle how rejected pages should be recovered.
            # Do not choose a new rule or silently turn this into a zero score.
            e, page_count, exact = None, None, None
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
              "missing": counts["missing"], "legal": sum(p.status == "legal" for p in predictions.values()),
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
