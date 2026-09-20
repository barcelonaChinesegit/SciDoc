from evaluation.check_reproduction import answer_bounds, compare_bound


def row(status, correct=False, pending=False, task="General"):
    return {"prediction_status": status, "answer_correct": correct, "task": task,
            "judge": {"decision": None if pending else "CORRECT"} if status == "legal" else None}


def test_bounds_keep_missing_illegal_and_failed_in_denominator():
    rows = [row("legal", True), row("legal", pending=True), row("missing"), row("illegal"), row("technical_failure")]
    assert answer_bounds(rows) == {"denominator": 5, "correct_known": 1, "pending_judge": 1,
                                  "minimum": 20, "maximum": 40, "exact": None}
    assert compare_bound("test", "All", 60, answer_bounds(rows))["status"] == "PROVEN_MISMATCH"


def test_unanswerable_is_deterministic_and_never_pending():
    result = answer_bounds([row("legal", True, task="Unanswerable"), row("missing")])
    assert result["exact"] == 50 and result["pending_judge"] == 0


def test_paper_rounding_precision_does_not_create_false_mismatch():
    rows = [row("legal", True)] + [row("missing") for _ in range(2)]
    assert compare_bound("test", "All", 33.33, answer_bounds(rows))["status"] == "DISPLAY_MATCH"
    assert compare_bound("test", "All", 33.34, answer_bounds(rows))["status"] == "PROVEN_MISMATCH"


def test_legal_predictions_without_judge_are_bounds_not_zero_scores():
    b = answer_bounds([row("legal", pending=True), row("missing")])
    assert b["exact"] is None and b["maximum"] == 50
    assert compare_bound("test", "All", 25, b)["status"] == "UNSCORED_BOUND"


def test_halfway_rounding_uses_the_evaluators_display():
    b = {"minimum": 12.125, "maximum": 12.125, "exact": 12.125}
    assert compare_bound("test", "All", float(f"{12.125:.2f}"), b)["status"] == "DISPLAY_MATCH"
