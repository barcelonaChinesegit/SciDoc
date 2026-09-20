from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/reporting/report_hard_expansion_pilot_eval.py"
)
SPEC = importlib.util.spec_from_file_location("hard_expansion_pilot_report", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summarize_joins_cohort_from_independent_gold(tmp_path) -> None:
    gold = {
        "p": {
            "QA": {
                "old": {"benchmark_cohort": "old_baseline"},
                "new": {"benchmark_cohort": "new_hard_expansion"},
            }
        }
    }
    judged = {
        "p": {
            "QA": {
                "old": {"answer_is_correct": True, "publication_eligible": True},
                "new": {"answer_is_correct": False, "publication_eligible": True},
            }
        }
    }
    gold_path = tmp_path / "gold.json"
    judged_path = tmp_path / "judge.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    judged_path.write_text(json.dumps(judged), encoding="utf-8")

    summary = MODULE.summarize(judged_path, MODULE.cohort_map(gold_path))

    assert summary["old_baseline"]["accuracy"] == 1.0
    assert summary["new_hard_expansion"]["accuracy"] == 0.0
