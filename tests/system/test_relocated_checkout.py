"""Verify entrypoints against a moved checkout without the installed package."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_release_and_evaluation_plan_follow_the_relocated_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "renamed-checkout"
    for relative in ("src", "schemas", "data/qa/7.final_2200"):
        shutil.copytree(
            ROOT / relative, checkout / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
    env = {
        **os.environ,
        "PYTHONPATH": str(checkout / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def run(module: str, *args: str):
        result = subprocess.run(
            [sys.executable, "-S", "-m", module, *args],
            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert str(ROOT) not in result.stdout
        return json.loads(result.stdout)

    release = run("pku_qa.workflows.operations.inspect_final_2200")
    assert release["status"] == "valid"
    assert release["qa_count"] == 2200
    plan = run("pku_qa.workflows.reporting.run_final_2200_evaluation", "--print-plan")
    assert plan["status"] == "ready"
    assert len(plan["commands"]) == 4
    for command in plan["commands"]:
        qa_path = checkout / command[command.index("--qa-json") + 1]
        output = Path(command[command.index("--output-dir") + 1])
        assert qa_path.is_file()
        assert output.is_relative_to(checkout / "data/results/evaluations/final_2200")
