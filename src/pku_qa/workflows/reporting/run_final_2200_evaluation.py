#!/usr/bin/env python3
"""Run the canonical four-file final-2200 evaluation with locked inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from pku_qa.workflows.selection.sync_final_2200_manifest import (
    MANIFEST_PATH,
    build_manifest,
    read_json,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = ROOT / "data/results/evaluations/final_2200"
COMPONENT_NAMES = ("ordinary", "unanswerable", "reasoning", "cross_pdf")
LOCKED_OPTIONS = {
    "--qa-json",
    "--pdf-dir",
    "--output-dir",
    "--input-mode",
    "--expected-qa-count",
    "--expected-qa-sha256",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the four manifest-locked final-2200 JSON files."
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=COMPONENT_NAMES,
        help="Component to run; repeat as needed. Default: all four in order.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Validate inputs and print commands without starting GPU work.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed to run_hard_eval.",
    )
    return parser.parse_args()


def validate_runner_args(values: Sequence[str]) -> list[str]:
    args = list(values)
    if args[:1] == ["--"]:
        args = args[1:]
    for value in args:
        option = value.split("=", 1)[0]
        if option in LOCKED_OPTIONS:
            raise ValueError(f"Final-2200 runner locks {option}")
    return args


def build_commands(
    manifest: dict[str, Any],
    components: Sequence[str],
    output_root: Path,
    runner_args: Sequence[str] = (),
) -> list[list[str]]:
    selected = set(components)
    unknown = selected.difference(COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown final-2200 components: {sorted(unknown)}")
    passthrough = validate_runner_args(runner_args)
    by_name = {
        str(component["dataset_id"]).removesuffix("_qa.json"): component
        for component in manifest["components"]
    }
    if set(by_name) != set(COMPONENT_NAMES):
        raise ValueError("Manifest does not contain the canonical four components")

    commands: list[list[str]] = []
    for name in COMPONENT_NAMES:
        if name not in selected:
            continue
        component = by_name[name]
        commands.append(
            [
                sys.executable,
                "-m",
                "pku_qa.evaluation.run_hard_eval",
                "--qa-json",
                str(component["path"]),
                "--pdf-dir",
                "data/pdfs",
                "--output-dir",
                str(output_root / name),
                "--input-mode",
                "pdf",
                "--expected-qa-count",
                str(component["qa_count"]),
                "--expected-qa-sha256",
                str(component["sha256"]),
                *passthrough,
            ]
        )
    return commands


def main() -> int:
    args = parse_args()
    expected = build_manifest()
    if read_json(MANIFEST_PATH) != expected:
        raise ValueError(f"Final-2200 manifest is stale: {MANIFEST_PATH}")
    selected = args.component or list(COMPONENT_NAMES)
    commands = build_commands(
        expected,
        selected,
        args.output_root,
        args.runner_args,
    )
    if args.print_plan:
        print(json.dumps({"status": "ready", "commands": commands}, indent=2))
        return 0
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"status": "completed", "components": selected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
