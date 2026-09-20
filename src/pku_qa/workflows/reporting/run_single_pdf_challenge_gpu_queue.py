#!/usr/bin/env python3
"""Run the canonical final single-PDF 1200 evaluation and technical report."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from progress_logging import progress_fields
from pku_qa.workflows.selection.single_pdf_release_views import (
    ORDINARY_VIEW,
    UNANSWERABLE_VIEW,
    sync_release_views,
    write_runtime_inputs,
)


EVAL_ROOT = ROOT / "data/results/evaluations"
SINGLE_PDF_EVAL_ROOT = EVAL_ROOT / "single_pdf_1200"
REPORT_DIR = ROOT / "data/results/reports/single_pdf_1200"
STATE_PATH = SINGLE_PDF_EVAL_ROOT / "state.json"
LOG_ROOT = SINGLE_PDF_EVAL_ROOT / "logs"
EVAL_PHASES = ("inference_4B", "inference_8B", "judge_4B", "judge_8B")
FULL_PDF_PIXEL_BUDGET_MP = 180


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json_if_valid(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def stage_work_snapshot(output: Path) -> dict[str, object]:
    """Aggregate the four checkpointed run_hard_eval phases."""
    completed_phases = 0
    current_phase = EVAL_PHASES[0]
    phase_completed = 0
    phase_total = 0
    work_state = "starting"
    resource_waiting = False
    inner_progress_heartbeat_at: float | None = None
    for index, phase in enumerate(EVAL_PHASES):
        scheduler = read_json_if_valid(
            output / ".adaptive_queue" / phase / "scheduler_state.json"
        )
        queue = scheduler.get("queue")
        queue = queue if isinstance(queue, dict) else {}
        completed = int(queue.get("completed_papers", 0) or 0)
        total = int(queue.get("total_papers", 0) or 0)
        phase_complete = (
            scheduler.get("status") == "completed"
            or scheduler.get("work_state") == "completed"
            or (total > 0 and completed >= total)
        )
        if phase_complete:
            completed_phases = index + 1
            continue
        current_phase = phase
        phase_completed = completed
        phase_total = total
        work_state = str(scheduler.get("work_state") or "starting")
        resource_waiting = bool(scheduler.get("resource_waiting"))
        raw_progress_at = scheduler.get(
            "progress_heartbeat_at", scheduler.get("last_progress_at")
        )
        if isinstance(raw_progress_at, (int, float)):
            inner_progress_heartbeat_at = float(raw_progress_at)
        break
    else:
        current_phase = "completed"
        work_state = "completed"

    phase_fraction = (
        min(1.0, phase_completed / phase_total) if phase_total else 0.0
    )
    stage_fraction = min(
        1.0,
        (completed_phases + phase_fraction) / len(EVAL_PHASES),
    )
    return {
        "phase": current_phase,
        "phase_completed": phase_completed,
        "phase_total": phase_total,
        "completed_phases": completed_phases,
        "stage_fraction": stage_fraction,
        "work_state": work_state,
        "resource_waiting": resource_waiting,
        "inner_progress_heartbeat_at": inner_progress_heartbeat_at,
    }


def qa_count_if_valid(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    if isinstance(data.get("QA"), dict):
        return len(data["QA"])
    return sum(
        len(paper.get("QA", {}))
        for paper in data.values()
        if isinstance(paper, dict) and isinstance(paper.get("QA"), dict)
    )


def judged_model_qa_progress(
    output: Path, expected_qa_count: int
) -> dict[str, object]:
    """Count durable, scored QA records for the two tested models."""
    by_model: dict[str, int] = {}
    for model in ("4B", "8B"):
        consolidated = output / f"judge_{model}.json"
        if consolidated.exists():
            completed = qa_count_if_valid(consolidated)
        else:
            completed = sum(
                qa_count_if_valid(path)
                for path in (
                    output / ".adaptive_queue" / f"judge_{model}" / "papers"
                ).glob("*.json")
            )
        by_model[model] = min(expected_qa_count, completed)
    return {
        "completed": sum(by_model.values()),
        "total": expected_qa_count * len(by_model),
        "by_model": by_model,
    }


def durable_phase_qa_count(
    output: Path,
    phase: str,
    model: str,
    expected_qa_count: int,
) -> int:
    prefix = "results" if phase == "inference" else "judge"
    consolidated = output / f"{prefix}_{model}.json"
    if consolidated.exists():
        completed = qa_count_if_valid(consolidated)
    else:
        completed = sum(
            qa_count_if_valid(path)
            for path in (
                output
                / ".adaptive_queue"
                / f"{phase}_{model}"
                / "papers"
            ).glob("*.json")
        )
    return min(expected_qa_count, completed)


def model_qa_pipeline_step_progress(
    output: Path, expected_qa_count: int
) -> dict[str, object]:
    by_model: dict[str, dict[str, int]] = {}
    for model in ("4B", "8B"):
        by_model[model] = {
            phase: durable_phase_qa_count(
                output, phase, model, expected_qa_count
            )
            for phase in ("inference", "judge")
        }
    completed = sum(
        count for phases in by_model.values() for count in phases.values()
    )
    scored_completed = sum(phases["judge"] for phases in by_model.values())
    return {
        "completed": completed,
        "total": expected_qa_count * len(by_model) * 2,
        "by_model": by_model,
        "scored_model_qa_completed": scored_completed,
        "scored_model_qa_total": expected_qa_count * len(by_model),
    }


def pipeline_scored_qa_progress(
    stages: list[dict[str, object]],
) -> dict[str, object]:
    breakdown: list[dict[str, object]] = []
    completed = 0
    total = 0
    for stage in stages:
        qa_total = int(stage.get("qa_count", 0) or 0)
        if not qa_total:
            qa_total = item_count(Path(stage["qa"]))
        progress = judged_model_qa_progress(Path(stage["output"]), qa_total)
        completed += int(progress["completed"])
        total += int(progress["total"])
        breakdown.append(
            {
                "stage": str(stage["name"]),
                "dataset": str(stage["qa"]),
                "qa_count": qa_total,
                "scored_model_qa_completed": progress["completed"],
                "scored_model_qa_total": progress["total"],
                "by_model": progress["by_model"],
            }
        )
    return {
        "completed": completed,
        "total": total,
        "unit": "scored_model_qa",
        "breakdown": breakdown,
    }


def pipeline_model_qa_step_progress(
    stages: list[dict[str, object]],
) -> dict[str, object]:
    breakdown: list[dict[str, object]] = []
    completed = 0
    total = 0
    scored_completed = 0
    scored_total = 0
    for stage in stages:
        qa_total = int(stage.get("qa_count", 0) or 0)
        if not qa_total:
            qa_total = item_count(Path(stage["qa"]))
        progress = model_qa_pipeline_step_progress(
            Path(stage["output"]), qa_total
        )
        completed += int(progress["completed"])
        total += int(progress["total"])
        scored_completed += int(progress["scored_model_qa_completed"])
        scored_total += int(progress["scored_model_qa_total"])
        breakdown.append(
            {
                "stage": str(stage["name"]),
                "dataset": str(stage["qa"]),
                "qa_count": qa_total,
                "pipeline_steps_completed": progress["completed"],
                "pipeline_steps_total": progress["total"],
                "scored_model_qa_completed": progress[
                    "scored_model_qa_completed"
                ],
                "scored_model_qa_total": progress["scored_model_qa_total"],
                "by_model": progress["by_model"],
            }
        )
    return {
        "completed": completed,
        "total": total,
        "unit": "model_qa_pipeline_step",
        "scored_model_qa_completed": scored_completed,
        "scored_model_qa_total": scored_total,
        "breakdown": breakdown,
    }


def run_checked_with_heartbeat(
    command: list[str],
    *,
    cwd: Path,
    heartbeat: Callable[[], None],
    env: dict[str, str] | None = None,
    stdout: TextIO | None = None,
    poll_seconds: float = 10.0,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
    )
    while True:
        heartbeat()
        returncode = process.poll()
        if returncode is not None:
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            return
        time.sleep(poll_seconds)


def resolve_npm_command(
    *,
    node_path: Path | None = None,
    npm_path: Path | None = None,
) -> list[str]:
    """Return a Node + npm-cli command that does not depend on service PATH."""
    home_bin = Path.home() / ".local/bin"
    node_candidates = [
        node_path,
        Path(value) if (value := shutil.which("node")) else None,
        home_bin / "node",
    ]
    npm_candidates = [
        npm_path,
        Path(value) if (value := shutil.which("npm")) else None,
        home_bin / "npm",
    ]
    node = next(
        (path.resolve() for path in node_candidates if path and path.is_file()),
        None,
    )
    npm_cli = next(
        (path.resolve() for path in npm_candidates if path and path.is_file()),
        None,
    )
    if node is None or npm_cli is None:
        raise FileNotFoundError(
            "Node/npm toolchain not found; checked PATH and ~/.local/bin"
        )
    return [str(node), str(npm_cli)]


def node_toolchain_env(npm_command: list[str]) -> dict[str, str]:
    """Expose the resolved Node binary to npm lifecycle scripts."""
    env = dict(os.environ)
    node_bin = str(Path(npm_command[0]).resolve().parent)
    existing = env.get("PATH", "")
    env["PATH"] = node_bin if not existing else f"{node_bin}{os.pathsep}{existing}"
    return env

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-skipped-illegal-rate", type=float, default=0.01)
    parser.add_argument("--expected-ordinary-sha256", default="")
    parser.add_argument("--expected-unanswerable-sha256", default="")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate immutable inputs and print the canonical stage plan.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def validate_single_pdf_inputs(
    args: argparse.Namespace, combined_path: Path, ablation_path: Path
) -> dict[str, int]:
    if args.expected_ordinary_sha256:
        actual = file_sha256(ORDINARY_VIEW)
        if actual != args.expected_ordinary_sha256:
            raise ValueError(
                f"ordinary hash changed: expected={args.expected_ordinary_sha256} "
                f"actual={actual}"
            )
    if args.expected_unanswerable_sha256:
        actual = file_sha256(UNANSWERABLE_VIEW)
        if actual != args.expected_unanswerable_sha256:
            raise ValueError(
                f"unanswerable hash changed: expected={args.expected_unanswerable_sha256} "
                f"actual={actual}"
            )

    ordinary = json.loads(ORDINARY_VIEW.read_text(encoding="utf-8"))
    unanswerable_dataset = json.loads(UNANSWERABLE_VIEW.read_text(encoding="utf-8"))
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    answerable = [
        qa
        for paper in ordinary.values()
        for qa in paper.get("QA", {}).values()
    ]
    unanswerable = [
        qa
        for paper in unanswerable_dataset.values()
        for qa in paper.get("QA", {}).values()
    ]
    combined_rows = [
        qa for paper in combined.values() for qa in paper.get("QA", {}).values()
    ]
    ablation_rows = [
        qa for paper in ablation.values() for qa in paper.get("QA", {}).values()
    ]
    if len(answerable) != 1000 or len(unanswerable) != 200 or len(combined_rows) != 1200:
        raise ValueError(
            "single-PDF size mismatch: "
            f"answerable={len(answerable)} unanswerable={len(unanswerable)} "
            f"combined={len(combined_rows)}"
        )
    if any(normalize_text(qa.get("answer")) == "unanswerable" for qa in answerable):
        raise ValueError("ordinary input contains an Unanswerable item")
    if any(qa.get("answer") != "Unanswerable" for qa in unanswerable):
        raise ValueError("unanswerable input contains a non-canonical answer")
    if any(
        qa.get("evidence_pages") != [] or qa.get("oracle_pages") != []
        for qa in unanswerable
    ):
        raise ValueError("unanswerable evidence_pages/oracle_pages must both be []")
    return {
        "answerable_qas": len(answerable),
        "unanswerable_qas": len(unanswerable),
        "combined_qas": len(combined_rows),
        "ablation_variants": len(ablation_rows),
    }


def build_stages(combined_path: Path, ablation_path: Path) -> list[dict[str, object]]:
    # Closed-book is a leakage diagnostic, not an answerability benchmark.
    # It intentionally excludes the 200 constructed unanswerable items.
    return [
        {
            "name": "question_only_1000",
            "qa": ORDINARY_VIEW,
            "output": SINGLE_PDF_EVAL_ROOT / "question_only_1000",
            "input_mode": "question_only",
            "policy": "full",
            "qa_page_field": "input_pages",
        },
        {
            "name": "full_1200",
            "qa": combined_path,
            "output": SINGLE_PDF_EVAL_ROOT / "full_1200",
            "policy": "full",
            "qa_page_field": "input_pages",
            "input_mode": "pdf",
        },
        {
            "name": "oracle_1200",
            "qa": combined_path,
            "output": SINGLE_PDF_EVAL_ROOT / "oracle_1200",
            "policy": "qa_field",
            "qa_page_field": "oracle_pages",
            "input_mode": "pdf",
        },
        {
            "name": f"ablation_{item_count(ablation_path)}",
            "qa": ablation_path,
            "output": SINGLE_PDF_EVAL_ROOT / f"ablation_{item_count(ablation_path)}",
            "policy": "qa_field",
            "qa_page_field": "input_pages",
            "input_mode": "pdf",
        },
    ]


def item_count(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sum(len(paper.get("QA", {})) for paper in data.values())


def build_stage_command(
    stage: dict[str, object], args: argparse.Namespace
) -> list[str]:
    qa = Path(stage["qa"])
    output = Path(stage["output"])
    cmd = [
        sys.executable,
        "src/pku_qa/evaluation/run_hard_eval.py",
        "--qa-json",
        str(qa),
        "--pdf-dir",
        "data/pdfs",
        "--output-dir",
        str(output),
        "--input-mode",
        str(stage["input_mode"]),
        "--page-input-policy",
        str(stage["policy"]),
        "--qa-page-field",
        str(stage["qa_page_field"]),
        "--dynamic-a800",
        "--a800-gpus",
        "2",
        "3",
        "4",
        "5",
        "--gpu-poll-seconds",
        "15",
        "--allow-shared-a800",
        "--shared-a800-min-free-mib",
        "56000",
        "--max-worker-backoff-seconds",
        "30",
        "--max-paper-failures",
        "20",
        "--inference-max-new-tokens",
        str(args.inference_max_new_tokens),
        "--judge-max-new-tokens",
        "32",
        "--dpi",
        "144",
        "--max-total-pdf-megapixels",
        str(FULL_PDF_PIXEL_BUDGET_MP),
        "--max-skipped-illegal-rate",
        str(args.max_skipped_illegal_rate),
    ]
    return cmd


def run_stage(
    stage: dict[str, object],
    stage_index: int,
    stage_total: int,
    args: argparse.Namespace,
    state_path: Path,
    state: dict[str, object],
    record: dict[str, object],
    stages: list[dict[str, object]],
) -> None:
    qa = Path(stage["qa"])
    output = Path(stage["output"])
    log_path = LOG_ROOT / f"{stage['name']}.log"
    cmd = build_stage_command(stage, args)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "GPU_REQUIRE_NAME": "A800",
            "GPU_WAIT_POLL_SECONDS": "15",
            "GPU_WAIT_TIMEOUT_SECONDS": "0",
            "GPU_ALLOW_SHARED": "1",
            "GPU_MIN_FREE_MEMORY_MIB": "56000",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"event=stage_started stage={stage['name']} "
            f"items={item_count(qa)} "
            f"{progress_fields(stage_index - 1, stage_total)} "
            "progress_scope=single_pdf_stages\n"
        )
        log.flush()
        def publish_heartbeat() -> None:
            snapshot = stage_work_snapshot(output)
            stage_fraction = float(snapshot["stage_fraction"])
            pipeline = pipeline_model_qa_step_progress(stages)
            completed_units = int(pipeline["completed"])
            total_units = int(pipeline["total"])
            previous_completed = int(state.get("completed", 0) or 0)
            timestamp = time.time()
            inner_progress_at = snapshot["inner_progress_heartbeat_at"]
            previous_progress_at = state.get("progress_heartbeat_at")
            if completed_units > previous_completed or (
                isinstance(inner_progress_at, (int, float))
                and (
                    not isinstance(previous_progress_at, (int, float))
                    or float(inner_progress_at) > float(previous_progress_at)
                )
            ):
                state["progress_heartbeat_at"] = timestamp
            state.update(
                {
                    "status": "running",
                    "completed": completed_units,
                    "total": total_units,
                    "percent": round(
                        100 * completed_units / total_units, 4
                    ),
                    "progress_unit": pipeline["unit"],
                    "progress_breakdown": pipeline["breakdown"],
                    "scored_model_qa_completed": pipeline[
                        "scored_model_qa_completed"
                    ],
                    "scored_model_qa_total": pipeline[
                        "scored_model_qa_total"
                    ],
                    "stage": str(stage["name"]),
                    "phase": snapshot["phase"],
                    "work_state": snapshot["work_state"],
                    "resource_waiting": snapshot["resource_waiting"],
                    "process_heartbeat_at": timestamp,
                    "updated_at": timestamp,
                }
            )
            record.update(
                {
                    "progress_percent": round(100 * stage_fraction, 4),
                    "pipeline_steps_completed": next(
                        row["pipeline_steps_completed"]
                        for row in pipeline["breakdown"]
                        if row["stage"] == str(stage["name"])
                    ),
                    "pipeline_steps_total": int(stage["qa_count"]) * 4,
                    "scored_model_qa_completed": next(
                        row["scored_model_qa_completed"]
                        for row in pipeline["breakdown"]
                        if row["stage"] == str(stage["name"])
                    ),
                    "scored_model_qa_total": int(stage["qa_count"]) * 2,
                    "phase": snapshot["phase"],
                    "phase_completed": snapshot["phase_completed"],
                    "phase_total": snapshot["phase_total"],
                    "work_state": snapshot["work_state"],
                    "resource_waiting": snapshot["resource_waiting"],
                    "process_heartbeat_at": timestamp,
                }
            )
            if snapshot["inner_progress_heartbeat_at"] is not None:
                record["inner_progress_heartbeat_at"] = snapshot[
                    "inner_progress_heartbeat_at"
                ]
            atomic_json(state_path, state)

        run_checked_with_heartbeat(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            heartbeat=publish_heartbeat,
        )


def build_metrics(
    stages: list[dict[str, object]],
    args: argparse.Namespace,
    heartbeat: Callable[[str], None] | None = None,
) -> None:
    by_kind = {
        "question_only": stages[0],
        "full": stages[1],
        "oracle": stages[2],
        "ablation": stages[3],
    }
    report_dir = REPORT_DIR
    cmd = [
        sys.executable,
        "src/pku_qa/workflows/reporting/report_single_pdf_challenge_eval.py",
        "--qa-json",
        str(stages[1]["qa"]),
        "--question-only-qa-json",
        str(stages[0]["qa"]),
        "--ablation-qa-json",
        str(stages[3]["qa"]),
        "--output-dir",
        str(report_dir),
        "--max-illegal-rate",
        str(args.max_skipped_illegal_rate),
    ]
    for label, stage in by_kind.items():
        output = Path(stage["output"])
        for model in ("4B", "8B"):
            cmd.extend(
                ["--eval", f"{label}_{model}={output / f'judge_{model}.json'}"]
            )
    run_checked_with_heartbeat(
        cmd,
        cwd=ROOT,
        heartbeat=lambda: heartbeat("metrics") if heartbeat else None,
    )
    plugin_candidates = sorted(
        glob.glob(
            str(
                Path.home()
                / ".codex/plugins/cache/openai-curated-remote/data-analytics/*"
            )
        )
    )
    if not plugin_candidates:
        raise FileNotFoundError("data-analytics report delivery plugin not found")
    plugin_root = Path(plugin_candidates[-1])
    npm_command = resolve_npm_command()
    run_checked_with_heartbeat(
        [
            *npm_command,
            "run",
            "report:deliver",
            "--",
            "--input",
            str(report_dir / "artifact.json"),
            "--output",
            str(report_dir / "technical_report.html"),
        ],
        cwd=plugin_root,
        env=node_toolchain_env(npm_command),
        heartbeat=lambda: heartbeat("technical_report") if heartbeat else None,
    )


def main() -> None:
    args = parse_args()
    sync_release_views()
    combined_path, ablation_path = write_runtime_inputs()
    input_validation = validate_single_pdf_inputs(args, combined_path, ablation_path)
    print(
        "[single-pdf-queue] input validation "
        + json.dumps(input_validation, ensure_ascii=False),
        flush=True,
    )
    stages = build_stages(combined_path, ablation_path)
    for stage in stages:
        stage["qa_count"] = item_count(Path(stage["qa"]))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "input_validation": input_validation,
                    "stages": [
                        {
                            "name": stage["name"],
                            "qa_count": stage["qa_count"],
                            "input_mode": stage["input_mode"],
                            "page_input_policy": stage["policy"],
                            "output": str(stage["output"]),
                        }
                        for stage in stages
                    ],
                    "report_dir": str(REPORT_DIR),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return
    state_path = STATE_PATH
    initial_pipeline = pipeline_model_qa_step_progress(stages)
    state: dict[str, object] = {
        "status": "running",
        "stages": [],
        "completed": initial_pipeline["completed"],
        "total": initial_pipeline["total"],
        "percent": round(
            100
            * int(initial_pipeline["completed"])
            / int(initial_pipeline["total"]),
            4,
        ),
        "progress_unit": initial_pipeline["unit"],
        "progress_breakdown": initial_pipeline["breakdown"],
        "scored_model_qa_completed": initial_pipeline[
            "scored_model_qa_completed"
        ],
        "scored_model_qa_total": initial_pipeline["scored_model_qa_total"],
        "data_contract": {
            **input_validation,
            "ordinary_path": str(ORDINARY_VIEW),
            "ordinary_sha256": file_sha256(ORDINARY_VIEW),
            "unanswerable_path": str(UNANSWERABLE_VIEW),
            "unanswerable_sha256": file_sha256(UNANSWERABLE_VIEW),
            "combined_runtime_path": str(combined_path),
            "combined_runtime_sha256": file_sha256(combined_path),
            "ablation_runtime_path": str(ablation_path),
            "ablation_runtime_sha256": file_sha256(ablation_path),
            "progress_definition": (
                "durable inference + judge QA steps across 4B and 8B; "
                "final scored_model_qa counts are reported separately; "
                "only the canonical current-run outputs are included"
            ),
        },
        "stage": "starting",
        "work_state": "starting",
        "resource_waiting": False,
        "process_heartbeat_at": time.time(),
        "updated_at": time.time(),
    }
    atomic_json(state_path, state)

    stage_total = len(stages)
    for stage_index, stage in enumerate(stages, start=1):
        record = {
            "name": stage["name"],
            "status": "running",
            "stage_index": stage_index,
            "stage_total": stage_total,
            "progress_percent": (
                100.0 * (stage_index - 1) / stage_total
            ),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        stages_state = state["stages"]
        assert isinstance(stages_state, list)
        stages_state.append(record)
        atomic_json(state_path, state)
        try:
            run_stage(
                stage,
                stage_index,
                stage_total,
                args,
                state_path,
                state,
                record,
                stages,
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            state["status"] = "failed"
            atomic_json(state_path, state)
            raise
        record["status"] = "completed"
        record["progress_percent"] = 100.0
        record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = LOG_ROOT / f"{stage['name']}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"event=stage_completed stage={stage['name']} "
                f"{progress_fields(stage_index, stage_total)} "
                "progress_scope=single_pdf_stages\n"
            )
        atomic_json(state_path, state)

    def publish_report_heartbeat(report_stage: str) -> None:
        timestamp = time.time()
        pipeline = pipeline_model_qa_step_progress(stages)
        state.update(
            {
                "status": "running",
                "completed": pipeline["completed"],
                "total": pipeline["total"],
                "percent": round(
                    100
                    * int(pipeline["completed"])
                    / int(pipeline["total"]),
                    4,
                ),
                "progress_unit": pipeline["unit"],
                "progress_breakdown": pipeline["breakdown"],
                "scored_model_qa_completed": pipeline[
                    "scored_model_qa_completed"
                ],
                "scored_model_qa_total": pipeline[
                    "scored_model_qa_total"
                ],
                "stage": report_stage,
                "phase": report_stage,
                "work_state": "running",
                "resource_waiting": False,
                "process_heartbeat_at": timestamp,
                "progress_heartbeat_at": state.get("progress_heartbeat_at", timestamp),
                "updated_at": timestamp,
            }
        )
        atomic_json(state_path, state)

    build_metrics(stages, args, publish_report_heartbeat)
    state["status"] = "completed"
    final_pipeline = pipeline_model_qa_step_progress(stages)
    state.update(
        {
            "completed": final_pipeline["completed"],
            "total": final_pipeline["total"],
            "percent": 100.0,
            "progress_unit": final_pipeline["unit"],
            "progress_breakdown": final_pipeline["breakdown"],
            "scored_model_qa_completed": final_pipeline[
                "scored_model_qa_completed"
            ],
            "scored_model_qa_total": final_pipeline[
                "scored_model_qa_total"
            ],
            "stage": "completed",
            "phase": "completed",
            "work_state": "completed",
            "resource_waiting": False,
            "process_heartbeat_at": time.time(),
            "progress_heartbeat_at": time.time(),
            "updated_at": time.time(),
        }
    )
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
