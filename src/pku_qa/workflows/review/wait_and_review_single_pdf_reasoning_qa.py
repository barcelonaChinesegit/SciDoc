#!/usr/bin/env python3
"""Wait for the expanded candidate pool, then run strict dual-model review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--quota-retries", type=int, default=30)
    parser.add_argument("--quota-retry-wait-seconds", type=int, default=45)
    parser.add_argument(
        "--hard-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def qa_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    return sum(
        len(paper.get("QA", {}))
        for paper in data.values()
        if isinstance(paper, dict) and isinstance(paper.get("QA"), dict)
    )


def main() -> int:
    args = parse_args()
    if args.expected_candidates < args.target:
        raise SystemExit("--expected-candidates must be at least --target")
    progress_path = args.output_dir / "review_progress.json"
    while True:
        completed = qa_count(args.candidates)
        if completed >= args.expected_candidates:
            break
        atomic_json(
            progress_path,
            {
                "status": "waiting_for_candidates",
                "completed": completed,
                "total": args.expected_candidates,
                "percent": round(
                    100 * completed / args.expected_candidates, 4
                ),
                "stage": "candidate_generation_wait",
                "updated_at": time.time(),
                "resource_waiting": True,
            },
        )
        print(
            json.dumps(
                {
                    "event": "waiting_for_candidates",
                    "completed": completed,
                    "total": args.expected_candidates,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        time.sleep(max(5, args.poll_seconds))

    command = [
        sys.executable,
        str(ROOT / "src/pku_qa/workflows/review/review_single_pdf_reasoning_qa_api.py"),
        "--candidates",
        str(args.candidates),
        "--output-dir",
        str(args.output_dir),
        "--final-output",
        str(args.final_output),
        "--target",
        str(args.target),
        "--workers",
        str(args.workers),
        "--min-confidence",
        str(args.min_confidence),
        "--max-retries",
        str(args.max_retries),
        "--gemini-model",
        args.gemini_model,
        "--quota-retries",
        str(args.quota_retries),
        "--quota-retry-wait-seconds",
        str(args.quota_retry_wait_seconds),
    ]
    if args.hard_mode:
        command.append("--hard-mode")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
