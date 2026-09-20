#!/usr/bin/env python3
"""Continuously scale the resumable 27B QA-composition run over free A800s."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pku_qa.evaluation.adaptive_gpu_pool import AdaptiveGpuPool


DEFAULT_PROGRESS = (
    ROOT / "data/qa/3.reasoning/progress.json"
)


class SearchSpaceExhaustedError(RuntimeError):
    """The configured generation rounds ended below the accepted target."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a800-gpus", default="2,3,4,5")
    parser.add_argument("--min-free-mib", type=int, default=64000)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--generation-rounds", type=int, default=1)
    parser.add_argument("--candidates-per-paper", type=int, default=3)
    parser.add_argument("--max-accepted-per-paper", type=int, default=2)
    parser.add_argument("--stable-qa-ids", action="store_true")
    parser.add_argument("--hard-mode", action="store_true")
    parser.add_argument("--require-semantic-chain", action="store_true")
    parser.add_argument(
        "--local-self-review",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-evidence-span", type=int, default=3)
    parser.add_argument("--max-papers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    return parser.parse_args()


def read_progress(path: Path, target: int) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    completed = int(value.get("completed", 0) or 0)
    return {
        "total_papers": target,
        "completed_papers": completed,
        "unfinished_papers": max(0, target - completed),
        "active_claims": 0,
        "retryable_papers": max(0, target - completed),
        "failures": [],
        "status": str(value.get("status", "unknown")),
        "exhausted_generation_rounds": int(
            value.get("exhausted_generation_rounds", 0) or 0
        ),
    }


def is_complete_or_raise(
    path: Path, target: int, configured_generation_rounds: int
) -> bool:
    status = read_progress(path, target)
    completed = int(status["completed_papers"])
    if completed >= target:
        return True
    exhausted_rounds = int(status["exhausted_generation_rounds"])
    if (
        status["status"] == "search_space_exhausted"
        and exhausted_rounds >= configured_generation_rounds
    ):
        raise SearchSpaceExhaustedError(
            "search_space_exhausted: "
            f"accepted={completed}/{target}, "
            f"configured_generation_rounds={configured_generation_rounds}; "
            "increase --generation-rounds or expand the eligible paper pool"
        )
    return False


def main() -> int:
    args = parse_args()
    allowed = [
        piece.strip()
        for piece in args.a800_gpus.split(",")
        if piece.strip()
    ]
    if not allowed:
        raise SystemExit("No A800 GPU IDs supplied")
    output_dir = args.output_dir or DEFAULT_PROGRESS.parent
    progress_path = args.progress_path or output_dir / "progress.json"

    def command_factory(gpu: str, attempt: int) -> list[str]:
        cmd = [
            sys.executable,
            str(ROOT / "src/pku_qa/workflows/generation/build_single_pdf_reasoning_qa.py"),
            "--target",
            str(args.target),
            "--gpus",
            gpu,
            "--generation-rounds",
            str(args.generation_rounds),
            "--candidates-per-paper",
            str(args.candidates_per_paper),
            "--max-accepted-per-paper",
            str(args.max_accepted_per_paper),
            "--worker-id",
            f"reasoning-gpu{gpu}-attempt{attempt}",
            "--output-dir",
            str(output_dir),
            "--min-evidence-span",
            str(args.min_evidence_span),
            "--max-papers",
            str(args.max_papers),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.stable_qa_ids:
            cmd.append("--stable-qa-ids")
        if args.hard_mode:
            cmd.append("--hard-mode")
        if args.require_semantic_chain:
            cmd.append("--require-semantic-chain")
        cmd.append(
            "--local-self-review"
            if args.local_self_review
            else "--no-local-self-review"
        )
        return cmd

    pool = AdaptiveGpuPool(
        command_factory=command_factory,
        is_complete=lambda: is_complete_or_raise(
            progress_path,
            args.target,
            args.generation_rounds,
        ),
        status_factory=lambda: read_progress(
            progress_path, args.target
        ),
        allowed_gpus=allowed,
        poll_seconds=max(5, args.poll_seconds),
        state_path=output_dir / "gpu_pool_state.json",
        log_dir=output_dir / "gpu_logs",
        max_restarts_per_gpu=30,
        max_restart_backoff_seconds=60,
        max_failures_without_progress=3,
        allow_shared_gpus=True,
        min_free_memory_mib=args.min_free_mib,
    )
    pool.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
