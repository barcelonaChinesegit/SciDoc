#!/usr/bin/env python3
"""Scale resumable hard Cross-PDF generation over currently free A800s."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pku_qa.evaluation.adaptive_gpu_pool import AdaptiveGpuPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepared-contexts", type=Path)
    parser.add_argument("--base-source", type=Path)
    parser.add_argument("--review-metadata-dir", type=Path)
    parser.add_argument("--a800-gpus", default="2,3,4,5")
    parser.add_argument("--min-free-mib", type=int, default=64000)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/qa/4.cross_pdf/hard_expansion/full",
    )
    parser.add_argument("--target", type=int, default=900)
    parser.add_argument("--generation-rounds", type=int, default=3)
    parser.add_argument("--candidates-per-bundle", type=int, default=5)
    parser.add_argument("--max-accepted-per-bundle", type=int, default=10)
    parser.add_argument("--max-bundles", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2200)
    parser.add_argument("--min-three-doc-ratio", type=float, default=0.6)
    parser.add_argument(
        "--source-policy",
        choices=(
            "base_single_qa",
            "evidence_facts",
            "audited_evidence_facts",
            "verified_only",
            "audited_keep_fix",
            "fact_valid",
        ),
        default="audited_evidence_facts",
    )
    parser.add_argument(
        "--require-source-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-shared-terms", type=int, default=1)
    parser.add_argument(
        "--require-relation-template",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--local-self-review",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    }


def build_worker_command(args: argparse.Namespace, gpu: str, attempt: int) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "src/pku_qa/workflows/generation/build_hard_cross_pdf_qa.py"),
        "--output-dir",
        str(args.output_dir),
        "--target",
        str(args.target),
        "--gpus",
        gpu,
        "--generation-rounds",
        str(args.generation_rounds),
        "--candidates-per-bundle",
        str(args.candidates_per_bundle),
        "--max-accepted-per-bundle",
        str(args.max_accepted_per_bundle),
        "--max-bundles",
        str(args.max_bundles),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-three-doc-ratio",
        str(args.min_three_doc_ratio),
        "--source-policy",
        args.source_policy,
        "--min-shared-terms",
        str(args.min_shared_terms),
        "--worker-id",
        f"hard-cross-gpu{gpu}-attempt{attempt}",
    ]
    for option, value in (
        ("--source", args.source),
        ("--prepared-contexts", args.prepared_contexts),
        ("--base-source", args.base_source),
        ("--review-metadata-dir", args.review_metadata_dir),
    ):
        if value is not None:
            command.extend([option, str(value)])
    if args.require_source_overlap:
        command.append("--require-source-overlap")
    if args.require_relation_template:
        command.append("--require-relation-template")
    command.append(
        "--local-self-review" if args.local_self_review else "--no-local-self-review"
    )
    return command


def main() -> int:
    args = parse_args()
    allowed = [piece.strip() for piece in args.a800_gpus.split(",") if piece.strip()]
    if not allowed:
        raise SystemExit("No A800 GPU IDs supplied")
    progress_path = args.output_dir / "progress.json"

    def command_factory(gpu: str, attempt: int) -> list[str]:
        return build_worker_command(args, gpu, attempt)

    pool = AdaptiveGpuPool(
        command_factory=command_factory,
        is_complete=lambda: int(read_progress(progress_path, args.target)["completed_papers"])
        >= args.target,
        status_factory=lambda: read_progress(progress_path, args.target),
        allowed_gpus=allowed,
        poll_seconds=max(5, args.poll_seconds),
        state_path=args.output_dir / "gpu_pool_state.json",
        log_dir=args.output_dir / "gpu_logs",
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
