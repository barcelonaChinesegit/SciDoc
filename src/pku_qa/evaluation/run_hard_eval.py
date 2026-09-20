#!/usr/bin/env python3
"""Hard short-answer evaluation scheduler.

This runner is intentionally conservative:
- 4B and 8B inference can run at the same time on separate GPU groups.
- The 27B judge starts only after all inference processes finish.
- Judge GPUs are required to be A800 cards unless explicitly overridden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from adaptive_gpu_pool import AdaptiveGpuPool
from durable_work_queue import DurablePaperQueue
from evaluation_protocol import (
    STRICT_INFERENCE_REQUIRED_QA_FIELDS,
    PDF_INPUT_MODE,
    configure_protocol,
    build_inference_queue_value_contract,
    build_inference_protocol_metadata,
    build_judge_queue_contract,
    build_judge_queue_value_contract,
    pdf_corpus_sha256,
    pdf_corpus_sha256_from_manifest,
    pdf_sha256_manifest,
    provider_runtime_identity,
    sha256_file,
    validate_dataset_protocol,
)
from eval_framework import load_provider_specs, print_gpu_overview, query_gpus
from gpu_reservation import (
    child_env_without_auto,
    managed_gpu_reservation,
)
from progress_logging import progress_fields


MODELS = ("4B", "8B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard short-answer PDF evaluation.")
    parser.add_argument("--qa-json", default="data/qa/1.base/rel__single_pdf__short_answer_unanswerable__n4451__v1.json")
    parser.add_argument("--pdf-dir", default="data/pdfs")
    parser.add_argument("--extra-pdf-dir", action="append", default=[])
    parser.add_argument("--output-dir", default="data/results/evaluations/short_answer_hard_pdf_eval")
    parser.add_argument("--input-mode", choices=["pdf", "question_only"], default="pdf")
    parser.add_argument(
        "--page-input-policy",
        choices=["full", "oracle", "qa_field"],
        default="full",
    )
    parser.add_argument("--qa-page-field", default="input_pages")
    parser.add_argument("--gpus-4b", nargs="+", default=[], help="GPUs used for 4B inference shards.")
    parser.add_argument("--gpus-8b", nargs="+", default=[], help="GPUs used for 8B inference shards.")
    parser.add_argument("--judge-gpus", nargs="+", default=[], help="A800 GPUs used for 27B judge shards.")
    parser.add_argument(
        "--dynamic-a800",
        action="store_true",
        help=(
            "Continuously discover free A800s and attach/detach crash-safe "
            "workers. A40 and CPU fallback are disabled."
        ),
    )
    parser.add_argument(
        "--a800-gpus",
        nargs="*",
        default=None,
        help="Optional physical A800 allowlist. Default: every A800.",
    )
    parser.add_argument("--gpu-poll-seconds", type=float, default=15)
    parser.add_argument("--max-worker-restarts", type=int, default=200)
    parser.add_argument(
        "--max-worker-backoff-seconds",
        type=float,
        default=30,
        help="Maximum delay before restarting a failed dynamic GPU worker.",
    )
    parser.add_argument(
        "--max-worker-failures-without-progress",
        type=int,
        default=3,
        help=(
            "Fail the stage after this many consecutive non-zero worker "
            "exits without an increase in durable completed-paper count."
        ),
    )
    parser.add_argument(
        "--allow-shared-a800",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow workers on A800s that contain foreign CUDA processes when "
            "the card still satisfies --shared-a800-min-free-mib."
        ),
    )
    parser.add_argument(
        "--shared-a800-min-free-mib",
        type=int,
        default=56000,
        help="Minimum free MiB required before launching a dynamic worker.",
    )
    parser.add_argument("--max-qa-retries", type=int, default=5)
    parser.add_argument("--max-paper-failures", type=int, default=20)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2)
    parser.add_argument("--provider-config", default=None)
    parser.add_argument("--inference-max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-new-tokens", type=int, default=128)
    parser.add_argument("--max-judge-retries", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--max-total-pdf-megapixels",
        type=float,
        default=0,
        help=(
            "Automatically reduce DPI for unusually long full-PDF inputs "
            "while retaining every page; 0 disables the budget."
        ),
    )
    parser.add_argument("--min-auto-dpi", type=int, default=72)
    parser.add_argument(
        "--serial-inference-models",
        action="store_true",
        help=(
            "Run the 4B and 8B inference stages sequentially. Use this for "
            "full-PDF inputs whose image tensors make concurrent models OOM."
        ),
    )
    parser.add_argument("--max-pdf-pages", type=int, default=0)
    parser.add_argument("--page-selection-seed", type=int, default=20260713)
    parser.add_argument(
        "--disable-pdf-vision-cache",
        action="store_true",
        help=(
            "Pass through to run_inference.py. Use this when multiple workers "
            "share one GPU and cached visual features leave too little memory headroom."
        ),
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument(
        "--max-skipped-illegal-rate",
        type=float,
        default=0.01,
        help=(
            "Maximum allowed fraction of structured answers that remain "
            "illegal after retries. The report stage fails above this rate."
        ),
    )
    parser.add_argument("--allow-non-a800-judge", action="store_true")
    parser.add_argument(
        "--cross-pdf-technical-report",
        action="store_true",
        help=(
            "Run the immutable Cross-PDF preflight before GPU work and write "
            "technical_report.md/json after the standard report."
        ),
    )
    parser.add_argument(
        "--reasoning-qa-technical-report",
        action="store_true",
        help=(
            "Validate the strict dual-reviewed reasoning-QA dataset before "
            "GPU work and write a reasoning-type technical report afterward."
        ),
    )
    parser.add_argument("--expected-qa-count", type=int)
    parser.add_argument("--expected-qa-sha256")
    parser.add_argument("--expected-pdf-corpus-sha256")
    parser.add_argument("--baseline-report")
    return parser.parse_args()


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def show_nvidia_smi(stage: str) -> None:
    print(f"\n[GPU] nvidia-smi before {stage}:", flush=True)
    subprocess.run(["nvidia-smi"], check=False)


def resolved_provider_identity(
    args: argparse.Namespace, provider_name: str
) -> dict[str, object]:
    cache = getattr(args, "_provider_runtime_identity_cache", None)
    if cache is None:
        cache = {}
        args._provider_runtime_identity_cache = cache
    if provider_name not in cache:
        provider_spec = load_provider_specs(args.provider_config)[provider_name]
        cache[provider_name] = provider_runtime_identity(
            provider_name, provider_spec
        )
    return cache[provider_name]


def inference_protocol_metadata(
    args: argparse.Namespace, model: str
) -> dict[str, object]:
    provider_name = (
        f"local_qwen3_vl_{model.lower()}_pdf"
        if args.pdf_mode
        else f"local_qwen3_vl_{model.lower()}"
    )
    return build_inference_protocol_metadata(
        {
            "qa_source_sha256": args.qa_source_sha256,
            "pdf_corpus_sha256": args.pdf_corpus_sha256,
            "input_mode": args.input_mode,
            "page_input_policy": args.page_input_policy,
            "qa_page_field": args.qa_page_field,
            "model": model,
            "prompt_style": (
                "pdf" if args.pdf_mode else "question_only"
            ),
            "pdf_mode": bool(args.pdf_mode),
            "max_new_tokens": args.inference_max_new_tokens,
            "max_qa_retries": args.max_qa_retries,
            "dpi": args.dpi,
            "max_total_pdf_megapixels": args.max_total_pdf_megapixels,
            "min_auto_dpi": args.min_auto_dpi,
            "max_pdf_pages": args.max_pdf_pages,
            "page_selection_seed": args.page_selection_seed,
            "provider_config_sha256": args.provider_config_sha256,
            "provider_runtime_identity": resolved_provider_identity(
                args, provider_name
            ),
        }
    )


def wait_all(processes: list[subprocess.Popen]) -> None:
    failed: list[int] = []
    for process in processes:
        returncode = process.wait()
        if returncode != 0:
            failed.append(returncode)
    if failed:
        raise subprocess.CalledProcessError(failed[0], "child process")


def merge_shards(shard_files: list[Path], out_file: Path) -> None:
    merged: dict = {}
    for shard_file in shard_files:
        if shard_file.exists():
            merged.update(json.loads(shard_file.read_text(encoding="utf-8")))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def gpu_name_map() -> dict[str, str]:
    return {str(gpu["index"]): str(gpu["name"]) for gpu in query_gpus()}


def validate_judge_gpus(judge_gpus: list[str], allow_non_a800: bool) -> None:
    names = gpu_name_map()
    missing = [gpu for gpu in judge_gpus if gpu not in names]
    if missing:
        raise ValueError(f"Unknown judge GPU id(s): {', '.join(missing)}")

    non_a800 = [f"{gpu} ({names[gpu]})" for gpu in judge_gpus if "A800" not in names[gpu].upper()]
    if non_a800 and not allow_non_a800:
        raise ValueError(
            "The 27B judge must run on A800 GPUs. Non-A800 judge GPU(s): "
            + ", ".join(non_a800)
            + ". Use --allow-non-a800-judge only if you have manually verified enough memory."
        )


def inference_cmd(
    args: argparse.Namespace,
    model: str,
    gpu: str,
    num_shards: int,
    shard_id: int,
    output_dir: Path,
) -> list[str]:
    configure_protocol(args)
    cmd = [
        sys.executable,
        "src/pku_qa/evaluation/run_inference.py",
        "--model",
        model,
        "--qa-json",
        args.qa_json,
        "--pdf-dir",
        args.pdf_dir,
        "--output-dir",
        str(output_dir),
        "--result-file",
        str(output_dir / f"results_{model}_shard{shard_id}.json"),
        "--progress-file",
        str(output_dir / f"progress_{model}_shard{shard_id}.txt"),
        "--input-mode",
        args.input_mode,
        "--page-input-policy",
        args.page_input_policy,
        "--qa-page-field",
        args.qa_page_field,
        "--gpu",
        gpu,
        "--num-shards",
        str(num_shards),
        "--shard-id",
        str(shard_id),
        "--max-new-tokens",
        str(args.inference_max_new_tokens),
        "--dpi",
        str(args.dpi),
        "--max-total-pdf-megapixels",
        str(args.max_total_pdf_megapixels),
        "--min-auto-dpi",
        str(args.min_auto_dpi),
        "--max-qa-retries",
        str(args.max_qa_retries),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
    ]
    if args.provider_config:
        cmd.extend(["--provider-config", args.provider_config])
    for pdf_dir in args.extra_pdf_dir:
        cmd.extend(["--extra-pdf-dir", pdf_dir])
    if args.disable_pdf_vision_cache:
        cmd.append("--disable-pdf-vision-cache")
    cmd.extend(["--max-pdf-pages", str(args.max_pdf_pages)])
    cmd.extend(["--page-selection-seed", str(args.page_selection_seed)])
    metadata = inference_protocol_metadata(args, model)
    cmd.extend(
        [
            "--protocol-fingerprint",
            str(metadata["protocol_fingerprint"]),
            "--qa-source-sha256",
            args.qa_source_sha256,
        ]
    )
    if args.pdf_corpus_sha256:
        cmd.extend(
            ["--pdf-corpus-sha256", args.pdf_corpus_sha256]
        )
    return cmd


def judge_cmd(
    args: argparse.Namespace,
    model: str,
    gpu: str,
    num_shards: int,
    shard_id: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        "src/pku_qa/evaluation/run_judge.py",
        "--result-file",
        str(output_dir / f"results_{model}.json"),
        "--qa-json",
        args.qa_json,
        "--output",
        str(output_dir / f"judge_{model}.json"),
        "--gpu",
        gpu,
        "--num-shards",
        str(num_shards),
        "--shard-id",
        str(shard_id),
        "--max-new-tokens",
        str(args.judge_max_new_tokens),
        "--max-judge-retries",
        str(args.max_judge_retries),
        "--expected-inference-protocol-fingerprint",
        str(inference_protocol_metadata(args, model)["protocol_fingerprint"]),
    ]
    if args.provider_config:
        cmd.extend(["--provider-config", args.provider_config])
    return cmd


def run_inference_stage(args: argparse.Namespace, output_dir: Path, child_env: dict[str, str]) -> None:
    show_nvidia_smi("inference")

    model_gpus = {"4B": args.gpus_4b, "8B": args.gpus_8b}

    def run_model(model: str) -> None:
        processes: list[subprocess.Popen] = []
        gpus = model_gpus[model]
        for shard_id, gpu in enumerate(gpus):
            cmd = inference_cmd(
                args, model, gpu, len(gpus), shard_id, output_dir
            )
            print(
                f"[inference] start {model} shard "
                f"{shard_id}/{len(gpus)} on GPU {gpu}",
                flush=True,
            )
            processes.append(subprocess.Popen(cmd, env=child_env))
        wait_all(processes)
        shard_files = [output_dir / f"results_{model}_shard{i}.json" for i in range(len(model_gpus[model]))]
        merge_shards(shard_files, output_dir / f"results_{model}.json")

    if args.serial_inference_models:
        for model in MODELS:
            run_model(model)
    else:
        processes: list[subprocess.Popen] = []
        for model in MODELS:
            gpus = model_gpus[model]
            for shard_id, gpu in enumerate(gpus):
                cmd = inference_cmd(
                    args, model, gpu, len(gpus), shard_id, output_dir
                )
                print(
                    f"[inference] start {model} shard "
                    f"{shard_id}/{len(gpus)} on GPU {gpu}",
                    flush=True,
                )
                processes.append(subprocess.Popen(cmd, env=child_env))
        wait_all(processes)
        for model in MODELS:
            shard_files = [
                output_dir / f"results_{model}_shard{i}.json"
                for i in range(len(model_gpus[model]))
            ]
            merge_shards(shard_files, output_dir / f"results_{model}.json")


def run_dynamic_inference_model(
    args: argparse.Namespace,
    model: str,
    output_dir: Path,
    child_env: dict[str, str],
) -> None:
    qa_data = json.loads(Path(args.qa_json).read_text(encoding="utf-8"))
    queue_dir = output_dir / ".adaptive_queue" / f"inference_{model}"
    protocol_metadata = inference_protocol_metadata(args, model)
    required_values, required_values_by_paper = (
        build_inference_queue_value_contract(
            qa_data,
            protocol_metadata=protocol_metadata,
            qa_source_sha256=args.qa_source_sha256,
            pdf_corpus_hash=args.pdf_corpus_sha256,
            pdf_sha256_by_paper=args.pdf_sha256_by_paper,
            input_mode=args.input_mode,
            page_input_policy=args.page_input_policy,
            prompt_style=(
                "pdf" if args.pdf_mode else "question_only"
            ),
            max_pdf_pages=args.max_pdf_pages,
            evaluated_model=model,
        )
    )
    queue = DurablePaperQueue(
        queue_dir,
        qa_data,
        max_paper_failures=args.max_paper_failures,
        require_structured_output=args.require_evidence_pages,
        required_qa_fields=STRICT_INFERENCE_REQUIRED_QA_FIELDS,
        required_qa_field_values=required_values,
        required_qa_field_values_by_paper=required_values_by_paper,
        manifest_metadata={
            "stage": "inference",
            **protocol_metadata,
        },
    )
    existing_artifacts = [
        output_dir / f"results_{model}.json",
        *sorted(output_dir.glob(f"results_{model}_shard*.json")),
    ]
    imported = queue.bootstrap(existing_artifacts)
    print(
        f"[adaptive-inference] model={model} imported={imported} "
        f"{progress_fields(queue.complete_count(), len(queue.source))} "
        f"progress_scope=inference_{model}",
        flush=True,
    )

    def command_factory(gpu: str, attempt: int) -> list[str]:
        cmd = inference_cmd(args, model, gpu, 1, 0, output_dir)
        cmd.extend(
            [
                "--work-queue-dir",
                str(queue_dir),
                "--worker-id",
                f"{model}-gpu{gpu}-attempt{attempt}",
                "--max-paper-failures",
                str(args.max_paper_failures),
            ]
        )
        return cmd

    runner = AdaptiveGpuPool(
        command_factory=command_factory,
        is_complete=queue.is_complete,
        status_factory=queue.status,
        allowed_gpus=args.a800_gpus,
        poll_seconds=args.gpu_poll_seconds,
        state_path=queue_dir / "scheduler_state.json",
        log_dir=output_dir / "adaptive_logs" / f"inference_{model}",
        max_restarts_per_gpu=args.max_worker_restarts,
        max_restart_backoff_seconds=args.max_worker_backoff_seconds,
        max_failures_without_progress=(
            args.max_worker_failures_without_progress
        ),
        allow_shared_gpus=args.allow_shared_a800,
        min_free_memory_mib=args.shared_a800_min_free_mib,
        env=child_env,
    )
    runner.run()
    queue.merge(output_dir / f"results_{model}.json")


def run_dynamic_inference_stage(
    args: argparse.Namespace,
    output_dir: Path,
    child_env: dict[str, str],
) -> None:
    show_nvidia_smi("dynamic A800 inference")
    # Full-PDF workloads remain model-serial to maximize per-card memory
    # headroom; every currently free A800 data-parallelizes the active model.
    for model in MODELS:
        run_dynamic_inference_model(args, model, output_dir, child_env)


def run_judge_stage(args: argparse.Namespace, output_dir: Path, child_env: dict[str, str]) -> None:
    show_nvidia_smi("27B judge")
    validate_judge_gpus(args.judge_gpus, args.allow_non_a800_judge)

    for model in MODELS:
        processes: list[subprocess.Popen] = []
        for shard_id, gpu in enumerate(args.judge_gpus):
            cmd = judge_cmd(args, model, gpu, len(args.judge_gpus), shard_id, output_dir)
            print(f"[judge] start {model} shard {shard_id}/{len(args.judge_gpus)} on GPU {gpu}", flush=True)
            processes.append(subprocess.Popen(cmd, env=child_env))

        wait_all(processes)
        # run_judge writes the canonical file directly when num_shards == 1.
        # Re-merging a nonexistent ``_shard0`` file would overwrite the valid
        # result with an empty object.
        if len(args.judge_gpus) > 1:
            shard_files = [
                output_dir / f"judge_{model}_shard{i}.json"
                for i in range(len(args.judge_gpus))
            ]
            merge_shards(shard_files, output_dir / f"judge_{model}.json")


def run_dynamic_judge_model(
    args: argparse.Namespace,
    model: str,
    output_dir: Path,
    child_env: dict[str, str],
) -> None:
    result_file = output_dir / f"results_{model}.json"
    inference_results = json.loads(result_file.read_text(encoding="utf-8"))
    queue_dir = output_dir / ".adaptive_queue" / f"judge_{model}"
    inference_fingerprint = str(
        inference_protocol_metadata(args, model)["protocol_fingerprint"]
    )
    required_fields, manifest_metadata = build_judge_queue_contract(
        input_mode=args.input_mode,
        judge_runner_path=Path(__file__).with_name("run_judge.py"),
        gold_source_sha256=args.qa_source_sha256,
        inference_protocol_fingerprint=inference_fingerprint,
        judge_provider_identity=resolved_provider_identity(
            args, "local_qwen3_6_27b_judge"
        ),
        judge_config={
            "judge_provider_name": "local_qwen3_6_27b_judge",
            "max_new_tokens": args.judge_max_new_tokens,
            "max_judge_retries": args.max_judge_retries,
        },
    )
    required_values, required_values_by_qa = (
        build_judge_queue_value_contract(
            inference_results,
            judge_metadata=manifest_metadata,
        )
    )
    queue = DurablePaperQueue(
        queue_dir,
        inference_results,
        max_paper_failures=args.max_paper_failures,
        required_qa_fields=required_fields,
        required_qa_field_values=required_values,
        required_qa_field_values_by_qa=required_values_by_qa,
        manifest_metadata=manifest_metadata,
    )
    existing_artifacts = [
        output_dir / f"judge_{model}.json",
        *sorted(output_dir.glob(f"judge_{model}_shard*.json")),
    ]
    imported = queue.bootstrap(existing_artifacts)
    print(
        f"[adaptive-judge] model={model} imported={imported} "
        f"{progress_fields(queue.complete_count(), len(queue.source))} "
        f"progress_scope=judge_{model}",
        flush=True,
    )

    def command_factory(gpu: str, attempt: int) -> list[str]:
        cmd = judge_cmd(args, model, gpu, 1, 0, output_dir)
        cmd.extend(
            [
                "--work-queue-dir",
                str(queue_dir),
                "--worker-id",
                f"judge-{model}-gpu{gpu}-attempt{attempt}",
            ]
        )
        return cmd

    runner = AdaptiveGpuPool(
        command_factory=command_factory,
        is_complete=queue.is_complete,
        status_factory=queue.status,
        allowed_gpus=args.a800_gpus,
        poll_seconds=args.gpu_poll_seconds,
        state_path=queue_dir / "scheduler_state.json",
        log_dir=output_dir / "adaptive_logs" / f"judge_{model}",
        max_restarts_per_gpu=args.max_worker_restarts,
        max_restart_backoff_seconds=args.max_worker_backoff_seconds,
        max_failures_without_progress=(
            args.max_worker_failures_without_progress
        ),
        allow_shared_gpus=args.allow_shared_a800,
        min_free_memory_mib=args.shared_a800_min_free_mib,
        env=child_env,
    )
    runner.run()
    queue.merge(output_dir / f"judge_{model}.json")


def run_dynamic_judge_stage(
    args: argparse.Namespace,
    output_dir: Path,
    child_env: dict[str, str],
) -> None:
    show_nvidia_smi("dynamic A800 judge")
    for model in MODELS:
        run_dynamic_judge_model(args, model, output_dir, child_env)


def run_report_stage(
    args: argparse.Namespace,
    output_dir: Path,
    child_env: dict[str, str] | None = None,
) -> None:
    cmd = [
        sys.executable,
        "src/pku_qa/evaluation/run_report.py",
        "--judge-files",
        str(output_dir / "judge_4B.json"),
        str(output_dir / "judge_8B.json"),
        "--inference-files",
        str(output_dir / "results_4B.json"),
        str(output_dir / "results_8B.json"),
        "--qa-json",
        args.qa_json,
        "--output",
        str(output_dir / "final_report.txt"),
        "--output-json",
        str(output_dir / "final_report.json"),
        "--max-skipped-illegal-rate",
        str(args.max_skipped_illegal_rate),
    ]
    for model in MODELS:
        _, judge_metadata = build_judge_queue_contract(
            input_mode=args.input_mode,
            judge_runner_path=Path(__file__).with_name("run_judge.py"),
            gold_source_sha256=args.qa_source_sha256,
            inference_protocol_fingerprint=str(
                inference_protocol_metadata(args, model)[
                    "protocol_fingerprint"
                ]
            ),
            judge_provider_identity=resolved_provider_identity(
                args, "local_qwen3_6_27b_judge"
            ),
            judge_config={
                "judge_provider_name": "local_qwen3_6_27b_judge",
                "max_new_tokens": args.judge_max_new_tokens,
                "max_judge_retries": args.max_judge_retries,
            },
        )
        cmd.extend(
            [
                "--expected-judge-fingerprint",
                f"{model}={judge_metadata['judge_protocol_fingerprint']}",
            ]
        )
    if args.input_mode == PDF_INPUT_MODE:
        for pdf_dir in (args.pdf_dir, *args.extra_pdf_dir):
            cmd.extend(["--pdf-dir", str(pdf_dir)])
    run_checked(cmd, env=child_env)


def cross_pdf_report_cmd(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    preflight_only: bool,
) -> list[str]:
    if args.expected_qa_count is None or not args.expected_qa_sha256:
        raise ValueError(
            "--cross-pdf-technical-report requires --expected-qa-count and "
            "--expected-qa-sha256"
        )
    cmd = [
        sys.executable,
        "-m",
        "pku_qa.workflows.reporting.report_cross_pdf_eval",
        "--qa-json",
        args.qa_json,
        "--pdf-dir",
        args.pdf_dir,
        "--expected-count",
        str(args.expected_qa_count),
        "--expected-sha256",
        args.expected_qa_sha256,
    ]
    if preflight_only:
        cmd.append("--preflight-only")
        return cmd
    cmd.extend(
        [
            "--judge-4b",
            str(output_dir / "judge_4B.json"),
            "--judge-8b",
            str(output_dir / "judge_8B.json"),
            "--final-report",
            str(output_dir / "final_report.json"),
            "--output",
            str(output_dir / "technical_report.md"),
            "--output-json",
            str(output_dir / "technical_report_data.json"),
        ]
    )
    if args.baseline_report:
        cmd.extend(["--baseline-report", args.baseline_report])
    return cmd


def reasoning_qa_report_cmd(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    preflight_only: bool,
) -> list[str]:
    if (
        args.expected_qa_count is None
        or not args.expected_qa_sha256
        or not args.expected_pdf_corpus_sha256
    ):
        raise ValueError(
            "--reasoning-qa-technical-report requires --expected-qa-count "
            "--expected-qa-sha256, and --expected-pdf-corpus-sha256"
        )
    cmd = [
        sys.executable,
        "-m",
        "pku_qa.workflows.reporting.report_reasoning_qa_eval",
        "--qa-json",
        args.qa_json,
        "--output-dir",
        str(output_dir),
        "--expected-count",
        str(args.expected_qa_count),
        "--expected-sha256",
        args.expected_qa_sha256,
        "--expected-pdf-corpus-sha256",
        args.expected_pdf_corpus_sha256,
    ]
    for pdf_dir in (args.pdf_dir, *args.extra_pdf_dir):
        cmd.extend(["--pdf-dir", str(pdf_dir)])
    if preflight_only:
        cmd.append("--preflight-only")
    return cmd


def main() -> None:
    args = parse_args()
    configure_protocol(args)
    qa_path = Path(args.qa_json)
    source_sha256 = hashlib.sha256(qa_path.read_bytes()).hexdigest()
    if args.expected_qa_sha256 and source_sha256 != args.expected_qa_sha256:
        raise ValueError(
            "QA source SHA256 mismatch: "
            f"expected {args.expected_qa_sha256}, found {source_sha256}"
        )
    qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
    args.qa_source_sha256 = source_sha256
    args.provider_config_sha256 = (
        sha256_file(args.provider_config) if args.provider_config else None
    )
    args.pdf_sha256_by_paper = {}
    if args.input_mode == PDF_INPUT_MODE:
        args.pdf_sha256_by_paper = pdf_sha256_manifest(
            qa_data, [args.pdf_dir, *args.extra_pdf_dir]
        )
        args.pdf_corpus_sha256 = pdf_corpus_sha256_from_manifest(
            args.pdf_sha256_by_paper
        )
    else:
        args.pdf_corpus_sha256 = None
    if (
        args.expected_pdf_corpus_sha256
        and args.pdf_corpus_sha256 != args.expected_pdf_corpus_sha256
    ):
        raise ValueError(
            "PDF corpus SHA256 mismatch: expected "
            f"{args.expected_pdf_corpus_sha256}, found "
            f"{args.pdf_corpus_sha256}"
        )
    protocol_profile = validate_dataset_protocol(qa_data, args.input_mode)
    print(
        "[protocol-preflight] "
        + json.dumps(protocol_profile, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    child_env = child_env_without_auto()
    child_env["GPU_REQUIRE_NAME"] = "A800"
    child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if args.cross_pdf_technical_report:
        run_checked(
            cross_pdf_report_cmd(args, output_dir, preflight_only=True),
            child_env,
        )
    if args.reasoning_qa_technical_report:
        run_checked(
            reasoning_qa_report_cmd(
                args, output_dir, preflight_only=True
            ),
            child_env,
        )
    needs_gpu = not (args.skip_inference and args.skip_judge)
    if not args.dynamic_a800:
        if not args.skip_inference and (not args.gpus_4b or not args.gpus_8b):
            raise ValueError("Static mode requires --gpus-4b and --gpus-8b")
        if not args.skip_judge and not args.judge_gpus:
            raise ValueError("Static mode requires --judge-gpus")
    with managed_gpu_reservation(
        "run_hard_eval.py",
        enabled=needs_gpu and not args.dynamic_a800,
    ):
        print_gpu_overview()

        if not args.skip_judge and not args.dynamic_a800:
            validate_judge_gpus(args.judge_gpus, args.allow_non_a800_judge)

        if not args.skip_inference:
            if args.dynamic_a800:
                run_dynamic_inference_stage(args, output_dir, child_env)
            else:
                run_inference_stage(args, output_dir, child_env)
        if not args.skip_judge:
            if args.dynamic_a800:
                run_dynamic_judge_stage(args, output_dir, child_env)
            else:
                run_judge_stage(args, output_dir, child_env)
        if not args.skip_report:
            run_report_stage(args, output_dir, child_env)
            if args.cross_pdf_technical_report:
                run_checked(
                    cross_pdf_report_cmd(
                        args,
                        output_dir,
                        preflight_only=False,
                    ),
                    child_env,
                )
            if args.reasoning_qa_technical_report:
                run_checked(
                    reasoning_qa_report_cmd(
                        args,
                        output_dir,
                        preflight_only=False,
                    ),
                    child_env,
                )


if __name__ == "__main__":
    main()
