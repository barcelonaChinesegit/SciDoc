#!/usr/bin/env python3
"""统一评测多卡并发调度器，支持带 PDF 和仅问题评测。"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from eval_framework import (
    load_provider_specs,
    print_gpu_overview,
    resolve_provider_name,
    sanitize_name,
)
from evaluation_protocol import (
    PDF_INPUT_MODE,
    configure_protocol,
    build_inference_protocol_metadata,
    build_judge_queue_contract,
    pdf_corpus_sha256_from_manifest,
    pdf_sha256_manifest,
    provider_runtime_identity,
    sha256_file,
    validate_dataset_protocol,
)
from gpu_reservation import child_env_without_auto, managed_gpu_reservation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", nargs="+", required=True)
    parser.add_argument("--provider-config", type=str, default=None)
    parser.add_argument("--inference-gpus", nargs="+", required=True, help="推理物理GPU列表 (如 2 3 4 5)")
    parser.add_argument("--judge-gpus", nargs="+", required=True, help="裁判物理GPU列表 (如 2 3 4 5)")
    parser.add_argument("--qa-json", type=str, default="data/qa/1.base/rel__single_pdf__mixed__n4211__v1.json")
    parser.add_argument("--pdf-dir", type=str, default="data/pdfs")
    parser.add_argument("--output-dir", type=str, default="data/results/evaluations/mixed_pdf_eval")
    parser.add_argument(
        "--input-mode",
        choices=["pdf", "question_only"],
        default="pdf",
        help="pdf: 输入 PDF 页面图像和问题；question_only: 只输入问题，不读取 PDF。",
    )
    parser.add_argument(
        "--page-input-policy",
        choices=["full", "oracle", "qa_field"],
        default="full",
    )
    parser.add_argument("--qa-page-field", default="input_pages")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--inference-max-new-tokens", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--max-skipped-illegal-rate", type=float, default=0.01
    )
    
    # 🌟 新增：并行模式选择
    parser.add_argument("--parallel-mode", type=str, choices=["data", "model"], default="model",
                        help="data: 每张卡跑一个子进程分片; model: 所有卡拼凑在一起跑一个大模型")
    return parser.parse_args()

def merge_shard_dicts(shard_files: list[Path], out_file: Path):
    merged = {}
    for f in shard_files:
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            merged.update(data)
            f.unlink()
    if merged:
        out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_processes(processes: list[subprocess.Popen]) -> None:
    failed = []
    for process in processes:
        returncode = process.wait()
        if returncode != 0:
            failed.append(returncode)
    if failed:
        raise subprocess.CalledProcessError(failed[0], "child process")


def actual_provider_name(original_name: str, pdf_mode: bool) -> str:
    if pdf_mode and original_name in {"4B", "8B"}:
        return f"local_qwen3_vl_{original_name.lower()}_pdf"
    return resolve_provider_name(
        model=original_name if original_name in {"4B", "8B"} else None,
        provider_name=None if original_name in {"4B", "8B"} else original_name,
    )


def inference_metadata(
    args: argparse.Namespace,
    *,
    original_name: str,
    provider_name: str,
    qa_sha256: str,
    pdf_corpus_hash: str | None,
    provider_spec: dict,
) -> dict:
    evaluated_model = (
        original_name if original_name in {"4B", "8B"} else provider_name
    )
    return build_inference_protocol_metadata(
        {
            "qa_source_sha256": qa_sha256,
            "pdf_corpus_sha256": pdf_corpus_hash,
            "input_mode": args.input_mode,
            "page_input_policy": args.page_input_policy,
            "qa_page_field": args.qa_page_field,
            "model": evaluated_model,
            "prompt_style": (
                "pdf" if args.pdf_mode else "question_only"
            ),
            "pdf_mode": bool(args.pdf_mode),
            "max_new_tokens": args.inference_max_new_tokens,
            "max_qa_retries": 3,
            "dpi": args.dpi,
            "max_total_pdf_megapixels": 0.0,
            "min_auto_dpi": 72,
            "max_pdf_pages": 0,
            "page_selection_seed": 20260713,
            "provider_config_sha256": (
                sha256_file(args.provider_config)
                if args.provider_config
                else None
            ),
            "provider_runtime_identity": provider_runtime_identity(
                provider_name, provider_spec
            ),
        }
    )

def main():
    args = parse_args()
    configure_protocol(args)
    qa_path = Path(args.qa_json)
    qa_data = json.loads(qa_path.read_text(encoding="utf-8"))
    validate_dataset_protocol(qa_data, args.input_mode)
    qa_sha256 = sha256_file(qa_path)
    pdf_corpus_hash = None
    if args.input_mode == PDF_INPUT_MODE:
        pdf_corpus_hash = pdf_corpus_sha256_from_manifest(
            pdf_sha256_manifest(qa_data, [args.pdf_dir])
        )
    provider_specs = load_provider_specs(args.provider_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_env = child_env_without_auto()
    needs_gpu = not (args.skip_inference and args.skip_judge)
    with managed_gpu_reservation("run_eval_pipeline.py", enabled=needs_gpu):
        print_gpu_overview()

        resolved = [actual_provider_name(i, args.pdf_mode) for i in args.providers]
        judge_files = []
        inference_files = []
        expected_judge_fingerprints: dict[str, str] = {}
        evaluated_model_names: dict[str, str] = {}

        for idx, provider_name in enumerate(resolved):
            original_name = args.providers[idx]
            result_key = sanitize_name(original_name)
            merged_result_file = output_dir / f"results_{result_key}.json"
            merged_judge_file = output_dir / f"judge_{result_key}.json"
            judge_files.append(str(merged_judge_file))
            inference_files.append(str(merged_result_file))
            evaluated_model = (
                original_name if original_name in {"4B", "8B"} else provider_name
            )
            evaluated_model_names[result_key] = evaluated_model
            protocol_metadata = inference_metadata(
                args,
                original_name=original_name,
                provider_name=provider_name,
                qa_sha256=qa_sha256,
                pdf_corpus_hash=pdf_corpus_hash,
                provider_spec=provider_specs[provider_name],
            )
            inference_fingerprint = str(
                protocol_metadata["protocol_fingerprint"]
            )

            # ================= 1. 推理阶段 =================
            if not args.skip_inference:
                print(f"\n🚀 开始推理 [{original_name}] | 模式: {args.parallel_mode.upper()} | 输入: {args.input_mode} | 显卡: {args.inference_gpus}")
                processes, shard_files = [], []

                if args.parallel_mode == "data":
                    num_shards = len(args.inference_gpus)
                    for shard_id, gpu in enumerate(args.inference_gpus):
                        shard_file = output_dir / f"results_{result_key}_shard{shard_id}.json"
                        progress_file = output_dir / f"progress_{result_key}_shard{shard_id}.txt"
                        shard_files.append(shard_file)
                        cmd = [sys.executable, "src/pku_qa/evaluation/run_inference.py", "--qa-json", args.qa_json, "--pdf-dir", args.pdf_dir,
                               "--input-mode", args.input_mode,
                               "--page-input-policy", args.page_input_policy,
                               "--qa-page-field", args.qa_page_field,
                               "--output-dir", str(output_dir), "--result-file", str(shard_file),
                               "--progress-file", str(progress_file),
                               "--max-new-tokens", str(args.inference_max_new_tokens), "--dpi", str(args.dpi),
                               "--gpu", gpu, "--num-shards", str(num_shards), "--shard-id", str(shard_id),
                               "--protocol-fingerprint", inference_fingerprint,
                               "--qa-source-sha256", qa_sha256]
                        if pdf_corpus_hash: cmd.extend(["--pdf-corpus-sha256", pdf_corpus_hash])
                        if args.provider_config: cmd.extend(["--provider-config", args.provider_config])
                        if original_name in {"4B", "8B"}: cmd.extend(["--model", original_name])
                        else: cmd.extend(["--provider-name", provider_name])
                        print(f"   [+] 启动 Data Shard {shard_id} on GPU {gpu}")
                        processes.append(subprocess.Popen(cmd, env=child_env))

                elif args.parallel_mode == "model":
                    # 🌟 模型并行逻辑：所有GPU打包成一个字符串传给单个进程
                    gpu_str = ",".join(args.inference_gpus)
                    cmd = [sys.executable, "src/pku_qa/evaluation/run_inference.py", "--qa-json", args.qa_json, "--pdf-dir", args.pdf_dir,
                           "--input-mode", args.input_mode,
                           "--page-input-policy", args.page_input_policy,
                           "--qa-page-field", args.qa_page_field,
                           "--output-dir", str(output_dir), "--result-file", str(merged_result_file),
                           "--progress-file", str(output_dir / f"progress_{result_key}.txt"),
                           "--max-new-tokens", str(args.inference_max_new_tokens), "--dpi", str(args.dpi),
                           "--gpu", gpu_str, "--num-shards", "1", "--shard-id", "0",
                           "--protocol-fingerprint", inference_fingerprint,
                           "--qa-source-sha256", qa_sha256]
                    if pdf_corpus_hash: cmd.extend(["--pdf-corpus-sha256", pdf_corpus_hash])
                    if args.provider_config: cmd.extend(["--provider-config", args.provider_config])
                    if original_name in {"4B", "8B"}: cmd.extend(["--model", original_name])
                    else: cmd.extend(["--provider-name", provider_name])
                    print(f"   [+] 启动超级模型实例 (Pooled VRAM) on GPUs: {gpu_str}")
                    processes.append(subprocess.Popen(cmd, env=child_env))

                wait_for_processes(processes)
                if args.parallel_mode == "data": merge_shard_dicts(shard_files, merged_result_file)

            # ================= 2. 阅卷阶段 =================
            if not args.skip_judge:
                print(f"\n⚖️ 开始裁判阅卷 [{original_name}] | 模式: {args.parallel_mode.upper()} | 显卡: {args.judge_gpus}")
                processes, shard_files = [], []

                if args.parallel_mode == "data":
                    num_shards = len(args.judge_gpus)
                    for shard_id, gpu in enumerate(args.judge_gpus):
                        shard_file = output_dir / f"judge_{result_key}_shard{shard_id}.json"
                        shard_files.append(shard_file)
                        # run_judge treats --output as the common base path
                        # and, when num_shards > 1, writes exactly the
                        # *_shard{shard_id}.json path above.  Passing
                        # shard_file itself would double-append the suffix.
                        cmd = [sys.executable, "src/pku_qa/evaluation/run_judge.py", "--result-file", str(merged_result_file),
                               "--qa-json", args.qa_json,
                               "--output", str(merged_judge_file), "--gpu", gpu, "--num-shards", str(num_shards), "--shard-id", str(shard_id),
                               "--expected-inference-protocol-fingerprint", inference_fingerprint]
                        if args.provider_config: cmd.extend(["--provider-config", args.provider_config])
                        print(f"   [+] 启动 Judge Shard {shard_id} on GPU {gpu}")
                        processes.append(subprocess.Popen(cmd, env=child_env))

                elif args.parallel_mode == "model":
                    gpu_str = ",".join(args.judge_gpus)
                    cmd = [sys.executable, "src/pku_qa/evaluation/run_judge.py", "--result-file", str(merged_result_file),
                           "--qa-json", args.qa_json,
                           "--output", str(merged_judge_file), "--gpu", gpu_str, "--num-shards", "1", "--shard-id", "0",
                           "--expected-inference-protocol-fingerprint", inference_fingerprint]
                    if args.provider_config: cmd.extend(["--provider-config", args.provider_config])
                    print(f"   [+] 启动超级裁判实例 on GPUs: {gpu_str}")
                    processes.append(subprocess.Popen(cmd, env=child_env))

                wait_for_processes(processes)
                if args.parallel_mode == "data": merge_shard_dicts(shard_files, merged_judge_file)

            _, judge_metadata = build_judge_queue_contract(
                input_mode=args.input_mode,
                judge_runner_path=Path(__file__).with_name("run_judge.py"),
                gold_source_sha256=qa_sha256,
                inference_protocol_fingerprint=inference_fingerprint,
                judge_provider_identity=provider_runtime_identity(
                    "local_qwen3_6_27b_judge",
                    provider_specs["local_qwen3_6_27b_judge"],
                ),
                judge_config={
                    "judge_provider_name": "local_qwen3_6_27b_judge",
                    "max_new_tokens": 128,
                    "max_judge_retries": 3,
                },
            )
            expected_judge_fingerprints[result_key] = str(
                judge_metadata["judge_protocol_fingerprint"]
            )

        # ================= 3. 报告生成 =================
        if not args.skip_report:
            print("\n📈 生成最终评测报告...")
            report_cmd = [sys.executable, "src/pku_qa/evaluation/run_report.py", "--judge-files", *judge_files,
                            "--inference-files", *inference_files,
                            "--qa-json", args.qa_json,
                            "--output", str(output_dir / "final_report.txt"),
                            "--output-json", str(output_dir / "final_report.json"),
                            "--max-skipped-illegal-rate",
                            str(args.max_skipped_illegal_rate),
                            ]
            for model_key in sorted(expected_judge_fingerprints):
                report_cmd.extend([
                    "--expected-judge-fingerprint",
                    f"{model_key}={expected_judge_fingerprints[model_key]}",
                    "--expected-evaluated-model",
                    f"{model_key}={evaluated_model_names[model_key]}",
                ])
            if args.input_mode == PDF_INPUT_MODE:
                report_cmd.extend(["--pdf-dir", args.pdf_dir])
            subprocess.run(report_cmd, check=True, env=child_env)

if __name__ == "__main__":
    main()
