#!/usr/bin/env python3
"""Timed local PDF → judge → official scorer diagnostic on immutable gold.

Explicit new-run settings; this does not recover historical experiment settings.
Inference and judging run in separate processes to release all GPU allocations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from evaluation import __version__
from evaluation.judge import Judge, object_hash
from evaluation.metrics import score
from evaluation.prompts import PROMPT_HASH
from evaluation.validation import file_hash, load_gold, load_submission
from pku_qa.evaluation.eval_framework import atomic_write_json

TASKS = ("General", "Unanswerable", "Reasoning", "Multi-Document")


def hashes(directory: Path) -> dict:
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*.json"))}


def checkpoint_hashes(directory: Path) -> dict:
    return {p.name: file_hash(p) for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja", ".txt"}}


def select_ids(gold: dict, per_task: int, explicit: list[str] | None) -> list[str]:
    if explicit:
        if len(set(explicit)) != len(explicit) or set(explicit) - set(gold):
            raise ValueError("QA IDs must be unique known release IDs")
        return sorted(explicit)
    # Select solely by input size, never by answers or observed model outcomes.
    return sorted(g.qa_id for task in TASKS for g in sorted(
        (g for g in gold.values() if g.task == task), key=lambda g: (g.page_count, g.qa_id))[:per_task])


def export_submission(out: Path, ids: list[str]) -> Path:
    records = [json.loads((out / "records" / (qid + ".json")).read_text()) for qid in ids]
    path = out / "predictions.jsonl"
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    temp.replace(path)
    return path


def main() -> int:
    raise SystemExit(
        "This binary-paper diagnostic is retired after the sxz v4 scoring change. "
        "Saved reports remain historical evidence. Use python evaluation/evaluate.py --help "
        "for recorded-cache replay or a new local v4 Judge run."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int, required=True, help="Physical idle A800 index")
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--qa-ids", nargs="+")
    parser.add_argument("--stage", choices=("all", "inference", "judge"), default="all")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT / "sxz") or out.is_relative_to(ROOT / "data/qa"):
        parser.error("Output must be outside read-only sxz and data/qa")
    if args.per_task < 1 or args.gpu < 0:
        parser.error("Positive per-task and nonnegative GPU required")
    if args.stage == "all":
        for stage in ("inference", "judge"):
            command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(out),
                       "--gpu", str(args.gpu), "--per-task", str(args.per_task), "--stage", stage]
            if args.qa_ids:
                command += ["--qa-ids", *args.qa_ids]
            subprocess.run(command, check=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    before = hashes(ROOT / "data/qa")
    gold, metadata = load_gold()
    ids = select_ids(gold, args.per_task, args.qa_ids)
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                   "--format=csv,noheader"], text=True).strip()
    if not uuid.startswith("GPU-") or "\n" in uuid:
        raise ValueError("Expected exactly one GPU UUID")
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, GPU_REQUIRE_NAME="A800", GPU_ALLOW_SHARED="0",
                      GPU_REPOSITORY_LOCK="1", GPU_COORDINATION_AUTO="1",
                      GPU_WAIT_TIMEOUT_SECONDS="60", GPU_WAIT_POLL_SECONDS="5")
    import numpy as np
    import torch
    import transformers
    from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation
    from pku_qa.evaluation.eval_framework import LocalTransformersProvider
    from pku_qa.evaluation.run_inference import PDF_SYSTEM_PROMPT, generate_with_retries, pdf_to_images

    model_name = "Qwen3-VL-4B-Instruct" if args.stage == "inference" else "Qwen3.6-27B"
    model_dir = ROOT / "models" / model_name
    print(f"Hashing {model_name} checkpoint", flush=True)
    artifacts = checkpoint_hashes(model_dir)
    spec = {"model_path": str(model_dir), "model_class": "Qwen3VLForConditionalGeneration"
            if args.stage == "inference" else "AutoModelForImageTextToText",
            "dtype": "bfloat16", "device_map": {"": "cuda:0"}, "attn_implementation": "sdpa",
            "trust_remote_code": False, "local_files_only": True, "enable_thinking": False}
    binding = {"qa_ids": ids, "dataset_hashes": metadata["dataset_hashes"],
               "pdf_hashes": metadata["pdf_hashes"], "qa_hashes": before,
               "selection": "explicit IDs" if args.qa_ids else "shortest full PDFs per task, then QA ID",
               "pdf_prompt_sha256": file_hash(ROOT / "evaluation/prompts/pdf_inference.txt"),
               "judge_prompt_sha256": PROMPT_HASH, "dpi": 144, "page_policy": "full",
               "seed": 42, "gpu_uuid": uuid, "physical_gpu": args.gpu,
               "torch": torch.__version__, "transformers": transformers.__version__,
               "code_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in [
                   Path(__file__).resolve(), ROOT / "src/pku_qa/evaluation/run_inference.py",
                   ROOT / "src/pku_qa/evaluation/eval_framework.py", ROOT / "evaluation/validation.py",
                   ROOT / "evaluation/metrics.py", ROOT / "evaluation/judge.py"]},
               "purpose": "new local diagnostic; historical reproduction unverified"}
    manifest = out / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != binding:
        raise ValueError("Run binding changed; use a new output directory")
    atomic_write_json(manifest, binding)
    stage_binding = {"checkpoint": artifacts, "spec": spec, "max_attempts": 3 if args.stage == "inference" else 2,
                     "max_new_tokens": 512 if args.stage == "inference" else 32,
                     "do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                     "retry_backoff_seconds": 0, "decode_cleanup": False,
                     "timeout_seconds": None, "timeout_note": "local synchronous token-bounded generation"}
    stage_path = out / (args.stage + "_binding.json")
    if stage_path.exists() and json.loads(stage_path.read_text()) != stage_binding:
        raise ValueError("Checkpoint/config changed; use a new output directory")
    atomic_write_json(stage_path, stage_binding)
    (out / "records").mkdir(exist_ok=True)
    pending = [qid for qid in ids if not (out / "records" / (qid + ".json")).exists()]
    if args.stage == "inference" and not pending:
        load_submission(export_submission(out, ids), gold)
        print("Inference resume: all terminal records verified; no generation", flush=True)
        return 0
    config_path = out / "judge_config.json"
    if args.stage == "judge" and config_path.exists():
        config = json.loads(config_path.read_text())
        predictions = load_submission(out / "predictions.jsonl", gold)
        replay = score(gold, predictions, Judge(config, out / "judge_cache.jsonl"))
        if not replay["technical_failures"]:
            atomic_write_json(out / "cache_replay_report.json", replay)
            print("Judge resume: all decisions cached; no model load", flush=True)
            return 2 if replay["illegal"] else 0
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    with managed_gpu_reservation("ScienceDoc local PDF pipeline", gpu_ids=[str(args.gpu)]):
        provider = LocalTransformersProvider(model_name, spec)
        if any(p.device.type != "cuda" for p in provider.model.parameters()):
            raise RuntimeError("CPU/disk offload forbidden")
        provider.set_generation_overrides({"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
        atomic_write_json(out / (args.stage + "_generation_defaults.json"), provider.model.generation_config.to_dict())
        if args.stage == "inference":
            assets = {a["pdf_id"]: a for a in json.loads((ROOT / "data/pdf_assets_manifest.json").read_text())["assets"]}
            settings = SimpleNamespace(max_qa_retries=3, max_new_tokens=512,
                                       require_evidence_pages=True, retry_backoff_seconds=0)
            for qid in pending:
                g = gold[qid]
                trace = out / "records" / (qid + ".attempts.json")
                audit = json.loads(trace.read_text()) if trace.exists() else {}
                pages = list(range(1, g.page_count + 1))
                images = pdf_to_images(str(ROOT / "data/pdfs" / assets[g.pdf_id]["filename"]), pages, 144)
                if [p for p, _ in images] != pages:
                    raise RuntimeError(f"{qid}: incomplete PDF render")
                content = []
                for number, img in images:
                    content.extend([{"type": "text", "text": f"[Page {number}]"}, {"type": "image", "image": img}])
                content.append({"type": "text", "text": "Question: " + g.question})
                messages = [{"role": "system", "content": [{"type": "text", "text": PDF_SYSTEM_PROMPT}]},
                            {"role": "user", "content": content}]
                begin = time.monotonic()
                try:
                    raw = generate_with_retries(provider, messages, settings, allowed_evidence_pages=pages,
                                                audit=audit, audit_checkpoint=lambda a: atomic_write_json(trace, a))
                except RuntimeError as exc:
                    if not hasattr(exc, "generation_audit"):
                        raise
                    raw = ""
                atomic_write_json(out / "records" / (qid + ".json"),
                                  {"qa_id": qid, "raw_model_output": raw, "generation_audit": audit})
                atomic_write_json(out / "records" / (qid + ".timing.json"),
                                  {"seconds": time.monotonic() - begin, "page_count": g.page_count})
                print(f"{qid} {g.task}: {g.page_count} pages, {audit['final_status']}, {time.monotonic()-begin:.1f}s", flush=True)
                for _, img in images:
                    img.close()
            load_submission(export_submission(out, ids), gold)
        else:
            config = {"identity": {"model": "Qwen/Qwen3.6-27B", "checkpoint": "sha256:" + object_hash(artifacts),
                                   "revision": None, "tokenizer_sha256": artifacts["tokenizer.json"],
                                   "chat_template_sha256": artifacts["chat_template.jinja"]},
                      "backend": "local_transformers_diagnostic", "max_attempts": 2,
                      "run_binding_sha256": object_hash(binding), "stage_binding": stage_binding,
                      "generation_defaults": provider.model.generation_config.to_dict(),
                      "purpose": "new-run settings, not recovered historical settings"}
            atomic_write_json(config_path, config)
            def generate(prompt):
                raw = provider.generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}], 32)
                print(f"Judge: {raw!r}", flush=True)
                return raw
            predictions = load_submission(out / "predictions.jsonl", gold)
            judge = Judge(config, out / "judge_cache.jsonl", generate)
            report = score(gold, predictions, judge)
            report.update(metadata)
            report.update(scope="partial local PDF pipeline diagnostic", official_reproduction_verified=False,
                          evaluator_version=__version__,
                          timestamp=datetime.now(timezone.utc).isoformat(), prompt_sha256=PROMPT_HASH,
                          judge_config_sha256=judge.config_hash,
                          prediction_file_sha256=file_hash(out / "predictions.jsonl"))
            atomic_write_json(out / "report.json", report)
            replay = score(gold, predictions, Judge(config, out / "judge_cache.jsonl"))
            atomic_write_json(out / "cache_replay_report.json", replay)
            if any(report[k] != replay[k] for k in ("answer_metrics", "evidence_metrics", "discipline_breakdown")):
                raise RuntimeError("Cache replay metrics changed")
        atomic_write_json(out / (args.stage + "_timing.json"),
                          {"seconds": time.monotonic() - started,
                           "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(0)})
    if before != hashes(ROOT / "data/qa"):
        raise RuntimeError("Immutable QA bytes changed")
    print(f"{args.stage} completed; QA hashes unchanged", flush=True)
    if args.stage == "judge" and (report["illegal"] or report["technical_failures"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
