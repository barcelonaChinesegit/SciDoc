#!/usr/bin/env python3
"""Opt-in real local Judge diagnostic; never a paper-table reproduction run.

Requires the repository's ML environment, frozen PDFs, local Qwen3.6 weights,
and historical Qwen3-VL-8B outputs. All QA and historical inputs are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.judge import Judge, object_hash
from evaluation.metrics import score
from evaluation.prompts import PROMPT_HASH
from evaluation.validation import digest, file_hash, load_gold, load_submission, validate_raw

TASKS = ("General", "Unanswerable", "Reasoning", "Multi-Document")
DATASETS = ("ordinary1190", "cross_old400", "cross_hard400", "reasoning_old100", "reasoning_hard100")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def qa_hashes() -> dict:
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "data/qa").rglob("*.json"))}


def select_predictions(gold: dict, source: Path, per_task: int) -> tuple[list, dict]:
    """First IDs per task among legal raw outputs with exactly matching gold."""
    index = {(g.legacy_paper_id, g.legacy_qa_id): qid for qid, g in gold.items()}
    candidates, sources, seen = {}, {}, set()
    for dataset in DATASETS:
        path = source / (dataset + ".json")
        sources[dataset] = file_hash(path)
        for paper_id, paper in json.loads(path.read_text(encoding="utf-8")).items():
            for legacy_qa, qa in paper.get("QA", {}).items():
                qid = index.get((str(paper_id), str(legacy_qa)))
                if qid is None:
                    continue
                if qid in seen:
                    raise ValueError(f"Duplicate historical binding: {qid}")
                seen.add(qid)
                g = gold[qid]
                if (qa.get("question"), qa.get("answer"), qa.get("evidence_pages")) != (g.question, g.answer, list(g.evidence_pages)):
                    continue
                raw = qa.get("answer_pre_raw")
                if isinstance(raw, str) and validate_raw(qid, raw, g.page_count).status == "legal":
                    candidates[qid] = {"qa_id": qid, "raw_model_output": raw}
    selected, counts = [], Counter()
    for qid in sorted(candidates):
        task = gold[qid].task
        if counts[task] < per_task:
            selected.append(candidates[qid])
            counts[task] += 1
    if any(counts[t] != per_task for t in TASKS):
        raise ValueError(f"Insufficient matching historical predictions: {dict(counts)}")
    return selected, {"source_sha256": sources, "selection_counts": dict(counts),
                      "eligible_count": len(candidates), "selection": "first global IDs per task, legal raw output and exact question/answer/evidence match"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Physical A800 index")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrite a run")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/results/Qwen3-VL-8B")
    parser.add_argument("--per-task", type=int, default=3)
    args = parser.parse_args()
    if args.per_task < 1 or args.gpu < 0:
        parser.error("--per-task must be positive and --gpu nonnegative")
    # Guard the collaborator boundary even for user-supplied output paths.
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT / "sxz") or out.is_relative_to(ROOT / "data/qa"):
        parser.error("Outputs must not be inside sxz or data/qa")
    out.mkdir(parents=True, exist_ok=False)
    before = qa_hashes()
    write_json(out / "qa_before.json", before)
    print("Validating all frozen gold/PDF inputs", flush=True)
    gold, metadata = load_gold()
    records, selection = select_predictions(gold, args.source_dir, args.per_task)
    submission = out / "predictions.jsonl"
    submission.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    predictions = load_submission(submission, gold)
    write_json(out / "selection.json", selection)
    print("Hashing checkpoint artifacts (including every weight shard)", flush=True)
    model_dir = args.model_dir.resolve()
    artifacts = {p.name: file_hash(p) for p in sorted(model_dir.iterdir())
                 if p.is_file() and p.suffix in {".json", ".safetensors", ".jinja", ".txt"}}
    write_json(out / "checkpoint_artifacts.json", artifacts)
    # CUDA ordinal order can differ from nvidia-smi's physical indices. Bind the
    # UUID so the loaded device is exactly the device checked by admission.
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
        "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not gpu_uuid.startswith("GPU-") or "\n" in gpu_uuid:
        raise ValueError("Could not identify exactly one physical GPU UUID")
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, GPU_REQUIRE_NAME="A800",
                      GPU_ALLOW_SHARED="0", GPU_REPOSITORY_LOCK="1", GPU_COORDINATION_AUTO="1",
                      GPU_WAIT_TIMEOUT_SECONDS="60", GPU_WAIT_POLL_SECONDS="5")
    from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation
    import torch
    import transformers
    import numpy as np
    from transformers import AutoModelForImageTextToText, AutoProcessor

    generation = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                  "max_new_tokens": 32}
    config = {"backend": "local_transformers_diagnostic", "max_attempts": 2,
              "purpose": "new smoke test; historical binary-judge configuration remains unresolved",
              "identity": {"model": "Qwen/Qwen3.6-27B", "checkpoint": "sha256:" + object_hash(artifacts),
                           "revision": None, "revision_note": "upstream commit unavailable; local bytes fully hashed",
                           "tokenizer_sha256": artifacts["tokenizer.json"],
                           "chat_template_sha256": artifacts["chat_template.jinja"],
                           "serving_version": transformers.__version__},
              "generation": generation, "seed": 42, "enable_thinking": False,
              "dtype": "bfloat16", "attention_implementation": "sdpa", "device_map": {"": "cuda:0"},
              "decode": {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
              "timeout_seconds": None, "timeout_note": "local synchronous generation, bounded by 32 new tokens; no wall-clock timeout",
              "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                          "cuda": torch.version.cuda, "python": sys.version, "physical_gpu": args.gpu,
                          "gpu_uuid": gpu_uuid}}
    calls = []
    cache = out / "judge_cache.jsonl"
    with managed_gpu_reservation("ScienceDoc local evaluation smoke test", gpu_ids=[str(args.gpu)]):
        # Avoid trainer_utils/set_seed: inference needs no optional PEFT trainer.
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        print("Loading local Qwen3.6-27B on one A800; no CPU/disk offload", flush=True)
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("CPU/disk offload is prohibited")
        config["model_default_generation_config"] = model.generation_config.to_dict()
        config["runtime"]["gpu_name"] = torch.cuda.get_device_name(0)
        write_json(out / "judge_config.json", config)

        def generate(prompt: str) -> str:
            text = processor.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda:0")
            with torch.inference_mode():
                tokens = model.generate(**inputs, **generation)
            raw = processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:], **config["decode"])[0]
            call = {"prompt_sha256": digest(prompt), "raw": raw,
                    "output_token_ids": tokens[0, inputs.input_ids.shape[1]:].tolist()}
            calls.append(call)
            with (out / "model_calls.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(call, ensure_ascii=False) + "\n")
                f.flush()
            print(f"Local judge call {len(calls)}: {raw!r}", flush=True)
            return raw

        judge = Judge(config, cache, generate)
        report = score(gold, predictions, judge)
        report.update(metadata)
        report.update(scope="partial-submission smoke diagnostic, not benchmark results",
                      prompt_sha256=PROMPT_HASH, judge_config_sha256=judge.config_hash,
                      prediction_file_sha256=file_hash(submission))
        write_json(out / "full_denominator_report.json", report)
        initial_calls = len(calls)
        replay = score(gold, predictions, Judge(config, cache))
        write_json(out / "cache_replay_report.json", replay)
        controls = []
        for task in ("General", "Reasoning", "Multi-Document"):
            g = next(gold[r["qa_id"]] for r in records if gold[r["qa_id"]].task == task)
            for label, answer, expected in (("reference_answer", g.answer, "CORRECT"),
                                             ("contradictory_control", "Purple elephants dance on the moon.", "INCORRECT")):
                p = validate_raw(g.qa_id, json.dumps({"answer_pre": answer, "evidence_pages": list(g.evidence_pages)}), g.page_count)
                decision = judge.decide(g, p)
                controls.append({"qa_id": g.qa_id, "control": label, "prediction": p.answer_pre,
                                 "expected": expected, "judge": decision})
        write_json(out / "controls.json", controls)
        peak_memory = torch.cuda.max_memory_allocated(0)
    after = qa_hashes()
    write_json(out / "qa_after.json", after)
    cache_hits = sum(bool(r["judge"] and r["judge"].get("cache_hit")) for r in replay["rows"])
    checks = {"qa_bytes_unchanged": before == after,
              "fixed_denominator": report["total"] == 2200 and report["missing"] == 2200 - len(records),
              "no_illegal_predictions": report["illegal"] == 0,
              "no_judge_failures": report["technical_failures"] == 0,
              "all_answerable_cache_hits": cache_hits == args.per_task * 3,
              "replay_metrics_identical": all(report[k] == replay[k] for k in ("answer_metrics", "evidence_metrics", "discipline_breakdown")),
              "all_controls_expected": all(c["judge"]["decision"] == c["expected"] for c in controls),
              "unanswerable_uses_exact_label_only": all(r["judge"]["method"] == "exact_canonical_refusal"
                  for r in report["rows"] if r["task"] == "Unanswerable" and r["prediction_status"] == "legal")}
    summary = {"timestamp": datetime.now(timezone.utc).isoformat(), "scope": "smoke test, not Table 2/3 reproduction",
               "checks": checks, "passed": all(checks.values()), "selected_predictions": len(records),
               "initial_model_calls": initial_calls, "total_model_calls": len(calls), "cache_replay_hits": cache_hits,
               "controls": len(controls), "peak_allocated_gpu_bytes": peak_memory,
               "qa_files_checked": len(before), "prompt_sha256": PROMPT_HASH, "judge_config_sha256": judge.config_hash,
               "dataset_hashes": metadata["dataset_hashes"], "selection": selection}
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
