#!/usr/bin/env python3
"""Run the bounded feasibility audit's semantic jobs with the paper local Judge."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from evaluation.check_reproduction import answer_bounds, compare_bound
from evaluation.judge import Judge, object_hash
from evaluation.metrics import score
from evaluation.prompts import JUDGE_PROMPT, PROMPT_HASH
from evaluation.reproduce import MODELS, load_baseline, write_csv
from evaluation.validation import Gold, Prediction, digest, file_hash, load_gold
from scripts.smoke_local_evaluation import qa_hashes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data/results")
    args = parser.parse_args()
    out = args.audit_dir.resolve()
    if out.is_relative_to(ROOT / "sxz") or out.is_relative_to(ROOT / "data/qa"):
        parser.error("Read-only input directory")
    cache = out / "local_judge_cache.jsonl"
    if cache.exists():
        parser.error("This run already has a cache; preserve it and select a fresh audit directory")
    before = qa_hashes()
    jobs = json.loads((out / "local_jobs.json").read_text())
    audit = json.loads((out / "feasibility.json").read_text())
    # Inputs, prompt and question/reference bindings must still match the audit.
    gold, metadata = load_gold()
    assert metadata["dataset_hashes"] == audit["dataset_hashes"]
    for source, expected in audit["source_hashes"].items():
        assert file_hash(args.results_dir / source) == expected
    for job in jobs:
        assert asdict(gold[job["gold"]["qa_id"]]) == {**job["gold"], "evidence_pages": tuple(job["gold"]["evidence_pages"])}
        assert job["prompt"] == JUDGE_PROMPT.format(question=job["gold"]["question"], correct=job["gold"]["answer"], model_answer=job["prediction"]["answer_pre"])
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu_uuid, GPU_REQUIRE_NAME="A800", GPU_ALLOW_SHARED="0",
                      GPU_REPOSITORY_LOCK="1", GPU_COORDINATION_AUTO="1", GPU_WAIT_TIMEOUT_SECONDS="60")
    print("Hashing local checkpoint", flush=True)
    artifacts = {p.name: file_hash(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()}
    write_json(out / "checkpoint_artifacts.json", artifacts)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    generation = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=32)
    config = {"identity": {"model": "Qwen/Qwen3.6-27B", "checkpoint": "sha256:" + object_hash(artifacts),
                           "revision": None, "tokenizer_sha256": artifacts["tokenizer.json"],
                           "chat_template_sha256": artifacts["chat_template.jinja"]},
              "backend": "local_transformers", "generation": generation, "seed": 42, "max_attempts": 2,
              "enable_thinking": False, "batch_size": len(jobs), "padding_side": "left", "attention": "sdpa",
              "dtype": "bfloat16", "timeout_seconds": None, "retry": "invalid batch replies retried individually",
              "purpose": "new bounded discrepancy verification; historical binary judge configuration unresolved",
              "torch": torch.__version__, "transformers": transformers.__version__, "gpu_uuid": gpu_uuid,
              "decode": {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}}
    with managed_gpu_reservation("paper discrepancy verification", gpu_ids=[str(args.gpu)]):
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        assert all(p.device.type == "cuda" for p in model.parameters())
        config["model_default_generation_config"] = model.generation_config.to_dict()
        write_json(out / "local_judge_config.json", config)

        def batch(prompts):
            texts = [processor.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                     add_generation_prompt=True, enable_thinking=False) for p in prompts]
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                ids = model.generate(**inputs, **generation)[:, inputs.input_ids.shape[1]:]
            replies = processor.batch_decode(ids, **config["decode"])
            with (out / "local_raw_calls.jsonl").open("a", encoding="utf-8") as f:
                for prompt, raw, tokens in zip(prompts, replies, ids.tolist()):
                    f.write(json.dumps({"prompt_sha256": digest(prompt), "raw": raw, "token_ids": tokens}) + "\n")
                    print(repr(raw), flush=True)
            return replies

        first = {job["prompt"]: raw for job, raw in zip(jobs, batch([j["prompt"] for j in jobs]))}

        def generate(prompt):
            return first.pop(prompt) if prompt in first else batch([prompt])[0]

        judge = Judge(config, cache, generate)
        decisions = [{"model": j["model"], "qa_id": j["gold"]["qa_id"],
                      "judge": judge.decide(gold[j["gold"]["qa_id"]], Prediction(**j["prediction"]))} for j in jobs]
        write_json(out / "local_decisions.json", decisions)
    paper = json.loads((ROOT / "evaluation/paper_reference.json").read_text())
    table2, table3, completed = [], [], {}
    for directory, name in MODELS.items():
        if not audit["models"][name]["complete_model_selected_for_local_judge"]:
            continue
        predictions, _, _, _ = load_baseline(args.results_dir, gold, directory)
        report = score(gold, predictions, Judge(config, cache))
        report.update(scope="current-release strict diagnostic, historical input defects retained", **metadata)
        write_json(out / (directory + "_locally_scored.json"), report)
        completed[name] = {k: report[k] for k in ("total", "missing", "illegal", "technical_failures", "answer_metrics")}
        for key, value in paper["table2"][name].items():
            if key not in report["answer_metrics"]:
                continue
            rows = report["rows"] if key == "All" else [r for r in report["rows"] if r["task"] == key]
            table2.append(compare_bound(name, key, value, answer_bounds(rows)))
        for key, value in paper["table3"][name].items():
            rows = report["rows"] if key == "All" else [r for r in report["rows"] if r["discipline"] == key]
            table3.append(compare_bound(name, key, value, answer_bounds(rows)))
    write_csv(out / "table2_local.csv", table2)
    write_csv(out / "table3_local.csv", table3)
    after = qa_hashes()
    assert before == after
    write_json(out / "qa_hashes.json", after)
    write_json(out / "local_summary.json", {"models": completed, "real_judge_items": len(jobs),
        "judge_failures": sum(d["judge"]["decision"] is None for d in decisions),
        "qa_files_unchanged": len(before), "prompt_sha256": PROMPT_HASH,
        "official_reproduction_verified": False, "result": "mismatch confirmed; not 11-model full rescore"})
    print("Local verification finished; tables differ. GPU released.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
