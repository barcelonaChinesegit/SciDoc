#!/usr/bin/env python3
"""Paired input-version sensitivity check with the unchanged paper Judge prompt.

Historical text is read from verified cache records. These pairs are diagnostic
inputs, not a benchmark release or alternate scoring mode. No baseline cache,
QA file, prediction or score is replaced by this experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from evaluation.judge import object_hash
from evaluation.prompts import JUDGE_PROMPT, PROMPT_HASH
from evaluation.validation import file_hash
from evaluation.audit_score_provenance import snapshot_qa


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--rescore-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT / "sxz") or out.is_relative_to(ROOT / "data/qa"):
        parser.error("Read-only output directory")
    out.mkdir(parents=True, exist_ok=False)
    before = snapshot_qa()
    jobs = json.loads((args.audit_dir / "paired_jobs.json").read_text())
    config = json.loads((args.rescore_dir / "judge_config.json").read_text())
    artifacts = json.loads((args.rescore_dir / "checkpoint_artifacts.json").read_text())
    model_dir = ROOT / "models/Qwen3.6-27B"
    started = time.monotonic()
    print("Verifying checkpoint; testing both input versions with the paper prompt", flush=True)
    if any(file_hash(model_dir / name) != expected for name, expected in artifacts.items()):
        raise ValueError("Checkpoint differs from the current-gold diagnostic")
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                   "--format=csv,noheader"], text=True).strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, GPU_REQUIRE_NAME="A800", GPU_ALLOW_SHARED="0",
                      GPU_REPOSITORY_LOCK="1", GPU_COORDINATION_AUTO="1", GPU_WAIT_TIMEOUT_SECONDS="60")
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation
    if torch.__version__ != config["torch"] or transformers.__version__ != config["transformers"]:
        raise ValueError("Runtime changed")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    run_config = {**config, "gpu_uuid": uuid,
                  "purpose": "paired historical/current text diagnostic with paper prompt; no score replacement"}
    requests = []
    for job in jobs:
        old = job["historical_inputs"]
        for variant, question, reference, prediction in (
            ("current", job["gold"]["question"], job["gold"]["answer"], job["prediction"]["answer_pre"]),
            ("historical_text", old["question"], old["gold_answer"], old["pred_answer"]),
        ):
            requests.append({"qa_id": job["qa_id"], "variant": variant,
                "prompt": JUDGE_PROMPT.format(question=question, correct=reference, model_answer=prediction)})
    # Both variants have identical generation settings and no truncation.
    requests.sort(key=lambda r: (len(r["prompt"]), r["qa_id"], r["variant"]))
    replies = {}
    with managed_gpu_reservation("ScienceDoc paired Judge input-version check", gpu_ids=[str(args.gpu)]):
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        processor.tokenizer.padding_side = config["padding_side"]
        model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation=config["attention"]).eval()
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("CPU/disk offload is forbidden")
        if model.generation_config.to_dict() != config["model_default_generation_config"]:
            raise ValueError("Model generation defaults changed")
        (out / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

        def generate(group):
            texts = [processor.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=config["enable_thinking"])
                     for r in group]
            inputs = processor(text=texts, padding=True, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                tokens = model.generate(**inputs, **config["generation"])[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(tokens, **config["decode"])
            with (out / "raw_calls.jsonl").open("a", encoding="utf-8") as f:
                for request, text, ids in zip(group, raw, tokens.tolist()):
                    f.write(json.dumps({**request, "prompt_sha256": object_hash(request["prompt"]),
                        "raw": text, "token_ids": ids, "batch_size": len(group)}, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return raw

        for start in range(0, len(requests), config["batch_size"]):
            group = requests[start:start + config["batch_size"]]
            for request, raw in zip(group, generate(group)):
                attempts = [raw]
                while raw not in {"CORRECT", "INCORRECT"} and len(attempts) < config["max_attempts"]:
                    raw = generate([request])[0]
                    attempts.append(raw)
                replies[(request["qa_id"], request["variant"])] = {
                    "decision": raw if raw in {"CORRECT", "INCORRECT"} else None, "attempts": attempts}
            print(f"Completed {min(start + len(group), len(requests))}/{len(requests)} checks", flush=True)
    rows = []
    for job in jobs:
        current = replies[(job["qa_id"], "current")]
        historical = replies[(job["qa_id"], "historical_text")]
        rows.append({"qa_id": job["qa_id"], "differences": job["differences"],
            "historical_recorded_decision": job["historical_inputs"]["answer_verdict"],
            "previous_current_decision": job["current_decision"], "current": current,
            "historical_text_paper_prompt": historical,
            "current_repeat_agrees": current["decision"] == job["current_decision"],
            "pair_agrees": current["decision"] == historical["decision"]})
    summary = {"scope": "paired input-version sensitivity diagnostic; no official score or cache replacement",
        "rows": rows, "pairs": len(rows), "current_repeat_agrees": sum(r["current_repeat_agrees"] for r in rows),
        "pair_agrees": sum(r["pair_agrees"] for r in rows),
        "failures": sum(r["decision"] is None for r in replies.values()),
        "retries": sum(len(r["attempts"]) - 1 for r in replies.values()),
        "seconds": time.monotonic() - started, "prompt_sha256": PROMPT_HASH,
        "input_jobs_sha256": file_hash(args.audit_dir / "paired_jobs.json"),
        "config_sha256": object_hash(run_config), "qa_unchanged": before == snapshot_qa()}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
    return 0 if summary["failures"] == 0 and summary["qa_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
