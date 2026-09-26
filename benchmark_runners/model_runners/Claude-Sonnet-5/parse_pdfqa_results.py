#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parents[1]
sys.path.insert(0,str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None

from settings.pdfqa_prompts import PROMPT_VERSION
import run_pdfqa as r

RAW=HERE/"output"/"raw_result"
PARSED=HERE/"output"/"parsed_result"
MANIFEST=HERE/"output"/"manifests"

def atomic_write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno()); tmp=f.name
    os.replace(tmp,path)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(1024*1024)
            if not b:break
            h.update(b)
    return h.hexdigest()

def name(ds): return f"{ds}__{PROMPT_VERSION}__dpi144__run-v1.json"

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset-id",required=True,choices=list(r.DATASETS));args=p.parse_args()
    ds=args.dataset_id; spec=r.DATASETS[ds]; src=RAW/name(ds)
    if not src.is_file():raise FileNotFoundError(src)
    data=json.loads(src.read_text(encoding="utf-8")); counts=Counter()
    for _,_,_,qa in r.iter_qa(data):
        status=qa.get("status")
        if status=="error":
            counts["technical"]+=1;continue
        raw=qa.get("answer_pre_raw")
        if raw is None or str(raw).strip()=="":
            if r.is_attempted(qa): counts["failed"]+=1
            else: counts["not_run"]+=1
            continue
        ok,a,pages,reason=r.parse_prediction(str(raw),qa.get("pdf_total_pages"))
        qa["posthoc_parse_ok"]=bool(ok);qa["posthoc_parse_reason"]=reason;qa["format_repair_used"]=False
        if ok:
            qa["answer_pre"]=a;qa["evidence_pages_pre"]=pages
            if reason != "strict_ok":
                qa["status"] = "normalized_completed"
                counts["normalized_strict"] += 1
            else:
                qa["status"] = "completed"
                counts["strict"] += 1
        else:
            qa["answer_pre"]="";qa["evidence_pages_pre"]=[];qa["status"]="parse_failed";counts["failed"]+=1
    out=PARSED/name(ds);atomic_write(out,data)
    manifest={
        "model_name":"Claude-Sonnet-5",
        "model_id":os.getenv("CLAUDE_MODEL","claude-sonnet-5"),
        "inference_backend":"anthropic_compatible_messages_api",
        "base_url":os.getenv("CLAUDE_BASE_URL",r.DEFAULT_BASE_URL),
        "dataset_id":ds,
        "dataset_path":str(spec["data"]),
        "dataset_sha256":sha256_file(spec["data"]),
        "expected_qa_count":spec["expected"],
        "prompt_id":spec["prompt_id"],
        "prompt_version":PROMPT_VERSION,
        "pdf_dpi":144,
        "whole_pdf":True,
        "page_dropping":False,
        "gold_answer_sent":False,
        "gold_evidence_sent":False,
        "format_repair":False,
        "semantic_repair":False,
        "thinking_mode":"disabled" if os.getenv("CLAUDE_DISABLE_THINKING","0")=="1" else "provider/model default",
        "compression_policy":spec["policy"],
        "max_output_tokens":spec["max_tokens"],
        "request_timeout":1800.0,
        "result_counts":dict(counts),
    }
    m=(MANIFEST / name(ds)).with_suffix(".manifest.json");atomic_write(m,manifest)
    print(f"[Parse] {ds} | {dict(counts)}");print(f"[Saved] {out}");print(f"[Manifest] {m}")

if __name__=="__main__":main()
