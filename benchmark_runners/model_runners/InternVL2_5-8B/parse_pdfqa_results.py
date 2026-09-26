#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

SXZ_ROOT = Path(__file__).resolve().parents[2]
if str(SXZ_ROOT) not in sys.path:
    sys.path.insert(0, str(SXZ_ROOT))

from settings.pdfqa_benchmark_config import DATASETS, PDF_DPI  # noqa: E402
from settings.pdfqa_prompts import PROMPT_VERSION  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent
RAW_DIR = MODEL_DIR / "output" / "raw_result"
PARSED_DIR = MODEL_DIR / "output" / "parsed_result"
RUN_VERSION = "run-v1"
SPECIAL_TOKENS = ("<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|assistant|>", "<|user|>")


def basename(dataset_id: str) -> str:
    return (
        f"{dataset_id}"
        f"__{PROMPT_VERSION}"
        f"__dpi{PDF_DPI}"
        f"__{RUN_VERSION}.json"
    )


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno()); tmp = Path(f.name)
    os.replace(tmp, path)


def clean_text(raw: str) -> str:
    text = str(raw or "").strip()
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text.strip()


def normalize_pages(raw: Any) -> List[int]:
    if raw is None or isinstance(raw, bool):
        return []
    items = raw if isinstance(raw, list) else [raw]
    out: List[int] = []
    for item in items:
        vals = [item] if isinstance(item, int) and not isinstance(item, bool) else re.findall(r"\d+", str(item))
        for value in vals:
            try: iv = int(value)
            except Exception: continue
            if iv > 0 and iv not in out: out.append(iv)
    return sorted(out)


def escape_invalid_json_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def parse_prediction(raw: str) -> tuple[Optional[Dict[str, Any]], str]:
    text = clean_text(raw)
    if not text: return None, "empty"
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fenced: text = fenced.group(1).strip()
    candidates = [text]
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s: candidates.append(text[s:e+1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict): return obj, "json"
        except Exception: pass
    for candidate in candidates:
        try:
            obj = json.loads(escape_invalid_json_backslashes(candidate))
            if isinstance(obj, dict): return obj, "json_backslash_repair"
        except Exception: pass
    for candidate in candidates:
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict): return obj, "literal_eval"
        except Exception: pass
    return None, "parse_failed"


def main() -> None:
    p = argparse.ArgumentParser(description="Parse InternVL2.5-8B raw PDF-QA outputs.")
    p.add_argument("--dataset-id", required=True, choices=list(DATASETS))
    args = p.parse_args()
    raw_path = RAW_DIR / basename(args.dataset_id)
    out_path = PARSED_DIR / basename(args.dataset_id)
    if not raw_path.is_file(): raise FileNotFoundError(raw_path)
    with raw_path.open("r", encoding="utf-8") as f: data = json.load(f)
    out = copy.deepcopy(data); counts: Dict[str, int] = {}
    for unit in out.values():
        if not isinstance(unit, dict) or not isinstance(unit.get("QA"), dict): continue
        for qa in unit["QA"].values():
            if not isinstance(qa, dict): continue
            raw = qa.get("answer_pre_raw", "")
            obj, mode = parse_prediction(raw)
            status = "ok"
            if obj is None:
                answer = ""; pages: List[int] = []; status = mode
            else:
                answer = str(obj.get("answer_pre", "")).strip()
                pages = normalize_pages(obj.get("evidence_pages"))
                if "answer_pre" not in obj or "evidence_pages" not in obj: status = "missing_required_key"
                elif not answer: status = "empty_answer"
            pdf_total = qa.get("pdf_total_pages")
            invalid = [p for p in pages if isinstance(pdf_total, int) and pdf_total > 0 and p > pdf_total]
            if invalid and status == "ok": status = "page_out_of_range"
            qa["answer_pre"] = answer
            qa["evidence_pages_pre"] = pages
            qa["prediction_parse_mode"] = mode
            qa["prediction_parse_status"] = status
            qa["predicted_page_issues"] = [f"out_of_range:{p}" for p in invalid]
            qa["parsed_valid"] = status == "ok"
            counts[status] = counts.get(status, 0) + 1
    atomic_write_json(out_path, out)
    print(f"[Parsed] {args.dataset_id} -> {out_path}")
    print(f"[Status] {counts}")


if __name__ == "__main__":
    main()
