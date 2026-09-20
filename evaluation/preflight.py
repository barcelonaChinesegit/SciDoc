#!/usr/bin/env python3
"""Read-only release validation and Google Drive upload inventories."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.validation import COMPONENTS, ROOT, file_hash, load_gold


def check_public_files() -> dict:
    """Conservative token/private-key scan; locations only, never print matches."""
    paths = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    patterns = [re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"), re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
                re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
    findings, checked = [], 0
    for relative in sorted(set(paths)):
        if not relative:
            continue
        path = ROOT / relative
        if not path.is_file() or path.suffix in {".png", ".jpg", ".xlsx", ".pdf"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        for number, line in enumerate(content.splitlines(), 1):
            if any(pattern.search(line) for pattern in patterns):
                findings.append({"path": relative, "line": number})
    if findings:
        raise ValueError("Potential credential material in public files (values hidden): " + json.dumps(findings))
    example = ROOT / "examples/predictions.example.jsonl"
    # The example consists solely of public QA content; schema is checked by the CLI/tests.
    return {"candidate_text_files_scanned": checked, "credential_pattern_findings": 0,
            "example_sha256": file_hash(example), "scope": "Git-tracked and nonignored candidate text files; pattern scan, not a proof of absence"}


def preflight(pdf_dir: Path | None = None) -> dict:
    gold, metadata = load_gold(pdf_dir=pdf_dir)
    duplicates = Counter((g.question, g.answer, g.pdf_id) for g in gold.values())
    fields = {}
    for filename, _, _ in COMPONENTS:
        data = json.loads((ROOT / "data/qa/7.final_2200" / filename).read_text(encoding="utf-8"))
        fields[filename] = {"paper_fields": dict(Counter(k for p in data.values() for k in p if k != "QA")),
                            "qa_fields": dict(Counter(k for p in data.values() for q in p["QA"].values() for k in q))}
    return {**metadata, "status": "valid", "public_file_checks": check_public_files(), "components": dict(Counter(g.task for g in gold.values())),
            "continuous_unique_ids": True, "duplicate_question_answer_pdf_records": sum(n - 1 for n in duplicates.values()),
            "gold_evidence_positive_integer_and_in_range": True, "canonical_unanswerable_count": 200,
            "pdf_count": len(metadata["pdf_hashes"]), "pdf_readability_and_hashes": "valid",
            "discipline_counts": dict(sorted(Counter(g.discipline for g in gold.values()).items())),
            "fine_grained_field_count": len({g.field for g in gold.values()}), "actual_field_inventory": fields}


def pdf_inventory(output_dir: Path, pdf_dir: Path | None = None) -> dict:
    assets = json.loads((ROOT / "data/pdf_assets_manifest.json").read_text(encoding="utf-8"))["assets"]
    needed = set()
    for filename, _, _ in COMPONENTS:
        needed.update(json.loads((ROOT / "data/qa/7.final_2200" / filename).read_text(encoding="utf-8")))
    rows = []
    for asset in assets:
        path = (pdf_dir or ROOT / "data/pdfs") / asset["filename"]
        if not path.is_file() or file_hash(path) != asset["sha256"]:
            raise ValueError(f"PDF inventory mismatch: {asset['filename']}")
        rows.append({"relative_path": "data/pdfs/" + asset["filename"], "pdf_id": asset["pdf_id"],
                     "kind": asset["kind"], "bytes": path.stat().st_size, "sha256": asset["sha256"],
                     "required_for_benchmark": asset["pdf_id"] in needed})
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("pdf_upload_all.csv", rows),
                         ("pdf_upload_benchmark.csv", [r for r in rows if r["required_for_benchmark"]])):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(subset)
    return {"all_pdf_count": len(rows), "all_pdf_bytes": sum(r["bytes"] for r in rows),
            "benchmark_pdf_count": len(needed),
            "benchmark_pdf_bytes": sum(r["bytes"] for r in rows if r["required_for_benchmark"]),
            "kinds": dict(Counter(r["kind"] for r in rows))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--pdf-inventory-dir", type=Path)
    args = parser.parse_args()
    result = preflight(args.pdf_dir)
    if args.pdf_inventory_dir:
        result["pdf_upload"] = pdf_inventory(args.pdf_inventory_dir, args.pdf_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"pdf_hashes", "pdf_page_counts", "actual_field_inventory"}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
