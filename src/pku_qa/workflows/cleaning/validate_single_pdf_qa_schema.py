#!/usr/bin/env python3
"""Validate the enriched single-PDF QA dataset against its JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=ROOT
        / "data/qa/1.base/rel__single_pdf__ordinary__batch01__n1000.json",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=ROOT / "schemas/single_pdf_qa.schema.json",
    )
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(dataset),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    payload = {
        "dataset": str(args.dataset),
        "schema": str(args.schema),
        "errors": len(errors),
        "first_errors": [
            {
                "path": "/".join(str(part) for part in error.absolute_path),
                "message": error.message,
            }
            for error in errors[:20]
        ],
        "passed": not errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
