#!/usr/bin/env python3
"""Flatten and deduplicate every PDF under data/pdfs with stable names."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PDF_ROOT = ROOT / "data/pdfs"
MANIFEST = ROOT / "data/pdf_assets_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def classify(relative: Path) -> str:
    parts = relative.parts
    if parts and parts[0] == "papers":
        return "single_paper"
    if len(parts) >= 2 and parts[:2] in {
        ("qa_expansion", "cross_paper_sources"),
        ("qa_expansion", "cross_pdf_revalidated_400_sources"),
    }:
        return "source_paper"
    if len(parts) == 1:
        stem = relative.stem
        if stem.startswith("paper_"):
            return "single_paper"
        if stem.startswith("source_"):
            return "source_paper"
        if stem.startswith("z_cross_"):
            return "cross_pdf"
    return "cross_pdf"


def preferred_path(paths: list[Path]) -> Path:
    ranks = {
        "single_paper": 0,
        "source_paper": 1,
        "cross_pdf": 2,
    }
    return min(paths, key=lambda path: (ranks[classify(path)], natural_key(str(path))))


def proposed_stem(path: Path) -> str:
    kind = classify(path)
    if kind == "single_paper":
        match = re.fullmatch(r"(?:paper_)?(\d+)", path.stem)
        if not match:
            raise ValueError(f"Unexpected single-paper filename: {path}")
        return f"paper_{int(match.group(1)):04d}"
    if kind == "source_paper":
        stem = path.stem.removeprefix("source_")
        return f"source_{stem}"
    return f"z_cross_{path.stem.removeprefix('z_cross_')}"


def build_manifest() -> dict[str, Any]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(PDF_ROOT.rglob("*.pdf"), key=lambda item: natural_key(str(item))):
        by_hash[sha256_file(path)].append(path.relative_to(PDF_ROOT))

    preliminary = []
    for digest, paths in by_hash.items():
        preferred = preferred_path(paths)
        preliminary.append(
            {
                "sha256": digest,
                "kind": classify(preferred),
                "preferred": preferred,
                "legacy_paths": paths,
                "stem": proposed_stem(preferred),
            }
        )

    by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preliminary:
        by_stem[row["stem"]].append(row)
    for stem, rows in by_stem.items():
        if len(rows) == 1:
            continue
        for row in rows:
            parent = row["preferred"].parent.name
            qualifier = parent.removeprefix("hard_cross_pdf_global_")
            row["stem"] = f"{stem}__{qualifier}"

    assets = []
    for row in sorted(
        preliminary,
        key=lambda item: (
            {"single_paper": 0, "source_paper": 1, "cross_pdf": 2}[item["kind"]],
            natural_key(item["stem"]),
        ),
    ):
        legacy_paths = [str(Path("data/pdfs") / path) for path in row["legacy_paths"]]
        assets.append(
            {
                "pdf_id": row["stem"],
                "filename": f"{row['stem']}.pdf",
                "kind": row["kind"],
                "sha256": row["sha256"],
                "legacy_ids": sorted(
                    {path.stem.removeprefix("z_cross_").removeprefix("source_") for path in row["legacy_paths"]},
                    key=natural_key,
                ),
                "legacy_paths": sorted(legacy_paths, key=natural_key),
            }
        )
    names = [row["filename"] for row in assets]
    if len(names) != len(set(names)):
        raise ValueError("Canonical PDF filenames are not unique")
    return {
        "schema_version": 1,
        "layout": "flat",
        "pdf_root": "data/pdfs",
        "ordering": ["single_paper", "source_paper", "cross_pdf"],
        "source_pdf_count": sum(len(row["legacy_paths"]) for row in assets),
        "asset_count": len(assets),
        "duplicate_copy_count": sum(len(row["legacy_paths"]) - 1 for row in assets),
        "assets": assets,
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def migrate(manifest: dict[str, Any]) -> None:
    for asset in manifest["assets"]:
        destination = PDF_ROOT / asset["filename"]
        sources = [ROOT / value for value in asset["legacy_paths"]]
        existing_sources = [path for path in sources if path.is_file()]
        if destination.is_file() and destination not in existing_sources:
            existing_sources.append(destination)
        if not existing_sources:
            raise FileNotFoundError(f"No source remains for {destination}")
        if any(sha256_file(path) != asset["sha256"] for path in existing_sources):
            raise ValueError(f"PDF changed during migration: {destination}")
        source = destination if destination in existing_sources else existing_sources[0]
        if source != destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
        for duplicate in existing_sources:
            if duplicate != destination and duplicate.is_file():
                duplicate.unlink()

    for directory, _, _ in os.walk(PDF_ROOT, topdown=False):
        path = Path(directory)
        if path != PDF_ROOT:
            try:
                path.rmdir()
            except OSError:
                pass


def rewrite_json_pdf_paths(manifest: dict[str, Any]) -> int:
    exact: dict[str, str] = {}
    basename: dict[str, set[str]] = defaultdict(set)
    for asset in manifest["assets"]:
        canonical = f"data/pdfs/{asset['filename']}"
        for old in asset["legacy_paths"]:
            exact[old] = canonical
            exact[str((ROOT / old).resolve())] = str((ROOT / canonical).resolve())
            basename[Path(old).name].add(canonical)

    legacy_dirs = {
        str(Path(old).parent)
        for asset in manifest["assets"]
        for old in asset["legacy_paths"]
    }
    legacy_dirs.update(
        str((ROOT / value).resolve())
        for value in tuple(legacy_dirs)
        if not Path(value).is_absolute()
    )

    def replace(value: Any) -> tuple[Any, bool]:
        if isinstance(value, list):
            rows = [replace(item) for item in value]
            return [item for item, _ in rows], any(changed for _, changed in rows)
        if isinstance(value, dict):
            rows = {key: replace(item) for key, item in value.items()}
            return {key: item for key, (item, _) in rows.items()}, any(
                changed for _, changed in rows.values()
            )
        if not isinstance(value, str):
            return value, False
        normalized = value.replace("qa_expansion_20260711", "qa_expansion")
        if normalized in exact:
            return exact[normalized], True
        if normalized in legacy_dirs:
            replacement = str(PDF_ROOT.resolve()) if normalized.startswith("/") else "data/pdfs"
            return replacement, replacement != value
        if normalized.casefold().endswith(".pdf"):
            candidates = basename.get(Path(normalized).name, set())
            if len(candidates) == 1:
                replacement = next(iter(candidates))
                if value.startswith("/"):
                    replacement = str((ROOT / replacement).resolve())
                return replacement, replacement != value
        return value, False

    changed_files = 0
    for path in sorted((ROOT / "data").rglob("*.json")):
        if "archive" in path.parts or path == MANIFEST:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        updated, changed = replace(payload)
        if changed:
            atomic_json(path, updated)
            changed_files += 1
    return changed_files


def check(manifest: dict[str, Any]) -> None:
    nested = [path for path in PDF_ROOT.rglob("*.pdf") if path.parent != PDF_ROOT]
    if nested:
        raise ValueError(f"Nested PDFs remain: {nested[:5]}")
    actual = sorted(path.name for path in PDF_ROOT.glob("*.pdf"))
    expected = sorted(row["filename"] for row in manifest["assets"])
    if actual != expected:
        raise ValueError("Flat PDF directory does not match its manifest")
    for asset in manifest["assets"]:
        path = PDF_ROOT / asset["filename"]
        if sha256_file(path) != asset["sha256"]:
            raise ValueError(f"PDF hash mismatch: {path}")


def main() -> int:
    args = parse_args()
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        changed_files = 0 if args.check else rewrite_json_pdf_paths(manifest)
    else:
        if args.check:
            raise FileNotFoundError(f"Missing PDF manifest: {MANIFEST}")
        manifest = build_manifest()
        migrate(manifest)
        atomic_json(MANIFEST, manifest)
        changed_files = rewrite_json_pdf_paths(manifest)
    check(manifest)
    print(
        json.dumps(
            {
                "status": "healthy",
                "asset_count": manifest["asset_count"],
                "duplicate_copy_count": manifest["duplicate_copy_count"],
                "rewritten_json_files": changed_files,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
