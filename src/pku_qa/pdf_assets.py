"""Resolve PDFs from the repository's canonical flat asset directory."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_ROOT = PROJECT_ROOT / "data/pdfs"
PDF_ASSET_MANIFEST = PROJECT_ROOT / "data/pdf_assets_manifest.json"


@lru_cache(maxsize=1)
def load_pdf_asset_manifest() -> dict[str, Any]:
    if not PDF_ASSET_MANIFEST.is_file():
        return {"assets": []}
    payload = json.loads(PDF_ASSET_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise ValueError(f"Invalid PDF asset manifest: {PDF_ASSET_MANIFEST}")
    return payload


def _under_project_pdf_root(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return resolved == PDF_ROOT.resolve() or PDF_ROOT.resolve() in resolved.parents


def _asset_candidates(
    legacy_id: str, pdf_dirs: Iterable[str | Path]
) -> list[dict[str, Any]]:
    directories = [Path(value) for value in pdf_dirs]
    if not any(_under_project_pdf_root(path) for path in directories):
        return []
    matches = []
    for asset in load_pdf_asset_manifest().get("assets", []):
        aliases = {str(value) for value in asset.get("legacy_ids", [])}
        if legacy_id not in aliases and legacy_id != asset.get("pdf_id"):
            continue
        legacy_paths = [str(Path(value)) for value in asset.get("legacy_paths", [])]
        expected_paths = []
        for directory in directories:
            expected = directory / f"{legacy_id}.pdf"
            try:
                expected = expected.resolve(strict=False).relative_to(PROJECT_ROOT)
            except ValueError:
                pass
            expected_paths.append(str(expected))
        if any(
            any(path.endswith(expected) for path in legacy_paths)
            for expected in expected_paths
        ):
            return [asset]
        matches.append(asset)
    return matches


def resolve_pdf_path(
    paper_id: str, pdf_dirs: Iterable[str | Path], *, required: bool = True
) -> Path | None:
    """Resolve a current PDF ID or a unique legacy ID.

    Direct paths are checked first so temporary test corpora and external PDF
    directories remain usable. Legacy IDs are then resolved through the
    migration manifest only when a caller searches inside ``data/pdfs``.
    """
    value = str(paper_id)
    directories = [Path(raw) for raw in pdf_dirs]
    for directory in directories:
        direct = directory / f"{value}.pdf"
        if direct.is_file():
            return direct

    nested = sorted(
        path
        for directory in directories
        for path in directory.glob(f"*/{value}.pdf")
        if path.is_file()
    )
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        raise ValueError(
            f"Ambiguous PDF ID {value!r}: {[str(path) for path in nested]}"
        )

    matches = _asset_candidates(value, directories)
    if len(matches) == 1:
        path = PDF_ROOT / str(matches[0]["filename"])
        if path.is_file():
            return path
    if len(matches) > 1:
        names = sorted(str(row["filename"]) for row in matches)
        raise ValueError(f"Ambiguous legacy PDF ID {value!r}: {names}")
    for directory in directories:
        conventional = [directory / f"z_cross_{value}.pdf"]
        if value.isdigit():
            conventional.insert(0, directory / f"paper_{int(value):04d}.pdf")
        conventional.extend(sorted(directory.glob(f"source_{value}_*.pdf")))
        existing = [path for path in conventional if path.is_file()]
        if len(existing) == 1:
            return existing[0]
    if required:
        raise FileNotFoundError(
            f"PDF for paper {value!r} was not found in "
            f"{[str(path) for path in directories]}"
        )
    return None


def resolve_pdf_id(legacy_id: str, *, legacy_dir: str | Path | None = None) -> str:
    """Return the canonical file stem for a legacy paper or bundle ID."""
    directories = [legacy_dir or PDF_ROOT]
    matches = _asset_candidates(str(legacy_id), directories)
    if len(matches) != 1:
        names = sorted(str(row.get("filename")) for row in matches)
        raise ValueError(f"Expected one PDF mapping for {legacy_id!r}, found {names}")
    return str(matches[0]["pdf_id"])


def resolve_source_pdf_path(paper_id: str, pdf_dir: str | Path = PDF_ROOT) -> Path:
    """Resolve one source-paper PDF whose legacy filename starts with its ID."""
    directory = Path(pdf_dir)
    direct = sorted(directory.glob(f"source_{paper_id}_*.pdf"))
    if len(direct) == 1:
        return direct[0]
    matches = _asset_candidates(str(paper_id), [directory])
    source_matches = [row for row in matches if row.get("kind") == "source_paper"]
    if len(source_matches) != 1:
        raise ValueError(
            f"Expected one source PDF for {paper_id}, found {len(source_matches)}"
        )
    return PDF_ROOT / str(source_matches[0]["filename"])


def output_pdf_path(
    pdf_dir: str | Path, legacy_id: str, *, kind: str = "cross_pdf"
) -> Path:
    """Return an output path that follows the flat naming convention."""
    directory = Path(pdf_dir)
    if directory.resolve(strict=False) != PDF_ROOT.resolve():
        return directory / f"{legacy_id}.pdf"
    prefixes = {
        "single_paper": "paper_",
        "source_paper": "source_",
        "cross_pdf": "z_cross_",
    }
    try:
        prefix = prefixes[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown PDF asset kind: {kind}") from exc
    value = f"{int(legacy_id):04d}" if kind == "single_paper" else legacy_id
    return directory / f"{prefix}{value}.pdf"
