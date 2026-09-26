from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from settings.pdfqa_benchmark_config import get_dataset_spec


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Dataset root must be a dict: {path}")
    return data


def count_qa(data: Dict[str, Any]) -> int:
    return sum(
        len(unit.get("QA", {}))
        for unit in data.values()
        if isinstance(unit, dict) and isinstance(unit.get("QA", {}), dict)
    )


def flatten_dataset(dataset_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    flat_index = 0
    for unit_id, unit in data.items():
        if not isinstance(unit, dict):
            continue
        qa_dict = unit.get("QA", {})
        if not isinstance(qa_dict, dict):
            continue
        for qa_id, qa in qa_dict.items():
            if not isinstance(qa, dict):
                continue
            flat_index += 1
            out.append(
                {
                    "dataset_id": dataset_id,
                    "flat_index": flat_index,
                    "unit_id": str(unit_id),
                    "qa_id": str(qa_id),
                    "item_uid": f"{dataset_id}::{unit_id}::{qa_id}",
                    "unit": unit,
                    "qa": qa,
                }
            )
    return out


def _candidate_paths(
    unit_id: str,
    unit: Dict[str, Any],
    pdf_root: Path,
) -> Iterable[Path]:
    """
    Generate possible PDF paths without assuming one filename convention.

    Supported examples:

        Historical:
            259 -> 259.pdf

        Current single-PDF corpus:
            259 -> paper_0259.pdf

        Cross-PDF / explicitly supplied paths:
            use pdf_path / paper_path / merged_pdf_path /
            pdf_path_resolved when present.

    Non-numeric IDs such as xb_0024 are left untouched and therefore
    remain compatible with the cross-document datasets.
    """

    # Dataset-provided explicit paths always have priority.
    for key in (
        "pdf_path",
        "paper_path",
        "merged_pdf_path",
        "pdf_path_resolved",
    ):
        value = unit.get(key)

        if not value:
            continue

        p = Path(str(value))

        if not p.is_absolute():
            p = pdf_root / p

        yield p

    uid = str(unit_id).strip()

    # ------------------------------------------------------------------
    # Historical convention:
    #     259 -> 259.pdf
    #     xb_0024 -> xb_0024.pdf
    # ------------------------------------------------------------------
    yield pdf_root / f"{uid}.pdf"

    # ------------------------------------------------------------------
    # Current single-paper corpus convention:
    #     0    -> paper_0000.pdf
    #     45   -> paper_0045.pdf
    #     259  -> paper_0259.pdf
    #     1000 -> paper_1000.pdf
    #
    # Only apply this to purely numeric IDs. Cross-document IDs such as
    # xb_0024 must NOT be converted.
    # ------------------------------------------------------------------
    try:
        numeric_id = int(uid)

        if 0 <= numeric_id <= 1000:
            yield pdf_root / f"paper_{numeric_id:04d}.pdf"

    except (TypeError, ValueError):
        pass


def resolve_pdf_path(
    dataset_id: str,
    unit_id: str,
    unit: Dict[str, Any],
) -> Path:
    """
    Resolve one benchmark unit to its actual PDF.

    Resolution order:

    1. Explicit path fields stored in the dataset;
    2. <unit_id>.pdf;
    3. paper_<4-digit-id>.pdf for numeric paper IDs;
    4. Recursive lookup using the same filename candidates.

    This keeps the old ordinary/cross-document behaviour while adding
    support for the current reasoning-PDF naming convention:
        259 -> paper_0259.pdf
    """

    spec = get_dataset_spec(dataset_id)
    pdf_root = Path(spec["pdf_root"])

    if not pdf_root.exists():
        raise FileNotFoundError(
            f"PDF root does not exist for {dataset_id}: {pdf_root}"
        )

    # ------------------------------------------------------------------
    # First try direct paths.
    # ------------------------------------------------------------------
    seen_candidates = set()

    for candidate in _candidate_paths(
        str(unit_id),
        unit,
        pdf_root,
    ):
        candidate = candidate.expanduser()

        key = str(candidate)
        if key in seen_candidates:
            continue

        seen_candidates.add(key)

        if candidate.is_file():
            return candidate.resolve()

    # ------------------------------------------------------------------
    # Recursive fallback.
    # ------------------------------------------------------------------
    uid = str(unit_id).strip()

    filename_patterns = [
        f"{uid}.pdf",
    ]

    try:
        numeric_id = int(uid)

        if 0 <= numeric_id <= 1000:
            filename_patterns.append(
                f"paper_{numeric_id:04d}.pdf"
            )

    except (TypeError, ValueError):
        pass

    matches = []
    seen_matches = set()

    for pattern in filename_patterns:
        for p in pdf_root.rglob(pattern):
            if not p.is_file():
                continue

            resolved = p.resolve()
            key = str(resolved)

            if key in seen_matches:
                continue

            seen_matches.add(key)
            matches.append(resolved)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple PDFs found for {dataset_id}/{unit_id}: "
            + " | ".join(str(p) for p in matches)
        )

    raise FileNotFoundError(
        f"PDF not found for {dataset_id}/{unit_id} "
        f"under {pdf_root}; tried filenames={filename_patterns}"
    )


def build_document_manifest(dataset_id: str, unit: Dict[str, Any]) -> str:
    spec = get_dataset_spec(dataset_id)
    if spec.get("document_manifest_mode") != "source_documents_full":
        return ""

    docs = unit.get("source_documents")
    if not isinstance(docs, list) or not docs:
        raise ValueError(
            f"{dataset_id}: expected non-gold source_documents manifest but none exists"
        )

    lines = ["Document map for the merged PDF:"]
    prev_end = 0
    for expected_doc, doc in enumerate(docs, start=1):
        if not isinstance(doc, dict):
            raise TypeError(f"Invalid source_documents entry: {doc!r}")
        dn = int(doc["doc_number"])
        start = int(doc["merged_start_page"])
        end = int(doc["merged_end_page"])
        if dn != expected_doc:
            raise ValueError(f"Expected Doc {expected_doc}, got {dn}")
        if start != prev_end + 1 or end < start:
            raise ValueError(
                f"Invalid merged page range for Doc {dn}: {start}-{end}, prev_end={prev_end}"
            )
        prev_end = end
        parts = [
            f"Doc {dn}",
            f"merged physical pages {start}-{end}",
        ]
        if doc.get("paper_id"):
            parts.append(f"source paper ID {doc['paper_id']}")
        if doc.get("title"):
            parts.append(f"title: {doc['title']}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)
