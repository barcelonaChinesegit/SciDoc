"""Strict PDF-mode submission validation. Never repair semantic content."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (("ordinary_qa.json", "General", 1000),
              ("unanswerable_qa.json", "Unanswerable", 200),
              ("reasoning_qa.json", "Reasoning", 200),
              ("cross_pdf_qa.json", "Multi-Document", 800))
MANIFEST = "rel__collection__final_2200__manifest.json"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def bad_constant(value):
        raise ValueError(f"non-JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_object, parse_constant=bad_constant)


@dataclass(frozen=True)
class Gold:
    qa_id: str
    question: str
    answer: str
    evidence_pages: tuple[int, ...]
    task: str
    discipline: str
    field: str
    pdf_id: str
    page_count: int
    legacy_paper_id: str = ""
    legacy_qa_id: str = ""


@dataclass
class Prediction:
    qa_id: str
    status: str
    answer_pre: str | None = None
    evidence_pages: list[int] | None = None
    errors: list[dict] = field(default_factory=list)
    audit: dict = field(default_factory=dict)


def load_gold(root: Path = ROOT, pdf_dir: Path | None = None) -> tuple[dict[str, Gold], dict]:
    """Use existing release/PDF manifests; verify the real physical PDF bounds."""
    from pypdf import PdfReader

    release = root / "data/qa/7.final_2200"
    manifest = strict_json((release / MANIFEST).read_text(encoding="utf-8"))
    assets_path = root / "data/pdf_assets_manifest.json"
    assets = strict_json(assets_path.read_text(encoding="utf-8"))["assets"]
    by_pdf = {a["pdf_id"]: a for a in assets}
    by_file = {a["dataset_id"]: a for a in manifest["components"]}
    pdf_dir = pdf_dir or root / "data/pdfs"
    pdf_counts, pdf_hashes, hashes, gold = {}, {}, {}, {}
    ordered_ids = []
    for filename, task, expected in COMPONENTS:
        path = release / filename
        hashes[filename] = file_hash(path)
        if hashes[filename] != by_file[filename]["sha256"]:
            raise ValueError(f"{filename}: release manifest SHA-256 mismatch")
        data = strict_json(path.read_text(encoding="utf-8"))
        count = 0
        for pdf_id, paper in data.items():
            if pdf_id not in pdf_counts:
                asset = by_pdf[pdf_id]
                pdf = pdf_dir / asset["filename"]
                if not pdf.is_file():
                    raise ValueError(f"Missing evaluation PDF: {pdf}")
                pdf_hashes[pdf_id] = file_hash(pdf)
                if pdf_hashes[pdf_id] != asset["sha256"]:
                    raise ValueError(f"{pdf.name}: PDF manifest SHA-256 mismatch")
                with pdf.open("rb") as f:
                    pdf_counts[pdf_id] = len(PdfReader(f).pages)
                if pdf_counts[pdf_id] <= 0:
                    raise ValueError(f"{pdf.name}: empty PDF")
            for qa_id, qa in paper["QA"].items():
                if qa_id in gold:
                    raise ValueError(f"Duplicate gold QA ID: {qa_id}")
                pages = qa["evidence_pages"]
                if not isinstance(pages, list) or any(type(p) is not int or not 0 < p <= pdf_counts[pdf_id] for p in pages):
                    raise ValueError(f"{qa_id}: invalid gold evidence bounds/types")
                if not isinstance(qa["question"], str) or not isinstance(qa["answer"], str):
                    raise ValueError(f"{qa_id}: gold question/answer must be strings")
                if not qa["question"].strip() or not qa["answer"].strip():
                    raise ValueError(f"{qa_id}: empty gold question/answer")
                if qa.get("question_type") not in {"Literal", "Inferential"} or not qa.get("question_category"):
                    raise ValueError(f"{qa_id}: invalid gold question classification")
                modalities = qa.get("modal_types")
                if not isinstance(modalities, list) or not modalities or any(m not in {"text", "image", "table", "formula"} for m in modalities):
                    raise ValueError(f"{qa_id}: invalid gold modalities")
                if task == "Unanswerable":
                    if qa["answer"] != "Unanswerable" or pages != []:
                        raise ValueError(f"{qa_id}: noncanonical gold Unanswerable")
                elif qa["answer"] == "Unanswerable" or not pages:
                    raise ValueError(f"{qa_id}: answerable gold contract violation")
                identity = qa["annotation_provenance"]["final_2200_identity"]
                if identity["global_qa_id"] != qa_id:
                    raise ValueError(f"{qa_id}: gold provenance identity mismatch")
                gold[qa_id] = Gold(qa_id, qa["question"], qa["answer"], tuple(pages), task,
                                   paper["primary_category"], paper["secondary_category"], pdf_id,
                                   pdf_counts[pdf_id], identity["legacy_paper_id"], identity["legacy_qa_id"])
                ordered_ids.append(qa_id)
                count += 1
        if count != expected:
            raise ValueError(f"{filename}: expected {expected}, found {count}")
    if ordered_ids != [f"QA{i:04d}" for i in range(1, 2201)]:
        raise ValueError("Gold IDs must be ordered, unique QA0001–QA2200")
    return gold, {"dataset_hashes": hashes, "release_manifest_sha256": file_hash(release / MANIFEST),
                  "pdf_manifest_sha256": file_hash(assets_path), "pdf_hashes": pdf_hashes,
                  "pdf_page_counts": pdf_counts, "total": len(gold)}


def validate_raw(qa_id: str, raw: str, page_count: int) -> Prediction:
    p = Prediction(qa_id, "legal", audit={"original_raw_output": raw,
        "raw_output_sha256": digest(raw), "correction_trigger": None, "retry_outputs": [],
        "deterministic_normalizations": [], "parse_status": "not_parsed",
        "format_status": "unchecked", "semantic_contract_status": "unchecked"})

    def error(kind, message):
        p.status = "illegal"
        p.errors.append({"type": kind, "message": message})

    try:
        obj = strict_json(raw)
    except ValueError as exc:
        error("invalid_json", str(exc))
        p.audit["parse_status"] = "invalid_json"
        p.audit["final_parse_status"] = "illegal"
        p.audit["final_normalized_output"] = None
        return p
    p.audit["parse_status"] = "parsed"
    if not isinstance(obj, dict):
        error("wrong_object_type", "raw output must be one JSON object")
        p.audit["final_parse_status"] = "illegal"
        p.audit["final_normalized_output"] = None
        return p
    if set(obj) != {"answer_pre", "evidence_pages"}:
        error("wrong_fields", "raw output must contain exactly answer_pre and evidence_pages")
    answer, pages = obj.get("answer_pre"), obj.get("evidence_pages")
    if not isinstance(answer, str):
        error("answer_type", "answer_pre must be a string")
    else:
        p.answer_pre = answer  # No trimming, casing, unit or numeric normalization.
        if not answer.strip():
            error("empty_answer", "answer_pre must contain a final answer")
    if not isinstance(pages, list):
        error("pages_type", "evidence_pages must be a JSON array")
    elif any(type(page) is not int for page in pages):
        error("page_type", "each evidence page must be an integer; bool/string/float are not allowed")
    else:
        p.evidence_pages = sorted(set(pages))
        if len(pages) != len(set(pages)):
            p.audit["deterministic_normalizations"].append("evidence_page_deduplication")
        if pages != sorted(pages):
            p.audit["deterministic_normalizations"].append("evidence_page_sorting")
        for page in p.evidence_pages:
            if page <= 0:
                error("page_nonpositive", f"evidence page {page} must be positive")
            elif page > page_count:
                error("page_out_of_range", f"evidence page {page} exceeds evaluation PDF page count {page_count}")
    p.audit["format_status"] = "valid" if not p.errors else "invalid"
    contract_errors = len(p.errors)
    if isinstance(answer, str) and isinstance(pages, list):
        if answer == "Unanswerable" and pages:
            error("refusal_with_pages", 'answer_pre is "Unanswerable" but evidence_pages is non-empty')
        elif answer != "Unanswerable" and not pages:
            error("answer_without_evidence", 'Only exact "Unanswerable" may have empty evidence_pages')
        # These are explicit prohibited labels, not semantic answer normalization.
        if (answer != "Unanswerable" and answer.strip().casefold() == "unanswerable") or answer in {"Not mentioned", "Not provided", "Unknown", "N/A", "None", "I cannot answer"}:
            error("noncanonical_refusal", 'Refusal must be exactly "Unanswerable"')
    p.audit["semantic_contract_status"] = "valid" if len(p.errors) == contract_errors else "invalid"
    if raw != raw.strip():
        p.audit["deterministic_normalizations"].append("outer_whitespace")
    p.audit["final_normalized_output"] = ({"answer_pre": answer, "evidence_pages": p.evidence_pages}
                                           if p.status == "legal" else None)
    p.audit["final_parse_status"] = p.status
    return p


class SubmissionError(ValueError):
    """Fatal ID/envelope ambiguity. Never guess a binding for invalid JSONL."""


def apply_generation_audit(p: Prediction, audit: dict, page_count: int) -> Prediction:
    """Verify a producer's complete retry trail without reconstructing output."""
    if not isinstance(audit, dict) or not isinstance(audit.get("attempts"), list) or not audit["attempts"]:
        raise SubmissionError(f"{p.qa_id}: generation_audit must contain nonempty attempts")
    first_raw = None
    for i, attempt in enumerate(audit["attempts"], 1):
        if not isinstance(attempt, dict) or type(attempt.get("attempt")) is not int or attempt["attempt"] != i:
            raise SubmissionError(f"{p.qa_id}: invalid attempt ordering")
        status = attempt.get("status")
        if status not in {"legal", "illegal", "technical_failure"}:
            raise SubmissionError(f"{p.qa_id}: invalid generation attempt status")
        raw = attempt.get("raw_output")
        if raw is not None:
            if not isinstance(raw, str) or attempt.get("raw_output_sha256") != digest(raw):
                raise SubmissionError(f"{p.qa_id}: attempt raw output hash mismatch")
            if first_raw is None:
                first_raw = raw
            if status != "technical_failure" and validate_raw(p.qa_id, raw, page_count).status != status:
                raise SubmissionError(f"{p.qa_id}: claimed attempt status disagrees with strict parsing")
        elif status != "technical_failure":
            raise SubmissionError(f"{p.qa_id}: successful generation has no raw bytes")
        if i < len(audit["attempts"]) and status == "legal":
            raise SubmissionError(f"{p.qa_id}: cannot correct an already legal prediction")
    last = audit["attempts"][-1]
    if audit.get("original_raw_output") != first_raw or audit.get("final_status") != last["status"]:
        raise SubmissionError(f"{p.qa_id}: first/final generation audit binding mismatch")
    final_raw = p.audit["original_raw_output"]
    if last["status"] == "technical_failure":
        if final_raw != "":
            raise SubmissionError(f"{p.qa_id}: failed generation cannot claim a final model answer")
        p.status, p.answer_pre, p.evidence_pages = "technical_failure", None, None
        p.errors = [{"type": "generation_failure", "message": "Generation exhausted its bounded attempts"}]
        p.audit["final_parse_status"] = "technical_failure"
    elif final_raw != last.get("raw_output"):
        raise SubmissionError(f"{p.qa_id}: final raw output is not the last generation attempt")
    p.audit["generation_audit"] = audit
    p.audit["retry_outputs"] = [a.get("raw_output") for a in audit["attempts"][1:]]
    p.audit["correction_trigger"] = [a.get("correction_trigger") for a in audit["attempts"][1:]]
    return p


def load_submission(path: Path, gold: dict[str, Gold]) -> dict[str, Prediction]:
    text = path.read_text(encoding="utf-8")
    try:
        records = strict_json(text) if text.lstrip().startswith("[") else [
            strict_json(line) for line in text.splitlines() if line.strip()]
    except ValueError as exc:
        raise SubmissionError(f"submission_parse_error: {exc}; cannot bind malformed records to QA IDs") from exc
    predictions = {}
    for line, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get("qa_id"), str):
            raise SubmissionError(f"record {line}: qa_id must be a string binding metadata field")
        qa_id = record["qa_id"]
        if qa_id in predictions:
            raise SubmissionError(f"{qa_id}: duplicate prediction QA ID (fatal)")
        if qa_id not in gold:
            raise SubmissionError(f"{qa_id}: unknown QA ID (fatal)")
        # Raw envelope is an audit ingestion format, not the recommended submission.
        if set(record) in ({"qa_id", "raw_model_output"}, {"qa_id", "raw_model_output", "generation_audit"}) and isinstance(record["raw_model_output"], str):
            raw = record["raw_model_output"]
            source = "raw_envelope"
        else:
            raw = json.dumps({k: v for k, v in record.items() if k != "qa_id"}, ensure_ascii=False)
            source = "submission_fields_original_model_bytes_unavailable"
        p = validate_raw(qa_id, raw, gold[qa_id].page_count)
        if source == "raw_envelope" and "generation_audit" in record:
            p = apply_generation_audit(p, record["generation_audit"], gold[qa_id].page_count)
        p.audit["source"] = source
        p.audit["submission_record"] = record
        predictions[qa_id] = p
    for qa_id in gold:
        predictions.setdefault(qa_id, Prediction(qa_id, "missing"))
    return predictions


def validation_report(predictions: dict[str, Prediction]) -> dict:
    counts = Counter(p.status for p in predictions.values())
    errors = [{"qa_id": p.qa_id, **error} for p in predictions.values() for error in p.errors]
    return {"total": len(predictions), "received": len(predictions) - counts["missing"],
            "known_qa_count": len(predictions) - counts["missing"], "missing": counts["missing"],
            "legal": counts["legal"], "illegal": counts["illegal"],
            "technical_failures": counts["technical_failure"], "duplicate_ids": [], "unknown_ids": [],
            "schema_validity": not any(e["type"] in {"invalid_json", "wrong_fields", "wrong_object_type", "answer_type", "pages_type", "page_type"} for e in errors),
            "evidence_range_validity": not any(e["type"] in {"page_nonpositive", "page_out_of_range", "page_type", "pages_type"} for e in errors),
            "refusal_validity": not any(e["type"] in {"refusal_with_pages", "noncanonical_refusal", "answer_without_evidence"} for e in errors),
            "errors": errors}
