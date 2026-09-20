#!/usr/bin/env python3
"""Single source of truth for benchmark input and output protocols.

There are exactly two supported evaluation protocols:

``question_only``
    The model receives no PDF and returns only its final answer.

``pdf``
    The model receives PDF page images and returns the canonical
    ``answer_pre`` + ``evidence_pages`` JSON object.  Answerable items require
    at least one gold evidence page.  An unanswerable item must use the exact
    reference label ``Unanswerable`` and has no evidence-page target.

Keeping these rules here prevents individual runners, judges, and reports from
silently defining slightly different experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterator


PDF_INPUT_MODE = "pdf"
QUESTION_ONLY_INPUT_MODE = "question_only"
SUPPORTED_INPUT_MODES = {PDF_INPUT_MODE, QUESTION_ONLY_INPUT_MODE}
UNANSWERABLE_LABEL = "Unanswerable"
INFERENCE_PROTOCOL_VERSION = 7
SCORING_PROTOCOL_VERSION = 6

JUDGE_REQUIRED_QA_FIELDS = (
    "is_correct",
    "answer_is_correct",
    "evidence_pages_is_correct",
    "input_mode",
    "require_structured_output",
    "require_evidence_pages",
    "output_is_legal",
)

INFERENCE_REQUIRED_QA_FIELDS = (
    "input_mode",
    "require_structured_output",
    "require_evidence_pages",
    "reference_evidence_pages",
    "prompt_style",
)
STRICT_INFERENCE_REQUIRED_QA_FIELDS = INFERENCE_REQUIRED_QA_FIELDS + (
    "protocol_fingerprint",
    "qa_source_sha256",
    "pdf_corpus_sha256",
    "pdf_sha256",
    "page_input_policy",
    "shown_pdf_pages",
    "total_pdf_pages",
    "max_pdf_pages",
    "evaluated_model",
    "raw_model_output",
    "raw_model_output_sha256",
    "deterministic_normalizations",
)

# Every Judge row is a score for one exact inference record.  Keep this list
# centralized so the Judge, durable queue, generic report, and specialist
# reports cannot silently bind to different subsets of the model output.
JUDGE_INFERENCE_BINDING_FIELDS = (
    "evaluated_model",
    "question",
    "correct_answer",
    "model_output",
    "raw_model_output",
    "raw_model_output_sha256",
    "deterministic_normalizations",
    "type",
    "input_mode",
    "require_structured_output",
    "require_evidence_pages",
    "reference_evidence_pages",
    "page_input_policy",
    "prompt_style",
    "shown_pdf_pages",
    "total_pdf_pages",
    "max_pdf_pages",
    "protocol_fingerprint",
    "qa_source_sha256",
    "pdf_corpus_sha256",
    "pdf_sha256",
)
STRICT_JUDGE_REQUIRED_QA_FIELDS = JUDGE_REQUIRED_QA_FIELDS + (
    "judge_protocol_fingerprint",
    "inference_binding_sha256",
    "protocol_fingerprint",
    "qa_source_sha256",
    "pdf_corpus_sha256",
    "pdf_sha256",
    "page_input_policy",
    "prompt_style",
    "shown_pdf_pages",
    "total_pdf_pages",
    "max_pdf_pages",
    "evaluated_model",
    "raw_model_output",
    "raw_model_output_sha256",
    "deterministic_normalizations",
    "publication_eligible",
    "protocol_validation",
)


@dataclass(frozen=True)
class EvaluationProtocol:
    input_mode: str
    require_structured_output: bool
    require_evidence_pages: bool


def is_exact_unanswerable(value: Any) -> bool:
    return str(value or "").strip() == UNANSWERABLE_LABEL


def is_unanswerable_spelling(value: Any) -> bool:
    """Return whether a value is a case-only variant of the canonical label."""
    return str(value or "").strip().casefold() == UNANSWERABLE_LABEL.casefold()


def canonical_gold_answer(qa: dict[str, Any], *, item_id: str) -> str:
    """Return the one allowed gold-answer field for source datasets."""
    if "answer" not in qa:
        raise ValueError(f"{item_id}: gold QA must contain the 'answer' field")
    value = qa["answer"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{item_id}: answer must be a non-empty string")
    return value.strip()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pdf_path(paper_id: str, pdf_dirs: list[str | Path]) -> Path:
    return resolve_asset_pdf_path(paper_id, pdf_dirs)


def pdf_corpus_sha256(
    dataset: dict[str, Any], pdf_dirs: list[str | Path]
) -> str:
    """Hash the exact PDF bytes used by a PDF-mode experiment."""
    return pdf_corpus_sha256_from_manifest(
        pdf_sha256_manifest(dataset, pdf_dirs)
    )


def pdf_sha256_manifest(
    dataset: dict[str, Any], pdf_dirs: list[str | Path]
) -> dict[str, str]:
    """Return the immutable per-paper PDF hashes for an experiment."""
    manifest: dict[str, str] = {}
    for paper_id in sorted(
        str(value)
        for value in dataset
        if not str(value).startswith("__")
    ):
        pdf_path = resolve_pdf_path(paper_id, pdf_dirs)
        manifest[paper_id] = sha256_file(pdf_path)
    return manifest


def pdf_corpus_sha256_from_manifest(manifest: dict[str, str]) -> str:
    """Hash a sorted paper/hash manifest using the publication encoding."""
    payload = json.dumps(
        sorted((str(key), str(value)) for key, value in manifest.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_inference_protocol_metadata(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable cache/experiment fingerprint for inference."""
    runner_path = Path(__file__).with_name("run_inference.py")
    normalized = json.loads(
        json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
    )
    payload = {
        "protocol_version": INFERENCE_PROTOCOL_VERSION,
        "inference_runner_sha256": sha256_file(runner_path),
        "source_code_sha256": {
            name: sha256_file(runner_path.with_name(name))
            for name in (
                "run_inference.py",
                "evaluation_protocol.py",
                "eval_framework.py",
            )
        },
        "config": normalized,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **payload,
        "protocol_fingerprint": hashlib.sha256(encoded).hexdigest(),
    }


def build_inference_queue_value_contract(
    dataset: dict[str, Any],
    *,
    protocol_metadata: dict[str, Any],
    qa_source_sha256: str,
    pdf_corpus_hash: str | None,
    pdf_sha256_by_paper: dict[str, str],
    input_mode: str,
    page_input_policy: str,
    prompt_style: str,
    max_pdf_pages: int,
    evaluated_model: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Bind queue cache rows to exact scheduler inputs and protocol values."""
    fingerprint = str(
        protocol_metadata.get("protocol_fingerprint", "")
    ).strip()
    if not fingerprint:
        raise ValueError("Inference queue contract requires a fingerprint")
    paper_ids = {
        str(paper_id)
        for paper_id in dataset
        if not str(paper_id).startswith("__")
    }
    if input_mode == PDF_INPUT_MODE and set(pdf_sha256_by_paper) != paper_ids:
        raise ValueError("Per-paper PDF hash manifest does not match QA papers")
    global_values = {
        "protocol_fingerprint": fingerprint,
        "qa_source_sha256": str(qa_source_sha256),
        "pdf_corpus_sha256": pdf_corpus_hash,
        "input_mode": input_mode,
        "page_input_policy": page_input_policy,
        "prompt_style": prompt_style,
        "max_pdf_pages": int(max_pdf_pages),
        "evaluated_model": str(evaluated_model),
    }
    per_paper = {
        paper_id: {
            "pdf_sha256": (
                pdf_sha256_by_paper[paper_id]
                if input_mode == PDF_INPUT_MODE
                else None
            )
        }
        for paper_id in sorted(paper_ids)
    }
    return global_values, per_paper


def _safe_provider_spec(value: Any, *, key: str = "") -> Any:
    """Remove credential values while retaining behavior-relevant config."""
    sensitive = ("password", "secret", "api_key", "access_token")
    if any(token in key.casefold() for token in sensitive):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(child_key): _safe_provider_spec(
                child_value, key=str(child_key)
            )
            for child_key, child_value in sorted(value.items())
        }
    if isinstance(value, list):
        return [_safe_provider_spec(item, key=key) for item in value]
    return value


def _asset_directory_identity(path_value: Any) -> dict[str, Any]:
    raw_path = Path(str(path_value))
    path = raw_path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model/processor directory not found: {path}")
    files: list[dict[str, Any]] = []
    for child in sorted(path.iterdir(), key=lambda value: value.name):
        if not child.is_file():
            continue
        stat = child.stat()
        record: dict[str, Any] = {
            "name": child.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        # Hash configs, tokenizer assets, and shard indexes exactly. Weight
        # shards are identity-locked by path/size/mtime without rereading
        # tens of gigabytes in every GPU worker startup.
        if child.suffix != ".safetensors":
            record["sha256"] = sha256_file(child)
        files.append(record)
    encoded = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "path": str(path),
        "files": files,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def provider_runtime_identity(
    provider_name: str, provider_spec: dict[str, Any]
) -> dict[str, Any]:
    """Describe the exact provider implementation and local model assets."""
    assets: dict[str, Any] = {}
    for field in ("model_path", "processor_path"):
        value = provider_spec.get(field)
        if value is None:
            continue
        identity = _asset_directory_identity(value)
        assets[field] = identity
    package_versions: dict[str, str | None] = {}
    for package in (
        "torch",
        "transformers",
        "modelscope",
        "qwen-vl-utils",
        "pypdfium2",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "provider_name": str(provider_name),
        "provider_spec": _safe_provider_spec(provider_spec),
        "assets": assets,
        "package_versions": package_versions,
    }


def build_judge_queue_contract(
    *,
    input_mode: str,
    judge_runner_path: str | Path | None = None,
    gold_source_sha256: str | None = None,
    inference_protocol_fingerprint: str | None = None,
    judge_provider_identity: dict[str, Any] | None = None,
    judge_config: dict[str, Any] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Return the one parent/worker manifest contract for Judge queues.

    The adaptive scheduler creates a queue before launching Judge workers.
    Both processes must write byte-for-byte equivalent validation metadata or
    the durable queue correctly rejects the worker as a different experiment.
    Keeping the constructor here prevents that contract from drifting.
    """
    protocol_for_item(input_mode)
    if judge_runner_path is None:
        raise ValueError("Judge contract requires judge_runner_path")
    gold_hash = str(gold_source_sha256 or "").strip()
    inference_fingerprint = str(
        inference_protocol_fingerprint or ""
    ).strip()
    if not gold_hash:
        raise ValueError("Judge contract requires gold_source_sha256")
    if not inference_fingerprint:
        raise ValueError(
            "Judge contract requires inference_protocol_fingerprint"
        )
    if not isinstance(judge_provider_identity, dict):
        raise ValueError(
            "Judge contract requires judge_provider_identity"
        )
    if not isinstance(judge_config, dict):
        raise ValueError("Judge contract requires judge_config")
    runner_path = Path(judge_runner_path)
    metadata = {
        "stage": "judge",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "judge_runner_sha256": sha256_file(runner_path),
        "source_code_sha256": {
            name: sha256_file(runner_path.with_name(name))
            for name in (
                "run_judge.py",
                "evaluation_protocol.py",
                "eval_framework.py",
            )
        },
        "judge_prompt_sha256": sha256_file(
            Path(__file__).resolve().parents[3] / "evaluation/prompts/semantic_judge.txt"
        ),
        "judge_provider_identity": judge_provider_identity,
        "judge_config": json.loads(
            json.dumps(
                judge_config,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        ),
        "gold_source_sha256": gold_hash,
        "inference_protocol_fingerprint": inference_fingerprint,
        "input_mode": input_mode,
    }
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata["judge_protocol_fingerprint"] = hashlib.sha256(
        encoded
    ).hexdigest()
    return STRICT_JUDGE_REQUIRED_QA_FIELDS, metadata


def judge_inference_binding_sha256(record: dict[str, Any]) -> str:
    """Hash every inference field that can affect one Judge verdict."""
    if not isinstance(record, dict):
        raise ValueError("Judge binding requires an inference QA object")
    payload = {
        field: record.get(field) for field in JUDGE_INFERENCE_BINDING_FIELDS
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_judge_queue_value_contract(
    inference_results: dict[str, Any],
    *,
    judge_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    """Bind cached Judge rows to the exact per-QA inference bytes.

    A queue source hash protects an existing queue directory.  This additional
    compact per-QA digest also protects a *new* queue when an older monolithic
    Judge file is offered to ``bootstrap``.
    """
    judge_fingerprint = str(
        judge_metadata.get("judge_protocol_fingerprint", "")
    ).strip()
    inference_fingerprint = str(
        judge_metadata.get("inference_protocol_fingerprint", "")
    ).strip()
    gold_hash = str(judge_metadata.get("gold_source_sha256", "")).strip()
    input_mode = str(judge_metadata.get("input_mode", "")).strip()
    if not all(
        (judge_fingerprint, inference_fingerprint, gold_hash, input_mode)
    ):
        raise ValueError("Incomplete strict Judge metadata")
    protocol_for_item(input_mode)
    per_qa: dict[str, dict[str, dict[str, Any]]] = {}
    corpus_hashes: set[Any] = set()
    for paper_id, paper in inference_results.items():
        if str(paper_id).startswith("__"):
            continue
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Malformed inference paper: {paper_id}")
        paper_values: dict[str, dict[str, Any]] = {}
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(
                    f"Malformed inference QA: {paper_id}/{qa_id}"
                )
            if qa.get("protocol_fingerprint") != inference_fingerprint:
                raise ValueError(
                    f"Inference fingerprint mismatch: {paper_id}/{qa_id}"
                )
            if qa.get("qa_source_sha256") != gold_hash:
                raise ValueError(
                    f"Inference source hash mismatch: {paper_id}/{qa_id}"
                )
            if qa.get("input_mode") != input_mode:
                raise ValueError(
                    f"Inference input mode mismatch: {paper_id}/{qa_id}"
                )
            corpus_hashes.add(qa.get("pdf_corpus_sha256"))
            paper_values[str(qa_id)] = {
                "inference_binding_sha256": (
                    judge_inference_binding_sha256(qa)
                )
            }
        per_qa[str(paper_id)] = paper_values
    if len(corpus_hashes) > 1:
        raise ValueError("Inference results mix PDF corpus hashes")
    global_values = {
        "judge_protocol_fingerprint": judge_fingerprint,
        "protocol_fingerprint": inference_fingerprint,
        "qa_source_sha256": gold_hash,
        "input_mode": input_mode,
        "pdf_corpus_sha256": (
            next(iter(corpus_hashes)) if corpus_hashes else None
        ),
    }
    return global_values, per_qa


def protocol_for_item(input_mode: str, reference_answer: Any = None) -> EvaluationProtocol:
    if input_mode not in SUPPORTED_INPUT_MODES:
        raise ValueError(
            f"Unsupported input_mode={input_mode!r}; expected one of "
            f"{sorted(SUPPORTED_INPUT_MODES)}"
        )
    if input_mode == QUESTION_ONLY_INPUT_MODE:
        return EvaluationProtocol(
            input_mode=input_mode,
            require_structured_output=False,
            require_evidence_pages=False,
        )
    return EvaluationProtocol(
        input_mode=input_mode,
        require_structured_output=True,
        require_evidence_pages=not is_exact_unanswerable(reference_answer),
    )


def configure_protocol(args: Any) -> EvaluationProtocol:
    """Derive all output requirements from the sole protocol selector.

    Command-line callers expose only ``input_mode``.  The attributes written
    here are internal runner state, so a hybrid answer/evidence protocol cannot
    be requested.
    """
    mode = str(getattr(args, "input_mode", ""))
    protocol = protocol_for_item(mode)
    page_policy = str(getattr(args, "page_input_policy", "full"))
    if mode == QUESTION_ONLY_INPUT_MODE:
        if page_policy != "full":
            raise ValueError(
                "--page-input-policy oracle/qa_field requires --input-mode pdf"
            )
        args.require_evidence_pages = False
        args.pdf_mode = False
    else:
        # PDF mode has one canonical schema throughout the project.
        args.require_evidence_pages = True
        args.pdf_mode = True
    return protocol


def iter_dataset_qas(
    dataset: Any,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    if not isinstance(dataset, dict):
        raise ValueError("QA dataset must be a JSON object keyed by paper id")
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict) or not isinstance(paper.get("QA"), dict):
            raise ValueError(f"Malformed paper record: {paper_id}")
        for qa_id, qa in paper["QA"].items():
            if not isinstance(qa, dict):
                raise ValueError(f"Malformed QA record: {paper_id}/{qa_id}")
            yield str(paper_id), str(qa_id), qa


def canonical_gold_pages(raw_pages: Any, *, item_id: str) -> list[int]:
    if not isinstance(raw_pages, list):
        raise ValueError(f"{item_id}: evidence_pages must be a JSON list")
    pages: list[int] = []
    for page in raw_pages:
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise ValueError(
                f"{item_id}: evidence_pages must contain positive integers"
            )
        if page in pages:
            raise ValueError(f"{item_id}: duplicate evidence page {page}")
        pages.append(page)
    if pages != sorted(pages):
        raise ValueError(f"{item_id}: evidence_pages must be sorted")
    return pages


def validate_dataset_protocol(dataset: Any, input_mode: str) -> dict[str, int | str]:
    """Fail before model loading when gold data cannot support the protocol."""
    protocol_for_item(input_mode)
    total = 0
    answerable = 0
    unanswerable = 0
    evidence_references = 0
    for paper_id, qa_id, qa in iter_dataset_qas(dataset):
        item_id = f"{paper_id}/{qa_id}"
        raw_question = qa.get("question")
        if not isinstance(raw_question, str) or not raw_question.strip():
            raise ValueError(
                f"{item_id}: question must be a non-empty string"
            )
        question = raw_question.strip()
        answer = canonical_gold_answer(qa, item_id=item_id)
        if is_unanswerable_spelling(answer) and not is_exact_unanswerable(answer):
            raise ValueError(
                f"{item_id}: unanswerable gold label must be exactly "
                f"{UNANSWERABLE_LABEL!r}"
            )
        total += 1
        if is_exact_unanswerable(answer):
            unanswerable += 1
            if input_mode == PDF_INPUT_MODE:
                pages = canonical_gold_pages(
                    qa.get("evidence_pages", []), item_id=item_id
                )
                if pages:
                    raise ValueError(
                        f"{item_id}: Unanswerable items must have evidence_pages=[]"
                    )
        else:
            answerable += 1
            if input_mode == PDF_INPUT_MODE:
                if "evidence_pages" not in qa or qa.get("evidence_pages") == []:
                    raise ValueError(
                        f"{item_id}: answerable PDF item has no gold evidence_pages"
                    )
                pages = canonical_gold_pages(
                    qa.get("evidence_pages"), item_id=item_id
                )
                if not pages:
                    raise ValueError(
                        f"{item_id}: answerable PDF item has no gold evidence_pages"
                    )
                evidence_references += len(pages)
    return {
        "input_mode": input_mode,
        "total": total,
        "answerable": answerable,
        "unanswerable": unanswerable,
        "evidence_references": evidence_references,
    }


def protocol_from_inference_record(record: dict[str, Any]) -> EvaluationProtocol:
    """Validate and resolve the protocol embedded in one inference record."""
    mode = str(record.get("input_mode", ""))
    answer = record.get("correct_answer", record.get("answer", ""))
    expected = protocol_for_item(mode, answer)
    actual_structured = record.get("require_structured_output") is True
    actual_evidence = bool(record.get("require_evidence_pages"))
    if actual_structured != expected.require_structured_output:
        raise ValueError(
            f"input_mode={mode} requires require_structured_output="
            f"{expected.require_structured_output}, found {actual_structured}"
        )
    if actual_evidence != expected.require_evidence_pages:
        raise ValueError(
            f"input_mode={mode}, answer={str(answer)[:80]!r} requires "
            f"require_evidence_pages={expected.require_evidence_pages}, "
            f"found {actual_evidence}"
        )
    if mode == PDF_INPUT_MODE:
        shown_pages = canonical_gold_pages(
            record.get("shown_pdf_pages"),
            item_id="inference record shown_pdf_pages",
        )
        if not shown_pages:
            raise ValueError("PDF inference record has no shown_pdf_pages")
        reference_pages = canonical_gold_pages(
            record.get("reference_evidence_pages", record.get("evidence_pages", [])),
            item_id="inference record",
        )
        if expected.require_evidence_pages and not reference_pages:
            raise ValueError("answerable PDF inference record has no gold evidence pages")
        if not expected.require_evidence_pages and reference_pages:
            raise ValueError("Unanswerable PDF inference record must have no gold evidence pages")
    return expected


def validate_pdf_prediction(
    answer: Any,
    raw_pages: Any,
    *,
    maximum_pages: int | None = None,
    allowed_pages: list[int] | tuple[int, ...] | set[int] | None = None,
) -> tuple[str, list[int]]:
    """Validate PDF fields and canonicalize the evidence page-set order.

    Evidence pages are a mathematical set for scoring. JSON list order does
    not change the prediction. Sort and deduplicate integer pages while keeping
    the answer text unchanged. Inference artifacts retain the exact raw response.
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer_pre must be a non-empty string")
    if not isinstance(raw_pages, list):
        raise ValueError("evidence_pages must be a JSON list")
    pages: list[int] = []
    for page in raw_pages:
        if isinstance(page, bool) or not isinstance(page, int) or page <= 0:
            raise ValueError("evidence_pages must contain positive integers")
        if page not in pages:
            pages.append(page)
    pages = sorted(pages)
    if maximum_pages is not None and len(pages) > maximum_pages:
        raise ValueError(
            f"evidence_pages contains more than {maximum_pages} pages"
        )
    if allowed_pages is not None:
        if not isinstance(allowed_pages, (list, tuple, set)):
            raise ValueError("allowed evidence pages must be a collection")
        allowed_values = (
            sorted(allowed_pages)
            if isinstance(allowed_pages, set)
            else list(allowed_pages)
        )
        allowed = set(
            canonical_gold_pages(
                allowed_values, item_id="shown_pdf_pages"
            )
        )
        outside = [page for page in pages if page not in allowed]
        if outside:
            raise ValueError(
                "evidence_pages contains pages not shown to the model: "
                + ", ".join(map(str, outside))
            )
    if answer == UNANSWERABLE_LABEL:
        if pages:
            raise ValueError("Unanswerable must have evidence_pages=[]")
    elif answer.strip().casefold() == UNANSWERABLE_LABEL.casefold() or answer in {
        "Not mentioned", "Not provided", "Unknown", "N/A", "None", "I cannot answer"
    }:
        raise ValueError(
            f"unanswerable label must be exactly {UNANSWERABLE_LABEL!r}"
        )
    elif not pages:
        raise ValueError("answerable prediction must include evidence_pages")
    return answer, pages


def _unique_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def parse_canonical_pdf_output(
    output: Any,
    *,
    maximum_pages: int | None = None,
    allowed_pages: list[int] | tuple[int, ...] | set[int] | None = None,
) -> tuple[str, list[int]]:
    """Parse the one legal PDF output container used by every scorer."""
    raw = str(output or "").strip()
    if not raw:
        raise ValueError("empty output")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_json_fields, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("output is not one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("output must be a JSON object")
    if set(value) != {"answer_pre", "evidence_pages"}:
        raise ValueError(
            "output keys must be exactly answer_pre and evidence_pages"
        )
    return validate_pdf_prediction(
        value["answer_pre"],
        value["evidence_pages"],
        maximum_pages=maximum_pages,
        allowed_pages=allowed_pages,
    )


def canonicalize_pdf_output_for_storage(
    output: Any,
    *,
    maximum_pages: int | None = None,
    allowed_pages: list[int] | tuple[int, ...] | set[int] | None = None,
) -> tuple[str, list[str]]:
    """Return canonical JSON plus an audit trail of safe normalizations.

    The input must already be the one legal JSON object with exactly the two
    canonical keys. Only outer JSON whitespace and evidence sorting/deduplication
    are normalized. The answer's whitespace is part of the preserved answer.
    Callers store the original response separately for publication audit.
    """
    raw = str(output or "").strip()
    answer, pages = parse_canonical_pdf_output(
        raw,
        maximum_pages=maximum_pages,
        allowed_pages=allowed_pages,
    )
    value = json.loads(raw)
    normalizations: list[str] = []
    if str(output or "") != raw:
        normalizations.append("outer_whitespace")
    if len(value["evidence_pages"]) != len(set(value["evidence_pages"])):
        normalizations.append("evidence_pages_deduplicated")
    if value["evidence_pages"] != sorted(value["evidence_pages"]):
        normalizations.append("evidence_pages_sorted")
    canonical = json.dumps(
        {"answer_pre": answer, "evidence_pages": pages},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return canonical, normalizations
from pku_qa.pdf_assets import resolve_pdf_path as resolve_asset_pdf_path
