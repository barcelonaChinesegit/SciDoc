from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/reporting/report_reasoning_qa_eval.py"
)
SPEC = importlib.util.spec_from_file_location("reasoning_eval_report", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def qa(question: str, answer: str, reasoning_type: str) -> dict:
    return {
        "question": question,
        "answer": answer,
        "reasoning_type": reasoning_type,
        "source_qa_ids": ["QA1", "QA2"],
        "evidence_pages": [1],
        "evidence_provenance": {
            "method": "sorted_union_of_source_qa_evidence_pages",
            "validation": {
                "all_source_qa_ids_resolved": True,
                "pages_are_positive_integers": True,
                "pages_are_sorted_and_unique": True,
                "pages_within_pdf_bounds": True,
            },
        },
        "dual_model_validation": {
            provider: {
                "decision": "KEEP",
                "confidence": 0.95,
                "reason": "supported",
            }
            for provider in MODULE.PROVIDERS
        },
    }


def dataset() -> dict:
    return {
        "1": {
            "paper": "1",
            "primary_category": "Science",
            "QA": {
                "R1": qa("Question one?", "A", "causal_chain"),
                "R2": qa("Question two?", "B", "comparison"),
            },
        }
    }


PDF_AUDIT = {
    "pdf_corpus_sha256": "corpus-fixture",
    "paper_count": 1,
    "papers": {
        "1": {
            "path": "/fixture/1.pdf",
            "sha256": "pdf-fixture",
            "total_pdf_pages": 2,
        }
    },
}


def judged(first: bool, second: bool) -> dict:
    source = dataset()
    for index, item in enumerate(source["1"]["QA"].values()):
        correct = first if index == 0 else second
        parsed_answer = item["answer"] if correct else "wrong"
        item.update(
            {
                "model_output": json.dumps(
                    {"answer_pre": parsed_answer, "evidence_pages": [1]},
                    separators=(",", ":"),
                ),
                "is_correct": correct,
                "answer_is_correct": correct,
                "evidence_pages_is_correct": True,
                "output_is_legal": True,
                "input_mode": "pdf",
                "correct_answer": item["answer"],
                "require_structured_output": True,
                "require_evidence_pages": True,
                "reference_evidence_pages": item["evidence_pages"],
                "predicted_evidence_pages": item["evidence_pages"],
                "parsed_answer": parsed_answer,
                "match_method": "exact_normalized",
                "page_input_policy": "full",
                "prompt_style": "pdf",
                "shown_pdf_pages": [1, 2],
                "protocol_fingerprint": "protocol-fixture",
                "qa_source_sha256": "source-fixture",
                "pdf_corpus_sha256": "corpus-fixture",
                "pdf_sha256": "pdf-fixture",
            }
        )
    return source


def write_artifacts(output: Path, left: dict, right: dict) -> None:
    judge_values = {}
    for model, value in (("4B", left), ("8B", right)):
        inference = json.loads(json.dumps(value))
        for item in inference["1"]["QA"].values():
            item["evaluated_model"] = model
            raw_output = str(item["model_output"])
            item["raw_model_output"] = raw_output
            item["raw_model_output_sha256"] = hashlib.sha256(
                raw_output.encode("utf-8")
            ).hexdigest()
            item["deterministic_normalizations"] = []
        (output / f"results_{model}.json").write_text(
            json.dumps(inference), encoding="utf-8"
        )
        judge = json.loads(json.dumps(inference))
        judge_metadata = {
            "stage": "judge",
            "scoring_protocol_version": MODULE.SCORING_PROTOCOL_VERSION,
            "judge_runner_sha256": "runner-fixture",
            "source_code_sha256": {"src/pku_qa/evaluation/run_judge.py": "code-fixture"},
            "judge_provider_identity": {"provider_name": "judge-fixture"},
            "judge_config": {
                "judge_provider_name": "judge-fixture",
                "max_new_tokens": 128,
                "max_judge_retries": 3,
            },
            "gold_source_sha256": "source-fixture",
            "inference_protocol_fingerprint": "protocol-fixture",
            "input_mode": "pdf",
        }
        judge_fingerprint = hashlib.sha256(
            json.dumps(
                judge_metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        judge_metadata["judge_protocol_fingerprint"] = judge_fingerprint
        per_qa = {"1": {}}
        for qa_id, item in judge["1"]["QA"].items():
            binding = MODULE.judge_inference_binding_sha256(
                inference["1"]["QA"][qa_id]
            )
            item["judge_protocol_fingerprint"] = judge_fingerprint
            item["inference_binding_sha256"] = binding
            item["publication_eligible"] = True
            item["protocol_validation"] = "publication_strict"
            per_qa["1"][qa_id] = {
                "inference_binding_sha256": binding
            }
        judge_values[model] = judge
        judge_path = output / f"judge_{model}.json"
        judge_path.write_text(json.dumps(judge), encoding="utf-8")
        manifest = {
            "version": 2,
            "metadata": {
                "stage": "inference",
                "protocol_version": MODULE.INFERENCE_PROTOCOL_VERSION,
                "protocol_fingerprint": "protocol-fixture",
                "config": {
                    "qa_source_sha256": "source-fixture",
                    "pdf_corpus_sha256": "corpus-fixture",
                    "input_mode": "pdf",
                    "page_input_policy": "full",
                    "model": model,
                    "prompt_style": "pdf",
                    "pdf_mode": True,
                    "max_pdf_pages": 0,
                },
            },
        }
        manifest_path = (
            output / ".adaptive_queue" / f"inference_{model}" / "manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        queue_source_sha256 = hashlib.sha256(
            json.dumps(
                inference,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        judge_manifest = {
            "version": 2,
            "source_sha256": queue_source_sha256,
            "metadata": judge_metadata,
            "validation_contract": {
                "require_structured_output": False,
                "required_qa_fields": list(
                    MODULE.STRICT_JUDGE_REQUIRED_QA_FIELDS
                ),
                "required_qa_field_values": {
                    "judge_protocol_fingerprint": judge_fingerprint,
                    "protocol_fingerprint": "protocol-fixture",
                    "qa_source_sha256": "source-fixture",
                    "input_mode": "pdf",
                    "pdf_corpus_sha256": "corpus-fixture",
                },
                "required_qa_field_values_by_paper": {},
                "required_qa_field_values_by_qa": per_qa,
            },
        }
        judge_manifest_path = (
            output / ".adaptive_queue" / f"judge_{model}" / "manifest.json"
        )
        judge_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        judge_manifest_path.write_text(
            json.dumps(judge_manifest), encoding="utf-8"
        )

    models = []
    profiles = []
    for model, value in judge_values.items():
        rows = list(value["1"]["QA"].values())
        models.append(
            {
                "model_name": model,
                "judge_file": str((output / f"judge_{model}.json").resolve()),
                "total": 2,
                "structured_required_total": 2,
                "evidence_required_total": 2,
                "evidence_required_legal_samples": sum(
                    row["output_is_legal"] for row in rows
                ),
                "answer_correct": sum(row["answer_is_correct"] for row in rows),
                "evidence_pages_correct": sum(
                    row["evidence_pages_is_correct"] for row in rows
                ),
                "both_correct": sum(row["is_correct"] for row in rows),
            }
        )
        profiles.append(
            {
                "judge_file": str((output / f"judge_{model}.json").resolve()),
                "input_mode": "pdf",
                "total": 2,
                "qa_source_sha256": "source-fixture",
                "pdf_corpus_sha256": "corpus-fixture",
                "protocol_fingerprint": "protocol-fixture",
                "judge_protocol_fingerprint": rows[0][
                    "judge_protocol_fingerprint"
                ],
                "evaluated_model": model,
                "inference_file_sha256": MODULE.sha256(
                    output / f"results_{model}.json"
                ),
            }
        )
    (output / "final_report.json").write_text(
        json.dumps(
            {
                "gold_source_sha256": "source-fixture",
                "publication_eligible": True,
                "protocol_validation": "publication_strict",
                "missing_files": [],
                "models": models,
                "protocol_profiles": profiles,
            }
        ),
        encoding="utf-8",
    )


def test_preflight_requires_strict_dual_keep() -> None:
    result = MODULE.validate_dataset(dataset(), 2, 0.8)
    assert result["status"] == "passed"
    broken = dataset()
    broken["1"]["QA"]["R1"]["dual_model_validation"]["claude"][
        "decision"
    ] = "REJECT"
    try:
        MODULE.validate_dataset(broken, 2, 0.8)
    except ValueError as exc:
        assert "not KEEP" in str(exc)
    else:
        raise AssertionError("Expected strict validation to fail")


def test_report_calculates_model_and_paired_accuracy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_artifacts(output, judged(True, False), judged(True, True))
        source = dataset()
        preflight = MODULE.validate_dataset(source, 2, 0.8)
        preflight["source_sha256"] = "source-fixture"
        report = MODULE.build_report(source, preflight, output, PDF_AUDIT)
        assert report["models"]["4B"]["answer_accuracy"] == 50.0
        assert report["models"]["4B"]["evidence_accuracy"] == 100.0
        assert report["models"]["4B"]["joint_accuracy"] == 50.0
        assert report["models"]["8B"]["answer_accuracy"] == 100.0
        assert report["models"]["8B"]["evidence_accuracy"] == 100.0
        assert report["models"]["8B"]["joint_accuracy"] == 100.0
        assert report["paired_outcomes"] == {
            "both_correct": 1,
            "only_8b_correct": 1,
        }


def test_illegal_output_counts_wrong_in_model_and_paired_metrics() -> None:
    left = judged(True, True)
    right = judged(True, True)
    for source in (left, right):
        item = source["1"]["QA"]["R1"]
        item["model_output"] = "not-json"
        item["parsed_answer"] = ""
        item["predicted_evidence_pages"] = []
        item["output_is_legal"] = False
        item["answer_is_correct"] = False
        item["evidence_pages_is_correct"] = False
        item["is_correct"] = False
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_artifacts(output, left, right)
        source = dataset()
        report = MODULE.build_report(
            source,
            {
                **MODULE.validate_dataset(source, 2, 0.8),
                "source_sha256": "source-fixture",
            },
            output,
            PDF_AUDIT,
        )
    assert report["models"]["4B"]["answer_correct"] == 1
    assert report["paired_answer_outcomes"] == {
        "both_correct": 1,
        "both_incorrect": 1,
    }


def test_report_rejects_nonzero_max_pdf_pages_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_artifacts(output, judged(True, True), judged(True, True))
        path = output / ".adaptive_queue/inference_4B/manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["metadata"]["config"]["max_pdf_pages"] = 8
        path.write_text(json.dumps(value), encoding="utf-8")
        source = dataset()
        preflight = {
            **MODULE.validate_dataset(source, 2, 0.8),
            "source_sha256": "source-fixture",
        }
        try:
            MODULE.build_report(source, preflight, output, PDF_AUDIT)
        except ValueError as exc:
            assert "max_pdf_pages" in str(exc)
        else:
            raise AssertionError("page-capped run must fail closed")


def test_report_rejects_incomplete_full_pdf_page_list() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        left = judged(True, True)
        left["1"]["QA"]["R1"]["shown_pdf_pages"] = [1]
        write_artifacts(output, left, judged(True, True))
        source = dataset()
        preflight = {
            **MODULE.validate_dataset(source, 2, 0.8),
            "source_sha256": "source-fixture",
        }
        try:
            MODULE.build_report(source, preflight, output, PDF_AUDIT)
        except ValueError as exc:
            assert "shown_pdf_pages" in str(exc)
        else:
            raise AssertionError("partial PDF input must fail closed")


def test_report_reparses_raw_output_instead_of_trusting_parsed_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        left = judged(True, True)
        # Raw output says page 2 while the pre-parsed/Judge fields claim page 1.
        left["1"]["QA"]["R1"]["model_output"] = (
            '{"answer_pre":"A","evidence_pages":[2]}'
        )
        write_artifacts(output, left, judged(True, True))
        source = dataset()
        preflight = {
            **MODULE.validate_dataset(source, 2, 0.8),
            "source_sha256": "source-fixture",
        }
        try:
            MODULE.build_report(source, preflight, output, PDF_AUDIT)
        except ValueError as exc:
            assert "independent_parsed_evidence" in str(exc)
        else:
            raise AssertionError("stale parsed evidence must fail closed")


def test_report_rejects_stale_judge_protocol_fingerprint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_artifacts(output, judged(True, True), judged(True, True))
        path = output / "judge_4B.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["1"]["QA"]["R1"]["judge_protocol_fingerprint"] = "stale"
        path.write_text(json.dumps(value), encoding="utf-8")
        source = dataset()
        preflight = {
            **MODULE.validate_dataset(source, 2, 0.8),
            "source_sha256": "source-fixture",
        }
        try:
            MODULE.build_report(source, preflight, output, PDF_AUDIT)
        except ValueError as exc:
            assert "Judge_protocol_fingerprint" in str(exc)
        else:
            raise AssertionError("stale Judge fingerprint must fail closed")
