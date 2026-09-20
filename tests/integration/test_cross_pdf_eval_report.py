from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pku_qa.workflows.reporting.report_cross_pdf_eval import (
    aggregate,
    generate_report,
    validate_dataset,
    validate_publication_artifacts,
    wilson_interval,
)


def qa_record(uid: str, hops: int = 2) -> dict:
    return {
        "question": "Compare two source documents.",
        "answer": "A concise comparison.",
        "evidence_pages": [1, 2],
        "evidence_items": [
            {
                "physical_pdf_page": 1,
                "source_paper_id": "source-a",
                "supported_fact": "fact a",
            },
            {
                "physical_pdf_page": 2,
                "source_paper_id": "source-b",
                "supported_fact": "fact b",
            },
        ],
        "evidence_hops": hops,
        "source_paper_ids": ["source-a", "source-b"],
        "modal_types": ["text"],
        "question_type": "Inferential",
        "review_status": "review_pass",
        "qa_uid": uid,
    }


def judge_record(answer_ok: bool, page_ok: bool, legal: bool = True) -> dict:
    return {
        "type": "fill",
        "require_evidence_pages": True,
        "parsed_answer": "answer" if legal else "",
        "predicted_evidence_pages": [1, 2] if legal else [],
        "output_parse_method": "json:answer_pre" if legal else "invalid_json",
        "answer_is_correct": answer_ok,
        "evidence_pages_is_correct": page_ok,
        "is_correct": answer_ok and page_ok,
    }


class CrossPdfEvalReportTests(unittest.TestCase):
    def test_publication_gate_rejects_missing_strict_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qa_path = root / "qa.json"
            qa_path.write_text("{}", encoding="utf-8")
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            final_report = root / "final_report.json"
            final_report.write_text(
                json.dumps(
                    {
                        "gold_source_sha256": hashlib.sha256(
                            qa_path.read_bytes()
                        ).hexdigest(),
                        "protocol_profiles": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strict protocol profiles"):
                validate_publication_artifacts(
                    dataset={},
                    qa_path=qa_path,
                    pdf_dir=pdf_dir,
                    judge_paths={
                        "4B": root / "judge_4B.json",
                        "8B": root / "judge_8B.json",
                    },
                    final_report=final_report,
                )

    def test_validate_dataset_checks_hash_count_and_pdfs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "bundle.pdf").write_bytes(b"%PDF-1.4\n")
            qa_path = root / "qa.json"
            qa_path.write_text(
                json.dumps(
                    {
                        "bundle": {
                            "paper": "bundle",
                            "QA": {"QA1": qa_record("bundle/QA1")},
                        }
                    }
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(qa_path.read_bytes()).hexdigest()
            _, summary = validate_dataset(
                qa_path,
                pdf_dir,
                expected_count=1,
                expected_sha256=digest,
            )
            self.assertEqual(summary["qas"], 1)
            self.assertEqual(summary["bundles"], 1)
            self.assertEqual(summary["evidence_hops"], {"2": 1})

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_dataset(
                    qa_path,
                    pdf_dir,
                    expected_count=1,
                    expected_sha256="0" * 64,
                )

    def test_aggregate_keeps_legal_and_end_to_end_denominators_separate(self) -> None:
        rows = [
            judge_record(True, True),
            judge_record(True, False),
            judge_record(False, False, legal=False),
        ]
        result = aggregate(rows)
        self.assertEqual(result["total_seen"], 3)
        self.assertEqual(result["legal_samples"], 2)
        self.assertEqual(result["answer_acc_legal"], 100.0)
        self.assertAlmostEqual(result["end_to_end_answer_success"], 200 / 3)
        low, high = wilson_interval(2, 3)
        self.assertLess(low, result["end_to_end_answer_success"])
        self.assertGreater(high, result["end_to_end_answer_success"])

    def test_generate_report_reconciles_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = {
                "bundle": {
                    "paper": "bundle",
                    "QA": {
                        "QA1": qa_record("bundle/QA1", 2),
                        "QA2": qa_record("bundle/QA2", 3),
                    },
                }
            }
            dataset_summary = {
                "qa_path": "qa.json",
                "pdf_dir": "pdfs",
                "sha256": "abc",
                "bundles": 1,
                "qas": 2,
                "pdfs": 1,
                "source_papers": 2,
                "evidence_items": 4,
                "evidence_hops": {"2": 1, "3": 1},
                "modalities": {"text": 2},
                "review_status": {"review_pass": 2},
            }
            judge_paths = {}
            for model, records in {
                "4B": {
                    "QA1": judge_record(True, True),
                    "QA2": judge_record(False, False),
                },
                "8B": {
                    "QA1": judge_record(True, False),
                    "QA2": judge_record(True, True),
                },
            }.items():
                path = root / f"judge_{model}.json"
                path.write_text(
                    json.dumps({"bundle": {"paper": "bundle", "QA": records}}),
                    encoding="utf-8",
                )
                judge_paths[model] = path

            standard = root / "final_report.json"
            standard.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_name": "4B",
                                "pdf_summary": {
                                    "total_seen": 2,
                                    "legal_samples": 2,
                                    "answer_correct": 1,
                                    "page_correct": 1,
                                    "both_correct": 1,
                                },
                            },
                            {
                                "model_name": "8B",
                                "pdf_summary": {
                                    "total_seen": 2,
                                    "legal_samples": 2,
                                    "answer_correct": 2,
                                    "page_correct": 1,
                                    "both_correct": 1,
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "technical_report.md"
            output_json = root / "technical_report_data.json"
            payload = generate_report(
                dataset=dataset,
                dataset_summary=dataset_summary,
                judge_4b=judge_paths["4B"],
                judge_8b=judge_paths["8B"],
                final_report=standard,
                baseline_report=None,
                output=output,
                output_json=output_json,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output_json.is_file())
            self.assertIn("Cross-PDF QA v3", output.read_text(encoding="utf-8"))
            self.assertEqual(payload["models"][1]["overall"]["answer_correct"], 2)


if __name__ == "__main__":
    unittest.main()
