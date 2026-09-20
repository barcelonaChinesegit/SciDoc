from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from run_report import (
    PDF_SUMMARY_COLUMNS,
    analyze,
    build_pdf_table,
    illegal_rate_failures,
    validate_judge_against_gold,
    write_pdf_csv,
)


class PdfReportTests(unittest.TestCase):
    def _write(self, root: Path, name: str, qas: dict) -> Path:
        path = root / name
        path.write_text(
            json.dumps({"paper-1": {"paper": "paper-1", "QA": qas}}),
            encoding="utf-8",
        )
        return path

    def test_pdf_summary_matches_collaborator_columns_and_legal_denominator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write(
                root,
                "judge_4B.json",
                {
                    "q1": {
                        "type": "fill",
                        "model_output": '{"answer_pre":"a","evidence_pages":[2]}',
                        "parsed_answer": "a",
                        "predicted_evidence_pages": [2],
                        "output_parse_method": "json:answer_pre",
                        "require_evidence_pages": True,
                        "answer_is_correct": True,
                        "evidence_pages_is_correct": True,
                        "is_correct": True,
                        "match_method": "exact_normalized",
                    },
                    "q2": {
                        "type": "fill",
                        "model_output": '{"answer_pre":"b","evidence_pages":[3]}',
                        "parsed_answer": "b",
                        "predicted_evidence_pages": [3],
                        "output_parse_method": "json:answer_pre",
                        "require_evidence_pages": True,
                        "answer_is_correct": True,
                        "evidence_pages_is_correct": False,
                        "is_correct": False,
                        "match_method": "exact_normalized",
                    },
                    "q3": {
                        "type": "fill",
                        "model_output": "not json",
                        "parsed_answer": "",
                        "predicted_evidence_pages": [],
                        "output_parse_method": "invalid_json",
                        "require_evidence_pages": True,
                        "answer_is_correct": False,
                        "evidence_pages_is_correct": False,
                        "is_correct": False,
                        "match_method": "empty_prediction",
                    },
                },
            )
            result = analyze(path)
            summary = result["pdf_summary"]
            self.assertEqual(tuple(summary), PDF_SUMMARY_COLUMNS)
            self.assertEqual(summary["model_name"], "Qwen3VL_4B")
            self.assertEqual(summary["total_seen"], 3)
            self.assertEqual(summary["legal_samples"], 2)
            self.assertEqual(summary["skipped_illegal_answer"], 1)
            self.assertEqual(summary["answer_correct"], 2)
            self.assertEqual(summary["page_correct"], 1)
            self.assertEqual(summary["both_correct"], 1)
            self.assertAlmostEqual(summary["answer_acc"], 200 / 3)
            self.assertAlmostEqual(summary["evidence_page_acc"], 100 / 3)
            self.assertAlmostEqual(summary["both_acc"], 100 / 3)
            self.assertEqual(summary["answer_acc_legal"], 100.0)
            self.assertEqual(summary["evidence_page_acc_legal"], 50.0)
            self.assertEqual(summary["both_acc_legal"], 50.0)
            self.assertAlmostEqual(result["joint_accuracy"], 100 / 3)
            table = build_pdf_table([result])
            self.assertTrue(table.startswith("\t".join(PDF_SUMMARY_COLUMNS)))
            self.assertIn(
                "Qwen3VL_4B\t3\t2\t1\t33.33%\t3\t2\t2\t1\t1", table
            )
            csv_path = root / "report.csv"
            write_pdf_csv(csv_path, [result])
            self.assertIn("Qwen3VL_4B", csv_path.read_text(encoding="utf-8-sig"))

    def test_question_only_result_does_not_add_evidence_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                "judge_4B.json",
                {
                    "q1": {
                        "type": "fill",
                        "model_output": "a",
                        "parsed_answer": "a",
                        "require_evidence_pages": False,
                        "answer_is_correct": True,
                        "evidence_pages_is_correct": True,
                        "is_correct": True,
                        "match_method": "exact_normalized",
                    }
                },
            )
            result = analyze(path)
            self.assertNotIn("pdf_summary", result)
            self.assertEqual(build_pdf_table([result]), "")
            self.assertEqual(result["total_accuracy"], 100.0)

    def test_unanswerable_is_legal_without_evidence_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                Path(tmp),
                "judge_8B.json",
                {
                    "answerable": {
                        "type": "fill",
                        "parsed_answer": "answer",
                        "predicted_evidence_pages": [2],
                        "output_parse_method": "json:answer_pre",
                        "require_structured_output": True,
                        "require_evidence_pages": True,
                        "answer_is_correct": True,
                        "evidence_pages_is_correct": True,
                        "is_correct": True,
                    },
                    "unanswerable": {
                        "type": "fill",
                        "parsed_answer": "Unanswerable",
                        "predicted_evidence_pages": [],
                        "output_parse_method": "json:answer_pre",
                        "require_structured_output": True,
                        "require_evidence_pages": False,
                        "answer_is_correct": True,
                        "evidence_pages_is_correct": True,
                        "is_correct": True,
                    },
                },
            )
            result = analyze(path)
            summary = result["pdf_summary"]
            self.assertEqual(summary["legal_samples"], 2)
            self.assertEqual(summary["skipped_illegal_answer"], 0)
            self.assertEqual(summary["answer_correct"], 2)
            self.assertEqual(summary["page_correct"], 1)
            self.assertEqual(summary["both_correct"], 2)
            self.assertEqual(result["evidence_required_legal_samples"], 1)
            self.assertEqual(result["evidence_pages_accuracy"], 100.0)
            self.assertEqual(result["joint_accuracy"], 100.0)

    def test_illegal_output_gate_is_strictly_at_one_percent(self) -> None:
        def result(rate: float) -> dict:
            return {
                "model_name": "8B",
                "pdf_summary": {
                    "skipped_illegal_rate": rate,
                    "skipped_illegal_answer": int(rate * 1000),
                    "total_seen": 1000,
                },
            }

        self.assertEqual(illegal_rate_failures([result(0.01)], 0.01), [])
        self.assertTrue(illegal_rate_failures([result(0.011)], 0.01))

    def test_gold_validation_reparses_raw_output_and_rejects_tampering(self) -> None:
        gold = {
            "paper-1": {
                "QA": {
                    "q1": {
                        "question": "Q?",
                        "answer": "A",
                        "evidence_pages": [2],
                    }
                }
            }
        }
        row = {
            "question": "Q?",
            "correct_answer": "A",
            "model_output": '{"answer_pre":"A","evidence_pages":[2]}',
            "parsed_answer": "A",
            "predicted_evidence_pages": [3],
            "reference_evidence_pages": [2],
            "input_mode": "pdf",
            "require_structured_output": True,
            "require_evidence_pages": True,
            "shown_pdf_pages": [2],
            "output_is_legal": True,
            "answer_is_correct": True,
            "evidence_pages_is_correct": False,
            "is_correct": False,
            "publication_eligible": True,
            "protocol_validation": "publication_strict",
            "protocol_fingerprint": "inference-fingerprint",
            "judge_protocol_fingerprint": "judge-fingerprint",
            "qa_source_sha256": "qa-source",
            "pdf_corpus_sha256": "pdf-corpus",
            "pdf_sha256": "pdf",
            "evaluated_model": "4B",
        }
        row["raw_model_output"] = row["model_output"]
        row["raw_model_output_sha256"] = hashlib.sha256(
            row["raw_model_output"].encode("utf-8")
        ).hexdigest()
        row["deterministic_normalizations"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), "judge_4B.json", {"q1": row})
            with self.assertRaisesRegex(ValueError, "parsed evidence mismatch"):
                validate_judge_against_gold(path, gold)


if __name__ == "__main__":
    unittest.main()
