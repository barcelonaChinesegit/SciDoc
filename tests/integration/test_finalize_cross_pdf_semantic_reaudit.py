from __future__ import annotations

import unittest

from pku_qa.workflows.cleaning.finalize_cross_pdf_semantic_reaudit import (
    all_criteria_true,
    apply_review_fix,
    find_near_duplicate_questions,
    structurally_valid_item,
    support_doc_numbers,
)


class FinalizeSemanticReauditTests(unittest.TestCase):
    def test_all_criteria_must_be_true(self) -> None:
        criteria = {
            "cross_document_required": True,
            "question_coherent": True,
            "question_specific": True,
            "answer_correct": True,
            "answer_complete": True,
            "evidence_sufficient": True,
            "no_unsupported_claim": True,
        }
        self.assertTrue(all_criteria_true({"criteria": criteria}))
        criteria["answer_complete"] = False
        self.assertFalse(all_criteria_true({"criteria": criteria}))

    def test_apply_review_fix_changes_only_non_null_fields(self) -> None:
        item = {
            "question": "Q",
            "answer": "A",
            "evidence_pages": [1, 5],
        }
        review = {
            "decision": "FIX",
            "corrected_question": None,
            "corrected_answer": "A2",
            "corrected_evidence_pages": [],
        }
        fixed, fields = apply_review_fix(item, review)
        self.assertEqual(fixed["question"], "Q")
        self.assertEqual(fixed["answer"], "A2")
        self.assertEqual(fixed["evidence_pages"], [1, 5])
        self.assertEqual(fields, ["answer"])

    def test_structural_validation_requires_two_papers(self) -> None:
        sources = [
            {"merged_start_page": 1, "merged_end_page": 4},
            {"merged_start_page": 5, "merged_end_page": 9},
        ]
        valid, docs, error = structurally_valid_item(
            {"question": "Q", "answer": "A", "evidence_pages": [2, 8]},
            sources,
        )
        self.assertTrue(valid)
        self.assertEqual(docs, [1, 2])
        self.assertIsNone(error)
        valid, docs, error = structurally_valid_item(
            {"question": "Q", "answer": "A", "evidence_pages": [2, 3]},
            sources,
        )
        self.assertFalse(valid)
        self.assertEqual(docs, [1])
        self.assertEqual(error, "evidence_spans_fewer_than_two_papers")

    def test_support_doc_numbers_ignores_non_integer_values(self) -> None:
        self.assertEqual(
            support_doc_numbers(
                {
                    "support": [
                        {"doc_number": 1},
                        {"doc_number": 2},
                        {"doc_number": "3"},
                    ]
                }
            ),
            {1, 2},
        )

    def test_detects_near_duplicate_questions(self) -> None:
        release = {
            "xb_1": {
                "QA": {
                    "QA1": {
                        "question": "How do methods in Doc 1 and Doc 2 differ?"
                    }
                }
            },
            "xb_2": {
                "QA": {
                    "QA1": {
                        "question": "How do methods in Doc 1 and Doc 3 differ?"
                    }
                }
            },
        }
        pairs = find_near_duplicate_questions(release)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["left"]["bundle_id"], "xb_1")


if __name__ == "__main__":
    unittest.main()
