from __future__ import annotations

import unittest

from pku_qa.workflows.generation.build_cross_pdf_gemini_fix_rereview import (
    apply_complete_gemini_fix,
    mapped_doc_numbers,
)


class BuildGeminiFixRereviewTests(unittest.TestCase):
    def test_apply_complete_fix_preserves_unchanged_metadata(self) -> None:
        original = {
            "question": "old",
            "answer": "old answer",
            "evidence_pages": [1, 5],
            "modal_types": ["text"],
        }
        review = {
            "corrected_question": "new",
            "corrected_answer": "new answer",
            "corrected_evidence_pages": None,
        }
        fixed = apply_complete_gemini_fix(original, review)
        self.assertEqual(
            fixed,
            {
                "question": "new",
                "answer": "new answer",
                "evidence_pages": [1, 5],
                "modal_types": ["text"],
            },
        )
        self.assertEqual(original["question"], "old")

    def test_rejects_malformed_evidence_pages(self) -> None:
        original = {
            "question": "old",
            "answer": "old answer",
            "evidence_pages": [1, 5],
        }
        review = {
            "corrected_question": "new",
            "corrected_answer": "new answer",
            "corrected_evidence_pages": [{"doc": 1, "pages": [1]}],
        }
        self.assertIsNone(apply_complete_gemini_fix(original, review))

    def test_maps_pages_to_distinct_documents(self) -> None:
        sources = [
            {"merged_start_page": 1, "merged_end_page": 4},
            {"merged_start_page": 5, "merged_end_page": 9},
        ]
        self.assertEqual(mapped_doc_numbers([2, 7], sources), {1, 2})
        self.assertEqual(mapped_doc_numbers([2, 4], sources), {1})
        self.assertEqual(mapped_doc_numbers([2, "7"], sources), set())


if __name__ == "__main__":
    unittest.main()
