from __future__ import annotations

import unittest

from qa_scoring import anls, split_list, typed_answer_match


class TypedAnswerScoringTests(unittest.TestCase):
    def test_unanswerable_requires_canonical_label(self) -> None:
        self.assertTrue(
            typed_answer_match(
                "Unanswerable", "Unanswerable", answer_format="Unanswerable"
            )[0]
        )
        self.assertFalse(
            typed_answer_match(
                "Unanswerable", "unanswerable", answer_format="Unanswerable"
            )[0]
        )
        self.assertFalse(
            typed_answer_match(
                "Unanswerable",
                "Not mentioned in the document",
                answer_format="Unanswerable",
            )[0]
        )

    def test_integer_requires_exact_value(self) -> None:
        self.assertEqual(
            typed_answer_match("42", "42.0", answer_format="Integer")[0],
            True,
        )
        self.assertEqual(
            typed_answer_match("42", "43", answer_format="Integer")[0],
            False,
        )

    def test_float_uses_relative_tolerance(self) -> None:
        self.assertEqual(
            typed_answer_match(
                "100.0",
                "100.9",
                answer_format="Float",
                tolerance={"type": "relative", "value": 0.01},
            )[0],
            True,
        )
        self.assertEqual(
            typed_answer_match(
                "100.0",
                "101.1",
                answer_format="Float",
                tolerance={"type": "relative", "value": 0.01},
            )[0],
            False,
        )

    def test_numeric_unit_must_match(self) -> None:
        decision, method, _ = typed_answer_match(
            "13.6 TeV",
            "13.6 GeV",
            answer_format="Float",
            expected_unit="tev",
        )
        self.assertFalse(decision)
        self.assertEqual(method, "typed_numeric_unit_mismatch")

    def test_alias_is_accepted(self) -> None:
        decision, method, _ = typed_answer_match(
            "area under the curve",
            "AUC",
            answer_format="String",
            aliases=["AUC"],
        )
        self.assertTrue(decision)
        self.assertEqual(method, "typed_string_exact_or_alias")

    def test_unknown_string_falls_through_to_llm(self) -> None:
        self.assertIsNone(
            typed_answer_match(
                "AdamW",
                "the Adam optimizer",
                answer_format="String",
            )[0]
        )

    def test_list_requires_same_length(self) -> None:
        self.assertFalse(
            typed_answer_match(
                "precision, recall, F1",
                "precision, recall",
                answer_format="List",
            )[0]
        )

    def test_list_accepts_order_independent_exact_items(self) -> None:
        self.assertTrue(
            typed_answer_match(
                "precision, recall, F1",
                "F1, precision, recall",
                answer_format="List",
            )[0]
        )

    def test_anls_thresholds_distant_strings(self) -> None:
        self.assertEqual(anls("abcdef", "uvwxyz"), 0.0)

    def test_split_list_supports_json(self) -> None:
        self.assertEqual(split_list('["a", "b"]'), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
