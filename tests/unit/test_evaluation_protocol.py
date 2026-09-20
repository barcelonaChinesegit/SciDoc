from __future__ import annotations

import unittest
from argparse import Namespace

from evaluation_protocol import (
    configure_protocol,
    canonicalize_pdf_output_for_storage,
    parse_canonical_pdf_output,
    protocol_for_item,
    protocol_from_inference_record,
    validate_dataset_protocol,
)


class EvaluationProtocolTests(unittest.TestCase):
    def test_pdf_cli_always_enables_canonical_structured_output(self) -> None:
        args = Namespace(
            input_mode="pdf",
            page_input_policy="full",
        )
        configure_protocol(args)
        self.assertTrue(args.require_evidence_pages)
        self.assertTrue(args.pdf_mode)

    def test_question_only_derives_plain_answer_protocol(self) -> None:
        args = Namespace(
            input_mode="question_only",
            page_input_policy="full",
        )
        configure_protocol(args)
        self.assertFalse(args.require_evidence_pages)
        self.assertFalse(args.pdf_mode)

    def test_unanswerable_pdf_is_structured_but_has_no_evidence_target(self) -> None:
        protocol = protocol_for_item("pdf", "Unanswerable")
        self.assertTrue(protocol.require_structured_output)
        self.assertFalse(protocol.require_evidence_pages)

    def test_pdf_dataset_requires_gold_evidence_for_answerable_items(self) -> None:
        dataset = {
            "p": {"QA": {"q": {"question": "Q?", "answer": "A"}}}
        }
        with self.assertRaisesRegex(ValueError, "no gold evidence_pages"):
            validate_dataset_protocol(dataset, "pdf")

    def test_gold_dataset_requires_one_string_answer_field(self) -> None:
        for qa in (
            {"question": "Q?", "correct_answer": "A", "evidence_pages": [1]},
            {"question": "Q?", "answer": None, "evidence_pages": [1]},
        ):
            with self.subTest(qa=qa):
                with self.assertRaisesRegex(ValueError, "answer"):
                    validate_dataset_protocol(
                        {"p": {"QA": {"q": qa}}}, "pdf"
                    )

    def test_gold_dataset_rejects_null_question(self) -> None:
        dataset = {
            "p": {
                "QA": {
                    "q": {
                        "question": None,
                        "answer": "A",
                        "evidence_pages": [1],
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            validate_dataset_protocol(dataset, "pdf")

    def test_dataset_accepts_exact_unanswerable_with_empty_pages(self) -> None:
        dataset = {
            "p": {
                "QA": {
                    "q": {
                        "question": "Q?",
                        "answer": "Unanswerable",
                        "evidence_pages": [],
                    }
                }
            }
        }
        profile = validate_dataset_protocol(dataset, "pdf")
        self.assertEqual(profile["unanswerable"], 1)

    def test_dataset_rejects_noncanonical_unanswerable_spelling(self) -> None:
        dataset = {
            "p": {
                "QA": {
                    "q": {
                        "question": "Q?",
                        "answer": "unanswerable",
                        "evidence_pages": [],
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "must be exactly"):
            validate_dataset_protocol(dataset, "pdf")

    def test_dataset_accepts_more_than_eight_gold_evidence_pages(self) -> None:
        dataset = {
            "p": {
                "QA": {
                    "q": {
                        "question": "Q?",
                        "answer": "A",
                        "evidence_pages": list(range(1, 10)),
                    }
                }
            }
        }
        profile = validate_dataset_protocol(dataset, "pdf")
        self.assertEqual(profile["evidence_references"], 9)

    def test_pdf_parser_does_not_turn_large_evidence_set_into_format_error(self) -> None:
        answer, pages = parse_canonical_pdf_output(
            '{"answer_pre":"A","evidence_pages":[1,2,3,4,5,6,7,8,9,10,11]}'
        )
        self.assertEqual(answer, "A")
        self.assertEqual(pages, list(range(1, 12)))

    def test_pdf_parser_rejects_pages_not_shown_to_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "not shown"):
            parse_canonical_pdf_output(
                '{"answer_pre":"A","evidence_pages":[4]}',
                allowed_pages=[1, 2, 3],
            )

    def test_pdf_parser_canonicalizes_page_set_order_without_changing_set(
        self,
    ) -> None:
        raw = '{"answer_pre":"A","evidence_pages":[7,6]}'
        answer, pages = parse_canonical_pdf_output(
            raw, allowed_pages=[1, 2, 3, 4, 5, 6, 7]
        )
        canonical, normalizations = canonicalize_pdf_output_for_storage(
            raw, allowed_pages=[1, 2, 3, 4, 5, 6, 7]
        )
        self.assertEqual((answer, pages), ("A", [6, 7]))
        self.assertEqual(
            canonical,
            '{"answer_pre":"A","evidence_pages":[6,7]}',
        )
        self.assertEqual(normalizations, ["evidence_pages_sorted"])

    def test_inference_record_rejects_old_pdf_answer_only_run(self) -> None:
        record = {
            "input_mode": "pdf",
            "correct_answer": "A",
            "reference_evidence_pages": [3],
            "require_structured_output": False,
            "require_evidence_pages": False,
            "prompt_style": "default",
        }
        with self.assertRaisesRegex(ValueError, "require_structured_output=True"):
            protocol_from_inference_record(record)

    def test_pdf_parser_rejects_markdown_and_schema_aliases(self) -> None:
        for output in (
            '```json\n{"answer_pre":"A","evidence_pages":[1]}\n```',
            '{"answer":"A","evidence_pages":[1]}',
            '{"answer_pre":"A","evidence_pages":"1"}',
        ):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    parse_canonical_pdf_output(output)

    def test_pdf_parser_enforces_unanswerable_empty_pages(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence_pages=\\[\\]"):
            parse_canonical_pdf_output(
                '{"answer_pre":"Unanswerable","evidence_pages":[9]}'
            )


if __name__ == "__main__":
    unittest.main()
