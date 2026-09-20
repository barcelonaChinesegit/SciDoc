from __future__ import annotations

import hashlib
import unittest
from argparse import Namespace

from eval_framework import is_numeric_match, parse_simple_numeric_answer
from run_judge import (
    direct_fill_match,
    build_judge_prompt,
    judge_fill,
    parse_structured_response,
    process_judge_paper,
    judge_row_matches_inference,
    should_process_existing,
)


class StrictNumericMatchTests(unittest.TestCase):
    def test_accepts_same_unitless_scalar(self) -> None:
        self.assertTrue(is_numeric_match("220", "220.0"))

    def test_accepts_same_scalar_and_normalized_unit(self) -> None:
        self.assertTrue(is_numeric_match("13.6 TeV", "13.6000 tev"))

    def test_accepts_same_percent_quantity(self) -> None:
        self.assertTrue(is_numeric_match("10%", "10.000%"))

    def test_accepts_scientific_notation(self) -> None:
        self.assertTrue(is_numeric_match("1e-4", "0.0001"))

    def test_rejects_missing_unit_on_one_side(self) -> None:
        self.assertFalse(is_numeric_match("13.6", "13.6 TeV"))

    def test_rejects_incompatible_units(self) -> None:
        self.assertFalse(is_numeric_match("10 ms", "10 s"))

    def test_rejects_ratio_even_when_one_component_matches(self) -> None:
        self.assertFalse(is_numeric_match("3:2", "5:2"))

    def test_rejects_formula_even_when_numbers_match(self) -> None:
        self.assertFalse(
            is_numeric_match(
                "min(max(0,E_normalized),2)",
                "min(max(0,x),1)",
            )
        )

    def test_rejects_range(self) -> None:
        self.assertFalse(is_numeric_match("10-20", "10-30"))

    def test_rejects_comparison(self) -> None:
        self.assertFalse(is_numeric_match("less than 0.10", "0.10"))

    def test_rejects_multiple_values(self) -> None:
        self.assertFalse(is_numeric_match("10 and 20", "10 and 30"))

    def test_rejects_long_explanation(self) -> None:
        self.assertFalse(
            is_numeric_match(
                "13.6",
                "The center-of-mass energy is 13.6 TeV in this experiment.",
            )
        )

    def test_rejects_percent_decimal_conversion_for_semantic_judge(self) -> None:
        self.assertFalse(is_numeric_match("0.1", "10%"))

    def test_parser_returns_normalized_quantity(self) -> None:
        self.assertEqual(parse_simple_numeric_answer("1,024 MB"), (1024.0, "mb"))


class DirectFillMatchTests(unittest.TestCase):
    def test_empty_prediction_is_directly_incorrect(self) -> None:
        self.assertEqual(
            direct_fill_match("answer", "  "),
            (False, "empty_prediction"),
        )

    def test_runtime_error_output_is_directly_incorrect(self) -> None:
        self.assertEqual(
            direct_fill_match("answer", "[ERROR: CUDA unavailable]"),
            (False, "error_output"),
        )

    def test_answerable_refusal_needs_semantic_judge(self) -> None:
        self.assertEqual(
            direct_fill_match("AdamW", "(Unanswerable)."),
            (None, "needs_llm_judge"),
        )

    def test_gold_unanswerable_requires_canonical_label(self) -> None:
        self.assertEqual(
            direct_fill_match("Unanswerable", "Unanswerable"),
            (True, "unanswerable_exact_label"),
        )
        self.assertEqual(
            direct_fill_match("Unanswerable", "Not mentioned in the paper"),
            (False, "unanswerable_exact_label"),
        )

    def test_normalized_equal_answer_requires_judge(self) -> None:
        self.assertEqual(
            direct_fill_match("Qwen3-VL-8B", "  qwen3-vl-8b\n"),
            (None, "needs_llm_judge"),
        )

    def test_loose_equal_answer_requires_judge(self) -> None:
        self.assertEqual(
            direct_fill_match("(AdamW)", "AdamW."),
            (None, "needs_llm_judge"),
        )

    def test_unsafe_numeric_answer_falls_through_to_llm(self) -> None:
        self.assertEqual(
            direct_fill_match("3:2", "5:2"),
            (None, "needs_llm_judge"),
        )


class StructuredSxzResponseTests(unittest.TestCase):
    def test_parses_answer_pre_and_evidence_pages(self) -> None:
        answer, pages, method = parse_structured_response(
            '{"answer_pre":"AdamW","evidence_pages":[3, 5]}'
        )
        self.assertEqual(answer, "AdamW")
        self.assertEqual(pages, [3, 5])
        self.assertEqual(method, "canonical_json")

    def test_parses_json_inside_markdown_fence(self) -> None:
        answer, pages, method = parse_structured_response(
            '```json\n{"answer_pre":"42","evidence_pages":[2, 4]}\n```'
        )
        self.assertEqual(answer, "")
        self.assertEqual(pages, [])
        self.assertEqual(method, "invalid_pdf_output")



class SelectiveRetryTests(unittest.TestCase):
    def test_retries_requested_match_method(self) -> None:
        self.assertTrue(
            should_process_existing(
                {"match_method": "numeric_match", "is_correct": True},
                retry_errors_only=False,
                retry_match_methods={"numeric_match"},
            )
        )


class EndToEndProtocolJudgeTests(unittest.TestCase):
    @staticmethod
    def stamp_raw_output(row: dict) -> None:
        raw = str(row["model_output"])
        row["raw_model_output"] = raw
        row["raw_model_output_sha256"] = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()
        row["deterministic_normalizations"] = []

    def test_unanswerable_rule_precedes_stale_mcq_metadata(self) -> None:
        gold = {
            "QA": {
                "Q": {
                    "question": "Can this be answered?",
                    "answer": "Unanswerable",
                    "evidence_pages": [],
                    "options": [
                        {"id": "A", "text": "yes"},
                        {"id": "B", "text": "no"},
                    ],
                }
            }
        }
        inference_qa = {
            "question": "Can this be answered?",
            "correct_answer": "Unanswerable",
            "reference_evidence_pages": [],
            "model_output": (
                '{"answer_pre":"Unanswerable","evidence_pages":[]}'
            ),
            "type": "mcq",
            "input_mode": "pdf",
            "require_structured_output": True,
            "require_evidence_pages": False,
            "prompt_style": "pdf",
            "shown_pdf_pages": [1],
        }
        self.stamp_raw_output(inference_qa)
        args = Namespace(
            retry_errors_only=False,
            worker_id="test",
            shard_id=0,
            max_judge_retries=1,
            max_new_tokens=8,
        )
        row = process_judge_paper(
            "P",
            {"QA": {"Q": inference_qa}},
            gold,
            None,
            judge_provider=None,
            args=args,
            retry_match_methods=set(),
        )["QA"]["Q"]
        self.assertTrue(row["answer_is_correct"])
        self.assertTrue(row["is_correct"])
        self.assertEqual(row["match_method"], "exact_unanswerable")

    def test_cached_judge_row_is_bound_to_raw_inference(self) -> None:
        inference = {
            "question": "Q?",
            "correct_answer": "A",
            "model_output": "A",
            "type": "fill",
            "input_mode": "question_only",
            "require_structured_output": False,
            "require_evidence_pages": False,
            "reference_evidence_pages": [],
        }
        existing = {
            **inference,
            "is_correct": True,
            "answer_is_correct": True,
            "evidence_pages_is_correct": True,
            "output_is_legal": True,
        }
        self.assertTrue(judge_row_matches_inference(existing, inference))
        changed = {**inference, "model_output": "B"}
        self.assertFalse(judge_row_matches_inference(existing, changed))

    def test_pdf_prediction_of_unshown_page_is_illegal(self) -> None:
        gold = {
            "QA": {
                "Q": {
                    "question": "Q?",
                    "answer": "A",
                    "evidence_pages": [1],
                }
            }
        }
        inference = {
            "QA": {
                "Q": {
                    "question": "Q?",
                    "correct_answer": "A",
                    "reference_evidence_pages": [1],
                    "model_output": (
                        '{"answer_pre":"A","evidence_pages":[3]}'
                    ),
                    "type": "fill",
                    "input_mode": "pdf",
                    "require_structured_output": True,
                    "require_evidence_pages": True,
                    "prompt_style": "pdf",
                    "shown_pdf_pages": [1, 2],
                }
            }
        }
        self.stamp_raw_output(inference["QA"]["Q"])
        args = Namespace(
            retry_errors_only=False,
            worker_id="test",
            shard_id=0,
            max_judge_retries=1,
            max_new_tokens=8,
        )
        row = process_judge_paper(
            "P",
            inference,
            gold,
            None,
            judge_provider=None,
            args=args,
            retry_match_methods=set(),
        )["QA"]["Q"]
        self.assertFalse(row["output_is_legal"])
        self.assertFalse(row["evidence_pages_is_correct"])
        self.assertFalse(row["is_correct"])

    def test_pdf_scores_answer_evidence_joint_and_illegal_independently(self) -> None:
        gold = {
            "QA": {
                "Q1": {
                    "question": "Answerable?",
                    "answer": "Alpha",
                    "evidence_pages": [1],
                },
                "Q2": {
                    "question": "Impossible?",
                    "answer": "Unanswerable",
                    "evidence_pages": [],
                },
            }
        }
        inference = {
            "QA": {
                "Q1": {
                    "question": "Answerable?",
                    "correct_answer": "Alpha",
                    "reference_evidence_pages": [1],
                    "model_output": '{"answer_pre":"Alpha","evidence_pages":[2]}',
                    "type": "fill",
                    "input_mode": "pdf",
                    "require_structured_output": True,
                    "require_evidence_pages": True,
                    "prompt_style": "pdf",
                    "shown_pdf_pages": [1, 2],
                },
                "Q2": {
                    "question": "Impossible?",
                    "correct_answer": "Unanswerable",
                    "reference_evidence_pages": [],
                    "model_output": '{"answer_pre":"Unanswerable","evidence_pages":[9]}',
                    "type": "fill",
                    "input_mode": "pdf",
                    "require_structured_output": True,
                    "require_evidence_pages": False,
                    "prompt_style": "pdf",
                    "shown_pdf_pages": [1, 9],
                },
            }
        }
        for row in inference["QA"].values():
            self.stamp_raw_output(row)
        args = Namespace(
            retry_errors_only=False,
            worker_id="test",
            shard_id=0,
            max_judge_retries=1,
            max_new_tokens=8,
        )
        class Provider:
            calls = []
            def generate(self, messages, max_tokens):
                self.calls.append(messages)
                return "CORRECT"
        provider = Provider()
        judged = process_judge_paper(
            "P",
            inference,
            gold,
            None,
            judge_provider=provider,
            args=args,
            retry_match_methods=set(),
        )["QA"]
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(judged["Q1"]["answer_is_correct"])
        self.assertFalse(judged["Q1"]["evidence_pages_is_correct"])
        self.assertFalse(judged["Q1"]["is_correct"])
        self.assertTrue(judged["Q1"]["output_is_legal"])
        self.assertFalse(judged["Q2"]["answer_is_correct"])
        self.assertFalse(judged["Q2"]["evidence_pages_is_correct"])
        self.assertFalse(judged["Q2"]["is_correct"])
        self.assertFalse(judged["Q2"]["output_is_legal"])

    def test_skips_other_match_method_during_selective_retry(self) -> None:
        self.assertFalse(
            should_process_existing(
                {"match_method": "llm_judge", "is_correct": False},
                retry_errors_only=False,
                retry_match_methods={"numeric_match"},
            )
        )

    def test_default_run_processes_missing_entry(self) -> None:
        self.assertTrue(
            should_process_existing(
                None,
                retry_errors_only=False,
                retry_match_methods=set(),
            )
        )

    def test_default_run_skips_completed_entry(self) -> None:
        self.assertFalse(
            should_process_existing(
                {
                    "match_method": "exact_normalized",
                    "is_correct": True,
                    "answer_is_correct": True,
                    "evidence_pages_is_correct": True,
                    "input_mode": "question_only",
                    "require_structured_output": False,
                    "require_evidence_pages": False,
                    "output_is_legal": True,
                },
                retry_errors_only=False,
                retry_match_methods=set(),
            )
        )

    def test_default_run_reprocesses_stale_answer_only_judge_entry(self) -> None:
        self.assertTrue(
            should_process_existing(
                {"match_method": "exact_normalized", "is_correct": True},
                retry_errors_only=False,
                retry_match_methods=set(),
            )
        )

    def test_retry_modes_are_independent(self) -> None:
        self.assertTrue(
            should_process_existing(
                {"match_method": "llm_judge_error", "judge_verdict": "ERROR"},
                retry_errors_only=True,
                retry_match_methods=set(),
            )
        )


if __name__ == "__main__":
    unittest.main()
