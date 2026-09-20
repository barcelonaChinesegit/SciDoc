from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src/pku_qa/workflows/review/review_cross_pdf_qa_api.py"
)
SPEC = importlib.util.spec_from_file_location("review_cross_pdf_qa_api", MODULE_PATH)
assert SPEC and SPEC.loader
review_api = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = review_api
SPEC.loader.exec_module(review_api)


class ReviewApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        with review_api._API_KEY_STATE_LOCK:
            review_api._ACTIVE_API_KEYS = ()
            review_api._EXHAUSTED_API_KEY_PROFILES.clear()

    def valid_review(self) -> dict:
        return {
            "bundle_id": "xb_test",
            "bundle_summary": "A coherent test bundle.",
            "items": [
                {
                    "qa_id": "QA1",
                    "decision": "KEEP",
                    "criteria": {name: True for name in review_api.CRITERIA},
                    "confidence": 0.95,
                    "severity": "none",
                    "reason": "All claims are supported.",
                    "support": [],
                    "corrected_question": None,
                    "corrected_answer": None,
                    "corrected_evidence_pages": None,
                }
            ],
        }

    def test_valid_unicode_replaces_lone_surrogate(self) -> None:
        cleaned = review_api.valid_unicode("left\ud800right")
        self.assertEqual(cleaned, "left?right")

    def test_load_key_selects_classmate_profile_without_exposing_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / ".env"
            key_file.write_text(
                "AICODEMIRROR_API_KEY=school-test-key\n"
                "AICODEMIRROR_API_KEY_CLASSMATE=classmate-test-key\n"
                "AICODEMIRROR_API_KEY_PROFILE=classmate\n",
                encoding="utf-8",
            )
            with patch.dict(review_api.os.environ, {}, clear=True):
                with patch("builtins.print") as print_mock:
                    key = review_api.load_key(key_file)
        self.assertEqual(key, "classmate-test-key")
        self.assertEqual(review_api._ACTIVE_API_KEYS, (("classmate", key),))
        printed = " ".join(str(call) for call in print_mock.call_args_list)
        self.assertIn("classmate", printed)
        self.assertNotIn(key, printed)

    def test_auto_profile_switches_only_after_quota_exhaustion(self) -> None:
        class Response:
            def __init__(self, status_code: int, text: str = "") -> None:
                self.status_code = status_code
                self.text = text

        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / ".env"
            key_file.write_text(
                "AICODEMIRROR_API_KEY=school-test-key\n"
                "AICODEMIRROR_API_KEY_CLASSMATE=classmate-test-key\n"
                "AICODEMIRROR_API_KEY_PROFILE=auto\n",
                encoding="utf-8",
            )
            with patch.dict(review_api.os.environ, {}, clear=True):
                primary_key = review_api.load_key(key_file)
            with patch.object(
                review_api.requests,
                "post",
                side_effect=[
                    Response(429, "insufficient_quota"),
                    Response(200),
                ],
            ) as post:
                response = review_api.post_with_retries(
                    "https://example.invalid",
                    {"x-api-key": primary_key},
                    {},
                    timeout=1,
                    retries=1,
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args_list[0].kwargs["headers"]["x-api-key"],
            "school-test-key",
        )
        self.assertEqual(
            post.call_args_list[1].kwargs["headers"]["x-api-key"],
            "classmate-test-key",
        )

    def test_parallel_worker_argv_replaces_parent_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "worker.txt"
            result = review_api.single_api_worker_argv(
                [
                    "--provider",
                    "claude",
                    "--api-workers=8",
                    "--bundle-id",
                    "xb_1",
                    "--bundle-list=parent.txt",
                    "--sample",
                    "10",
                ],
                3,
                shard,
            )
        self.assertEqual(
            result,
            [
                "--provider",
                "claude",
                "--api-workers",
                "1",
                "--worker-id",
                "cross-review-api-3",
                "--bundle-list",
                str(shard),
            ],
        )

    def test_parse_json_response_accepts_fenced_json(self) -> None:
        parsed = review_api.parse_json_response(
            '```json\n{"bundle_id":"xb_test","items":[]}\n```'
        )
        self.assertEqual(parsed["bundle_id"], "xb_test")

    def test_normalizes_plain_integer_page_string(self) -> None:
        review = self.valid_review()
        review["items"][0]["decision"] = "FIX"
        review["items"][0]["corrected_evidence_pages"] = "5, 24, 36"
        normalizations = review_api.normalize_review_types(review)
        self.assertEqual(
            review["items"][0]["corrected_evidence_pages"], [5, 24, 36]
        )
        self.assertEqual(len(normalizations), 1)

    def test_does_not_normalize_page_ranges_or_objects(self) -> None:
        review = self.valid_review()
        review["items"][0]["corrected_evidence_pages"] = "5-7"
        self.assertEqual(review_api.normalize_review_types(review), [])
        self.assertEqual(review["items"][0]["corrected_evidence_pages"], "5-7")

    def test_validate_review_accepts_complete_record(self) -> None:
        errors = review_api.validate_review(
            self.valid_review(), "xb_test", ["QA1"]
        )
        self.assertEqual(errors, [])

    def test_validate_review_rejects_wrong_id_and_invalid_decision(self) -> None:
        review = self.valid_review()
        review["items"][0]["qa_id"] = "QA2"
        review["items"][0]["decision"] = "MAYBE"
        errors = review_api.validate_review(review, "xb_test", ["QA1"])
        self.assertTrue(any("qa_id set mismatch" in error for error in errors))
        self.assertTrue(any("invalid decision" in error for error in errors))

    def test_validate_review_allows_different_item_order(self) -> None:
        review = self.valid_review()
        second = dict(review["items"][0])
        second["qa_id"] = "QA2"
        review["items"] = [second, review["items"][0]]
        errors = review_api.validate_review(review, "xb_test", ["QA1", "QA2"])
        self.assertEqual(errors, [])

    def test_validate_review_rejects_duplicate_ids(self) -> None:
        review = self.valid_review()
        review["items"] = [review["items"][0], dict(review["items"][0])]
        errors = review_api.validate_review(review, "xb_test", ["QA1", "QA2"])
        self.assertTrue(any("duplicate qa_ids" in error for error in errors))
        self.assertTrue(any("qa_id set mismatch" in error for error in errors))

    def test_fix_requires_at_least_one_corrected_field(self) -> None:
        review = self.valid_review()
        review["items"][0]["decision"] = "FIX"
        errors = review_api.validate_review(review, "xb_test", ["QA1"])
        self.assertEqual(errors, ["QA1: FIX has no corrected field"])

    def test_fix_allows_unchanged_fields_as_null(self) -> None:
        review = self.valid_review()
        review["items"][0]["decision"] = "FIX"
        review["items"][0]["corrected_evidence_pages"] = [1, 9]
        errors = review_api.validate_review(review, "xb_test", ["QA1"])
        self.assertEqual(errors, [])

    def test_rejects_structured_corrected_evidence_pages(self) -> None:
        review = self.valid_review()
        review["items"][0]["decision"] = "FIX"
        review["items"][0]["corrected_evidence_pages"] = [
            {"doc_number": 1, "pages": [1]}
        ]
        errors = review_api.validate_review(review, "xb_0001", ["QA1"])
        self.assertIn("QA1: invalid corrected_evidence_pages", errors)

    def test_rejects_malformed_support(self) -> None:
        review = self.valid_review()
        review["items"][0]["support"] = [
            {"doc_number": "1", "merged_page": 0, "supported_fact": ""}
        ]
        errors = review_api.validate_review(review, "xb_0001", ["QA1"])
        self.assertIn("QA1: support[0] has invalid doc_number", errors)
        self.assertIn("QA1: support[0] has invalid merged_page", errors)
        self.assertIn("QA1: support[0] has empty supported_fact", errors)

    def test_expected_schema_preserves_qa_order(self) -> None:
        schema = review_api.expected_schema("xb_test", ["QA7", "QA2"])
        self.assertEqual(
            [item["qa_id"] for item in schema["items"]], ["QA7", "QA2"]
        )

    def test_hard_candidate_payload_preserves_reasoning_ledger(self) -> None:
        payload = review_api.qa_payload(
            "xb_test",
            {
                "QA": {
                    "QA1": {
                        "question": "How do the modules interact?",
                        "answer": "One feeds the next.",
                        "source_doc_numbers": [1, 2, 3],
                        "reasoning_type": "adjacent_module_dependency",
                        "intermediate_facts": [{"doc_numbers": [1]}],
                    }
                }
            },
        )
        self.assertEqual(payload[0]["source_doc_numbers"], [1, 2, 3])
        self.assertEqual(
            payload[0]["reasoning_type"], "adjacent_module_dependency"
        )

    def test_hard_policy_allows_explicit_cross_paper_constraints(self) -> None:
        self.assertIn(
            "source papers do not need to",
            review_api.HARD_REVIEW_PROMPT,
        )
        self.assertIn(
            "deterministically yield one compatibility decision",
            review_api.HARD_REVIEW_PROMPT,
        )
        self.assertIn(
            "treat an explicit constraint in the question as a legitimate benchmark",
            review_api.HARD_REVIEW_PROMPT.replace("\n", " "),
        )

    def test_evidence_context_expands_without_crossing_doc_boundaries(self) -> None:
        sources = [
            review_api.Source(1, "One", "p1", Path("p1.pdf"), 1, 10),
            review_api.Source(2, "Two", "p2", Path("p2.pdf"), 11, 20),
        ]
        pages = review_api.selected_context_pages(
            {"QA": {"Q1": {"evidence_pages": [10, 11]}}},
            sources,
            context_pages=1,
        )
        self.assertEqual(pages, {9, 10, 11, 12})

    def test_cloudflare_524_is_retried(self) -> None:
        class Response:
            def __init__(self, status_code: int, text: str = "") -> None:
                self.status_code = status_code
                self.text = text

        with (
            patch.object(
                review_api.requests,
                "post",
                side_effect=[Response(524, "timeout"), Response(200)],
            ) as post,
            patch.object(review_api.time, "sleep"),
        ):
            response = review_api.post_with_retries(
                "https://example.invalid",
                {},
                {},
                timeout=1,
                retries=2,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)

    def test_non_retryable_401_fails_immediately(self) -> None:
        class Response:
            status_code = 401
            text = "unauthorized"

        with patch.object(review_api.requests, "post", return_value=Response()) as post:
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                review_api.post_with_retries(
                    "https://example.invalid",
                    {},
                    {},
                    timeout=1,
                    retries=4,
                )
        self.assertEqual(post.call_count, 1)

    def test_account_quota_exhaustion_is_not_retried(self) -> None:
        class Response:
            status_code = 429
            text = '{"error":{"type":"insufficient_quota","message":"check billing"}}'

        with patch.object(review_api.requests, "post", return_value=Response()) as post:
            with self.assertRaises(review_api.ApiQuotaExhaustedError):
                review_api.post_with_retries(
                    "https://example.invalid",
                    {},
                    {},
                    timeout=1,
                    retries=6,
                )
        self.assertEqual(post.call_count, 1)

    def test_plain_rate_limit_remains_retryable(self) -> None:
        class Response:
            def __init__(self, status_code: int, text: str = "") -> None:
                self.status_code = status_code
                self.text = text

        with (
            patch.object(
                review_api.requests,
                "post",
                side_effect=[
                    Response(429, "requests per minute exceeded"),
                    Response(200),
                ],
            ) as post,
            patch.object(review_api.time, "sleep"),
        ):
            response = review_api.post_with_retries(
                "https://example.invalid",
                {},
                {},
                timeout=1,
                retries=2,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
