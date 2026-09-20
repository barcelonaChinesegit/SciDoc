from __future__ import annotations

from argparse import Namespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation_protocol import (
    STRICT_INFERENCE_REQUIRED_QA_FIELDS,
    build_judge_queue_contract,
    pdf_corpus_sha256,
    pdf_corpus_sha256_from_manifest,
    pdf_sha256_manifest,
)
from run_hard_eval import inference_protocol_metadata
from run_inference import protocol_metadata_for_run


class ProtocolFingerprintTests(unittest.TestCase):
    def test_strict_inference_contract_keeps_raw_and_canonical_audit(self) -> None:
        self.assertTrue(
            {
                "raw_model_output",
                "raw_model_output_sha256",
                "deterministic_normalizations",
            }
            <= set(STRICT_INFERENCE_REQUIRED_QA_FIELDS)
        )

    def test_scheduler_and_inference_worker_build_identical_fingerprint(
        self,
    ) -> None:
        runtime_identity = {
            "provider_name": "local_qwen3_vl_4b_pdf",
            "provider_spec": {"enable_thinking": False},
            "assets": {"model_path": {"manifest_sha256": "model"}},
            "package_versions": {"transformers": "test"},
        }
        parent = Namespace(
            qa_source_sha256="qa",
            pdf_corpus_sha256="pdf",
            input_mode="pdf",
            page_input_policy="full",
            qa_page_field="input_pages",
            pdf_mode=True,
            inference_max_new_tokens=512,
            max_qa_retries=5,
            dpi=144,
            max_total_pdf_megapixels=0.0,
            min_auto_dpi=72,
            max_pdf_pages=0,
            page_selection_seed=20260713,
            provider_config_sha256=None,
        )
        worker = Namespace(
            provider_config=None,
            input_mode="pdf",
            page_input_policy="full",
            qa_page_field="input_pages",
            pdf_mode=True,
            max_new_tokens=512,
            max_qa_retries=5,
            dpi=144,
            max_total_pdf_megapixels=0.0,
            min_auto_dpi=72,
            max_pdf_pages=0,
            page_selection_seed=20260713,
        )
        with patch(
            "run_hard_eval.resolved_provider_identity",
            return_value=runtime_identity,
        ):
            parent_metadata = inference_protocol_metadata(parent, "4B")
        with (
            patch(
                "run_inference.load_provider_specs",
                return_value={"local_qwen3_vl_4b_pdf": {}},
            ),
            patch(
                "run_inference.provider_runtime_identity",
                return_value=runtime_identity,
            ),
        ):
            worker_metadata = protocol_metadata_for_run(
                worker,
                model_name="4B",
                provider_name="local_qwen3_vl_4b_pdf",
                qa_source_sha256="qa",
                pdf_corpus_hash="pdf",
            )
        self.assertEqual(
            parent_metadata["protocol_fingerprint"],
            worker_metadata["protocol_fingerprint"],
        )
        self.assertEqual(parent_metadata["config"], worker_metadata["config"])

    def test_scheduler_and_judge_worker_share_one_queue_contract(self) -> None:
        kwargs = {
            "input_mode": "pdf",
            "judge_runner_path": Path("src/pku_qa/evaluation/run_judge.py"),
            "gold_source_sha256": "gold",
            "inference_protocol_fingerprint": "inference",
            "judge_provider_identity": {"provider_name": "judge"},
            "judge_config": {
                "judge_provider_name": "judge",
                "max_new_tokens": 128,
                "max_judge_retries": 3,
            },
        }
        parent_contract = build_judge_queue_contract(**kwargs)
        worker_contract = build_judge_queue_contract(**kwargs)
        self.assertEqual(parent_contract, worker_contract)

    def test_pdf_corpus_hash_is_derived_from_per_paper_manifest(self) -> None:
        dataset = {"b": {"QA": {}}, "a": {"QA": {}}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.pdf").write_bytes(b"pdf-a")
            (root / "b.pdf").write_bytes(b"pdf-b")
            manifest = pdf_sha256_manifest(dataset, [root])
            self.assertEqual(set(manifest), {"a", "b"})
            self.assertEqual(
                pdf_corpus_sha256(dataset, [root]),
                pdf_corpus_sha256_from_manifest(manifest),
            )


if __name__ == "__main__":
    unittest.main()
