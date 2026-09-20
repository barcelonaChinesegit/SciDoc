from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from eval_framework import DEFAULT_PROVIDER_SPECS
from evaluation_protocol import (
    JUDGE_INFERENCE_BINDING_FIELDS,
    build_inference_protocol_metadata,
    build_judge_queue_contract,
    judge_inference_binding_sha256,
)
from pypdf import PdfWriter
from run_inference import PDF_SYSTEM_PROMPT, build_fill_prompt
from pku_qa.workflows.reporting.run_single_pdf_challenge_gpu_queue import (
    FULL_PDF_PIXEL_BUDGET_MP,
    build_stage_command,
    build_stages,
    file_sha256,
    run_checked_with_heartbeat,
    judged_model_qa_progress,
    model_qa_pipeline_step_progress,
    node_toolchain_env,
    resolve_npm_command,
    stage_work_snapshot,
    validate_single_pdf_inputs,
)
from pku_qa.workflows.reporting.report_single_pdf_challenge_eval import (
    audit_pdf_corpus,
    build_ablation_expectations,
    eval_rows,
    gold_index,
    normalized_queue_source_sha256,
    sha256,
    validate_eval_provenance,
)
from pku_qa.workflows.selection.single_pdf_release_views import (
    ORDINARY_VIEW,
    UNANSWERABLE_VIEW,
    write_runtime_inputs,
)


ROOT = Path(__file__).resolve().parents[2]


class SinglePdfChallengeV2Tests(unittest.TestCase):
    def test_report_entrypoint_imports_project_modules_when_run_as_script(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pku_qa.workflows.reporting.report_single_pdf_challenge_eval",
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output-dir", completed.stdout)

    @staticmethod
    def _pdf_audit(page_count: int = 3) -> dict:
        return {
            "pdf_corpus_sha256": "fixture-corpus",
            "paper_count": 1,
            "papers": {
                "1": {
                    "path": "/fixture/1.pdf",
                    "sha256": "fixture-pdf",
                    "total_pdf_pages": page_count,
                }
            },
        }

    @staticmethod
    def _gold_dataset(two_rows: bool = False) -> dict:
        qas = {
            "QA1": {
                "question": "Question?",
                "answer": "Answer",
                "evidence_pages": [2],
                "oracle_pages": [2],
            }
        }
        if two_rows:
            qas["QA2"] = {
                "question": "Second question?",
                "answer": "Second answer",
                "evidence_pages": [3],
                "oracle_pages": [3],
            }
        return {"1": {"QA": qas}}

    @staticmethod
    def _pdf_judged_row(
        *,
        question: str = "Question?",
        answer: str = "Answer",
        reference_pages: list[int] | None = None,
        predicted_pages: list[int] | None = None,
        shown_pages: list[int] | None = None,
        page_input_policy: str = "full",
    ) -> dict:
        reference_pages = [2] if reference_pages is None else reference_pages
        predicted_pages = [2] if predicted_pages is None else predicted_pages
        shown_pages = [1, 2, 3] if shown_pages is None else shown_pages
        evidence_correct = predicted_pages == reference_pages
        return {
            "question": question,
            "correct_answer": answer,
            "model_output": json.dumps(
                {"answer_pre": answer, "evidence_pages": predicted_pages}
            ),
            "parsed_answer": answer,
            "reference_evidence_pages": reference_pages,
            "predicted_evidence_pages": predicted_pages,
            "input_mode": "pdf",
            "page_input_policy": page_input_policy,
            "shown_pdf_pages": shown_pages,
            "require_structured_output": True,
            "require_evidence_pages": answer != "Unanswerable",
            "output_is_legal": True,
            "answer_is_correct": True,
            "evidence_pages_is_correct": evidence_correct,
            "is_correct": evidence_correct,
        }

    @staticmethod
    def _write_strict_full_provenance_fixture(root: Path) -> dict:
        source = {
            "1": {
                "QA": {
                    "QA1": {
                        "question": "Question?",
                        "answer": "Answer",
                        "type": "fill",
                        "evidence_pages": [2],
                        "oracle_pages": [2],
                    }
                }
            }
        }
        source_path = root / "source.json"
        source_path.write_text(json.dumps(source), encoding="utf-8")
        pdf_dir = root / "pdfs"
        pdf_dir.mkdir()
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=72, height=72)
        with (pdf_dir / "1.pdf").open("wb") as handle:
            writer.write(handle)
        pdf_audit = audit_pdf_corpus(source, [pdf_dir])
        source_hash = sha256(source_path)
        protocol_metadata = build_inference_protocol_metadata(
            {
                "qa_source_sha256": source_hash,
                "pdf_corpus_sha256": pdf_audit["pdf_corpus_sha256"],
                "input_mode": "pdf",
                "page_input_policy": "full",
                "qa_page_field": "input_pages",
                "model": "4B",
                "prompt_style": "pdf",
                "pdf_mode": True,
                "max_pdf_pages": 0,
            }
        )
        fingerprint = protocol_metadata["protocol_fingerprint"]
        raw_output = json.dumps(
            {"answer_pre": "Answer", "evidence_pages": [2]}
        )
        inference_row = {
            "question": "Question?",
            "correct_answer": "Answer",
            "model_output": raw_output,
            "type": "fill",
            "input_mode": "pdf",
            "require_structured_output": True,
            "require_evidence_pages": True,
            "reference_evidence_pages": [2],
            "page_input_policy": "full",
            "prompt_style": "pdf",
            "shown_pdf_pages": [1, 2, 3],
            "total_pdf_pages": 3,
            "max_pdf_pages": 0,
            "protocol_fingerprint": fingerprint,
            "qa_source_sha256": source_hash,
            "pdf_corpus_sha256": pdf_audit["pdf_corpus_sha256"],
            "pdf_sha256": pdf_audit["papers"]["1"]["sha256"],
        }
        inferred = {"1": {"QA": {"QA1": inference_row}}}
        _, judge_metadata = build_judge_queue_contract(
            input_mode="pdf",
            judge_runner_path=ROOT / "src/pku_qa/evaluation/run_judge.py",
            gold_source_sha256=source_hash,
            inference_protocol_fingerprint=fingerprint,
            judge_provider_identity={"fixture": True},
            judge_config={
                "judge_provider_name": "fixture",
                "max_new_tokens": 32,
                "max_judge_retries": 3,
            },
        )
        judge_fingerprint = judge_metadata["judge_protocol_fingerprint"]
        judge_row = {
            field: inference_row.get(field)
            for field in JUDGE_INFERENCE_BINDING_FIELDS
        }
        judge_row.update(
            {
                "parsed_answer": "Answer",
                "predicted_evidence_pages": [2],
                "output_is_legal": True,
                "answer_is_correct": True,
                "evidence_pages_is_correct": True,
                "is_correct": True,
                "inference_binding_sha256": judge_inference_binding_sha256(
                    inference_row
                ),
                "judge_protocol_fingerprint": judge_fingerprint,
            }
        )
        judged = {"1": {"QA": {"QA1": judge_row}}}
        output = root / "output"
        output.mkdir()
        inference_path = output / "results_4B.json"
        judge_path = output / "judge_4B.json"
        inference_path.write_text(json.dumps(inferred), encoding="utf-8")
        judge_path.write_text(json.dumps(judged), encoding="utf-8")

        inference_manifest = {
            "version": 2,
            "papers": {"fixture": "1"},
            "source_sha256": normalized_queue_source_sha256(source),
            "validation_contract": {
                "required_qa_field_values": {
                    "protocol_fingerprint": fingerprint,
                    "qa_source_sha256": source_hash,
                    "pdf_corpus_sha256": pdf_audit["pdf_corpus_sha256"],
                    "input_mode": "pdf",
                    "page_input_policy": "full",
                    "prompt_style": "pdf",
                    "max_pdf_pages": 0,
                },
                "required_qa_field_values_by_paper": {
                    "1": {
                        "pdf_sha256": pdf_audit["papers"]["1"]["sha256"]
                    }
                },
            },
            "metadata": {"stage": "inference", **protocol_metadata},
        }
        judge_manifest = {
            "version": 2,
            "papers": {"fixture": "1"},
            "source_sha256": normalized_queue_source_sha256(inferred),
            "validation_contract": {
                "required_qa_field_values": {
                    "judge_protocol_fingerprint": judge_fingerprint,
                    "protocol_fingerprint": fingerprint,
                    "qa_source_sha256": source_hash,
                    "input_mode": "pdf",
                    "pdf_corpus_sha256": pdf_audit["pdf_corpus_sha256"],
                },
                "required_qa_field_values_by_qa": {
                    "1": {
                        "QA1": {
                            "inference_binding_sha256": (
                                judge_inference_binding_sha256(inference_row)
                            )
                        }
                    }
                },
            },
            "metadata": judge_metadata,
        }
        for stage, manifest in (
            ("inference_4B", inference_manifest),
            ("judge_4B", judge_manifest),
        ):
            path = output / ".adaptive_queue" / stage / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "source": source,
            "source_path": source_path,
            "pdf_dir": pdf_dir,
            "pdf_audit": pdf_audit,
            "output": output,
            "judge_path": judge_path,
        }

    def test_full_pdf_stage_bounds_pixels_and_reuses_vision_features(self) -> None:
        combined_path, ablation_path = write_runtime_inputs()
        stage = build_stages(combined_path, ablation_path)[1]
        command = build_stage_command(
            stage,
            Namespace(
                inference_max_new_tokens=256,
                max_skipped_illegal_rate=0.01,
            ),
        )
        self.assertIn("--max-total-pdf-megapixels", command)
        budget_index = command.index("--max-total-pdf-megapixels") + 1
        self.assertEqual(command[budget_index], str(FULL_PDF_PIXEL_BUDGET_MP))
        self.assertNotIn("--disable-pdf-vision-cache", command)

    def test_outer_stage_snapshot_aggregates_inner_checkpoint_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for phase in ("inference_4B", "inference_8B"):
                path = output / ".adaptive_queue" / phase / "scheduler_state.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "work_state": "completed",
                            "queue": {
                                "completed_papers": 389,
                                "total_papers": 389,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            judge = (
                output
                / ".adaptive_queue"
                / "judge_4B"
                / "scheduler_state.json"
            )
            judge.parent.mkdir(parents=True, exist_ok=True)
            judge.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "work_state": "running",
                        "queue": {
                            "completed_papers": 100,
                            "total_papers": 400,
                        },
                        "progress_heartbeat_at": 123.0,
                    }
                ),
                encoding="utf-8",
            )
            snapshot = stage_work_snapshot(output)
            self.assertEqual(snapshot["phase"], "judge_4B")
            self.assertEqual(snapshot["phase_completed"], 100)
            self.assertEqual(snapshot["phase_total"], 400)
            self.assertAlmostEqual(snapshot["stage_fraction"], 0.5625)
            self.assertEqual(snapshot["inner_progress_heartbeat_at"], 123.0)

    def test_checked_subprocess_refreshes_process_heartbeat(self) -> None:
        heartbeats = []
        run_checked_with_heartbeat(
            ["python", "-c", "pass"],
            cwd=ROOT,
            heartbeat=lambda: heartbeats.append(1),
            poll_seconds=0.01,
        )
        self.assertGreaterEqual(len(heartbeats), 1)

    def test_npm_command_uses_explicit_node_and_resolved_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "node"
            npm_cli = root / "npm-cli.js"
            npm = root / "npm"
            node.write_text("fixture", encoding="utf-8")
            npm_cli.write_text("fixture", encoding="utf-8")
            npm.symlink_to(npm_cli)
            self.assertEqual(
                resolve_npm_command(node_path=node, npm_path=npm),
                [str(node.resolve()), str(npm_cli.resolve())],
            )
            toolchain_env = node_toolchain_env(
                [str(node.resolve()), str(npm_cli.resolve())]
            )
            self.assertEqual(
                toolchain_env["PATH"].split(":", 1)[0],
                str(root.resolve()),
            )

    def test_progress_counts_real_scored_model_qa_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            partial = (
                output
                / ".adaptive_queue"
                / "judge_4B"
                / "papers"
                / "paper-1.json"
            )
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(
                json.dumps({"paper": "1", "QA": {"QA1": {}, "QA2": {}}}),
                encoding="utf-8",
            )
            (output / "judge_8B.json").write_text(
                json.dumps({"1": {"QA": {"QA1": {}, "QA2": {}, "QA3": {}}}}),
                encoding="utf-8",
            )
            progress = judged_model_qa_progress(output, 5)
            self.assertEqual(progress["by_model"], {"4B": 2, "8B": 3})
            self.assertEqual(progress["completed"], 5)
            self.assertEqual(progress["total"], 10)

    def test_pipeline_progress_moves_during_inference_before_judging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            inference = (
                output
                / ".adaptive_queue"
                / "inference_4B"
                / "papers"
                / "paper-1.json"
            )
            inference.parent.mkdir(parents=True, exist_ok=True)
            inference.write_text(
                json.dumps(
                    {"paper": "1", "QA": {"QA1": {}, "QA2": {}}}
                ),
                encoding="utf-8",
            )
            progress = model_qa_pipeline_step_progress(output, 5)
            self.assertEqual(progress["completed"], 2)
            self.assertEqual(progress["total"], 20)
            self.assertEqual(progress["scored_model_qa_completed"], 0)
            self.assertEqual(progress["scored_model_qa_total"], 10)

    def test_single_pdf_combined_dataset_contract(self) -> None:
        combined_path, _ = write_runtime_inputs()
        dataset = json.loads(
            combined_path.read_text(encoding="utf-8")
        )
        rows = [qa for paper in dataset.values() for qa in paper["QA"].values()]
        unanswerable = [qa for qa in rows if qa.get("answer") == "Unanswerable"]
        answerable = [qa for qa in rows if qa.get("answer") != "Unanswerable"]

        self.assertEqual(len(rows), 1200)
        self.assertEqual(len(answerable), 1000)
        self.assertEqual(len(unanswerable), 200)
        self.assertTrue(all(qa.get("evidence_pages") == [] for qa in unanswerable))
        self.assertTrue(all(qa.get("oracle_pages") == [] for qa in unanswerable))
        self.assertTrue(
            all(qa.get("answer_format") == "Unanswerable" for qa in unanswerable)
        )

    def test_v2_closed_book_excludes_unanswerable_items(self) -> None:
        combined_path, ablation_path = write_runtime_inputs()
        stages = build_stages(combined_path, ablation_path)
        self.assertEqual(stages[0]["qa"], ORDINARY_VIEW)
        self.assertIn("question_only_1000", str(stages[0]["name"]))
        self.assertEqual(stages[1]["qa"], combined_path)
        self.assertEqual(stages[2]["qa"], combined_path)

    def test_prompts_define_one_refusal_label_and_no_refusal_evidence(self) -> None:
        self.assertIn('exactly "Unanswerable"', PDF_SYSTEM_PROMPT)
        self.assertIn("evidence_pages must be []", " ".join(PDF_SYSTEM_PROMPT.split()))
        self.assertIn("all and only", PDF_SYSTEM_PROMPT)
        self.assertNotIn("at most 8", PDF_SYSTEM_PROMPT)
        self.assertNotIn("Never enumerate every shown page", PDF_SYSTEM_PROMPT)
        prompt = build_fill_prompt("Question?", input_mode="pdf")
        self.assertIn("exactly 'Unanswerable'", prompt)
        self.assertIn("do not use any synonym", prompt)

    def test_human_reviewed_preflight_contract(self) -> None:
        combined_path, ablation_path = write_runtime_inputs()
        result = validate_single_pdf_inputs(
            Namespace(
                expected_ordinary_sha256=file_sha256(ORDINARY_VIEW),
                expected_unanswerable_sha256=file_sha256(UNANSWERABLE_VIEW),
            ),
            combined_path,
            ablation_path,
        )
        self.assertEqual(
            result,
            {
                "answerable_qas": 1000,
                "unanswerable_qas": 200,
                "combined_qas": 1200,
                "ablation_variants": 676,
            },
        )
    def test_all_local_evaluation_models_disable_thinking(self) -> None:
        for name in (
            "local_qwen3_vl_4b",
            "local_qwen3_vl_8b",
            "local_qwen3_vl_4b_pdf",
            "local_qwen3_vl_8b_pdf",
            "local_qwen3_6_27b_judge",
        ):
            self.assertIs(DEFAULT_PROVIDER_SPECS[name]["enable_thinking"], False)

    def test_ablation_variant_uses_base_question_gold(self) -> None:
        dataset = {
            "1": {
                "QA": {
                    "QA1": {
                        "question": "Question?",
                        "answer": "Answer",
                        "evidence_pages": [2, 5],
                        "oracle_pages": [2, 5],
                    }
                }
            }
        }
        ablation = {
            "1": {
                "QA": {
                    "QA1__drop_p2": {
                        "question": "Question?",
                        "answer": "Answer",
                        "evidence_pages": [2, 5],
                        "input_pages": [5],
                        "ablation": {
                            "source_qa_id": "QA1",
                            "removed_physical_pdf_page": 2,
                            "original_evidence_pages": [2, 5],
                        },
                    }
                }
            }
        }
        judged = {
            "1": {
                "QA": {
                    "QA1__drop_p2": self._pdf_judged_row(
                        reference_pages=[2, 5],
                        predicted_pages=[5],
                        shown_pages=[5],
                        page_input_policy="qa_field",
                    )
                }
            }
        }
        gold = gold_index(dataset)
        rows = eval_rows(
            judged,
            gold,
            mode="ablation",
            ablation_expectations=build_ablation_expectations(
                ablation, gold
            ),
            expected_count=1,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["answer_correct"])
        self.assertEqual(rows[0]["qa_id"], "QA1__drop_p2")

    def test_report_rejects_tampered_judge_parse(self) -> None:
        dataset = self._gold_dataset()
        row = self._pdf_judged_row()
        row["model_output"] = json.dumps(
            {"answer_pre": "Tampered", "evidence_pages": [2]}
        )
        judged = {"1": {"QA": {"QA1": row}}}
        with self.assertRaisesRegex(ValueError, "parsed_answer mismatch"):
            eval_rows(
                judged,
                gold_index(dataset),
                mode="full",
                pdf_audit=self._pdf_audit(),
                expected_count=1,
            )

    def test_report_rejects_missing_non_ablation_row(self) -> None:
        dataset = self._gold_dataset(two_rows=True)
        judged = {
            "1": {"QA": {"QA1": self._pdf_judged_row()}}
        }
        with self.assertRaisesRegex(ValueError, "fixed evaluation set"):
            eval_rows(
                judged,
                gold_index(dataset),
                mode="full",
                pdf_audit=self._pdf_audit(),
                expected_count=2,
            )

    def test_report_rejects_prediction_from_unshown_page(self) -> None:
        dataset = self._gold_dataset()
        row = self._pdf_judged_row(
            predicted_pages=[3],
            shown_pages=[2],
            page_input_policy="qa_field",
        )
        judged = {"1": {"QA": {"QA1": row}}}
        with self.assertRaisesRegex(ValueError, "parsed_answer mismatch|legality mismatch"):
            eval_rows(
                judged,
                gold_index(dataset),
                mode="oracle",
                pdf_audit=self._pdf_audit(),
                expected_count=1,
            )

    def test_pdf_audit_reads_physical_page_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_dir = Path(tmp)
            writer = PdfWriter()
            for _ in range(4):
                writer.add_blank_page(width=72, height=72)
            with (pdf_dir / "1.pdf").open("wb") as handle:
                writer.write(handle)
            audit = audit_pdf_corpus(self._gold_dataset(), [pdf_dir])
        self.assertEqual(audit["papers"]["1"]["total_pdf_pages"], 4)
        self.assertEqual(audit["paper_count"], 1)
        self.assertEqual(len(audit["papers"]["1"]["sha256"]), 64)

    def test_full_pdf_rejects_contiguous_prefix_of_physical_pdf(self) -> None:
        dataset = self._gold_dataset()
        judged = {
            "1": {
                "QA": {
                    "QA1": self._pdf_judged_row(shown_pages=[1, 2])
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "all physical PDF pages"):
            eval_rows(
                judged,
                gold_index(dataset),
                mode="full",
                pdf_audit=self._pdf_audit(page_count=3),
                expected_count=1,
            )

    def test_oracle_answerable_requires_gold_evidence_page_set(self) -> None:
        dataset = self._gold_dataset()
        dataset["1"]["QA"]["QA1"]["oracle_pages"] = [3]
        judged = {
            "1": {
                "QA": {
                    "QA1": self._pdf_judged_row(
                        shown_pages=[3], page_input_policy="qa_field"
                    )
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "oracle/evidence pages differ"):
            eval_rows(
                judged,
                gold_index(dataset),
                mode="oracle",
                pdf_audit=self._pdf_audit(),
                expected_count=1,
            )

    def test_oracle_unanswerable_uses_full_physical_pdf_fallback(self) -> None:
        dataset = {
            "1": {
                "QA": {
                    "UA1": {
                        "question": "Can this be answered?",
                        "answer": "Unanswerable",
                        "evidence_pages": [],
                        "oracle_pages": [],
                    }
                }
            }
        }
        judged = {
            "1": {
                "QA": {
                    "UA1": self._pdf_judged_row(
                        question="Can this be answered?",
                        answer="Unanswerable",
                        reference_pages=[],
                        predicted_pages=[],
                        shown_pages=[1, 2, 3],
                        page_input_policy="qa_field",
                    )
                }
            }
        }
        rows = eval_rows(
            judged,
            gold_index(dataset),
            mode="oracle",
            pdf_audit=self._pdf_audit(),
            expected_count=1,
        )
        self.assertEqual(rows[0]["shown_pages"], [1, 2, 3])
        self.assertTrue(rows[0]["answer_correct"])

    def test_oracle_unanswerable_rejects_empty_shown_pages(self) -> None:
        dataset = {
            "1": {
                "QA": {
                    "UA1": {
                        "question": "Can this be answered?",
                        "answer": "Unanswerable",
                        "evidence_pages": [],
                        "oracle_pages": [],
                    }
                }
            }
        }
        judged = {
            "1": {
                "QA": {
                    "UA1": self._pdf_judged_row(
                        question="Can this be answered?",
                        answer="Unanswerable",
                        reference_pages=[],
                        predicted_pages=[],
                        shown_pages=[],
                        page_input_policy="qa_field",
                    )
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "Oracle shown pages mismatch"):
            eval_rows(
                judged,
                gold_index(dataset),
                mode="oracle",
                pdf_audit=self._pdf_audit(),
                expected_count=1,
            )

    def test_strict_report_provenance_binds_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._write_strict_full_provenance_fixture(Path(tmp))
            binding = validate_eval_provenance(
                label="full_4B",
                judge_path=fixture["judge_path"],
                source_path=fixture["source_path"],
                source_dataset=fixture["source"],
                report_gold=gold_index(fixture["source"]),
                pdf_audit=fixture["pdf_audit"],
            )
            expected_source_sha256 = sha256(fixture["source_path"])
        self.assertEqual(len(binding["inference_protocol_fingerprint"]), 64)
        self.assertEqual(len(binding["judge_protocol_fingerprint"]), 64)
        self.assertEqual(binding["source_sha256"], expected_source_sha256)

    def test_strict_report_rejects_current_pdf_byte_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._write_strict_full_provenance_fixture(Path(tmp))
            writer = PdfWriter()
            for _ in range(4):
                writer.add_blank_page(width=72, height=72)
            with (fixture["pdf_dir"] / "1.pdf").open("wb") as handle:
                writer.write(handle)
            changed_audit = audit_pdf_corpus(
                fixture["source"], [fixture["pdf_dir"]]
            )
            with self.assertRaisesRegex(ValueError, "inference config mismatch"):
                validate_eval_provenance(
                    label="full_4B",
                    judge_path=fixture["judge_path"],
                    source_path=fixture["source_path"],
                    source_dataset=fixture["source"],
                    report_gold=gold_index(fixture["source"]),
                    pdf_audit=changed_audit,
                )

    def test_strict_report_rejects_tampered_judge_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self._write_strict_full_provenance_fixture(Path(tmp))
            manifest_path = (
                fixture["output"]
                / ".adaptive_queue"
                / "judge_4B"
                / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["judge_protocol_fingerprint"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint is internally invalid"):
                validate_eval_provenance(
                    label="full_4B",
                    judge_path=fixture["judge_path"],
                    source_path=fixture["source_path"],
                    source_dataset=fixture["source"],
                    report_gold=gold_index(fixture["source"]),
                    pdf_audit=fixture["pdf_audit"],
                )


if __name__ == "__main__":
    unittest.main()
