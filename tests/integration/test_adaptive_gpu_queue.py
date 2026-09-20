from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import adaptive_gpu_pool
import gpu_reservation
from durable_work_queue import DurablePaperQueue
from eval_framework import LocalTransformersProvider, atomic_write_json
from run_inference import (
    build_per_qa_page_plan,
    canonical_structured_output,
    choose_pdf_render_dpi,
    generate_with_retries,
    has_runaway_evidence_pages,
    has_runaway_repetition,
    prepare_pdf_output_for_storage,
    repair_json_string_backslashes,
    structured_output_is_valid,
)


class PdfPagePlanTests(unittest.TestCase):
    def test_oracle_unanswerable_falls_back_to_full_pdf(self) -> None:
        rendered, per_qa = build_per_qa_page_plan(
            {
                "answerable": {"evidence_pages": [2, 4]},
                "unanswerable": {"evidence_pages": []},
            },
            "evidence_pages",
            5,
        )
        self.assertEqual(rendered, [1, 2, 3, 4, 5])
        self.assertEqual(per_qa["answerable"], [2, 4])
        self.assertEqual(per_qa["unanswerable"], [1, 2, 3, 4, 5])


def source(papers: int = 1) -> dict:
    return {
        f"paper-{index}": {
            "QA": {
                "q1": {"question": "q", "answer": "a"},
                "q2": {"question": "q2", "answer": "b"},
            }
        }
        for index in range(papers)
    }


def paper_result(paper_id: str, *, judged: bool = False) -> dict:
    qa = {
        qa_id: {
            "model_output": f"answer-{qa_id}",
            **({"is_correct": True} if judged else {}),
        }
        for qa_id in ("q1", "q2")
    }
    return {"paper": paper_id, "QA": qa}


class DurablePaperQueueTests(unittest.TestCase):
    def test_manifest_rejects_same_ids_with_changed_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = source()
            DurablePaperQueue(
                tmp,
                original,
                manifest_metadata={"protocol": "question_only_v1"},
            )
            changed = source()
            changed["paper-0"]["QA"]["q1"]["answer"] = "changed"
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                DurablePaperQueue(
                    tmp,
                    changed,
                    manifest_metadata={"protocol": "question_only_v1"},
                )

    def test_manifest_rejects_protocol_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            DurablePaperQueue(
                tmp,
                source(),
                manifest_metadata={"protocol": "question_only_v1"},
            )
            with self.assertRaisesRegex(ValueError, "metadata"):
                DurablePaperQueue(
                    tmp,
                    source(),
                    manifest_metadata={"protocol": "pdf_evidence_v1"},
                )

    def test_claim_is_exclusive_between_concurrent_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(tmp, source())
            barrier = threading.Barrier(3)
            claims = []

            def claim(worker: str) -> None:
                barrier.wait()
                claims.append(queue.claim_next(worker))

            threads = [
                threading.Thread(target=claim, args=(f"worker-{i}",))
                for i in range(2)
            ]
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join()
            self.assertEqual(sum(item is not None for item in claims), 1)

    def test_claim_path_is_never_visible_as_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(tmp, source())
            original_link = os.link
            observed = []

            def inspect_before_link(source_path, destination_path):
                value = json.loads(Path(source_path).read_text())
                observed.append(value)
                self.assertFalse(Path(destination_path).exists())
                return original_link(source_path, destination_path)

            with mock.patch(
                "durable_work_queue.os.link", side_effect=inspect_before_link
            ):
                claim = queue.claim_next("worker")
            self.assertIsNotNone(claim)
            self.assertEqual(observed[0]["paper_id"], "paper-0")
            self.assertEqual(
                json.loads(claim.path.read_text())["worker_id"], "worker"
            )

    def test_dead_owner_claim_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(tmp, source())
            path = queue.claim_path("paper-0")
            atomic_write_json(
                path,
                {
                    "paper_id": "paper-0",
                    "worker_id": "dead",
                    "pid": 999_999_999,
                    "hostname": __import__("socket").gethostname(),
                    "updated_at": 0,
                },
            )
            claim = queue.claim_next("replacement")
            self.assertIsNotNone(claim)
            self.assertEqual(claim.paper_id, "paper-0")

    def test_stale_claims_are_reaped_before_new_work_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(tmp, source(2))
            stale_path = queue.claim_path("paper-0")
            atomic_write_json(
                stale_path,
                {
                    "paper_id": "paper-0",
                    "worker_id": "dead",
                    "pid": 999_999_999,
                    "hostname": __import__("socket").gethostname(),
                    "updated_at": 0,
                },
            )
            queue.claim_next("replacement")
            if stale_path.exists():
                replacement = json.loads(
                    stale_path.read_text(encoding="utf-8")
                )
                self.assertNotEqual(replacement["pid"], 999_999_999)

    def test_partial_result_is_not_complete_and_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(tmp, source())
            partial = {"paper": "paper-0", "QA": {"q1": {"model_output": "a"}}}
            queue.save_partial("paper-0", partial)
            self.assertFalse(queue.is_complete())
            claim = queue.claim_next("resume")
            self.assertIsNotNone(claim)
            resumed = queue.load_result("paper-0")
            self.assertIn("q1", resumed["QA"])

    def test_structured_output_is_screened_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(), require_structured_output=True
            )
            invalid = paper_result("paper-0")
            queue.save_partial("paper-0", invalid)
            self.assertFalse(queue.is_complete())
            valid = paper_result("paper-0")
            for item in valid["QA"].values():
                item["model_output"] = (
                    '{"answer_pre":"ok","evidence_pages":[1]}'
                )
            claim = queue.claim_next("worker")
            queue.finish(claim, valid)
            self.assertTrue(queue.is_complete())

    def test_structured_output_rejects_empty_answer_and_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(), require_structured_output=True
            )
            result = paper_result("paper-0")
            result["QA"]["q1"]["model_output"] = (
                '{"answer_pre":"","evidence_pages":[]}'
            )
            result["QA"]["q2"]["model_output"] = (
                '{"answer_pre":"answer","evidence_pages":[]}'
            )
            valid, reason = queue.validate_result("paper-0", result)
            self.assertFalse(valid)
            self.assertTrue(reason.startswith("malformed_structured_output:"))

    def test_structured_unanswerable_requires_empty_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(), require_structured_output=True
            )
            valid = paper_result("paper-0")
            for item in valid["QA"].values():
                item["model_output"] = (
                    '{"answer_pre":"Unanswerable","evidence_pages":[]}'
                )
            self.assertEqual(
                queue.validate_result("paper-0", valid), (True, "ok")
            )
            valid["QA"]["q1"]["model_output"] = (
                '{"answer_pre":"Unanswerable","evidence_pages":[1]}'
            )
            self.assertFalse(queue.validate_result("paper-0", valid)[0])

    def test_audited_illegal_output_can_complete_for_judge_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(), require_structured_output=True
            )
            result = paper_result("paper-0")
            for item in result["QA"].values():
                raw = '{"answer_pre":"ok","evidence_pages":[221]}'
                item.update(
                    {
                        "model_output": raw,
                        "raw_model_output": raw,
                        "raw_model_output_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                        "deterministic_normalizations": [],
                        "generation_status": (
                            "invalid_model_output_after_retries"
                        ),
                        "shown_pdf_pages": [1, 2],
                    }
                )
            claim = queue.claim_next("worker")
            queue.finish(claim, result)
            self.assertTrue(queue.is_complete())

            result["QA"]["q1"]["model_output"] += " "
            self.assertEqual(
                queue.validate_result("paper-0", result)[1],
                "invalid_output_audit_mismatch:q1",
            )

    def test_judge_required_fields_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(), required_qa_fields=("is_correct",)
            )
            queue.save_partial("paper-0", paper_result("paper-0"))
            self.assertFalse(queue.is_complete())
            claim = queue.claim_next("judge")
            queue.finish(claim, paper_result("paper-0", judged=True))
            self.assertTrue(queue.is_complete())

    def test_bootstrap_and_merge_preserve_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.json"
            atomic_write_json(existing, {"paper-0": paper_result("paper-0")})
            queue = DurablePaperQueue(root / "queue", source())
            self.assertEqual(queue.bootstrap([existing]), 1)
            output = root / "merged.json"
            queue.merge(output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["paper-0"][
                    "paper"
                ],
                "paper-0",
            )

    def test_bootstrap_rejects_stale_per_qa_binding_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.json"
            stale = paper_result("paper-0", judged=True)
            for item in stale["QA"].values():
                item["inference_binding_sha256"] = "old"
            atomic_write_json(existing, {"paper-0": stale})
            expected = {
                "paper-0": {
                    qa_id: {"inference_binding_sha256": "new"}
                    for qa_id in source()["paper-0"]["QA"]
                }
            }
            queue = DurablePaperQueue(
                root / "queue",
                source(),
                required_qa_field_values_by_qa=expected,
            )
            self.assertEqual(queue.bootstrap([existing]), 0)
            self.assertFalse(queue.is_complete())

            current = paper_result("paper-0", judged=True)
            for item in current["QA"].values():
                item["inference_binding_sha256"] = "new"
            atomic_write_json(existing, {"paper-0": current})
            self.assertEqual(queue.bootstrap([existing]), 1)
            self.assertTrue(queue.is_complete())

    def test_failure_quarantines_after_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(), max_paper_failures=1
            )
            claim = queue.claim_next("worker")
            queue.fail(claim, RuntimeError("boom"))
            failure = json.loads(
                queue.failure_path("paper-0").read_text(encoding="utf-8")
            )
            self.assertTrue(failure["quarantined"])
            self.assertIsNone(queue.claim_next("another"))

    def test_status_distinguishes_retryable_and_quarantined_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            queue = DurablePaperQueue(
                tmp, source(2), max_paper_failures=1
            )
            claim = queue.claim_next("worker")
            queue.fail(claim, RuntimeError("malformed model output"))

            status = queue.status()

            self.assertEqual(status["quarantined_papers"], 1)
            self.assertEqual(status["retryable_papers"], 1)
            self.assertEqual(status["runnable_papers"], 1)


class PdfRenderBudgetTests(unittest.TestCase):
    class FakePage:
        def get_size(self):
            return (612.0, 792.0)

    class FakePdf:
        def __init__(self, pages: int):
            self.pages = [
                PdfRenderBudgetTests.FakePage() for _ in range(pages)
            ]

        def __len__(self):
            return len(self.pages)

        def __getitem__(self, index):
            return self.pages[index]

    def test_short_pdf_keeps_requested_dpi(self) -> None:
        pdf = self.FakePdf(20)
        self.assertEqual(
            choose_pdf_render_dpi(
                pdf, list(range(1, 21)), 144, 180, 72
            ),
            144,
        )

    def test_long_pdf_is_downscaled_without_dropping_pages(self) -> None:
        pdf = self.FakePdf(155)
        selected = list(range(1, 156))
        effective = choose_pdf_render_dpi(
            pdf, selected, 144, 180, 72
        )
        self.assertGreaterEqual(effective, 72)
        self.assertLess(effective, 144)
        rendered_pixels = (
            155 * 612 * 792 * (effective / 72.0) ** 2
        )
        self.assertLessEqual(rendered_pixels, 180_000_000)

    def test_disabled_budget_preserves_requested_dpi(self) -> None:
        pdf = self.FakePdf(155)
        self.assertEqual(
            choose_pdf_render_dpi(
                pdf, list(range(1, 156)), 144, 0, 72
            ),
            144,
        )


class AdaptiveGpuSelectionTests(unittest.TestCase):
    def test_a40_is_filtered_even_when_allowed(self) -> None:
        inventory = {
            "0": {"name": "NVIDIA A40", "uuid": "a40"},
            "2": {"name": "NVIDIA A800 80GB PCIe", "uuid": "a800-2"},
            "3": {"name": "NVIDIA A800 80GB PCIe", "uuid": "a800-3"},
        }
        with mock.patch.object(
            adaptive_gpu_pool, "_gpu_inventory", return_value=inventory
        ):
            self.assertEqual(
                adaptive_gpu_pool.a800_gpu_ids(["0", "2"]), ["2"]
            )

    def test_no_a800_never_falls_back_to_cpu_or_a40(self) -> None:
        with mock.patch.object(
            adaptive_gpu_pool,
            "_gpu_inventory",
            return_value={"0": {"name": "NVIDIA A40", "uuid": "a40"}},
        ):
            with self.assertRaisesRegex(RuntimeError, "No allowed A800"):
                adaptive_gpu_pool.AdaptiveGpuPool(
                    command_factory=lambda gpu, attempt: ["true"],
                    is_complete=lambda: False,
                    status_factory=dict,
                )

    def test_pool_launches_worker_and_finishes_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "done"
            state = root / "state.json"
            inventory = {
                "2": {
                    "name": "NVIDIA A800 80GB PCIe",
                    "uuid": "a800-2",
                }
            }
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(done)!r}).write_text('ok')"
                ),
            ]
            with (
                mock.patch.object(
                    adaptive_gpu_pool,
                    "_gpu_inventory",
                    return_value=inventory,
                ),
                mock.patch.object(
                    adaptive_gpu_pool,
                    "gpu_foreign_processes",
                    return_value=[],
                ),
                mock.patch.object(
                    adaptive_gpu_pool,
                    "try_gpu_lease",
                    return_value=mock.Mock(close=mock.Mock()),
                ),
            ):
                pool = adaptive_gpu_pool.AdaptiveGpuPool(
                    command_factory=lambda gpu, attempt: command,
                    is_complete=done.exists,
                    status_factory=lambda: {"done": done.exists()},
                    allowed_gpus=["2"],
                    poll_seconds=1,
                    state_path=state,
                    log_dir=root / "logs",
                )
                pool.run()
            self.assertTrue(done.exists())
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["workers"], {})

    def test_shared_pool_launches_with_foreign_process_when_memory_is_enough(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            done = root / "done"
            inventory = {
                "3": {
                    "name": "NVIDIA A800 80GB PCIe",
                    "uuid": "a800-3",
                }
            }
            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    f"Path({str(done)!r}).write_text('ok')"
                ),
            ]
            with (
                mock.patch.object(
                    adaptive_gpu_pool,
                    "_gpu_inventory",
                    return_value=inventory,
                ),
                mock.patch.object(
                    adaptive_gpu_pool,
                    "gpu_foreign_processes",
                    return_value=[
                        {"pid": "999", "used_memory_mib": "16000"}
                    ],
                ),
                mock.patch.object(
                    adaptive_gpu_pool,
                    "gpu_free_memory_mib",
                    return_value=64000,
                ),
                mock.patch.object(
                    adaptive_gpu_pool,
                    "try_gpu_lease",
                    return_value=mock.Mock(close=mock.Mock()),
                ),
            ):
                pool = adaptive_gpu_pool.AdaptiveGpuPool(
                    command_factory=lambda gpu, attempt: command,
                    is_complete=done.exists,
                    status_factory=lambda: {"done": done.exists()},
                    allowed_gpus=["3"],
                    poll_seconds=1,
                    allow_shared_gpus=True,
                    min_free_memory_mib=56000,
                    log_dir=root / "logs",
                )
                pool.run()
            self.assertTrue(done.exists())
            started = [
                event
                for event in pool.events
                if event["event"] == "worker_started"
            ]
            self.assertTrue(started[0]["shared"])
            self.assertEqual(started[0]["foreign_pids"], ["999"])

    def test_shared_pool_rejects_card_below_memory_threshold(self) -> None:
        lease = mock.Mock(close=mock.Mock())
        inventory = {
            "3": {
                "name": "NVIDIA A800 80GB PCIe",
                "uuid": "a800-3",
            }
        }
        with (
            mock.patch.object(
                adaptive_gpu_pool,
                "_gpu_inventory",
                return_value=inventory,
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "gpu_foreign_processes",
                return_value=[
                    {"pid": "999", "used_memory_mib": "40000"}
                ],
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "gpu_free_memory_mib",
                return_value=39000,
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "try_gpu_lease",
                return_value=lease,
            ),
        ):
            pool = adaptive_gpu_pool.AdaptiveGpuPool(
                command_factory=lambda gpu, attempt: ["true"],
                is_complete=lambda: False,
                status_factory=dict,
                allowed_gpus=["3"],
                allow_shared_gpus=True,
                min_free_memory_mib=56000,
            )
            self.assertFalse(pool._start("3"))
        lease.close.assert_called_once()
        self.assertEqual(
            pool.blocked_reasons["3"]["reason"],
            "insufficient_free_memory",
        )

    def test_pool_fails_immediately_when_all_remaining_work_is_quarantined(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            inventory = {
                "2": {
                    "name": "NVIDIA A800 80GB PCIe",
                    "uuid": "a800-2",
                }
            }
            queue_status = {
                "total_papers": 2,
                "completed_papers": 1,
                "unfinished_papers": 1,
                "active_claims": 0,
                "failures": [
                    {
                        "paper_id": "stuck",
                        "count": 20,
                        "quarantined": True,
                    }
                ],
            }
            with mock.patch.object(
                adaptive_gpu_pool,
                "_gpu_inventory",
                return_value=inventory,
            ):
                pool = adaptive_gpu_pool.AdaptiveGpuPool(
                    command_factory=lambda gpu, attempt: ["should-not-run"],
                    is_complete=lambda: False,
                    status_factory=lambda: queue_status,
                    allowed_gpus=["2"],
                    state_path=state,
                )
                with self.assertRaisesRegex(
                    adaptive_gpu_pool.QuarantinedWorkError,
                    "stuck",
                ):
                    pool.run()
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")
            self.assertIn("QuarantinedWorkError", saved["error"])
            self.assertEqual(pool.workers, {})

    def test_repeated_worker_exits_without_progress_fail_the_pool(self) -> None:
        inventory = {
            "2": {
                "name": "NVIDIA A800 80GB PCIe",
                "uuid": "a800-2",
            }
        }
        queue_status = {
            "total_papers": 2,
            "completed_papers": 1,
            "unfinished_papers": 1,
            "active_claims": 0,
            "retryable_papers": 1,
            "failures": [],
        }
        with mock.patch.object(
            adaptive_gpu_pool,
            "_gpu_inventory",
            return_value=inventory,
        ):
            pool = adaptive_gpu_pool.AdaptiveGpuPool(
                command_factory=lambda gpu, attempt: ["false"],
                is_complete=lambda: False,
                status_factory=lambda: queue_status,
                allowed_gpus=["2"],
                max_failures_without_progress=3,
            )

        for attempt in range(3):
            process = mock.Mock()
            process.pid = 5000 + attempt
            process.args = ["false"]
            process.returncode = 1
            process.poll.return_value = 1
            pool.workers["2"] = adaptive_gpu_pool.Worker(
                gpu_id="2",
                process=process,
                lease=mock.Mock(close=mock.Mock()),
                started_at=0,
                progress_completed_at_start=1,
            )
            if attempt < 2:
                pool._reap_and_monitor()
            else:
                with self.assertRaisesRegex(
                    adaptive_gpu_pool.WorkerProgressStallError,
                    "3 consecutive worker failures",
                ):
                    pool._reap_and_monitor()

        self.assertEqual(pool.consecutive_failures_without_progress, 3)

    def test_process_heartbeat_does_not_advance_progress_heartbeat(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            inventory = {
                "2": {
                    "name": "NVIDIA A800 80GB PCIe",
                    "uuid": "a800-2",
                }
            }
            queue_status = {
                "total_papers": 2,
                "completed_papers": 1,
                "unfinished_papers": 1,
                "active_claims": 0,
                "retryable_papers": 1,
                "failures": [],
            }
            with mock.patch.object(
                adaptive_gpu_pool,
                "_gpu_inventory",
                return_value=inventory,
            ):
                pool = adaptive_gpu_pool.AdaptiveGpuPool(
                    command_factory=lambda gpu, attempt: ["true"],
                    is_complete=lambda: False,
                    status_factory=lambda: queue_status,
                    allowed_gpus=["2"],
                    state_path=state,
                )
            pool.last_progress_at = 100.0
            with mock.patch.object(
                adaptive_gpu_pool.time, "time", side_effect=[200.0, 201.0]
            ):
                pool._write_state()
                first = json.loads(state.read_text(encoding="utf-8"))
                pool._write_state()
                second = json.loads(state.read_text(encoding="utf-8"))

            self.assertEqual(first["progress_heartbeat_at"], 100.0)
            self.assertEqual(second["progress_heartbeat_at"], 100.0)
            self.assertGreater(
                second["process_heartbeat_at"],
                first["process_heartbeat_at"],
            )

    def test_shared_pool_rejects_orphan_from_same_work_queue(self) -> None:
        lease = mock.Mock(close=mock.Mock())
        inventory = {
            "3": {
                "name": "NVIDIA A800 80GB PCIe",
                "uuid": "a800-3",
            }
        }
        command = [
            sys.executable,
            "src/pku_qa/evaluation/run_inference.py",
            "--work-queue-dir",
            "queue/formal4211",
        ]
        with (
            mock.patch.object(
                adaptive_gpu_pool,
                "_gpu_inventory",
                return_value=inventory,
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "gpu_foreign_processes",
                return_value=[
                    {"pid": "999", "used_memory_mib": "16000"}
                ],
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "_process_command",
                return_value=command,
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "gpu_free_memory_mib",
                return_value=64000,
            ),
            mock.patch.object(
                adaptive_gpu_pool,
                "try_gpu_lease",
                return_value=lease,
            ),
        ):
            pool = adaptive_gpu_pool.AdaptiveGpuPool(
                command_factory=lambda gpu, attempt: command,
                is_complete=lambda: False,
                status_factory=dict,
                allowed_gpus=["3"],
                allow_shared_gpus=True,
                min_free_memory_mib=56000,
            )
            self.assertFalse(pool._start("3"))
        lease.close.assert_called_once()
        self.assertEqual(
            pool.blocked_reasons["3"]["reason"],
            "same_work_queue_process",
        )

    def test_shutdown_signal_notifies_all_worker_process_groups(self) -> None:
        inventory = {
            "2": {
                "name": "NVIDIA A800 80GB PCIe",
                "uuid": "a800-2",
            }
        }
        with mock.patch.object(
            adaptive_gpu_pool,
            "_gpu_inventory",
            return_value=inventory,
        ):
            pool = adaptive_gpu_pool.AdaptiveGpuPool(
                command_factory=lambda gpu, attempt: ["true"],
                is_complete=lambda: False,
                status_factory=dict,
                allowed_gpus=["2"],
            )
        process = mock.Mock()
        process.pid = 4321
        process.poll.return_value = None
        pool.workers["2"] = adaptive_gpu_pool.Worker(
            gpu_id="2",
            process=process,
            lease=mock.Mock(),
            started_at=0,
        )
        with mock.patch.object(adaptive_gpu_pool.os, "killpg") as killpg:
            with self.assertRaises(adaptive_gpu_pool.PoolShutdownSignal):
                pool._handle_shutdown_signal(signal.SIGTERM, None)
        killpg.assert_called_once_with(4321, signal.SIGTERM)
        self.assertTrue(pool.stop_requested)

    def test_project_gpu_wait_allows_foreign_process_with_memory_gate(
        self,
    ) -> None:
        inventory = {
            "3": {"name": "NVIDIA A800", "uuid": "a800-3"}
        }
        processes = {
            "a800-3": [{"pid": "999", "used_memory_mib": "16000"}]
        }
        with (
            mock.patch.dict(
                gpu_reservation.os.environ,
                {
                    "GPU_ALLOW_SHARED": "1",
                    "GPU_MIN_FREE_MEMORY_MIB": "56000",
                },
                clear=False,
            ),
            mock.patch.object(
                gpu_reservation,
                "_gpu_inventory",
                return_value=inventory,
            ),
            mock.patch.object(
                gpu_reservation,
                "_gpu_compute_processes",
                return_value=processes,
            ),
            mock.patch.object(
                gpu_reservation,
                "_gpu_free_memory",
                return_value={"3": 64000},
            ),
        ):
            gpu_reservation.wait_for_gpus(["3"], label="shared-test")


class RetryAndOutputValidationTests(unittest.TestCase):
    def test_runaway_repetition_detection(self) -> None:
        self.assertTrue(has_runaway_repetition("\u2009" * 64))
        self.assertTrue(has_runaway_repetition(r"\u2009" * 32))
        self.assertFalse(has_runaway_repetition("normal concise answer"))

    def test_runaway_evidence_page_detection_uses_explicit_degen_guard(self) -> None:
        output = (
            '{"answer_pre":"ok","evidence_pages":'
            "[3,4,5,6,7,8,9,10,11,12"
        )
        self.assertTrue(has_runaway_evidence_pages(output, maximum_pages=8))
        self.assertFalse(
            has_runaway_evidence_pages(
                '{"answer_pre":"ok","evidence_pages":[3,7,11]}'
            )
        )

    def test_structured_validator_rejects_thinking_or_malformed_output(self) -> None:
        self.assertFalse(structured_output_is_valid("reasoning...", True))
        self.assertFalse(
            structured_output_is_valid(
                'prefix {"answer_pre":"x","evidence_pages":[2]} suffix',
                True,
            )
        )

    def test_structured_validator_enforces_unanswerable_protocol(self) -> None:
        self.assertTrue(
            structured_output_is_valid(
                '{"answer_pre":"Unanswerable","evidence_pages":[]}', True
            )
        )
        self.assertFalse(
            structured_output_is_valid(
                '{"answer_pre":"not mentioned","evidence_pages":[]}', True
            )
        )
        self.assertFalse(
            structured_output_is_valid(
                '{"answer_pre":"unanswerable","evidence_pages":[]}', True
            )
        )
        self.assertFalse(
            structured_output_is_valid(
                '{"answer_pre":"Unanswerable","evidence_pages":[2]}', True
            )
        )
        self.assertFalse(
            structured_output_is_valid(
                '{"answer_pre":"","evidence_pages":[]}', True
            )
        )

    def test_structured_validator_accepts_more_than_eight_evidence_pages(
        self,
    ) -> None:
        output = json.dumps(
            {"answer_pre": "ok", "evidence_pages": list(range(1, 10))}
        )
        self.assertTrue(structured_output_is_valid(output, True))

    def test_canonicalizer_finds_valid_object_after_unrelated_braces(self) -> None:
        output = (
            'analysis {"noise": 1} final '
            '{"answer":"ok","evidence_pages":["Page 3", 2, 2]}'
        )
        self.assertEqual(
            json.loads(canonical_structured_output(output)),
            {"answer_pre": "ok", "evidence_pages": [2, 3]},
        )

    def test_canonicalizer_repairs_unescaped_latex_in_json_string(self) -> None:
        output = (
            r'{"answer_pre":"D_{qc}(X) gives \tau_\alpha and '
            r'\operatorname{pgdim}\leq 1","evidence_pages":[14]}'
        )
        canonical = canonical_structured_output(output)
        self.assertEqual(
            json.loads(canonical),
            {
                "answer_pre": (
                    r"D_{qc}(X) gives \tau_\alpha and "
                    r"\operatorname{pgdim}\leq 1"
                ),
                "evidence_pages": [14],
            },
        )
        repaired = repair_json_string_backslashes(output)
        self.assertIn(r"\\tau", repaired)
        self.assertIn(r"\\leq", repaired)

    def test_generation_retries_invalid_then_succeeds(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                if self.calls == 1:
                    return "[ERROR: transient]"
                return '{"answer_pre":"ok","evidence_pages":[1]}'

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=10,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        output = generate_with_retries(provider, [], args)
        self.assertIn("answer_pre", output)
        self.assertEqual(provider.calls, 2)

    def test_generation_retries_evidence_page_not_shown(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                page = 99 if self.calls == 1 else 2
                return (
                    '{"answer_pre":"ok","evidence_pages":['
                    f"{page}]}}"
                )

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=10,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        output = generate_with_retries(
            provider, [], args, allowed_evidence_pages=[1, 2]
        )
        self.assertEqual(json.loads(output)["evidence_pages"], [2])
        self.assertEqual(provider.calls, 2)

    def test_exhausted_invalid_output_is_preserved_for_judge(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                return '{"answer_pre":"ok","evidence_pages":[2,221]}'

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=10,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        raw = generate_with_retries(
            provider, [], args, allowed_evidence_pages=[1, 2]
        )
        model_output, normalizations, status = (
            prepare_pdf_output_for_storage(raw, allowed_pages=[1, 2])
        )
        self.assertEqual(
            model_output,
            '{"answer_pre":"ok","evidence_pages":[2,221]}',
        )
        self.assertEqual(normalizations, [])
        self.assertEqual(status, "invalid_model_output_after_retries")
        self.assertEqual(provider.calls, 2)

    def test_illegal_output_preserves_exact_raw_whitespace(self) -> None:
        raw = '  {"answer_pre":"ok","evidence_pages":[99]}\n'
        model_output, normalizations, status = (
            prepare_pdf_output_for_storage(raw, allowed_pages=[1, 2])
        )
        self.assertEqual(model_output, raw)
        self.assertEqual(normalizations, [])
        self.assertEqual(status, "invalid_model_output_after_retries")

    def test_generation_retry_adds_json_correction_message(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.messages = []

            def generate(self, messages, max_new_tokens):
                self.messages.append(messages)
                if len(self.messages) == 1:
                    return "not json"
                return '{"answer_pre":"ok","evidence_pages":[1]}'

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=10,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        generate_with_retries(
            provider,
            [{"role": "user", "content": [{"type": "text", "text": "Q"}]}],
            args,
        )
        self.assertEqual(len(provider.messages[1]), 3)
        self.assertEqual(provider.messages[1][-2]["role"], "assistant")
        self.assertEqual(
            provider.messages[1][-2]["content"][0]["text"], "not json"
        )
        correction = provider.messages[1][-1]["content"][0]["text"]
        self.assertIn("previous response was invalid", correction)
        self.assertIn("pages that directly support the answer", correction)
        self.assertNotIn("at most 8", correction)

    def test_strict_validator_does_not_silently_repair_model_schema(self) -> None:
        invalid_outputs = (
            '{"answer":"ok","evidence_pages":[1]}',
            '{"answer_pre":"ok","evidence_pages":[1],"extra":true}',
            '{"answer_pre":"ok","evidence_pages":["1"]}',
            '{"answer_pre":"ok","evidence_pages":[1,1]}',
            '{"answer_pre":"ok","evidence_pages":[0]}',
        )
        for output in invalid_outputs:
            with self.subTest(output=output):
                self.assertFalse(structured_output_is_valid(output, True))

    def test_more_than_eight_evidence_pages_is_scored_without_retry(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.calls = 0
                self.cache_clears = 0

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                return '{"answer_pre":"ok","evidence_pages":[1,2,3,4,5,6,7,8,9,10,11]}'

            def clear_vision_cache(self):
                self.cache_clears += 1

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=32,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        output = generate_with_retries(provider, [], args)
        self.assertEqual(
            json.loads(output)["evidence_pages"], list(range(1, 12))
        )
        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.cache_clears, 0)

    def test_unsorted_evidence_set_is_accepted_without_identical_retry(
        self,
    ) -> None:
        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                return '{"answer_pre":"ok","evidence_pages":[7,6]}'

        provider = Provider()
        args = Namespace(
            max_qa_retries=5,
            max_new_tokens=512,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        output = generate_with_retries(
            provider, [], args, allowed_evidence_pages=list(range(1, 9))
        )
        self.assertEqual(json.loads(output)["evidence_pages"], [7, 6])
        self.assertEqual(provider.calls, 1)

    def test_generation_retry_corrects_runaway_evidence_page_list(self) -> None:
        class Provider:
            def __init__(self) -> None:
                self.messages = []
                self.overrides = []

            def set_generation_overrides(self, overrides):
                self.overrides.append(dict(overrides))

            def generate(self, messages, max_new_tokens):
                self.messages.append(messages)
                if len(self.messages) == 1:
                    return (
                        '{"answer_pre":"ok","evidence_pages":'
                        "[3,4,5,6,7,8,9,10,11,12"
                    )
                return '{"answer_pre":"ok","evidence_pages":[3,7]}'

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=512,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        output = generate_with_retries(provider, [], args)
        self.assertEqual(
            json.loads(output),
            {"answer_pre": "ok", "evidence_pages": [3, 7]},
        )
        correction = provider.messages[1][-1]["content"][0]["text"]
        self.assertIn("consecutive range is valid", correction)
        self.assertEqual(
            provider.overrides[1],
            {},
        )

    def test_repeated_malformed_output_is_preserved_after_finite_retries(
        self,
    ) -> None:
        class Provider:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                return (
                    '{"answer_pre":"semantically plausible","evidence_pages":'
                    "[3,4,5,6,7,8,9,10,11,12"
                )

        provider = Provider()
        args = Namespace(
            max_qa_retries=5,
            max_new_tokens=512,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        raw = generate_with_retries(provider, [], args)
        model_output, normalizations, status = (
            prepare_pdf_output_for_storage(
                raw, allowed_pages=list(range(1, 20))
            )
        )
        self.assertEqual(model_output, raw)
        self.assertEqual(normalizations, [])
        self.assertEqual(status, "invalid_model_output_after_retries")
        self.assertEqual(provider.calls, 5)


    def test_generation_retry_enables_repetition_fallback_only_after_loop(
        self,
    ) -> None:
        class Provider:
            def __init__(self) -> None:
                self.overrides = []
                self.calls = 0
                self.max_tokens = []

            def set_generation_overrides(self, overrides):
                self.overrides.append(dict(overrides))

            def generate(self, messages, max_new_tokens):
                self.calls += 1
                self.max_tokens.append(max_new_tokens)
                if self.calls == 1:
                    return '{"answer_pre":"' + "\u2009" * 64
                return '{"answer_pre":"linear decrease","evidence_pages":[4]}'

        provider = Provider()
        args = Namespace(
            max_qa_retries=2,
            max_new_tokens=512,
            require_evidence_pages=True,
            retry_backoff_seconds=0,
        )
        output = generate_with_retries(provider, [], args)
        self.assertIn("linear decrease", output)
        self.assertEqual(provider.overrides[0], {})
        self.assertEqual(
            provider.overrides[1],
            {
                "repetition_penalty": 1.12,
            },
        )
        self.assertEqual(provider.max_tokens, [512, 256])
        self.assertEqual(provider.overrides[-1], {})

    def test_local_provider_forwards_safe_generation_overrides(self) -> None:
        import torch

        class Inputs(dict):
            def to(self, _device):
                return self

        class Processor:
            eos_token_id = 0

            def __init__(self):
                self.template_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.template_kwargs = kwargs
                return "prompt"

            def __call__(self, values, return_tensors):
                return Inputs(input_ids=torch.tensor([[1, 2]]))

            def batch_decode(self, values, **kwargs):
                return ['{"answer_pre":"ok","evidence_pages":[1]}']

        class Model:
            device = "cpu"

            def __init__(self):
                self.kwargs = None

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[1, 2, 3]])

        provider = object.__new__(LocalTransformersProvider)
        provider.spec = {
            "model_class": "AutoModelForCausalLM",
            "enable_thinking": False,
        }
        provider.processor = Processor()
        provider.model = Model()
        provider._generation_overrides = {}
        provider.set_generation_overrides(
            {"repetition_penalty": 1.1, "no_repeat_ngram_size": 8}
        )
        provider.generate([], 32)
        self.assertEqual(provider.model.kwargs["repetition_penalty"], 1.1)
        self.assertEqual(provider.model.kwargs["no_repeat_ngram_size"], 8)
        self.assertIs(provider.processor.template_kwargs["enable_thinking"], False)


if __name__ == "__main__":
    unittest.main()
