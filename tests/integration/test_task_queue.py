from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from http.server import ThreadingHTTPServer

from task_queue_api import ApiHandler
from task_queue_daemon import CodexRemediator, QueueDaemon
from task_queue_manager import (
    ROOT,
    TaskStore,
    classify_failure,
    delete_task_results,
    make_dynamic_a800,
    read_progress,
    task_result_targets,
    task_stage_snapshots,
    task_log_snapshot,
)
from task_queue_runner import api_quota_pause_requested


class TaskStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = TaskStore(self.root / "tasks.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_api_quota_progress_requests_manual_pause(self) -> None:
        self.assertTrue(
            api_quota_pause_requested(
                {"status": "waiting_for_api_quota"}
            )
        )
        self.assertFalse(
            api_quota_pause_requested({"status": "failed"})
        )

    def test_crud_and_reordering(self) -> None:
        first = self.store.add(
            name="first",
            command=["true"],
            cwd=self.root,
            metadata={"description": "验证任务说明持久化"},
        )
        second = self.store.add(name="second", command=["true"], cwd=self.root)
        self.assertEqual([task.id for task in self.store.list_tasks()], [first.id, second.id])
        self.assertEqual(
            self.store.get(first.id).to_dict()["description"],
            "验证任务说明持久化",
        )
        self.store.move(second.id, 0)
        self.assertEqual(self.store.list_tasks()[0].id, second.id)
        updated = self.store.update(first.id, name="renamed", priority=5)
        self.assertEqual(updated.name, "renamed")
        self.store.delete(second.id)
        self.assertEqual([task.id for task in self.store.list_tasks()], [first.id])

    def test_active_task_requires_force_delete(self) -> None:
        task = self.store.add(
            name="active",
            command=["true"],
            cwd=self.root,
            status="running",
        )
        with self.assertRaises(RuntimeError):
            self.store.delete(task.id)
        self.store.delete(task.id, force=True)

    def test_delete_creates_restorable_history_record(self) -> None:
        task = self.store.add(
            name="finished",
            command=["true"],
            cwd=self.root,
            status="succeeded",
        )

        deletion = self.store.delete(task.id)

        self.assertEqual(deletion["task_id"], task.id)
        self.assertEqual(deletion["task_snapshot"]["status"], "succeeded")
        self.assertEqual(len(self.store.list_deletions()), 1)
        with self.assertRaises(KeyError):
            self.store.get(task.id)

        restored = self.store.restore_deletion(deletion["id"])

        self.assertEqual(restored.id, task.id)
        self.assertEqual(restored.status, "succeeded")
        self.assertFalse(restored.adopted)
        self.assertIsNone(restored.pid)
        self.assertIsNotNone(
            self.store.get_deletion(deletion["id"])["restored_at"]
        )

    def test_delete_task_results_only_removes_declared_project_outputs(self) -> None:
        output_root = ROOT / "data/results/evaluations"
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="delete-results-test-",
            dir=output_root,
        ) as tmp:
            output = Path(tmp)
            (output / "checkpoint.json").write_text("{}", encoding="utf-8")
            task = self.store.add(
                name="finished",
                command=[
                    sys.executable,
                    "run.py",
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                status="succeeded",
            )

            self.assertEqual(task_result_targets(task), [output.resolve()])
            deleted, errors = delete_task_results(task)

            self.assertEqual(deleted, [str(output.resolve())])
            self.assertEqual(errors, [])
            self.assertFalse(output.exists())

    def test_dependencies_require_success(self) -> None:
        first = self.store.add(name="first", command=["true"], cwd=self.root)
        second = self.store.add(
            name="second",
            command=["true"],
            cwd=self.root,
            depends_on=[first.id],
        )
        self.assertFalse(self.store.dependencies_satisfied(second))
        self.store.update(first.id, status="succeeded")
        self.assertTrue(self.store.dependencies_satisfied(second))

    def test_running_service_satisfies_dependency(self) -> None:
        service = self.store.add(
            name="api",
            command=["true"],
            cwd=self.root,
            kind="service",
            status="running",
            pid=os.getpid(),
        )
        client = self.store.add(
            name="client",
            command=["true"],
            cwd=self.root,
            depends_on=[service.id],
        )
        self.assertTrue(self.store.dependencies_satisfied(client))

    def test_progress_reads_adaptive_scheduler_shape(self) -> None:
        path = self.root / "state.json"
        path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "queue": {"completed_papers": 25, "total_papers": 100},
                    "workers": {"2": {"pid": 123}},
                }
            ),
            encoding="utf-8",
        )
        progress = read_progress(str(path))
        self.assertEqual(progress["percent"], 25.0)
        self.assertEqual(progress["workers"]["2"]["pid"], 123)

    def test_single_stage_task_preserves_real_progress_denominator(self) -> None:
        path = self.root / "challenge_state.json"
        path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed": 2000,
                    "total": 8460,
                    "process_heartbeat_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        task = self.store.add(
            name="challenge",
            command=["true"],
            cwd=self.root,
            progress_path=path,
            status="running",
        )
        progress = task.to_dict()["progress"]
        self.assertEqual(progress["completed"], 2000)
        self.assertEqual(progress["total"], 8460)

    def test_progress_follows_freshest_running_pipeline_stage(self) -> None:
        queue_root = self.root / ".adaptive_queue"
        stage_4b = queue_root / "inference_4B" / "scheduler_state.json"
        stage_8b = queue_root / "inference_8B" / "scheduler_state.json"
        stage_4b.parent.mkdir(parents=True)
        stage_8b.parent.mkdir(parents=True)
        stage_4b.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "queue": {
                        "completed_papers": 100,
                        "total_papers": 100,
                    },
                }
            ),
            encoding="utf-8",
        )
        stage_8b.write_text(
            json.dumps(
                {
                    "status": "running",
                    "queue": {
                        "completed_papers": 10,
                        "total_papers": 100,
                    },
                    "workers": {"3": {"pid": 456}},
                }
            ),
            encoding="utf-8",
        )

        progress = read_progress(str(stage_4b))

        self.assertEqual(progress["stage"], "inference_8B")
        self.assertEqual(progress["path"], str(stage_8b))
        self.assertEqual(progress["configured_path"], str(stage_4b))
        self.assertEqual(progress["percent"], 10.0)
        self.assertEqual(progress["workers"]["3"]["pid"], 456)

    def test_hard_eval_is_split_into_atomic_stage_progress(self) -> None:
        output = self.root / "formal"
        stage_4b = output / ".adaptive_queue/inference_4B/scheduler_state.json"
        stage_8b = output / ".adaptive_queue/inference_8B/scheduler_state.json"
        stage_4b.parent.mkdir(parents=True)
        stage_8b.parent.mkdir(parents=True)
        stage_4b.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "queue": {
                        "completed_papers": 100,
                        "total_papers": 100,
                    },
                }
            ),
            encoding="utf-8",
        )
        stage_8b.write_text(
            json.dumps(
                {
                    "status": "running",
                    "queue": {
                        "completed_papers": 50,
                        "total_papers": 100,
                    },
                    "workers": {"2": {"pid": 123}},
                }
            ),
            encoding="utf-8",
        )
        task = self.store.add(
            name="formal",
            command=[
                sys.executable,
                "src/pku_qa/evaluation/run_hard_eval.py",
                "--output-dir",
                str(output),
            ],
            cwd=self.root,
            status="running",
        )

        payload = task.to_dict()

        self.assertEqual(
            [stage["id"] for stage in payload["stages"]],
            [
                "inference_4B",
                "inference_8B",
                "judge_4B",
                "judge_8B",
                "report",
            ],
        )
        self.assertEqual(payload["progress"]["stage"], "inference_8B")
        self.assertEqual(payload["progress"]["percent"], 30.0)
        self.assertEqual(payload["progress"]["completed"], 1)
        self.assertEqual(task_stage_snapshots(task)[1]["percent"], 50.0)

    def test_stage_log_snapshot_filters_worker_logs(self) -> None:
        output = self.root / "formal"
        inference = output / "adaptive_logs/inference_8B"
        judge = output / "adaptive_logs/judge_4B"
        inference.mkdir(parents=True)
        judge.mkdir(parents=True)
        (inference / "gpu_2.log").write_text("inference line", encoding="utf-8")
        (judge / "gpu_2.log").write_text("judge line", encoding="utf-8")
        task = self.store.add(
            name="formal",
            command=[
                sys.executable,
                "src/pku_qa/evaluation/run_hard_eval.py",
                "--output-dir",
                str(output),
            ],
            cwd=self.root,
        )

        snapshot = task_log_snapshot(task, stage_id="inference_8B")

        self.assertIn("inference line", snapshot["current_log"])
        self.assertNotIn("judge line", snapshot["log"])
        self.assertEqual(snapshot["stage_id"], "inference_8B")

    def test_wrapper_pipeline_aggregates_configured_hard_eval_stages(self) -> None:
        wrapper_state = self.root / "wrapper.json"
        wrapper_state.write_text(
            json.dumps(
                {
                    "status": "running",
                    "stages": [
                        {
                            "name": "closedbook",
                            "status": "completed",
                        },
                        {"name": "full_pdf", "status": "running"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        closedbook = self.root / "closedbook"
        full_pdf = self.root / "full"
        closedbook.mkdir()
        (closedbook / "final_report.json").write_text("{}", encoding="utf-8")
        running = (
            full_pdf
            / ".adaptive_queue/inference_4B/scheduler_state.json"
        )
        running.parent.mkdir(parents=True)
        running.write_text(
            json.dumps(
                {
                    "status": "running",
                    "queue": {
                        "completed_papers": 50,
                        "total_papers": 100,
                    },
                    "workers": {"2": {"pid": 123}},
                }
            ),
            encoding="utf-8",
        )
        task = self.store.add(
            name="wrapper",
            command=["python", "wrapper.py"],
            cwd=self.root,
            status="running",
            progress_path=wrapper_state,
            metadata={
                "atomic_tasks": [
                    {
                        "id": "closedbook",
                        "name": "Closed book",
                        "output_dir": str(closedbook),
                    },
                    {
                        "id": "full_pdf",
                        "name": "Full PDF",
                        "output_dir": str(full_pdf),
                    },
                ]
            },
        )

        payload = task.to_dict()

        self.assertEqual(payload["stages"][0]["status"], "succeeded")
        self.assertEqual(payload["stages"][0]["percent"], 100.0)
        self.assertEqual(payload["stages"][1]["status"], "running")
        self.assertEqual(payload["stages"][1]["percent"], 10.0)
        self.assertEqual(payload["progress"]["percent"], 55.0)
        self.assertEqual(payload["progress"]["workers"]["2"]["pid"], 123)

    def test_wrapper_stage_logs_include_nested_workers_and_percent_each_line(
        self,
    ) -> None:
        wrapper_state = self.root / "wrapper.json"
        wrapper_state.write_text(
            json.dumps(
                {
                    "status": "running",
                    "stages": [
                        {"name": "full_pdf", "status": "running"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "full"
        state_path = (
            output
            / ".adaptive_queue/inference_4B/scheduler_state.json"
        )
        state_path.parent.mkdir(parents=True)
        state_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "queue": {
                        "completed_papers": 50,
                        "total_papers": 100,
                    },
                }
            ),
            encoding="utf-8",
        )
        worker_log = output / "adaptive_logs/inference_4B/gpu_2.log"
        worker_log.parent.mkdir(parents=True)
        worker_log.write_text(
            "event=qa_started paper=p1 qa=q1 progress=1/3\n",
            encoding="utf-8",
        )
        task = self.store.add(
            name="wrapper",
            command=["python", "wrapper.py"],
            cwd=self.root,
            status="running",
            progress_path=wrapper_state,
            metadata={
                "atomic_tasks": [
                    {
                        "id": "full_pdf",
                        "name": "Full PDF",
                        "output_dir": str(output),
                    }
                ]
            },
        )

        snapshot = task_log_snapshot(task, stage_id="full_pdf")

        self.assertIn(str(worker_log), snapshot["current_log"])
        self.assertEqual(snapshot["sources"][0]["progress_percent"], 10.0)
        self.assertEqual(task_stage_snapshots(task)[0]["log_count"], 1)
        for line in snapshot["current_log"].splitlines():
            if line.strip():
                self.assertIn("progress_percent=", line)

    def test_adopted_task_logs_fall_back_to_latest_dynamic_stage(self) -> None:
        output = self.root / "output"
        stage_4b = output / "adaptive_logs/inference_4B"
        stage_8b = output / "adaptive_logs/inference_8B"
        stage_4b.mkdir(parents=True)
        stage_8b.mkdir(parents=True)
        old_log = stage_4b / "gpu_2.log"
        current_log = stage_8b / "gpu_3.log"
        old_log.write_text("old-stage-line\n", encoding="utf-8")
        current_log.write_text("current-stage-line\n", encoding="utf-8")
        os.utime(old_log, (100, 100))
        os.utime(current_log, (200, 200))
        task = self.store.add(
            name="adopted",
            command=["python", "run.py", "--output-dir", str(output)],
            cwd=self.root,
            adopted=True,
            metadata={"source_logs": [str(old_log)]},
        )

        snapshot = task_log_snapshot(task, 8_000)

        self.assertTrue(snapshot["primary_log_missing"])
        self.assertEqual(snapshot["sources"][0]["stage"], "inference_8B")
        self.assertTrue(snapshot["sources"][0]["current"])
        self.assertIn("current-stage-line", snapshot["log"])
        self.assertIn("old-stage-line", snapshot["log"])
        self.assertIn("current-stage-line", snapshot["current_log"])
        self.assertNotIn("old-stage-line", snapshot["current_log"])


class FailureRecoveryTests(unittest.TestCase):
    def test_failure_classifier(self) -> None:
        self.assertEqual(classify_failure("CUDA out of memory"), "oom")
        self.assertEqual(classify_failure("No space left on device"), "disk_full")
        self.assertEqual(
            classify_failure(
                "WorkerProgressStallError: 3 consecutive worker failures"
            ),
            "worker_failure",
        )
        self.assertEqual(classify_failure("mystery"), "unknown")

    def test_static_hard_eval_is_converted_to_dynamic_a800(self) -> None:
        command = [
            "python",
            "src/pku_qa/evaluation/run_hard_eval.py",
            "--gpus-4b",
            "2",
            "2",
            "--gpus-8b",
            "2",
            "--judge-gpus",
            "2",
            "--dpi",
            "144",
        ]
        updated = make_dynamic_a800(command)
        self.assertNotIn("--gpus-4b", updated)
        self.assertNotIn("--judge-gpus", updated)
        self.assertIn("--dynamic-a800", updated)
        self.assertEqual(updated.count("2"), 1)

    def test_oom_recovery_applies_shared_memory_gate_to_any_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.sqlite3")
            task = store.add(
                name="generic gpu task",
                command=[sys.executable, "gpu_program.py"],
                cwd=tmp,
                env={},
                auto_codex=False,
            )
            daemon = QueueDaemon(store.db_path)

            recovered, message = daemon.builtin_recovery(task, "oom")
            updated = store.get(task.id)

            self.assertTrue(recovered)
            self.assertIn("shared A800", message)
            self.assertEqual(updated.command, task.command)
            self.assertEqual(updated.env["GPU_REQUIRE_NAME"], "A800")
            self.assertEqual(updated.env["GPU_ALLOW_SHARED"], "1")
            self.assertEqual(
                updated.env["GPU_MIN_FREE_MEMORY_MIB"],
                "56000",
            )
            self.assertEqual(
                updated.env["PYTORCH_CUDA_ALLOC_CONF"],
                "expandable_segments:True",
            )

    def test_worker_failure_skips_blind_retry_and_invokes_codex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.sqlite3")
            task = store.add(
                name="stalled worker",
                command=["false"],
                cwd=tmp,
                status="failed",
                max_codex_attempts=1,
            )
            task = store.update(
                task.id,
                last_error=(
                    "WorkerProgressStallError: "
                    "3 consecutive worker failures"
                ),
            )
            daemon = QueueDaemon(store.db_path)
            daemon.remediator.remediate = mock.Mock(return_value=True)

            daemon.recover_failed(task)

            daemon.remediator.remediate.assert_called_once()
            call = daemon.remediator.remediate.call_args
            self.assertEqual(call.args[1], "worker_failure")
            self.assertEqual(store.get(task.id).status, "queued")


class ResourceLaneSchedulingTests(unittest.TestCase):
    def test_cpu_launches_while_high_vram_waits_for_active_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.sqlite3")
            store.set_setting("max_concurrent_batch", 4)
            store.set_setting("max_concurrent_cpu_batch", 2)
            store.set_setting("max_concurrent_gpu_batch", 2)
            store.add(
                name="active gpu",
                command=["gpu-worker"],
                cwd=tmp,
                status="running",
                metadata={"resource_class": "gpu"},
            )
            cpu = store.add(
                name="cpu summary",
                command=["cpu-worker"],
                cwd=tmp,
                priority=20,
                metadata={"resource_class": "cpu"},
            )
            store.add(
                name="27b",
                command=["large-model"],
                cwd=tmp,
                priority=10,
                metadata={"resource_class": "gpu_high_vram"},
            )
            daemon = QueueDaemon(store.db_path)
            launched = []
            daemon.launch = lambda task: launched.append(task.id)

            daemon.launch_ready()

            self.assertEqual(launched, [cpu.id])

    def test_high_vram_blocks_later_regular_gpu_but_not_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.sqlite3")
            store.set_setting("max_concurrent_batch", 4)
            store.set_setting("max_concurrent_cpu_batch", 2)
            store.set_setting("max_concurrent_gpu_batch", 2)
            high = store.add(
                name="27b",
                command=["large-model"],
                cwd=tmp,
                priority=30,
                metadata={"resource_class": "gpu_high_vram"},
            )
            cpu = store.add(
                name="cpu",
                command=["summary"],
                cwd=tmp,
                priority=20,
                metadata={"resource_class": "cpu"},
            )
            store.add(
                name="ordinary gpu",
                command=["small-model"],
                cwd=tmp,
                priority=10,
                metadata={"resource_class": "gpu"},
            )
            daemon = QueueDaemon(store.db_path)
            launched = []
            daemon.launch = lambda task: launched.append(task.id)

            daemon.launch_ready()

            self.assertEqual(launched, [high.id, cpu.id])

    def test_daemon_launches_detached_runner_to_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite3"
            store = TaskStore(db)
            marker = root / "done.txt"
            task = store.add(
                name="quick",
                command=[
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).write_text('ok')",
                ],
                cwd=root,
                auto_codex=False,
            )
            daemon = QueueDaemon(db, poll_seconds=1)
            daemon.tick()
            deadline = time.time() + 10
            while time.time() < deadline:
                current = store.get(task.id)
                if current.status == "succeeded":
                    break
                time.sleep(0.2)
                daemon.tick()
            self.assertEqual(store.get(task.id).status, "succeeded")
            self.assertEqual(marker.read_text(), "ok")

    def test_adopted_waiter_is_marked_running_from_progress_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            progress = root / "progress.json"
            progress.write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )
            task = store.add(
                name="adopted",
                command=["true"],
                cwd=root,
                status="waiting",
                adopted=True,
                pid=os.getpid(),
                progress_path=progress,
                auto_codex=False,
            )
            daemon = QueueDaemon(store.db_path)

            daemon.monitor_adopted(task)

            self.assertEqual(store.get(task.id).status, "running")

    def test_resource_wait_does_not_trigger_progress_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            progress = root / "scheduler_state.json"
            progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "work_state": "waiting_for_resources",
                        "resource_waiting": True,
                        "process_heartbeat_at": time.time(),
                        "progress_heartbeat_at": 1,
                        "queue": {
                            "completed_papers": 9,
                            "total_papers": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = store.add(
                name="gpu wait",
                command=["true"],
                cwd=root,
                status="running",
                progress_path=progress,
                auto_codex=False,
            )
            store.update(
                task.id,
                progress_heartbeat_at=1,
                progress_signature="old",
            )
            daemon = QueueDaemon(store.db_path)

            failure = daemon.monitor_effective_progress(
                store.get(task.id)
            )

            self.assertIsNone(failure)
            self.assertGreater(
                store.get(task.id).progress_heartbeat_at,
                1,
            )

    def test_stale_checkpoint_heartbeat_gets_launch_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            progress = root / "progress.json"
            progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "process_heartbeat_at": 1,
                        "progress_heartbeat_at": 1,
                        "completed": 4,
                        "total": 500,
                        "percent": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            task = store.add(
                name="resumed generation",
                command=["true"],
                cwd=root,
                status="running",
                progress_path=progress,
                auto_codex=False,
                metadata={"scheduler_stale_seconds": 300},
            )
            store.update(
                task.id,
                started_at=time.time(),
                progress_heartbeat_at=time.time(),
                progress_signature=None,
            )
            daemon = QueueDaemon(store.db_path)

            failure = daemon.monitor_effective_progress(store.get(task.id))

            self.assertIsNone(failure)

    def test_stale_checkpoint_heartbeat_fails_after_launch_grace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            progress = root / "progress.json"
            progress.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "process_heartbeat_at": 1,
                        "progress_heartbeat_at": 1,
                        "completed": 4,
                        "total": 500,
                        "percent": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            task = store.add(
                name="stuck resumed generation",
                command=["true"],
                cwd=root,
                status="running",
                progress_path=progress,
                auto_codex=False,
                metadata={"scheduler_stale_seconds": 1},
            )
            store.update(
                task.id,
                started_at=time.time() - 2,
                progress_heartbeat_at=time.time() - 2,
                progress_signature=None,
            )
            daemon = QueueDaemon(store.db_path)

            failure = daemon.monitor_effective_progress(store.get(task.id))

            self.assertIn("did not refresh", failure or "")

    def test_failed_scheduler_state_propagates_to_queue_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            progress = root / "scheduler_state.json"
            progress.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "work_state": "failed",
                        "error": (
                            "WorkerProgressStallError: "
                            "3 consecutive worker failures"
                        ),
                        "queue": {
                            "completed_papers": 9,
                            "total_papers": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            task = store.add(
                name="broken gpu stage",
                command=["true"],
                cwd=root,
                status="running",
                progress_path=progress,
                auto_codex=False,
            )
            daemon = QueueDaemon(store.db_path)

            failure = daemon.monitor_effective_progress(task)

            self.assertIn("WorkerProgressStallError", failure)

    def test_codex_remediator_is_bounded_and_runs_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            task = store.add(
                name="broken",
                command=["false"],
                cwd=root,
                validation_command=["true"],
                max_codex_attempts=1,
            )
            incident = store.add_incident(task.id, "unknown", "broken")
            completed = mock.Mock(returncode=0)
            with mock.patch("task_queue_daemon.subprocess.run", return_value=completed) as run:
                remediator = CodexRemediator(store)
                remediator.lock_path = root / "codex.lock"
                # Keep test artifacts inside the temporary root.
                with mock.patch("task_queue_daemon.DEFAULT_INCIDENT_DIR", root / "incidents"):
                    self.assertTrue(
                        remediator.remediate(task, "unknown", "trace", incident)
                    )
            self.assertEqual(store.get(task.id).codex_attempts, 1)
            self.assertEqual(run.call_count, 2)
            self.assertFalse(remediator.allowed(store.get(task.id)))

    def test_zero_codex_attempt_setting_is_unbounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.sqlite3")
            task = store.add(
                name="broken",
                command=["false"],
                cwd=tmp,
                max_codex_attempts=0,
            )
            remediator = CodexRemediator(store)
            self.assertTrue(remediator.allowed(task))
            task = store.update(task.id, codex_attempts=1000)
            self.assertTrue(remediator.allowed(task))

    def test_codex_invocation_ignores_user_mcp_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = TaskStore(root / "tasks.sqlite3")
            task = store.add(
                name="broken",
                command=["false"],
                cwd=root,
                validation_command=["true"],
                max_codex_attempts=1,
            )
            incident = store.add_incident(task.id, "unknown", "broken")
            completed = mock.Mock(returncode=0)
            with mock.patch(
                "task_queue_daemon.subprocess.run",
                return_value=completed,
            ) as run:
                remediator = CodexRemediator(store)
                remediator.lock_path = root / "codex.lock"
                with mock.patch(
                    "task_queue_daemon.DEFAULT_INCIDENT_DIR",
                    root / "incidents",
                ):
                    self.assertTrue(
                        remediator.remediate(
                            task, "unknown", "trace", incident
                        )
                    )
            command = run.call_args_list[0].args[0]
            self.assertIn("--ignore-user-config", command)
            self.assertEqual(command[-1], "-")

    def test_durable_progress_resets_codex_attempt_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "scheduler_state.json"
            state.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "queue": {
                            "completed_papers": 2,
                            "total_papers": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )
            store = TaskStore(root / "tasks.sqlite3")
            task = store.add(
                name="recovering",
                command=["true"],
                cwd=root,
                progress_path=state,
                status="running",
            )
            task = store.update(
                task.id,
                codex_attempts=2,
                progress_signature=json.dumps(
                    {
                        "stage": "main",
                        "completed": 1,
                        "total": 10,
                        "percent": 10.0,
                        "status": "running",
                    },
                    sort_keys=True,
                ),
            )
            daemon = QueueDaemon(store.db_path)

            daemon.monitor_effective_progress(task)

            self.assertEqual(store.get(task.id).codex_attempts, 0)


class TaskQueueApiTests(unittest.TestCase):
    def test_api_crud_and_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = TaskStore(Path(tmp) / "tasks.sqlite3")
            server = ThreadingHTTPServer(("127.0.0.1", 0), ApiHandler)
            server.store = store
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"

            def call(path: str, method: str = "GET", body=None):
                data = None if body is None else json.dumps(body).encode()
                request = urllib.request.Request(
                    base + path,
                    data=data,
                    method=method,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    return json.loads(response.read())

            try:
                self.assertEqual(call("/api/health")["status"], "ok")
                created = call(
                    "/api/tasks",
                    "POST",
                    {
                        "name": "api task",
                        "command": [sys.executable, "-c", "print('ok')"],
                        "cwd": tmp,
                        "auto_codex": False,
                        "description": "运行一个 API 测试任务",
                    },
                )
                task_id = created["id"]
                self.assertEqual(created["max_codex_attempts"], 0)
                self.assertEqual(
                    created["description"],
                    "运行一个 API 测试任务",
                )
                self.assertEqual(call("/api/tasks")["tasks"][0]["id"], task_id)
                fallback_log = Path(tmp) / "adopted.log"
                fallback_log.write_text("visible fallback log", encoding="utf-8")
                store.update(
                    task_id,
                    metadata={
                        "description": created["description"],
                        "source_logs": [str(fallback_log)],
                    },
                )
                logs = call(f"/api/tasks/{task_id}/logs?bytes=4096")
                self.assertIn("visible fallback log", logs["log"])
                self.assertEqual(logs["sources"][0]["path"], str(fallback_log))
                changed = call(
                    f"/api/tasks/{task_id}",
                    "PATCH",
                    {
                        "name": "renamed",
                        "description": "已更新的任务说明",
                    },
                )
                self.assertEqual(changed["name"], "renamed")
                self.assertEqual(changed["description"], "已更新的任务说明")
                self.assertEqual(
                    changed["metadata"]["source_logs"],
                    [str(fallback_log)],
                )
                store.update(
                    task_id,
                    status="running",
                    pid=987654,
                    runner_pid=987653,
                )
                def assert_pause_precedes_signal(_task) -> None:
                    self.assertEqual(store.get(task_id).status, "paused")

                with mock.patch(
                    "task_queue_api.terminate_task",
                    side_effect=assert_pause_precedes_signal,
                ) as terminate:
                    paused = call(
                        f"/api/tasks/{task_id}/actions/pause",
                        "POST",
                        {},
                    )
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(terminate.call_args.args[0].pid, 987654)
                deleted = call(f"/api/tasks/{task_id}", "DELETE")
                self.assertEqual(deleted["deleted"], task_id)
                self.assertFalse(deleted["files_deleted"])
                deletion_records = call("/api/deletions")
                self.assertEqual(
                    deletion_records["deletions"][0]["task_id"],
                    task_id,
                )
                restored = call(
                    (
                        "/api/deletions/"
                        f"{deletion_records['deletions'][0]['id']}/restore"
                    ),
                    "POST",
                    {},
                )
                self.assertEqual(restored["restored"], task_id)
                self.assertEqual(
                    call(f"/api/tasks/{task_id}")["name"],
                    "renamed",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


class TaskQueueCliTests(unittest.TestCase):
    def test_cli_add_list_move_update_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "tasks.sqlite3")
            cli = str(Path(__file__).resolve().parents[2] / "src/pku_qa/services/task_queue/task_queue_cli.py")
            added = subprocess.run(
                [
                    sys.executable,
                    cli,
                    "--db",
                    db,
                    "add",
                    "cli task",
                    "--cwd",
                    tmp,
                    "--description",
                    "CLI 新建任务说明",
                    "--",
                    sys.executable,
                    "-c",
                    "print('ok')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            added_task = json.loads(added.stdout)
            task_id = added_task["id"]
            self.assertEqual(added_task["description"], "CLI 新建任务说明")
            dependency = TaskStore(db).add(
                name="dependency", command=["true"], cwd=tmp
            )
            listed = subprocess.run(
                [sys.executable, cli, "--db", db, "list", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(listed.stdout)[0]["id"], task_id)
            subprocess.run(
                [
                    sys.executable,
                    cli,
                    "--db",
                    db,
                    "update",
                    task_id,
                    "--name",
                    "updated",
                    "--description",
                    "CLI 更新任务说明",
                    "--depends-on",
                    dependency.id,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            self.assertEqual(
                TaskStore(db).get(task_id).to_dict()["description"],
                "CLI 更新任务说明",
            )
            self.assertEqual(
                TaskStore(db).get(task_id).depends_on,
                [dependency.id],
            )
            subprocess.run(
                [sys.executable, cli, "--db", db, "delete", task_id],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            TaskStore(db).delete(dependency.id)
            self.assertEqual(TaskStore(db).list_tasks(), [])


if __name__ == "__main__":
    unittest.main()
