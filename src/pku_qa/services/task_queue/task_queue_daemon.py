#!/usr/bin/env python3
"""Persistent scheduler, watchdog, recovery engine, and Codex bridge."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval_framework import atomic_write_json
from task_queue_manager import (
    ACTIVE_STATES,
    DEFAULT_DB,
    DEFAULT_INCIDENT_DIR,
    ROOT,
    Task,
    TaskStore,
    classify_failure,
    make_dynamic_a800,
    now,
    pid_alive,
    read_progress,
    task_stage_snapshots,
    task_log_snapshot,
    task_resource_class,
    tmux_session_exists,
)


LOCK_PATH = Path("/tmp/czj_project_task_queue.lock")
STATE_PATH = ROOT / "data/web/task_queue/daemon_state.json"
DAEMON_LOG = ROOT / "data/web/task_queue/daemon.log"
CODEX_TIMEOUT_SECONDS = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def completion_detected(task: Task) -> bool:
    paths = task.metadata.get("completion_paths", [])
    if paths and all(Path(path).exists() for path in paths):
        return True
    state_path = task.metadata.get("completion_state_path")
    expected = task.metadata.get("completion_state", "completed")
    if state_path and Path(state_path).exists():
        try:
            value = json.loads(Path(state_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return value.get("status") == expected
    return False


def sanitized_task_context(task: Task) -> dict[str, Any]:
    sensitive = ("key", "token", "secret", "password", "credential")
    env = {
        key: ("***" if any(word in key.lower() for word in sensitive) else value)
        for key, value in task.env.items()
    }
    command = [
        "***" if any(str(item).lower().startswith(f"--{word}=") for word in sensitive)
        else str(item)
        for item in task.command
    ]
    return {
        "id": task.id,
        "name": task.name,
        "command": command,
        "cwd": task.cwd,
        "env": env,
        "status": task.status,
        "restart_count": task.restart_count,
        "progress_path": task.progress_path,
        "validation_command": task.validation_command,
    }


class CodexRemediator:
    def __init__(self, store: TaskStore) -> None:
        self.store = store
        self.lock_path = ROOT / "data/web/task_queue/codex_remediation.lock"

    def allowed(self, task: Task) -> bool:
        if not task.auto_codex:
            return False
        if task.max_codex_attempts == 0:
            return True
        return task.codex_attempts < task.max_codex_attempts

    def remediate(
        self, task: Task, category: str, log_tail: str, incident_id: int
    ) -> bool:
        if not self.allowed(task):
            return False
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            attempt = task.codex_attempts + 1
            incident_dir = DEFAULT_INCIDENT_DIR / task.id
            incident_dir.mkdir(parents=True, exist_ok=True)
            output_path = incident_dir / f"codex_attempt_{attempt}.md"
            event_path = incident_dir / f"codex_attempt_{attempt}.jsonl"
            context = json.dumps(
                sanitized_task_context(task), ensure_ascii=False, indent=2
            )
            template_path = ROOT / "docs/prompts/MAINTENANCE_REMEDIATION.md"
            # Public checkouts do not include local maintenance prompts; in
            # that case keep ordinary queue recovery operational and skip the
            # optional external remediation step.
            if not template_path.is_file():
                return False
            template = template_path.read_text(encoding="utf-8")
            prompt = (
                template.replace("{{TASK_CONTEXT}}", context)
                .replace("{{FAILURE_CATEGORY}}", category)
                .replace("{{LOG_TAIL}}", log_tail[-12000:])
            )
            command = [
                "codex",
                "-a",
                "never",
                "exec",
                "--ignore-user-config",
                "-C",
                task.cwd,
                "-s",
                "workspace-write",
                "--skip-git-repo-check",
                "--ephemeral",
                "--json",
                "-o",
                str(output_path),
                "-",
            ]
            self.store.update(
                task.id,
                status="recovering",
                codex_attempts=attempt,
                heartbeat_at=now(),
            )
            with event_path.open("a", encoding="utf-8") as events:
                try:
                    result = subprocess.run(
                        command,
                        input=prompt,
                        text=True,
                        cwd=task.cwd,
                        stdout=events,
                        stderr=subprocess.STDOUT,
                        timeout=CODEX_TIMEOUT_SECONDS,
                        check=False,
                    )
                except Exception as exc:
                    self.store.resolve_incident(
                        incident_id,
                        f"Codex invocation failed: {type(exc).__name__}: {exc}",
                        attempt,
                    )
                    return False
            if result.returncode != 0:
                self.store.resolve_incident(
                    incident_id,
                    f"Codex returned {result.returncode}",
                    attempt,
                )
                return False
            validation = task.validation_command or [
                sys.executable,
                "-m",
                "pytest",
                "-q",
            ]
            validation_log = incident_dir / f"validation_{attempt}.log"
            with validation_log.open("a", encoding="utf-8") as handle:
                validated = subprocess.run(
                    validation,
                    cwd=task.cwd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    check=False,
                )
            if validated.returncode != 0:
                self.store.resolve_incident(
                    incident_id,
                    f"Codex changed code but validation returned {validated.returncode}",
                    attempt,
                )
                return False
            self.store.resolve_incident(
                incident_id,
                f"Codex attempt {attempt} passed validation",
                attempt,
            )
            return True


class QueueDaemon:
    def __init__(self, db_path: str | Path, poll_seconds: float = 5) -> None:
        self.store = TaskStore(db_path)
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.stop_requested = False
        self.remediator = CodexRemediator(self.store)
        self.children: dict[str, subprocess.Popen] = {}
        DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with DAEMON_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def request_stop(self, *_args) -> None:
        self.stop_requested = True

    def launch(self, task: Task) -> None:
        launched_at = now()
        command = [
            sys.executable,
            str(ROOT / "src/pku_qa/services/task_queue/task_queue_runner.py"),
            "--db",
            str(self.store.db_path),
            "--task-id",
            task.id,
        ]
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.children[task.id] = process
        self.store.update(
            task.id,
            status="starting",
            runner_pid=process.pid,
            pid=None,
            started_at=launched_at,
            finished_at=None,
            heartbeat_at=launched_at,
            progress_heartbeat_at=launched_at,
            progress_signature=None,
            last_error=None,
        )
        self.log(f"launched task={task.id} name={task.name} runner={process.pid}")

    def monitor_effective_progress(self, task: Task) -> str | None:
        """Track durable progress separately from the runner liveness pulse."""

        if task.kind != "batch":
            return None
        stages = task_stage_snapshots(task)
        if not stages:
            return None
        stage = next(
            (
                item
                for item in stages
                if item.get("status") == "failed"
            ),
            next(
                (
                    item
                    for item in stages
                    if item.get("status")
                    in {"running", "recovering", "starting"}
                ),
                None,
            ),
        )
        if stage is None:
            return None

        signature = json.dumps(
            {
                "stage": stage.get("id"),
                "completed": stage.get("completed"),
                "total": stage.get("total"),
                "percent": round(float(stage.get("percent") or 0), 6),
                "status": stage.get("status"),
            },
            sort_keys=True,
        )
        timestamp = now()
        scheduler_progress_at = stage.get("progress_heartbeat_at")
        effective_progress_at = task.progress_heartbeat_at
        changes: dict[str, object] = {}
        if signature != task.progress_signature:
            effective_progress_at = timestamp
            changes["progress_signature"] = signature
            changes["progress_heartbeat_at"] = timestamp
            try:
                previous_signature = json.loads(
                    task.progress_signature or "{}"
                )
            except json.JSONDecodeError:
                previous_signature = {}
            previous_completed = previous_signature.get("completed")
            current_completed = stage.get("completed")
            stage_advanced = (
                previous_signature.get("stage") not in {None, stage.get("id")}
            )
            durable_count_advanced = (
                isinstance(previous_completed, (int, float))
                and isinstance(current_completed, (int, float))
                and current_completed > previous_completed
            )
            if task.codex_attempts and (
                stage_advanced or durable_count_advanced
            ):
                changes["codex_attempts"] = 0
        elif isinstance(scheduler_progress_at, (int, float)) and (
            effective_progress_at is None
            or float(scheduler_progress_at) > effective_progress_at
        ):
            effective_progress_at = float(scheduler_progress_at)
            changes["progress_heartbeat_at"] = effective_progress_at
        if changes:
            self.store.update(task.id, **changes)

        work_state = str(stage.get("work_state") or "")
        stage_error = str(stage.get("error") or "").strip()
        if stage.get("status") == "failed" or work_state == "failed":
            return stage_error or (
                f"stage {stage.get('id')} reported failed"
            )

        # Resource scarcity is an explicit, healthy wait state. Its process
        # heartbeat must stay fresh, but lack of completed work is not a stall.
        if bool(stage.get("resource_waiting")) or (
            work_state == "waiting_for_resources"
        ):
            return None

        process_heartbeat_at = stage.get("process_heartbeat_at")
        scheduler_stale_seconds = float(
            task.metadata.get("scheduler_stale_seconds", 300)
        )
        if (
            isinstance(process_heartbeat_at, (int, float))
            and isinstance(task.started_at, (int, float))
            and float(process_heartbeat_at) < float(task.started_at)
        ):
            if timestamp - float(task.started_at) <= scheduler_stale_seconds:
                return None
            return (
                f"stage {stage.get('id')} did not refresh its scheduler "
                f"heartbeat within {scheduler_stale_seconds:.0f}s of launch"
            )
        if (
            isinstance(process_heartbeat_at, (int, float))
            and timestamp - float(process_heartbeat_at)
            > scheduler_stale_seconds
        ):
            return (
                f"stage {stage.get('id')} scheduler heartbeat stale for "
                f"{timestamp - float(process_heartbeat_at):.0f}s"
            )

        consecutive_failures = int(
            stage.get(
                "consecutive_worker_failures_without_progress", 0
            )
            or 0
        )
        progress_stall_seconds = float(
            task.metadata.get("progress_stall_seconds", 1800)
        )
        if (
            consecutive_failures > 0
            and work_state in {"worker_retry_backoff", "starting"}
            and effective_progress_at is not None
            and timestamp - effective_progress_at > progress_stall_seconds
        ):
            return (
                f"stage {stage.get('id')} made no durable progress for "
                f"{timestamp - effective_progress_at:.0f}s after "
                f"{consecutive_failures} worker failure(s)"
            )
        return None

    def fail_managed_task(self, task: Task, message: str) -> None:
        runner_pid = task.runner_pid
        if pid_alive(runner_pid):
            try:
                os.killpg(int(runner_pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.store.update(
            task.id,
            status="failed",
            last_error=message,
            finished_at=now(),
        )
        self.log(f"managed task failed task={task.id}: {message}")

    def monitor_adopted(self, task: Task) -> None:
        alive = pid_alive(task.pid) or tmux_session_exists(task.tmux_session)
        if alive:
            updates: dict[str, object] = {"heartbeat_at": now()}
            progress = read_progress(task.progress_path)
            if (
                task.status in {"waiting", "starting", "recovering"}
                and (progress or {}).get("status") == "running"
            ):
                updates["status"] = "running"
                self.log(
                    f"adopted task entered running state task={task.id}"
                )
            self.store.update(task.id, **updates)
            return
        if completion_detected(task):
            self.store.update(
                task.id,
                status="succeeded",
                pid=None,
                runner_pid=None,
                finished_at=now(),
                heartbeat_at=now(),
            )
            self.log(f"adopted task completed task={task.id}")
            return
        message = "adopted process/session disappeared before completion"
        self.store.update(
            task.id,
            status="failed",
            adopted=False,
            pid=None,
            runner_pid=None,
            last_error=message,
            finished_at=now(),
        )
        self.log(f"adopted task failed task={task.id}: {message}")

    def monitor_managed(self, task: Task) -> None:
        runner_pid = task.runner_pid
        if pid_alive(runner_pid):
            progress_failure = self.monitor_effective_progress(task)
            if progress_failure:
                self.fail_managed_task(task, progress_failure)
                return
            if task.timeout_seconds and task.started_at:
                if now() - task.started_at > task.timeout_seconds:
                    self.fail_managed_task(
                        task, "task timeout exceeded"
                    )
            return
        # The runner normally writes succeeded/failed before exiting. Give it
        # one cycle to commit, then classify an unclean disappearance.
        refreshed = self.store.get(task.id)
        if refreshed.status in {"succeeded", "failed", "cancelled", "paused"}:
            self.children.pop(task.id, None)
            return
        if refreshed.heartbeat_at and now() - refreshed.heartbeat_at < 2 * self.poll_seconds:
            return
        self.store.update(
            task.id,
            status="failed",
            pid=None,
            runner_pid=None,
            last_error="task runner disappeared without terminal status",
            finished_at=now(),
        )

    def builtin_recovery(
        self, task: Task, category: str
    ) -> tuple[bool, str]:
        command = task.command
        env = dict(task.env)
        if category == "oom":
            dynamic = (
                make_dynamic_a800(command)
                if any(
                    Path(part).name == "run_hard_eval.py"
                    for part in command
                )
                else command
            )
            env["GPU_REQUIRE_NAME"] = "A800"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["GPU_ALLOW_SHARED"] = "1"
            env["GPU_MIN_FREE_MEMORY_MIB"] = env.get(
                "GPU_MIN_FREE_MEMORY_MIB", "56000"
            )
            env["GPU_WAIT_POLL_SECONDS"] = env.get(
                "GPU_WAIT_POLL_SECONDS", "15"
            )
            self.store.update(task.id, command=dynamic, env=env)
            return (
                True,
                "checkpoint-safe OOM retry with shared A800 memory gate",
            )
        if category in {"gpu_busy", "transient"}:
            return True, f"safe delayed retry for {category}"
        if category == "data_error" and task.restart_count == 0:
            return True, "one clean checkpoint retry before code remediation"
        if category == "unknown" and task.restart_count == 0:
            return True, "one clean retry before unknown-failure code remediation"
        return False, "no deterministic built-in recovery"

    def recover_failed(self, task: Task) -> None:
        if now() < task.next_retry_at:
            return
        log_tail = task_log_snapshot(task, 64_000)["log"]
        failure_text = f"{task.last_error or ''}\n{log_tail}"
        category = classify_failure(failure_text)
        incident_id = self.store.add_incident(
            task.id,
            category,
            task.last_error or "task failed",
            log_tail,
        )
        built_in, resolution = self.builtin_recovery(task, category)
        refreshed = self.store.get(task.id)
        if built_in and refreshed.restart_count < refreshed.max_restarts:
            delay = min(
                300.0,
                refreshed.retry_delay_seconds
                * (2 ** min(refreshed.restart_count, 5)),
            )
            self.store.update(
                task.id,
                status="queued",
                restart_count=refreshed.restart_count + 1,
                next_retry_at=now() + delay,
                pid=None,
                runner_pid=None,
            )
            self.store.resolve_incident(incident_id, resolution)
            self.log(
                f"built-in recovery task={task.id} category={category}: {resolution}"
            )
            return
        if self.remediator.allowed(refreshed):
            success = self.remediator.remediate(
                refreshed, category, log_tail, incident_id
            )
            after = self.store.get(task.id)
            if success:
                self.store.update(
                    task.id,
                    status="queued",
                    restart_count=after.restart_count + 1,
                    next_retry_at=now() + after.retry_delay_seconds,
                    pid=None,
                    runner_pid=None,
                    last_error=None,
                )
                self.log(f"Codex remediation passed task={task.id}")
                return
            # A failed Codex attempt is retried with backoff. A zero
            # max_codex_attempts value means unbounded remediation, matching
            # the CLI and project queue contract.
            after = self.store.get(task.id)
            self.store.update(
                task.id,
                status="failed",
                next_retry_at=now() + min(900, 60 * max(1, after.codex_attempts)),
            )
            return
        self.store.update(task.id, status="blocked")
        self.log(
            f"blocked task={task.id}; recovery and Codex attempt budget exhausted"
        )

    def launch_ready(self) -> None:
        tasks = self.store.list_tasks()
        max_batch = int(self.store.setting("max_concurrent_batch", 1))
        max_cpu = int(
            self.store.setting("max_concurrent_cpu_batch", max_batch)
        )
        max_gpu = int(
            self.store.setting("max_concurrent_gpu_batch", max_batch)
        )
        active_batch = sum(
            task.status in ACTIVE_STATES and task.kind == "batch"
            for task in tasks
        )
        active_cpu = sum(
            task.status in ACTIVE_STATES
            and task.kind == "batch"
            and task_resource_class(task) == "cpu"
            for task in tasks
        )
        active_gpu = sum(
            task.status in ACTIVE_STATES
            and task.kind == "batch"
            and task_resource_class(task) in {"gpu", "gpu_high_vram"}
            for task in tasks
        )
        high_vram_active = any(
            task.status in ACTIVE_STATES
            and task.kind == "batch"
            and task_resource_class(task) == "gpu_high_vram"
            for task in tasks
        )
        for task in tasks:
            if (
                not task.enabled
                or task.status != "queued"
                or now() < task.next_retry_at
                or not self.store.dependencies_satisfied(task)
            ):
                continue
            if task.kind == "batch" and active_batch >= max_batch:
                continue
            resource_class = task_resource_class(task)
            if task.kind == "batch" and resource_class == "cpu":
                if active_cpu >= max_cpu:
                    continue
            elif task.kind == "batch" and resource_class == "gpu":
                if high_vram_active or active_gpu >= max_gpu:
                    continue
            elif task.kind == "batch" and resource_class == "gpu_high_vram":
                # Large-model work may use nearly a whole A800 or span cards.
                # It can overlap CPU work, but no other GPU batch.
                if active_gpu:
                    continue
            self.launch(task)
            if task.kind == "batch":
                active_batch += 1
                if resource_class == "cpu":
                    active_cpu += 1
                else:
                    active_gpu += 1
                    if resource_class == "gpu_high_vram":
                        high_vram_active = True

    def snapshot(self) -> None:
        tasks = [task.to_dict() for task in self.store.list_tasks()]
        atomic_write_json(
            STATE_PATH,
            {
                "status": "running",
                "pid": os.getpid(),
                "updated_at": now(),
                "tasks": tasks,
                "incidents": self.store.list_incidents()[:100],
            },
        )

    def tick(self) -> None:
        for task in self.store.list_tasks():
            if task.adopted and task.status in ACTIVE_STATES:
                self.monitor_adopted(task)
            elif not task.adopted and task.status in ACTIVE_STATES:
                self.monitor_managed(task)
        for task in self.store.list_tasks():
            if task.status == "failed" and task.enabled:
                self.recover_failed(task)
        self.launch_ready()
        self.snapshot()

    def run(self, once: bool = False) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        self.log(f"queue daemon started pid={os.getpid()} db={self.store.db_path}")
        while not self.stop_requested:
            try:
                self.tick()
            except Exception as exc:
                self.log(f"daemon tick error: {type(exc).__name__}: {exc}")
            if once:
                break
            time.sleep(self.poll_seconds)
        self.snapshot()
        self.log("queue daemon stopped")


def main() -> None:
    args = parse_args()
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another task_queue_daemon is already running")
        QueueDaemon(args.db, args.poll_seconds).run(once=args.once)


if __name__ == "__main__":
    main()
