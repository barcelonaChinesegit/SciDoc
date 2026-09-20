#!/usr/bin/env python3
"""Dynamic A800-only worker pool with repository locks and auto-recovery."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from eval_framework import atomic_write_json
from gpu_reservation import (
    GPU_LOCK_DIR,
    _gpu_compute_processes,
    _gpu_inventory,
)
from progress_logging import progress_percent


@dataclass
class GpuLease:
    gpu_id: str
    handle: object

    def close(self) -> None:
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


@dataclass
class Worker:
    gpu_id: str
    process: subprocess.Popen
    lease: GpuLease
    started_at: float
    progress_completed_at_start: int = 0
    log_handle: object | None = None


class PoolShutdownSignal(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"GPU pool received signal {signum}")
        self.signum = signum


class QuarantinedWorkError(RuntimeError):
    """Raised when every unfinished queue item is quarantined."""


class WorkerProgressStallError(RuntimeError):
    """Raised after repeated worker failures without durable progress."""


RESOURCE_WAIT_REASONS = {
    "foreign_process",
    "insufficient_free_memory",
    "repository_lock",
    "same_work_queue_process",
}


def _command_option(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return command[index + 1]


def _process_command(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    ]


def same_work_queue_processes(
    processes: list[dict[str, str]], command: list[str]
) -> list[dict[str, str]]:
    queue_dir = _command_option(command, "--work-queue-dir")
    if not queue_dir:
        return []
    matches = []
    for process in processes:
        process_command = _process_command(int(process["pid"]))
        if _command_option(process_command, "--work-queue-dir") == queue_dir:
            matches.append(process)
    return matches


def a800_gpu_ids(
    allowed: Iterable[str] | None = None,
    *,
    required_name: str = "A800",
) -> list[str]:
    inventory = _gpu_inventory()
    allowed_set = None if allowed is None else {str(item) for item in allowed}
    return sorted(
        [
            gpu_id
            for gpu_id, gpu in inventory.items()
            if required_name.lower() in gpu["name"].lower()
            and (allowed_set is None or gpu_id in allowed_set)
        ],
        key=int,
    )


def try_gpu_lease(gpu_id: str) -> GpuLease | None:
    GPU_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    handle = (GPU_LOCK_DIR / f"gpu_{gpu_id}.lock").open(
        "a+", encoding="utf-8"
    )
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return GpuLease(gpu_id=gpu_id, handle=handle)


def gpu_foreign_processes(
    gpu_id: str, ignored_pids: set[int] | None = None
) -> list[dict[str, str]]:
    inventory = _gpu_inventory()
    ignored = set(ignored_pids or set())
    return [
        process
        for process in _gpu_compute_processes().get(
            inventory[gpu_id]["uuid"], []
        )
        if int(process["pid"]) not in ignored
    ]


def gpu_free_memory_mib(gpu_id: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "nvidia-smi memory query failed"
        )
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[0] == str(gpu_id):
            return int(parts[1])
    raise KeyError(f"GPU {gpu_id} missing from memory query")


def _descendant_pids(pid: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        capture_output=True,
        text=True,
        check=False,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        try:
            child, parent = map(int, line.split())
        except (ValueError, TypeError):
            continue
        children.setdefault(parent, []).append(child)
    found = {pid}
    pending = [pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, []):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


class AdaptiveGpuPool:
    """Continuously add free A800s and restart failed workers."""

    def __init__(
        self,
        *,
        command_factory: Callable[[str, int], list[str]],
        is_complete: Callable[[], bool],
        status_factory: Callable[[], dict],
        allowed_gpus: Iterable[str] | None = None,
        poll_seconds: float = 15,
        state_path: str | Path | None = None,
        log_dir: str | Path | None = None,
        max_restarts_per_gpu: int = 50,
        max_restart_backoff_seconds: float = 300,
        max_failures_without_progress: int = 3,
        startup_grace_seconds: float = 45,
        allow_shared_gpus: bool = False,
        min_free_memory_mib: int = 0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command_factory = command_factory
        self.is_complete = is_complete
        self.status_factory = status_factory
        self.gpu_ids = a800_gpu_ids(allowed_gpus)
        if not self.gpu_ids:
            raise RuntimeError("No allowed A800 GPU found; A40/CPU fallback is disabled")
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.state_path = Path(state_path) if state_path else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.max_restarts_per_gpu = int(max_restarts_per_gpu)
        self.max_restart_backoff_seconds = max(
            self.poll_seconds, float(max_restart_backoff_seconds)
        )
        self.max_failures_without_progress = max(
            1, int(max_failures_without_progress)
        )
        self.startup_grace_seconds = float(startup_grace_seconds)
        self.allow_shared_gpus = bool(allow_shared_gpus)
        self.min_free_memory_mib = max(0, int(min_free_memory_mib))
        self.env = dict(os.environ if env is None else env)
        self.workers: dict[str, Worker] = {}
        self.restarts = {gpu_id: 0 for gpu_id in self.gpu_ids}
        self.next_start = {gpu_id: 0.0 for gpu_id in self.gpu_ids}
        self.blocked_reasons: dict[str, dict[str, object]] = {}
        self.events: list[dict] = []
        self.stop_requested = False
        self.failure_message: str | None = None
        initial_status = self.status_factory()
        self.last_completed = int(
            initial_status.get("completed_papers") or 0
        )
        self.last_progress_at = time.time()
        self.consecutive_failures_without_progress = 0

    def _event(self, event: str, **fields: object) -> None:
        status = self.status_factory()
        completed = int(status.get("completed_papers") or 0)
        total = int(status.get("total_papers") or 0)
        fields.setdefault("progress_completed", completed)
        fields.setdefault("progress_total", total)
        fields.setdefault(
            "progress_percent",
            f"{progress_percent(completed, total):.2f}%",
        )
        fields.setdefault("progress_scope", "paper_queue")
        record = {"time": time.time(), "event": event, **fields}
        self.events.append(record)
        self.events = self.events[-200:]
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [adaptive-gpu] "
            + event
            + (" " + json.dumps(fields, ensure_ascii=False) if fields else ""),
            flush=True,
        )

    def _write_state(self) -> None:
        if self.state_path is None:
            return
        timestamp = time.time()
        work_state = self._work_state()
        state = {
            "status": (
                "completed"
                if self.is_complete()
                else ("failed" if self.failure_message else "running")
            ),
            "allowed_a800_gpus": self.gpu_ids,
            "allow_shared_gpus": self.allow_shared_gpus,
            "min_free_memory_mib": self.min_free_memory_mib,
            "max_restart_backoff_seconds": self.max_restart_backoff_seconds,
            "max_failures_without_progress": (
                self.max_failures_without_progress
            ),
            "blocked_reasons": self.blocked_reasons,
            "workers": {
                gpu_id: {
                    "pid": worker.process.pid,
                    "started_at": worker.started_at,
                    "command": worker.process.args,
                }
                for gpu_id, worker in self.workers.items()
            },
            "restarts": self.restarts,
            "queue": self.status_factory(),
            "error": self.failure_message,
            "work_state": work_state,
            "resource_waiting": work_state == "waiting_for_resources",
            "process_heartbeat_at": timestamp,
            "progress_heartbeat_at": self.last_progress_at,
            "last_progress_at": self.last_progress_at,
            "last_progress_completed": self.last_completed,
            "consecutive_worker_failures_without_progress": (
                self.consecutive_failures_without_progress
            ),
            "events": self.events,
            "updated_at": timestamp,
        }
        atomic_write_json(self.state_path, state)

    def _refresh_progress(self) -> int:
        status = self.status_factory()
        completed = int(status.get("completed_papers") or 0)
        if completed > self.last_completed:
            previous = self.last_completed
            self.last_completed = completed
            self.last_progress_at = time.time()
            self.consecutive_failures_without_progress = 0
            self._event(
                "durable_progress",
                previous_completed=previous,
                completed=completed,
            )
        return completed

    def _resource_waiting(self) -> bool:
        if self.workers:
            return False
        if not self.blocked_reasons:
            return False
        return all(
            str(value.get("reason")) in RESOURCE_WAIT_REASONS
            for value in self.blocked_reasons.values()
        )

    def _work_state(self) -> str:
        if self.failure_message:
            return "failed"
        if self.is_complete():
            return "completed"
        if self.workers:
            return "running"
        if self._resource_waiting():
            return "waiting_for_resources"
        if self.consecutive_failures_without_progress:
            return "worker_retry_backoff"
        return "starting"

    def _start(self, gpu_id: str) -> bool:
        if time.time() < self.next_start[gpu_id]:
            return False
        attempt = self.restarts[gpu_id]
        command = self.command_factory(gpu_id, attempt)
        lease = try_gpu_lease(gpu_id)
        if lease is None:
            self.blocked_reasons[gpu_id] = {"reason": "repository_lock"}
            return False
        # Recheck after obtaining our repository lock to close the local race.
        foreign = gpu_foreign_processes(gpu_id)
        duplicate_workers = same_work_queue_processes(foreign, command)
        if duplicate_workers:
            self.blocked_reasons[gpu_id] = {
                "reason": "same_work_queue_process",
                "pids": [process["pid"] for process in duplicate_workers],
            }
            lease.close()
            return False
        if foreign and not self.allow_shared_gpus:
            self.blocked_reasons[gpu_id] = {
                "reason": "foreign_process",
                "pids": [process["pid"] for process in foreign],
            }
            lease.close()
            return False
        free_memory_mib = gpu_free_memory_mib(gpu_id)
        if free_memory_mib < self.min_free_memory_mib:
            self.blocked_reasons[gpu_id] = {
                "reason": "insufficient_free_memory",
                "free_memory_mib": free_memory_mib,
                "required_free_memory_mib": self.min_free_memory_mib,
            }
            lease.close()
            return False
        log_handle = None
        stdout = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"gpu_{gpu_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            stdout = log_handle
        process = subprocess.Popen(
            command,
            env=self.env,
            stdout=stdout,
            stderr=subprocess.STDOUT if stdout is not None else None,
            start_new_session=True,
        )
        self.workers[gpu_id] = Worker(
            gpu_id=gpu_id,
            process=process,
            lease=lease,
            started_at=time.time(),
            progress_completed_at_start=self.last_completed,
            log_handle=log_handle,
        )
        self.blocked_reasons.pop(gpu_id, None)
        self._event(
            "worker_started",
            gpu=gpu_id,
            pid=process.pid,
            attempt=attempt,
            free_memory_mib=free_memory_mib,
            shared=bool(foreign),
            foreign_pids=[process["pid"] for process in foreign],
        )
        return True

    def _stop(self, gpu_id: str, reason: str) -> None:
        worker = self.workers[gpu_id]
        if worker.process.poll() is None:
            try:
                os.killpg(worker.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                worker.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(worker.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.process.wait()
        self._event(
            "worker_stopped",
            gpu=gpu_id,
            pid=worker.process.pid,
            reason=reason,
            returncode=worker.process.returncode,
        )
        worker.lease.close()
        if worker.log_handle is not None:
            worker.log_handle.close()
        del self.workers[gpu_id]

    def _reap_and_monitor(self) -> None:
        completed = self._refresh_progress()
        for gpu_id in list(self.workers):
            worker = self.workers[gpu_id]
            returncode = worker.process.poll()
            if returncode is not None:
                normal = returncode == 0
                self._stop(gpu_id, "worker_exit")
                if not normal and not self.is_complete():
                    if completed <= worker.progress_completed_at_start:
                        self.consecutive_failures_without_progress += 1
                    else:
                        self.consecutive_failures_without_progress = 0
                    self._event(
                        "worker_failure_without_progress",
                        gpu=gpu_id,
                        returncode=returncode,
                        consecutive_failures=(
                            self.consecutive_failures_without_progress
                        ),
                        max_failures=(
                            self.max_failures_without_progress
                        ),
                    )
                    if (
                        self.consecutive_failures_without_progress
                        >= self.max_failures_without_progress
                    ):
                        raise WorkerProgressStallError(
                            f"{self.consecutive_failures_without_progress} "
                            "consecutive worker failures without durable "
                            f"progress (completed={completed})"
                        )
                    self.restarts[gpu_id] += 1
                    if self.restarts[gpu_id] > self.max_restarts_per_gpu:
                        raise RuntimeError(
                            f"GPU {gpu_id} worker exceeded restart limit"
                        )
                    delay = min(
                        self.max_restart_backoff_seconds,
                        2.0 ** min(self.restarts[gpu_id], 8),
                    )
                    self.next_start[gpu_id] = time.time() + delay
                continue
            if time.time() - worker.started_at < self.startup_grace_seconds:
                continue
            own_pids = _descendant_pids(worker.process.pid)
            foreign = gpu_foreign_processes(gpu_id, own_pids)
            if foreign and not self.allow_shared_gpus:
                self._stop(gpu_id, "foreign_process_appeared")
                self.next_start[gpu_id] = time.time() + self.poll_seconds

    def _raise_if_all_unfinished_are_quarantined(self) -> None:
        status = self.status_factory()
        unfinished = int(status.get("unfinished_papers") or 0)
        active_claims = int(status.get("active_claims") or 0)
        quarantined = [
            item
            for item in status.get("failures", [])
            if isinstance(item, dict) and item.get("quarantined")
        ]
        retryable = status.get("retryable_papers")
        no_retryable_work = (
            int(retryable) == 0
            if isinstance(retryable, (int, float))
            else len(quarantined) >= unfinished
        )
        if unfinished > 0 and active_claims == 0 and no_retryable_work:
            paper_ids = ", ".join(
                str(item.get("paper_id", "?"))
                for item in quarantined[:5]
            )
            raise QuarantinedWorkError(
                f"{unfinished} unfinished paper(s) are quarantined"
                + (f": {paper_ids}" if paper_ids else "")
            )

    def _handle_shutdown_signal(self, signum: int, _frame: object) -> None:
        if self.stop_requested:
            return
        self.stop_requested = True
        self._event("pool_shutdown_requested", signal=signum)
        # Workers have their own sessions, so the parent process-group signal
        # does not reach them. Notify every worker immediately, then let the
        # normal finally block reap processes and release leases/log handles.
        for worker in list(self.workers.values()):
            if worker.process.poll() is None:
                try:
                    os.killpg(worker.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        raise PoolShutdownSignal(signum)

    def run(self) -> None:
        previous_handlers: dict[int, object] = {}
        install_signal_handlers = (
            threading.current_thread() is threading.main_thread()
        )
        if install_signal_handlers:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_shutdown_signal)
        self._event(
            "pool_started",
            a800_gpus=self.gpu_ids,
            allow_shared_gpus=self.allow_shared_gpus,
            min_free_memory_mib=self.min_free_memory_mib,
        )
        shutdown_signal: int | None = None
        try:
            while not self.is_complete() and not self.stop_requested:
                self._reap_and_monitor()
                self._raise_if_all_unfinished_are_quarantined()
                status = self.status_factory()
                unfinished_value = status.get("unfinished_papers")
                desired_workers = (
                    len(self.gpu_ids)
                    if unfinished_value is None
                    else min(len(self.gpu_ids), int(unfinished_value or 0))
                )
                for gpu_id in self.gpu_ids:
                    if len(self.workers) >= desired_workers:
                        break
                    if (
                        gpu_id not in self.workers
                        and not self.is_complete()
                        and not self.stop_requested
                    ):
                        self._start(gpu_id)
                self._write_state()
                if not self.workers:
                    self._event(
                        "waiting_for_free_a800"
                        if self._resource_waiting()
                        else "worker_retry_backoff"
                    )
                time.sleep(self.poll_seconds)
            if self.is_complete():
                for gpu_id in list(self.workers):
                    self._stop(gpu_id, "queue_complete")
                self._event("pool_completed")
                self._write_state()
        except PoolShutdownSignal as exc:
            shutdown_signal = exc.signum
        except Exception as exc:
            self.failure_message = f"{type(exc).__name__}: {exc}"
            self._event("pool_failed", error=self.failure_message)
            self._write_state()
            raise
        finally:
            for gpu_id in list(self.workers):
                self._stop(gpu_id, "pool_shutdown")
            self._write_state()
            if install_signal_handlers:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        if shutdown_signal is not None:
            raise SystemExit(128 + shutdown_signal)
