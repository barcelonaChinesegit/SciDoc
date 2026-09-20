#!/usr/bin/env python3
"""Detached runner for one managed project task."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from task_queue_manager import TaskStore, now, read_progress, tail_text


def api_quota_pause_requested(progress: dict | None) -> bool:
    return bool(progress) and progress.get("status") == "waiting_for_api_quota"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--task-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = TaskStore(args.db)
    task = store.get(args.task_id)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(5):
            try:
                store.update(task.id, heartbeat_at=now())
            except Exception:
                pass

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    child: subprocess.Popen | None = None

    def terminate(signum, _frame) -> None:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    Path(task.log_path).parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(task.env)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    # Uniform policy for every queue-managed GPU command. Individual tasks may
    # override these values explicitly, but the project default is safe
    # co-location on sufficiently empty cards rather than exclusive ownership.
    environment.setdefault("GPU_ALLOW_SHARED", "1")
    environment.setdefault("GPU_MIN_FREE_MEMORY_MIB", "56000")
    environment.setdefault("GPU_REPOSITORY_LOCK", "0")
    environment.setdefault("GPU_WAIT_POLL_SECONDS", "15")
    started = now()
    store.update(
        task.id,
        status="running",
        pid=None,
        runner_pid=os.getpid(),
        started_at=task.started_at or started,
        heartbeat_at=started,
        finished_at=None,
        last_error=None,
    )
    with Path(task.log_path).open("a", encoding="utf-8") as log:
        log.write(
            f"\n[task-runner] start={started:.3f} pid={os.getpid()} "
            f"command={task.command!r}\n"
        )
        log.flush()
        try:
            child = subprocess.Popen(
                task.command,
                cwd=task.cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            store.update(task.id, pid=child.pid, heartbeat_at=now())
            returncode = child.wait()
            finished = now()
            log.write(
                f"[task-runner] event=child_exited time={finished:.3f} "
                f"pid={child.pid} returncode={returncode} "
                f"duration_seconds={finished - started:.3f}\n"
            )
            log.flush()
        except BaseException as exc:
            log.write(f"[task-runner] launch error: {type(exc).__name__}: {exc}\n")
            log.flush()
            returncode = 127
    stop.set()
    thread.join(timeout=1)
    requested = store.get(task.id)
    if requested.status in {"paused", "cancelled", "queued"}:
        store.update(
            task.id,
            pid=None,
            runner_pid=None,
            heartbeat_at=now(),
            finished_at=now() if requested.status == "cancelled" else None,
        )
        return 0
    if returncode != 0 and api_quota_pause_requested(
        read_progress(task.progress_path)
    ):
        store.update(
            task.id,
            status="paused",
            pid=None,
            runner_pid=None,
            heartbeat_at=now(),
            finished_at=None,
            last_error=(
                "API quota exhausted; durable provider/paper checkpoints were "
                "saved. Resume manually after renewing the API quota."
            ),
        )
        return 0
    if returncode == 0:
        store.update(
            task.id,
            status="succeeded",
            pid=None,
            runner_pid=None,
            heartbeat_at=now(),
            finished_at=now(),
            last_error=None,
        )
    else:
        error = tail_text(task.log_path, 20_000)[-20_000:]
        store.update(
            task.id,
            status="failed",
            pid=None,
            runner_pid=None,
            heartbeat_at=now(),
            finished_at=now(),
            last_error=f"exit_code={returncode}\n{error}",
        )
    return int(returncode)


if __name__ == "__main__":
    raise SystemExit(main())
