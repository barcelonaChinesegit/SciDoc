#!/usr/bin/env python3
"""Idempotently register the local data-review API with the task daemon."""

from __future__ import annotations

import sys
from pku_qa.services.task_queue.task_queue_manager import DEFAULT_DB, ROOT, TaskStore


def main() -> None:
    store = TaskStore(DEFAULT_DB)
    command = [
        sys.executable,
        "-m",
        "pku_qa.services.review.data_manager_api",
        "--host",
        "127.0.0.1",
        "--port",
        "8770",
    ]
    validation = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/unit/test_data_review_manager.py",
    ]
    try:
        task = store.get("data_api")
    except KeyError:
        task = store.add(
            task_id="data_api",
            name="数据集人工审核本地 API",
            command=command,
            cwd=ROOT,
            priority=195,
            kind="service",
            max_restarts=100,
            retry_delay_seconds=3,
            validation_command=validation,
            auto_codex=True,
            max_codex_attempts=0,
        )
    else:
        task = store.update(
            task.id,
            name="数据集人工审核本地 API",
            command=command,
            cwd=str(ROOT),
            priority=195,
            kind="service",
            enabled=True,
            max_restarts=100,
            retry_delay_seconds=3,
            validation_command=validation,
            auto_codex=True,
            max_codex_attempts=0,
        )
        if task.status not in {"starting", "running"}:
            task = store.update(
                task.id,
                status="queued",
                next_retry_at=0,
                finished_at=None,
            )

    try:
        web = store.get("queue_web")
    except KeyError:
        pass
    else:
        dependencies = list(
            dict.fromkeys([*web.depends_on, "queue_api", "data_api"])
        )
        store.update(web.id, depends_on=dependencies)
    print(task.id, task.status, task.name)


if __name__ == "__main__":
    main()
