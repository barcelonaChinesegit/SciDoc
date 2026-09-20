#!/usr/bin/env python3
"""CRUD, ordering, control, progress, and log CLI for project tasks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from task_queue_manager import (
    ACTIVE_STATES,
    DEFAULT_DB,
    RESOURCE_CLASSES,
    ROOT,
    TaskStore,
    pid_alive,
    task_resource_class,
    task_log_snapshot,
    tmux_session_exists,
)


SUPERVISOR_SESSION = "pku_task_queue_supervisor"


def store_from(args: argparse.Namespace) -> TaskStore:
    return TaskStore(args.db)


def print_tasks(tasks, as_json: bool = False) -> None:
    if as_json:
        print(
            json.dumps(
                [task.to_dict() for task in tasks],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print(
        f"{'#':>3} {'ID':<12} {'STATUS':<11} {'RESOURCE':<13} {'RESTART':>7} "
        f"{'PROGRESS':<14} NAME"
    )
    for index, task in enumerate(tasks, start=1):
        progress = task.to_dict().get("progress") or {}
        percent = progress.get("percent")
        progress_text = f"{percent:6.2f}%" if percent is not None else "-"
        print(
            f"{index:>3} {task.id:<12} {task.status:<11} "
            f"{task_resource_class(task):<13} "
            f"{task.restart_count:>3}/{task.max_restarts:<3} "
            f"{progress_text:<14} {task.name}"
        )


def stop_process(task) -> None:
    pid = task.runner_pid or task.pid
    if pid_alive(pid):
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5
        while pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if pid_alive(pid):
            for process_group in (task.pid, task.runner_pid):
                if not process_group:
                    continue
                try:
                    os.killpg(int(process_group), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent project task queue")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    sub = parser.add_subparsers(dest="action", required=True)

    list_parser = sub.add_parser("list")
    list_parser.add_argument("--json", action="store_true")

    show = sub.add_parser("show")
    show.add_argument("task_id")

    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--cwd", default=str(ROOT))
    add.add_argument("--env", action="append", default=[])
    add.add_argument("--priority", type=int, default=0)
    add.add_argument(
        "--description",
        default="",
        help="任务目的、主要步骤和预期输出",
    )
    add.add_argument("--kind", choices=["batch", "service"], default="batch")
    add.add_argument(
        "--resource-class",
        choices=sorted(RESOURCE_CLASSES),
        default="cpu",
    )
    add.add_argument("--depends-on", action="append", default=[])
    add.add_argument("--progress-path")
    add.add_argument("--validation", default="")
    add.add_argument("--max-restarts", type=int, default=20)
    add.add_argument("--retry-delay", type=float, default=15)
    add.add_argument("--timeout", type=float, default=0)
    add.add_argument("--no-codex", action="store_true")
    add.add_argument(
        "--max-codex-attempts",
        type=int,
        default=0,
        help="Codex 自动修复上限；0 表示持续修复直到成功",
    )
    add.add_argument(
        "command",
        nargs="*",
        help="运行命令；使用 -- 与任务队列参数分隔",
    )

    update = sub.add_parser("update")
    update.add_argument("task_id")
    update.add_argument("--name")
    update.add_argument("--description")
    update.add_argument("--priority", type=int)
    update.add_argument("--max-restarts", type=int)
    update.add_argument("--max-codex-attempts", type=int)
    update.add_argument("--retry-delay", type=float)
    update.add_argument("--progress-path")
    update.add_argument(
        "--depends-on",
        action="append",
        help="Replace task dependencies with the supplied task IDs.",
    )
    update.add_argument(
        "--resource-class",
        choices=sorted(RESOURCE_CLASSES),
    )
    update.add_argument(
        "--codex",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或关闭本地 Codex 自动修复",
    )
    update.add_argument("--enable", action="store_true")
    update.add_argument("--disable", action="store_true")
    update.add_argument("--command", default="")

    delete = sub.add_parser("delete")
    delete.add_argument("task_id")
    delete.add_argument("--force", action="store_true")

    move = sub.add_parser("move")
    move.add_argument("task_id")
    move.add_argument("index", type=int, help="1-based destination")

    for action in ("pause", "resume", "retry", "cancel"):
        control = sub.add_parser(action)
        control.add_argument("task_id")

    logs = sub.add_parser("logs")
    logs.add_argument("task_id")
    logs.add_argument("--bytes", type=int, default=32000)

    incidents = sub.add_parser("incidents")
    incidents.add_argument("--task-id")

    daemon = sub.add_parser("daemon")
    daemon.add_argument("operation", choices=["start", "stop", "status"])

    config = sub.add_parser("config")
    config.add_argument("--max-concurrent-batch", type=int)
    config.add_argument("--max-concurrent-cpu-batch", type=int)
    config.add_argument("--max-concurrent-gpu-batch", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = store_from(args)

    if args.action == "list":
        print_tasks(store.list_tasks(), args.json)
    elif args.action == "show":
        print(json.dumps(store.get(args.task_id).to_dict(), ensure_ascii=False, indent=2))
    elif args.action == "add":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        env = {}
        for assignment in args.env:
            key, separator, value = assignment.partition("=")
            if not separator or not key:
                raise SystemExit(f"Invalid --env value: {assignment!r}")
            env[key] = value
        validation = shlex.split(args.validation) if args.validation else []
        task = store.add(
            name=args.name,
            command=command,
            cwd=args.cwd,
            env=env,
            priority=args.priority,
            kind=args.kind,
            depends_on=args.depends_on,
            progress_path=args.progress_path,
            validation_command=validation,
            max_restarts=args.max_restarts,
            retry_delay_seconds=args.retry_delay,
            timeout_seconds=args.timeout,
            auto_codex=not args.no_codex,
            max_codex_attempts=args.max_codex_attempts,
            metadata={
                "resource_class": args.resource_class,
                "description": args.description.strip(),
            },
        )
        print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
    elif args.action == "update":
        changes = {}
        for field in (
            "name",
            "priority",
            "max_restarts",
            "max_codex_attempts",
            "progress_path",
        ):
            value = getattr(args, field)
            if value is not None:
                changes[field] = value
        if args.retry_delay is not None:
            changes["retry_delay_seconds"] = args.retry_delay
        if args.codex is not None:
            changes["auto_codex"] = args.codex
        if args.enable and args.disable:
            raise SystemExit("--enable and --disable are mutually exclusive")
        if args.enable or args.disable:
            changes["enabled"] = args.enable
        if args.command:
            changes["command"] = shlex.split(args.command)
        if args.depends_on is not None:
            changes["depends_on"] = args.depends_on
        if args.resource_class or args.description is not None:
            current = store.get(args.task_id)
            metadata = dict(current.metadata)
            if args.resource_class:
                metadata["resource_class"] = args.resource_class
            if args.description is not None:
                metadata["description"] = args.description.strip()
            changes["metadata"] = metadata
        task = store.update(args.task_id, **changes)
        print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
    elif args.action == "delete":
        task = store.get(args.task_id)
        if args.force and task.status in ACTIVE_STATES:
            stop_process(task)
        store.delete(args.task_id, force=args.force)
        print(f"deleted {args.task_id}")
    elif args.action == "move":
        tasks = store.move(args.task_id, args.index - 1)
        print_tasks(tasks)
    elif args.action in {"pause", "resume", "retry", "cancel"}:
        task = store.get(args.task_id)
        if args.action == "pause":
            store.update(task.id, status="paused")
            stop_process(task)
            task = store.update(
                task.id, status="paused", pid=None, runner_pid=None
            )
        elif args.action == "resume":
            task = store.update(
                task.id,
                status="queued",
                enabled=True,
                next_retry_at=0,
                finished_at=None,
            )
        elif args.action == "retry":
            store.update(task.id, status="paused")
            stop_process(task)
            task = store.update(
                task.id,
                status="queued",
                enabled=True,
                adopted=False,
                pid=None,
                runner_pid=None,
                next_retry_at=0,
                finished_at=None,
            )
        else:
            store.update(task.id, status="cancelled", enabled=False)
            stop_process(task)
            task = store.update(
                task.id,
                status="cancelled",
                enabled=False,
                pid=None,
                runner_pid=None,
            )
        print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2))
    elif args.action == "logs":
        snapshot = task_log_snapshot(store.get(args.task_id), args.bytes)
        print(snapshot["log"])
    elif args.action == "incidents":
        print(
            json.dumps(
                store.list_incidents(args.task_id),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.action == "daemon":
        if args.operation == "start":
            if tmux_session_exists(SUPERVISOR_SESSION):
                print("task queue supervisor already running")
            else:
                subprocess.run(
                    [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        SUPERVISOR_SESSION,
                        sys.executable,
                        str(ROOT / "src/pku_qa/services/task_queue/task_queue_supervisor.py"),
                        "--db",
                        str(store.db_path),
                    ],
                    check=True,
                )
                print("task queue supervisor started")
        elif args.operation == "stop":
            subprocess.run(
                ["tmux", "kill-session", "-t", SUPERVISOR_SESSION],
                check=False,
            )
            print("task queue supervisor stopped")
        else:
            print(
                "running"
                if tmux_session_exists(SUPERVISOR_SESSION)
                else "stopped"
            )
    elif args.action == "config":
        settings = {
            "max_concurrent_batch": args.max_concurrent_batch,
            "max_concurrent_cpu_batch": args.max_concurrent_cpu_batch,
            "max_concurrent_gpu_batch": args.max_concurrent_gpu_batch,
        }
        for key, value in settings.items():
            if value is None:
                continue
            if value < 1:
                raise SystemExit(f"{key} must be >= 1")
            store.set_setting(key, value)
        print(
            json.dumps(
                {
                    "max_concurrent_batch": store.setting(
                        "max_concurrent_batch", 1
                    ),
                    "max_concurrent_cpu_batch": store.setting(
                        "max_concurrent_cpu_batch", 1
                    ),
                    "max_concurrent_gpu_batch": store.setting(
                        "max_concurrent_gpu_batch", 1
                    ),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
