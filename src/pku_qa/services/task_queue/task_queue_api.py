#!/usr/bin/env python3
"""Loopback-only JSON API for the local task queue GUI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from eval_framework import query_gpus
from task_queue_manager import (
    DEFAULT_DB,
    ROOT,
    TaskStore,
    delete_task_results,
    task_log_snapshot,
    task_result_targets,
)
MAX_TASK_DESCRIPTION_CHARS = 4000
GPU_STATUS_CACHE_TTL_SECONDS = 30.0


class GpuStatusCache:
    """Bound GPU probes and coalesce concurrent dashboard refreshes."""

    def __init__(self, ttl_seconds: float = GPU_STATUS_CACHE_TTL_SECONDS) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._lock = threading.Lock()
        self._value: list[dict] | None = None
        self._expires_at = 0.0

    def get(self) -> list[dict]:
        with self._lock:
            current = time.monotonic()
            if self._value is not None and current < self._expires_at:
                return self._value
            try:
                value = query_gpus()
            except Exception as exc:
                value = [{"error": str(exc)}]
            self._value = value
            self._expires_at = time.monotonic() + self.ttl_seconds
            return value


GPU_STATUS_CACHE = GpuStatusCache()


def normalize_task_description(value) -> str:
    description = str(value or "").strip()
    if len(description) > MAX_TASK_DESCRIPTION_CHARS:
        raise ValueError(
            f"description must not exceed {MAX_TASK_DESCRIPTION_CHARS} characters"
        )
    return description


def task_metadata(body: dict, current: dict | None = None) -> dict:
    raw_metadata = body.get("metadata", current or {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("metadata must be an object")
    metadata = dict(raw_metadata)
    if "description" in body:
        metadata["description"] = normalize_task_description(
            body["description"]
        )
    return metadata


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "PkuTaskQueue/1.0"

    @property
    def store(self) -> TaskStore:
        return self.server.store  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin.startswith(("http://127.0.0.1:", "http://localhost:")):
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PATCH,DELETE,OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type,X-PKU-Actor-User-Id,X-PKU-Actor-Username",
        )
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def json_response(self, value, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def route(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        return parts, parse_qs(parsed.query)

    def actor(self) -> dict[str, str | None]:
        return {
            "actor_user_id": self.headers.get("X-PKU-Actor-User-Id") or None,
            "actor_username": self.headers.get("X-PKU-Actor-Username") or "system",
        }

    def audit(
        self,
        action: str,
        *,
        task_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.store.record_audit(
            action,
            task_id=task_id,
            details=details,
            **self.actor(),
        )

    def handle_error(self, exc: Exception) -> None:
        status = 404 if isinstance(exc, KeyError) else 400
        self.json_response(
            {"error": f"{type(exc).__name__}: {exc}"}, status=status
        )

    def do_GET(self) -> None:
        try:
            parts, query = self.route()
            if parts == ["api", "health"]:
                self.json_response({"status": "ok", "pid": os.getpid()})
            elif parts == ["api", "tasks"]:
                self.json_response(
                    {"tasks": [task.to_dict() for task in self.store.list_tasks()]}
                )
            elif len(parts) == 3 and parts[:2] == ["api", "tasks"]:
                self.json_response(self.store.get(parts[2]).to_dict())
            elif len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "logs":
                size = int(query.get("bytes", ["64000"])[0])
                task = self.store.get(parts[2])
                stage_id = query.get("stage", [None])[0]
                self.json_response(
                    task_log_snapshot(task, size, stage_id=stage_id)
                )
            elif (
                len(parts) == 4
                and parts[:2] == ["api", "tasks"]
                and parts[3] == "result-targets"
            ):
                task = self.store.get(parts[2])
                self.json_response(
                    {
                        "task_id": task.id,
                        "targets": [
                            str(path) for path in task_result_targets(task)
                        ],
                    }
                )
            elif parts == ["api", "incidents"]:
                self.json_response({"incidents": self.store.list_incidents()})
            elif parts == ["api", "deletions"]:
                self.json_response(
                    {
                        "deletions": self.store.list_deletions(),
                        "result_deletions": (
                            self.store.list_result_deletions()
                        ),
                    }
                )
            elif parts == ["api", "audit"]:
                limit = int(query.get("limit", ["200"])[0])
                self.json_response({"events": self.store.list_audit(limit)})
            elif parts == ["api", "system"]:
                self.json_response(
                    {
                        "gpus": GPU_STATUS_CACHE.get(),
                        "max_concurrent_batch": self.store.setting(
                            "max_concurrent_batch", 1
                        ),
                    }
                )
            else:
                self.json_response({"error": "not found"}, 404)
        except Exception as exc:
            self.handle_error(exc)

    def do_POST(self) -> None:
        try:
            parts, _ = self.route()
            body = self.read_json()
            if parts == ["api", "tasks"]:
                command = body.get("command", [])
                if isinstance(command, str):
                    command = shlex.split(command)
                metadata = task_metadata(body)
                task = self.store.add(
                    name=str(body["name"]),
                    command=command,
                    cwd=body.get("cwd", str(ROOT)),
                    env=body.get("env", {}),
                    priority=int(body.get("priority", 0)),
                    kind=body.get("kind", "batch"),
                    depends_on=body.get("depends_on", []),
                    progress_path=body.get("progress_path"),
                    validation_command=body.get("validation_command", []),
                    max_restarts=int(body.get("max_restarts", 20)),
                    retry_delay_seconds=float(body.get("retry_delay_seconds", 15)),
                    timeout_seconds=float(body.get("timeout_seconds", 0)),
                    auto_codex=bool(body.get("auto_codex", True)),
                    max_codex_attempts=int(body.get("max_codex_attempts", 0)),
                    metadata=metadata,
                )
                self.audit(
                    "task.create",
                    task_id=task.id,
                    details={"name": task.name, "kind": task.kind},
                )
                self.json_response(task.to_dict(), 201)
                return
            if (
                len(parts) == 4
                and parts[:2] == ["api", "deletions"]
                and parts[3] == "restore"
            ):
                task = self.store.restore_deletion(int(parts[2]))
                self.audit(
                    "task.restore",
                    task_id=task.id,
                    details={"deletion_id": int(parts[2])},
                )
                self.json_response(
                    {
                        "restored": task.id,
                        "task": task.to_dict(),
                    }
                )
                return
            if (
                len(parts) == 5
                and parts[:2] == ["api", "tasks"]
                and parts[3] == "actions"
            ):
                task = self.store.get(parts[2])
                action = parts[4]
                if action == "pause":
                    self.store.update(task.id, status="paused")
                    terminate_task(task)
                    task = self.store.update(
                        task.id, status="paused", pid=None, runner_pid=None
                    )
                elif action in {"resume", "retry"}:
                    if action == "retry":
                        self.store.update(task.id, status="paused")
                        terminate_task(task)
                    task = self.store.update(
                        task.id,
                        status="queued",
                        enabled=True,
                        adopted=False if action == "retry" else task.adopted,
                        pid=None if action == "retry" else task.pid,
                        runner_pid=None,
                        next_retry_at=0,
                        finished_at=None,
                    )
                elif action == "cancel":
                    self.store.update(
                        task.id, status="cancelled", enabled=False
                    )
                    terminate_task(task)
                    task = self.store.update(
                        task.id,
                        status="cancelled",
                        enabled=False,
                        pid=None,
                        runner_pid=None,
                    )
                elif action == "move":
                    tasks = self.store.move(task.id, int(body["index"]))
                    self.audit(
                        "task.move",
                        task_id=task.id,
                        details={"index": int(body["index"])},
                    )
                    self.json_response(
                        {"tasks": [item.to_dict() for item in tasks]}
                    )
                    return
                elif action == "delete-results":
                    if body.get("confirm_task_id") != task.id:
                        raise ValueError(
                            "confirm_task_id must exactly match the task id"
                        )
                    deleted, errors = delete_task_results(task)
                    record = self.store.add_result_deletion(
                        task, deleted, errors
                    )
                    self.audit(
                        "task.delete_results",
                        task_id=task.id,
                        details={
                            "deleted_target_count": len(deleted),
                            "error_count": len(errors),
                        },
                    )
                    self.json_response(
                        {
                            "task_id": task.id,
                            "deleted_targets": deleted,
                            "errors": errors,
                            "record": record,
                        }
                    )
                    return
                else:
                    raise ValueError(f"Unknown task action: {action}")
                self.audit(
                    f"task.{action}",
                    task_id=task.id,
                    details={"status": task.status},
                )
                self.json_response(task.to_dict())
                return
            self.json_response({"error": "not found"}, 404)
        except Exception as exc:
            self.handle_error(exc)

    def do_PATCH(self) -> None:
        try:
            parts, _ = self.route()
            if len(parts) != 3 or parts[:2] != ["api", "tasks"]:
                self.json_response({"error": "not found"}, 404)
                return
            body = self.read_json()
            allowed = {
                "name",
                "command",
                "cwd",
                "env",
                "priority",
                "kind",
                "depends_on",
                "enabled",
                "max_restarts",
                "retry_delay_seconds",
                "timeout_seconds",
                "progress_path",
                "validation_command",
                "auto_codex",
                "max_codex_attempts",
                "metadata",
                "description",
            }
            changes = {key: value for key, value in body.items() if key in allowed}
            if isinstance(changes.get("command"), str):
                changes["command"] = shlex.split(changes["command"])
            if "description" in changes:
                current = self.store.get(parts[2])
                changes["metadata"] = task_metadata(
                    changes,
                    changes.get("metadata", current.metadata),
                )
                changes.pop("description", None)
            elif "metadata" in changes and not isinstance(
                changes["metadata"], dict
            ):
                raise ValueError("metadata must be an object")
            task = self.store.update(parts[2], **changes)
            self.audit(
                "task.update",
                task_id=task.id,
                details={"fields": sorted(changes)},
            )
            self.json_response(task.to_dict())
        except Exception as exc:
            self.handle_error(exc)

    def do_DELETE(self) -> None:
        try:
            parts, query = self.route()
            if len(parts) != 3 or parts[:2] != ["api", "tasks"]:
                self.json_response({"error": "not found"}, 404)
                return
            force = query.get("force", ["false"])[0].lower() == "true"
            task = self.store.get(parts[2])
            if force:
                terminate_task(task)
            deletion = self.store.delete(task.id, force=force)
            self.audit(
                "task.delete",
                task_id=task.id,
                details={"force": force, "deletion_id": deletion["id"]},
            )
            self.json_response(
                {
                    "deleted": task.id,
                    "deletion_record": deletion,
                    "files_deleted": False,
                }
            )
        except Exception as exc:
            self.handle_error(exc)

    def log_message(self, fmt: str, *args) -> None:
        # Keep the daemon log clean; task/API errors are returned as JSON.
        return


def terminate_task(task) -> None:
    pid = task.runner_pid or task.pid
    if pid:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + 5
        while process_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if process_alive(pid):
            for process_group in (task.pid, task.runner_pid):
                if not process_group:
                    continue
                try:
                    os.killpg(int(process_group), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The task API is local-only; bind to loopback")
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    server.store = TaskStore(Path(args.db))  # type: ignore[attr-defined]
    print(f"Task queue API listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
