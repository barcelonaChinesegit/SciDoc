#!/usr/bin/env python3
"""Persistent project task queue primitives.

The SQLite store is deliberately independent from the GPU evaluation queue:
it schedules whole experiments, while ``src/pku_qa/evaluation/durable_work_queue.py`` checkpoints
papers/questions inside one experiment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from glob import glob
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QUEUE_DIR = ROOT / "data/web/task_queue"
DEFAULT_DB = DEFAULT_QUEUE_DIR / "tasks.sqlite3"
DEFAULT_LOG_DIR = DEFAULT_QUEUE_DIR / "logs"
DEFAULT_INCIDENT_DIR = DEFAULT_QUEUE_DIR / "incidents"
ACTIVE_STATES = {"starting", "running", "waiting", "recovering"}
TERMINAL_STATES = {"succeeded", "cancelled", "blocked"}
RESOURCE_CLASSES = {"cpu", "gpu", "gpu_high_vram"}


def task_resource_class(task: "Task") -> str:
    """Return the scheduling lane for a batch task."""
    explicit = str(task.metadata.get("resource_class", "")).strip().lower()
    if explicit in RESOURCE_CLASSES:
        return explicit
    command = " ".join(task.command).lower()
    if any(
        marker in command
        for marker in ("qwen3.6-27b", "qwen3_6_27b", "high_vram")
    ):
        return "gpu_high_vram"
    if any(
        marker in command
        for marker in (
            "--gpu",
            "--gpus",
            "--dynamic-a800",
            "src/pku_qa/evaluation/run_hard_eval.py",
            "gpu_queue",
            "ablation",
        )
    ):
        return "gpu"
    return "cpu"


def now() -> float:
    return time.time()


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_load(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class Task:
    id: str
    name: str
    command: list[str]
    cwd: str
    env: dict[str, str]
    status: str
    position: int
    priority: int
    kind: str
    depends_on: list[str]
    enabled: bool
    adopted: bool
    pid: int | None
    runner_pid: int | None
    tmux_session: str | None
    max_restarts: int
    restart_count: int
    retry_delay_seconds: float
    next_retry_at: float
    timeout_seconds: float
    heartbeat_at: float | None
    progress_heartbeat_at: float | None
    progress_signature: str | None
    progress_path: str | None
    log_path: str
    validation_command: list[str]
    auto_codex: bool
    max_codex_attempts: int
    codex_attempts: int
    last_error: str | None
    metadata: dict[str, Any]
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None

    def to_dict(self, include_progress: bool = True) -> dict[str, Any]:
        value = asdict(self)
        description = self.metadata.get("description", "")
        value["description"] = (
            str(description).strip() if description is not None else ""
        )
        value["resource_class"] = task_resource_class(self)
        value["alive"] = pid_alive(self.runner_pid or self.pid)
        if include_progress:
            stages = task_stage_snapshots(self)
            value["stages"] = stages
            value["progress"] = aggregate_stage_progress(
                self, stages
            )
        return value


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    command_json TEXT NOT NULL,
    cwd TEXT NOT NULL,
    env_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued',
    position INTEGER NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'batch',
    depends_on_json TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1,
    adopted INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,
    runner_pid INTEGER,
    tmux_session TEXT,
    max_restarts INTEGER NOT NULL DEFAULT 20,
    restart_count INTEGER NOT NULL DEFAULT 0,
    retry_delay_seconds REAL NOT NULL DEFAULT 15,
    next_retry_at REAL NOT NULL DEFAULT 0,
    timeout_seconds REAL NOT NULL DEFAULT 0,
    heartbeat_at REAL,
    progress_heartbeat_at REAL,
    progress_signature TEXT,
    progress_path TEXT,
    log_path TEXT NOT NULL,
    validation_command_json TEXT NOT NULL DEFAULT '[]',
    auto_codex INTEGER NOT NULL DEFAULT 1,
    max_codex_attempts INTEGER NOT NULL DEFAULT 0,
    codex_attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_order
ON tasks(status, enabled, priority DESC, position ASC);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL,
    log_tail TEXT NOT NULL DEFAULT '',
    resolution TEXT,
    codex_attempt INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    resolved_at REAL,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);
CREATE TABLE IF NOT EXISTS task_deletions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    task_snapshot_json TEXT NOT NULL,
    incidents_snapshot_json TEXT NOT NULL DEFAULT '[]',
    deleted_at REAL NOT NULL,
    restored_at REAL,
    restored_task_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_deletions_time
ON task_deletions(deleted_at DESC);
CREATE TABLE IF NOT EXISTS task_result_deletions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    task_name TEXT NOT NULL,
    targets_json TEXT NOT NULL DEFAULT '[]',
    errors_json TEXT NOT NULL DEFAULT '[]',
    deleted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_result_deletions_time
ON task_result_deletions(deleted_at DESC);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS task_audit_events (
    audit_id TEXT PRIMARY KEY,
    actor_user_id TEXT,
    actor_username TEXT NOT NULL,
    action TEXT NOT NULL,
    task_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_audit_events_time
ON task_audit_events(created_at DESC);
"""


class TaskStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "progress_heartbeat_at" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN progress_heartbeat_at REAL"
                )
            if "progress_signature" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN progress_signature TEXT"
                )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path, timeout=30, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def record_audit(
        self,
        action: str,
        *,
        task_id: str | None = None,
        actor_user_id: str | None = None,
        actor_username: str = "system",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "audit_id": uuid.uuid4().hex,
            "actor_user_id": actor_user_id,
            "actor_username": actor_username or "system",
            "action": action,
            "task_id": task_id,
            "details": details or {},
            "created_at": now(),
        }
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO task_audit_events(
                    audit_id,actor_user_id,actor_username,action,task_id,
                    details_json,created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    event["audit_id"],
                    event["actor_user_id"],
                    event["actor_username"],
                    event["action"],
                    event["task_id"],
                    json_dump(event["details"]),
                    event["created_at"],
                ),
            )
        return event

    def list_audit(self, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_audit_events
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["details"] = json_load(value.pop("details_json"), {})
            values.append(value)
        return values

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            name=row["name"],
            command=json_load(row["command_json"], []),
            cwd=row["cwd"],
            env=json_load(row["env_json"], {}),
            status=row["status"],
            position=int(row["position"]),
            priority=int(row["priority"]),
            kind=row["kind"],
            depends_on=json_load(row["depends_on_json"], []),
            enabled=bool(row["enabled"]),
            adopted=bool(row["adopted"]),
            pid=row["pid"],
            runner_pid=row["runner_pid"],
            tmux_session=row["tmux_session"],
            max_restarts=int(row["max_restarts"]),
            restart_count=int(row["restart_count"]),
            retry_delay_seconds=float(row["retry_delay_seconds"]),
            next_retry_at=float(row["next_retry_at"]),
            timeout_seconds=float(row["timeout_seconds"]),
            heartbeat_at=row["heartbeat_at"],
            progress_heartbeat_at=row["progress_heartbeat_at"],
            progress_signature=row["progress_signature"],
            progress_path=row["progress_path"],
            log_path=row["log_path"],
            validation_command=json_load(
                row["validation_command_json"], []
            ),
            auto_codex=bool(row["auto_codex"]),
            max_codex_attempts=int(row["max_codex_attempts"]),
            codex_attempts=int(row["codex_attempts"]),
            last_error=row["last_error"],
            metadata=json_load(row["metadata_json"], {}),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def list_tasks(self) -> list[Task]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                ORDER BY priority DESC, position ASC, created_at ASC
                """
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def get(self, task_id: str) -> Task:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown task: {task_id}")
        return self._row_to_task(row)

    def add(
        self,
        *,
        name: str,
        command: Iterable[str],
        cwd: str | Path = ROOT,
        env: dict[str, str] | None = None,
        priority: int = 0,
        kind: str = "batch",
        depends_on: Iterable[str] = (),
        enabled: bool = True,
        adopted: bool = False,
        pid: int | None = None,
        tmux_session: str | None = None,
        max_restarts: int = 20,
        retry_delay_seconds: float = 15,
        timeout_seconds: float = 0,
        progress_path: str | Path | None = None,
        validation_command: Iterable[str] = (),
        auto_codex: bool = True,
        max_codex_attempts: int = 0,
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
        status: str | None = None,
    ) -> Task:
        command = [str(item) for item in command]
        if not command:
            raise ValueError("Task command cannot be empty")
        task_id = task_id or uuid.uuid4().hex[:12]
        timestamp = now()
        cwd_path = str(Path(cwd).resolve())
        log_path = str(DEFAULT_LOG_DIR / f"{task_id}.log")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                position = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(position), 0) + 10 FROM tasks"
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    INSERT INTO tasks (
                        id,name,command_json,cwd,env_json,status,position,
                        priority,kind,depends_on_json,enabled,adopted,pid,
                        tmux_session,max_restarts,retry_delay_seconds,
                        timeout_seconds,progress_path,log_path,
                        validation_command_json,auto_codex,max_codex_attempts,
                        metadata_json,created_at,updated_at,started_at,
                        heartbeat_at,progress_heartbeat_at,progress_signature
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        task_id,
                        name,
                        json_dump(command),
                        cwd_path,
                        json_dump(env or {}),
                        status
                        or ("running" if adopted and pid_alive(pid) else "queued"),
                        position,
                        int(priority),
                        kind,
                        json_dump(list(depends_on)),
                        int(enabled),
                        int(adopted),
                        pid,
                        tmux_session,
                        int(max_restarts),
                        float(retry_delay_seconds),
                        float(timeout_seconds),
                        str(progress_path) if progress_path else None,
                        log_path,
                        json_dump(list(validation_command)),
                        int(auto_codex),
                        int(max_codex_attempts),
                        json_dump(metadata or {}),
                        timestamp,
                        timestamp,
                        timestamp if adopted and pid_alive(pid) else None,
                        timestamp if adopted and pid_alive(pid) else None,
                        timestamp if adopted and pid_alive(pid) else None,
                        None,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(task_id)

    def update(self, task_id: str, **changes: Any) -> Task:
        allowed = {
            "name",
            "command",
            "cwd",
            "env",
            "status",
            "priority",
            "kind",
            "depends_on",
            "enabled",
            "adopted",
            "pid",
            "runner_pid",
            "tmux_session",
            "max_restarts",
            "restart_count",
            "retry_delay_seconds",
            "next_retry_at",
            "timeout_seconds",
            "heartbeat_at",
            "progress_heartbeat_at",
            "progress_signature",
            "progress_path",
            "validation_command",
            "auto_codex",
            "max_codex_attempts",
            "codex_attempts",
            "last_error",
            "metadata",
            "started_at",
            "finished_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Unsupported task fields: {sorted(unknown)}")
        mapping = {
            "command": "command_json",
            "env": "env_json",
            "depends_on": "depends_on_json",
            "validation_command": "validation_command_json",
            "metadata": "metadata_json",
        }
        bool_fields = {"enabled", "adopted", "auto_codex"}
        assignments = []
        values = []
        for field, value in changes.items():
            column = mapping.get(field, field)
            if field in mapping:
                value = json_dump(value)
            elif field in bool_fields:
                value = int(bool(value))
            elif field == "cwd":
                value = str(Path(value).resolve())
            assignments.append(f"{column}=?")
            values.append(value)
        assignments.append("updated_at=?")
        values.append(now())
        values.append(task_id)
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE tasks SET {', '.join(assignments)} WHERE id=?",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown task: {task_id}")
        return self.get(task_id)

    def delete(self, task_id: str, force: bool = False) -> dict[str, Any]:
        task = self.get(task_id)
        if task.status in ACTIVE_STATES and not force:
            raise RuntimeError(
                "Refusing to delete an active task; pause/cancel it or use force"
            )
        snapshot = task.to_dict(include_progress=False)
        incidents = self.list_incidents(task_id)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO task_deletions(
                        task_id,task_name,task_snapshot_json,
                        incidents_snapshot_json,deleted_at
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        task.id,
                        task.name,
                        json_dump(snapshot),
                        json_dump(incidents),
                        now(),
                    ),
                )
                deletion_id = int(cursor.lastrowid)
                connection.execute(
                    "DELETE FROM incidents WHERE task_id=?", (task_id,)
                )
                connection.execute("DELETE FROM tasks WHERE id=?", (task_id,))
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        self.normalize_positions()
        return self.get_deletion(deletion_id)

    def get_deletion(self, deletion_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_deletions WHERE id=?",
                (int(deletion_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown deletion record: {deletion_id}")
        value = dict(row)
        value["task_snapshot"] = json_load(
            value.pop("task_snapshot_json"), {}
        )
        value["incidents_snapshot"] = json_load(
            value.pop("incidents_snapshot_json"), []
        )
        return value

    def list_deletions(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM task_deletions ORDER BY deleted_at DESC"
            ).fetchall()
        return [self.get_deletion(int(row["id"])) for row in rows]

    def add_result_deletion(
        self,
        task: Task,
        targets: list[str],
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        timestamp = now()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO task_result_deletions(
                    task_id,task_name,targets_json,errors_json,deleted_at
                ) VALUES (?,?,?,?,?)
                """,
                (
                    task.id,
                    task.name,
                    json_dump(targets),
                    json_dump(errors),
                    timestamp,
                ),
            )
            record_id = int(cursor.lastrowid)
        return {
            "id": record_id,
            "task_id": task.id,
            "task_name": task.name,
            "targets": targets,
            "errors": errors,
            "deleted_at": timestamp,
        }

    def list_result_deletions(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM task_result_deletions
                ORDER BY deleted_at DESC
                """
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["targets"] = json_load(value.pop("targets_json"), [])
            value["errors"] = json_load(value.pop("errors_json"), [])
            values.append(value)
        return values

    def restore_deletion(self, deletion_id: int) -> Task:
        record = self.get_deletion(deletion_id)
        if record.get("restored_at"):
            restored_id = record.get("restored_task_id")
            if restored_id:
                return self.get(str(restored_id))
            raise RuntimeError("Deletion record was already restored")
        snapshot = record["task_snapshot"]
        task_id = str(snapshot["id"])
        try:
            self.get(task_id)
        except KeyError:
            pass
        else:
            raise RuntimeError(
                f"Cannot restore task {task_id}: id already exists"
            )

        original_status = str(snapshot.get("status") or "paused")
        safe_status = (
            original_status
            if original_status
            in {"succeeded", "cancelled", "paused", "failed", "blocked"}
            else "paused"
        )
        timestamp = now()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        id,name,command_json,cwd,env_json,status,position,
                        priority,kind,depends_on_json,enabled,adopted,pid,
                        runner_pid,tmux_session,max_restarts,restart_count,
                        retry_delay_seconds,next_retry_at,timeout_seconds,
                        heartbeat_at,progress_heartbeat_at,progress_signature,
                        progress_path,log_path,
                        validation_command_json,auto_codex,max_codex_attempts,
                        codex_attempts,last_error,metadata_json,created_at,
                        updated_at,started_at,finished_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        task_id,
                        snapshot["name"],
                        json_dump(snapshot.get("command", [])),
                        snapshot.get("cwd", str(ROOT)),
                        json_dump(snapshot.get("env", {})),
                        safe_status,
                        int(snapshot.get("position", 10)),
                        int(snapshot.get("priority", 0)),
                        snapshot.get("kind", "batch"),
                        json_dump(snapshot.get("depends_on", [])),
                        int(bool(snapshot.get("enabled", True))),
                        0,
                        None,
                        None,
                        None,
                        int(snapshot.get("max_restarts", 20)),
                        int(snapshot.get("restart_count", 0)),
                        float(snapshot.get("retry_delay_seconds", 15)),
                        0.0,
                        float(snapshot.get("timeout_seconds", 0)),
                        None,
                        snapshot.get("progress_heartbeat_at"),
                        snapshot.get("progress_signature"),
                        snapshot.get("progress_path"),
                        snapshot.get(
                            "log_path",
                            str(DEFAULT_LOG_DIR / f"{task_id}.log"),
                        ),
                        json_dump(snapshot.get("validation_command", [])),
                        int(bool(snapshot.get("auto_codex", True))),
                        int(snapshot.get("max_codex_attempts", 0)),
                        int(snapshot.get("codex_attempts", 0)),
                        snapshot.get("last_error"),
                        json_dump(snapshot.get("metadata", {})),
                        float(snapshot.get("created_at", timestamp)),
                        timestamp,
                        snapshot.get("started_at"),
                        snapshot.get("finished_at"),
                    ),
                )
                for incident in record.get("incidents_snapshot", []):
                    connection.execute(
                        """
                        INSERT INTO incidents(
                            task_id,category,message,log_tail,resolution,
                            codex_attempt,created_at,resolved_at
                        ) VALUES (?,?,?,?,?,?,?,?)
                        """,
                        (
                            task_id,
                            incident.get("category", "restored"),
                            incident.get("message", ""),
                            incident.get("log_tail", ""),
                            incident.get("resolution"),
                            int(incident.get("codex_attempt", 0)),
                            float(incident.get("created_at", timestamp)),
                            incident.get("resolved_at"),
                        ),
                    )
                connection.execute(
                    """
                    UPDATE task_deletions
                    SET restored_at=?,restored_task_id=?
                    WHERE id=?
                    """,
                    (timestamp, task_id, int(deletion_id)),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        self.normalize_positions()
        return self.get(task_id)

    def normalize_positions(self) -> None:
        tasks = self.list_tasks()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for index, task in enumerate(tasks, start=1):
                    connection.execute(
                        "UPDATE tasks SET position=?,updated_at=? WHERE id=?",
                        (index * 10, now(), task.id),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def move(self, task_id: str, new_index: int) -> list[Task]:
        tasks = self.list_tasks()
        selected = next((task for task in tasks if task.id == task_id), None)
        if selected is None:
            raise KeyError(f"Unknown task: {task_id}")
        tasks.remove(selected)
        new_index = max(0, min(int(new_index), len(tasks)))
        tasks.insert(new_index, selected)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for index, task in enumerate(tasks, start=1):
                    connection.execute(
                        "UPDATE tasks SET position=?,updated_at=? WHERE id=?",
                        (index * 10, now(), task.id),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.list_tasks()

    def dependencies_satisfied(self, task: Task) -> bool:
        if not task.depends_on:
            return True
        dependencies = {item.id: item for item in self.list_tasks()}
        for task_id in task.depends_on:
            dependency = dependencies.get(task_id)
            if dependency is None:
                return False
            if dependency.kind == "service":
                if dependency.status != "running" or not dependency.to_dict(
                    include_progress=False
                )["alive"]:
                    return False
            elif dependency.status != "succeeded":
                return False
        return True

    def add_incident(
        self, task_id: str, category: str, message: str, log_tail: str = ""
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO incidents(
                    task_id,category,message,log_tail,created_at
                ) VALUES (?,?,?,?,?)
                """,
                (task_id, category, message, log_tail[-20000:], now()),
            )
            return int(cursor.lastrowid)

    def resolve_incident(
        self, incident_id: int, resolution: str, codex_attempt: int = 0
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE incidents SET resolution=?,codex_attempt=?,resolved_at=?
                WHERE id=?
                """,
                (resolution, codex_attempt, now(), int(incident_id)),
            )

    def list_incidents(self, task_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as connection:
            if task_id:
                rows = connection.execute(
                    "SELECT * FROM incidents WHERE task_id=? ORDER BY id DESC",
                    (task_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM incidents ORDER BY id DESC"
                ).fetchall()
        return [dict(row) for row in rows]

    def setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key=?", (key,)
            ).fetchone()
        return default if row is None else json_load(row[0], default)

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO settings(key,value_json,updated_at) VALUES (?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, json_dump(value), now()),
            )


def tail_text(path: str | Path, max_bytes: int = 64_000) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    with target.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def task_log_paths(task: Task) -> list[Path]:
    """Return every existing log associated with a task.

    Adopted tasks may not have written the queue-owned ``log_path``. Their
    actual output can live in metadata-provided files or in dynamically
    created stage directories below ``adaptive_logs``.
    """

    candidates: list[Path] = []

    def add_path(value: str | Path | None) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(task.cwd) / path
        candidates.append(path)

    add_path(task.log_path)
    for value in task.metadata.get("source_logs", []):
        add_path(value)

    for pattern in task.metadata.get("log_globs", []):
        pattern_path = Path(pattern).expanduser()
        if not pattern_path.is_absolute():
            pattern_path = Path(task.cwd) / pattern_path
        for match in glob(str(pattern_path), recursive=True):
            add_path(match)

    # Wrapper pipelines (for example Challenge question-only/full/oracle/
    # ablation) keep each nested run_hard_eval worker log under the atomic
    # task's output directory rather than next to the wrapper log.
    for item in task.metadata.get("atomic_tasks", []):
        if not isinstance(item, dict) or not item.get("output_dir"):
            continue
        output_dir = Path(str(item["output_dir"])).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(task.cwd) / output_dir
        adaptive_root = output_dir / "adaptive_logs"
        if adaptive_root.is_dir():
            candidates.extend(adaptive_root.glob("**/*.log"))

    output_dirs: list[Path] = []
    for index, token in enumerate(task.command[:-1]):
        if token == "--output-dir":
            output_dir = Path(task.command[index + 1]).expanduser()
            if not output_dir.is_absolute():
                output_dir = Path(task.cwd) / output_dir
            output_dirs.append(output_dir)

    adaptive_roots = [output_dir / "adaptive_logs" for output_dir in output_dirs]
    for candidate in list(candidates):
        for parent in candidate.parents:
            if parent.name == "adaptive_logs":
                adaptive_roots.append(parent)
                break
    for adaptive_root in adaptive_roots:
        if adaptive_root.is_dir():
            candidates.extend(adaptive_root.glob("**/*.log"))

    unique: dict[str, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file():
                unique[str(resolved)] = resolved
        except OSError:
            continue
    return list(unique.values())


def task_log_snapshot(
    task: Task,
    max_bytes: int = 64_000,
    stage_id: str | None = None,
) -> dict[str, Any]:
    """Read a bounded, newest-first snapshot of all logs for ``task``."""

    max_bytes = max(1_024, min(int(max_bytes), 1_000_000))
    stage_progress = log_progress_context(task, stage_id)
    sources: list[dict[str, Any]] = []
    for path in task_log_paths(task):
        if stage_id and not log_matches_task_stage(task, path, stage_id):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        sources.append(
            {
                "path": str(path),
                "name": path.name,
                "stage": path.parent.name,
                "size_bytes": stat.st_size,
                "updated_at": stat.st_mtime,
                "progress_percent": stage_progress["percent"],
                "progress_scope": stage_progress["scope"],
                "_path": path,
            }
        )
    sources.sort(key=lambda item: (item["updated_at"], item["path"]), reverse=True)

    # Keep the response useful when an experiment has accumulated many old
    # worker logs. The freshest log receives most of the byte budget.
    visible = sources[:8]
    chunks: list[str] = []
    remaining = max_bytes
    for index, source in enumerate(visible):
        if remaining <= 0:
            break
        if index == 0:
            allocation = max(1_024, int(max_bytes * 0.7))
        else:
            later = len(visible) - index
            allocation = max(1_024, remaining // max(1, later))
        allocation = min(allocation, remaining)
        label = "当前日志" if index == 0 else "关联日志"
        header = (
            f"===== {label} · {source['stage']}/{source['name']} =====\n"
            f"{source['path']}\n"
        )
        chunks.append(
            annotate_log_progress(
                header + tail_text(source["_path"], allocation),
                source["progress_percent"],
                source["progress_scope"],
            )
        )
        remaining -= allocation

    public_sources = []
    for index, source in enumerate(sources):
        public_sources.append(
            {
                key: value
                for key, value in source.items()
                if key != "_path"
            }
            | {"current": index == 0}
        )
    current_log = ""
    if sources:
        source = sources[0]
        current_log = annotate_log_progress(
            f"===== 当前日志 · {source['stage']}/{source['name']} =====\n"
            f"{source['path']}\n"
            + tail_text(source["_path"], max_bytes),
            source["progress_percent"],
            source["progress_scope"],
        )
    return {
        "task_id": task.id,
        "stage_id": stage_id,
        "log": "\n\n".join(chunks),
        # The GUI follows only the freshest worker log. ``log`` remains the
        # combined audit view for CLI callers and backwards compatibility.
        "current_log": current_log,
        "sources": public_sources,
        "primary_log_missing": not Path(task.log_path).is_file(),
    }


STAGE_NAMES = {
    "inference_4B": "4B 推理",
    "inference_8B": "8B 推理",
    "judge_4B": "4B 判卷",
    "judge_8B": "8B 判卷",
    "report": "汇总报告",
    "main": "主程序",
}
HARD_EVAL_STAGE_ORDER = (
    "inference_4B",
    "inference_8B",
    "judge_4B",
    "judge_8B",
    "report",
)


def log_matches_stage(path: Path, stage_id: str) -> bool:
    normalized = stage_id.lower()
    return (
        path.parent.name.lower() == normalized
        or path.stem.lower() == normalized
        or path.name.lower().startswith(normalized)
        or (normalized in {"main", "report"} and path.parent.name == "logs")
    )


def log_matches_task_stage(task: Task, path: Path, stage_id: str) -> bool:
    if log_matches_stage(path, stage_id):
        return True
    for item in task.metadata.get("atomic_tasks", []):
        if not isinstance(item, dict) or str(item.get("id")) != stage_id:
            continue
        output_value = item.get("output_dir")
        if not output_value:
            continue
        output_dir = Path(str(output_value)).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(task.cwd) / output_dir
        try:
            path.resolve().relative_to(output_dir.resolve())
            return True
        except (OSError, ValueError):
            continue
    return False


def log_progress_context(
    task: Task, stage_id: str | None
) -> dict[str, Any]:
    stages = task_stage_snapshots(task)
    selected = next(
        (stage for stage in stages if stage.get("id") == stage_id),
        None,
    )
    if selected is not None:
        return {
            "scope": str(selected.get("id") or task.id),
            "percent": float(selected.get("percent") or 0),
        }
    aggregate = aggregate_stage_progress(task, stages) or {}
    return {
        "scope": str(aggregate.get("stage") or task.id),
        "percent": float(aggregate.get("percent") or 0),
    }


def annotate_log_progress(
    text: str,
    percent: float,
    scope: str,
) -> str:
    """Ensure every displayed non-empty log line carries a percentage."""

    prefix = (
        f"[progress_scope={scope} progress_percent={percent:.2f}% "
        "progress_basis=current_stage_snapshot] "
    )
    lines = text.splitlines(keepends=True)
    annotated = []
    for line in lines:
        if not line.strip() or "progress_percent=" in line:
            annotated.append(line)
            continue
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        annotated.append(prefix + body + ending)
    return "".join(annotated)


def command_output_dirs(task: Task) -> list[Path]:
    output_dirs: list[Path] = []
    for index, token in enumerate(task.command[:-1]):
        if token != "--output-dir":
            continue
        output_dir = Path(task.command[index + 1]).expanduser()
        if not output_dir.is_absolute():
            output_dir = Path(task.cwd) / output_dir
        output_dirs.append(output_dir)
    return output_dirs


def task_result_targets(task: Task) -> list[Path]:
    """Resolve only task-owned outputs, checkpoints, and logs.

    Input QA/PDF paths are deliberately excluded. Every returned target must
    remain inside the project root, and broad project directories are refused.
    """

    candidates: list[Path] = []

    def add(value: str | Path | None) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(task.cwd) / path
        candidates.append(path)

    for path in command_output_dirs(task):
        add(path)
    add(task.log_path)
    add(task.progress_path)
    for key in (
        "completion_paths",
        "source_logs",
    ):
        for value in task.metadata.get(key, []):
            add(value)
    add(task.metadata.get("completion_state_path"))
    for item in task.metadata.get("atomic_tasks", []):
        if isinstance(item, dict):
            add(item.get("output_dir"))
            add(item.get("progress_path"))
    for pattern in task.metadata.get("log_globs", []):
        pattern_path = Path(pattern).expanduser()
        if not pattern_path.is_absolute():
            pattern_path = Path(task.cwd) / pattern_path
        for match in glob(str(pattern_path), recursive=True):
            add(match)

    root = ROOT.resolve()
    protected = {
        root,
        (root / "data").resolve(),
        (root / "logs").resolve(),
        (root / "data/results/evaluations").resolve(),
        (root / "data/web/task_queue").resolve(),
    }
    safe: dict[str, Path] = {}
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved in protected or not resolved.exists():
            continue
        if resolved.is_dir() and len(resolved.relative_to(root).parts) < 3:
            continue
        safe[str(resolved)] = resolved

    # If an output directory is selected, do not list its children again.
    ordered = sorted(safe.values(), key=lambda path: len(path.parts))
    roots: list[Path] = []
    for candidate in ordered:
        if any(candidate == parent or parent in candidate.parents for parent in roots):
            continue
        roots.append(candidate)
    return roots


def delete_task_results(task: Task) -> tuple[list[str], list[dict[str, str]]]:
    if task.status in ACTIVE_STATES:
        raise RuntimeError(
            "Refusing to delete results for an active task; pause it first"
        )
    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    for target in task_result_targets(task):
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(str(target))
        except OSError as exc:
            errors.append({"path": str(target), "error": str(exc)})
    return deleted, errors


def configured_stage_statuses(task: Task) -> dict[str, str]:
    """Read a wrapper pipeline's stage states from its main progress file."""

    progress = read_progress(task.progress_path, follow_pipeline=False)
    raw = (progress or {}).get("raw")
    values = raw.get("stages") if isinstance(raw, dict) else None
    if not isinstance(values, list):
        return {}
    statuses: dict[str, str] = {}
    for item in values:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        statuses[str(item["name"])] = str(item.get("status") or "queued")
    return statuses


def hard_eval_output_summary(
    output_dir: Path,
    *,
    status_hint: str = "queued",
) -> dict[str, Any]:
    """Aggregate one run_hard_eval output directory into one wrapper stage."""

    adaptive_root = output_dir / ".adaptive_queue"
    percents: list[float] = []
    completed_stages = 0
    current_progress: dict[str, Any] | None = None
    failed_progress: dict[str, Any] | None = None
    updated_values: list[float] = []
    observed_failure = status_hint == "failed"

    for stage_id in HARD_EVAL_STAGE_ORDER[:-1]:
        state_path = adaptive_root / stage_id / "scheduler_state.json"
        progress = read_progress(str(state_path), follow_pipeline=False)
        if not state_path.exists():
            percents.append(0.0)
            continue
        raw_status = str((progress or {}).get("status") or "queued")
        stage_percent = float((progress or {}).get("percent") or 0)
        if raw_status == "completed":
            stage_percent = 100.0
            completed_stages += 1
        elif raw_status == "failed":
            observed_failure = True
            failed_progress = progress
        elif raw_status == "running":
            current_progress = progress
        percents.append(stage_percent)
        updated = (progress or {}).get("updated_at")
        if isinstance(updated, (int, float)):
            updated_values.append(float(updated))

    report_path = output_dir / "final_report.json"
    report_done = report_path.exists()
    if report_done:
        # A final report is the pipeline completion contract even when old
        # scheduler-state files have been pruned.
        percents = [100.0] * len(HARD_EVAL_STAGE_ORDER)
        completed_stages = len(HARD_EVAL_STAGE_ORDER)
        updated_values.append(report_path.stat().st_mtime)
    else:
        percents.append(0.0)

    normalized_hint = {
        "completed": "succeeded",
        "running": "running",
        "failed": "failed",
    }.get(status_hint, status_hint)
    if report_done:
        status = "succeeded"
    elif observed_failure:
        status = "failed"
    elif current_progress is not None or normalized_hint == "running":
        status = "running"
    else:
        status = normalized_hint or "queued"

    health_progress = failed_progress or current_progress or {}
    return {
        "status": status,
        "completed": completed_stages,
        "total": len(HARD_EVAL_STAGE_ORDER),
        "percent": sum(percents) / len(percents),
        "workers": (current_progress or {}).get("workers") or {},
        "updated_at": max(updated_values) if updated_values else None,
        "progress_path": health_progress.get("path"),
        "process_heartbeat_at": health_progress.get(
            "process_heartbeat_at"
        ),
        "progress_heartbeat_at": health_progress.get(
            "progress_heartbeat_at"
        ),
        "work_state": health_progress.get("work_state"),
        "resource_waiting": bool(
            health_progress.get("resource_waiting")
        ),
        "consecutive_worker_failures_without_progress": (
            health_progress.get(
                "consecutive_worker_failures_without_progress", 0
            )
        ),
        "error": health_progress.get("error"),
    }


def task_stage_snapshots(task: Task) -> list[dict[str, Any]]:
    """Expose a long command as independently inspectable atomic stages."""

    output_dirs = command_output_dirs(task)
    adaptive_root = next(
        (
            output_dir / ".adaptive_queue"
            for output_dir in output_dirs
            if (output_dir / ".adaptive_queue").is_dir()
        ),
        None,
    )
    if adaptive_root is not None:
        output_dir = adaptive_root.parent
        stages: list[dict[str, Any]] = []
        prior_complete = True
        for stage_id in HARD_EVAL_STAGE_ORDER:
            if stage_id == "report":
                report_path = output_dir / "final_report.json"
                completed = int(report_path.exists())
                status = (
                    "succeeded"
                    if completed
                    else (
                        "running"
                        if prior_complete and task.status in ACTIVE_STATES
                        else "queued"
                    )
                )
                stage = {
                    "id": stage_id,
                    "name": STAGE_NAMES[stage_id],
                    "status": status,
                    "completed": completed,
                    "total": 1,
                    "percent": float(completed * 100),
                    "progress_path": str(report_path),
                    "workers": {},
                    "updated_at": (
                        report_path.stat().st_mtime
                        if report_path.exists()
                        else None
                    ),
                }
            else:
                state_path = adaptive_root / stage_id / "scheduler_state.json"
                progress = read_progress(
                    str(state_path), follow_pipeline=False
                )
                if not state_path.exists():
                    status = "queued"
                    completed = 0
                    total = None
                    percent = 0.0
                    workers = {}
                    updated_at = None
                else:
                    raw_status = (progress or {}).get("status")
                    status = (
                        "succeeded"
                        if raw_status == "completed"
                        else (
                            "running"
                            if raw_status == "running"
                            else str(raw_status or "queued")
                        )
                    )
                    completed = (progress or {}).get("completed")
                    total = (progress or {}).get("total")
                    percent = float((progress or {}).get("percent") or 0)
                    workers = (progress or {}).get("workers") or {}
                    updated_at = (progress or {}).get("updated_at")
                stage = {
                    "id": stage_id,
                    "name": STAGE_NAMES[stage_id],
                    "status": status,
                    "completed": completed,
                    "total": total,
                    "percent": percent,
                    "progress_path": str(state_path),
                    "workers": workers,
                    "updated_at": updated_at,
                    "process_heartbeat_at": (progress or {}).get(
                        "process_heartbeat_at"
                    ),
                    "progress_heartbeat_at": (progress or {}).get(
                        "progress_heartbeat_at"
                    ),
                    "work_state": (progress or {}).get("work_state"),
                    "resource_waiting": bool(
                        (progress or {}).get("resource_waiting")
                    ),
                    "consecutive_worker_failures_without_progress": (
                        (progress or {}).get(
                            "consecutive_worker_failures_without_progress",
                            0,
                        )
                    ),
                    "error": (progress or {}).get("error"),
                }
            stage["log_count"] = sum(
                log_matches_task_stage(task, path, stage_id)
                for path in task_log_paths(task)
            )
            stages.append(stage)
            prior_complete = prior_complete and status == "succeeded"
        return stages

    configured = task.metadata.get("atomic_tasks", [])
    if isinstance(configured, list) and configured:
        stages = []
        status_hints = configured_stage_statuses(task)
        for index, item in enumerate(configured):
            if not isinstance(item, dict):
                continue
            stage_id = str(item.get("id") or f"stage_{index + 1}")
            output_dir_value = item.get("output_dir")
            if output_dir_value:
                output_dir = Path(str(output_dir_value)).expanduser()
                if not output_dir.is_absolute():
                    output_dir = Path(task.cwd) / output_dir
                progress = hard_eval_output_summary(
                    output_dir,
                    status_hint=status_hints.get(stage_id, "queued"),
                )
            else:
                progress = read_progress(item.get("progress_path"))
            stages.append(
                {
                    "id": stage_id,
                    "name": str(item.get("name") or stage_id),
                    "status": (progress or {}).get("status", "queued"),
                    "completed": (progress or {}).get("completed"),
                    "total": (progress or {}).get("total"),
                    "percent": float((progress or {}).get("percent") or 0),
                    "progress_path": (
                        (progress or {}).get("progress_path")
                        or item.get("progress_path")
                    ),
                    "workers": (progress or {}).get("workers") or {},
                    "updated_at": (progress or {}).get("updated_at"),
                    "process_heartbeat_at": (progress or {}).get(
                        "process_heartbeat_at"
                    ),
                    "progress_heartbeat_at": (progress or {}).get(
                        "progress_heartbeat_at"
                    ),
                    "work_state": (progress or {}).get("work_state"),
                    "resource_waiting": bool(
                        (progress or {}).get("resource_waiting")
                    ),
                    "consecutive_worker_failures_without_progress": (
                        (progress or {}).get(
                            "consecutive_worker_failures_without_progress",
                            0,
                        )
                    ),
                    "error": (progress or {}).get("error"),
                    "log_count": sum(
                        log_matches_task_stage(task, path, stage_id)
                        for path in task_log_paths(task)
                    ),
                }
            )
        if stages:
            return stages

    progress = read_progress(task.progress_path)
    return [
        {
            "id": "main",
            "name": STAGE_NAMES["main"],
            "status": (
                "succeeded" if task.status == "succeeded" else task.status
            ),
            "completed": (progress or {}).get("completed"),
            "total": (progress or {}).get("total"),
            "percent": (
                100.0
                if task.status == "succeeded"
                else float((progress or {}).get("percent") or 0)
            ),
            "progress_path": task.progress_path,
            "workers": (progress or {}).get("workers") or {},
            "updated_at": (progress or {}).get("updated_at"),
            "process_heartbeat_at": (progress or {}).get(
                "process_heartbeat_at"
            ),
            "progress_heartbeat_at": (progress or {}).get(
                "progress_heartbeat_at"
            ),
            "work_state": (progress or {}).get("work_state"),
            "resource_waiting": bool(
                (progress or {}).get("resource_waiting")
            ),
            "consecutive_worker_failures_without_progress": (
                (progress or {}).get(
                    "consecutive_worker_failures_without_progress", 0
                )
            ),
            "error": (progress or {}).get("error"),
            "log_count": len(task_log_paths(task)),
        }
    ]


def aggregate_stage_progress(
    task: Task, stages: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not stages:
        return read_progress(task.progress_path)
    percent = sum(float(stage.get("percent") or 0) for stage in stages) / len(
        stages
    )
    current = next(
        (
            stage
            for stage in stages
            if stage.get("status") in {"running", "recovering", "starting"}
        ),
        next(
            (
                stage
                for stage in stages
                if stage.get("status") not in {"succeeded", "completed"}
            ),
            stages[-1],
        ),
    )
    if len(stages) == 1:
        aggregate_completed = stages[0].get("completed")
        aggregate_total = stages[0].get("total")
    else:
        aggregate_completed = sum(
            1
            for stage in stages
            if stage.get("status") in {"succeeded", "completed"}
        )
        aggregate_total = len(stages)
    return {
        "status": task.status,
        "completed": aggregate_completed,
        "total": aggregate_total,
        "percent": percent,
        "stage": current.get("id"),
        "stage_name": current.get("name"),
        "stage_percent": current.get("percent"),
        "workers": current.get("workers") or {},
        "updated_at": current.get("updated_at"),
        "process_heartbeat_at": current.get("process_heartbeat_at"),
        "progress_heartbeat_at": current.get("progress_heartbeat_at"),
        "work_state": current.get("work_state"),
        "resource_waiting": bool(current.get("resource_waiting")),
        "consecutive_worker_failures_without_progress": current.get(
            "consecutive_worker_failures_without_progress", 0
        ),
        "error": current.get("error"),
    }


def read_progress(
    path: str | None, *, follow_pipeline: bool = True
) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path)
    configured_target = target
    # A hard-eval pipeline keeps one scheduler state per stage. The task was
    # historically registered against inference_4B, which made the GUI stay
    # at 100% after the pipeline had already advanced to 8B or judging. Follow
    # the freshest running stage automatically while retaining the configured
    # path for audit/debugging.
    if (
        follow_pipeline
        and
        target.name == "scheduler_state.json"
        and target.parent.parent.name == ".adaptive_queue"
    ):
        candidates = list(
            target.parent.parent.glob("*/scheduler_state.json")
        )
        parsed: list[tuple[Path, dict[str, Any]]] = []
        for candidate in candidates:
            try:
                candidate_value = json.loads(
                    candidate.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(candidate_value, dict):
                parsed.append((candidate, candidate_value))
        running = [
            item for item in parsed if item[1].get("status") == "running"
        ]
        selectable = running or parsed
        if selectable:
            target = max(
                selectable,
                key=lambda item: item[0].stat().st_mtime,
            )[0]
    if not target.exists():
        return {"path": str(target), "status": "missing"}
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        stat = target.stat()
        return {
            "path": str(target),
            "status": "present",
            "size_bytes": stat.st_size,
            "updated_at": stat.st_mtime,
        }
    if not isinstance(value, dict):
        return {"path": str(target), "status": "present", "value": value}
    queue = value.get("queue") if isinstance(value.get("queue"), dict) else {}
    completed = queue.get("completed_papers", value.get("completed"))
    total = queue.get("total_papers", value.get("total"))
    percent = None
    if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total:
        percent = completed / total * 100
    return {
        "path": str(target),
        "configured_path": str(configured_target),
        "stage": (
            target.parent.name
            if target.name == "scheduler_state.json"
            else None
        ),
        "status": value.get("status", "present"),
        "completed": completed,
        "total": total,
        "percent": percent,
        "workers": value.get("workers"),
        "updated_at": value.get("updated_at", target.stat().st_mtime),
        "process_heartbeat_at": value.get(
            "process_heartbeat_at",
            value.get("updated_at", target.stat().st_mtime),
        ),
        "progress_heartbeat_at": value.get(
            "progress_heartbeat_at",
            value.get("last_progress_at"),
        ),
        "work_state": value.get("work_state"),
        "resource_waiting": bool(value.get("resource_waiting")),
        "consecutive_worker_failures_without_progress": value.get(
            "consecutive_worker_failures_without_progress", 0
        ),
        "error": value.get("error"),
        "blocked_reasons": value.get("blocked_reasons") or {},
        "raw": value,
    }


FAILURE_PATTERNS = (
    (
        "worker_failure",
        re.compile(
            r"(quarantinedworkerror|paper\(s\) quarantined|"
            r"workerprogressstallerror|consecutive worker failures|"
            r"worker_failure_without_progress|malformed model output|"
            r"invalid structured output)",
            re.IGNORECASE,
        ),
    ),
    (
        "oom",
        re.compile(
            r"(out of memory|cuda oom|cublas.*alloc|memory allocation)",
            re.IGNORECASE,
        ),
    ),
    (
        "gpu_busy",
        re.compile(
            r"(device.*busy|waiting for foreign jobs|no free a800|gpu.*occupied)",
            re.IGNORECASE,
        ),
    ),
    (
        "disk_full",
        re.compile(r"(no space left on device|disk quota)", re.IGNORECASE),
    ),
    (
        "data_error",
        re.compile(
            r"(jsondecodeerror|invalid json|pdf not found|corrupt|schema)",
            re.IGNORECASE,
        ),
    ),
    (
        "transient",
        re.compile(
            r"(timeout|temporarily unavailable|connection reset|broken pipe|epipe)",
            re.IGNORECASE,
        ),
    ),
)


def classify_failure(text: str) -> str:
    for category, pattern in FAILURE_PATTERNS:
        if pattern.search(text or ""):
            return category
    return "unknown"


def strip_option_with_values(command: list[str], option: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(command):
        if command[index] != option:
            output.append(command[index])
            index += 1
            continue
        index += 1
        while index < len(command) and not command[index].startswith("--"):
            index += 1
    return output


def make_dynamic_a800(command: list[str]) -> list[str]:
    updated = list(command)
    for option in ("--gpus-4b", "--gpus-8b", "--judge-gpus"):
        updated = strip_option_with_values(updated, option)
    if "--dynamic-a800" not in updated:
        updated.extend(
            [
                "--dynamic-a800",
                "--a800-gpus",
                "2",
                "3",
                "4",
                "5",
                "--gpu-poll-seconds",
                "15",
            ]
        )
    if "--allow-shared-a800" not in updated:
        updated.append("--allow-shared-a800")
    if "--shared-a800-min-free-mib" not in updated:
        updated.extend(["--shared-a800-min-free-mib", "56000"])
    if "--max-worker-backoff-seconds" not in updated:
        updated.extend(["--max-worker-backoff-seconds", "30"])
    return updated


def tmux_session_exists(session: str | None) -> bool:
    if not session:
        return False
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0
