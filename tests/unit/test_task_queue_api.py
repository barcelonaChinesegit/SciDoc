from __future__ import annotations

import tempfile
from pathlib import Path

import task_queue_api
from task_queue_manager import TaskStore


def test_gpu_status_cache_reuses_probe_within_ttl(monkeypatch) -> None:
    calls = []
    clock = [10.0]

    def fake_query_gpus():
        calls.append(clock[0])
        return [{"index": len(calls)}]

    monkeypatch.setattr(task_queue_api, "query_gpus", fake_query_gpus)
    monkeypatch.setattr(task_queue_api.time, "monotonic", lambda: clock[0])
    cache = task_queue_api.GpuStatusCache(ttl_seconds=30)

    assert cache.get() == [{"index": 1}]
    clock[0] = 39.9
    assert cache.get() == [{"index": 1}]
    assert len(calls) == 1

    clock[0] = 40.0
    assert cache.get() == [{"index": 2}]
    assert len(calls) == 2


def test_gpu_status_cache_temporarily_caches_probe_error(monkeypatch) -> None:
    calls = []

    def failing_query_gpus():
        calls.append(True)
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(task_queue_api, "query_gpus", failing_query_gpus)
    cache = task_queue_api.GpuStatusCache(ttl_seconds=30)

    expected = [{"error": "probe unavailable"}]
    assert cache.get() == expected
    assert cache.get() == expected
    assert len(calls) == 1


def test_task_store_records_actor_audit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = TaskStore(Path(tmp) / "tasks.sqlite3")
        recorded = store.record_audit(
            "task.pause",
            task_id="task-1",
            actor_user_id="user-1",
            actor_username="review-admin",
            details={"status": "paused"},
        )
        assert recorded["actor_username"] == "review-admin"
        events = store.list_audit()
        assert events == [recorded]
