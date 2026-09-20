from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pku_qa.services.task_queue.task_queue_manager import TaskStore
from pku_qa.workflows.operations import bootstrap_data_review_service as bootstrap


def test_registered_review_service_uses_working_module_and_validation_paths(
    tmp_path: Path, monkeypatch,
) -> None:
    database = tmp_path / "tasks.sqlite3"
    monkeypatch.setattr(bootstrap, "DEFAULT_DB", database)
    bootstrap.main()
    bootstrap.main()
    store = TaskStore(database)
    task = store.get("data_api")
    assert task.command[:3] == [
        sys.executable, "-m", "pku_qa.services.review.data_manager_api",
    ]
    assert Path(task.cwd).resolve() == bootstrap.ROOT
    assert (Path(task.cwd) / task.validation_command[-1]).is_file()
    result = subprocess.run(
        [*task.command, "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(bootstrap.ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
