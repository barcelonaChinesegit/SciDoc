#!/usr/bin/env python3
"""Tiny outer watchdog that restarts the queue daemon itself."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from task_queue_manager import DEFAULT_DB, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    log_path = ROOT / "data/web/task_queue/supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    while True:
        environment = dict(os.environ)
        source_root = str(ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            source_root
            if not existing_pythonpath
            else f"{source_root}{os.pathsep}{existing_pythonpath}"
        )
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pku_qa.services.task_queue.task_queue_daemon",
                    "--db",
                    args.db,
                    "--poll-seconds",
                    str(args.poll_seconds),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(
                f"[supervisor] daemon exited={process.returncode}; restarting\n"
            )
        failures += 1
        time.sleep(min(60, 2 ** min(failures, 6)))


if __name__ == "__main__":
    main()
