#!/usr/bin/env python3
"""GPU admission and repository-local coordination helpers."""

from __future__ import annotations

import contextlib
import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterator


GPU_COORDINATION_ENV = "GPU_COORDINATION_AUTO"
GPU_POLL_ENV = "GPU_WAIT_POLL_SECONDS"
GPU_WAIT_TIMEOUT_ENV = "GPU_WAIT_TIMEOUT_SECONDS"
GPU_REQUIRE_NAME_ENV = "GPU_REQUIRE_NAME"
GPU_ALLOW_SHARED_ENV = "GPU_ALLOW_SHARED"
GPU_MIN_FREE_MEMORY_ENV = "GPU_MIN_FREE_MEMORY_MIB"
GPU_REPOSITORY_LOCK_ENV = "GPU_REPOSITORY_LOCK"
GPU_LOCK_DIR = Path("/tmp/czj_gpu_task_locks")
GPU_ARGUMENTS = {
    "--gpu",
    "--gpus",
    "--inference-gpus",
    "--judge-gpus",
    "--gpus-4b",
    "--gpus-8b",
}
SCRIPT_DEFAULT_GPUS = {
    "run_inference.py": ["2"],
    "run_judge.py": ["2"],
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def gpu_coordination_enabled() -> bool:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return False
    value = os.environ.get(GPU_COORDINATION_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off", "disable", "disabled"}


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "disable",
        "disabled",
    }


def child_env_without_auto(env: dict[str, str] | None = None) -> dict[str, str]:
    child_env = dict(os.environ if env is None else env)
    child_env[GPU_COORDINATION_ENV] = "0"
    return child_env


def _parse_gpu_values(values: list[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece.isdigit() and piece not in parsed:
                parsed.append(piece)
    return parsed


def infer_requested_gpu_ids(argv: list[str] | None = None) -> list[str]:
    """Infer physical GPU ids from common project CLI arguments."""
    args = list(sys.argv[1:] if argv is None else argv)
    values: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item in GPU_ARGUMENTS:
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                values.append(args[index])
                index += 1
            continue
        index += 1

    parsed = _parse_gpu_values(values)
    if parsed:
        return parsed

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parsed = _parse_gpu_values([visible])
        if parsed:
            return parsed

    # Audit scripts use device_map="cuda:0" without a separate --gpu flag.
    for index, item in enumerate(args):
        if item == "--hf-device-map" and index + 1 < len(args):
            match = args[index + 1].strip().lower()
            if match.startswith("cuda:"):
                return _parse_gpu_values([match.split(":", 1)[1]])
    return SCRIPT_DEFAULT_GPUS.get(Path(sys.argv[0]).name, [])


def _gpu_inventory() -> dict[str, dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid",
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
            or "nvidia-smi GPU query failed"
        )
    inventory: dict[str, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3:
            index, name, uuid = parts
            inventory[index] = {"name": name, "uuid": uuid}
    return inventory


def _gpu_compute_processes() -> dict[str, list[dict[str, str]]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
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
            or "nvidia-smi process query failed"
        )
    processes: dict[str, list[dict[str, str]]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        uuid, pid, used_memory = parts
        processes.setdefault(uuid, []).append(
            {"pid": pid, "used_memory_mib": used_memory}
        )
    return processes


def _gpu_free_memory() -> dict[str, int]:
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
    free_memory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2 and parts[0].isdigit():
            free_memory[parts[0]] = int(parts[1])
    return free_memory


def wait_for_gpus(
    gpu_ids: list[str],
    label: str = "gpu task",
    ignore_pids: set[int] | None = None,
) -> None:
    """Wait for target GPUs using the project-wide shared-memory policy.

    Foreign processes are allowed by default when every requested GPU still
    has at least ``GPU_MIN_FREE_MEMORY_MIB`` available. Setting
    ``GPU_ALLOW_SHARED=0`` restores exclusive waiting.
    """
    if not gpu_ids:
        return

    inventory = _gpu_inventory()
    missing = [gpu for gpu in gpu_ids if gpu not in inventory]
    if missing:
        raise ValueError(f"Unknown physical GPU id(s): {', '.join(missing)}")

    required_name = os.environ.get(GPU_REQUIRE_NAME_ENV, "").strip().lower()
    if required_name:
        wrong_type = [
            f"{gpu} ({inventory[gpu]['name']})"
            for gpu in gpu_ids
            if required_name not in inventory[gpu]["name"].lower()
        ]
        if wrong_type:
            raise RuntimeError(
                f"{label} requires GPU name containing "
                f"{required_name!r}; rejected: {', '.join(wrong_type)}"
            )

    ignored = set(ignore_pids or set())
    ignored.add(os.getpid())
    poll_seconds = max(1.0, float(os.environ.get(GPU_POLL_ENV, "30")))
    timeout_seconds = max(
        0.0, float(os.environ.get(GPU_WAIT_TIMEOUT_ENV, "0"))
    )
    allow_shared = env_flag(GPU_ALLOW_SHARED_ENV, True)
    min_free_memory_mib = max(
        0, int(os.environ.get(GPU_MIN_FREE_MEMORY_ENV, "56000"))
    )
    started = time.monotonic()
    last_report: tuple[tuple[str, tuple[str, ...]], ...] | None = None

    while True:
        by_uuid = _gpu_compute_processes()
        free_memory = _gpu_free_memory()
        busy: dict[str, list[dict[str, str]]] = {}
        for gpu in gpu_ids:
            active = [
                process
                for process in by_uuid.get(inventory[gpu]["uuid"], [])
                if int(process["pid"]) not in ignored
            ]
            enough_shared_memory = (
                allow_shared
                and free_memory.get(gpu, 0) >= min_free_memory_mib
            )
            if active and not enough_shared_memory:
                busy[gpu] = active

        if not busy:
            waited = time.monotonic() - started
            print(
                f"[gpu-policy] event=ready task={label!r} "
                f"gpus={','.join(gpu_ids)} shared={allow_shared} "
                f"min_free_mib={min_free_memory_mib} "
                f"free_mib={{{', '.join(f'{gpu}:{free_memory.get(gpu, 0)}' for gpu in gpu_ids)}}} "
                f"waited_seconds={waited:.1f}",
                flush=True,
            )
            return

        report_key = tuple(
            (gpu, tuple(process["pid"] for process in processes))
            for gpu, processes in sorted(busy.items())
        )
        if report_key != last_report:
            detail = "; ".join(
                f"GPU {gpu} ({inventory[gpu]['name']}): "
                + ", ".join(
                    f"pid={process['pid']} memory={process['used_memory_mib']}MiB"
                    for process in processes
                )
                for gpu, processes in sorted(busy.items())
            )
            print(
                f"[gpu-policy] event=waiting task={label!r} "
                f"shared={allow_shared} min_free_mib={min_free_memory_mib} "
                f"detail={detail}",
                flush=True,
            )
            last_report = report_key

        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for GPU(s) {','.join(gpu_ids)} "
                f"after {timeout_seconds:.0f}s"
            )
        time.sleep(poll_seconds)


@contextlib.contextmanager
def _gpu_task_locks(gpu_ids: list[str], label: str) -> Iterator[None]:
    """Serialize this repository's GPU tasks while foreign jobs are monitored."""
    if not gpu_ids:
        yield
        return

    GPU_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_files = []
        for gpu in sorted(gpu_ids, key=int):
            lock_file = stack.enter_context(
                (GPU_LOCK_DIR / f"gpu_{gpu}.lock").open("a+", encoding="utf-8")
            )
            print(
                f"[gpu-lock] {label}: waiting for repository lock on GPU {gpu}",
                flush=True,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            lock_files.append(lock_file)
        try:
            yield
        finally:
            for lock_file in reversed(lock_files):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def managed_gpu_reservation(
    label: str = "gpu task",
    enabled: bool = True,
    gpu_ids: list[str] | None = None,
) -> Iterator[None]:
    should_coordinate = enabled and gpu_coordination_enabled()
    requested_gpus = (
        _parse_gpu_values(gpu_ids)
        if gpu_ids is not None
        else infer_requested_gpu_ids()
    )
    should_lock_repository = env_flag(GPU_REPOSITORY_LOCK_ENV, False)
    lock_context = (
        _gpu_task_locks(requested_gpus, label)
        if should_coordinate and should_lock_repository
        else contextlib.nullcontext()
    )
    with lock_context:
        if should_coordinate:
            wait_for_gpus(requested_gpus, label=label)
        yield
