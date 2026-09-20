#!/usr/bin/env python3
"""Check and optionally repair the local APIs, web app, SSH tunnel and HTTPS front."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STATE_PATH = ROOT / "data/web/task_queue/web_health.json"
LOCAL_ENDPOINTS = {
    "web": "http://127.0.0.1:3780/",
    "queue_api": "http://127.0.0.1:8765/api/health",
    "data_api": "http://127.0.0.1:8770/api/health",
}
SERVICE_BY_ENDPOINT = {
    "web": "pku-task-queue-web.service",
    "queue_api": "pku-task-queue-api.service",
    "data_api": "pku-data-manager-api.service",
}
PUBLIC_URL = "https://pku.chenzijian.com/"
VPS_TUNNEL_HOST = os.environ.get("PKU_VPS_TUNNEL_HOST", "45.59.102.64")
SSH = [
    "/usr/bin/ssh",
    "-i", "/data/czj/.ssh/pku-vps-tunnel",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ConnectTimeout=10",
    f"pku-tunnel@{VPS_TUNNEL_HOST}",
    "curl -sS --max-time 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:13780/",
]


def http_status(url: str, timeout: float = 15) -> int:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (OSError, urllib.error.URLError):
        return 0


def public_front_is_healthy(status: int) -> bool:
    # The public front must reach the application's own login/registration
    # page. A 401 means an obsolete Nginx Basic Auth prompt intercepted it.
    return status == 200


def tunnel_backend_status() -> int:
    result = subprocess.run(SSH, text=True, capture_output=True, timeout=20)
    try:
        return int(result.stdout.strip()) if result.returncode == 0 else 0
    except ValueError:
        return 0


def snapshot() -> dict[str, object]:
    local = {name: http_status(url) for name, url in LOCAL_ENDPOINTS.items()}
    tunnel = tunnel_backend_status()
    public = http_status(PUBLIC_URL)
    healthy = (
        all(status == 200 for status in local.values())
        and tunnel == 200
        and public_front_is_healthy(public)
    )
    return {
        "healthy": healthy,
        "checked_at": time.time(),
        "local": local,
        "tunnel_backend": tunnel,
        "public_front": public,
        "public_expected": "200 application login page",
    }


def restart(service: str) -> None:
    subprocess.run(["systemctl", "--user", "restart", service], check=True)


def repair(first: dict[str, object]) -> list[str]:
    restarted: list[str] = []
    local = first["local"]
    assert isinstance(local, dict)
    for name, status in local.items():
        if status != 200:
            service = SERVICE_BY_ENDPOINT[name]
            restart(service)
            restarted.append(service)
    if first["tunnel_backend"] != 200:
        restart("pku-web-tunnel.service")
        restarted.append("pku-web-tunnel.service")
    if restarted:
        time.sleep(5)
    return restarted


def write_state(value: dict[str, object]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_name(f".{STATE_PATH.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    first = snapshot()
    restarted = repair(first) if args.repair and not first["healthy"] else []
    final = snapshot() if restarted else first
    final["restarted_services"] = restarted
    write_state(final)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))
    if not final["healthy"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
