#!/usr/bin/env python3
"""Crash-safe, multi-process paper work queue for GPU evaluation.

The queue stores one result file per paper.  Claims use ``O_EXCL`` and results
use atomic replacement, so a killed worker can lose at most the in-flight
model call.  Existing monolithic/sharded result files can be imported before
workers start.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from eval_framework import atomic_write_json, looks_like_error_output
from evaluation_protocol import parse_canonical_pdf_output


def _key(paper_id: str) -> str:
    digest = hashlib.sha256(str(paper_id).encode("utf-8")).hexdigest()[:20]
    return digest


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True)
class Claim:
    paper_id: str
    path: Path
    owner_pid: int


class DurablePaperQueue:
    """A filesystem queue shared by independently started GPU workers."""

    def __init__(
        self,
        root: str | Path,
        source: dict[str, Any],
        *,
        claim_timeout_seconds: float = 6 * 60 * 60,
        max_paper_failures: int = 8,
        require_structured_output: bool = False,
        required_qa_fields: tuple[str, ...] = (),
        required_qa_field_values: dict[str, Any] | None = None,
        required_qa_field_values_by_paper: (
            dict[str, dict[str, Any]] | None
        ) = None,
        required_qa_field_values_by_qa: (
            dict[str, dict[str, dict[str, Any]]] | None
        ) = None,
        manifest_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.source = {
            str(paper_id): paper
            for paper_id, paper in source.items()
            if not str(paper_id).startswith("__") and isinstance(paper, dict)
        }
        self.claim_timeout_seconds = float(claim_timeout_seconds)
        self.max_paper_failures = int(max_paper_failures)
        self.require_structured_output = bool(require_structured_output)
        self.required_qa_fields = tuple(required_qa_fields)
        self.required_qa_field_values = dict(
            required_qa_field_values or {}
        )
        self.required_qa_field_values_by_paper = {
            str(paper_id): dict(values)
            for paper_id, values in (
                required_qa_field_values_by_paper or {}
            ).items()
        }
        self.required_qa_field_values_by_qa = {
            str(paper_id): {
                str(qa_id): dict(values)
                for qa_id, values in qa_values.items()
            }
            for paper_id, qa_values in (
                required_qa_field_values_by_qa or {}
            ).items()
        }
        self.manifest_metadata = dict(manifest_metadata or {})
        self.results_dir = self.root / "papers"
        self.claims_dir = self.root / "claims"
        self.failures_dir = self.root / "failures"
        for directory in (self.results_dir, self.claims_dir, self.failures_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

    def _write_manifest(self) -> None:
        source_sha256 = hashlib.sha256(
            json.dumps(
                self.source,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest = {
            "version": 2,
            "paper_count": len(self.source),
            "papers": {_key(paper_id): paper_id for paper_id in self.source},
            "source_sha256": source_sha256,
            "validation_contract": {
                "require_structured_output": self.require_structured_output,
                "required_qa_fields": list(self.required_qa_fields),
                "required_qa_field_values": self.required_qa_field_values,
                "required_qa_field_values_by_paper": (
                    self.required_qa_field_values_by_paper
                ),
                "required_qa_field_values_by_qa": (
                    self.required_qa_field_values_by_qa
                ),
            },
            "metadata": self.manifest_metadata,
        }
        path = self.root / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("papers") != manifest["papers"]:
                raise ValueError(
                    f"Queue manifest does not match current QA input: {path}"
                )
            if int(existing.get("version", 1)) >= 2:
                for field in (
                    "source_sha256",
                    "validation_contract",
                    "metadata",
                ):
                    existing_value = existing.get(field)
                    if field == "validation_contract" and isinstance(
                        existing_value, dict
                    ):
                        # Version-2 manifests written before exact-value
                        # contracts are equivalent only for callers that do
                        # not request them.  Strict new queues still mismatch.
                        existing_value = dict(existing_value)
                        existing_value.setdefault(
                            "required_qa_field_values", {}
                        )
                        existing_value.setdefault(
                            "required_qa_field_values_by_paper", {}
                        )
                        existing_value.setdefault(
                            "required_qa_field_values_by_qa", {}
                        )
                    if existing_value != manifest[field]:
                        raise ValueError(
                            f"Queue manifest {field} does not match current "
                            f"dataset/protocol: {path}"
                        )
            else:
                # Version 1 only locked paper IDs. Upgrade it so every future
                # process is content- and protocol-locked. Per-QA validators
                # below still reject stale answer-only artifacts.
                manifest["upgraded_from_version"] = 1
                atomic_write_json(path, manifest)
        else:
            atomic_write_json(path, manifest)

    def result_path(self, paper_id: str) -> Path:
        return self.results_dir / f"{_key(paper_id)}.json"

    def claim_path(self, paper_id: str) -> Path:
        return self.claims_dir / f"{_key(paper_id)}.json"

    def failure_path(self, paper_id: str) -> Path:
        return self.failures_dir / f"{_key(paper_id)}.json"

    def load_result(self, paper_id: str) -> dict[str, Any] | None:
        path = self.result_path(paper_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def expected_qa_ids(self, paper_id: str) -> set[str]:
        paper = self.source[paper_id]
        qa = paper.get("QA", {})
        return {str(qa_id) for qa_id in qa} if isinstance(qa, dict) else set()

    def validate_result(
        self, paper_id: str, result: dict[str, Any] | None = None
    ) -> tuple[bool, str]:
        result = self.load_result(paper_id) if result is None else result
        if not isinstance(result, dict):
            return False, "missing_or_invalid_json"
        if result.get("error"):
            return False, "paper_error"
        actual = result.get("QA")
        if not isinstance(actual, dict):
            return False, "missing_qa_object"
        expected_ids = self.expected_qa_ids(paper_id)
        if set(map(str, actual)) != expected_ids:
            return False, "incomplete_qa_ids"
        for qa_id in expected_ids:
            item = actual.get(qa_id)
            if not isinstance(item, dict):
                return False, f"invalid_qa:{qa_id}"
            output = item.get("model_output")
            if output is None or not str(output).strip():
                return False, f"empty_output:{qa_id}"
            if looks_like_error_output(str(output)):
                return False, f"error_output:{qa_id}"
            missing_fields = [
                field for field in self.required_qa_fields if field not in item
            ]
            if missing_fields:
                return False, f"missing_fields:{qa_id}:{','.join(missing_fields)}"
            for field, expected in self.required_qa_field_values.items():
                if item.get(field) != expected:
                    return False, f"field_value_mismatch:{qa_id}:{field}"
            paper_values = self.required_qa_field_values_by_paper.get(
                str(paper_id), {}
            )
            for field, expected in paper_values.items():
                if item.get(field) != expected:
                    return False, f"field_value_mismatch:{qa_id}:{field}"
            qa_values = (
                self.required_qa_field_values_by_qa
                .get(str(paper_id), {})
                .get(str(qa_id), {})
            )
            for field, expected in qa_values.items():
                if item.get(field) != expected:
                    return False, f"field_value_mismatch:{qa_id}:{field}"
            if self.require_structured_output:
                raw = str(output).strip()
                try:
                    parse_canonical_pdf_output(
                        raw, allowed_pages=item.get("shown_pdf_pages")
                    )
                except ValueError:
                    if item.get("generation_status") != (
                        "invalid_model_output_after_retries"
                    ):
                        return False, f"malformed_structured_output:{qa_id}"
                    exact_raw = str(item.get("raw_model_output", ""))
                    expected_hash = hashlib.sha256(
                        exact_raw.encode("utf-8")
                    ).hexdigest()
                    if (
                        output != exact_raw
                        or item.get("raw_model_output_sha256")
                        != expected_hash
                        or item.get("deterministic_normalizations") != []
                    ):
                        return False, f"invalid_output_audit_mismatch:{qa_id}"
        return True, "ok"

    def save_partial(self, paper_id: str, result: dict[str, Any]) -> None:
        atomic_write_json(self.result_path(paper_id), result)

    def complete_count(self) -> int:
        return sum(self.validate_result(paper_id)[0] for paper_id in self.source)

    def is_complete(self) -> bool:
        return self.complete_count() == len(self.source)

    def unfinished_count(self) -> int:
        return len(self.source) - self.complete_count()

    def _read_claim(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _claim_is_stale(self, path: Path) -> bool:
        claim = self._read_claim(path)
        if not claim:
            return True
        updated = float(claim.get("updated_at", 0))
        if time.time() - updated > self.claim_timeout_seconds:
            return True
        if claim.get("hostname") == socket.gethostname():
            try:
                return not _pid_alive(int(claim.get("pid", -1)))
            except (TypeError, ValueError):
                return True
        return False

    def _try_create_claim(self, paper_id: str, worker_id: str) -> Claim | None:
        path = self.claim_path(paper_id)
        payload = {
            "paper_id": paper_id,
            "worker_id": worker_id,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.time(),
            "updated_at": time.time(),
        }
        temporary = self.claims_dir / (
            f".{_key(paper_id)}.{os.getpid()}.{time.time_ns()}.claim"
        )
        atomic_write_json(temporary, payload)
        try:
            # A hard-link create is atomic and fails when ``path`` exists.
            # Unlike O_EXCL followed by json.dump, contenders can never see
            # a zero-length or partially written claim and misclassify it as
            # stale.
            os.link(temporary, path)
        except FileExistsError:
            if self._claim_is_stale(path):
                stale = path.with_suffix(
                    f".stale.{os.getpid()}.{time.time_ns()}.json"
                )
                try:
                    os.replace(path, stale)
                except FileNotFoundError:
                    pass
                else:
                    stale.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
                return self._try_create_claim(paper_id, worker_id)
            temporary.unlink(missing_ok=True)
            return None
        temporary.unlink(missing_ok=True)
        return Claim(paper_id=paper_id, path=path, owner_pid=os.getpid())

    def _failure_record(self, paper_id: str) -> dict[str, Any]:
        path = self.failure_path(paper_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"count": 0}
        return value if isinstance(value, dict) else {"count": 0}

    def reap_stale_claims(self) -> int:
        """Remove claims whose local owner died or whose lease expired."""
        reaped = 0
        for path in self.claims_dir.glob("*.json"):
            if not self._claim_is_stale(path):
                continue
            stale = path.with_suffix(
                f".stale.{os.getpid()}.{time.time_ns()}.json"
            )
            try:
                os.replace(path, stale)
            except FileNotFoundError:
                continue
            stale.unlink(missing_ok=True)
            reaped += 1
        return reaped

    def claim_next(self, worker_id: str) -> Claim | None:
        self.reap_stale_claims()
        # Rotate the deterministic order to avoid every worker contending on
        # the first path after a simultaneous start.
        ids = sorted(self.source)
        if ids:
            offset = int(hashlib.sha256(worker_id.encode()).hexdigest()[:8], 16)
            offset %= len(ids)
            ids = ids[offset:] + ids[:offset]
        for paper_id in ids:
            if self.validate_result(paper_id)[0]:
                continue
            failure = self._failure_record(paper_id)
            if int(failure.get("count", 0)) >= self.max_paper_failures:
                continue
            retry_after = float(failure.get("retry_after", 0))
            if time.time() < retry_after:
                continue
            claim = self._try_create_claim(paper_id, worker_id)
            if claim is not None:
                return claim
        return None

    def heartbeat(self, claim: Claim) -> None:
        current = self._read_claim(claim.path)
        if not current or int(current.get("pid", -1)) != claim.owner_pid:
            raise RuntimeError(f"Lost queue claim for {claim.paper_id}")
        current["updated_at"] = time.time()
        atomic_write_json(claim.path, current)

    def finish(self, claim: Claim, result: dict[str, Any]) -> None:
        self.save_partial(claim.paper_id, result)
        valid, reason = self.validate_result(claim.paper_id, result)
        if not valid:
            raise ValueError(
                f"Cannot finish invalid paper {claim.paper_id}: {reason}"
            )
        claim.path.unlink(missing_ok=True)
        self.failure_path(claim.paper_id).unlink(missing_ok=True)

    def fail(self, claim: Claim, error: BaseException) -> None:
        failure = self._failure_record(claim.paper_id)
        count = int(failure.get("count", 0)) + 1
        delay = min(300.0, 2.0 ** min(count, 8))
        atomic_write_json(
            self.failure_path(claim.paper_id),
            {
                "paper_id": claim.paper_id,
                "count": count,
                "last_error": f"{type(error).__name__}: {error}",
                "updated_at": time.time(),
                "retry_after": time.time() + delay,
                "quarantined": count >= self.max_paper_failures,
            },
        )
        claim.path.unlink(missing_ok=True)

    def release(self, claim: Claim) -> None:
        claim.path.unlink(missing_ok=True)

    def bootstrap(self, result_files: Iterable[str | Path]) -> int:
        """Import valid completed papers from older monolithic shard files."""
        imported = 0
        for result_file in result_files:
            path = Path(result_file)
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            for paper_id, result in data.items():
                paper_id = str(paper_id)
                if paper_id not in self.source or not isinstance(result, dict):
                    continue
                if self.validate_result(paper_id)[0]:
                    continue
                valid, _ = self.validate_result(paper_id, result)
                if valid:
                    self.save_partial(paper_id, result)
                    imported += 1
        return imported

    def merge(self, output: str | Path) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        invalid: dict[str, str] = {}
        for paper_id in sorted(self.source):
            result = self.load_result(paper_id)
            valid, reason = self.validate_result(paper_id, result)
            if valid and result is not None:
                merged[paper_id] = result
            else:
                invalid[paper_id] = reason
        if invalid:
            preview = ", ".join(
                f"{paper_id}:{reason}"
                for paper_id, reason in list(invalid.items())[:5]
            )
            raise RuntimeError(
                f"Queue has {len(invalid)} invalid/unfinished papers: {preview}"
            )
        atomic_write_json(output, merged)
        return merged

    def status(self) -> dict[str, Any]:
        failures = []
        failures_by_paper: dict[str, dict[str, Any]] = {}
        for path in self.failures_dir.glob("*.json"):
            try:
                failure = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            failures.append(failure)
            if isinstance(failure, dict) and failure.get("paper_id") is not None:
                failures_by_paper[str(failure["paper_id"])] = failure
        runnable_papers = 0
        retry_delayed_papers = 0
        quarantined_papers = 0
        active_papers = 0
        timestamp = time.time()
        for paper_id in self.source:
            if self.validate_result(paper_id)[0]:
                continue
            claim_path = self.claim_path(paper_id)
            if claim_path.exists() and not self._claim_is_stale(claim_path):
                active_papers += 1
                continue
            failure = failures_by_paper.get(paper_id, {})
            if int(failure.get("count", 0)) >= self.max_paper_failures:
                quarantined_papers += 1
            elif timestamp < float(failure.get("retry_after", 0)):
                retry_delayed_papers += 1
            else:
                runnable_papers += 1
        return {
            "total_papers": len(self.source),
            "completed_papers": self.complete_count(),
            "unfinished_papers": self.unfinished_count(),
            "active_claims": active_papers,
            "runnable_papers": runnable_papers,
            "retry_delayed_papers": retry_delayed_papers,
            "retryable_papers": runnable_papers + retry_delayed_papers,
            "quarantined_papers": quarantined_papers,
            "failures": failures,
        }
