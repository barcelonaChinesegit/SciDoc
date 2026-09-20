#!/usr/bin/env python3
"""Loopback-only API for authenticated, reversible QA review."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import mimetypes
import os
import secrets
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import socks

from data_review_manager import (
    DEFAULT_DB,
    SESSION_TTL_SECONDS,
    STATE_DIR,
    DatasetRegistry,
    ReviewStore,
    atomic_write,
)
SESSION_COOKIE = "pku_session"
INTERNAL_TOKEN_FILE = STATE_DIR / "web_internal_token"
LOGGER = logging.getLogger("pku_qa.review.email")


def smtp_proxy_options(proxy_url: str) -> dict[str, object]:
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"socks5", "socks5h"} or not parsed.hostname:
        raise ValueError("PKU_SMTP_PROXY must be a socks5:// or socks5h:// URL")
    return {
        "proxy_type": socks.SOCKS5,
        "proxy_addr": parsed.hostname,
        "proxy_port": parsed.port or 1080,
        "proxy_rdns": parsed.scheme == "socks5h",
        "proxy_username": parsed.username,
        "proxy_password": parsed.password,
    }


class SocksSMTPSSL(smtplib.SMTP_SSL):
    def __init__(self, *args, proxy_url: str, **kwargs) -> None:
        self.proxy_options = smtp_proxy_options(proxy_url)
        super().__init__(*args, **kwargs)

    def _get_socket(self, host: str, port: int, timeout: float):
        raw_socket = socks.create_connection(
            (host, port),
            timeout=timeout,
            source_address=self.source_address,
            **self.proxy_options,
        )
        return self.context.wrap_socket(raw_socket, server_hostname=host)


class AuthenticationError(PermissionError):
    pass


class AuthorizationError(PermissionError):
    pass


def ensure_internal_token(path: Path = INTERNAL_TOKEN_FILE) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(48)
    atomic_write(path, (token + "\n").encode("ascii"))
    os.chmod(path, 0o600)
    return token


class DataApiHandler(BaseHTTPRequestHandler):
    server_version = "PkuDataReview/2.0"

    @property
    def store(self) -> ReviewStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def registry(self) -> DatasetRegistry:
        return self.store.registry

    @property
    def internal_token(self) -> str:
        return getattr(self.server, "internal_token", "")

    def end_headers(self) -> None:
        self.send_header(
            "Cache-Control",
            getattr(self, "response_cache_control", "no-store"),
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def json_response(
        self,
        value: object,
        status: int = 200,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        for name, content in (headers or {}).items():
            self.send_header(name, content)
        self.end_headers()
        self.wfile.write(raw)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def route(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        return parts, parse_qs(parsed.query)

    def request_metadata(self) -> dict[str, str]:
        forwarded = self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        return {
            "remote_addr": forwarded or self.client_address[0],
            "user_agent": self.headers.get("User-Agent", "")[:500],
        }

    def session_token(self) -> str:
        cookie = SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else ""

    def require_user(self) -> dict:
        token = self.session_token()
        if not token:
            raise AuthenticationError("Authentication required")
        try:
            return self.store.session_user(token)
        except PermissionError as exc:
            raise AuthenticationError(str(exc)) from exc

    def require_admin(self) -> dict:
        user = self.require_user()
        if not self.store.has_role(user, "admin"):
            raise AuthorizationError("Administrator permission required")
        return user

    @staticmethod
    def email_delivery_available() -> bool:
        return bool(
            os.environ.get("PKU_SMTP_USERNAME", "").strip()
            and os.environ.get("PKU_SMTP_APP_PASSWORD", "").replace(" ", "").strip()
        )

    @staticmethod
    def send_verification_email(email: str, code: str, purpose: str) -> None:
        username = os.environ.get("PKU_SMTP_USERNAME", "").strip()
        password = os.environ.get("PKU_SMTP_APP_PASSWORD", "").replace(" ", "").strip()
        if not username or not password:
            raise RuntimeError("Email verification is not configured")
        host = os.environ.get("PKU_SMTP_HOST", "smtp.gmail.com").strip()
        port = int(os.environ.get("PKU_SMTP_PORT", "465"))
        message = EmailMessage()
        message["From"] = f"PKU QA Review <{username}>"
        message["To"] = email
        message["Subject"] = "PKU QA 人工校验邮箱验证码"
        action = "注册账户" if purpose == "register" else "换绑邮箱"
        message.set_content(
            f"你正在为 PKU QA 人工校验台{action}。\n\n"
            f"验证码：{code}\n\n"
            "验证码 10 分钟内有效。若非本人操作，请忽略此邮件。"
        )
        context = ssl.create_default_context()
        proxy_url = os.environ.get("PKU_SMTP_PROXY", "").strip()
        smtp_class = SocksSMTPSSL if proxy_url else smtplib.SMTP_SSL
        options = {"proxy_url": proxy_url} if proxy_url else {}
        with smtp_class(host, port, timeout=20, context=context, **options) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)

    def session_cookie(self, token: str, *, clear: bool = False) -> str:
        max_age = 0 if clear else SESSION_TTL_SECONDS
        value = "" if clear else token
        attributes = [
            f"{SESSION_COOKIE}={value}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            f"Max-Age={max_age}",
        ]
        if self.headers.get("X-Forwarded-Proto", "").lower() == "https":
            attributes.append("Secure")
        return "; ".join(attributes)

    def handle_exception(self, exc: Exception) -> None:
        if isinstance(exc, AuthenticationError):
            status = 401
        elif isinstance(exc, AuthorizationError):
            status = 403
        elif isinstance(exc, KeyError):
            status = 404
        elif isinstance(exc, RuntimeError):
            status = 409
        elif isinstance(exc, PermissionError):
            status = 403
        else:
            status = 400
        self.json_response({"error": f"{type(exc).__name__}: {exc}"}, status)

    def do_GET(self) -> None:
        try:
            parts, query = self.route()
            if parts == ["api", "health"]:
                self.json_response({"status": "ok", "pid": os.getpid()})
                return
            if parts == ["api", "auth", "me"]:
                self.json_response({"user": self.require_user()})
                return
            if parts == ["api", "auth", "email-config"]:
                self.json_response({"available": self.email_delivery_available()})
                return
            user = self.require_user()
            if parts == ["api", "datasets"]:
                collection = query.get("collection", [None])[0] or None
                if collection not in {None, "final_2200", "other"}:
                    raise ValueError("Invalid dataset collection")
                rows = self.registry.list(collection_id=collection)
                summary = query.get("detail", [""])[0] == "summary"
                self.json_response({
                    "datasets": [
                        row.to_summary_dict() if summary else row.to_dict()
                        for row in rows
                    ]
                })
            elif len(parts) == 3 and parts[:2] == ["api", "datasets"]:
                dataset_id = parts[2]
                info = self.registry.inspect(
                    self.registry.resolve(dataset_id), dataset_id
                )
                if info is None:
                    raise KeyError(f"Dataset is not QA-shaped: {dataset_id}")
                self.json_response({"dataset": info.to_dict()})
            elif (
                len(parts) == 4
                and parts[:2] == ["api", "datasets"]
                and parts[3] == "item"
            ):
                index = int(query.get("index", ["0"])[0])
                payload = self.store.item(parts[2], index, actor=user)
                paper_id = payload["item"]["paper_id"]
                try:
                    pdf = self.registry.pdf_for(parts[2], paper_id)
                    payload["item"]["pdf_available"] = True
                    payload["item"]["pdf_size_bytes"] = pdf.stat().st_size
                except KeyError:
                    payload["item"]["pdf_available"] = False
                self.json_response(payload)
            elif parts == ["api", "events"]:
                dataset_id = query.get("dataset", [None])[0]
                limit = int(query.get("limit", ["100"])[0])
                self.json_response(
                    {"events": self.store.events(dataset_id, limit, actor=user)}
                )
            elif parts == ["api", "pdf"]:
                dataset_id = query.get("dataset", [""])[0]
                paper_id = query.get("paper", [""])[0]
                self.serve_pdf(self.registry.pdf_for(dataset_id, paper_id))
            elif parts == ["api", "users"]:
                self.require_admin()
                self.json_response({"users": self.store.list_users()})
            elif parts == ["api", "assignments"]:
                scope = query.get("scope", [""])[0]
                current_user = self.require_user()
                if scope == "mine":
                    assignments = self.store.list_assignments(str(current_user["user_id"]))
                else:
                    self.require_admin()
                    assignments = self.store.list_assignments()
                self.json_response(
                    {"assignments": assignments}
                )
            elif parts == ["api", "reviewer-progress"]:
                self.require_admin()
                self.json_response({"progress": self.store.reviewer_progress()})
            elif parts == ["api", "audit"]:
                self.require_admin()
                limit = int(query.get("limit", ["200"])[0])
                self.json_response({"events": self.store.audit_events(limit)})
            else:
                self.json_response({"error": "not found"}, 404)
        except Exception as exc:
            self.handle_exception(exc)

    def do_POST(self) -> None:
        try:
            parts, _ = self.route()
            body = self.read_json()
            metadata = self.request_metadata()
            if parts == ["api", "auth", "login"]:
                username = str(body.get("username", ""))
                try:
                    user = self.store.authenticate_password(
                        username, str(body.get("password", ""))
                    )
                except PermissionError:
                    self.store.audit(
                        "auth.login_failed",
                        "user",
                        username,
                        details={"username": username},
                        **metadata,
                    )
                    raise AuthenticationError("Invalid username or password")
                token = self.store.create_session(user, **metadata)
                self.store.audit(
                    "auth.login",
                    "user",
                    user["user_id"],
                    actor=user,
                    details={"method": "password"},
                    **metadata,
                )
                self.json_response(
                    {"user": user},
                    headers={"Set-Cookie": self.session_cookie(token)},
                )
                return
            if parts == ["api", "auth", "email-code"]:
                if not self.email_delivery_available():
                    raise RuntimeError("Email verification is not configured")
                purpose = str(body.get("purpose", "register"))
                email, code = self.store.prepare_email_verification(
                    str(body.get("email", "")),
                    purpose,
                    remote_addr=metadata["remote_addr"],
                )
                try:
                    self.send_verification_email(email, code, purpose)
                except Exception as exc:
                    self.store.cancel_email_verification(email)
                    LOGGER.error(
                        "verification email delivery failed via %s: %s",
                        "SOCKS5 proxy" if os.environ.get("PKU_SMTP_PROXY") else "direct SMTP",
                        type(exc).__name__,
                    )
                    raise RuntimeError(
                        "验证码邮件发送失败，请稍后重试；若持续失败请联系管理员检查 VPN"
                    ) from exc
                self.store.audit(
                    "auth.email_code_sent",
                    "email",
                    hashlib.sha256(email.encode("utf-8")).hexdigest()[:16],
                    details={"purpose": purpose},
                    **metadata,
                )
                self.json_response({"ok": True, "expires_in": 600})
                return
            if parts == ["api", "auth", "register"]:
                if not self.email_delivery_available():
                    raise RuntimeError("Email verification is not configured")
                user = self.store.register_user(
                    str(body.get("username", "")),
                    str(body.get("password", "")),
                    display_name=str(body.get("display_name", "")),
                    email=str(body.get("email", "")),
                    email_code=str(body.get("email_code", "")),
                    **metadata,
                )
                token = self.store.create_session(user, **metadata)
                self.json_response(
                    {"user": user},
                    201,
                    headers={"Set-Cookie": self.session_cookie(token)},
                )
                return
            if parts == ["api", "auth", "perimeter"]:
                supplied = self.headers.get("X-PKU-Internal-Token", "")
                if not self.internal_token or not secrets.compare_digest(
                    supplied, self.internal_token
                ):
                    raise AuthorizationError("Trusted proxy token required")
                user = self.store.perimeter_user(
                    self.headers.get("X-PKU-Perimeter-User", "")
                )
                token = self.store.create_session(user, **metadata)
                self.store.audit(
                    "auth.login",
                    "user",
                    user["user_id"],
                    actor=user,
                    details={"method": "nginx_basic_auth"},
                    **metadata,
                )
                self.json_response(
                    {"user": user},
                    headers={"Set-Cookie": self.session_cookie(token)},
                )
                return
            if parts == ["api", "auth", "logout"]:
                user = self.require_user()
                token = self.session_token()
                self.store.delete_session(token)
                self.store.audit(
                    "auth.logout", "user", user["user_id"], actor=user, **metadata
                )
                self.json_response(
                    {"ok": True},
                    headers={"Set-Cookie": self.session_cookie("", clear=True)},
                )
                return
            user = self.require_user()
            if parts == ["api", "profile"]:
                self.json_response(
                    self.store.update_profile(
                        user["user_id"],
                        username=body.get("username"),
                        display_name=body.get("display_name"),
                        email=body.get("email"),
                        email_code=body.get("email_code"),
                        actor=user,
                    )
                )
                return
            if parts == ["api", "profile", "password"]:
                self.json_response(
                    self.store.change_password(
                        user["user_id"],
                        str(body.get("current_password", "")),
                        str(body.get("new_password", "")),
                        session_token=self.session_token(),
                        actor=user,
                    )
                )
                return
            if (
                len(parts) == 4
                and parts[:2] == ["api", "datasets"]
                and parts[3] == "review"
            ):
                result = self.store.review(
                    parts[2],
                    str(body["paper_id"]),
                    str(body["qa_id"]),
                    str(body["action"]),
                    expected_sha256=body.get("expected_sha256"),
                    note=body.get("note"),
                    actor=user,
                )
                self.json_response(result)
            elif (
                len(parts) == 4
                and parts[:2] == ["api", "datasets"]
                and parts[3] == "edit"
            ):
                self.json_response(
                    self.store.edit(
                        parts[2],
                        str(body["paper_id"]),
                        str(body["qa_id"]),
                        question=str(body["question"]),
                        answer=str(body["answer"]),
                        evidence_pages=body["evidence_pages"],
                        evidence_page_changes=body.get("evidence_page_changes"),
                        expected_sha256=body.get("expected_sha256"),
                        actor=user,
                    )
                )
            elif (
                len(parts) == 4
                and parts[:2] == ["api", "datasets"]
                and parts[3] == "undo"
            ):
                self.json_response(
                    self.store.undo(parts[2], body.get("event_id"), actor=user)
                )
            elif parts == ["api", "users"]:
                admin = self.require_admin()
                created = self.store.create_user(
                    str(body["username"]),
                    str(body["password"]),
                    display_name=str(body.get("display_name", "")),
                    role=str(body.get("role", "reviewer")),
                    roles=body.get("roles"),
                    email=body.get("email"),
                    actor=admin,
                )
                self.json_response(created, 201)
            elif parts == ["api", "assignments"]:
                admin = self.require_admin()
                created = self.store.create_assignment(
                    str(body["user_id"]),
                    str(body["dataset_id"]),
                    int(body["start_index"]),
                    int(body["end_index"]),
                    actor=admin,
                )
                self.json_response(created, 201)
            elif (
                len(parts) == 4
                and parts[:2] == ["api", "users"]
                and parts[3] == "password"
            ):
                admin = self.require_admin()
                self.json_response(
                    self.store.set_password(
                        parts[2], str(body["password"]), actor=admin
                    )
                )
            else:
                self.json_response({"error": "not found"}, 404)
        except Exception as exc:
            self.handle_exception(exc)

    def do_PATCH(self) -> None:
        try:
            parts, _ = self.route()
            if len(parts) != 3:
                self.json_response({"error": "not found"}, 404)
                return
            admin = self.require_admin()
            body = self.read_json()
            if parts[:2] == ["api", "users"]:
                self.json_response(
                    self.store.update_user(
                        parts[2],
                        display_name=body.get("display_name"),
                        role=body.get("role"),
                        roles=body.get("roles"),
                        disabled=body.get("disabled"),
                        actor=admin,
                    )
                )
            elif parts[:2] == ["api", "assignments"]:
                self.json_response(
                    self.store.update_assignment(
                        parts[2],
                        user_id=str(body["user_id"]),
                        dataset_id=str(body["dataset_id"]),
                        start_index=int(body["start_index"]),
                        end_index=int(body["end_index"]),
                        actor=admin,
                    )
                )
            else:
                self.json_response({"error": "not found"}, 404)
        except Exception as exc:
            self.handle_exception(exc)

    def do_DELETE(self) -> None:
        try:
            parts, _ = self.route()
            if len(parts) != 3 or parts[:2] != ["api", "assignments"]:
                self.json_response({"error": "not found"}, 404)
                return
            admin = self.require_admin()
            self.store.delete_assignment(parts[2], actor=admin)
            self.json_response({"ok": True})
        except Exception as exc:
            self.handle_exception(exc)

    def serve_pdf(self, path: Path) -> None:
        stat = path.stat()
        size = stat.st_size
        etag = f'"{stat.st_size:x}-{stat.st_mtime_ns:x}"'
        self.response_cache_control = "private, max-age=86400, must-revalidate"
        range_header = self.headers.get("Range")
        if self.headers.get("If-None-Match") == etag and not range_header:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
            self.end_headers()
            return
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if range_header:
            if not range_header.startswith("bytes="):
                raise ValueError("Invalid Range header")
            raw_start, separator, raw_end = range_header[6:].partition("-")
            if not separator:
                raise ValueError("Invalid Range header")
            start = int(raw_start) if raw_start else 0
            end = int(raw_end) if raw_end else size - 1
            if start < 0 or end < start or end >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "application/pdf"
        )
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--internal-token-file", type=Path, default=INTERNAL_TOKEN_FILE)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Data review API must remain loopback-only")
    server = ThreadingHTTPServer((args.host, args.port), DataApiHandler)
    store = ReviewStore(args.db)
    # Warm only the four formal final-review components.  Historical/process
    # JSON files are scanned only after an explicit collection=other request.
    store.registry.list(collection_id="final_2200")
    server.store = store  # type: ignore[attr-defined]
    server.internal_token = ensure_internal_token(  # type: ignore[attr-defined]
        args.internal_token_file
    )
    print(f"Data review API listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
