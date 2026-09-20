from __future__ import annotations

import json
import tempfile
import threading
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from data_review_manager import (
    DATASET_CATALOG,
    FINAL_2200_DATASET_IDS,
    PROJECT_DATASETS,
    DatasetRegistry,
    ReviewStore,
    dataset_profile,
)
from data_manager_api import DataApiHandler, smtp_proxy_options


def fixture_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "p1": {
                    "paper": "p1",
                    "QA": {
                        "Q1": {
                            "question": "Question?",
                            "answer": "Answer.",
                            "evidence_pages": [3],
                            "oracle_pages": [3],
                            "evidence_items": [
                                {
                                    "physical_pdf_page": 3,
                                    "source_doc_number": 2,
                                    "supported_fact": "Original fact",
                                }
                            ],
                            "evidence_source_docs": [2],
                            "source_evidence_pages": [3],
                            "construction_key_evidence_pages": [3],
                            "intermediate_facts": [
                                {"merged_page": 3, "fact": "Original fact"}
                            ],
                            "review_evidence_ledger": [
                                {"merged_page": 3, "supported_fact": "Original fact"}
                            ],
                            "evidence_provenance": {
                                "source_qa_contributions": [
                                    {"source_qa_id": "Q0", "evidence_pages": [3]}
                                ]
                            },
                            "evidence_span": 0,
                            "evidence_span_ratio": 0,
                        },
                        "Q2": {
                            "question": "Another?",
                            "answer": "Another answer.",
                            "evidence_pages": [5],
                        },
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_smtp_proxy_options_require_socks_and_enable_remote_dns() -> None:
    options = smtp_proxy_options("socks5h://proxy-user:proxy-pass@127.0.0.1:10808")
    assert options["proxy_addr"] == "127.0.0.1"
    assert options["proxy_port"] == 10808
    assert options["proxy_rdns"] is True
    assert options["proxy_username"] == "proxy-user"
    assert options["proxy_password"] == "proxy-pass"
    with pytest.raises(ValueError, match="socks5"):
        smtp_proxy_options("http://127.0.0.1:8080")


def test_keep_delete_and_undo_are_durable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        dataset_path = qa_dir / "qa_test.json"
        fixture_dataset(dataset_path)
        registry = DatasetRegistry(qa_dir, pdf_dir)
        store = ReviewStore(
            root / "review.sqlite3",
            registry,
            root / "snapshots",
        )

        current = store.item("qa_test.json", 0)
        kept = store.review(
            "qa_test.json",
            "p1",
            "Q1",
            "keep",
            expected_sha256=current["dataset_sha256"],
        )
        assert kept["status"] == "kept"
        assert store.item("qa_test.json", 0)["kept"] == 1

        current = store.item("qa_test.json", 0)
        deleted = store.review(
            "qa_test.json",
            "p1",
            "Q1",
            "delete",
            expected_sha256=current["dataset_sha256"],
        )
        assert deleted["status"] == "deleted"
        assert "Q1" not in json.loads(dataset_path.read_text())["p1"]["QA"]
        store.undo("qa_test.json", deleted["event_id"])
        assert "Q1" in json.loads(dataset_path.read_text())["p1"]["QA"]
        assert len(store.events("qa_test.json")) == 2


def test_registry_catalog_cache_is_explicitly_invalidated_after_mutation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        dataset_path = qa_dir / "qa_test.json"
        fixture_dataset(dataset_path)
        registry = DatasetRegistry(qa_dir, pdf_dir)

        assert registry.list()[0].qas == 2
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset["p1"]["QA"]["Q3"] = {
            "question": "Third?", "answer": "Third.", "evidence_pages": [1]
        }
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

        # The catalog remains stable during its short TTL until a known write
        # invalidates it; review mutations call this method immediately.
        assert registry.list()[0].qas == 2
        registry.invalidate("qa_test.json")
        assert registry.list()[0].qas == 3


def test_stale_browser_hash_cannot_delete() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        dataset_path = qa_dir / "qa_test.json"
        fixture_dataset(dataset_path)
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        with pytest.raises(RuntimeError, match="reload"):
            store.review(
                "qa_test.json",
                "p1",
                "Q1",
                "delete",
                expected_sha256="stale",
            )
        assert "Q1" in json.loads(dataset_path.read_text())["p1"]["QA"]


def test_edit_updates_short_answer_fields_and_is_reversible() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        dataset_path = qa_dir / "qa_test.json"
        fixture_dataset(dataset_path)
        dataset = json.loads(dataset_path.read_text())
        dataset["p1"]["QA"]["Q1"]["options"] = [
            {"id": "A", "text": "stale option"}
        ]
        dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )

        current = store.item("qa_test.json", 0)
        edited = store.edit(
            "qa_test.json",
            "p1",
            "Q1",
            question="Corrected question?",
            answer="Corrected answer.",
            evidence_pages=[7, 4, 4],
            evidence_page_changes=[{"from": 3, "to": 4}],
            expected_sha256=current["dataset_sha256"],
        )
        assert edited["status"] == "edited"
        payload = json.loads(dataset_path.read_text())
        qa = payload["p1"]["QA"]["Q1"]
        assert qa["question"] == "Corrected question?"
        assert qa["answer"] == "Corrected answer."
        assert qa["evidence_pages"] == [4, 7]
        assert qa["oracle_pages"] == [4, 7]
        assert qa["evidence_items"][0]["physical_pdf_page"] == 4
        assert qa["evidence_source_docs"] == [2]
        assert qa["source_evidence_pages"] == [4, 7]
        assert qa["construction_key_evidence_pages"] == [4]
        assert qa["intermediate_facts"][0]["merged_page"] == 4
        assert qa["review_evidence_ledger"][0]["merged_page"] == 4
        assert qa["evidence_provenance"]["source_qa_contributions"][0][
            "evidence_pages"
        ] == [4]
        assert "options" not in qa
        assert store.item("qa_test.json", 0)["edited"] == 1
        store.undo("qa_test.json", edited["event_id"])
        restored = json.loads(dataset_path.read_text())["p1"]["QA"]["Q1"]
        assert restored["question"] == "Question?"
        assert restored["evidence_pages"] == [3]
        assert restored["options"][0]["id"] == "A"


def test_edit_and_undo_refresh_final_collection_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        dataset_path = qa_dir / "qa_test.json"
        fixture_dataset(dataset_path)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "collection_id": "final_2200",
                    "qa_count": 2,
                    "components": [
                        {
                            "dataset_id": "qa_test.json",
                            "qa_count": 2,
                            "paper_count": 1,
                            "answerable": 2,
                            "unanswerable": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
            manifest_path,
        )
        current = store.item("qa_test.json", 0)
        edited = store.edit(
            "qa_test.json",
            "p1",
            "Q1",
            question="Corrected question?",
            answer="Answer.",
            evidence_pages=[4],
            evidence_page_changes=[{"from": 3, "to": 4}],
            expected_sha256=current["dataset_sha256"],
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["qa_format"] == "short_answer_only"
        assert manifest["qa_count"] == 2
        assert manifest["components"][0]["sha256"] == edited["after_sha256"]

        store.undo("qa_test.json", edited["event_id"])
        restored_manifest = json.loads(manifest_path.read_text())
        assert restored_manifest["components"][0]["sha256"] == current[
            "dataset_sha256"
        ]


def test_edit_enforces_answer_and_evidence_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        fixture_dataset(qa_dir / "qa_test.json")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        current = store.item("qa_test.json", 0)
        with pytest.raises(ValueError, match="必须为空"):
            store.edit(
                "qa_test.json",
                "p1",
                "Q1",
                question="Question?",
                answer="Unanswerable",
                evidence_pages=[1],
                expected_sha256=current["dataset_sha256"],
            )


def test_dataset_profile_ignores_malformed_evidence_page_objects() -> None:
    profile = dataset_profile(
        "qa_unknown.json",
        {
            "p1": {
                "QA": {
                    "Q1": {
                        "question": "Q?",
                        "answer": "A",
                        "evidence_pages": [{"page": 3}, "4", 5, 0],
                    }
                }
            }
        },
    )
    assert profile["statistics"]["with_evidence_pages"] == 1
    assert profile["statistics"]["multi_page_evidence"] == 1
    fields = {row["name"]: row for row in profile["qa_fields"]}
    assert fields["evidence_pages"]["value_types"] == [
        "array<integer|object|string>"
    ]


def test_final_cross_pdf_release_uses_materialized_bundle_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        bundle_dir = pdf_dir / "cross_pdf_hard_expansion"
        bundle_dir.mkdir(parents=True)
        expected = bundle_dir / "hxf_0001.pdf"
        expected.write_bytes(b"%PDF-release")

        resolved = DatasetRegistry(qa_dir, pdf_dir).pdf_for(
            "cross_pdf_qa.json",
            "hxf_0001",
        )

        assert resolved == expected.resolve()


def test_historical_reasoning_sources_are_labeled_as_lineage() -> None:
    refreshed = "rel__reasoning__refreshed__batch01__n100__v1.json"
    incremental = "rel__reasoning__incremental__batch02__n100__v1.json"

    assert PROJECT_DATASETS[refreshed].name == refreshed
    assert PROJECT_DATASETS[incremental].name == incremental
    assert DATASET_CATALOG[refreshed]["status"] == "历史构建来源"
    assert DATASET_CATALOG[incremental]["status"] == "历史构建来源"


def test_final_2200_files_share_one_review_collection() -> None:
    expected_order = (
        "ordinary_qa.json",
        "unanswerable_qa.json",
        "reasoning_qa.json",
        "cross_pdf_qa.json",
    )
    expected = set(expected_order)

    assert FINAL_2200_DATASET_IDS == expected_order
    assert expected == {
        dataset_id
        for dataset_id in PROJECT_DATASETS
        if DATASET_CATALOG.get(dataset_id, {}).get("collection_id") == "final_2200"
    }
    assert {
        DATASET_CATALOG[dataset_id]["collection_id"]
        for dataset_id in expected
    } == {"final_2200"}
    assert [
        row.id for row in DatasetRegistry().list(collection_id="final_2200")
    ] == list(expected_order)
    assert dataset_profile("qa_unknown.json", {"p": {"QA": {"q": {
        "question": "Q?", "answer": "A", "evidence_pages": [1]
    }}}})["collection_id"] == "other"


def test_categorized_final_release_views_are_read_only() -> None:
    ordinary_view = "rel__single_pdf__ordinary__batch01__n1000.json"
    profile = dataset_profile(
        ordinary_view,
        {"p": {"QA": {"q": {"question": "Q?", "answer": "A"}}}},
    )

    assert profile["collection_id"] == "other"
    assert profile["status"] == "最终集只读镜像"
    assert profile["reviewable"] is False


def test_registry_discovers_nested_qa_files_with_safe_relative_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        nested = qa_dir / "output" / "run1"
        nested.mkdir(parents=True)
        path = nested / "responses.json"
        fixture_dataset(path)
        rows = DatasetRegistry(qa_dir, root / "pdfs").list()
        assert [row.id for row in rows] == ["output/run1/responses.json"]
        profile = rows[0].profile
        assert profile["relative_path"] == "output/run1/responses.json"
        assert profile["source_summary"]
        assert profile["intended_use"]
        assert profile["reviewable"] is False
        with pytest.raises(ValueError):
            DatasetRegistry(qa_dir, root / "pdfs").resolve("../responses.json")


def test_data_api_lists_reviews_undoes_and_serves_pdf_ranges() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        (pdf_dir / "test").mkdir(parents=True)
        dataset_path = qa_dir / "qa_test.json"
        fixture_dataset(dataset_path)
        pdf_bytes = b"%PDF-test-content-for-range"
        (pdf_dir / "test/p1.pdf").write_bytes(pdf_bytes)
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), DataApiHandler)
        server.store = store
        server.internal_token = "test-internal-token"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )

        def call(path: str, method: str = "GET", body=None, headers=None):
            raw = None if body is None else json.dumps(body).encode()
            request = urllib.request.Request(
                base + path,
                data=raw,
                method=method,
                headers={"Content-Type": "application/json", **(headers or {})},
            )
            with opener.open(request, timeout=5) as response:
                return json.loads(response.read())

        try:
            assert call("/api/health")["status"] == "ok"
            with pytest.raises(urllib.error.HTTPError) as unauthorized:
                call("/api/datasets")
            assert unauthorized.value.code == 401
            signed_in = call(
                "/api/auth/perimeter",
                "POST",
                {},
                {
                    "X-PKU-Internal-Token": "test-internal-token",
                    "X-PKU-Perimeter-User": "czj-web",
                },
            )
            assert signed_in["user"]["role"] == "admin"
            dual_admin = call(
                f"/api/users/{signed_in['user']['user_id']}",
                "PATCH",
                {"roles": ["admin", "reviewer"]},
            )
            assert dual_admin["roles"] == ["admin", "reviewer"]
            created_user = call(
                "/api/users",
                "POST",
                {
                    "username": "classmate-one",
                    "display_name": "Classmate One",
                    "password": "classmate-pass-123",
                    "role": "reviewer",
                },
            )
            assert created_user["role"] == "reviewer"

            reviewer_opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(CookieJar())
            )

            def reviewer_call(path: str, method: str = "GET", body=None):
                raw = None if body is None else json.dumps(body).encode()
                request = urllib.request.Request(
                    base + path,
                    data=raw,
                    method=method,
                    headers={"Content-Type": "application/json"},
                )
                with reviewer_opener.open(request, timeout=5) as response:
                    return json.loads(response.read())

            reviewer_login = reviewer_call(
                "/api/auth/login",
                "POST",
                {
                    "username": "classmate-one",
                    "password": "classmate-pass-123",
                },
            )
            assert reviewer_login["user"]["role"] == "reviewer"
            with pytest.raises(urllib.error.HTTPError) as forbidden:
                reviewer_call("/api/users")
            assert forbidden.value.code == 403
            assignment = call(
                "/api/assignments",
                "POST",
                {
                    "user_id": created_user["user_id"],
                    "dataset_id": "qa_test.json",
                    "start_index": 2,
                    "end_index": 2,
                },
            )
            assert assignment["assigned_count"] == 1
            admin_assignment = call(
                "/api/assignments",
                "POST",
                {
                    "user_id": signed_in["user"]["user_id"],
                    "dataset_id": "qa_test.json",
                    "start_index": 1,
                    "end_index": 1,
                },
            )
            assert admin_assignment["assigned_count"] == 1
            assert {
                row["username"]
                for row in call("/api/assignments")["assignments"]
            } == {"classmate-one", "czj-web"}
            reviewer_assignments = reviewer_call(
                "/api/assignments?scope=mine"
            )["assignments"]
            assert [row["username"] for row in reviewer_assignments] == [
                "classmate-one"
            ]
            assert reviewer_assignments[0]["reviewed_count"] == 0
            progress_rows = call("/api/reviewer-progress")["progress"]
            progress_by_user = {row["username"]: row for row in progress_rows}
            assert progress_by_user["classmate-one"]["assigned_count"] == 1
            assert progress_by_user["classmate-one"]["reviewed_count"] == 0
            assert progress_by_user["classmate-one"]["last_reviewed_at"] is None
            assigned_item = reviewer_call(
                "/api/datasets/qa_test.json/item?index=1"
            )
            assert assigned_item["can_modify"] is True
            assert assigned_item["assignment"]["start_index"] == 2
            read_only_item = reviewer_call(
                "/api/datasets/qa_test.json/item?index=0"
            )
            assert read_only_item["can_modify"] is False
            with pytest.raises(urllib.error.HTTPError) as admin_unassigned:
                call(
                    "/api/datasets/qa_test.json/review",
                    "POST",
                    {
                        "paper_id": "p1",
                        "qa_id": "Q2",
                        "action": "keep",
                        "expected_sha256": assigned_item["dataset_sha256"],
                    },
                )
            assert admin_unassigned.value.code == 403
            reviewer_call(
                "/api/datasets/qa_test.json/review",
                "POST",
                {
                    "paper_id": "p1",
                    "qa_id": "Q2",
                    "action": "keep",
                    "expected_sha256": assigned_item["dataset_sha256"],
                },
            )
            edited_item = reviewer_call(
                "/api/datasets/qa_test.json/edit",
                "POST",
                {
                    "paper_id": "p1",
                    "qa_id": "Q2",
                    "question": "Corrected second question?",
                    "answer": "Corrected second answer.",
                    "evidence_pages": [6],
                    "evidence_page_changes": [{"from": 5, "to": 6}],
                    "expected_sha256": assigned_item["dataset_sha256"],
                },
            )
            assert edited_item["status"] == "edited"
            progress = {
                row["username"]: row
                for row in call("/api/reviewer-progress")["progress"]
            }["classmate-one"]
            assert progress["reviewed_count"] == 1
            assert progress["kept_count"] == 0
            assert progress["edited_count"] == 1
            assert progress["last_reviewed_at"] is not None
            with pytest.raises(urllib.error.HTTPError) as unassigned:
                reviewer_call(
                    "/api/datasets/qa_test.json/review",
                    "POST",
                    {
                        "paper_id": "p1",
                        "qa_id": "Q1",
                        "action": "keep",
                        "expected_sha256": read_only_item["dataset_sha256"],
                    },
                )
            assert unassigned.value.code == 403
            datasets = call("/api/datasets")["datasets"]
            assert datasets[0]["qas"] == 2
            profile = datasets[0]["profile"]
            assert profile["statistics"]["with_evidence_pages"] == 2
            assert profile["statistics"]["multi_page_evidence"] == 0
            qa_fields = {
                row["name"]: row for row in profile["qa_fields"]
            }
            assert qa_fields["question"]["coverage_percent"] == 100.0
            assert qa_fields["evidence_pages"]["value_types"] == [
                "array<integer>"
            ]
            summary = call("/api/datasets?detail=summary")["datasets"]
            assert summary[0]["qas"] == 2
            assert "paper_fields" not in summary[0]["profile"]
            detail = call("/api/datasets/qa_test.json")["dataset"]
            assert detail["profile"]["paper_fields"]
            with pytest.raises(urllib.error.HTTPError) as invalid_collection:
                call("/api/datasets?collection=unknown")
            assert invalid_collection.value.code == 400
            current = call("/api/datasets/qa_test.json/item?index=0")
            event = call(
                "/api/datasets/qa_test.json/review",
                "POST",
                {
                    "paper_id": "p1",
                    "qa_id": "Q1",
                    "action": "delete",
                    "expected_sha256": current["dataset_sha256"],
                },
            )
            assert event["status"] == "deleted"
            review_events = call(
                "/api/events?dataset=qa_test.json"
            )["events"]
            assert review_events[0]["actor_username"] == "czj-web"
            undone = call(
                "/api/datasets/qa_test.json/undo",
                "POST",
                {"event_id": event["event_id"]},
            )
            assert undone["qa_id"] == "Q1"
            request = urllib.request.Request(
                base + "/api/pdf?dataset=qa_test.json&paper=p1",
                headers={"Range": "bytes=0-3"},
            )
            request.add_header(
                "Cookie",
                "; ".join(
                    f"{cookie.name}={cookie.value}"
                    for cookie in cookie_jar
                ),
            )
            with opener.open(request, timeout=5) as response:
                assert response.status == 206
                assert response.read() == pdf_bytes[:4]
                assert response.headers["Content-Range"].startswith("bytes 0-3/")
                assert response.headers["Cache-Control"].startswith("private, max-age=86400")
                etag = response.headers["ETag"]
                assert etag

            conditional = urllib.request.Request(
                base + "/api/pdf?dataset=qa_test.json&paper=p1",
                headers={"If-None-Match": etag},
            )
            conditional.add_header(
                "Cookie",
                "; ".join(
                    f"{cookie.name}={cookie.value}"
                    for cookie in cookie_jar
                ),
            )
            with pytest.raises(urllib.error.HTTPError) as not_modified:
                opener.open(conditional, timeout=5)
            assert not_modified.value.code == 304
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_user_password_sessions_roles_and_audit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        fixture_dataset(qa_dir / "qa_test.json")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        admin = store.perimeter_user("czj-web")
        reviewer = store.create_user(
            "reviewer-one",
            "strong-pass-123",
            display_name="Reviewer One",
            actor=admin,
        )
        assert reviewer["role"] == "reviewer"
        authenticated = store.authenticate_password(
            "reviewer-one", "strong-pass-123"
        )
        token = store.create_session(authenticated)
        assert store.session_user(token)["username"] == "reviewer-one"
        store.update_user(reviewer["user_id"], disabled=True, actor=admin)
        with pytest.raises(PermissionError, match="Authentication required"):
            store.session_user(token)
        actions = [event["action"] for event in store.audit_events()]
        assert "user.create" in actions
        assert "user.update" in actions


def test_assignments_use_stable_qa_membership_and_reviewer_undo_scope() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        fixture_dataset(qa_dir / "qa_test.json")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        admin = store.perimeter_user("czj-web")
        reviewer = store.create_user(
            "reviewer-two",
            "strong-pass-456",
            display_name="Reviewer Two",
            actor=admin,
        )
        assignment = store.create_assignment(
            reviewer["user_id"], "qa_test.json", 2, 2, actor=admin
        )
        assert assignment["assigned_count"] == 1
        assert assignment["selection_mode"] == "range"
        assert assignment["first_index"] == 2

        first = store.item("qa_test.json", 0, actor=admin)
        store.review(
            "qa_test.json",
            "p1",
            "Q1",
            "delete",
            expected_sha256=first["dataset_sha256"],
            actor=admin,
        )
        shifted = store.item("qa_test.json", 0, actor=reviewer)
        assert shifted["item"]["qa_id"] == "Q2"
        assert shifted["can_modify"] is True
        kept = store.review(
            "qa_test.json",
            "p1",
            "Q2",
            "keep",
            expected_sha256=shifted["dataset_sha256"],
            actor=reviewer,
        )
        assert kept["status"] == "kept"
        store.undo("qa_test.json", kept["event_id"], actor=reviewer)
        with pytest.raises(PermissionError, match="只能撤销自己"):
            store.undo("qa_test.json", actor=reviewer)


def test_migrated_stable_assignment_must_be_recreated_before_editing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        fixture_dataset(qa_dir / "qa_test.json")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        admin = store.perimeter_user("czj-web")
        reviewer = store.create_user(
            "reviewer-stable", "strong-pass-789", actor=admin
        )
        assignment = store.create_assignment(
            reviewer["user_id"], "qa_test.json", 1, 1, actor=admin
        )
        with store.connect() as connection:
            connection.execute(
                "UPDATE review_assignments SET selection_mode='stable_members' "
                "WHERE assignment_id=?",
                (assignment["assignment_id"],),
            )
            connection.commit()

        migrated = store.list_assignments(reviewer["user_id"])[0]
        assert migrated["selection_mode"] == "stable_members"
        assert migrated["first_index"] == 1
        with pytest.raises(ValueError, match="deleted and recreated"):
            store.update_assignment(
                assignment["assignment_id"],
                user_id=reviewer["user_id"],
                dataset_id="qa_test.json",
                start_index=1,
                end_index=2,
                actor=admin,
            )


def test_assignment_ranges_cannot_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        fixture_dataset(qa_dir / "qa_test.json")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        admin = store.perimeter_user("czj-web")
        first = store.create_user(
            "reviewer-a", "strong-pass-a1", actor=admin
        )
        second = store.create_user(
            "reviewer-b", "strong-pass-b2", actor=admin
        )
        store.create_assignment(
            first["user_id"], "qa_test.json", 1, 1, actor=admin
        )
        with pytest.raises(ValueError, match="overlaps assignment"):
            store.create_assignment(
                second["user_id"], "qa_test.json", 1, 2, actor=admin
            )


def test_accounts_support_multiple_roles_registration_and_profile_updates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        qa_dir = root / "qa"
        pdf_dir = root / "pdfs"
        qa_dir.mkdir()
        pdf_dir.mkdir()
        fixture_dataset(qa_dir / "qa_test.json")
        store = ReviewStore(
            root / "review.sqlite3",
            DatasetRegistry(qa_dir, pdf_dir),
            root / "snapshots",
        )
        email, code = store.prepare_email_verification(
            "new.user@example.com", "register", remote_addr="127.0.0.1"
        )
        registered = store.register_user(
            "multi-role-user",
            "strong-pass-987",
            display_name="New User",
            email=email,
            email_code=code,
        )
        assert registered["roles"] == ["reviewer"]
        updated = store.update_user(
            registered["user_id"],
            roles=["reviewer", "admin"],
            actor=store.perimeter_user("czj-web"),
        )
        assert updated["roles"] == ["admin", "reviewer"]
        assert store.has_role(updated, "admin")
        assert store.has_role(updated, "reviewer")
        profile = store.update_profile(
            registered["user_id"],
            display_name="Updated User",
            actor=updated,
        )
        assert profile["display_name"] == "Updated User"
