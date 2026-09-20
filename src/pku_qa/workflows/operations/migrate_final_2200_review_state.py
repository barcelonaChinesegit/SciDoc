#!/usr/bin/env python3
"""Migrate manual-review state from five legacy components to four final files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pku_qa.services.review.data_review_manager import DEFAULT_DB, ReviewStore
from pku_qa.workflows.selection.normalize_final_2200_release import atomic_json
from pku_qa.workflows.selection.sync_final_2200_manifest import (
    FINAL_COMPONENTS,
    FINAL_DIR,
    MANIFEST_PATH,
    build_manifest,
)


LEGACY_DATASET_TARGETS = {
    "rel__challenge__ordinary_unanswerable__batch01__n1200__v1.json": {
        "ordinary_qa.json",
        "unanswerable_qa.json",
    },
    "rel__reasoning__refreshed__batch01__n100__claude_hard__v1.json": {
        "reasoning_qa.json"
    },
    "rel__reasoning__incremental__batch02__n100__claude_hard__v1.json": {
        "reasoning_qa.json"
    },
    "rel__cross_pdf__challenge__batch01__n400__v1.json": {"cross_pdf_qa.json"},
    "rel__cross_pdf__strict_dual_review__batch02__n400__v1.json": {
        "cross_pdf_qa.json"
    },
}
FINAL_DATASET_IDS = {filename for _, filename, _ in FINAL_COMPONENTS}


@dataclass(frozen=True)
class FinalItem:
    dataset_id: str
    paper_id: str
    qa_id: str
    source_index: int
    supplement: bool


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def final_items(final_dir: Path = FINAL_DIR) -> tuple[
    dict[tuple[str, str], FinalItem], dict[tuple[str, str, str], FinalItem]
]:
    by_legacy: dict[tuple[str, str], FinalItem] = {}
    by_current: dict[tuple[str, str, str], FinalItem] = {}
    for _, dataset_id, _ in FINAL_COMPONENTS:
        dataset = read_json(final_dir / dataset_id)
        source_index = 0
        for paper_id, paper in dataset.items():
            for qa_id, qa in paper["QA"].items():
                source_index += 1
                identity = qa.get("annotation_provenance", {}).get(
                    "final_2200_identity"
                )
                if not isinstance(identity, dict):
                    raise ValueError(f"Missing final identity: {dataset_id}:{paper_id}/{qa_id}")
                legacy = (
                    str(identity.get("legacy_paper_id", "")),
                    str(identity.get("legacy_qa_id", "")),
                )
                item = FinalItem(
                    dataset_id=dataset_id,
                    paper_id=str(paper_id),
                    qa_id=str(qa_id),
                    source_index=source_index,
                    supplement=isinstance(
                        qa.get("annotation_provenance", {}).get(
                            "post_manual_review_supplement"
                        ),
                        dict,
                    ),
                )
                if not all(legacy) or legacy in by_legacy:
                    raise ValueError(f"Duplicate or empty legacy identity: {legacy}")
                by_legacy[legacy] = item
                by_current[(dataset_id, str(paper_id), str(qa_id))] = item
    if len(by_current) != 2200:
        raise ValueError(f"Expected 2,200 final QA, found {len(by_current)}")
    return by_legacy, by_current


def target_for_legacy_dataset(dataset_id: str, item: FinalItem | None) -> str:
    allowed = LEGACY_DATASET_TARGETS.get(dataset_id)
    if allowed is None:
        raise ValueError(f"Unexpected legacy dataset: {dataset_id}")
    if item is None:
        if len(allowed) != 1:
            raise ValueError(f"Cannot infer tombstone target for {dataset_id}")
        return next(iter(allowed))
    if item.dataset_id not in allowed:
        raise ValueError(
            f"Legacy identity from {dataset_id} mapped to {item.dataset_id}"
        )
    return item.dataset_id


def backup_database(db_path: Path) -> Path:
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_path = backup_dir / f"review-before-four-file-{stamp}.sqlite3"
    with sqlite3.connect(db_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path


def deterministic_assignment_id(original: str, dataset_id: str) -> str:
    return hashlib.sha256(f"{original}:{dataset_id}".encode("utf-8")).hexdigest()[:32]


def is_current(
    connection: sqlite3.Connection,
    current_keys: set[tuple[str, str, str]],
) -> bool:
    legacy_count = sum(
        connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE dataset_id NOT IN ({','.join('?' for _ in FINAL_DATASET_IDS)})",
            tuple(sorted(FINAL_DATASET_IDS)),
        ).fetchone()[0]
        for table in ("reviews", "events", "review_assignments", "review_assignment_members")
    )
    if legacy_count:
        return False
    review_keys = {
        tuple(row)
        for row in connection.execute(
            "SELECT dataset_id,paper_id,qa_id FROM reviews"
        )
    }
    member_keys = {
        tuple(row)
        for row in connection.execute(
            "SELECT dataset_id,paper_id,qa_id FROM review_assignment_members"
        )
    }
    return review_keys == current_keys and member_keys == current_keys


def migrate(
    db_path: Path = DEFAULT_DB,
    final_dir: Path = FINAL_DIR,
    manifest_path: Path = MANIFEST_PATH,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    by_legacy, by_current = final_items(final_dir)
    current_keys = set(by_current)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        if is_current(connection, current_keys):
            return {
                "status": "current",
                "review_count": len(current_keys),
                "tombstone_count": connection.execute(
                    "SELECT COUNT(*) FROM review_tombstones"
                ).fetchone()[0],
                "assignment_count": connection.execute(
                    "SELECT COUNT(*) FROM review_assignments"
                ).fetchone()[0],
                "assignment_member_count": len(current_keys),
                "event_count": connection.execute(
                    "SELECT COUNT(*) FROM events"
                ).fetchone()[0],
            }

        reviews = connection.execute("SELECT * FROM reviews").fetchall()
        events = connection.execute("SELECT * FROM events").fetchall()
        assignments = connection.execute(
            "SELECT * FROM review_assignments ORDER BY created_at,assignment_id"
        ).fetchall()
        members = connection.execute(
            "SELECT * FROM review_assignment_members ORDER BY assignment_id,source_index"
        ).fetchall()

    migrated_reviews: list[tuple[str, str, str, str, float]] = []
    tombstones: list[tuple[str, str, str, str, str, float]] = []
    for row in reviews:
        key = (str(row["paper_id"]), str(row["qa_id"]))
        item = by_legacy.get(key)
        if item is None:
            if row["status"] != "deleted":
                raise ValueError(f"Unmapped non-deleted review: {dict(row)}")
            target = target_for_legacy_dataset(str(row["dataset_id"]), None)
            tombstones.append(
                (
                    target,
                    str(row["dataset_id"]),
                    key[0],
                    key[1],
                    str(row["status"]),
                    float(row["updated_at"]),
                )
            )
            continue
        target_for_legacy_dataset(str(row["dataset_id"]), item)
        migrated_reviews.append(
            (item.dataset_id, item.paper_id, item.qa_id, str(row["status"]), float(row["updated_at"]))
        )
    if len(migrated_reviews) != 2200 or {row[:3] for row in migrated_reviews} != current_keys:
        raise ValueError("Review migration does not cover the exact final 2,200 QA")

    assignments_by_id = {str(row["assignment_id"]): row for row in assignments}
    grouped_members: dict[tuple[str, str], list[FinalItem]] = defaultdict(list)
    tombstone_assignments: list[tuple[str, str]] = []
    assigned_current: set[tuple[str, str, str]] = set()
    for row in members:
        original_id = str(row["assignment_id"])
        item = by_legacy.get((str(row["paper_id"]), str(row["qa_id"])))
        if item is None:
            target = target_for_legacy_dataset(str(row["dataset_id"]), None)
            tombstone_assignments.append((original_id, target))
            continue
        target_for_legacy_dataset(str(row["dataset_id"]), item)
        grouped_members[(original_id, item.dataset_id)].append(item)
        assigned_current.add((item.dataset_id, item.paper_id, item.qa_id))

    unassigned_supplements = sorted(
        (item for key, item in by_current.items() if key not in assigned_current and item.supplement),
        key=lambda item: (item.dataset_id, item.source_index),
    )
    if len(unassigned_supplements) != len(tombstone_assignments):
        raise ValueError(
            "Deleted assignment slots and unassigned supplement count differ: "
            f"{len(tombstone_assignments)} != {len(unassigned_supplements)}"
        )
    supplement_by_target: dict[str, list[FinalItem]] = defaultdict(list)
    for item in unassigned_supplements:
        supplement_by_target[item.dataset_id].append(item)
    for original_id, target in sorted(tombstone_assignments):
        choices = supplement_by_target[target]
        if not choices:
            raise ValueError(f"No replacement QA for deleted assignment in {target}")
        item = choices.pop(0)
        grouped_members[(original_id, target)].append(item)
        assigned_current.add((item.dataset_id, item.paper_id, item.qa_id))
    if assigned_current != current_keys:
        raise ValueError("Assignment migration does not cover the exact final 2,200 QA")

    migrated_assignments: list[tuple[Any, ...]] = []
    migrated_members: list[tuple[str, str, str, str, int]] = []
    groups_by_original: dict[str, list[str]] = defaultdict(list)
    for original_id, target in grouped_members:
        groups_by_original[original_id].append(target)
    for original_id, targets in sorted(groups_by_original.items()):
        original = assignments_by_id[original_id]
        ordered_targets = sorted(targets)
        for position, target in enumerate(ordered_targets):
            assignment_id = (
                original_id
                if position == 0
                else deterministic_assignment_id(original_id, target)
            )
            items = sorted(
                grouped_members[(original_id, target)],
                key=lambda item: item.source_index,
            )
            migrated_assignments.append(
                (
                    assignment_id,
                    str(original["user_id"]),
                    target,
                    items[0].source_index,
                    items[-1].source_index,
                    len(items),
                    "stable_members",
                    float(original["created_at"]),
                    time.time(),
                    original["created_by_user_id"],
                )
            )
            migrated_members.extend(
                (assignment_id, target, item.paper_id, item.qa_id, item.source_index)
                for item in items
            )

    summary = {
        "status": "ready" if not apply else "migrated",
        "review_count": len(migrated_reviews),
        "tombstone_count": len(tombstones),
        "event_count": len(events),
        "assignment_count_before": len(assignments),
        "assignment_count_after": len(migrated_assignments),
        "assignment_member_count": len(migrated_members),
    }
    if not apply:
        return summary

    backup_path = backup_database(db_path)
    ReviewStore(
        db_path=db_path,
        snapshot_dir=db_path.parent / "snapshots",
        final_manifest_path=manifest_path,
    )
    migrated_at = time.time()
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM reviews")
        connection.executemany(
            "INSERT INTO reviews(dataset_id,paper_id,qa_id,status,updated_at) VALUES(?,?,?,?,?)",
            migrated_reviews,
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO review_tombstones(
                dataset_id,legacy_dataset_id,legacy_paper_id,legacy_qa_id,
                status,updated_at,migrated_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            [(*row, migrated_at) for row in tombstones],
        )
        for row in events:
            legacy_dataset = str(row["dataset_id"])
            legacy_paper = str(row["paper_id"])
            legacy_qa = str(row["qa_id"])
            item = by_legacy.get((legacy_paper, legacy_qa))
            target = target_for_legacy_dataset(legacy_dataset, item)
            connection.execute(
                """
                UPDATE events SET dataset_id=?,paper_id=?,qa_id=?,reversible=0,
                    legacy_dataset_id=?,legacy_paper_id=?,legacy_qa_id=?
                WHERE event_id=?
                """,
                (
                    target,
                    item.paper_id if item else legacy_paper,
                    item.qa_id if item else legacy_qa,
                    legacy_dataset,
                    legacy_paper,
                    legacy_qa,
                    row["event_id"],
                ),
            )
        connection.execute("DELETE FROM review_assignment_members")
        connection.execute("DELETE FROM review_assignments")
        connection.executemany(
            """
            INSERT INTO review_assignments(
                assignment_id,user_id,dataset_id,start_index,end_index,
                assigned_count,selection_mode,created_at,updated_at,created_by_user_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            migrated_assignments,
        )
        connection.executemany(
            """
            INSERT INTO review_assignment_members(
                assignment_id,dataset_id,paper_id,qa_id,source_index
            ) VALUES(?,?,?,?,?)
            """,
            migrated_members,
        )
        connection.execute(
            """
            INSERT INTO audit_events(
                audit_id,actor_user_id,actor_username,action,target_type,target_id,
                details_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                uuid.uuid4().hex,
                None,
                "system",
                "review_state.migrate_final_2200_four_files",
                "dataset_collection",
                "final_2200",
                json.dumps(summary, ensure_ascii=False, sort_keys=True),
                migrated_at,
            ),
        )
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise ValueError(f"Foreign-key errors after migration: {foreign_key_errors}")
        connection.commit()

    atomic_json(manifest_path, build_manifest(final_dir))
    summary["backup"] = str(backup_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(
                args.db,
                args.final_dir,
                args.manifest,
                apply=args.apply,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
