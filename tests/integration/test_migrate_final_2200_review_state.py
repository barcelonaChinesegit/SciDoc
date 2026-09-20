import shutil
import sqlite3
from pathlib import Path

from pku_qa.workflows.operations.migrate_final_2200_review_state import migrate


ROOT = Path(__file__).resolve().parents[2]


def test_production_review_state_maps_to_exact_four_file_membership(tmp_path: Path) -> None:
    backup_dir = ROOT / "data/web/review/backups"
    backups = sorted(backup_dir.glob("review-before-four-file-*.sqlite3"))
    source = backups[-1] if backups else ROOT / "data/web/review/review.sqlite3"
    migrated = tmp_path / "review.sqlite3"
    manifest = tmp_path / "manifest.json"
    shutil.copy2(source, migrated)

    preview = migrate(migrated, manifest_path=manifest)
    assert preview["review_count"] == 2200
    if preview["status"] == "current":
        assert preview["assignment_member_count"] == 2200
        return
    assert preview["tombstone_count"] == 2
    assert preview["assignment_member_count"] == 2200

    result = migrate(migrated, manifest_path=manifest, apply=True)
    assert result["status"] == "migrated"
    with sqlite3.connect(migrated) as connection:
        assert connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2200
        assert connection.execute(
            "SELECT COUNT(*) FROM review_assignment_members"
        ).fetchone()[0] == 2200
        assert connection.execute(
            "SELECT COUNT(*) FROM review_tombstones"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM events WHERE reversible!=0"
        ).fetchone()[0] == 0
        assert {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT dataset_id FROM reviews"
            )
        } == {
            "ordinary_qa.json",
            "unanswerable_qa.json",
            "reasoning_qa.json",
            "cross_pdf_qa.json",
        }

    assert migrate(migrated, manifest_path=manifest)["status"] == "current"
