import json
from pathlib import Path

from pku_qa.workflows.selection.sync_final_2200_manifest import (
    FINAL_COMPONENTS,
    build_manifest,
)
from pku_qa.workflows.selection.single_pdf_release_views import (
    ORDINARY_SOURCE,
    ORDINARY_VIEW,
    UNANSWERABLE_SOURCE,
    UNANSWERABLE_VIEW,
    build_evidence_ablation_dataset,
    merge_single_pdf_release,
    qa_count,
    read_dataset,
    sync_release_views,
)


ROOT = Path(__file__).resolve().parents[2]


def test_manifest_binds_exactly_the_four_canonical_files() -> None:
    manifest = build_manifest()

    assert manifest["qa_count"] == 2200
    assert manifest["component_count"] == 4
    assert [row["dataset_id"] for row in manifest["components"]] == [
        filename for _, filename, _ in FINAL_COMPONENTS
    ]
    assert sum(row["qa_count"] for row in manifest["components"]) == 2200


def test_checked_in_manifest_matches_canonical_files() -> None:
    expected = build_manifest()
    actual = json.loads(
        (
            ROOT
            / "data/qa/7.final_2200/rel__collection__final_2200__manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert actual == expected


def test_categorized_single_pdf_views_match_final_files() -> None:
    sync_release_views(check_only=True)

    assert ORDINARY_VIEW.read_bytes() == ORDINARY_SOURCE.read_bytes()
    assert UNANSWERABLE_VIEW.read_bytes() == UNANSWERABLE_SOURCE.read_bytes()


def test_runtime_single_pdf_inputs_are_derived_from_final_files() -> None:
    ordinary = read_dataset(ORDINARY_SOURCE)
    unanswerable = read_dataset(UNANSWERABLE_SOURCE)

    assert qa_count(merge_single_pdf_release(ordinary, unanswerable)) == 1200
    assert qa_count(build_evidence_ablation_dataset(ordinary)) == 676
