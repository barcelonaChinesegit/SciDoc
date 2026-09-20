import json
from pathlib import Path

import jsonschema

from pku_qa.workflows.selection.final_2200_contract import (
    canonical_modal_types,
    category_for_reasoning_type,
    validate_final_dataset,
)


ROOT = Path(__file__).resolve().parents[2]


def test_canonical_modal_types_normalizes_aliases_and_order() -> None:
    assert canonical_modal_types(["table", "figure", "text", "table"]) == [
        "text",
        "image",
        "table",
    ]


def test_cross_pdf_reasoning_type_maps_into_discipline_taxonomy() -> None:
    assert (
        category_for_reasoning_type("Computer Science", "metric_reasoning")
        == "Experiment & Result Validation"
    )
    assert (
        category_for_reasoning_type("Mathematics", "component_hierarchy")
        == "Definition & Theorem Statement"
    )


def test_final_2200_components_follow_shared_core_schema() -> None:
    manifest_path = (
        ROOT / "data/qa/7.final_2200/rel__collection__final_2200__manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schemas/final_2200_qa.schema.json").read_text(encoding="utf-8")
    )
    total = 0
    for component in manifest["components"]:
        dataset = json.loads(Path(component["path"]).read_text(encoding="utf-8"))
        jsonschema.validate(dataset, schema)
        validate_final_dataset(dataset, label=component["id"])
        total += sum(len(paper["QA"]) for paper in dataset.values())
    assert total == 2200


def test_four_file_delivery_has_global_ids_and_cross_pdf_last() -> None:
    names = (
        "ordinary_qa.json",
        "unanswerable_qa.json",
        "reasoning_qa.json",
        "cross_pdf_qa.json",
    )
    qa_ids: list[str] = []
    for name in names:
        dataset = json.loads((ROOT / "data/qa/7.final_2200" / name).read_text())
        validate_final_dataset(dataset, label=name)
        qa_ids.extend(
            str(qa_id)
            for paper in dataset.values()
            for qa_id in paper["QA"]
        )
        assert all(Path(paper["pdf_path"]).is_file() for paper in dataset.values())

    assert qa_ids == [f"QA{index:04d}" for index in range(1, 2201)]
    cross = json.loads(
        (ROOT / "data/qa/7.final_2200/cross_pdf_qa.json").read_text()
    )
    assert all(paper_id.startswith("z_cross_") for paper_id in cross)
