import pytest

from pku_qa.workflows.cleaning.build_final_2200_modality_review import merge_review_input


def test_merge_review_input_preserves_source_file_and_rejects_collisions() -> None:
    first = {"p1": {"paper": "p1", "QA": {"Q1": {"answer": "a"}}}}
    second = {"p1": {"paper": "p1", "QA": {"Q2": {"answer": "b"}}}}
    merged = merge_review_input(("a.json", first), ("b.json", second))
    assert merged["p1"]["QA"]["Q1"]["modality_review_source_file"] == "a.json"
    assert merged["p1"]["QA"]["Q2"]["modality_review_source_file"] == "b.json"

    with pytest.raises(ValueError, match="Duplicate QA key"):
        merge_review_input(("a.json", first), ("duplicate.json", first))
