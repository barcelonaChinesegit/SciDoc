from pku_qa.workflows.reporting.build_final_2200_classification import (
    manifest_from_four_files,
    source_document_count,
)


def test_source_document_count_uses_qa_evidence_documents() -> None:
    assert source_document_count(
        "cross_pdf_first_400", {"evidence_source_docs": [1, 2, 2, 3]}
    ) == 3
    assert source_document_count(
        "cross_pdf_second_400", {"source_document_count": 2}
    ) == 2
    assert source_document_count("reasoning_refreshed_100", {}) == 1


def test_manifest_from_four_files_requires_exact_delivery_counts(tmp_path) -> None:
    import json

    def dataset(count: int, *, answer: str = "answer") -> dict:
        return {
            "p": {
                "paper": "p",
                "primary_category": "Computer Science",
                "secondary_category": "Artificial Intelligence",
                "QA": {
                    f"QA{i}": {
                        "question": "q",
                        "answer": answer,
                        "evidence_pages": [] if answer == "Unanswerable" else [1],
                        "modal_types": ["text"],
                        "question_type": "Literal",
                        "question_category": "Algorithm & Architecture Detail",
                    }
                    for i in range(count)
                },
            }
        }

    for name, count, answer in (
        ("ordinary_qa.json", 1000, "answer"),
        ("unanswerable_qa.json", 200, "Unanswerable"),
        ("reasoning_qa.json", 200, "answer"),
        ("cross_pdf_qa.json", 800, "answer"),
    ):
        (tmp_path / name).write_text(json.dumps(dataset(count, answer=answer)))

    manifest = manifest_from_four_files(tmp_path)
    assert [component["qa_count"] for component in manifest["components"]] == [
        1000,
        200,
        200,
        800,
    ]
    assert [component["label"] for component in manifest["components"]] == [
        "ordinary_qa.json",
        "unanswerable_qa.json",
        "reasoning_qa.json",
        "cross_pdf_qa.json",
    ]
