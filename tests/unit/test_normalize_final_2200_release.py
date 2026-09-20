from pku_qa.workflows.selection.normalize_final_2200_release import normalize_reasoning_qa


def test_reasoning_normalization_preserves_complete_visual_api_review() -> None:
    qa = {
        "question": "q",
        "answer": "a",
        "evidence_pages": [1],
        "modal_types": ["text", "formula"],
        "question_type": "Inferential",
        "question_category": "Algorithm & Architecture Detail",
        "source_qa_ids": ["Q1"],
        "annotation_provenance": {
            "modality_api_review": {
                "modal_types": ["text", "formula"],
                "page_support_complete": True,
            }
        },
    }
    source = {
        "primary_category": "Computer Science",
        "QA": {
            "Q1": {
                "modal_types": ["text"],
                "question_category": "Algorithm & Architecture Detail",
            }
        },
    }
    result = normalize_reasoning_qa(qa, source)
    assert result["modal_types"] == ["text", "formula"]
    assert result["annotation_provenance"]["final_2200_core_normalization"][
        "modal_types_method"
    ] == "vision_api_pdf_page_review"
