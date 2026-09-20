from __future__ import annotations

from pku_qa.workflows.cleaning.audit_reasoning_pdf_dependency import (
    audit_dataset,
    balanced_candidate_rows,
    build_refreshed_and_incremental,
    separated_evidence_selection,
)


def qa(question: str, answer: str, pages: list[int]) -> dict:
    return {
        "question": question,
        "answer": answer,
        "source_qa_ids": ["QA1", "QA2"],
        "reasoning_focus": "sequential_reasoning",
        "intermediate_facts": [
            "Alpha identifies the method",
            "The method reduces error",
        ],
        "dual_model_validation": {
            "claude": {"decision": "KEEP", "confidence": 0.95},
            "gemini": {"decision": "KEEP", "confidence": 0.95},
        },
        "closed_book_pdf_dependency": {
            "models": {"Qwen4B": False, "Qwen8B": False},
            "both_models_correct": False,
        },
        "evidence_pages": pages,
        "evidence_provenance": {
            "source_qa_contributions": [
                {"source_qa_id": "QA1", "evidence_pages": [pages[0]]},
                {"source_qa_id": "QA2", "evidence_pages": [pages[-1]]},
            ]
        },
        "derivation": [
            {"source_qa_id": "QA1", "fact": "Alpha identifies the method"},
            {"source_qa_id": "QA2", "fact": "The method reduces error"},
            {"inference": "The identified method therefore reduces error"},
        ],
    }


def dataset(item: dict, qa_id: str = "RQA1") -> dict:
    return {
        "1": {
            "paper": "1",
            "primary_category": "Science",
            "secondary_category": "Test",
            "QA": {qa_id: item},
        }
    }


SOURCE = {
    "1": {
        "primary_category": "Science",
        "secondary_category": "Test",
        "QA": {
            "QA1": {"question": "Which method?", "answer": "Alpha"},
            "QA2": {"question": "What does it do?", "answer": "reduces error"},
        },
    }
}


def test_separated_evidence_requires_distinct_non_adjacent_source_pages() -> None:
    assert separated_evidence_selection(qa("Question", "Alpha reduces error", [1, 5]), 3) == [1, 5]
    assert separated_evidence_selection(qa("Question", "Alpha reduces error", [2, 3]), 3) is None


def test_audit_flags_leakage_and_closed_book_success() -> None:
    item = qa("Does Alpha reduce error?", "Alpha reduces error", [1, 5])
    result = audit_dataset(
        dataset(item),
        SOURCE,
        {("1", "RQA1"): {"gpt": True, "qwen": True}},
        ["gpt", "qwen"],
        3,
    )
    row = result["items"][0]
    assert row["replace"] is True
    assert "source_or_final_answer_leaked_in_question" in row["reasons"]
    assert "all_required_models_answered_correctly_without_pdf" in row["reasons"]


def test_refresh_and_incremental_outputs_do_not_reuse_candidates() -> None:
    existing_item = qa("Does Alpha reduce error?", "Alpha reduces error", [1, 5])
    pool = {
        "1": {
            **SOURCE["1"],
            "paper": "1",
            "QA": {
                "NEW1": qa(
                    "Which hidden method satisfies both distant constraints?",
                    "Alpha reduces error after the hidden method is identified from evidence.",
                    [1, 5],
                ),
                "NEW2": qa(
                    "Which hidden method satisfies the alternative constraints?",
                    "Alpha improves stability after the hidden method is identified from evidence.",
                    [2, 8],
                ),
            },
        }
    }
    audit = {
        "items": [
            {"paper_id": "1", "qa_id": "RQA1", "replace": True}
        ]
    }
    refreshed, incremental, summary = build_refreshed_and_incremental(
        dataset(existing_item), SOURCE, pool, audit, 1
    )
    refreshed_questions = {
        row["question"]
        for paper in refreshed.values()
        for row in paper["QA"].values()
    }
    incremental_questions = {
        row["question"]
        for paper in incremental.values()
        for row in paper["QA"].values()
    }
    assert len(refreshed_questions) == 1
    assert len(incremental_questions) == 1
    assert refreshed_questions.isdisjoint(incremental_questions)
    assert summary["replaced_existing"] == 1
    assert summary["incremental_reasoning_focus_distribution"] == {
        "sequential_reasoning": 1
    }


def test_balanced_candidates_use_weighted_focus_rotation() -> None:
    rows = []
    foci = (
        "sequential_reasoning",
        "conditional_filtering",
        "similar_concept_discrimination",
        "model_relationship",
        "metric_reasoning",
    )
    for focus in foci:
        for index in range(10):
            rows.append((focus, str(index), {"reasoning_focus": focus}))
    ordered = balanced_candidate_rows(rows)
    first_twenty = [row[2]["reasoning_focus"] for row in ordered[:20]]
    assert first_twenty.count("sequential_reasoning") == 6
    assert first_twenty.count("conditional_filtering") == 5
    assert first_twenty.count("similar_concept_discrimination") == 4
    assert first_twenty.count("model_relationship") == 3
    assert first_twenty.count("metric_reasoning") == 2
