import json
from pathlib import Path

from pku_qa.workflows.cleaning.sync_human_reviewed_qa import (
    apply_core_gold,
    atomic_write_json,
    authority_index,
    effective_answer,
    iter_qa_refs,
    match_gold,
    sync_unanswerable,
)


ROOT = Path(__file__).resolve().parents[2]


def authority_payload():
    return {
        "7": {
            "paper": "paper.pdf",
            "QA": {
                "QA2": {
                    "question": "  What is X? ",
                    "answer": "human answer",
                    "evidence_pages": [2, 4],
                    "modal_types": ["text", "table"],
                    "question_type": "Inferential",
                    "question_category": "Methods",
                }
            },
        }
    }


def test_same_question_in_different_paper_does_not_cross_match():
    by_paper, by_question, _ = authority_index(authority_payload())
    payload = {
        "8": {
            "QA": {
                "QA1": {"question": "What is X?", "answer": "other paper"}
            }
        }
    }
    ref = next(iter(iter_qa_refs(payload)))
    assert ref.paper_id == "8"
    assert match_gold(ref, by_paper, by_question) is None


def test_short_answer_receives_all_human_core_fields():
    by_paper, _, _ = authority_index(authority_payload())
    gold = by_paper[("7", "what is x?")]
    qa = {
        "question": "What is X?",
        "answer": "old",
        "evidence_pages": [2],
        "modal_types": ["text"],
        "question_type": "Literal",
        "question_category": "Methods",
    }
    changes = apply_core_gold(qa, gold)
    assert qa["answer"] == "human answer"
    assert qa["evidence_pages"] == [2, 4]
    assert qa["modal_types"] == ["text", "table"]
    assert qa["question_type"] == "Inferential"
    assert set(changes) == {
        "question",
        "answer",
        "evidence_pages",
        "modal_types",
        "question_type",
    }


def test_multiple_choice_keeps_label_and_updates_selected_text():
    by_paper, _, _ = authority_index(authority_payload())
    gold = by_paper[("7", "what is x?")]
    qa = {
        "question": "What is X?",
        "answer": "B",
        "options": [
            {"id": "A", "text": "distractor"},
            {"id": "B", "text": "old answer"},
        ],
    }
    changes = apply_core_gold(qa, gold)
    assert qa["answer"] == "B"
    assert qa["options"][1]["text"] == "human answer"
    assert changes["effective_answer"] == ["old answer", "human answer"]


def test_unanswerable_has_no_public_evidence_but_keeps_neighbor_metadata():
    by_paper, by_question, _ = authority_index(authority_payload())
    payload = {
        "7": {
            "QA": {
                "UQA1": {
                    "question": "What is Y?",
                    "answer": "Unanswerable",
                    "evidence_pages": [],
                    "oracle_pages": [2],
                    "evidence_items": [],
                    "evidence_hops": 0,
                    "evidence_span": 0,
                    "evidence_span_ratio": 0,
                    "unanswerable_construction": {
                        "original_question": "What is X?",
                        "original_answer": "old",
                        "nearest_neighbor_pages": [2],
                    },
                }
            }
        }
    }
    ref = next(iter(iter_qa_refs(payload)))
    changes = sync_unanswerable(ref, by_paper, by_question)
    assert ref.qa["oracle_pages"] == []
    assert ref.qa["unanswerable_construction"]["original_answer"] == "human answer"
    assert ref.qa["unanswerable_construction"]["nearest_neighbor_pages"] == [2, 4]
    assert "oracle_pages" in changes


def test_authority_fixture_is_valid_json_round_trip(tmp_path: Path):
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(authority_payload()), encoding="utf-8")
    by_paper, _, _ = authority_index(json.loads(path.read_text()))
    assert len(by_paper) == 1


def test_atomic_writer_preserves_existing_crlf(tmp_path: Path):
    path = tmp_path / "dataset.json"
    path.write_bytes(b'{\r\n  "old": true\r\n}\r\n')
    atomic_write_json(path, {"new": True})
    content = path.read_bytes()
    assert b"\r\n" in content
    assert b"\n" not in content.replace(b"\r\n", b"")


def test_repository_source_follows_human_gold_and_release_view_follows_final():
    authority = json.loads(
        (ROOT / "data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json").read_text()
    )
    by_paper, by_question, _ = authority_index(authority)
    assert len(by_paper) == 983
    assert len(by_question) == 983

    source = json.loads(
        (ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json").read_text()
    )
    source_matches = 0
    for ref in iter_qa_refs(source):
        gold = match_gold(ref, by_paper, by_question)
        if gold is None:
            continue
        source_matches += 1
        for field in (
            "question",
            "answer",
            "evidence_pages",
            "modal_types",
            "question_type",
            "question_category",
        ):
            assert ref.qa.get(field) == gold.qa.get(field)
    assert source_matches == 983

    release_view = (
        ROOT / "data/qa/1.base/rel__single_pdf__ordinary__batch01__n1000.json"
    )
    final = ROOT / "data/qa/7.final_2200/ordinary_qa.json"
    assert release_view.read_bytes() == final.read_bytes()


def test_repository_multiple_choice_effective_answers_follow_human_gold():
    authority = json.loads(
        (ROOT / "data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json").read_text()
    )
    by_paper, by_question, _ = authority_index(authority)
    dataset = json.loads((ROOT / "data/qa/1.base/work__single_pdf__raw_mixed__n6204.json").read_text())
    matches = 0
    for ref in iter_qa_refs(dataset):
        gold = match_gold(ref, by_paper, by_question)
        if gold is None:
            continue
        matches += 1
        assert effective_answer(ref.qa) == gold.qa["answer"]
    assert matches == 983
