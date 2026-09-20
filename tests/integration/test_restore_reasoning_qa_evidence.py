import copy
import json
from pathlib import Path
import sys

import pytest
from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pku_qa.pdf_assets import resolve_pdf_path
from pku_qa.workflows.cleaning.restore_reasoning_qa_evidence import (
    EvidenceRestoreError,
    dataset_question_answer_sha256,
    question_answer_sha256,
    restore_from_paths,
    restore_reasoning_qa_evidence,
)


REASONING_DATASET = (
    ROOT
    / "data/qa/3.reasoning/"
    "work__reasoning__historical_clean__batch00__n100.json"
)
SOURCE_DATASET = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
PDF_DIR = ROOT / "data/pdfs"


def _write_pdf(path: Path, page_count: int) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)
    with path.open("wb") as stream:
        writer.write(stream)


def _small_datasets() -> tuple[dict, dict]:
    reasoning = {
        "7": {
            "paper": "7",
            "source_pdf_id": "7",
            "QA": {
                "RQA1": {
                    "question": "Combined question?",
                    "answer": "Combined answer",
                    "source_qa_ids": ["QA1", "QA2"],
                    "reasoning_type": "conditional_inference",
                }
            },
        }
    }
    source = {
        "7": {
            "paper": "7",
            "QA": {
                "QA1": {
                    "question": "Source one?",
                    "answer": "One",
                    "evidence_pages": [3, 1, 3],
                },
                "QA2": {
                    "question": "Source two?",
                    "answer": "Two",
                    "evidence_pages": [2, 3],
                },
            },
        }
    }
    return reasoning, source


def test_real_reasoning_100_restores_every_source_and_valid_page() -> None:
    original = json.loads(REASONING_DATASET.read_text(encoding="utf-8"))
    restored, report = restore_from_paths(
        REASONING_DATASET,
        SOURCE_DATASET,
        PDF_DIR,
        expected_count=100,
    )

    assert report["reasoning_qa_count"] == 100
    assert report["resolved_reasoning_qa_count"] == 100
    assert report["source_qa_reference_count"] == 214
    assert report["source_pdf_count"] == 68
    assert report["union_evidence_page_reference_count"] == 211
    assert report["union_evidence_page_count_distribution"] == {
        "1": 29,
        "2": 39,
        "3": 26,
        "4": 4,
        "5": 2,
    }
    assert report["question_answer_hash_unchanged"] is True
    assert report["input_question_answer_sha256"] == report[
        "output_question_answer_sha256"
    ]
    assert dataset_question_answer_sha256(original) == (
        dataset_question_answer_sha256(restored)
    )

    source = json.loads(SOURCE_DATASET.read_text(encoding="utf-8"))
    observed = 0
    for paper_id, paper in restored.items():
        pdf = resolve_pdf_path(paper_id, [PDF_DIR])
        page_count = len(PdfReader(str(pdf)).pages)
        for qa_id, qa in paper["QA"].items():
            original_qa = original[paper_id]["QA"][qa_id]
            for key, value in original_qa.items():
                if key in {"annotation_provenance", "evidence_provenance"} and isinstance(value, dict):
                    # The restore operation refreshes the source path after the
                    # numbered QA directory migration; all other provenance
                    # fields must remain byte-for-byte equivalent.
                    restored_provenance = qa[key]
                    expected_provenance = json.loads(json.dumps(value).replace("data/qa/base/", "data/qa/1.base/"))
                    observed_provenance = dict(restored_provenance)
                    assert observed_provenance == expected_provenance
                else:
                    assert qa[key] == value
            assert set(qa) == set(original_qa) | {
                "evidence_pages",
                "evidence_provenance",
            }
            assert question_answer_sha256(qa) == question_answer_sha256(
                original_qa
            )

            pages = qa["evidence_pages"]
            assert pages == sorted(set(pages))
            assert pages
            assert all(type(page) is int for page in pages)
            assert min(pages) >= 1
            assert max(pages) <= page_count

            provenance = qa["evidence_provenance"]
            contributions = provenance["source_qa_contributions"]
            assert [row["source_qa_id"] for row in contributions] == qa[
                "source_qa_ids"
            ]
            assert len(contributions) == len(qa["source_qa_ids"])
            expected_union: set[int] = set()
            for row in contributions:
                source_qa = source[paper_id]["QA"][row["source_qa_id"]]
                expected_pages = sorted(set(source_qa["evidence_pages"]))
                assert row["evidence_pages"] == expected_pages
                assert row["source_question_answer_sha256"] == (
                    question_answer_sha256(source_qa)
                )
                expected_union.update(expected_pages)
                observed += 1
            assert pages == sorted(expected_union)
            assert provenance["source_pdf_page_count"] == page_count
            assert provenance["reasoning_question_answer_sha256"] == (
                question_answer_sha256(original_qa)
            )
            assert all(provenance["validation"].values())
    assert observed == 214


def test_restore_is_copying_deterministic_and_records_each_contribution(
    tmp_path: Path,
) -> None:
    reasoning, source = _small_datasets()
    before = copy.deepcopy(reasoning)
    _write_pdf(tmp_path / "7.pdf", 3)

    first, first_report = restore_reasoning_qa_evidence(
        reasoning,
        source,
        pdf_dir=tmp_path,
        source_dataset_path=tmp_path / "source.json",
        source_dataset_sha256="a" * 64,
        expected_count=1,
    )
    second, second_report = restore_reasoning_qa_evidence(
        reasoning,
        source,
        pdf_dir=tmp_path,
        source_dataset_path=tmp_path / "source.json",
        source_dataset_sha256="a" * 64,
        expected_count=1,
    )

    assert reasoning == before
    assert first == second
    assert first_report == second_report
    qa = first["7"]["QA"]["RQA1"]
    assert qa["evidence_pages"] == [1, 2, 3]
    assert [
        row["evidence_pages"]
        for row in qa["evidence_provenance"]["source_qa_contributions"]
    ] == [[1, 3], [2, 3]]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda reasoning, source: reasoning["7"]["QA"]["RQA1"].update(
                source_qa_ids=["QA1", "QA404"]
            ),
            "does not exist",
        ),
        (
            lambda reasoning, source: source["7"]["QA"]["QA1"].update(
                evidence_pages=[]
            ),
            "non-empty evidence_pages",
        ),
        (
            lambda reasoning, source: source["7"]["QA"]["QA1"].update(
                evidence_pages=[True]
            ),
            "non-integer page",
        ),
        (
            lambda reasoning, source: source["7"]["QA"]["QA1"].update(
                evidence_pages=[4]
            ),
            "outside the physical PDF range",
        ),
    ],
)
def test_restore_fails_closed_on_unverifiable_evidence(
    tmp_path: Path, mutation, message: str
) -> None:
    reasoning, source = _small_datasets()
    mutation(reasoning, source)
    _write_pdf(tmp_path / "7.pdf", 3)

    with pytest.raises(EvidenceRestoreError, match=message):
        restore_reasoning_qa_evidence(
            reasoning,
            source,
            pdf_dir=tmp_path,
            source_dataset_path=tmp_path / "source.json",
            source_dataset_sha256="a" * 64,
            expected_count=1,
        )


def test_restore_rejects_wrong_expected_count(tmp_path: Path) -> None:
    reasoning, source = _small_datasets()
    _write_pdf(tmp_path / "7.pdf", 3)

    with pytest.raises(EvidenceRestoreError, match="expected 100.*found 1"):
        restore_reasoning_qa_evidence(
            reasoning,
            source,
            pdf_dir=tmp_path,
            source_dataset_path=tmp_path / "source.json",
            source_dataset_sha256="a" * 64,
            expected_count=100,
        )
