from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pypdf import PdfWriter


SCRIPT = Path(__file__).resolve().parents[2] / "src/pku_qa/workflows/generation/build_global_cross_pdf_bundle_plan.py"
SPEC = importlib.util.spec_from_file_location("global_cross_plan", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_global_plan_builds_deterministic_three_document_context(tmp_path) -> None:
    source = {}
    for index in range(1, 7):
        source[str(index)] = {
            "primary_category": "Computer Science",
            "secondary_category": "Machine Learning",
            "QA": {
                "QA1": {"question": "How does weighted kappa measure ordinal agreement?", "answer": "Weighted kappa measures ordinal agreement.", "evidence_pages": [1]},
                "QA2": {"question": "Why is ordinal agreement calibrated?", "answer": "Ordinal agreement supports calibrated ranking.", "evidence_pages": [1]},
            },
        }
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with (tmp_path / f"{index}.pdf").open("wb") as handle:
            writer.write(handle)
    papers = MODULE.paper_rows(source)
    pairs, df = MODULE.related_pairs(papers, 50, 2)
    triples = MODULE.ranked_triples(papers, pairs, df, 10)
    selected = MODULE.select_triples(triples, 2, 1)
    assert len(selected) == 2
    planned, contexts, manifest, ledger = MODULE.build_plan(source, tmp_path, selected, df, 2)
    assert list(planned) == ["hxb_0001", "hxb_0002"]
    assert len(manifest[0]["sources"]) == 3
    assert len(contexts["contexts"]["hxb_0001"]["source_qas"]) == 6
    assert ledger[0]["source_qa_count"] == 6

    merged_dir = tmp_path / "merged"
    page_counts = MODULE.materialize_bundle_pdfs(manifest, merged_dir)
    assert page_counts == {"hxb_0001": 3, "hxb_0002": 3}
    assert (merged_dir / "hxb_0001.pdf").is_file()

    prefixed, prefixed_contexts, prefixed_manifest, prefixed_ledger = (
        MODULE.build_plan(source, tmp_path, selected[:1], df, 2, bundle_prefix="hxc")
    )
    assert list(prefixed) == ["hxc_0001"]
    assert list(prefixed_contexts["contexts"]) == ["hxc_0001"]
    assert prefixed_manifest[0]["id"] == "hxc_0001"
    assert prefixed_ledger[0]["bundle_id"] == "hxc_0001"


def test_paper_rows_rejects_out_of_range_evidence_before_planning(tmp_path) -> None:
    source = {
        "1": {
            "primary_category": "Statistics",
            "secondary_category": "Measurement",
            "QA": {
                "valid1": {"question": "weighted kappa", "answer": "ordinal", "evidence_pages": [1]},
                "valid2": {"question": "ordinal agreement", "answer": "kappa", "evidence_pages": [1]},
                "invalid": {"question": "bad offset", "answer": "bad", "evidence_pages": [555]},
            },
        }
    }
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_metadata({"/Title": "Weighted Kappa Calibration"})
    with (tmp_path / "1.pdf").open("wb") as handle:
        writer.write(handle)
    audit = {}

    rows = MODULE.paper_rows(source, tmp_path, audit)

    assert [qa["qa_id"] for qa in rows["1"]["qas"]] == ["valid1", "valid2"]
    assert rows["1"]["title"] == "Weighted Kappa Calibration"
    assert {"weighted", "kappa", "calibration"} <= rows["1"]["title_terms"]
    assert audit == {"invalid_evidence_qa_count": 1}


def test_related_pairs_can_require_shared_title_terms() -> None:
    papers = {
        "1": {
            "primary_category": "Statistics",
            "secondary_category": "Measurement",
            "terms": {"calibration", "ordinal", "agreement"},
            "title_terms": {"ordinal", "calibration"},
        },
        "2": {
            "primary_category": "Statistics",
            "secondary_category": "Measurement",
            "terms": {"calibration", "ordinal", "agreement"},
            "title_terms": {"ordinal", "rubric"},
        },
        "3": {
            "primary_category": "Statistics",
            "secondary_category": "Measurement",
            "terms": {"calibration", "ordinal", "agreement"},
            "title_terms": {"unrelated", "estimator"},
        },
    }

    pairs, df = MODULE.related_pairs(
        papers,
        max_df=10,
        min_shared=2,
        min_shared_title_terms=1,
    )

    assert set(pairs) == {("1", "2")}

    pair_bundles = MODULE.ranked_pair_bundles(papers, pairs, df)
    assert [row["paper_ids"] for row in pair_bundles] == [["1", "2"]]
    assert pair_bundles[0]["connected_edges"] == [["1", "2"]]

    triangle = {"paper_ids": ["1", "2", "3"]}
    assert not MODULE.complete_triangle(triangle, pairs)
    complete_pairs = dict(pairs)
    complete_pairs[("1", "3")] = {"calibration"}
    complete_pairs[("2", "3")] = {"agreement"}
    assert MODULE.complete_triangle(triangle, complete_pairs)


def test_build_plan_supports_two_document_fallback(tmp_path) -> None:
    source = {}
    papers = {}
    for paper_id, title in (("1", "Ordinal Calibration"), ("2", "Ordinal Rubrics")):
        source[paper_id] = {
            "primary_category": "Statistics",
            "secondary_category": "Measurement",
            "QA": {
                "QA1": {
                    "question": "How is ordinal agreement measured?",
                    "answer": "With weighted kappa.",
                    "evidence_pages": [1],
                },
                "QA2": {
                    "question": "What is calibrated?",
                    "answer": "The ordinal rubric.",
                    "evidence_pages": [1],
                },
            },
        }
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        with (tmp_path / f"{paper_id}.pdf").open("wb") as handle:
            writer.write(handle)
        papers[paper_id] = {
            "title": title,
            "primary_category": "Statistics",
            "secondary_category": "Measurement",
            "qas": [
                {
                    "qa_id": qa_id,
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "evidence_pages": qa["evidence_pages"],
                    "terms": {"ordinal", "calibration"},
                }
                for qa_id, qa in source[paper_id]["QA"].items()
            ],
        }
    selected = [
        {
            "paper_ids": ["1", "2"],
            "anchors": ["ordinal", "calibration"],
            "connected_edges": [["1", "2"]],
            "score": 1.0,
        }
    ]

    _, contexts, manifest, _ = MODULE.build_plan(
        source, tmp_path, selected, {"ordinal": 2, "calibration": 2}, 2, papers
    )

    assert len(manifest[0]["sources"]) == 2
    assert len(contexts["contexts"]["hxb_0001"]["source_qas"]) == 4


def test_choose_qas_keeps_context_for_title_only_anchor() -> None:
    paper = {
        "qas": [
            {
                "qa_id": "QA2",
                "question": "Which component is used?",
                "answer": "Encoder B",
                "terms": {"component", "encoder"},
            },
            {
                "qa_id": "QA1",
                "question": "Which score is reported?",
                "answer": "0.8",
                "terms": {"score", "reported"},
            },
        ]
    }

    selected = MODULE.choose_qas(
        paper, {"vision-language"}, {"vision-language": 2}, 1
    )

    assert [qa["qa_id"] for qa in selected] == ["QA1"]
