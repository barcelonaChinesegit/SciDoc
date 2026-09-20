#!/usr/bin/env python3
"""Prepare inspectable evidence packets for manual cross-paper QA review.

This script does not score QA items and does not call a model.  It only joins
the QA dataset to the bundle manifest, maps evidence pages to source papers,
and extracts the cited PDF pages so a human reviewer can make the decision.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from pku_qa.pdf_assets import resolve_pdf_path


DOC_RE = re.compile(r"\bDoc(?:ument)?\s*([1-9]\d*)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
STOPWORDS = {
    "about",
    "according",
    "also",
    "and",
    "are",
    "based",
    "between",
    "both",
    "compare",
    "compared",
    "does",
    "document",
    "from",
    "have",
    "how",
    "into",
    "paper",
    "that",
    "their",
    "these",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-json",
        type=Path,
        default=Path("data/qa/qa_expansion_cross_pdf_cleaned_20260711.json"),
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path(
            "data/qa_generation/expansion_20260711/"
            "cross_pdf_bundle_manifest.json"
        ),
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/qa/6.review/cross_pdf"),
    )
    parser.add_argument("--min-qa-per-pdf", type=int, default=10)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def relevant_excerpt(page_text: str, question: str, answer: str) -> str:
    """Return high-overlap sentences as a navigation aid, not a verdict."""
    query_tokens = {
        token.lower()
        for token in WORD_RE.findall(f"{question} {answer}")
        if token.lower() not in STOPWORDS
    }
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", page_text)
        if len(sentence.strip()) >= 30
    ]
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_tokens = {token.lower() for token in WORD_RE.findall(sentence)}
        score = len(query_tokens & sentence_tokens)
        scored.append((score, -index, sentence))
    selected = [
        sentence
        for score, _, sentence in sorted(scored, reverse=True)[:4]
        if score > 0
    ]
    excerpt = " ".join(selected)
    return excerpt[:2400]


def source_for_page(sources: list[dict[str, Any]], page: int) -> int | None:
    for index, source in enumerate(sources, start=1):
        if source["merged_start_page"] <= page <= source["merged_end_page"]:
            return index
    return None


def main() -> None:
    args = parse_args()
    qa_data = json.loads(args.qa_json.read_text(encoding="utf-8"))
    manifest_rows = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    manifest = {row["id"]: row for row in manifest_rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_ids = sorted(
        (
            bundle_id
            for bundle_id, record in qa_data.items()
            if len(record["QA"]) >= args.min_qa_per_pdf
        ),
        key=lambda bundle_id: (-len(qa_data[bundle_id]["QA"]), bundle_id),
    )

    records: list[dict[str, Any]] = []
    structural_counts: Counter[str] = Counter()
    for bundle_id in candidate_ids:
        bundle = manifest[bundle_id]
        reader = PdfReader(resolve_pdf_path(bundle_id, [args.pdf_dir]))
        page_cache: dict[int, str] = {}
        for qa_id, qa in qa_data[bundle_id]["QA"].items():
            evidence_pages = qa.get("evidence_pages") or []
            page_doc_numbers = [
                source_for_page(bundle["sources"], int(page))
                for page in evidence_pages
                if isinstance(page, int) or str(page).isdigit()
            ]
            evidence_doc_numbers = sorted(
                {number for number in page_doc_numbers if number is not None}
            )
            question_doc_numbers = sorted(
                {int(value) for value in DOC_RE.findall(qa.get("question", ""))}
            )
            answer_doc_numbers = sorted(
                {int(value) for value in DOC_RE.findall(qa.get("answer", ""))}
            )

            flags: list[str] = []
            if len(evidence_doc_numbers) < 2:
                flags.append("evidence_spans_fewer_than_two_source_papers")
            if len(question_doc_numbers) < 2:
                flags.append("question_names_fewer_than_two_documents")
            if len(answer_doc_numbers) < 2:
                flags.append("answer_names_fewer_than_two_documents")
            if any(number is None for number in page_doc_numbers):
                flags.append("evidence_page_not_mapped_to_source")
            if any(
                not isinstance(page, int) or page < 1 or page > len(reader.pages)
                for page in evidence_pages
            ):
                flags.append("evidence_page_out_of_bounds")
            structural_counts.update(flags)

            evidence: list[dict[str, Any]] = []
            for page in evidence_pages:
                if not isinstance(page, int) or page < 1 or page > len(reader.pages):
                    continue
                if page not in page_cache:
                    page_cache[page] = normalize_text(
                        reader.pages[page - 1].extract_text() or ""
                    )
                doc_number = source_for_page(bundle["sources"], page)
                source = (
                    bundle["sources"][doc_number - 1]
                    if doc_number is not None
                    else None
                )
                evidence.append(
                    {
                        "page": page,
                        "doc_number": doc_number,
                        "source_title": source["title"] if source else None,
                        "relevant_excerpt": relevant_excerpt(
                            page_cache[page],
                            qa.get("question", ""),
                            qa.get("answer", ""),
                        ),
                        "page_text_characters": len(page_cache[page]),
                        "text": page_cache[page],
                    }
                )

            records.append(
                {
                    "bundle_id": bundle_id,
                    "qa_id": qa_id,
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "evidence_pages": evidence_pages,
                    "evidence_doc_numbers": evidence_doc_numbers,
                    "question_doc_numbers": question_doc_numbers,
                    "answer_doc_numbers": answer_doc_numbers,
                    "structural_flags": flags,
                    "sources": [
                        {
                            "doc_number": index,
                            "paper_id": source["paper_id"],
                            "arxiv_id": source["arxiv_id"],
                            "title": source["title"],
                            "merged_start_page": source["merged_start_page"],
                            "merged_end_page": source["merged_end_page"],
                        }
                        for index, source in enumerate(bundle["sources"], start=1)
                    ],
                    "evidence": evidence,
                }
            )

    packet_path = args.output_dir / "review_packet.jsonl"
    with packet_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "source_qa_json": str(args.qa_json),
        "source_manifest_json": str(args.manifest_json),
        "candidate_rule": f"at least {args.min_qa_per_pdf} cleaned QA per PDF",
        "candidate_pdf_count": len(candidate_ids),
        "candidate_qa_count": len(records),
        "qa_per_pdf_distribution": dict(
            sorted(
                Counter(
                    len(qa_data[bundle_id]["QA"]) for bundle_id in candidate_ids
                ).items()
            )
        ),
        "structural_flag_counts": dict(structural_counts),
        "candidate_pdf_ids": candidate_ids,
    }
    (args.output_dir / "profile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
