#!/usr/bin/env python3
"""Plan coherent three-paper QA-only bundles from the 4211 base QA corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from pku_qa.pdf_assets import output_pdf_path, resolve_pdf_path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = ROOT / "data/qa/1.base/rel__single_pdf__short_answer__n4211__v1.json"
DEFAULT_PDF_DIR = ROOT / "data/pdfs"
STOPWORDS = {
    "about", "across", "after", "approach", "approaches", "based", "between",
    "compared", "comparison", "different", "during", "evaluation", "method",
    "methods", "model", "models", "paper", "performance", "question", "result",
    "results", "specific", "study", "system", "systems", "their", "these",
    "using", "which", "would", "framework", "proposed", "reported", "shows",
    "adaptation", "advantage", "architecture", "challenge", "dataset", "direct",
    "engineering", "features", "limitation", "limited", "mentioned", "primary",
    "reviewed", "technique", "traditional", "property", "state", "through", "where",
    "according", "assess", "compare", "design", "develop", "effects", "identified",
    "included", "learning", "level", "multiple", "related", "research", "review",
    "showed", "significance", "significant", "statistical", "studies", "tests", "three",
    "analysis", "survey", "inference", "large", "further", "toward", "application",
    "applications",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-bundles", type=int, default=240)
    parser.add_argument(
        "--bundle-prefix",
        default="hxb",
        help="Stable ID prefix used to keep independently planned pools disjoint.",
    )
    parser.add_argument("--max-paper-reuse", type=int, default=2)
    parser.add_argument("--max-term-paper-frequency", type=int, default=50)
    parser.add_argument("--min-shared-terms", type=int, default=2)
    parser.add_argument(
        "--min-shared-title-terms",
        type=int,
        default=0,
        help=(
            "Require each related-paper edge to share this many filtered title "
            "terms in addition to QA anchors."
        ),
    )
    parser.add_argument("--qas-per-paper", type=int, default=4)
    parser.add_argument("--max-neighbors-per-paper", type=int, default=40)
    parser.add_argument(
        "--include-pair-bundles",
        action="store_true",
        help=(
            "After all ranked three-paper groups, use remaining related pairs "
            "to reach the requested bundle count."
        ),
    )
    parser.add_argument(
        "--pair-bundles-only",
        action="store_true",
        help="Plan only related two-paper bundles.",
    )
    parser.add_argument(
        "--require-complete-triangle",
        action="store_true",
        help="Keep a three-paper bundle only when all three paper pairs are related.",
    )
    parser.add_argument(
        "--materialize-pdf-dir",
        type=Path,
        default=None,
        help="Optionally merge each planned bundle into <bundle_id>.pdf for review/eval.",
    )
    parser.add_argument(
        "--category-level", choices=("secondary", "primary"), default="secondary"
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def scientific_terms(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9]+(?:-[a-z0-9]+)*", str(value).casefold())
        if len(token) >= 5 and token not in STOPWORDS and not token.startswith("paper")
    }


def paper_rows(
    source: dict[str, Any],
    pdf_dir: Path | None = None,
    audit: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for paper_id, paper in source.items():
        page_count = None
        title = f"Paper {paper_id}"
        if pdf_dir is not None:
            pdf_path = resolve_pdf_path(paper_id, [pdf_dir], required=False)
            if pdf_path is None:
                if audit is not None:
                    audit["missing_pdf_paper_count"] = audit.get(
                        "missing_pdf_paper_count", 0
                    ) + 1
                continue
            reader = PdfReader(str(pdf_path))
            page_count = len(reader.pages)
            metadata_title = str(
                reader.metadata.title if reader.metadata else ""
            ).strip()
            if metadata_title:
                title = metadata_title
        title_terms = scientific_terms(title)
        qas = []
        for qa_id, qa in paper.get("QA", {}).items():
            if not isinstance(qa, dict) or not qa.get("evidence_pages"):
                continue
            evidence_pages = sorted(
                set(int(page) for page in qa["evidence_pages"])
            )
            if page_count is not None and any(
                page < 1 or page > page_count for page in evidence_pages
            ):
                if audit is not None:
                    audit["invalid_evidence_qa_count"] = audit.get(
                        "invalid_evidence_qa_count", 0
                    ) + 1
                continue
            text = f"{qa.get('question', '')} {qa.get('answer', '')}"
            qas.append(
                {
                    "qa_id": str(qa_id),
                    "question": str(qa.get("question", "")).strip(),
                    "answer": str(qa.get("answer", "")).strip(),
                    "evidence_pages": evidence_pages,
                    "terms": scientific_terms(text),
                }
            )
        if len(qas) < 2:
            if audit is not None:
                audit["insufficient_valid_qa_paper_count"] = audit.get(
                    "insufficient_valid_qa_paper_count", 0
                ) + 1
            continue
        rows[str(paper_id)] = {
            "title": title,
            "title_terms": title_terms,
            "primary_category": str(paper.get("primary_category", "")),
            "secondary_category": str(paper.get("secondary_category", "")),
            "qas": qas,
            "terms": title_terms | set().union(*(qa["terms"] for qa in qas)),
        }
    return rows


def related_pairs(
    papers: dict[str, dict[str, Any]],
    max_df: int,
    min_shared: int,
    category_level: str = "secondary",
    min_shared_title_terms: int = 0,
) -> tuple[dict[tuple[str, str], set[str]], dict[str, int]]:
    postings: dict[str, list[str]] = defaultdict(list)
    for paper_id, paper in papers.items():
        for term in paper["terms"]:
            postings[term].append(paper_id)
    df = {term: len(set(ids)) for term, ids in postings.items()}
    pair_terms: dict[tuple[str, str], set[str]] = defaultdict(set)
    for term, ids in postings.items():
        unique = sorted(set(ids))
        if not 2 <= len(unique) <= max_df:
            continue
        for left, right in combinations(unique, 2):
            category_field = f"{category_level}_category"
            if (
                not papers[left][category_field]
                or papers[left][category_field] != papers[right][category_field]
            ):
                continue
            if len(
                papers[left].get("title_terms", set())
                & papers[right].get("title_terms", set())
            ) < min_shared_title_terms:
                continue
            pair_terms[(left, right)].add(term)
    return (
        {pair: terms for pair, terms in pair_terms.items() if len(terms) >= min_shared},
        df,
    )


def pair_weight(terms: set[str], paper_count: int, df: dict[str, int]) -> float:
    return sum(math.log((paper_count + 1) / (df[term] + 1)) for term in terms)


def ranked_pair_bundles(
    papers: dict[str, dict[str, Any]],
    pairs: dict[tuple[str, str], set[str]],
    df: dict[str, int],
) -> list[dict[str, Any]]:
    """Return deterministic two-paper fallbacks after three-paper planning."""
    count = len(papers)
    rows = [
        {
            "paper_ids": [left, right],
            "anchors": sorted(anchors),
            "connected_edges": [[left, right]],
            "score": round(pair_weight(anchors, count, df) + len(anchors), 6),
        }
        for (left, right), anchors in pairs.items()
    ]
    return sorted(
        rows,
        key=lambda row: (-row["score"], -len(row["anchors"]), row["paper_ids"]),
    )


def complete_triangle(
    row: dict[str, Any], pairs: dict[tuple[str, str], set[str]]
) -> bool:
    return len(row["paper_ids"]) == 3 and all(
        tuple(sorted(edge)) in pairs
        for edge in combinations(row["paper_ids"], 2)
    )


def ranked_triples(
    papers: dict[str, dict[str, Any]],
    pairs: dict[tuple[str, str], set[str]],
    df: dict[str, int],
    max_neighbors: int,
) -> list[dict[str, Any]]:
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    count = len(papers)
    for (left, right), anchors in pairs.items():
        weight = pair_weight(anchors, count, df)
        neighbors[left].append((right, weight))
        neighbors[right].append((left, weight))
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for center, rows in neighbors.items():
        top = sorted(rows, key=lambda row: (-row[1], row[0]))[:max_neighbors]
        for (left, _), (right, _) in combinations(top, 2):
            triple = tuple(sorted((center, left, right)))
            if len(set(triple)) != 3:
                continue
            edges = []
            anchors: set[str] = set()
            weight = 0.0
            for a, b in combinations(triple, 2):
                edge_terms = pairs.get(tuple(sorted((a, b))))
                if not edge_terms:
                    continue
                edges.append([a, b])
                anchors.update(edge_terms)
                weight += pair_weight(edge_terms, count, df)
            if len(edges) < 2:
                continue
            row = {
                "paper_ids": list(triple),
                "anchors": sorted(anchors),
                "connected_edges": edges,
                "score": round(weight + len(anchors), 6),
            }
            prior = candidates.get(triple)
            if prior is None or row["score"] > prior["score"]:
                candidates[triple] = row
    return sorted(
        candidates.values(),
        key=lambda row: (-row["score"], -len(row["anchors"]), row["paper_ids"]),
    )


def select_triples(
    ranked: list[dict[str, Any]], target: int, max_reuse: int
) -> list[dict[str, Any]]:
    usage: Counter[str] = Counter()
    selected = []
    for row in ranked:
        if any(usage[paper_id] >= max_reuse for paper_id in row["paper_ids"]):
            continue
        selected.append(row)
        usage.update(row["paper_ids"])
        if len(selected) >= target:
            break
    return selected


def choose_qas(
    paper: dict[str, Any], anchors: set[str], df: dict[str, int], limit: int
) -> list[dict[str, Any]]:
    scored = []
    for qa in paper["qas"]:
        overlap = qa["terms"] & anchors
        score = sum(1 / max(1, df.get(term, 1)) for term in overlap)
        scored.append((len(overlap), score, qa["qa_id"], qa))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    overlapping = [row[3] for row in scored if row[0] > 0]
    if overlapping:
        return overlapping[:limit]
    # A title-qualified pair can use model/component names absent from the
    # question wording. Preserve deterministic QA context; semantic source
    # grouping still decides whether the bundle is eligible for generation.
    return [row[3] for row in scored[:limit]]


def build_plan(
    source: dict[str, Any],
    pdf_dir: Path,
    selected: list[dict[str, Any]],
    df: dict[str, int],
    qas_per_paper: int,
    validated_papers: dict[str, dict[str, Any]] | None = None,
    bundle_prefix: str = "hxb",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    papers = validated_papers or paper_rows(source, pdf_dir)
    output_source: dict[str, Any] = {}
    contexts: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    page_count_cache: dict[str, int] = {}
    for index, row in enumerate(selected, start=1):
        bundle_id = f"{bundle_prefix}_{index:04d}"
        starts_at = 1
        source_rows = []
        source_qas = []
        anchors = set(row["anchors"])
        for doc_number, paper_id in enumerate(row["paper_ids"], start=1):
            pdf_path = resolve_pdf_path(paper_id, [pdf_dir])
            if not pdf_path.is_file():
                raise FileNotFoundError(pdf_path)
            page_count = page_count_cache.setdefault(
                paper_id, len(PdfReader(str(pdf_path)).pages)
            )
            end_page = starts_at + page_count - 1
            source_rows.append(
                {
                    "paper_id": paper_id,
                    "title": papers[paper_id].get("title") or f"Paper {paper_id}",
                    "source_pdf_path": str(pdf_path.resolve()),
                    "source_page_count": page_count,
                    "merged_start_page": starts_at,
                    "merged_end_page": end_page,
                    "primary_category": papers[paper_id]["primary_category"],
                    "secondary_category": papers[paper_id]["secondary_category"],
                }
            )
            for qa in choose_qas(papers[paper_id], anchors, df, qas_per_paper):
                source_qas.append(
                    {
                        "source_qa_id": f"base/{paper_id}/{qa['qa_id']}",
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "evidence_pages": [starts_at + page - 1 for page in qa["evidence_pages"]],
                        "audit_decision": "BASE_CORRECT",
                    }
                )
            starts_at = end_page + 1
        manifest_row = {
            "id": bundle_id,
            "paper": bundle_id,
            "primary_category": papers[row["paper_ids"][0]]["primary_category"],
            "secondary_category": papers[row["paper_ids"][0]]["secondary_category"],
            "page_count": starts_at - 1,
            "sources": source_rows,
        }
        output_source[bundle_id] = {
            "paper": bundle_id,
            "primary_category": manifest_row["primary_category"],
            "secondary_category": manifest_row["secondary_category"],
            "QA": {},
        }
        contexts[bundle_id] = {
            "manifest": manifest_row,
            "source_qas": source_qas,
            "prior_bundle_summary": (
                "Related papers selected within one primary category using shared "
                "high-IDF scientific anchors: " + ", ".join(row["anchors"][:12])
            ),
        }
        manifest.append(manifest_row)
        ledger.append({"bundle_id": bundle_id, **row, "source_qa_count": len(source_qas)})
    return output_source, {"contexts": contexts}, manifest, ledger


def materialize_bundle_pdfs(
    manifest: list[dict[str, Any]], output_dir: Path
) -> dict[str, int]:
    """Create deterministic merged PDFs and verify their physical page counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    page_counts: dict[str, int] = {}
    for bundle in manifest:
        bundle_id = str(bundle["id"])
        destination = output_pdf_path(output_dir, bundle_id)
        temporary = destination.with_name(f".{destination.name}.{time.time_ns()}.tmp")
        writer = PdfWriter()
        for source in bundle["sources"]:
            reader = PdfReader(str(source["source_pdf_path"]))
            if len(reader.pages) != int(source["source_page_count"]):
                raise ValueError(
                    f"source page count changed for {source['source_pdf_path']}: "
                    f"expected {source['source_page_count']}, got {len(reader.pages)}"
                )
            for page in reader.pages:
                writer.add_page(page)
        with temporary.open("wb") as handle:
            writer.write(handle)
        actual_pages = len(PdfReader(str(temporary)).pages)
        expected_pages = int(bundle["page_count"])
        if actual_pages != expected_pages:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"merged page count mismatch for {bundle_id}: "
                f"expected {expected_pages}, got {actual_pages}"
            )
        temporary.replace(destination)
        page_counts[bundle_id] = actual_pages
    return page_counts


def main() -> None:
    args = parse_args()
    if args.target_bundles < 1 or args.max_paper_reuse < 1:
        raise SystemExit("bundle target and paper reuse must be positive")
    if args.min_shared_title_terms < 0:
        raise SystemExit("--min-shared-title-terms cannot be negative")
    if args.pair_bundles_only and args.include_pair_bundles:
        raise SystemExit(
            "--pair-bundles-only and --include-pair-bundles are mutually exclusive"
        )
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.bundle_prefix):
        raise SystemExit(
            "--bundle-prefix must contain lowercase letters, digits, or underscores"
        )
    source = read_json(args.source)
    source_audit: dict[str, int] = {}
    papers = paper_rows(source, args.pdf_dir, source_audit)
    pairs, df = related_pairs(
        papers,
        args.max_term_paper_frequency,
        args.min_shared_terms,
        args.category_level,
        args.min_shared_title_terms,
    )
    triples = ranked_triples(papers, pairs, df, args.max_neighbors_per_paper)
    if args.require_complete_triangle:
        triples = [
            row
            for row in triples
            if complete_triangle(row, pairs)
        ]
    if args.pair_bundles_only:
        ranked_bundles = ranked_pair_bundles(papers, pairs, df)
    else:
        ranked_bundles = list(triples)
    if args.include_pair_bundles:
        ranked_bundles.extend(ranked_pair_bundles(papers, pairs, df))
    selected = select_triples(
        ranked_bundles, args.target_bundles, args.max_paper_reuse
    )
    if len(selected) < args.target_bundles:
        raise SystemExit(
            f"insufficient global bundles: selected={len(selected)}/{args.target_bundles}"
        )
    output_source, contexts, manifest, ledger = build_plan(
        source,
        args.pdf_dir,
        selected,
        df,
        args.qas_per_paper,
        papers,
        args.bundle_prefix,
    )
    materialized_page_counts = None
    if args.materialize_pdf_dir is not None:
        materialized_page_counts = materialize_bundle_pdfs(
            manifest, args.materialize_pdf_dir
        )
    atomic_json(args.output_dir / "source_dataset.json", output_source)
    atomic_json(args.output_dir / "prepared_contexts.json", contexts)
    atomic_json(args.output_dir / "bundle_manifest.json", manifest)
    atomic_json(args.output_dir / "selection_ledger.json", ledger)
    summary = {
        "status": "complete",
        "base_papers_with_at_least_two_qa": len(papers),
        "source_evidence_validation": source_audit,
        "related_pair_count": len(pairs),
        "candidate_triple_count": len(triples),
        "candidate_pair_bundle_count": len(pairs),
        "selected_bundle_count": len(selected),
        "selected_three_document_bundle_count": sum(
            len(row["paper_ids"]) == 3 for row in selected
        ),
        "selected_two_document_bundle_count": sum(
            len(row["paper_ids"]) == 2 for row in selected
        ),
        "bundle_prefix": args.bundle_prefix,
        "three_document_ratio": (
            sum(len(row["paper_ids"]) == 3 for row in selected) / len(selected)
        ),
        "max_paper_reuse": args.max_paper_reuse,
        "category_level": args.category_level,
        "min_shared_title_terms": args.min_shared_title_terms,
        "pair_bundles_only": args.pair_bundles_only,
        "require_complete_triangle": args.require_complete_triangle,
        "qwen_pdf_input": False,
        "materialized_pdf_dir": (
            str(args.materialize_pdf_dir) if args.materialize_pdf_dir is not None else None
        ),
        "materialized_pdf_count": (
            len(materialized_page_counts) if materialized_page_counts is not None else 0
        ),
        "outputs": {
            "source": str(args.output_dir / "source_dataset.json"),
            "contexts": str(args.output_dir / "prepared_contexts.json"),
            "manifest": str(args.output_dir / "bundle_manifest.json"),
            "ledger": str(args.output_dir / "selection_ledger.json"),
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
