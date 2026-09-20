#!/usr/bin/env python3
"""Create a reproducible baseline-versus-v2 data-quality summary."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from pku_qa.workflows.generation.build_cross_pdf_quality_v2 import mapped_docs


ROOT = Path(__file__).resolve().parents[4]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prefix(question: str) -> str:
    words = question.casefold().split()
    return " ".join(words[:2]).strip(" ,.:;?!")


def profile(
    qa: dict[str, Any], manifest_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    manifests = {row["id"]: row for row in manifest_rows}
    source_counts: Counter[str] = Counter()
    hops: Counter[int] = Counter()
    modalities: Counter[str] = Counter()
    prefixes: Counter[str] = Counter()
    qa_per_pdf = []
    page_counts = []
    spans = []
    evidence_items_complete = 0
    total_qas = 0
    for bundle_id, bundle in qa.items():
        manifest = manifests[bundle_id]
        page_counts.append(int(manifest["page_count"]))
        qa_per_pdf.append(len(bundle["QA"]))
        source_counts.update(
            str(source["paper_id"]) for source in manifest["sources"]
        )
        for item in bundle["QA"].values():
            total_qas += 1
            pages = sorted(set(item.get("evidence_pages", [])))
            hops[len(mapped_docs(pages, manifest["sources"]))] += 1
            modalities.update(item.get("modal_types", []))
            prefixes[prefix(item["question"])] += 1
            if pages:
                spans.append(max(pages) - min(pages))
            evidence_item_pages = {
                row.get("physical_pdf_page")
                for row in item.get("evidence_items", [])
                if isinstance(row, dict)
            }
            if pages and set(pages).issubset(evidence_item_pages):
                evidence_items_complete += 1
    templated = sum(
        count
        for question_prefix, count in prefixes.items()
        if question_prefix in {"how do", "how does", "compare the", "what is"}
    )
    return {
        "bundles": len(qa),
        "qas": total_qas,
        "source_references": sum(source_counts.values()),
        "unique_source_papers": len(source_counts),
        "max_source_reuse": max(source_counts.values(), default=0),
        "source_reuse_histogram": dict(
            sorted(Counter(source_counts.values()).items())
        ),
        "physical_pages_total": sum(page_counts),
        "physical_pages_per_pdf": {
            "min": min(page_counts, default=0),
            "median": statistics.median(page_counts) if page_counts else 0,
            "max": max(page_counts, default=0),
        },
        "qas_per_pdf": {
            "min": min(qa_per_pdf, default=0),
            "median": statistics.median(qa_per_pdf) if qa_per_pdf else 0,
            "max": max(qa_per_pdf, default=0),
        },
        "evidence_hops": dict(sorted(hops.items())),
        "evidence_span_median": (
            statistics.median(spans) if spans else 0
        ),
        "modalities": dict(modalities),
        "top_question_prefixes": prefixes.most_common(15),
        "fixed_template_prefix_count": templated,
        "fixed_template_prefix_rate": templated / total_qas if total_qas else 0,
        "evidence_items_complete": evidence_items_complete,
        "evidence_items_complete_rate": (
            evidence_items_complete / total_qas if total_qas else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-qa",
        type=Path,
        default=ROOT / "data/qa/qa_cross_pdf_semantic_verified_20260724.json",
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=(
            ROOT
            / "data/qa/4.cross_pdf/semantic_reaudit/final_release/"
            "selected_bundle_manifest.json"
        ),
    )
    parser.add_argument(
        "--final-qa",
        type=Path,
        default=ROOT / "data/qa/4.cross_pdf/quality/qa_cross_pdf_challenge_v2.json",
    )
    parser.add_argument(
        "--final-manifest",
        type=Path,
        default=(
            ROOT
            / "data/qa/4.cross_pdf/quality/"
            "selected_bundle_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/qa/4.cross_pdf/quality",
    )
    args = parser.parse_args()
    baseline = profile(
        load_json(args.baseline_qa), load_json(args.baseline_manifest)
    )
    final = profile(
        load_json(args.final_qa), load_json(args.final_manifest)
    )
    summary = {
        "page_numbering_definition": (
            "evidence_pages are 1-based physical pages in the merged PDF; "
            "printed page labels are not used for scoring"
        ),
        "baseline": baseline,
        "final": final,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "quality_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = f"""# 多文档 PDF QA Challenge v2：质量改造技术总结

## 技术结论

新版不是对 369 条旧题的宽松扩充，而是零源论文复用的高难核心集。它只保留
能够构造真实三文档问题、经独立模型逐页复审、并通过结构门禁的拼接 PDF。
`evidence_pages` 统一表示拼接 PDF 中从 1 开始计数的物理页，不使用论文页脚的
印刷页码。

## 核心指标

| 指标 | 旧发布集 | Challenge v2 |
| --- | ---: | ---: |
| 拼接 PDF | {baseline['bundles']} | {final['bundles']} |
| QA | {baseline['qas']} | {final['qas']} |
| 源论文出现次数 | {baseline['source_references']} | {final['source_references']} |
| 不重复源论文 | {baseline['unique_source_papers']} | {final['unique_source_papers']} |
| 单篇源论文最大复用 | {baseline['max_source_reuse']} | {final['max_source_reuse']} |
| 真正三文档 QA | {baseline['evidence_hops'].get(3, 0)} | {final['evidence_hops'].get(3, 0)} |
| 证据事实字段完整率 | {baseline['evidence_items_complete_rate']:.1%} | {final['evidence_items_complete_rate']:.1%} |
| 四类固定开头占比 | {baseline['fixed_template_prefix_rate']:.1%} | {final['fixed_template_prefix_rate']:.1%} |

## 改造流程

1. 用整数规划选择候选组合，并要求同一源论文最多出现一次。
2. Gemini 读取三篇完整论文，生成必须同时依赖 Doc 1、Doc 2、Doc 3 的候选题，
   同时改写部分模板化旧问题。
3. Claude 在不知道生成结论是否可信的前提下重新阅读完整论文，逐条检查答案、
   三文档必要性、证据充分性和模板多样性。
4. 只接受全部质量布尔项为真的 `KEEP/FIX`；支持事实必须覆盖三篇论文。
5. 使用复审支持事实重建物理页证据，不沿用存在争议的旧页码。
6. 再次求解零重叠最大子集，并验证 PDF 页数、证据范围、问题唯一性、文件哈希、
   字段完整性及源论文复用。

## 页码口径

若 PDF 文件第 3 张物理页的页脚印着“1”，答案句出现在这一张，则标注和评分均
使用 `3`。源论文内页码另存为 `evidence_items[].source_page`，仅用于人工查阅，
不会替代 `evidence_pages`。页码评分要求模型给出的排序去重页集合与标准集合
完全一致。

## 限制

- Challenge v2 为高精度小规模核心集，不替代旧 369 条覆盖型发布集。
- 模型复审仍不是领域专家人工金标准，因此网页审核结果会作为后续人工门禁。
- 三文档题越严格，能够通过的组合越少；流程选择宁缺毋滥，不以降低门槛凑数。
"""
    (args.output_dir / "REPORT.md").write_text(
        markdown, encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
