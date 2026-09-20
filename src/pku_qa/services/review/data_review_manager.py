#!/usr/bin/env python3
"""Durable, reversible manual review for local QA JSON datasets."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from pku_qa.pdf_assets import resolve_pdf_path


ROOT = Path(__file__).resolve().parents[4]
QA_DIR = ROOT / "data/qa"
PDF_ROOT = ROOT / "data/pdfs"
STATE_DIR = ROOT / "data/web/review"
DEFAULT_DB = STATE_DIR / "review.sqlite3"
SNAPSHOT_DIR = STATE_DIR / "snapshots"
FINAL_2200_MANIFEST = ROOT / "data/qa/7.final_2200/rel__collection__final_2200__manifest.json"
_MUTATION_LOCK = threading.RLock()
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14
PASSWORD_MIN_LENGTH = 10
BOOTSTRAP_ADMIN_USERNAME = "czj-web"
ACCOUNT_ROLES = ("admin", "reviewer")
EMAIL_CODE_TTL_SECONDS = 10 * 60
EMAIL_CODE_RESEND_SECONDS = 60
EMAIL_CODE_MAX_ATTEMPTS = 6

# Main reviewable datasets live in purpose-specific directories.  The base
# directory remains configurable for tests and one-off review sessions.
PROJECT_DATASETS: dict[str, Path] = {
    "ordinary_qa.json": ROOT / "data/qa/7.final_2200/ordinary_qa.json",
    "unanswerable_qa.json": ROOT / "data/qa/7.final_2200/unanswerable_qa.json",
    "reasoning_qa.json": ROOT / "data/qa/7.final_2200/reasoning_qa.json",
    "cross_pdf_qa.json": ROOT / "data/qa/7.final_2200/cross_pdf_qa.json",
    "rel__human_reviewed__authority__batch01__n983.json": ROOT / "data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json",
    "work__reasoning__historical_clean__batch00__n100.json": (
        ROOT / "data/qa/3.reasoning/work__reasoning__historical_clean__batch00__n100.json"
    ),
    "rel__reasoning__refreshed__batch01__n100__v1.json": (
        ROOT
        / "data/qa/3.reasoning/"
        "rel__reasoning__refreshed__batch01__n100__v1.json"
    ),
    "rel__reasoning__incremental__batch02__n100__v1.json": (
        ROOT
        / "data/qa/3.reasoning/hard_expansion/"
        "rel__reasoning__incremental__batch02__n100__v1.json"
    ),
}

FINAL_2200_DATASET_IDS = (
    "ordinary_qa.json",
    "unanswerable_qa.json",
    "reasoning_qa.json",
    "cross_pdf_qa.json",
)


DATASET_CATALOG: dict[str, dict[str, str]] = {
    "rel__human_reviewed__authority__batch01__n983.json": {
        "title": "历史人工审核来源 983",
        "status": "历史构建来源",
        "role": "人工审核与同步来源",
        "description": "保留供历史单 PDF 同步工作流使用；不代表最终 2,200 条的独立发布输入。",
        "lineage": "最终四文件形成前的人工审核来源记录。",
        "caution": "当前发表、人工终审和评测使用 7.final_2200/ 四文件。",
    },
    "ordinary_qa.json": {
        "title": "最终普通题 1000",
        "status": "发表前最终人工校验",
        "role": "普通可回答 QA",
        "description": "最终 2,200 条中的 1,000 条普通可回答单论文 QA。",
        "lineage": "由已完成人工终审的单论文正式组件按精确 Unanswerable 规则拆分并统一全局 QA ID。",
        "caution": "这是 final_2200 的规范文件；修改会直接写回并刷新四文件 manifest。",
        "collection_id": "final_2200",
        "collection_title": "发表前最终人工校验（2200 条）",
    },
    "unanswerable_qa.json": {
        "title": "最终不可回答题 200",
        "status": "发表前最终人工校验",
        "role": "不可回答 QA",
        "description": "最终 2,200 条中答案精确为 Unanswerable、证据页为空的 200 条单论文 QA。",
        "lineage": "由已完成人工终审的单论文正式组件按精确 Unanswerable 规则拆分并统一全局 QA ID。",
        "caution": "不可回答题只允许精确字符串 Unanswerable，且 evidence_pages 必须为 []。",
        "collection_id": "final_2200",
        "collection_title": "发表前最终人工校验（2200 条）",
    },
    "reasoning_qa.json": {
        "title": "最终 Reasoning 200",
        "status": "发表前最终人工校验",
        "role": "单论文高难推理",
        "description": "最终 2,200 条中的 200 条单论文多步推理 QA。",
        "lineage": "合并两批各 100 条的严格双审与难度校准结果，并统一全局 QA ID。",
        "caution": "题目必须保持真实多步推理依赖，证据页使用 1-based PDF 物理页。",
        "collection_id": "final_2200",
        "collection_title": "发表前最终人工校验（2200 条）",
    },
    "cross_pdf_qa.json": {
        "title": "最终 Cross-PDF 800",
        "status": "发表前最终人工校验",
        "role": "跨论文多文档推理",
        "description": "最终 2,200 条中的 800 条 Cross-PDF QA，使用规范 z_cross_ PDF ID。",
        "lineage": "合并两批各 400 条的严格复审结果，并统一全局 QA ID。",
        "caution": "题目必须保持多文档必要性，证据页使用合并 PDF 的 1-based 物理页。",
        "collection_id": "final_2200",
        "collection_title": "发表前最终人工校验（2200 条）",
    },
    "rel__single_pdf__ordinary__batch01__n1000.json": {
        "title": "最终普通题分类镜像 1000",
        "status": "最终集只读镜像",
        "role": "普通单 PDF QA",
        "description": "按用途归入 base 的最终普通题镜像；内容必须与 ordinary_qa.json 完全一致。",
        "lineage": "从 data/qa/7.final_2200/ordinary_qa.json 按字节同步。",
        "caution": "最终文件是唯一权威来源；请勿单独修改此镜像。",
        "reviewable": False,
        "read_only_reason": "只读镜像；修改必须在 7.final_2200/ordinary_qa.json 中完成。",
    },
    "data/qa/2.unanswerable/rel__single_pdf__unanswerable__batch01__n200.json": {
        "title": "最终不可回答题分类镜像 200",
        "status": "最终集只读镜像",
        "role": "不可回答单 PDF QA",
        "description": "单独归档的最终不可回答题镜像；内容必须与 unanswerable_qa.json 完全一致。",
        "lineage": "从 data/qa/7.final_2200/unanswerable_qa.json 按字节同步。",
        "caution": "最终文件是唯一权威来源；请勿单独修改此镜像。",
        "reviewable": False,
        "read_only_reason": "只读镜像；修改必须在 7.final_2200/unanswerable_qa.json 中完成。",
    },
    "data/qa/3.reasoning/rel__reasoning__refreshed__batch01__n100__claude_hard__v1.json": {
        "title": "迁移前 Reasoning 第一批 100",
        "status": "历史生成谱系",
        "role": "五文件迁移前源组件",
        "description": "保留用于追溯 reasoning_qa.json 的第一批来源。",
        "lineage": "reasoning_qa.json 的迁移前第一批组件。",
        "caution": "四文件迁移已完成；本文件只读。",
        "reviewable": False,
        "read_only_reason": "迁移前源组件仅用于生成谱系；正式审核使用 reasoning_qa.json。",
    },
    "data/qa/3.reasoning/hard_expansion/rel__reasoning__incremental__batch02__n100__claude_hard__v1.json": {
        "title": "迁移前 Reasoning 第二批 100",
        "status": "历史生成谱系",
        "role": "五文件迁移前源组件",
        "description": "保留用于追溯 reasoning_qa.json 的第二批来源。",
        "lineage": "reasoning_qa.json 的迁移前第二批组件。",
        "caution": "四文件迁移已完成；本文件只读。",
        "reviewable": False,
        "read_only_reason": "迁移前源组件仅用于生成谱系；正式审核使用 reasoning_qa.json。",
    },
    "data/qa/4.cross_pdf/challenge/rel__cross_pdf__challenge__batch01__n400__v1.json": {
        "title": "迁移前 Cross-PDF 第一批 400",
        "status": "历史生成谱系",
        "role": "五文件迁移前源组件",
        "description": "保留用于追溯 cross_pdf_qa.json 的第一批来源。",
        "lineage": "cross_pdf_qa.json 的迁移前第一批组件。",
        "caution": "四文件迁移已完成；本文件只读。",
        "reviewable": False,
        "read_only_reason": "迁移前源组件仅用于生成谱系；正式审核使用 cross_pdf_qa.json。",
    },
    "data/qa/4.cross_pdf/hard_expansion/rel__cross_pdf__strict_dual_review__batch02__n400__v1.json": {
        "title": "迁移前 Cross-PDF 第二批 400",
        "status": "历史生成谱系",
        "role": "五文件迁移前源组件",
        "description": "保留用于追溯 cross_pdf_qa.json 的第二批来源。",
        "lineage": "cross_pdf_qa.json 的迁移前第二批组件。",
        "caution": "四文件迁移已完成；本文件只读。",
        "reviewable": False,
        "read_only_reason": "迁移前源组件仅用于生成谱系；正式审核使用 cross_pdf_qa.json。",
    },
    "work__reasoning__historical_clean__batch00__n100.json": {
        "title": "历史 Reasoning 100",
        "status": "历史正式输入",
        "role": "结果追溯",
        "description": "刷新前的第一批 Reasoning 100；其中 98 条后来经 PDF 依赖审计替换。",
        "lineage": "旧 Reasoning 正式输入，保留用于复现实验和核对历史结果。",
        "caution": "当前发布与评测使用 7.final_2200/reasoning_qa.json；本文件仅供历史追溯。",
    },
    "rel__reasoning__refreshed__batch01__n100__v1.json": {
        "title": "Reasoning 第一批（刷新 100）",
        "status": "历史构建来源",
        "role": "单论文高难推理",
        "description": "对旧第一批替换 98 条并保留 2 条后的历史 Reasoning 100，已完成 PDF 依赖和证据页审计。",
        "lineage": "由旧 Reasoning 100 与严格双审候选按 209 条选择池确定性刷新。",
        "caution": "证据页对应 data/pdfs 中由 PDF 清单映射的 paper_* 论文 PDF 物理页。",
    },
    "rel__reasoning__incremental__batch02__n100__v1.json": {
        "title": "Reasoning 第二批（新增 100）",
        "status": "历史构建来源",
        "role": "单论文高难推理扩展",
        "description": "严格 Claude/Gemini 双 KEEP、证据恢复和 4B/8B 闭卷依赖筛选后的新增 Reasoning 100。",
        "lineage": "从同一 209 条严格候选池选择且不与刷新第一批重复。",
        "caution": "证据页对应 data/pdfs 中由 PDF 清单映射的 paper_* 论文 PDF 物理页。",
    },
    "work__single_pdf__raw_mixed__n6204.json": {
        "title": "原始 6204 题混合题型集",
        "status": "原始数据",
        "role": "溯源基线",
        "description": "703 篇论文的原始混合题型数据，包含选择题与简答题；尚未经过后续标准集的清洗和筛选。",
        "lineage": "最上游原始版本，是 option-shuffled 和 4211 标准集的来源。",
        "caution": "适合追溯和对照，不建议直接作为当前正式评测输入。",
    },
    "work__single_pdf__raw_mixed_option_shuffled__n6204.json": {
        "title": "原始 6204 题选项重排版",
        "status": "原始派生",
        "role": "偏差对照",
        "description": "保持原始 6204 道题内容不变，对选择题选项顺序进行重排，用于检查和降低选项位置偏差。",
        "lineage": "由 work__single_pdf__raw_mixed__n6204.json 派生。",
        "caution": "仍保留原始数据质量问题，不是清洗后的正式集。",
    },
    "rel__single_pdf__mixed__n4211__v1.json": {
        "title": "4211 题清洗混合标准集",
        "status": "历史构建基线",
        "role": "混合题型评测",
        "description": "PDF 评测清洗后的 4211 题标准集，保留选择题和简答题两类格式。",
        "lineage": "由原始 6204 题清洗得到，也是全简答 4211 集的直接来源。",
        "caution": "与全简答版本比较结果时，应注意选择题和简答题计分方式不同。",
    },
    "rel__single_pdf__short_answer__n4211__v1.json": {
        "title": "4211 题全简答正式集",
        "status": "历史构建基线",
        "role": "上游简答基线",
        "description": "将清洗混合集中的选择题全部转换为简答题，统一使用短答案生成与判卷。",
        "lineage": "由 rel__single_pdf__mixed__n4211__v1.json 转换得到；用于构建谱系；当前正式评测使用 7.final_2200/ 四文件。",
        "caution": "证据页使用从 1 开始的 PDF 物理页码，不使用论文页脚印刷页码。",
    },
    "work__single_pdf__short_answer_filtered__n4154.json": {
        "title": "4154 题去除闭卷易题集",
        "status": "过滤版本",
        "role": "难度控制",
        "description": "从全简答 4211 集移除 57 道闭卷与带 PDF 条件下 4B、8B 均答对的题，降低常识泄漏和过易题比例。",
        "lineage": "由 rel__single_pdf__short_answer__n4211__v1.json 过滤得到。",
        "caution": "仍覆盖 693 篇论文，但不包含额外不可回答题。",
    },
    "work__single_pdf__short_answer_balanced_hard__n3000.json": {
        "title": "3000 题均衡困难候选集",
        "status": "发布候选",
        "role": "多模型对比",
        "description": "在领域和题型尽量均衡的前提下优先保留小模型错题、推理题、跨页题和多模态题。",
        "lineage": "从 4154 题去除闭卷易题集筛选；保留全部 2006 道至少一个小模型答错的题。",
        "caution": "这是难度导向的子集，不能用来估计原始数据的自然分布。",
    },
    "rel__single_pdf__short_answer_unanswerable__n4451__v1.json": {
        "title": "4451 题全简答 Hard 集",
        "status": "困难派生",
        "role": "拒答能力评测",
        "description": "在全简答 4211 题基础上追加 240 道不可回答题，用于同时考察作答能力和正确拒答能力。",
        "lineage": "rel__single_pdf__short_answer__n4211__v1.json + work__single_pdf__unanswerable_pool__n240.json。",
        "caution": "报告结果时应分别给出可回答题正确率、不可回答召回率和错误拒答率。",
    },
    "rel__single_pdf__short_answer_unanswerable_filtered__n4394__v1.json": {
        "title": "4394 题过滤 Hard 集",
        "status": "困难候选",
        "role": "高难综合评测",
        "description": "Hard 集移除 57 道闭卷和带 PDF 均容易的问题，保留额外不可回答题。",
        "lineage": "由 rel__single_pdf__short_answer_unanswerable__n4451__v1.json 按双模型易题键过滤。",
        "caution": "同时混合可回答和不可回答题，不能只报告一个总体正确率。",
    },
    "work__single_pdf__unanswerable_pool__n240.json": {
        "title": "240 道额外不可回答题",
        "status": "辅助题源",
        "role": "拒答压力测试",
        "description": "40 篇论文上的额外不可回答题，用于构建 Hard 数据集并评测模型是否会编造答案。",
        "lineage": "独立辅助题源，合并进入 rel__single_pdf__short_answer_unanswerable__n4451__v1.json。",
        "caution": "不应单独当作普通可回答 QA 集统计答案正确率。",
    },
}


FIELD_DESCRIPTIONS = {
    "paper": "论文或拼接 PDF 的标识符。",
    "QA": "以 QA ID 为键的问题集合。",
    "primary_category": "论文所属一级领域。",
    "primary category": "历史格式的一级领域字段。",
    "secondary_category": "论文所属二级领域。",
    "secondary category": "历史格式的二级领域字段。",
    "pdf_page_numbering": "证据页采用的页码口径。",
    "source_documents": "拼接 PDF 中各来源论文的范围与元数据。",
    "source_paper_ids": "该记录涉及的去重源论文 ID。",
    "question": "向被测试模型提出的问题。",
    "answer": "用于判卷的标准答案；不可回答题必须精确为 Unanswerable。",
    "evidence_pages": "支持答案的 PDF 物理页集合，从 1 开始计数。",
    "modal_types": "解题所需模态，如 text、table、image、formula。",
    "question_type": "作答形式或推理类型的历史分类字段。",
    "question_category": "问题的语义或推理类别。",
    "answer_format": "期望答案的数据格式，如整数、浮点、字符串或列表。",
    "answer_aliases": "可接受的等价答案表达。",
    "answer_unit": "数值答案的单位。",
    "numeric_tolerance": "数值答案规则判分允许的误差。",
    "evidence_items": "逐证据页保存的来源映射和具体支持事实。",
    "evidence_hops": "回答问题需要整合的必要证据来源或推理跳数。",
    "evidence_span": "最远与最近证据物理页之间的跨度。",
    "evidence_page_numbering": "QA 级证据页编号规范。",
    "evidence_source_docs": "标准证据覆盖的来源文档编号。",
    "qa_uid": "跨文件或流水线使用的稳定 QA 唯一标识。",
    "question_style": "问题当前的语言表达风格。",
    "question_style_normalization": "模板化问题改写和风格去重记录。",
    "annotation_provenance": "生成、复审、修正和发布来源记录。",
    "review_status": "该 QA 的模型或人工复审状态。",
}


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        item_types = sorted({value_type(item) for item in value[:100]})
        suffix = "|".join(item_types) if item_types else "empty"
        return f"array<{suffix}>"
    return type(value).__name__


def field_profile(
    records: list[dict[str, Any]], total: int
) -> list[dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for record in records:
        for name, value in record.items():
            row = fields.setdefault(name, {"present": 0, "types": set()})
            row["present"] += 1
            row["types"].add(value_type(value))
    return [
        {
            "name": name,
            "description": FIELD_DESCRIPTIONS.get(
                name, "数据集自定义字段；请结合生成或标注流程解释。"
            ),
            "present": values["present"],
            "total": total,
            "coverage_percent": round(
                100.0 * values["present"] / total, 1
            ) if total else 0.0,
            "value_types": sorted(values["types"]),
        }
        for name, values in sorted(fields.items())
    ]


def dataset_profile(
    dataset_id: str, dataset: dict[str, Any], *, path: Path | None = None
) -> dict[str, Any]:
    paper_records = [
        paper for paper in dataset.values() if isinstance(paper, dict)
    ]
    qa_records = [
        qa
        for paper in paper_records
        for qa in (paper.get("QA", {}) or {}).values()
        if isinstance(qa, dict)
    ]
    qas = len(qa_records)
    catalog = dict(DATASET_CATALOG.get(dataset_id, _inferred_catalog(dataset_id, path)))
    catalog = {
        "collection_id": "other",
        "collection_title": "其他数据与历史版本",
        **catalog,
    }
    if path is not None:
        try:
            relative_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            relative_path = dataset_id
    elif dataset_id.startswith("data/"):
        relative_path = dataset_id
    else:
        relative_path = f"data/qa/{dataset_id}"
    catalog.setdefault("source_summary", catalog.get("lineage", ""))
    catalog.setdefault("acquisition", catalog.get("lineage", ""))
    catalog.setdefault("intended_use", catalog.get("description", ""))
    catalog.setdefault("artifact_class", catalog.get("role", "QA 数据集"))
    catalog.setdefault("reviewable", True)
    catalog.setdefault("read_only_reason", None)
    catalog["path"] = str(path.resolve()) if path is not None else str(ROOT / relative_path)
    catalog["relative_path"] = relative_path
    question_types = Counter(
        str(qa.get("question_type", "未标注")) for qa in qa_records
    )
    modalities = Counter(
        str(modality)
        for qa in qa_records
        for modality in (
            qa.get("modal_types")
            if isinstance(qa.get("modal_types"), list)
            else []
        )
    )
    def valid_physical_pages(qa: dict[str, Any]) -> set[int]:
        raw_pages = qa.get("evidence_pages")
        if not isinstance(raw_pages, list):
            return set()
        pages = set()
        for page in raw_pages:
            if isinstance(page, bool):
                continue
            if isinstance(page, int) and page >= 1:
                pages.add(page)
            elif isinstance(page, str) and page.strip().isdigit():
                parsed = int(page.strip())
                if parsed >= 1:
                    pages.add(parsed)
        return pages

    evidence_page_sets = [valid_physical_pages(qa) for qa in qa_records]
    with_evidence = sum(bool(pages) for pages in evidence_page_sets)
    multi_page = sum(len(pages) >= 2 for pages in evidence_page_sets)
    unanswerable = sum(
        qa.get("answer") == "Unanswerable" for qa in qa_records
    )
    rich_evidence = sum(
        isinstance(qa.get("evidence_items"), list)
        and bool(qa["evidence_items"])
        for qa in qa_records
    )
    page_numbering = (
        "PDF 文件物理页，从 1 开始；论文印刷页码不参与评分。"
        if with_evidence
        else "该数据集未稳定提供非空 evidence_pages。"
    )
    return {
        **catalog,
        "page_numbering": page_numbering,
        "statistics": {
            "avg_qas_per_paper": round(qas / len(paper_records), 2)
            if paper_records else 0,
            "short_answer": qas,
            "with_evidence_pages": with_evidence,
            "multi_page_evidence": multi_page,
            "unanswerable": unanswerable,
            "with_evidence_items": rich_evidence,
            "question_types": dict(question_types.most_common()),
            "modalities": dict(modalities.most_common()),
        },
        "paper_fields": field_profile(paper_records, len(paper_records)),
        "qa_fields": field_profile(qa_records, qas),
    }


def _inferred_catalog(dataset_id: str, path: Path | None = None) -> dict[str, Any]:
    """Return a complete, deterministic description for discovered QA files.

    The registry intentionally discovers nested and historical QA-shaped files.  Most
    of those are process artifacts rather than safe review inputs, so their purpose and
    read-only status are derived from the path instead of being left undocumented.
    """
    if path is not None:
        try:
            raw_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            raw_path = str(path)
    else:
        raw_path = dataset_id
    raw_path = raw_path.replace("\\", "/")
    parts = set(raw_path.lower().split("/"))
    name = Path(dataset_id).name
    lower_name = name.lower()
    process_markers = {
        "output", "outputs", "checkpoint", "checkpoints", "retry_split",
        "review", "dual_review", "pilot", "calibration", "runtime",
    }
    read_only = bool(parts & process_markers) or any(
        marker in lower_name
        for marker in ("summary", "manifest", "progress", "ledger", "validation", "response")
    )
    if "raw" in parts or "expansion" in parts or "candidate" in lower_name:
        artifact_class = "候选/原始 QA 数据"
        status = "过程候选"
        purpose = "作为后续清洗、复审或正式集构建的输入，不代表已发布质量。"
    elif read_only:
        artifact_class = "审核/运行过程 QA 材料"
        status = "过程产物（只读）"
        purpose = "保存生成、模型复审、审核或运行过程中的 QA 快照，用于追溯和复现。"
    elif name.startswith("rel__"):
        artifact_class = "正式/当前 QA 数据集"
        status = "正式数据"
        purpose = "作为当前评测、人工校验或论文发布前的数据输入。"
    else:
        artifact_class = "工作数据集"
        status = "工作版本"
        purpose = "供数据处理流程和人工检查使用，后续可能被筛选或替换。"
    source = (
        f"由项目数据目录自动发现：{raw_path}。文件内每个顶层记录包含论文或 PDF 标识及其 QA 字段；"
        "未登记为显式正式集的文件，其来源关系以该文件所在目录和 annotation_provenance 为准。"
    )
    reason = (
        "该文件属于模型输出、审核记录、断点或汇总过程材料，网页仅提供查看，避免误改可复现产物。"
        if read_only else "该文件包含可供人工检查的问题、标准答案和证据页；修改会写回原 JSON，并保留快照与审计日志。"
    )
    stem = name.removesuffix(".json")
    if "cross_pdf" in raw_path.lower():
        title_prefix = "Cross-PDF 过程数据"
    elif "reasoning" in raw_path.lower():
        title_prefix = "Reasoning 过程数据"
    elif "challenge" in raw_path.lower():
        title_prefix = "Challenge 过程数据"
    elif "output" in raw_path.lower():
        title_prefix = "模型输出过程数据"
    else:
        title_prefix = "QA 工作数据"
    display_title = f"{title_prefix} · {stem}"
    return {
        "title": display_title,
        "status": status,
        "role": artifact_class,
        "description": purpose,
        "lineage": source,
        "caution": reason,
        "source_summary": source,
        "acquisition": "由项目清洗/生成/复审流水线产生；具体阶段见路径、文件名和文件内 annotation_provenance。",
        "intended_use": purpose,
        "artifact_class": artifact_class,
        "reviewable": not read_only,
        "read_only_reason": reason if read_only else None,
        "collection_id": "other",
        "collection_title": "其他数据与历史版本",
    }


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_dataset(path: Path) -> tuple[dict[str, Any], bytes, str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("QA dataset must be an object keyed by paper id")
    return value, raw, sha256_bytes(raw)


def count_qas(dataset: dict[str, Any]) -> int:
    return sum(
        len(paper.get("QA", {}))
        for paper in dataset.values()
        if isinstance(paper, dict) and isinstance(paper.get("QA"), dict)
    )


def atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".review.tmp")
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    with temporary.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def encode_dataset(dataset: dict[str, Any]) -> bytes:
    return (
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class DatasetInfo:
    id: str
    path: str
    papers: int
    qas: int
    size_bytes: int
    updated_at: float
    sha256: str
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def to_summary_dict(self) -> dict[str, Any]:
        """Return the fields needed by dataset selectors without every field map.

        Full profiles contain per-field coverage for every discovered QA JSON and
        are intentionally loaded only for the dataset currently being reviewed.
        """
        profile_keys = (
            "collection_id",
            "collection_title",
            "title",
            "status",
            "role",
            "reviewable",
            "read_only_reason",
            "path",
            "relative_path",
        )
        profile = {key: self.profile.get(key) for key in profile_keys}
        return {
            "id": self.id,
            "path": self.path,
            "papers": self.papers,
            "qas": self.qas,
            "size_bytes": self.size_bytes,
            "updated_at": self.updated_at,
            "sha256": self.sha256,
            "profile": profile,
        }


class DatasetRegistry:
    LIST_CACHE_TTL_SECONDS = 300.0

    def __init__(self, qa_dir: Path = QA_DIR, pdf_root: Path = PDF_ROOT) -> None:
        self.qa_dir = qa_dir.resolve()
        self.pdf_root = pdf_root.resolve()
        self._cache: dict[str, tuple[int, int, DatasetInfo]] = {}
        self._list_cache: dict[str, tuple[float, tuple[tuple[str, int, int], ...], list[DatasetInfo]]] = {}

    def invalidate(self, dataset_id: str | None = None) -> None:
        """Invalidate catalog/profile caches after a dataset mutation.

        Mutations are serialized by the review store, so clearing the small
        in-memory caches here is enough to make the next catalog request
        observe the newly written JSON without rescanning on every request.
        """
        if dataset_id is None:
            self._cache.clear()
            self._list_cache.clear()
            return
        try:
            path = self.resolve(dataset_id)
            self._cache.pop(str(path.resolve()), None)
        except (KeyError, ValueError, OSError):
            self._cache.clear()
        self._list_cache.clear()

    def _id_for_path(self, path: Path) -> str:
        path = path.resolve()
        if self.qa_dir == QA_DIR.resolve():
            if path in {candidate.resolve() for candidate in PROJECT_DATASETS.values()}:
                return path.name
            if path.parent == (ROOT / "data/qa/1.base").resolve():
                return path.name
            relative = path.relative_to(ROOT.resolve())
            return relative.as_posix()
        relative = path.relative_to(self.qa_dir)
        return relative.name if len(relative.parts) == 1 else relative.as_posix()

    def resolve(self, dataset_id: str) -> Path:
        if (
            not dataset_id
            or not dataset_id.endswith(".json")
            or Path(dataset_id).is_absolute()
            or ".." in Path(dataset_id).parts
        ):
            raise ValueError("Invalid dataset id")
        if self.qa_dir == QA_DIR.resolve() and dataset_id in PROJECT_DATASETS:
            path = PROJECT_DATASETS[dataset_id].resolve()
        elif self.qa_dir == QA_DIR.resolve() and dataset_id.startswith("data/qa/"):
            candidate = (ROOT / dataset_id).resolve()
            if ROOT / "data/qa" not in candidate.parents:
                raise ValueError("Invalid dataset id")
            path = candidate
        elif self.qa_dir == QA_DIR.resolve() and Path(dataset_id).name == dataset_id:
            matches = [candidate for candidate in self.qa_dir.rglob(dataset_id) if candidate.is_file()]
            if len(matches) != 1:
                raise KeyError(f"Unknown or ambiguous dataset: {dataset_id}")
            path = matches[0].resolve()
        else:
            path = (self.qa_dir / dataset_id).resolve()
        if not path.is_file() or self.qa_dir not in path.parents:
            raise KeyError(f"Unknown dataset: {dataset_id}")
        return path

    def inspect(self, path: Path, dataset_id: str | None = None) -> DatasetInfo | None:
        dataset_id = dataset_id or self._id_for_path(path)
        stat = path.stat()
        cache_key = str(path.resolve())
        cached = self._cache.get(cache_key)
        signature = (stat.st_mtime_ns, stat.st_size)
        if cached and cached[:2] == signature:
            return cached[2]
        try:
            dataset, raw, digest = load_dataset(path)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        papers = sum(
            isinstance(paper, dict) and isinstance(paper.get("QA"), dict)
            for paper in dataset.values()
        )
        qas = count_qas(dataset)
        if not papers or not qas:
            return None
        info = DatasetInfo(
            id=dataset_id,
            path=str(path),
            papers=papers,
            qas=qas,
            size_bytes=len(raw),
            updated_at=stat.st_mtime,
            sha256=digest,
            profile=dataset_profile(dataset_id, dataset, path=path),
        )
        self._cache[cache_key] = (*signature, info)
        return info

    def reviewable(self, dataset_id: str) -> bool:
        path = self.resolve(dataset_id)
        info = self.inspect(path, dataset_id)
        return bool(info and info.profile.get("reviewable", True))

    def list(self, collection_id: str | None = None) -> list[DatasetInfo]:
        cache_key = collection_id or "__all__"
        now = time.monotonic()
        cached = self._list_cache.get(cache_key)
        if cached and now - cached[0] < self.LIST_CACHE_TTL_SECONDS:
            return list(cached[2])
        if collection_id == "final_2200" and self.qa_dir == QA_DIR.resolve():
            paths = [
                path for dataset_id, path in PROJECT_DATASETS.items()
                if DATASET_CATALOG.get(dataset_id, {}).get("collection_id") == "final_2200"
            ]
        else:
            paths = list(self.qa_dir.rglob("*.json"))
            if self.qa_dir == QA_DIR.resolve():
                paths.extend(PROJECT_DATASETS.values())
        unique_paths = sorted(set(paths))
        signatures: list[tuple[str, int, int]] = []
        for path in unique_paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            signatures.append((str(path), stat.st_mtime_ns, stat.st_size))
        signature_key = tuple(signatures)
        if cached:
            cached_at, cached_signature, cached_rows = cached
            if now - cached_at < self.LIST_CACHE_TTL_SECONDS:
                return list(cached_rows)
            if cached_signature == signature_key:
                self._list_cache[cache_key] = (now, cached_signature, cached_rows)
                return list(cached_rows)
        results = []
        for path in unique_paths:
            if not path.is_file():
                continue
            # Avoid hashing and fully decoding tens of thousands of manifests,
            # ledgers and model-response files that cannot contain a QA map.
            try:
                if b'"QA"' not in path.read_bytes():
                    continue
            except OSError:
                continue
            info = self.inspect(path, self._id_for_path(path))
            if info:
                results.append(info)
        if collection_id == "final_2200":
            rank = {
                dataset_id: index
                for index, dataset_id in enumerate(FINAL_2200_DATASET_IDS)
            }
            ordered = sorted(results, key=lambda row: rank[row.id])
        else:
            ordered = sorted(results, key=lambda row: row.updated_at, reverse=True)
        if collection_id == "other":
            ordered = [
                row for row in ordered
                if row.profile.get("collection_id") != "final_2200"
            ]
        self._list_cache[cache_key] = (now, signature_key, ordered)
        return list(ordered)

    def pdf_for(self, dataset_id: str, paper_id: str) -> Path:
        if (
            not paper_id
            or Path(paper_id).name != paper_id
            or "/" in paper_id
        ):
            raise ValueError("Invalid paper id")
        try:
            resolved = resolve_pdf_path(paper_id, [self.pdf_root])
        except (FileNotFoundError, ValueError) as exc:
            raise KeyError(f"PDF not found for {dataset_id}/{paper_id}") from exc
        if self.pdf_root.resolve() not in resolved.resolve().parents:
            raise KeyError(f"PDF escaped asset root for {dataset_id}/{paper_id}")
        return resolved.resolve()


class ReviewStore:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        registry: DatasetRegistry | None = None,
        snapshot_dir: Path = SNAPSHOT_DIR,
        final_manifest_path: Path = FINAL_2200_MANIFEST,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry or DatasetRegistry()
        self.final_manifest_path = Path(final_manifest_path)
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS reviews (
                    dataset_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    qa_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(dataset_id, paper_id, qa_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    qa_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    before_status TEXT,
                    after_status TEXT,
                    before_sha256 TEXT NOT NULL,
                    after_sha256 TEXT NOT NULL,
                    snapshot_path TEXT,
                    item_json TEXT NOT NULL,
                    note TEXT,
                    created_at REAL NOT NULL,
                    undone_at REAL,
                    reversible INTEGER NOT NULL DEFAULT 1,
                    legacy_dataset_id TEXT,
                    legacy_paper_id TEXT,
                    legacy_qa_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_review_events
                ON events(dataset_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin','reviewer')),
                    roles_json TEXT NOT NULL DEFAULT '["reviewer"]',
                    email TEXT,
                    email_verified_at REAL,
                    password_hash TEXT,
                    disabled_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    created_by_user_id TEXT,
                    FOREIGN KEY(created_by_user_id) REFERENCES users(user_id)
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    user_agent TEXT,
                    remote_addr TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_expiry
                ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS email_verifications (
                    email TEXT PRIMARY KEY COLLATE NOCASE,
                    purpose TEXT NOT NULL,
                    code_salt TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    sent_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    requested_remote_addr TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_email_verifications_expiry
                ON email_verifications(expires_at);
                CREATE TABLE IF NOT EXISTS audit_events (
                    audit_id TEXT PRIMARY KEY,
                    actor_user_id TEXT,
                    actor_username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    remote_addr TEXT,
                    user_agent TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_audit_events_time
                ON audit_events(created_at DESC);
                CREATE TABLE IF NOT EXISTS review_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    start_index INTEGER NOT NULL,
                    end_index INTEGER NOT NULL,
                    assigned_count INTEGER NOT NULL,
                    selection_mode TEXT NOT NULL DEFAULT 'range',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    created_by_user_id TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    FOREIGN KEY(created_by_user_id) REFERENCES users(user_id)
                );
                CREATE INDEX IF NOT EXISTS idx_review_assignments_user
                ON review_assignments(user_id, dataset_id);
                CREATE TABLE IF NOT EXISTS review_assignment_members (
                    assignment_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    qa_id TEXT NOT NULL,
                    source_index INTEGER NOT NULL,
                    PRIMARY KEY(assignment_id, paper_id, qa_id),
                    UNIQUE(dataset_id, paper_id, qa_id),
                    FOREIGN KEY(assignment_id) REFERENCES review_assignments(assignment_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_review_assignment_members_lookup
                ON review_assignment_members(dataset_id, paper_id, qa_id);
                CREATE TABLE IF NOT EXISTS review_tombstones (
                    dataset_id TEXT NOT NULL,
                    legacy_dataset_id TEXT NOT NULL,
                    legacy_paper_id TEXT NOT NULL,
                    legacy_qa_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    migrated_at REAL NOT NULL,
                    PRIMARY KEY(legacy_dataset_id, legacy_paper_id, legacy_qa_id)
                );
                """
            )
            self._ensure_column(connection, "events", "actor_user_id", "TEXT")
            self._ensure_column(connection, "events", "actor_username", "TEXT")
            self._ensure_column(connection, "events", "undone_by_user_id", "TEXT")
            self._ensure_column(connection, "events", "undone_by_username", "TEXT")
            self._ensure_column(connection, "events", "reversible", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(connection, "events", "legacy_dataset_id", "TEXT")
            self._ensure_column(connection, "events", "legacy_paper_id", "TEXT")
            self._ensure_column(connection, "events", "legacy_qa_id", "TEXT")
            self._ensure_column(
                connection,
                "review_assignments",
                "selection_mode",
                "TEXT NOT NULL DEFAULT 'range'",
            )
            self._ensure_column(connection, "users", "roles_json", "TEXT")
            self._ensure_column(connection, "users", "email", "TEXT")
            self._ensure_column(connection, "users", "email_verified_at", "REAL")
            connection.execute(
                """
                UPDATE users SET roles_json=CASE role
                    WHEN 'admin' THEN '["admin"]'
                    ELSE '["reviewer"]'
                END
                WHERE roles_json IS NULL OR trim(roles_json)=''
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique
                ON users(lower(email)) WHERE email IS NOT NULL
                """
            )
            timestamp = time.time()
            connection.execute(
                """
                INSERT INTO users(
                    user_id,username,display_name,role,roles_json,password_hash,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(username) DO UPDATE SET
                    role='admin',updated_at=excluded.updated_at
                """,
                (
                    uuid.uuid4().hex,
                    BOOTSTRAP_ADMIN_USERNAME,
                    "CZJ 管理员",
                    "admin",
                    '["admin"]',
                    None,
                    timestamp,
                    timestamp,
                ),
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _password_hash(password: str) -> str:
        if len(password) < PASSWORD_MIN_LENGTH:
            raise ValueError(
                f"Password must contain at least {PASSWORD_MIN_LENGTH} characters"
            )
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1
        )
        return "scrypt$16384$8$1$%s$%s" % (
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )

    @staticmethod
    def _password_matches(password: str, encoded: str | None) -> bool:
        if not encoded:
            return False
        try:
            algorithm, raw_n, raw_r, raw_p, raw_salt, raw_digest = encoded.split("$")
            if algorithm != "scrypt":
                return False
            salt = base64.urlsafe_b64decode(raw_salt.encode("ascii"))
            expected = base64.urlsafe_b64decode(raw_digest.encode("ascii"))
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=salt,
                n=int(raw_n),
                r=int(raw_r),
                p=int(raw_p),
            )
            return hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, Any]:
        try:
            roles = json.loads(row["roles_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            roles = []
        roles = [role for role in ACCOUNT_ROLES if role in roles]
        if not roles:
            roles = [str(row["role"])]
        return {
            "user_id": str(row["user_id"]),
            "username": str(row["username"]),
            "display_name": str(row["display_name"]),
            "role": "admin" if "admin" in roles else "reviewer",
            "roles": roles,
            "email": str(row["email"] or ""),
            "email_verified": row["email_verified_at"] is not None,
            "disabled": row["disabled_at"] is not None,
            "password_configured": bool(row["password_hash"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    @staticmethod
    def has_role(user: dict[str, Any] | None, role: str) -> bool:
        if user is None:
            return False
        roles = user.get("roles")
        if isinstance(roles, list):
            return role in roles
        return user.get("role") == role

    @staticmethod
    def _normalize_roles(roles: Any) -> list[str]:
        if isinstance(roles, str):
            roles = [roles]
        if not isinstance(roles, (list, tuple, set)):
            raise ValueError("Roles must be a list containing admin and/or reviewer")
        normalized = [role for role in ACCOUNT_ROLES if role in roles]
        if not normalized:
            raise ValueError("An account must keep at least one identity")
        unknown = {str(role) for role in roles} - set(ACCOUNT_ROLES)
        if unknown:
            raise ValueError(f"Unknown account roles: {', '.join(sorted(unknown))}")
        return normalized

    @staticmethod
    def _normalize_email(email: str) -> str:
        normalized = email.strip().lower()
        if len(normalized) > 254 or not re.fullmatch(
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+",
            normalized,
        ):
            raise ValueError("Please enter a valid email address")
        return normalized

    @staticmethod
    def _normalize_display_name(display_name: str) -> str:
        normalized = " ".join(display_name.strip().split())
        if len(normalized) < 2 or len(normalized) > 80:
            raise ValueError("Real name must contain 2-80 characters")
        return normalized

    def audit(
        self,
        action: str,
        target_type: str,
        target_id: str | None,
        *,
        actor: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        remote_addr: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        with self.connect() as connection:
            return self._insert_audit(
                connection,
                action,
                target_type,
                target_id,
                actor=actor,
                details=details,
                remote_addr=remote_addr,
                user_agent=user_agent,
            )

    @staticmethod
    def _insert_audit(
        connection: sqlite3.Connection,
        action: str,
        target_type: str,
        target_id: str | None,
        *,
        actor: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        remote_addr: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        audit_id = uuid.uuid4().hex
        username = str(actor["username"]) if actor else "system"
        user_id = str(actor["user_id"]) if actor else None
        connection.execute(
            """
            INSERT INTO audit_events(
                audit_id,actor_user_id,actor_username,action,target_type,
                target_id,details_json,created_at,remote_addr,user_agent
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                audit_id,
                user_id,
                username,
                action,
                target_type,
                target_id,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                time.time(),
                remote_addr,
                user_agent,
            ),
        )
        return audit_id

    def audit_events(self, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            value = dict(row)
            value["details"] = json.loads(value.pop("details_json") or "{}")
            results.append(value)
        return results

    def list_users(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM users ORDER BY disabled_at IS NOT NULL, username"
            ).fetchall()
        return [self._public_user(row) for row in rows]

    @staticmethod
    def _public_assignment(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "assignment_id": str(row["assignment_id"]),
            "user_id": str(row["user_id"]),
            "username": str(row["username"]),
            "display_name": str(row["display_name"]),
            "dataset_id": str(row["dataset_id"]),
            "start_index": int(row["start_index"]),
            "end_index": int(row["end_index"]),
            "assigned_count": int(row["assigned_count"]),
            "selection_mode": str(row["selection_mode"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def list_assignments(self, user_id: str | None = None) -> list[dict[str, Any]]:
        user_filter = "WHERE assignments.user_id=?" if user_id else ""
        params: tuple[Any, ...] = (user_id,) if user_id else ()
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT assignments.*,users.username,users.display_name,
                       MIN(members.source_index) AS first_index,
                       COALESCE(SUM(CASE WHEN reviews.status IS NOT NULL THEN 1 ELSE 0 END), 0) AS reviewed_count,
                       COALESCE(SUM(CASE WHEN reviews.status='kept' THEN 1 ELSE 0 END), 0) AS kept_count,
                       COALESCE(SUM(CASE WHEN reviews.status='deleted' THEN 1 ELSE 0 END), 0) AS deleted_count,
                       COALESCE(SUM(CASE WHEN reviews.status='edited' THEN 1 ELSE 0 END), 0) AS edited_count
                FROM review_assignments AS assignments
                JOIN users ON users.user_id=assignments.user_id
                LEFT JOIN review_assignment_members AS members
                  ON members.assignment_id=assignments.assignment_id
                LEFT JOIN reviews
                  ON reviews.dataset_id=members.dataset_id
                 AND reviews.paper_id=members.paper_id
                 AND reviews.qa_id=members.qa_id
                {user_filter}
                GROUP BY assignments.assignment_id
                ORDER BY assignments.dataset_id,assignments.start_index,
                         users.username
                """,
                params,
            ).fetchall()
        return [
            {
                **self._public_assignment(row),
                "first_index": int(row["first_index"] or row["start_index"]),
                "reviewed_count": int(row["reviewed_count"]),
                "kept_count": int(row["kept_count"]),
                "deleted_count": int(row["deleted_count"]),
                "edited_count": int(row["edited_count"]),
            }
            for row in rows
        ]

    def reviewer_progress(self) -> list[dict[str, Any]]:
        """Return progress aggregated by reviewer over stable assignment members."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT users.user_id, users.username, users.display_name,
                       users.disabled_at,
                       COUNT(DISTINCT members.assignment_id || ':' || members.paper_id || ':' || members.qa_id)
                           AS assigned_count,
                       COUNT(DISTINCT CASE WHEN reviews.status IS NOT NULL
                           THEN members.assignment_id || ':' || members.paper_id || ':' || members.qa_id END)
                           AS reviewed_count,
                       COUNT(DISTINCT CASE WHEN reviews.status='kept'
                           THEN members.assignment_id || ':' || members.paper_id || ':' || members.qa_id END)
                           AS kept_count,
                       COUNT(DISTINCT CASE WHEN reviews.status='edited'
                           THEN members.assignment_id || ':' || members.paper_id || ':' || members.qa_id END)
                           AS edited_count,
                       COUNT(DISTINCT CASE WHEN reviews.status='deleted'
                           THEN members.assignment_id || ':' || members.paper_id || ':' || members.qa_id END)
                           AS deleted_count,
                       MAX(reviews.updated_at) AS last_reviewed_at
                FROM users
                LEFT JOIN review_assignments AS assignments
                  ON assignments.user_id=users.user_id
                LEFT JOIN review_assignment_members AS members
                  ON members.assignment_id=assignments.assignment_id
                LEFT JOIN reviews
                  ON reviews.dataset_id=members.dataset_id
                 AND reviews.paper_id=members.paper_id
                 AND reviews.qa_id=members.qa_id
                WHERE users.role='reviewer'
                   OR instr(COALESCE(users.roles_json, ''), '"reviewer"') > 0
                GROUP BY users.user_id
                ORDER BY users.disabled_at IS NOT NULL, users.username
                """
            ).fetchall()
        return [
            {
                "user_id": str(row["user_id"]),
                "username": str(row["username"]),
                "display_name": str(row["display_name"]),
                "disabled": row["disabled_at"] is not None,
                "assigned_count": int(row["assigned_count"]),
                "reviewed_count": int(row["reviewed_count"]),
                "kept_count": int(row["kept_count"]),
                "edited_count": int(row["edited_count"]),
                "deleted_count": int(row["deleted_count"]),
                "last_reviewed_at": float(row["last_reviewed_at"])
                if row["last_reviewed_at"] is not None else None,
            }
            for row in rows
        ]

    def _dataset_rows(self, dataset_id: str) -> tuple[list[dict[str, Any]], str]:
        path = self.registry.resolve(dataset_id)
        dataset, _, digest = load_dataset(path)
        rows = []
        for paper_id, paper in dataset.items():
            if not isinstance(paper, dict):
                continue
            for qa_id, qa in paper.get("QA", {}).items():
                rows.append(
                    {
                        "paper_id": str(paper_id),
                        "qa_id": str(qa_id),
                        "qa": qa,
                        "paper": {
                            key: value for key, value in paper.items() if key != "QA"
                        },
                    }
                )
        if not rows:
            raise KeyError("Dataset contains no reviewable QA")
        return rows, digest

    def _validate_assignment_range(
        self, dataset_id: str, start_index: int, end_index: int
    ) -> list[dict[str, Any]]:
        rows, _ = self._dataset_rows(dataset_id)
        start_index = int(start_index)
        end_index = int(end_index)
        if start_index < 1 or end_index < start_index or end_index > len(rows):
            raise ValueError(
                f"Assignment range must be between 1 and {len(rows)}"
            )
        return rows[start_index - 1 : end_index]

    def _assignment_conflict(
        self,
        connection: sqlite3.Connection,
        dataset_id: str,
        members: list[dict[str, Any]],
        *,
        exclude_assignment_id: str | None = None,
    ) -> sqlite3.Row | None:
        for member in members:
            row = connection.execute(
                """
                SELECT assignments.assignment_id,assignments.start_index,
                       assignments.end_index,users.username
                FROM review_assignment_members AS membership
                JOIN review_assignments AS assignments
                  ON assignments.assignment_id=membership.assignment_id
                JOIN users ON users.user_id=assignments.user_id
                WHERE membership.dataset_id=? AND membership.paper_id=?
                  AND membership.qa_id=? AND assignments.assignment_id!=?
                LIMIT 1
                """,
                (
                    dataset_id,
                    member["paper_id"],
                    member["qa_id"],
                    exclude_assignment_id or "",
                ),
            ).fetchone()
            if row is not None:
                return row
        return None

    def create_assignment(
        self,
        user_id: str,
        dataset_id: str,
        start_index: int,
        end_index: int,
        *,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = self.user(user_id)
        if not self.has_role(user, "reviewer") or user["disabled"]:
            raise ValueError("Assignments require an active reviewer account")
        members = self._validate_assignment_range(dataset_id, start_index, end_index)
        assignment_id = uuid.uuid4().hex
        timestamp = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            conflict = self._assignment_conflict(connection, dataset_id, members)
            if conflict is not None:
                raise ValueError(
                    "QA range overlaps assignment for "
                    f"{conflict['username']} ({conflict['start_index']}-"
                    f"{conflict['end_index']})"
                )
            connection.execute(
                """
                INSERT INTO review_assignments(
                    assignment_id,user_id,dataset_id,start_index,end_index,
                    assigned_count,selection_mode,created_at,updated_at,created_by_user_id
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    assignment_id,
                    user_id,
                    dataset_id,
                    int(start_index),
                    int(end_index),
                    len(members),
                    "range",
                    timestamp,
                    timestamp,
                    actor["user_id"] if actor else None,
                ),
            )
            connection.executemany(
                """
                INSERT INTO review_assignment_members(
                    assignment_id,dataset_id,paper_id,qa_id,source_index
                ) VALUES(?,?,?,?,?)
                """,
                [
                    (
                        assignment_id,
                        dataset_id,
                        member["paper_id"],
                        member["qa_id"],
                        position,
                    )
                    for position, member in enumerate(members, int(start_index))
                ],
            )
            self._insert_audit(
                connection,
                "assignment.create",
                "review_assignment",
                assignment_id,
                actor=actor,
                details={
                    "user_id": user_id,
                    "username": user["username"],
                    "dataset_id": dataset_id,
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "assigned_count": len(members),
                },
            )
            connection.commit()
        return next(
            row for row in self.list_assignments()
            if row["assignment_id"] == assignment_id
        )

    def update_assignment(
        self,
        assignment_id: str,
        *,
        user_id: str,
        dataset_id: str,
        start_index: int,
        end_index: int,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = self.user(user_id)
        if not self.has_role(user, "reviewer") or user["disabled"]:
            raise ValueError("Assignments require an active reviewer account")
        members = self._validate_assignment_range(dataset_id, start_index, end_index)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT assignment_id,selection_mode FROM review_assignments WHERE assignment_id=?",
                (assignment_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Unknown assignment: {assignment_id}")
            if existing["selection_mode"] == "stable_members":
                raise ValueError(
                    "Migrated stable-member assignments must be deleted and recreated as a range"
                )
            conflict = self._assignment_conflict(
                connection,
                dataset_id,
                members,
                exclude_assignment_id=assignment_id,
            )
            if conflict is not None:
                raise ValueError(
                    "QA range overlaps assignment for "
                    f"{conflict['username']} ({conflict['start_index']}-"
                    f"{conflict['end_index']})"
                )
            connection.execute(
                """
                UPDATE review_assignments SET
                    user_id=?,dataset_id=?,start_index=?,end_index=?,
                    assigned_count=?,selection_mode='range',updated_at=?
                WHERE assignment_id=?
                """,
                (
                    user_id,
                    dataset_id,
                    int(start_index),
                    int(end_index),
                    len(members),
                    time.time(),
                    assignment_id,
                ),
            )
            connection.execute(
                "DELETE FROM review_assignment_members WHERE assignment_id=?",
                (assignment_id,),
            )
            connection.executemany(
                """
                INSERT INTO review_assignment_members(
                    assignment_id,dataset_id,paper_id,qa_id,source_index
                ) VALUES(?,?,?,?,?)
                """,
                [
                    (
                        assignment_id,
                        dataset_id,
                        member["paper_id"],
                        member["qa_id"],
                        position,
                    )
                    for position, member in enumerate(members, int(start_index))
                ],
            )
            self._insert_audit(
                connection,
                "assignment.update",
                "review_assignment",
                assignment_id,
                actor=actor,
                details={
                    "user_id": user_id,
                    "username": user["username"],
                    "dataset_id": dataset_id,
                    "start_index": int(start_index),
                    "end_index": int(end_index),
                    "assigned_count": len(members),
                },
            )
            connection.commit()
        return next(
            row for row in self.list_assignments()
            if row["assignment_id"] == assignment_id
        )

    def delete_assignment(
        self,
        assignment_id: str,
        *,
        actor: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM review_assignments WHERE assignment_id=?",
                (assignment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown assignment: {assignment_id}")
            details = dict(row)
            connection.execute(
                "DELETE FROM review_assignments WHERE assignment_id=?",
                (assignment_id,),
            )
            self._insert_audit(
                connection,
                "assignment.delete",
                "review_assignment",
                assignment_id,
                actor=actor,
                details=details,
            )
            connection.commit()

    def assignment_for_item(
        self,
        dataset_id: str,
        paper_id: str,
        qa_id: str,
        user_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any] | None:
        owns_connection = connection is None
        db = connection or self.connect()
        try:
            row = db.execute(
                """
                SELECT assignments.*,users.username,users.display_name
                FROM review_assignment_members AS membership
                JOIN review_assignments AS assignments
                  ON assignments.assignment_id=membership.assignment_id
                JOIN users ON users.user_id=assignments.user_id
                WHERE membership.dataset_id=? AND membership.paper_id=?
                  AND membership.qa_id=? AND assignments.user_id=?
                """,
                (dataset_id, paper_id, qa_id, user_id),
            ).fetchone()
            return self._public_assignment(row) if row is not None else None
        finally:
            if owns_connection:
                db.close()

    def _require_item_permission(
        self,
        dataset_id: str,
        paper_id: str,
        qa_id: str,
        actor: dict[str, Any] | None,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if not self.registry.reviewable(dataset_id):
            info = self.registry.inspect(self.registry.resolve(dataset_id), dataset_id)
            reason = (info.profile.get("read_only_reason") if info else None) or "该数据集是只读过程产物。"
            raise PermissionError(f"{reason}")
        if actor is None or (
            self.has_role(actor, "admin") and not self.has_role(actor, "reviewer")
        ):
            return
        if not self.has_role(actor, "reviewer"):
            raise PermissionError(
                "当前账户没有校验员身份，仅可查看，不能修改。"
            )
        assignment = self.assignment_for_item(
            dataset_id,
            paper_id,
            qa_id,
            str(actor["user_id"]),
            connection=connection,
        )
        if assignment is None:
            raise PermissionError(
                "此 QA 未分配给当前账户，仅可查看，不能修改。"
            )

    def user(self, user_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown user: {user_id}")
        return self._public_user(row)

    def create_user(
        self,
        username: str,
        password: str,
        *,
        display_name: str = "",
        role: str = "reviewer",
        roles: Any = None,
        email: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        username = username.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,31}", username):
            raise ValueError(
                "Username must be 3-32 characters using letters, numbers, dot, dash or underscore"
            )
        normalized_roles = self._normalize_roles(roles if roles is not None else role)
        normalized_email = self._normalize_email(email) if email else None
        normalized_name = self._normalize_display_name(display_name or username)
        timestamp = time.time()
        user_id = uuid.uuid4().hex
        try:
            with self.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users(
                        user_id,username,display_name,role,roles_json,email,
                        email_verified_at,password_hash,created_at,updated_at,
                        created_by_user_id
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        user_id,
                        username,
                        normalized_name,
                        "admin" if "admin" in normalized_roles else "reviewer",
                        json.dumps(normalized_roles),
                        normalized_email,
                        timestamp if normalized_email else None,
                        self._password_hash(password),
                        timestamp,
                        timestamp,
                        actor["user_id"] if actor else None,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username or email is already registered") from exc
        self.audit(
            "user.create",
            "user",
            user_id,
            actor=actor,
            details={"username": username, "roles": normalized_roles},
        )
        return self.user(user_id)

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        role: str | None = None,
        roles: Any = None,
        disabled: bool | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.user(user_id)
        normalized_roles = None
        if roles is not None or role is not None:
            normalized_roles = self._normalize_roles(roles if roles is not None else role)
        if current["username"].lower() == BOOTSTRAP_ADMIN_USERNAME and (
            (normalized_roles is not None and "admin" not in normalized_roles)
            or disabled is True
        ):
            raise ValueError("The bootstrap czj-web administrator cannot be disabled or demoted")
        updates = []
        values: list[Any] = []
        if display_name is not None:
            updates.append("display_name=?")
            values.append(self._normalize_display_name(display_name))
        if normalized_roles is not None:
            updates.extend(["role=?", "roles_json=?"])
            values.extend([
                "admin" if "admin" in normalized_roles else "reviewer",
                json.dumps(normalized_roles),
            ])
        if disabled is not None:
            updates.append("disabled_at=?")
            values.append(time.time() if disabled else None)
        if updates:
            updates.append("updated_at=?")
            values.extend([time.time(), user_id])
            with self.connect() as connection:
                connection.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE user_id=?",
                    values,
                )
                if disabled:
                    connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        updated = self.user(user_id)
        self.audit(
            "user.update",
            "user",
            user_id,
            actor=actor,
            details={
                "username": updated["username"],
                "roles": updated["roles"],
                "disabled": updated["disabled"],
            },
        )
        return updated

    def prepare_email_verification(
        self,
        email: str,
        purpose: str,
        *,
        remote_addr: str | None = None,
    ) -> tuple[str, str]:
        normalized_email = self._normalize_email(email)
        if purpose not in {"register", "profile"}:
            raise ValueError("Invalid email verification purpose")
        timestamp = time.time()
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM email_verifications WHERE expires_at<=?", (timestamp,)
            )
            existing = connection.execute(
                "SELECT sent_at FROM email_verifications WHERE email=? COLLATE NOCASE",
                (normalized_email,),
            ).fetchone()
            if existing is not None and timestamp - float(existing["sent_at"]) < EMAIL_CODE_RESEND_SECONDS:
                retry_after = int(
                    EMAIL_CODE_RESEND_SECONDS - (timestamp - float(existing["sent_at"]))
                )
                raise RuntimeError(f"Please wait {max(1, retry_after)} seconds before requesting another code")
            owner = connection.execute(
                "SELECT user_id FROM users WHERE email=? COLLATE NOCASE",
                (normalized_email,),
            ).fetchone()
            if purpose == "register" and owner is not None:
                raise ValueError("This email is already registered")
            code = f"{secrets.randbelow(1_000_000):06d}"
            salt = secrets.token_hex(16)
            code_hash = hashlib.sha256(
                f"{salt}:{normalized_email}:{purpose}:{code}".encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT INTO email_verifications(
                    email,purpose,code_salt,code_hash,sent_at,expires_at,
                    attempt_count,requested_remote_addr
                ) VALUES(?,?,?,?,?,?,0,?)
                ON CONFLICT(email) DO UPDATE SET
                    purpose=excluded.purpose,code_salt=excluded.code_salt,
                    code_hash=excluded.code_hash,sent_at=excluded.sent_at,
                    expires_at=excluded.expires_at,attempt_count=0,
                    requested_remote_addr=excluded.requested_remote_addr
                """,
                (
                    normalized_email,
                    purpose,
                    salt,
                    code_hash,
                    timestamp,
                    timestamp + EMAIL_CODE_TTL_SECONDS,
                    remote_addr,
                ),
            )
        return normalized_email, code

    def cancel_email_verification(self, email: str) -> None:
        normalized_email = self._normalize_email(email)
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM email_verifications WHERE email=? COLLATE NOCASE",
                (normalized_email,),
            )

    def verify_email_code(self, email: str, purpose: str, code: str) -> str:
        normalized_email = self._normalize_email(email)
        timestamp = time.time()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM email_verifications WHERE email=? COLLATE NOCASE",
                (normalized_email,),
            ).fetchone()
            if row is None or row["purpose"] != purpose or float(row["expires_at"]) <= timestamp:
                raise ValueError("The email verification code is missing or expired")
            if int(row["attempt_count"]) >= EMAIL_CODE_MAX_ATTEMPTS:
                raise ValueError("Too many invalid verification attempts; request a new code")
            actual = hashlib.sha256(
                f"{row['code_salt']}:{normalized_email}:{purpose}:{code.strip()}".encode("utf-8")
            ).hexdigest()
            if not hmac.compare_digest(actual, str(row["code_hash"])):
                connection.execute(
                    "UPDATE email_verifications SET attempt_count=attempt_count+1 WHERE email=? COLLATE NOCASE",
                    (normalized_email,),
                )
                connection.commit()
                raise ValueError("Invalid email verification code")
        return normalized_email

    def register_user(
        self,
        username: str,
        password: str,
        *,
        display_name: str,
        email: str,
        email_code: str,
        remote_addr: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        normalized_email = self.verify_email_code(email, "register", email_code)
        with _MUTATION_LOCK:
            user = self.create_user(
                username,
                password,
                display_name=display_name,
                roles=["reviewer"],
                email=normalized_email,
                actor=None,
            )
            self.cancel_email_verification(normalized_email)
        self.audit(
            "auth.register",
            "user",
            user["user_id"],
            actor=user,
            details={"email": normalized_email, "roles": ["reviewer"]},
            remote_addr=remote_addr,
            user_agent=user_agent,
        )
        return user

    def update_profile(
        self,
        user_id: str,
        *,
        username: str | None = None,
        display_name: str | None = None,
        email: str | None = None,
        email_code: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.user(user_id)
        updates: list[str] = []
        values: list[Any] = []
        changed: list[str] = []
        if username is not None:
            normalized_username = username.strip()
            if current["username"].lower() == BOOTSTRAP_ADMIN_USERNAME and normalized_username.lower() != BOOTSTRAP_ADMIN_USERNAME:
                raise ValueError("The bootstrap czj-web username cannot be changed")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,31}", normalized_username):
                raise ValueError("Username must be 3-32 characters using letters, numbers, dot, dash or underscore")
            if normalized_username.lower() != current["username"].lower():
                updates.append("username=?")
                values.append(normalized_username)
                changed.append("username")
        if display_name is not None:
            normalized_name = self._normalize_display_name(display_name)
            if normalized_name != current["display_name"]:
                updates.append("display_name=?")
                values.append(normalized_name)
                changed.append("display_name")
        if email is not None:
            normalized_email = self._normalize_email(email)
            if normalized_email != current["email"].lower():
                if not email_code:
                    raise ValueError("Email verification code is required")
                self.verify_email_code(normalized_email, "profile", email_code)
                updates.extend(["email=?", "email_verified_at=?"])
                values.extend([normalized_email, time.time()])
                changed.append("email")
        if not updates:
            return current
        updates.append("updated_at=?")
        values.extend([time.time(), user_id])
        try:
            with self.connect() as connection:
                connection.execute(
                    f"UPDATE users SET {', '.join(updates)} WHERE user_id=?",
                    values,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username or email is already registered") from exc
        if "email" in changed:
            self.cancel_email_verification(str(email))
        updated = self.user(user_id)
        self.audit(
            "user.profile_update",
            "user",
            user_id,
            actor=actor or updated,
            details={"changed_fields": changed},
        )
        return updated

    def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        *,
        session_token: str,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
            if row is None or not self._password_matches(current_password, row["password_hash"]):
                raise PermissionError("Current password is incorrect")
            connection.execute(
                "UPDATE users SET password_hash=?,updated_at=? WHERE user_id=?",
                (self._password_hash(new_password), time.time(), user_id),
            )
            current_hash = hashlib.sha256(session_token.encode("utf-8")).hexdigest()
            connection.execute(
                "DELETE FROM sessions WHERE user_id=? AND token_hash!=?",
                (user_id, current_hash),
            )
        updated = self.user(user_id)
        self.audit(
            "user.password_change",
            "user",
            user_id,
            actor=actor or updated,
        )
        return updated

    def set_password(
        self,
        user_id: str,
        password: str,
        *,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user = self.user(user_id)
        with self.connect() as connection:
            connection.execute(
                "UPDATE users SET password_hash=?,updated_at=? WHERE user_id=?",
                (self._password_hash(password), time.time(), user_id),
            )
            connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
        self.audit(
            "user.password_reset",
            "user",
            user_id,
            actor=actor,
            details={"username": user["username"]},
        )
        return self.user(user_id)

    def authenticate_password(self, username: str, password: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
        if row is None or row["disabled_at"] is not None or not self._password_matches(
            password, row["password_hash"]
        ):
            raise PermissionError("Invalid username or password")
        return self._public_user(row)

    def perimeter_user(self, username: str) -> dict[str, Any]:
        if username.strip().lower() != BOOTSTRAP_ADMIN_USERNAME:
            raise PermissionError("Only the czj-web perimeter account can bootstrap admin access")
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE",
                (BOOTSTRAP_ADMIN_USERNAME,),
            ).fetchone()
        if row is None or row["disabled_at"] is not None:
            raise PermissionError("Bootstrap administrator is unavailable")
        return self._public_user(row)

    def create_session(
        self,
        user: dict[str, Any],
        *,
        remote_addr: str | None = None,
        user_agent: str | None = None,
    ) -> str:
        token = secrets.token_urlsafe(32)
        timestamp = time.time()
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE expires_at<=?", (timestamp,))
            connection.execute(
                """
                INSERT INTO sessions(
                    token_hash,user_id,created_at,expires_at,last_seen_at,
                    user_agent,remote_addr
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    hashlib.sha256(token.encode("utf-8")).hexdigest(),
                    user["user_id"],
                    timestamp,
                    timestamp + SESSION_TTL_SECONDS,
                    timestamp,
                    user_agent,
                    remote_addr,
                ),
            )
        return token

    def session_user(self, token: str) -> dict[str, Any]:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        timestamp = time.time()
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT users.* FROM sessions
                JOIN users ON users.user_id=sessions.user_id
                WHERE sessions.token_hash=? AND sessions.expires_at>?
                  AND users.disabled_at IS NULL
                """,
                (token_hash, timestamp),
            ).fetchone()
            if row is None:
                raise PermissionError("Authentication required")
            connection.execute(
                "UPDATE sessions SET last_seen_at=? WHERE token_hash=?",
                (timestamp, token_hash),
            )
        return self._public_user(row)

    def delete_session(self, token: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash=?",
                (hashlib.sha256(token.encode("utf-8")).hexdigest(),),
            )

    def status_map(self, dataset_id: str) -> dict[tuple[str, str], str]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT paper_id,qa_id,status FROM reviews WHERE dataset_id=?",
                (dataset_id,),
            ).fetchall()
        return {
            (str(row["paper_id"]), str(row["qa_id"])): str(row["status"])
            for row in rows
        }

    def item(
        self,
        dataset_id: str,
        index: int = 0,
        *,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rows, digest = self._dataset_rows(dataset_id)
        statuses = self.status_map(dataset_id)
        for row in rows:
            row["review_status"] = statuses.get(
                (row["paper_id"], row["qa_id"]), "unreviewed"
            )
        index = max(0, min(int(index), len(rows) - 1))
        kept = sum(status == "kept" for status in statuses.values())
        deleted = sum(status == "deleted" for status in statuses.values())
        edited = sum(status == "edited" for status in statuses.values())
        selected = rows[index]
        assignment = None
        can_modify = self.registry.reviewable(dataset_id) and (
            actor is None
            or (
                self.has_role(actor, "admin")
                and not self.has_role(actor, "reviewer")
            )
        )
        if actor is not None and self.has_role(actor, "reviewer"):
            assignment = self.assignment_for_item(
                dataset_id,
                selected["paper_id"],
                selected["qa_id"],
                str(actor["user_id"]),
            )
            can_modify = assignment is not None
        return {
            "dataset_id": dataset_id,
            "dataset_sha256": digest,
            "index": index,
            "total": len(rows),
            "reviewed": kept + deleted + edited,
            "kept": kept,
            "deleted": deleted,
            "edited": edited,
            "can_modify": can_modify,
            "assigned_to_current_user": assignment is not None,
            "assignment": assignment,
            "item": selected,
        }

    def _current_status(
        self, connection: sqlite3.Connection, dataset_id: str, paper_id: str, qa_id: str
    ) -> str | None:
        row = connection.execute(
            """
            SELECT status FROM reviews
            WHERE dataset_id=? AND paper_id=? AND qa_id=?
            """,
            (dataset_id, paper_id, qa_id),
        ).fetchone()
        return str(row["status"]) if row else None

    def _final_manifest_snapshot(self, dataset_id: str) -> bytes | None:
        if not self.final_manifest_path.is_file():
            return None
        raw = self.final_manifest_path.read_bytes()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError("Invalid final 2200 manifest")
        components = manifest.get("components", [])
        if not isinstance(components, list):
            raise ValueError("Invalid final 2200 manifest components")
        return raw if any(
            isinstance(component, dict)
            and component.get("dataset_id") == dataset_id
            for component in components
        ) else None

    def _refresh_final_manifest(
        self, dataset_id: str, dataset: dict[str, Any] | None = None
    ) -> None:
        snapshot = self._final_manifest_snapshot(dataset_id)
        if snapshot is None:
            return
        manifest = json.loads(snapshot)
        path = self.registry.resolve(dataset_id)
        if dataset is None:
            dataset, _, dataset_sha = load_dataset(path)
        else:
            dataset_sha = sha256_bytes(encode_dataset(dataset))
        rows = [
            qa
            for paper in dataset.values()
            if isinstance(paper, dict) and isinstance(paper.get("QA"), dict)
            for qa in paper["QA"].values()
            if isinstance(qa, dict)
        ]
        if any("options" in qa for qa in rows):
            raise ValueError("Final publication QA must be short-answer only")
        component = next(
            item
            for item in manifest["components"]
            if isinstance(item, dict) and item.get("dataset_id") == dataset_id
        )
        component.update(
            {
                "sha256": dataset_sha,
                "qa_count": len(rows),
                "paper_count": sum(
                    isinstance(paper, dict)
                    and isinstance(paper.get("QA"), dict)
                    and bool(paper["QA"])
                    for paper in dataset.values()
                ),
                "answerable": sum(
                    qa.get("answer") != "Unanswerable" for qa in rows
                ),
                "unanswerable": sum(
                    qa.get("answer") == "Unanswerable" for qa in rows
                ),
                "qa_format": "short_answer_only",
            }
        )
        manifest["qa_count"] = sum(
            int(item.get("qa_count", 0))
            for item in manifest["components"]
            if isinstance(item, dict)
        )
        manifest["qa_format"] = "short_answer_only"
        atomic_write(self.final_manifest_path, encode_dataset(manifest))

    def review(
        self,
        dataset_id: str,
        paper_id: str,
        qa_id: str,
        action: str,
        *,
        expected_sha256: str | None = None,
        note: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if action not in {"keep", "delete"}:
            raise ValueError("Action must be keep or delete")
        self._require_item_permission(dataset_id, paper_id, qa_id, actor)
        path = self.registry.resolve(dataset_id)
        with _MUTATION_LOCK:
            manifest_snapshot = self._final_manifest_snapshot(dataset_id)
            dataset, raw, before_sha = load_dataset(path)
            if expected_sha256 and expected_sha256 != before_sha:
                raise RuntimeError(
                    "Dataset changed after it was displayed; reload before reviewing"
                )
            paper = dataset.get(paper_id)
            if not isinstance(paper, dict):
                raise KeyError(f"Unknown paper: {paper_id}")
            qas = paper.get("QA")
            if not isinstance(qas, dict) or qa_id not in qas:
                raise KeyError(f"Unknown QA: {paper_id}/{qa_id}")
            item = qas[qa_id]
            event_id = uuid.uuid4().hex
            snapshot_path: Path | None = None
            after_sha = before_sha
            if action == "delete":
                snapshot_path = (
                    self.snapshot_dir
                    / dataset_id.removesuffix(".json")
                    / f"{time.time_ns()}_{event_id}.json"
                )
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, snapshot_path)
                del qas[qa_id]
                if not qas:
                    del dataset[paper_id]
                encoded = encode_dataset(dataset)
                atomic_write(path, encoded)
                try:
                    self._refresh_final_manifest(dataset_id, dataset)
                except Exception:
                    atomic_write(path, raw)
                    if manifest_snapshot is not None:
                        atomic_write(self.final_manifest_path, manifest_snapshot)
                    raise
                after_sha = sha256_bytes(encoded)
                self.registry.invalidate(dataset_id)

            timestamp = time.time()
            after_status = "kept" if action == "keep" else "deleted"
            try:
                with self.connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    before_status = self._current_status(
                        connection, dataset_id, paper_id, qa_id
                    )
                    connection.execute(
                        """
                        INSERT INTO events(
                            event_id,dataset_id,paper_id,qa_id,action,
                            before_status,after_status,before_sha256,after_sha256,
                            snapshot_path,item_json,note,created_at,
                            actor_user_id,actor_username
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            event_id,
                            dataset_id,
                            paper_id,
                            qa_id,
                            action,
                            before_status,
                            after_status,
                            before_sha,
                            after_sha,
                            str(snapshot_path) if snapshot_path else None,
                            json.dumps(item, ensure_ascii=False),
                            note,
                            timestamp,
                            actor["user_id"] if actor else None,
                            actor["username"] if actor else "system",
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO reviews(dataset_id,paper_id,qa_id,status,updated_at)
                        VALUES(?,?,?,?,?)
                        ON CONFLICT(dataset_id,paper_id,qa_id)
                        DO UPDATE SET status=excluded.status,updated_at=excluded.updated_at
                        """,
                        (dataset_id, paper_id, qa_id, after_status, timestamp),
                    )
                    self._insert_audit(
                        connection,
                        f"review.{action}",
                        "qa",
                        f"{dataset_id}:{paper_id}:{qa_id}",
                        actor=actor,
                        details={
                            "dataset_id": dataset_id,
                            "paper_id": paper_id,
                            "qa_id": qa_id,
                            "before_sha256": before_sha,
                            "after_sha256": after_sha,
                        },
                    )
                    connection.commit()
            except Exception:
                if snapshot_path and snapshot_path.is_file():
                    atomic_write(path, snapshot_path.read_bytes())
                if manifest_snapshot is not None:
                    atomic_write(self.final_manifest_path, manifest_snapshot)
                raise
            return {
                "event_id": event_id,
                "dataset_id": dataset_id,
                "paper_id": paper_id,
                "qa_id": qa_id,
                "action": action,
                "status": after_status,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "reversible": True,
            }

    def edit(
        self,
        dataset_id: str,
        paper_id: str,
        qa_id: str,
        *,
        question: str,
        answer: str,
        evidence_pages: list[Any],
        evidence_page_changes: list[dict[str, Any]] | None = None,
        expected_sha256: str | None = None,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_item_permission(dataset_id, paper_id, qa_id, actor)
        normalized_question = str(question).strip()
        normalized_answer = str(answer).strip()
        if not normalized_question:
            raise ValueError("问题不能为空。")
        if not normalized_answer:
            raise ValueError("标准答案不能为空。")
        if normalized_answer.casefold() == "unanswerable" and (
            normalized_answer != "Unanswerable"
        ):
            raise ValueError("不可回答答案必须精确写为 Unanswerable。")
        if not isinstance(evidence_pages, list):
            raise ValueError("证据物理页必须是数组。")
        pages: list[int] = []
        for raw_page in evidence_pages:
            if isinstance(raw_page, bool) or not isinstance(raw_page, int):
                raise ValueError("证据物理页必须是正整数。")
            if raw_page < 1:
                raise ValueError("证据物理页必须从 1 开始。")
            pages.append(raw_page)
        pages = sorted(set(pages))
        if normalized_answer == "Unanswerable" and pages:
            raise ValueError("Unanswerable 的证据物理页必须为空。")
        if normalized_answer != "Unanswerable" and not pages:
            raise ValueError("可回答题至少需要一个证据物理页。")

        path = self.registry.resolve(dataset_id)
        with _MUTATION_LOCK:
            manifest_snapshot = self._final_manifest_snapshot(dataset_id)
            dataset, _, before_sha = load_dataset(path)
            if expected_sha256 and expected_sha256 != before_sha:
                raise RuntimeError(
                    "Dataset changed after it was displayed; reload before editing"
                )
            paper = dataset.get(paper_id)
            if not isinstance(paper, dict):
                raise KeyError(f"Unknown paper: {paper_id}")
            qas = paper.get("QA")
            if not isinstance(qas, dict) or qa_id not in qas:
                raise KeyError(f"Unknown QA: {paper_id}/{qa_id}")
            item = qas[qa_id]
            if not isinstance(item, dict):
                raise ValueError(f"Invalid QA: {paper_id}/{qa_id}")

            total_pdf_pages: int | None = None
            try:
                pdf_path = self.registry.pdf_for(dataset_id, paper_id)
                total_pdf_pages = len(PdfReader(str(pdf_path)).pages)
            except (KeyError, ValueError, OSError, PdfReadError):
                total_pdf_pages = None
            if total_pdf_pages is not None and any(
                page > total_pdf_pages for page in pages
            ):
                raise ValueError(
                    f"证据物理页不能超过当前 PDF 的 {total_pdf_pages} 页。"
                )

            old_pages = [
                page
                for page in item.get("evidence_pages", [])
                if isinstance(page, int) and not isinstance(page, bool) and page >= 1
            ]
            old_page_set = set(old_pages)
            new_page_set = set(pages)
            page_mapping: dict[int, int] = {}
            for change in evidence_page_changes or []:
                if not isinstance(change, dict):
                    continue
                old_page = change.get("from")
                new_page = change.get("to")
                if (
                    isinstance(old_page, int)
                    and not isinstance(old_page, bool)
                    and isinstance(new_page, int)
                    and not isinstance(new_page, bool)
                    and new_page in pages
                ):
                    page_mapping[old_page] = new_page

            updated_item = copy.deepcopy(item)
            updated_item["question"] = normalized_question
            updated_item["answer"] = normalized_answer
            updated_item["evidence_pages"] = pages
            updated_item.pop("options", None)
            if normalized_answer == "Unanswerable":
                updated_item["answer_format"] = "Unanswerable"
            elif updated_item.get("answer_format") == "Unanswerable":
                updated_item["answer_format"] = "String"
            if "oracle_pages" in updated_item:
                updated_item["oracle_pages"] = copy.deepcopy(pages)
            if "source_evidence_pages" in updated_item:
                updated_item["source_evidence_pages"] = copy.deepcopy(pages)
            if isinstance(updated_item.get("construction_key_evidence_pages"), list):
                updated_item["construction_key_evidence_pages"] = sorted(
                    {
                        page_mapping.get(page, page)
                        for page in updated_item["construction_key_evidence_pages"]
                        if isinstance(page, int)
                        and not isinstance(page, bool)
                        and page_mapping.get(page, page) in new_page_set
                    }
                )
            if "evidence_span" in updated_item:
                updated_item["evidence_span"] = (
                    max(pages) - min(pages) if pages else 0
                )
            if "evidence_span_ratio" in updated_item:
                span = max(pages) - min(pages) if pages else 0
                updated_item["evidence_span_ratio"] = (
                    span / total_pdf_pages if total_pdf_pages else 0
                )
            if isinstance(updated_item.get("evidence_items"), list):
                synchronized_items = []
                for evidence in updated_item["evidence_items"]:
                    if not isinstance(evidence, dict):
                        synchronized_items.append(evidence)
                        continue
                    current_page = evidence.get("physical_pdf_page")
                    synchronized = copy.deepcopy(evidence)
                    if isinstance(current_page, int) and current_page in old_page_set:
                        target_page = page_mapping.get(current_page, current_page)
                        if target_page not in new_page_set:
                            continue
                        synchronized["physical_pdf_page"] = target_page
                    synchronized_items.append(synchronized)
                synchronized_items.sort(
                    key=lambda evidence: (
                        evidence.get("physical_pdf_page", 10**9)
                        if isinstance(evidence, dict)
                        else 10**9
                    )
                )
                updated_item["evidence_items"] = synchronized_items
                if "evidence_source_docs" in updated_item:
                    updated_item["evidence_source_docs"] = sorted(
                        {
                            evidence["source_doc_number"]
                            for evidence in synchronized_items
                            if isinstance(evidence, dict)
                            and isinstance(evidence.get("source_doc_number"), int)
                        }
                    )
            for field in ("intermediate_facts", "review_evidence_ledger"):
                if not isinstance(updated_item.get(field), list):
                    continue
                for fact in updated_item[field]:
                    if not isinstance(fact, dict):
                        continue
                    merged_page = fact.get("merged_page")
                    if isinstance(merged_page, int) and merged_page in page_mapping:
                        fact["merged_page"] = page_mapping[merged_page]
            provenance_detail = updated_item.get("evidence_provenance")
            if isinstance(provenance_detail, dict):
                contributions = provenance_detail.get("source_qa_contributions")
                if isinstance(contributions, list):
                    for contribution in contributions:
                        if not isinstance(contribution, dict) or not isinstance(
                            contribution.get("evidence_pages"), list
                        ):
                            continue
                        contribution["evidence_pages"] = sorted(
                            {
                                page_mapping.get(page, page)
                                for page in contribution["evidence_pages"]
                                if isinstance(page, int)
                                and not isinstance(page, bool)
                                and page_mapping.get(page, page) in new_page_set
                            }
                        )

            changed_fields = [
                name
                for name in ("question", "answer", "evidence_pages")
                if item.get(name) != updated_item.get(name)
            ]
            if not changed_fields and "options" not in item:
                raise ValueError("没有检测到需要保存的修改。")
            provenance = updated_item.setdefault("annotation_provenance", {})
            if isinstance(provenance, dict):
                edits = provenance.setdefault("manual_review_edits", [])
                if isinstance(edits, list):
                    edits.append(
                        {
                            "actor_username": (
                                actor["username"] if actor else "system"
                            ),
                            "edited_at": time.time(),
                            "changed_fields": changed_fields,
                            "before_evidence_pages": old_pages,
                            "after_evidence_pages": pages,
                        }
                    )

            event_id = uuid.uuid4().hex
            snapshot_path = (
                self.snapshot_dir
                / dataset_id.removesuffix(".json")
                / f"{time.time_ns()}_{event_id}.json"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, snapshot_path)
            qas[qa_id] = updated_item
            encoded = encode_dataset(dataset)
            atomic_write(path, encoded)
            try:
                self._refresh_final_manifest(dataset_id, dataset)
            except Exception:
                atomic_write(path, snapshot_path.read_bytes())
                if manifest_snapshot is not None:
                    atomic_write(self.final_manifest_path, manifest_snapshot)
                raise
            after_sha = sha256_bytes(encoded)
            self.registry.invalidate(dataset_id)
            timestamp = time.time()
            try:
                with self.connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    before_status = self._current_status(
                        connection, dataset_id, paper_id, qa_id
                    )
                    connection.execute(
                        """
                        INSERT INTO events(
                            event_id,dataset_id,paper_id,qa_id,action,
                            before_status,after_status,before_sha256,after_sha256,
                            snapshot_path,item_json,note,created_at,
                            actor_user_id,actor_username
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            event_id,
                            dataset_id,
                            paper_id,
                            qa_id,
                            "edit",
                            before_status,
                            "edited",
                            before_sha,
                            after_sha,
                            str(snapshot_path),
                            json.dumps(item, ensure_ascii=False),
                            None,
                            timestamp,
                            actor["user_id"] if actor else None,
                            actor["username"] if actor else "system",
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO reviews(dataset_id,paper_id,qa_id,status,updated_at)
                        VALUES(?,?,?,?,?)
                        ON CONFLICT(dataset_id,paper_id,qa_id)
                        DO UPDATE SET status=excluded.status,updated_at=excluded.updated_at
                        """,
                        (dataset_id, paper_id, qa_id, "edited", timestamp),
                    )
                    self._insert_audit(
                        connection,
                        "review.edit",
                        "qa",
                        f"{dataset_id}:{paper_id}:{qa_id}",
                        actor=actor,
                        details={
                            "dataset_id": dataset_id,
                            "paper_id": paper_id,
                            "qa_id": qa_id,
                            "changed_fields": changed_fields,
                            "before_evidence_pages": old_pages,
                            "after_evidence_pages": pages,
                            "before_sha256": before_sha,
                            "after_sha256": after_sha,
                        },
                    )
                    connection.commit()
            except Exception:
                atomic_write(path, snapshot_path.read_bytes())
                if manifest_snapshot is not None:
                    atomic_write(self.final_manifest_path, manifest_snapshot)
                raise
            return {
                "event_id": event_id,
                "dataset_id": dataset_id,
                "paper_id": paper_id,
                "qa_id": qa_id,
                "action": "edit",
                "status": "edited",
                "changed_fields": changed_fields,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "reversible": True,
            }

    def undo(
        self,
        dataset_id: str,
        event_id: str | None = None,
        *,
        actor: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self.registry.resolve(dataset_id)
        with _MUTATION_LOCK, self.connect() as connection:
            if event_id:
                row = connection.execute(
                    """
                    SELECT * FROM events
                    WHERE dataset_id=? AND event_id=? AND undone_at IS NULL
                      AND reversible=1
                    """,
                    (dataset_id, event_id),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM events
                    WHERE dataset_id=? AND undone_at IS NULL AND reversible=1
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (dataset_id,),
                ).fetchone()
            if row is None:
                raise KeyError("No reversible review operation found")
            if actor is not None and not self.has_role(actor, "admin"):
                if row["actor_user_id"] != actor.get("user_id"):
                    raise PermissionError(
                        "校验员只能撤销自己执行的审核操作。"
                    )
                self._require_item_permission(
                    dataset_id,
                    str(row["paper_id"]),
                    str(row["qa_id"]),
                    actor,
                    connection=connection,
                )
            _, current_raw, current_sha = load_dataset(path)
            if current_sha != row["after_sha256"]:
                raise RuntimeError(
                    "Only the latest dataset mutation can be undone safely"
                )
            snapshot_path = row["snapshot_path"]
            if row["action"] in {"delete", "edit"}:
                snapshot = Path(snapshot_path)
                if not snapshot.is_file():
                    raise RuntimeError("Review snapshot is missing")
                manifest_snapshot = self._final_manifest_snapshot(dataset_id)
                atomic_write(path, snapshot.read_bytes())
                try:
                    self._refresh_final_manifest(dataset_id)
                except Exception:
                    atomic_write(path, current_raw)
                    if manifest_snapshot is not None:
                        atomic_write(self.final_manifest_path, manifest_snapshot)
                    raise
                self.registry.invalidate(dataset_id)
            previous = row["before_status"]
            connection.execute("BEGIN IMMEDIATE")
            if previous is None:
                connection.execute(
                    """
                    DELETE FROM reviews
                    WHERE dataset_id=? AND paper_id=? AND qa_id=?
                    """,
                    (dataset_id, row["paper_id"], row["qa_id"]),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO reviews(dataset_id,paper_id,qa_id,status,updated_at)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(dataset_id,paper_id,qa_id)
                    DO UPDATE SET status=excluded.status,updated_at=excluded.updated_at
                    """,
                    (
                        dataset_id,
                        row["paper_id"],
                        row["qa_id"],
                        previous,
                        time.time(),
                    ),
                )
            connection.execute(
                """
                UPDATE events SET
                    undone_at=?,undone_by_user_id=?,undone_by_username=?
                WHERE event_id=?
                """,
                (
                    time.time(),
                    actor["user_id"] if actor else None,
                    actor["username"] if actor else "system",
                    row["event_id"],
                ),
            )
            result = {
                "undone_event_id": row["event_id"],
                "dataset_id": dataset_id,
                "paper_id": row["paper_id"],
                "qa_id": row["qa_id"],
                "restored_status": previous or "unreviewed",
            }
            self._insert_audit(
                connection,
                "review.undo",
                "qa",
                f"{dataset_id}:{result['paper_id']}:{result['qa_id']}",
                actor=actor,
                details=result,
            )
            connection.commit()
        return result

    def events(
        self,
        dataset_id: str | None = None,
        limit: int = 100,
        *,
        actor: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self.connect() as connection:
            if dataset_id:
                rows = connection.execute(
                    """
                    SELECT * FROM events WHERE dataset_id=?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (dataset_id, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        results = [
            {
                key: row[key]
                for key in row.keys()
                if key not in {"item_json"}
            }
            for row in rows
        ]
        for result in results:
            can_undo = not result["undone_at"] and bool(result["reversible"])
            if actor is not None and not self.has_role(actor, "admin"):
                can_undo = can_undo and result["actor_user_id"] == actor.get("user_id")
                if can_undo:
                    can_undo = self.assignment_for_item(
                        str(result["dataset_id"]),
                        str(result["paper_id"]),
                        str(result["qa_id"]),
                        str(actor["user_id"]),
                    ) is not None
            result["can_undo"] = can_undo
        return results
