"""Shared classification contract for the four final-2200 QA files."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


CORE_QA_FIELDS = (
    "question",
    "answer",
    "evidence_pages",
    "modal_types",
    "question_type",
    "question_category",
)
CANONICAL_MODALITIES = ("text", "image", "table", "formula")
MODALITY_ALIASES = {
    "figure": "image",
    "fig": "image",
    "equation": "formula",
    "math": "formula",
}
QUESTION_TYPES = {"Literal", "Inferential"}

CATEGORY_TAXONOMY = {
    "Computer Science": (
        "Algorithm & Architecture Detail",
        "Experiment & Result Validation",
        "Method Innovation",
    ),
    "Economics": (
        "Theoretical Framework & Concept Definition",
        "Empirical Design & Econometric Method",
        "Causal Inference & Result Interpretation",
    ),
    "Electrical Engineering and Systems Science": (
        "System Architecture & Signal Processing Method",
        "Experiment & Performance Test",
        "System Optimization & Engineering Implementation",
    ),
    "Mathematics": (
        "Definition & Theorem Statement",
        "Formula & Derivation Detail",
        "Proposition Proof & Logical Reasoning",
    ),
    "Physics": (
        "Physical Concept & Theoretical Model",
        "Computation & Experimental Validation",
        "Computation & Experimental Validation",
    ),
    "Quantitative Biology": (
        "Biological Entity & Quantitative Model Definition",
        "Experimental Data & Statistical Analysis",
        "Biological Network & Dynamic Modeling",
    ),
    "Quantitative Finance": (
        "Financial Theory & Pricing Model Definition",
        "Quantitative Strategy & Performance Analysis",
        "Quantitative Strategy & Performance Analysis",
    ),
    "Statistics": (
        "Statistical Concept & Probability Model Definition",
        "Statistical Inference & Method Application",
        "Statistical Inference & Method Application",
    ),
}

CATEGORY_DEFINITIONS = {
    "Algorithm & Architecture Detail": "模型架构、算法流程、核心模块和实现逻辑。",
    "Experiment & Result Validation": "实验设置、数据集、指标、消融和结果对比。",
    "Method Innovation": "方法创新、适用场景、差异、限制和方法推理。",
    "Theoretical Framework & Concept Definition": "经济理论、概念、模型假设和研究范围。",
    "Empirical Design & Econometric Method": "识别策略、计量模型、数据、变量和工具变量。",
    "Causal Inference & Result Interpretation": "因果效应、回归结果、稳健性和内生性解释。",
    "System Architecture & Signal Processing Method": "系统结构、信号处理、模块和协议逻辑。",
    "Experiment & Performance Test": "工程实验、仿真、硬件参数和性能测试。",
    "System Optimization & Engineering Implementation": "系统优化、实现、瓶颈和部署约束。",
    "Definition & Theorem Statement": "数学定义、公理、定理和引理陈述。",
    "Formula & Derivation Detail": "公式、符号、推导步骤和不等式缩放。",
    "Proposition Proof & Logical Reasoning": "证明逻辑、命题推导、条件、反例和边界。",
    "Physical Concept & Theoretical Model": "物理概念、理论模型、假设和物理量定义。",
    "Computation & Experimental Validation": "物理计算、量纲、实验设计和理论实验对照。",
    "Biological Entity & Quantitative Model Definition": "生物实体、网络、定量模型和过程描述。",
    "Experimental Data & Statistical Analysis": "生物实验数据、统计检验和差异分析。",
    "Biological Network & Dynamic Modeling": "网络动力学、拟合、敏感性和生物解释。",
    "Financial Theory & Pricing Model Definition": "金融理论、定价模型、假设和市场机制。",
    "Quantitative Strategy & Performance Analysis": "量化策略、收益风险指标、回测和尾部风险。",
    "Statistical Concept & Probability Model Definition": "统计概念、分布、模型假设和统计量。",
    "Statistical Inference & Method Application": "估计、检验、区间、拟合和方法应用。",
}

ANALYTICAL_REASONING_TYPES = {
    "conflict_resolution",
    "compatibility_judgment",
    "method_transfer",
}
EMPIRICAL_REASONING_TYPES = {"metric_reasoning"}
STRUCTURAL_REASONING_TYPES = {
    "component_hierarchy",
    "adjacent_module_dependency",
}


def canonical_modal_types(values: Any) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("modal_types must be a non-empty list")
    normalized = {
        MODALITY_ALIASES.get(str(value).strip().casefold(), str(value).strip().casefold())
        for value in values
    }
    invalid = normalized - set(CANONICAL_MODALITIES)
    if invalid:
        raise ValueError(f"unsupported modal_types: {sorted(invalid)}")
    return [value for value in CANONICAL_MODALITIES if value in normalized]


def category_for_role(primary_category: str, role: str) -> str:
    categories = CATEGORY_TAXONOMY.get(primary_category)
    if categories is None:
        raise ValueError(f"unknown primary_category: {primary_category}")
    index = {"structural": 0, "empirical": 1, "analytical": 2}[role]
    return categories[index]


def category_for_reasoning_type(primary_category: str, reasoning_type: str) -> str:
    if reasoning_type in STRUCTURAL_REASONING_TYPES:
        role = "structural"
    elif reasoning_type in EMPIRICAL_REASONING_TYPES:
        role = "empirical"
    elif reasoning_type in ANALYTICAL_REASONING_TYPES:
        role = "analytical"
    else:
        raise ValueError(f"unsupported cross-PDF reasoning_type: {reasoning_type}")
    return category_for_role(primary_category, role)


def normalize_nonstandard_category(primary_category: str, value: str) -> str:
    allowed = set(CATEGORY_TAXONOMY.get(primary_category, ()))
    if value in allowed:
        return value
    lowered = value.casefold()
    if any(token in lowered for token in ("experiment", "statistical", "claim")):
        role = "empirical"
    elif any(
        token in lowered
        for token in ("method", "theory", "failure", "synthesis", "comparison", "research")
    ):
        role = "analytical"
    else:
        role = "analytical"
    return category_for_role(primary_category, role)


def representative_source_category(
    source_categories: Iterable[str], primary_category: str
) -> str:
    ordered = list(source_categories)
    allowed = set(CATEGORY_TAXONOMY.get(primary_category, ()))
    if not ordered or any(value not in allowed for value in ordered):
        raise ValueError(
            f"invalid source question categories for {primary_category}: {ordered}"
        )
    counts = Counter(ordered)
    maximum = max(counts.values())
    return next(value for value in ordered if counts[value] == maximum)


def ordered_qa(qa: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: qa[field] for field in CORE_QA_FIELDS},
        **{key: value for key, value in qa.items() if key not in CORE_QA_FIELDS},
    }


def validate_final_dataset(dataset: dict[str, Any], *, label: str) -> None:
    if not isinstance(dataset, dict) or not dataset:
        raise ValueError(f"{label}: dataset must be a non-empty object")
    for paper_id, paper in dataset.items():
        if not isinstance(paper, dict):
            raise ValueError(f"{label}:{paper_id}: paper must be an object")
        if str(paper.get("paper")) != str(paper_id):
            raise ValueError(f"{label}:{paper_id}: paper field does not match key")
        primary = paper.get("primary_category")
        secondary = paper.get("secondary_category")
        if primary not in CATEGORY_TAXONOMY or not isinstance(secondary, str) or not secondary:
            raise ValueError(f"{label}:{paper_id}: missing or invalid discipline metadata")
        qas = paper.get("QA")
        if not isinstance(qas, dict) or not qas:
            raise ValueError(f"{label}:{paper_id}: QA must be a non-empty object")
        for qa_id, qa in qas.items():
            if not isinstance(qa, dict):
                raise ValueError(f"{label}:{paper_id}/{qa_id}: QA must be an object")
            missing = [field for field in CORE_QA_FIELDS if field not in qa]
            if missing:
                raise ValueError(f"{label}:{paper_id}/{qa_id}: missing {missing}")
            if not isinstance(qa["question"], str) or not qa["question"].strip():
                raise ValueError(f"{label}:{paper_id}/{qa_id}: invalid question")
            if not isinstance(qa["answer"], str) or not qa["answer"].strip():
                raise ValueError(f"{label}:{paper_id}/{qa_id}: invalid answer")
            pages = qa["evidence_pages"]
            if (
                not isinstance(pages, list)
                or pages != sorted(set(pages))
                or any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in pages)
            ):
                raise ValueError(f"{label}:{paper_id}/{qa_id}: invalid evidence_pages")
            if qa["modal_types"] != canonical_modal_types(qa["modal_types"]):
                raise ValueError(f"{label}:{paper_id}/{qa_id}: non-canonical modal_types")
            if qa["question_type"] not in QUESTION_TYPES:
                raise ValueError(f"{label}:{paper_id}/{qa_id}: invalid question_type")
            if qa["question_category"] not in CATEGORY_TAXONOMY[primary]:
                raise ValueError(f"{label}:{paper_id}/{qa_id}: invalid question_category")
            if qa["answer"] == "Unanswerable" and pages:
                raise ValueError(f"{label}:{paper_id}/{qa_id}: refusal has evidence pages")
            if qa["answer"] != "Unanswerable" and not pages:
                raise ValueError(f"{label}:{paper_id}/{qa_id}: answerable QA has no evidence")
