#!/usr/bin/env python3
"""Typed short-answer scoring helpers for single-PDF QA evaluation."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from eval_framework import normalize_loose_text


NUMBER_RE = re.compile(
    r"^[\s$€£¥]*([+-]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?(?:e[+-]?\d+)?)"
    r"\s*([A-Za-z°%][A-Za-z0-9°%./^-]*)?\s*$",
    re.IGNORECASE,
)


def normalized_string(value: Any) -> str:
    return normalize_loose_text(str(value or ""))


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls(reference: str, prediction: str) -> float:
    left = normalized_string(reference)
    right = normalized_string(prediction)
    if not left and not right:
        return 1.0
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    similarity = 1.0 - levenshtein_distance(left, right) / denominator
    return similarity if similarity >= 0.5 else 0.0


def parse_quantity(value: Any) -> tuple[float, str | None] | None:
    text = str(value or "").strip()
    match = NUMBER_RE.fullmatch(text)
    if not match or not match.group(1):
        return None
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = match.group(2).lower() if match.group(2) else None
    return number, unit


def split_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    return [
        item.strip()
        for item in re.split(r"\s*(?:;|\n|,\s+(?=[A-Za-z0-9]))\s*", text)
        if item.strip()
    ]


def string_alias_match(
    reference: str,
    prediction: str,
    aliases: list[Any] | None,
) -> bool:
    accepted = [reference, *(aliases or [])]
    pred = normalized_string(prediction)
    return any(normalized_string(candidate) == pred for candidate in accepted)


def typed_answer_match(
    reference: Any,
    prediction: Any,
    *,
    answer_format: str | None = None,
    aliases: list[Any] | None = None,
    expected_unit: str | None = None,
    tolerance: dict[str, Any] | None = None,
    string_metric: str = "exact_or_alias",
) -> tuple[bool | None, str, float | None]:
    """Return (decision, method, score).

    ``decision=None`` deliberately falls through to the semantic LLM judge.
    """
    fmt = str(answer_format or "String").strip().lower()
    ref = str(reference or "").strip()
    pred = str(prediction or "").strip()

    if not pred:
        return False, "typed_empty_prediction", 0.0
    if fmt == "unanswerable":
        # This is a controlled abstention label, not a semantic short answer.
        # Requiring the canonical spelling prevents vague phrases such as
        # "not mentioned" from being silently converted into a refusal.
        matched = pred == "Unanswerable"
        return matched, "typed_unanswerable_exact", float(matched)

    if fmt in {"integer", "float"}:
        ref_quantity = parse_quantity(ref)
        pred_quantity = parse_quantity(pred)
        if ref_quantity is None or pred_quantity is None:
            return None, "typed_numeric_needs_llm", None
        ref_value, ref_unit = ref_quantity
        pred_value, pred_unit = pred_quantity
        unit = expected_unit.lower() if expected_unit else ref_unit
        if unit and pred_unit != unit:
            return False, "typed_numeric_unit_mismatch", 0.0
        if ref_unit and pred_unit and ref_unit != pred_unit:
            return False, "typed_numeric_unit_mismatch", 0.0
        if fmt == "integer":
            matched = (
                ref_value.is_integer()
                and pred_value.is_integer()
                and int(ref_value) == int(pred_value)
            )
            return matched, "typed_integer_exact", float(matched)

        rule = tolerance or {"type": "relative", "value": 0.01}
        value = float(rule.get("value", 0.01))
        if rule.get("type") == "absolute":
            matched = math.isclose(ref_value, pred_value, abs_tol=value, rel_tol=0)
        else:
            matched = math.isclose(ref_value, pred_value, rel_tol=value, abs_tol=1e-12)
        return matched, "typed_float_tolerance", float(matched)

    if fmt == "list":
        ref_items = split_list(reference)
        pred_items = split_list(prediction)
        if len(ref_items) != len(pred_items):
            return False, "typed_list_length_mismatch", 0.0
        remaining = list(pred_items)
        for ref_item in ref_items:
            match_index = next(
                (
                    index
                    for index, pred_item in enumerate(remaining)
                    if normalized_string(ref_item) == normalized_string(pred_item)
                ),
                None,
            )
            if match_index is None:
                return None, "typed_list_needs_llm", None
            remaining.pop(match_index)
        return True, "typed_list_exact", 1.0

    if string_alias_match(ref, pred, aliases):
        return True, "typed_string_exact_or_alias", 1.0
    if string_metric == "anls":
        score = max(
            anls(str(candidate), pred)
            for candidate in [ref, *(aliases or [])]
        )
        return score >= 0.5, "typed_string_anls", score
    return None, "typed_string_needs_llm", None
