"""Scoring kernel copied verbatim from the sxz v4 experiment source.

Only portable orchestration lives outside this module. Do not edit the scoring
functions: source hashes and parity tests bind them to the experiment.
"""

from __future__ import annotations
import os
import re
import json
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable

FIXED_DENOMINATORS = {
    "ordinary1190": 1200,
    "cross_old400": 400,
    "cross_hard400": 400,
    "reasoning_old100": 100,
    "reasoning_hard100": 100,
}


ORDINARY_SPLIT_DENOMINATORS = {
    "ordinary1190_answerable": 1000,
    "ordinary1190_unanswerable": 200,
}


SEED = 42


ENABLE_THINKING = False


JUDGE_MAX_NEW_TOKENS = 32


QWEN36_JUDGE_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models/Qwen3.6-27B")

JUDGE_RULES = r"""
You are performing frozen semantic answer review for a PDF-QA benchmark.

Apply exactly the same standard to every model, dataset, and review session.

You will receive:
- Dataset name
- Question
- Gold answer
- Model prediction answer

Judge ONLY the semantic answer.
Evidence pages are evaluated separately by deterministic code and MUST NOT
influence the answer verdict.

Return exactly ONE label:
CORRECT
PARTIAL
WRONG

============================================================
A. CENTRAL CALIBRATION RULE: GOLD IS A REFERENCE, NOT A CHECKLIST
============================================================

First determine what the QUESTION explicitly requires.
Then use the Gold answer as the semantic reference for those required parts.

The prediction does NOT need to reproduce every detail, example, implementation
detail, explanatory sentence, parenthetical remark, number, method name, or
supporting fact that happens to appear in the Gold answer.

A prediction may be much shorter than the Gold answer and still be CORRECT if
it accurately and sufficiently answers every aspect explicitly required by the
question.

Do NOT downgrade an answer merely because the Gold answer is more detailed.

However, do not accept a generic restatement of the question. The prediction
must provide the requested information, distinction, relation, mechanism,
entity, value, direction, condition, or conclusion.

============================================================
B. CORRECT
============================================================

Output CORRECT when:
- every semantically necessary component explicitly requested by the question
  is answered correctly;
- the answer reaches the same requested conclusion as Gold;
- no material statement in the prediction contradicts the Gold answer.

For comparison questions:
- both requested sides must be addressed;
- the essential requested contrast must be correct;
- supporting Gold details that the question does not require may be omitted.

For "how" / mechanism questions:
- the essential mechanism or process asked for must be stated;
- incidental implementation details present only in Gold are not mandatory.

For yes/no or existence questions:
- the correct yes/no conclusion can be sufficient when that is what the
  question asks, even if Gold additionally explains why.

For numerical/entity questions:
- the requested value/entity must be correct.
- extra Gold context is not mandatory unless the question asks for it.

For multi-hop/reasoning questions:
- a bridge entity, condition, intermediate relation, causal step, comparison
  side, or document-specific relation is mandatory ONLY when the wording of
  the question actually requires that component to answer the target.
- do not require hidden intermediate reasoning that is not requested in the
  answer itself.

============================================================
C. PARTIAL
============================================================

Output PARTIAL when the main answer/direction is substantially correct but at
least one EXPLICITLY REQUIRED component is missing or locally wrong, while the
response still demonstrates meaningful correct understanding.

Typical PARTIAL cases:
- one required comparison side is incomplete;
- a required condition/qualification is missing;
- the requested mechanism is only partly specified;
- one required sub-answer is omitted;
- a minor numerical/formula/detail error occurs while the main requested
  conclusion remains correct;
- the answer correctly identifies the broad distinction but does not provide
  enough information to satisfy the question as written.

Do NOT use PARTIAL just because the prediction is less detailed than Gold.

============================================================
D. WRONG
============================================================

Output WRONG when:
- the core requested conclusion is wrong;
- the requested entity/value/sign/direction/ordering is wrong;
- a central requested mechanism or causal/comparative relation is wrong;
- documents, methods, or roles are materially mismatched;
- the answer is unrelated, fabricated, or contradicted by Gold;
- the response is so generic/vacuous that it does not actually answer the
  requested target;
- it merely repeats information already stated in the question without
  supplying the requested answer;
- an answerable question receives no substantive semantic answer.

============================================================
E. DATASET ANSWERABILITY
============================================================

Ordinary (ordinary1190):
- Gold = Unanswerable and prediction = Unanswerable -> CORRECT.
- Gold = Unanswerable but prediction gives a substantive answer -> WRONG.
- Gold is answerable but prediction says Unanswerable -> WRONG.

Cross Old / Cross Hard / Reasoning Old / Reasoning Hard:
- every item is answerable;
- any prediction of Unanswerable -> WRONG.

============================================================
F. FINAL CONSISTENCY CHECK
============================================================

Before selecting the label, check:

1. What exact information does the QUESTION ask for?
2. Did the prediction answer all of those required parts?
3. Is any required part materially wrong?
4. Am I penalizing the prediction only because Gold contains extra detail?
   If yes, do NOT downgrade for that reason.
5. Am I accepting a vague paraphrase that does not actually answer the target?
   If yes, do NOT mark it CORRECT.

Use CORRECT for a concise but sufficient answer.
Use PARTIAL for a substantively right but explicitly incomplete answer.
Use WRONG for a materially wrong, unrelated, or non-answer.

Output exactly one label and nothing else:
CORRECT
PARTIAL
WRONG
"""


JUDGE_RULES_SHA256 = hashlib.sha256(
    JUDGE_RULES.encode("utf-8")
).hexdigest()


SAVE_JUDGE_RAW = True


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|assistant|>",
    "<|user|>",
]


INVALID_ANSWER_PATTERNS = [
    r"^\s*$",
    r"^\s*none\s*$",
    r"^\s*null\s*$",
    r"^\s*nan\s*$",
    r"^\s*error\s*:",
    r"outofmemoryerror",
    r"cuda out of memory",
    r"traceback \(most recent call last\)",
    r"runtimeerror",
    r"exception",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_space(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(s).replace("\u00a0", " "),
    ).strip()


def compact_json(x: Any) -> str:
    return json.dumps(
        x,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def stable_key(*parts: Any) -> str:
    text = "\n---\n".join(str(x) for x in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _combine_multiple_json_values(values: List[Any], path: Path) -> Any:
    """
    Normalize files that contain multiple top-level JSON values.

    Supported cases:
    1) JSONL / NDJSON: one QA dict per line -> return list[dict].
    2) Multiple top-level list chunks -> flatten them.
    3) Multiple dataset-mapping dict chunks such as
         {paper_id: {"QA": {...}}}
       -> merge them when top-level keys do not collide.

    We never silently overwrite duplicate dataset keys.
    """
    if not values:
        return []

    if len(values) == 1:
        return values[0]

    # Multiple list chunks -> flatten.
    if all(isinstance(v, list) for v in values):
        out: List[Any] = []
        for v in values:
            out.extend(v)
        return out

    # Multiple canonical dataset-mapping chunks -> merge safely.
    if all(isinstance(v, dict) for v in values):
        def looks_like_dataset_mapping(d: Dict[str, Any]) -> bool:
            if not d:
                return False
            qa_units = 0
            for obj in d.values():
                if isinstance(obj, dict) and isinstance(obj.get("QA"), dict):
                    qa_units += 1
            return qa_units > 0

        if all(looks_like_dataset_mapping(v) for v in values):
            merged: Dict[str, Any] = {}
            for chunk in values:
                for key, value in chunk.items():
                    if key in merged:
                        raise ValueError(
                            f"Multiple top-level JSON chunks in {path} contain "
                            f"duplicate dataset key {key!r}; refusing to overwrite."
                        )
                    merged[key] = value
            return merged

        # Most .json files that trigger Extra data here are actually JSONL:
        # one complete QA/result dict per line/object.
        return values

    # Mixed top-level values are unusual, but returning a list keeps every
    # object rather than discarding data. iter_qa_records() will validate shape.
    return values


def load_json(path: Path) -> Any:
    """
    Read either:
      - normal JSON,
      - JSONL / NDJSON,
      - concatenated top-level JSON values.

    Some Claude result files use a .json suffix but contain one complete JSON
    object per line. Standard json.load() raises JSONDecodeError: Extra data on
    those files, so we fall back to JSONDecoder.raw_decode() and keep all
    top-level values.
    """
    text = path.read_text(encoding="utf-8-sig")

    if not text.strip():
        raise ValueError(f"Empty JSON/result file: {path}")

    try:
        return json.loads(text)
    except json.JSONDecodeError as first_exc:
        decoder = json.JSONDecoder()
        values: List[Any] = []
        idx = 0
        n = len(text)

        while idx < n:
            # Skip whitespace between complete JSON values / JSONL records.
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break

            try:
                value, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Failed normal JSON parse ({first_exc.msg}); also failed "
                    f"multi-value/JSONL parse for {path}: {exc.msg}",
                    text,
                    exc.pos,
                ) from exc

            values.append(value)
            idx = end

        if not values:
            raise first_exc

        print(
            f"[Flexible JSON] {path} | detected {len(values)} top-level "
            "JSON value(s); treating as JSONL/concatenated JSON.",
            flush=True,
        )
        return _combine_multiple_json_values(values, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp, path)


def iter_qa_records(
    data: Any,
) -> Iterable[
    Tuple[
        str,
        str,
        Dict[str, Any],
        Dict[str, Any],
    ]
]:
    if isinstance(data, dict):
        yielded = False

        # Normal benchmark structure.
        for unit_id, unit in data.items():
            if not isinstance(unit, dict):
                continue

            qa_dict = unit.get("QA")
            if not isinstance(qa_dict, dict):
                continue

            for qa_id, qa in qa_dict.items():
                if not isinstance(qa, dict):
                    continue

                yielded = True
                yield (
                    str(unit_id),
                    str(qa_id),
                    unit,
                    qa,
                )

        if yielded:
            return

        # Some tools wrap the actual result under one of these keys.
        for key in (
            "data",
            "results",
            "items",
            "records",
        ):
            if key in data:
                yield from iter_qa_records(data[key])
                return

        raise ValueError(
            "Unsupported dict result structure: "
            "no unit['QA'] records and no known wrapper key."
        )

    if isinstance(data, list):
        for idx, qa in enumerate(data):
            if not isinstance(qa, dict):
                continue

            # A JSONL file may contain wrapper/dataset chunks rather than
            # direct QA rows. Recurse into those shapes instead of treating
            # the wrapper itself as one QA sample.
            if isinstance(qa.get("QA"), dict) or any(
                key in qa for key in ("data", "results", "items", "records")
            ):
                try:
                    yield from iter_qa_records(qa)
                    continue
                except ValueError:
                    pass

            unit_id = str(
                qa.get(
                    "paper_id",
                    qa.get(
                        "paper",
                        qa.get(
                            "unit_id",
                            qa.get(
                                "doc_id",
                                "",
                            ),
                        ),
                    ),
                )
            )

            qa_id = str(
                qa.get(
                    "qa_id",
                    qa.get(
                        "id",
                        idx,
                    ),
                )
            )

            unit = {
                "paper": unit_id,
                "primary_category": qa.get(
                    "primary_category",
                    "",
                ),
                "secondary_category": qa.get(
                    "secondary_category",
                    "",
                ),
                "pdf_path": qa.get(
                    "pdf_path",
                    "",
                ),
            }

            yield (
                unit_id,
                qa_id,
                unit,
                qa,
            )
        return

    raise ValueError(
        f"Unsupported JSON structure: {type(data)}"
    )


def get_item_uid(
    dataset_id: str,
    unit_id: str,
    qa_id: str,
    qa: Dict[str, Any],
) -> str:
    return str(
        qa.get("item_uid")
        or qa.get("uid")
        or ""
    ).strip()


def build_alignment_key(
    dataset_id: str,
    unit_id: str,
    qa_id: str,
    qa: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Priority chosen for cross-model compatibility:

    1) unit_id + qa_id:
       Most result files retain these benchmark-native identifiers.

    2) item_uid:
       Used only when unit/QA IDs are unavailable.

    3) normalized question hash:
       Last-resort alignment fallback.
    """
    uid = get_item_uid(
        dataset_id,
        unit_id,
        qa_id,
        qa,
    )

    if str(unit_id).strip() and str(qa_id).strip():
        return (
            f"unitqa::{unit_id}::{qa_id}",
            "unit_id+qa_id",
        )

    if uid:
        return (
            f"uid::{uid}",
            "item_uid",
        )

    question = normalize_space(
        qa.get("question", "")
    )
    if question:
        return (
            "question_sha256::"
            + hashlib.sha256(
                question.encode("utf-8")
            ).hexdigest(),
            "question_sha256",
        )

    raise ValueError(
        f"Cannot align sample in {dataset_id}: "
        f"unit_id={unit_id!r}, qa_id={qa_id!r}, "
        "no item_uid and no question."
    )


def get_gold_answer_from_qa(
    qa: Dict[str, Any],
) -> Tuple[str, str]:
    for key in (
        "answer",
        "gold_answer",
        "reference_answer",
    ):
        if key in qa:
            return (
                normalize_space(qa.get(key, "")),
                key,
            )

    return "", "missing"


def extract_pages(value: Any) -> List[int]:
    """
    Parse page-number-like values.

    Intended ONLY for evidence-page fields / fragments, never arbitrary
    answer text.
    """
    pages: List[int] = []

    def add_int(x: Any) -> None:
        try:
            if isinstance(x, bool):
                return
            n = int(x)
            if n > 0:
                pages.append(n)
        except Exception:
            return

    if value is None:
        return []

    if isinstance(value, int):
        add_int(value)

    elif isinstance(value, float):
        if value.is_integer():
            add_int(value)

    elif isinstance(
        value,
        (list, tuple, set),
    ):
        for item in value:
            pages.extend(
                extract_pages(item)
            )

    elif isinstance(value, dict):
        matched = False

        for key in (
            "evidence_pages_pre",
            "pred_evidence_pages",
            "predicted_evidence_pages",
            "evidence_pages",
            "gold_evidence_pages",
            "pages",
        ):
            if key in value:
                pages.extend(
                    extract_pages(
                        value.get(key)
                    )
                )
                matched = True
                break

        if not matched:
            return []

    else:
        text = str(value)

        # Explicit labels first.
        for m in re.finditer(
            r"(?:PDF_PAGE_|PAGE_|Page\s*|page\s*[:=]?\s*)(\d+)",
            text,
            flags=re.I,
        ):
            add_int(
                m.group(1)
            )

        # Plain integers, because this function is only used on page fields.
        for m in re.finditer(
            r"(?<![A-Za-z0-9_.-])(\d{1,4})(?![A-Za-z0-9_.-])",
            text,
        ):
            add_int(
                m.group(1)
            )

    # Unique, preserve order.
    seen = set()
    out: List[int] = []

    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return out


def get_gold_pages_from_qa(
    qa: Dict[str, Any],
) -> Tuple[List[int], str]:
    for key in (
        "evidence_pages",
        "gold_evidence_pages",
    ):
        if key in qa:
            return (
                extract_pages(
                    qa.get(key)
                ),
                key,
            )

    return [], "missing"


def strip_code_fence(s: str) -> str:
    s = s.strip()

    if s.startswith("```"):
        s = re.sub(
            r"^```(?:json|JSON)?\s*",
            "",
            s,
        )
        s = re.sub(
            r"\s*```$",
            "",
            s,
        )

    return s.strip()


def clean_model_text(s: Any) -> str:
    if s is None:
        return ""

    if isinstance(
        s,
        (dict, list),
    ):
        return json.dumps(
            s,
            ensure_ascii=False,
        )

    s = str(s)

    for tok in SPECIAL_TOKENS:
        s = s.replace(
            tok,
            "",
        )

    return strip_code_fence(s)


def safe_json_loads(
    s: str,
) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_json_field_string(
    text: str,
    field: str,
) -> Optional[str]:
    key_pat = re.compile(
        r'"'
        + re.escape(field)
        + r'"\s*:\s*"'
    )

    m = key_pat.search(text)

    if not m:
        return None

    i = m.end()
    out: List[str] = []
    escaped = False

    while i < len(text):
        ch = text[i]

        if escaped:
            if ch == "n":
                out.append("\n")
            elif ch == "t":
                out.append("\t")
            elif ch == "r":
                out.append("\r")
            elif ch in [
                '"',
                "\\",
                "/",
            ]:
                out.append(ch)
            else:
                out.append(ch)

            escaped = False

        else:
            if ch == "\\":
                escaped = True

            elif ch == '"':
                return "".join(out).strip()

            else:
                out.append(ch)

        i += 1

    return (
        "".join(out).strip()
        if out
        else None
    )


def parse_answer_pre_value(
    raw: Any,
) -> Tuple[str, str]:
    if isinstance(raw, dict):
        ans = raw.get(
            "answer_pre",
            raw.get(
                "pred_answer",
                raw.get(
                    "answer",
                    "",
                ),
            ),
        )

        return (
            normalize_space(ans),
            "dict",
        )

    text = clean_model_text(raw)

    if not text:
        return "", "empty"

    obj = safe_json_loads(text)

    if isinstance(obj, dict):
        ans = obj.get(
            "answer_pre",
            obj.get(
                "pred_answer",
                obj.get(
                    "answer",
                    "",
                ),
            ),
        )

        return (
            normalize_space(ans),
            "json",
        )

    first = text.find("{")
    last = text.rfind("}")

    if first >= 0 and last > first:
        obj2 = safe_json_loads(
            text[first:last + 1]
        )

        if isinstance(obj2, dict):
            ans = obj2.get(
                "answer_pre",
                obj2.get(
                    "pred_answer",
                    obj2.get(
                        "answer",
                        "",
                    ),
                ),
            )

            return (
                normalize_space(ans),
                "json_extracted",
            )

    field_ans = extract_json_field_string(
        text,
        "answer_pre",
    )

    if field_ans is not None:
        return (
            normalize_space(field_ans),
            "field_scan",
        )

    return (
        normalize_space(text),
        "raw_text",
    )


def _recover_answer_from_prediction_container(
    raw: Any,
    depth: int = 0,
) -> Tuple[str, str]:
    """
    Deterministic mechanical recovery ONLY.

    This function is called only on fields that are already designated as
    prediction/raw-output containers. It never searches qa["answer"] or other
    gold/reference fields.

    Supports common local/API output shapes:
      - plain text
      - JSON text containing answer_pre / pred_answer / final_answer / answer
      - dict output containers
      - OpenAI-style choices[0].message.content / choices[0].text
      - Anthropic-style content=[{"type":"text","text":"..."}]
      - nested response/output/result/message/generation/completion containers
    """
    if depth > 6 or raw is None:
        return "", "empty"

    if isinstance(raw, str):
        return parse_answer_pre_value(raw)

    if isinstance(raw, (int, float, bool)):
        return normalize_space(raw), "scalar"

    if isinstance(raw, list):
        # Anthropic-style content blocks.
        text_parts = []
        for item in raw:
            if isinstance(item, dict):
                item_type = normalize_space(
                    item.get("type", "")
                ).lower()
                if item_type in {"text", "output_text"} and "text" in item:
                    txt = normalize_space(item.get("text", ""))
                    if txt:
                        text_parts.append(txt)

        if text_parts:
            return "\n".join(text_parts), "content_blocks"

        # Otherwise inspect list entries in order and take the first
        # deterministically recoverable answer.
        for item in raw:
            ans, status = _recover_answer_from_prediction_container(
                item,
                depth + 1,
            )
            if ans:
                return ans, f"list:{status}"

        return "", "empty_list"

    if isinstance(raw, dict):
        # Direct semantic-answer keys inside a prediction container.
        direct_answer_keys = (
            "answer_pre",
            "pred_answer",
            "prediction",
            "model_answer",
            "final_answer",
            "generated_answer",
            "answer",
        )

        for key in direct_answer_keys:
            if key not in raw:
                continue

            ans, status = _recover_answer_from_prediction_container(
                raw.get(key),
                depth + 1,
            )
            if ans:
                return ans, f"{key}:{status}"

        # OpenAI-compatible choices.
        choices = raw.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                if "message" in choice:
                    ans, status = _recover_answer_from_prediction_container(
                        choice.get("message"),
                        depth + 1,
                    )
                    if ans:
                        return ans, f"choices.message:{status}"

                if "text" in choice:
                    ans, status = _recover_answer_from_prediction_container(
                        choice.get("text"),
                        depth + 1,
                    )
                    if ans:
                        return ans, f"choices.text:{status}"

        # Common explicit text fields INSIDE an already-known output container.
        text_keys = (
            "generated_text",
            "output_text",
            "text",
            "content",
        )

        for key in text_keys:
            if key not in raw:
                continue

            ans, status = _recover_answer_from_prediction_container(
                raw.get(key),
                depth + 1,
            )
            if ans:
                return ans, f"{key}:{status}"

        # Common nested output containers.
        nested_keys = (
            "response",
            "raw_response",
            "raw_output",
            "output",
            "result",
            "message",
            "generation",
            "completion",
            "data",
        )

        for key in nested_keys:
            if key not in raw:
                continue

            ans, status = _recover_answer_from_prediction_container(
                raw.get(key),
                depth + 1,
            )
            if ans:
                return ans, f"{key}:{status}"

        return "", "empty_dict"

    return "", "unsupported_container"


def get_pred_answer_from_qa(
    qa: Dict[str, Any],
) -> Tuple[str, str, str, Any]:
    """
    Deterministic prediction recovery.

    IMPORTANT:
    qa["answer"], qa["gold_answer"], and qa["reference_answer"] are NEVER used
    as model predictions.

    The function first checks normalized prediction fields, then raw-generation
    fields. If all recoveries fail, the sample remains a technical failure.
    """
    candidate_keys = (
        # normalized prediction fields
        "answer_pre",
        "pred_answer",
        "prediction",
        "model_answer",
        "final_answer",
        "generated_answer",

        # preserved/raw prediction fields
        "answer_pre_raw",
        "raw_answer",
        "model_response",
        "assistant_response",
        "generated_text",
        "output_text",
        "generation",
        "completion",
        "response",
        "raw_output",
        "raw_response",
        "output",
        "result",
    )

    first_seen = (
        "",
        "missing",
        "",
        None,
    )

    for key in candidate_keys:
        if key not in qa:
            continue

        raw = qa.get(key)

        ans, status = _recover_answer_from_prediction_container(
            raw
        )

        ans = normalize_space(ans)

        if ans:
            return (
                ans,
                status,
                key,
                raw,
            )

        if first_seen[1] == "missing":
            first_seen = (
                "",
                status,
                key,
                raw,
            )

    return first_seen


def is_illegal_answer(
    ans: str,
    raw: Any = None,
) -> bool:
    text = clean_model_text(
        raw
        if raw is not None
        else ans
    )

    ans2 = normalize_space(ans)

    combined = (
        text
        + "\n"
        + ans2
    ).strip().lower()

    for pat in INVALID_ANSWER_PATTERNS:
        if re.search(
            pat,
            combined,
            flags=re.I,
        ):
            return True

    return False


def extract_pages_from_jsonish(
    raw: Any,
) -> List[int]:
    if isinstance(raw, dict):
        for key in (
            "evidence_pages_pre",
            "pred_evidence_pages",
            "predicted_evidence_pages",
            "evidence_pages",
        ):
            if key in raw:
                return extract_pages(
                    raw.get(key)
                )

        return []

    text = clean_model_text(raw)

    if not text:
        return []

    obj = safe_json_loads(text)

    if isinstance(obj, dict):
        return extract_pages_from_jsonish(obj)

    first = text.find("{")
    last = text.rfind("}")

    if first >= 0 and last > first:
        obj2 = safe_json_loads(
            text[first:last + 1]
        )

        if isinstance(obj2, dict):
            return extract_pages_from_jsonish(
                obj2
            )

    # Strict field-level fallback: inspect only evidence_pages array.
    m = re.search(
        r'"evidence_pages"\s*:\s*\[([^\]]*)\]',
        text,
        flags=re.I | re.S,
    )

    if m:
        return extract_pages(
            m.group(1)
        )

    return []


def get_pred_pages_from_qa(
    qa: Dict[str, Any],
) -> Tuple[List[int], str]:
    primary_keys = (
        "evidence_pages_pre",
        "pred_evidence_pages",
        "predicted_evidence_pages",
        "evidence_pages_pred",
    )

    # Historical strict parsers may keep a normalized field empty while
    # preserving the actual model-emitted pages in evidence_pages_pre_raw.
    # Match the reference evaluator's behavior: use raw only when the
    # normalized field exists but is empty and raw contains pages.
    for key in primary_keys:
        if key in qa:
            pages = extract_pages(
                qa.get(key)
            )

            if pages:
                return (
                    pages,
                    key,
                )

            raw_pages = extract_pages(
                qa.get(
                    "evidence_pages_pre_raw"
                )
            )

            if raw_pages:
                return (
                    raw_pages,
                    "evidence_pages_pre_raw_fallback",
                )

            # Explicit normalized [] + no raw pages = valid empty prediction.
            return (
                [],
                key,
            )

    if "evidence_pages_pre_raw" in qa:
        return (
            extract_pages(
                qa.get(
                    "evidence_pages_pre_raw"
                )
            ),
            "evidence_pages_pre_raw",
        )

    # Recover from raw model JSON only when no normalized prediction page
    # field exists.
    for key in (
        "answer_pre",
        "response",
        "raw_output",
        "raw_response",
    ):
        if key not in qa:
            continue

        pages = extract_pages_from_jsonish(
            qa.get(key)
        )

        if pages:
            return (
                pages,
                f"{key}:jsonish_evidence_pages",
            )

    # IMPORTANT:
    # Do NOT use qa["evidence_pages"] here.
    # That is the GOLD field.
    return [], "missing"


def calc_evidence_metrics(
    gold_pages: List[int],
    pred_pages: List[int],
) -> Dict[str, Any]:
    gold = set(gold_pages)
    pred = set(pred_pages)

    exact_correct = (
        gold == pred
    )

    inter = len(
        gold & pred
    )

    if not gold and not pred:
        precision = 1.0
        recall = 1.0
        f1 = 1.0

    elif not pred:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    elif not gold:
        precision = 0.0
        recall = 0.0
        f1 = 0.0

    else:
        precision = (
            inter / len(pred)
        )

        recall = (
            inter / len(gold)
        )

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    return {
        "evidence_correct": int(
            exact_correct
        ),
        "e_precision": precision,
        "e_recall": recall,
        "e_f1": f1,
        "gold_page_count": len(gold),
        "pred_page_count": len(pred),
        "intersection_count": inter,
    }


def build_judge_prompt(
    dataset_id: str,
    question: str,
    gold_answer: str,
    pred_answer: str,
) -> str:
    rules = JUDGE_RULES.strip()

    return (
        "You are a strict semantic reviewer for PDF question answering.\n"
        "Follow the frozen review protocol below exactly.\n\n"
        "[Frozen Review Protocol]\n"
        f"{rules}\n\n"
        "[Dataset]\n"
        f"{dataset_id}\n\n"
        "[Question]\n"
        f"{question}\n\n"
        "[Gold Answer]\n"
        f"{gold_answer}\n\n"
        "[Model Prediction]\n"
        f"{pred_answer}\n\n"
        "Return exactly one label and nothing else:\n"
        "CORRECT\nPARTIAL\nWRONG\n"
    )


def apply_chat_template_no_thinking(
    processor,
    messages,
):
    """
    Thinking MUST be disabled.

    No fallback is allowed. If this processor does not support
    enable_thinking=False, the formal evaluation stops rather than silently
    changing the judge protocol.
    """
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


def set_global_seed(
    seed: int,
) -> None:
    import random

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_first_param_device(model):
    try:
        return next(
            model.parameters()
        ).device
    except Exception:
        return "cuda"


def load_qwen36_judge():
    import torch
    from transformers import AutoProcessor

    set_global_seed(SEED)

    print(
        f"[Judge Protocol] seed={SEED}",
        flush=True,
    )
    print(
        f"[Judge Protocol] "
        f"enable_thinking={ENABLE_THINKING} "
        f"(required; NO FALLBACK)",
        flush=True,
    )
    print(
        "[Judge Protocol] do_sample=False",
        flush=True,
    )

    try:
        from transformers import (
            AutoModelForImageTextToText,
        )
        ModelClass = (
            AutoModelForImageTextToText
        )

        print(
            "[Judge Load] using "
            "AutoModelForImageTextToText",
            flush=True,
        )

    except Exception:
        from transformers import AutoModel
        ModelClass = AutoModel

        print(
            "[Judge Load] fallback model class "
            "to AutoModel",
            flush=True,
        )

    model_path = Path(
        QWEN36_JUDGE_MODEL_PATH
    )

    if not model_path.is_dir():
        raise FileNotFoundError(
            "Qwen3.6 judge model directory "
            f"does not exist: {model_path}"
        )

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    try:
        model = ModelClass.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

    except TypeError:
        model = ModelClass.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ).eval()

    print(
        "[Judge] model class  =",
        type(model),
        flush=True,
    )
    print(
        "[Judge] processor    =",
        type(processor),
        flush=True,
    )
    print(
        "[Judge] hf_device_map=",
        getattr(
            model,
            "hf_device_map",
            None,
        ),
        flush=True,
    )
    print(
        "[Judge] CUDA_VISIBLE_DEVICES =",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "",
        ),
        flush=True,
    )

    return model, processor


def parse_judge_label(
    text: str,
) -> Tuple[Optional[str], str]:
    raw = normalize_space(text)
    up = raw.upper()

    # Accept legacy INCORRECT as WRONG defensively, but formal prompt requests
    # only CORRECT / PARTIAL / WRONG.
    if re.search(r"\bINCORRECT\b", up):
        return "wrong", raw
    if re.search(r"\bPARTIAL\b", up):
        return "partial", raw
    if re.search(r"\bWRONG\b", up):
        return "wrong", raw
    if re.search(r"\bCORRECT\b", up):
        return "correct", raw

    return None, raw


def run_qwen36_judge(
    model,
    processor,
    dataset_id: str,
    question: str,
    gold_answer: str,
    pred_answer: str,
) -> Tuple[str, str]:
    import torch

    prompt = build_judge_prompt(
        dataset_id,
        question,
        gold_answer,
        pred_answer,
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = apply_chat_template_no_thinking(
        processor,
        messages,
    )

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        get_first_param_device(model)
    )

    with torch.inference_mode():
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        except TypeError:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
            )

    gen_ids = [
        out[len(inp):]
        for inp, out in zip(
            inputs.input_ids,
            output_ids,
        )
    ]

    out_text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    verdict, raw = parse_judge_label(
        out_text
    )

    # A malformed JUDGE output is an evaluator failure, not a model semantic
    # error. Stop the worker instead of silently counting it as WRONG.
    if verdict is None:
        raise RuntimeError(
            "Qwen3.6 judge returned an invalid label. "
            f"Raw judge output: {raw!r}"
        )

    return verdict, raw


def save_cache(
    path: Path,
    cache: Dict[str, Any],
) -> None:
    atomic_write_json(
        path,
        cache,
    )


CORE_QA_FIELDS = {
    "question",
    "answer",
    "gold_answer",
    "reference_answer",
    "evidence_pages",
    "gold_evidence_pages",
    "answer_pre",
    "pred_answer",
    "prediction",
    "model_answer",
    "response",
    "raw_output",
    "raw_response",
    "evidence_pages_pre",
    "evidence_pages_pre_raw",
    "pred_evidence_pages",
    "predicted_evidence_pages",
    "evidence_pages_pred",
}


def build_extra_metadata_json(
    qa: Dict[str, Any],
) -> str:
    extra: Dict[str, Any] = {}

    for key, value in qa.items():
        if key in CORE_QA_FIELDS:
            continue

        # These already have dedicated detail columns.
        if key in {
            "item_uid",
            "uid",
            "question_type",
            "question_category",
            "modal_types",
            "answerable",
            "is_answerable",
            "status",
            "run_status",
            "elapsed_seconds",
            "model_name",
            "prompt_id",
            "prompt_version",
        }:
            continue

        try:
            json.dumps(
                value,
                ensure_ascii=False,
            )
            extra[key] = value

        except Exception:
            extra[key] = str(value)

    return (
        compact_json(extra)
        if extra
        else ""
    )


def get_paper_or_unit(
    unit_id: str,
    unit: Dict[str, Any],
    qa: Dict[str, Any],
) -> str:
    for obj in (
        qa,
        unit,
    ):
        for key in (
            "paper",
            "paper_id",
            "doc_id",
            "unit_id",
        ):
            value = obj.get(key)

            if value not in (
                None,
                "",
            ):
                return str(value)

    return str(unit_id)


def get_pdf_path_display(
    unit: Dict[str, Any],
    qa: Dict[str, Any],
) -> str:
    for obj in (
        qa,
        unit,
    ):
        for key in (
            "pdf_path_resolved",
            "pdf_path",
            "paper_path",
            "merged_pdf_path",
        ):
            value = obj.get(key)

            if value not in (
                None,
                "",
            ):
                return str(value)

    return ""


def get_answerable_display(
    qa: Dict[str, Any],
    gold_answer: str,
) -> Any:
    for key in (
        "answerable",
        "is_answerable",
    ):
        if key in qa:
            return qa.get(key)

    # Display-only inference for the ordinary benchmark.
    if gold_answer:
        return (
            0
            if gold_answer.lower()
            == "unanswerable"
            else 1
        )

    return ""


def is_unanswerable_gold_detail(
    row: Dict[str, Any],
) -> bool:
    gold = normalize_space(
        row.get(
            "gold_answer",
            "",
        )
    ).strip().lower()

    if gold == "unanswerable":
        return True

    flag = row.get(
        "answerable",
        "",
    )

    if isinstance(flag, bool):
        return not flag

    if isinstance(flag, (int, float)):
        return int(flag) == 0

    flag_text = normalize_space(flag).lower()
    return flag_text in {
        "0",
        "false",
        "no",
        "unanswerable",
    }


def build_fixed_denominator_summary(
    model_name: str,
    dataset_label: str,
    source_dataset: str,
    subset: str,
    result_path: Path,
    rows: List[Dict[str, Any]],
    fixed_denominator: int,
) -> Dict[str, Any]:
    present = len(rows)

    correct = sum(
        1
        for row in rows
        if str(
            row.get(
                "answer_verdict",
                "",
            )
        ).lower()
        == "correct"
    )

    partial = sum(
        1
        for row in rows
        if str(
            row.get(
                "answer_verdict",
                "",
            )
        ).lower()
        == "partial"
    )

    wrong = sum(
        1
        for row in rows
        if str(
            row.get(
                "answer_verdict",
                "",
            )
        ).lower()
        == "wrong"
    )

    technical = sum(
        1
        for row in rows
        if str(
            row.get(
                "answer_verdict",
                "",
            )
        ).lower()
        == "technical_failure"
    )

    judged = sum(
        int(
            row.get(
                "answer_judge_called",
                0,
            )
            or 0
        )
        for row in rows
    )

    evidence_correct = sum(
        int(
            row.get(
                "evidence_correct",
                0,
            )
            or 0
        )
        for row in rows
    )

    sum_ep = sum(
        float(
            row.get(
                "E_Precision",
                0.0,
            )
            or 0.0
        )
        for row in rows
    )

    sum_er = sum(
        float(
            row.get(
                "E_Recall",
                0.0,
            )
            or 0.0
        )
        for row in rows
    )

    sum_ef1 = sum(
        float(
            row.get(
                "E_F1",
                0.0,
            )
            or 0.0
        )
        for row in rows
    )

    sum_pred_pages = sum(
        int(
            row.get(
                "pred_page_count",
                0,
            )
            or 0
        )
        for row in rows
    )

    both_correct = sum(
        int(
            row.get(
                "both_correct",
                0,
            )
            or 0
        )
        for row in rows
    )

    missing_vs_fixed = max(
        fixed_denominator - present,
        0,
    )
    over_present = max(
        present - fixed_denominator,
        0,
    )

    denom = float(
        fixed_denominator
    )

    return {
        "Model": model_name,
        "Dataset": dataset_label,
        "Source Dataset": source_dataset,
        "Subset": subset,

        # IMPORTANT: Samples is deliberately the fixed benchmark denominator.
        "Samples": fixed_denominator,
        "Fixed Denominator": fixed_denominator,
        "Present Samples": present,
        "Missing vs Fixed Denominator": missing_vs_fixed,
        "Over Present vs Fixed Denominator": over_present,

        "Answer Judged": judged,
        "Correct": correct,
        "Partial": partial,
        "Wrong": wrong,
        "Technical Failure": technical,

        # User-requested unified denominator:
        # Correct / fixed benchmark denominator.
        "Semantic Strict Accuracy": (
            correct / denom
            if fixed_denominator
            else 0.0
        ),
        # Alias retained for downstream compatibility.
        "Answer Accuracy": (
            correct / denom
            if fixed_denominator
            else 0.0
        ),

        "Evidence Correct": evidence_correct,
        "Evidence Accuracy": (
            evidence_correct / denom
            if fixed_denominator
            else 0.0
        ),
        "E_Precision": (
            sum_ep / denom
            if fixed_denominator
            else 0.0
        ),
        "E_Recall": (
            sum_er / denom
            if fixed_denominator
            else 0.0
        ),
        "E_F1": (
            sum_ef1 / denom
            if fixed_denominator
            else 0.0
        ),
        "Avg Pred Pages": (
            sum_pred_pages / denom
            if fixed_denominator
            else 0.0
        ),
        "Both Correct": both_correct,
        "Both Accuracy": (
            both_correct / denom
            if fixed_denominator
            else 0.0
        ),
        "Result File": str(
            result_path
        ),
    }


def build_summary_rows_for_job(
    model_name: str,
    dataset_id: str,
    result_path: Path,
    details: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = [
        build_fixed_denominator_summary(
            model_name=model_name,
            dataset_label=dataset_id,
            source_dataset=dataset_id,
            subset="all",
            result_path=result_path,
            rows=details,
            fixed_denominator=FIXED_DENOMINATORS[
                dataset_id
            ],
        )
    ]

    if dataset_id == "ordinary1190":
        answerable_rows = [
            row
            for row in details
            if not is_unanswerable_gold_detail(
                row
            )
        ]

        unanswerable_rows = [
            row
            for row in details
            if is_unanswerable_gold_detail(
                row
            )
        ]

        rows.append(
            build_fixed_denominator_summary(
                model_name=model_name,
                dataset_label="ordinary1190_answerable",
                source_dataset=dataset_id,
                subset="answerable",
                result_path=result_path,
                rows=answerable_rows,
                fixed_denominator=ORDINARY_SPLIT_DENOMINATORS[
                    "ordinary1190_answerable"
                ],
            )
        )

        rows.append(
            build_fixed_denominator_summary(
                model_name=model_name,
                dataset_label="ordinary1190_unanswerable",
                source_dataset=dataset_id,
                subset="unanswerable",
                result_path=result_path,
                rows=unanswerable_rows,
                fixed_denominator=ORDINARY_SPLIT_DENOMINATORS[
                    "ordinary1190_unanswerable"
                ],
            )
        )

    return rows


def evaluate_one_result(
    model_name: str,
    dataset_id: str,
    result_path: Path,
    judge_model,
    judge_processor,
    cache: Dict[str, Any],
    cache_path: Path,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    data = load_json(
        result_path
    )

    details: List[
        Dict[str, Any]
    ] = []

    sample_count = 0
    invalid_or_empty_answer = 0
    missing_gold_answer = 0
    missing_gold_evidence_field = 0

    answer_judged_count = 0
    answer_correct_count = 0
    evidence_correct_count = 0
    both_correct_count = 0

    sum_ep = 0.0
    sum_er = 0.0
    sum_ef1 = 0.0
    sum_pred_pages = 0.0

    seen_alignment_keys = set()

    for (
        unit_id,
        qa_id,
        unit,
        qa,
    ) in iter_qa_records(data):
        sample_count += 1

        (
            alignment_key,
            alignment_method,
        ) = build_alignment_key(
            dataset_id,
            unit_id,
            qa_id,
            qa,
        )

        if alignment_key in seen_alignment_keys:
            raise RuntimeError(
                f"Duplicate alignment key inside "
                f"{model_name}/{dataset_id}: "
                f"{alignment_key}. "
                "The formal result file contains duplicate sample IDs."
            )

        seen_alignment_keys.add(
            alignment_key
        )

        item_uid = get_item_uid(
            dataset_id,
            unit_id,
            qa_id,
            qa,
        )

        question = normalize_space(
            qa.get(
                "question",
                "",
            )
        )

        (
            gold_answer,
            gold_answer_source,
        ) = get_gold_answer_from_qa(
            qa
        )

        (
            gold_pages,
            gold_pages_source,
        ) = get_gold_pages_from_qa(
            qa
        )

        if gold_answer_source == "missing":
            missing_gold_answer += 1

        if gold_pages_source == "missing":
            missing_gold_evidence_field += 1

        (
            pred_answer,
            answer_parse_status,
            pred_answer_source,
            raw_pred_answer,
        ) = get_pred_answer_from_qa(
            qa
        )

        (
            pred_pages,
            pred_pages_source,
        ) = get_pred_pages_from_qa(
            qa
        )

        if gold_pages_source == "missing":
            # Missing gold evidence is a malformed/self-incomplete result file,
            # NOT an unanswerable sample with gold=[].
            evidence = {
                "evidence_correct": 0,
                "e_precision": 0.0,
                "e_recall": 0.0,
                "e_f1": 0.0,
                "gold_page_count": 0,
                "pred_page_count": len(set(pred_pages)),
                "intersection_count": 0,
            }
        else:
            evidence = calc_evidence_metrics(
                gold_pages,
                pred_pages,
            )

        evidence_correct = bool(
            evidence["evidence_correct"]
        )

        answer_correct = False
        answer_verdict = "technical_failure"
        answer_judge_called = 0
        judge_raw = ""

        if is_illegal_answer(
            pred_answer,
            raw_pred_answer,
        ):
            invalid_or_empty_answer += 1
            prediction_status = (
                "technical_failure"
            )
            judge_raw = (
                "SKIPPED_UNRECOVERABLE_OR_EMPTY_PREDICTION"
            )

        elif not question:
            prediction_status = (
                "technical_failure_missing_question"
            )
            judge_raw = (
                "SKIPPED_MISSING_QUESTION"
            )

        elif gold_answer_source == "missing":
            prediction_status = (
                "technical_failure_missing_gold_answer"
            )
            judge_raw = (
                "SKIPPED_MISSING_GOLD_ANSWER"
            )

        else:
            prediction_status = (
                "evaluated"
            )
            answer_judge_called = 1
            answer_judged_count += 1

            cache_key = stable_key(
                "qwen36_calibrated_semantic_triclass_v4",
                JUDGE_RULES_SHA256,
                dataset_id,
                question,
                gold_answer,
                pred_answer,
            )

            if cache_key in cache:
                answer_verdict = str(
                    cache[
                        cache_key
                    ].get(
                        "answer_verdict",
                        "",
                    )
                ).strip().lower()

                judge_raw = str(
                    cache[
                        cache_key
                    ].get(
                        "judge_raw",
                        "",
                    )
                )

                if answer_verdict not in {
                    "correct",
                    "partial",
                    "wrong",
                }:
                    raise RuntimeError(
                        "Invalid cached answer_verdict "
                        f"{answer_verdict!r} for cache key {cache_key}"
                    )

                prediction_status = (
                    "evaluated_cached"
                )

            else:
                (
                    answer_verdict,
                    judge_raw,
                ) = run_qwen36_judge(
                    judge_model,
                    judge_processor,
                    dataset_id,
                    question,
                    gold_answer,
                    pred_answer,
                )

                cache[
                    cache_key
                ] = {
                    "dataset_id": dataset_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "answer_verdict": answer_verdict,
                    "judge_raw": judge_raw,
                    "judge_rules_sha256": (
                        JUDGE_RULES_SHA256
                    ),
                }

                # Periodic worker-local checkpoint.
                if (
                    len(cache) % 20
                    == 0
                ):
                    save_cache(
                        cache_path,
                        cache,
                    )

        answer_correct = (
            answer_verdict == "correct"
        )

        both_correct = bool(
            answer_correct
            and evidence_correct
        )

        answer_correct_count += int(
            answer_correct
        )

        evidence_correct_count += int(
            evidence_correct
        )

        both_correct_count += int(
            both_correct
        )

        sum_ep += evidence[
            "e_precision"
        ]

        sum_er += evidence[
            "e_recall"
        ]

        sum_ef1 += evidence[
            "e_f1"
        ]

        sum_pred_pages += evidence[
            "pred_page_count"
        ]

        detail = {
            # ----------------------------
            # identity / alignment
            # ----------------------------
            "model_name": model_name,
            "dataset": dataset_id,
            "sample_index": sample_count,
            "alignment_key": alignment_key,
            "alignment_method": alignment_method,
            "unit_id": unit_id,
            "qa_id": qa_id,
            "item_uid": item_uid,

            # ----------------------------
            # sample attributes
            # ----------------------------
            "paper_or_unit": get_paper_or_unit(
                unit_id,
                unit,
                qa,
            ),
            "pdf_path": get_pdf_path_display(
                unit,
                qa,
            ),
            "primary_category": unit.get(
                "primary_category",
                qa.get(
                    "primary_category",
                    "",
                ),
            ),
            "secondary_category": unit.get(
                "secondary_category",
                qa.get(
                    "secondary_category",
                    "",
                ),
            ),
            "question_type": qa.get(
                "question_type",
                "",
            ),
            "question_category": qa.get(
                "question_category",
                "",
            ),
            "modal_types": compact_json(
                qa.get(
                    "modal_types",
                    [],
                )
            ),
            "answerable": get_answerable_display(
                qa,
                gold_answer,
            ),
            "question": question,

            # ----------------------------
            # gold/reference fields
            # ----------------------------
            "gold_answer": gold_answer,
            "gold_answer_source": (
                gold_answer_source
            ),
            "gold_evidence_pages": (
                compact_json(
                    gold_pages
                )
            ),
            "gold_evidence_pages_source": (
                gold_pages_source
            ),
            "gold_page_count": evidence[
                "gold_page_count"
            ],
            "gold_extra_metadata": (
                build_extra_metadata_json(
                    qa
                )
            ),

            # ----------------------------
            # prediction / answer judge
            # ----------------------------
            "pred_answer": pred_answer,
            "pred_answer_source": (
                pred_answer_source
            ),
            "pred_answer_parse_status": (
                answer_parse_status
            ),
            "prediction_status": (
                prediction_status
            ),
            "answer_judge_called": (
                answer_judge_called
            ),
            "answer_verdict": (
                answer_verdict
            ),
            "answer_correct": int(
                answer_correct
            ),

            # ----------------------------
            # predicted evidence
            # ----------------------------
            "pred_evidence_pages": (
                compact_json(
                    pred_pages
                )
            ),
            "pred_evidence_pages_source": (
                pred_pages_source
            ),
            "evidence_correct": int(
                evidence_correct
            ),
            "E_Precision": evidence[
                "e_precision"
            ],
            "E_Recall": evidence[
                "e_recall"
            ],
            "E_F1": evidence[
                "e_f1"
            ],
            "pred_page_count": evidence[
                "pred_page_count"
            ],
            "page_intersection_count": (
                evidence[
                    "intersection_count"
                ]
            ),
            "both_correct": int(
                both_correct
            ),

            # ----------------------------
            # source-result audit fields
            # ----------------------------
            "source_status": qa.get(
                "status",
                "",
            ),
            "source_run_status": qa.get(
                "run_status",
                "",
            ),
            "source_model_name": qa.get(
                "model_name",
                "",
            ),
            "source_prompt_id": qa.get(
                "prompt_id",
                "",
            ),
            "source_prompt_version": qa.get(
                "prompt_version",
                "",
            ),
            "source_elapsed_seconds": qa.get(
                "elapsed_seconds",
                "",
            ),
            "result_file": str(
                result_path
            ),
        }

        if SAVE_JUDGE_RAW:
            detail[
                "judge_raw"
            ] = judge_raw

        details.append(
            detail
        )

        if (
            sample_count % 25
            == 0
        ):
            print(
                f"[{model_name} / "
                f"{dataset_id}] "
                f"{sample_count} samples | "
                f"answer_acc="
                f"{answer_correct_count / sample_count:.4f} | "
                f"evidence_acc="
                f"{evidence_correct_count / sample_count:.4f} | "
                f"E_F1="
                f"{sum_ef1 / sample_count:.4f}",
                flush=True,
            )

    summary = build_fixed_denominator_summary(
        model_name=model_name,
        dataset_label=dataset_id,
        source_dataset=dataset_id,
        subset="all",
        result_path=result_path,
        rows=details,
        fixed_denominator=FIXED_DENOMINATORS[
            dataset_id
        ],
    )

    return (
        details,
        summary,
    )


