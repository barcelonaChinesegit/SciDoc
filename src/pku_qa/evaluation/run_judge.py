#!/usr/bin/env python3
"""
答案评判脚本 (支持多卡并发分片版)

评判脚本
- 非法输出与不可回答题：按严格 contract 判定
- 可回答题：统一使用论文原版提示词和 Qwen3.6-27B 语义判定
"""
from __future__ import annotations
import argparse, json, os, time
import hashlib
from pathlib import Path

from durable_work_queue import DurablePaperQueue
from eval_framework import (
    ProviderError, atomic_write_json, configure_hf_environment, create_provider,
    iter_paper_items, load_json, looks_like_error_output,
    load_provider_specs,
)
from gpu_reservation import managed_gpu_reservation
from progress_logging import progress_fields
from evaluation_protocol import (
    JUDGE_INFERENCE_BINDING_FIELDS,
    PDF_INPUT_MODE,
    build_judge_queue_contract,
    build_judge_queue_value_contract,
    canonical_gold_answer,
    canonical_gold_pages,
    canonicalize_pdf_output_for_storage,
    is_exact_unanswerable,
    judge_inference_binding_sha256,
    parse_canonical_pdf_output,
    protocol_from_inference_record,
    provider_runtime_identity,
    sha256_file,
    validate_dataset_protocol,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str, required=True)
    parser.add_argument(
        "--qa-json",
        type=str,
        required=True,
        help="Independent immutable gold QA file used for every score.",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--judge-provider-name", type=str, default="local_qwen3_6_27b_judge")
    parser.add_argument("--provider-config", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--disable-hf-mirror", action="store_true")
    # 🌟 新增：分片参数
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="Only rerun existing entries whose previous LLM judge call failed.",
    )
    parser.add_argument(
        "--retry-match-method",
        action="append",
        default=[],
        help=(
            "Only rerun existing entries with this match_method. May be "
            "repeated, for example --retry-match-method numeric_match."
        ),
    )
    parser.add_argument("--work-queue-dir", default=None)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--worker-idle-seconds", type=float, default=3.0)
    parser.add_argument("--max-judge-retries", type=int, default=3)
    parser.add_argument(
        "--expected-inference-protocol-fingerprint",
        default=None,
        help=(
            "Fail closed unless every inference row has this exact protocol "
            "fingerprint. Used by the parent scheduler and skip/resume runs."
        ),
    )
    return parser.parse_args()

def parse_structured_response(text: str) -> tuple[str, list[int], str]:
    try:
        answer, pages = parse_canonical_pdf_output(text)
    except ValueError:
        return "", [], "invalid_pdf_output"
    return answer, pages, "canonical_json"


def first_existing(data: dict, field_names: tuple[str, ...], default=None):
    for field_name in field_names:
        if field_name in data:
            return data.get(field_name)
    return default


def qa_type(qa_data: dict) -> str:
    explicit = str(qa_data.get("type", "")).strip().lower()
    if explicit in {"mcq", "fill"}:
        return explicit
    return "mcq" if qa_data.get("options") else "fill"

def direct_fill_match(reference: str, prediction: str) -> tuple[bool | None, str]:
    """Only empty/error output and exact refusal are deterministic decisions.

    Every answerable response goes to the paper semantic judge, including
    identical text, numeric equivalents and aliases. Never normalize answers.
    """
    if not prediction.strip():
        return False, "empty_prediction"
    if looks_like_error_output(prediction):
        return False, "error_output"
    if reference == "Unanswerable":
        return prediction == "Unanswerable", "unanswerable_exact_label"
    return None, "needs_llm_judge"


def build_judge_prompt(question: str, correct: str, model_answer: str) -> str:
    path = Path(__file__).resolve().parents[3] / "evaluation/prompts/semantic_judge.txt"
    return path.read_text(encoding="utf-8").format(
        question=question, correct=correct, model_answer=model_answer
    )


def judge_fill(provider, question: str, correct: str, model_answer: str, max_new_tokens: int) -> tuple[str, str]:
    prompt = build_judge_prompt(question, correct, model_answer)
    response = provider.generate(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_new_tokens,
    )
    if response not in {"CORRECT", "INCORRECT"}:
        error = ProviderError("Judge returned an invalid exact binary label")
        error.raw_response = response
        raise error
    return response, response


def fill_needs_llm_judge(qa: dict) -> bool:
    output = qa.get("model_output", "")
    try:
        protocol = protocol_from_inference_record(qa)
    except ValueError:
        return False
    if protocol.require_structured_output:
        try:
            output, _ = parse_canonical_pdf_output(
                output, allowed_pages=qa.get("shown_pdf_pages")
            )
        except ValueError:
            return False
    correct = first_existing(qa, ("correct_answer", "answer"), "")
    if is_exact_unanswerable(correct):
        return False
    return direct_fill_match(correct, output)[0] is None


def is_existing_judge_error(qa: dict | None) -> bool:
    if not isinstance(qa, dict):
        return False
    return qa.get("judge_verdict") == "ERROR" or qa.get("match_method") == "llm_judge_error"


def should_process_existing(
    existing_qa: dict | None,
    *,
    retry_errors_only: bool,
    retry_match_methods: set[str],
) -> bool:
    if retry_errors_only:
        return is_existing_judge_error(existing_qa)
    if retry_match_methods:
        return (
            isinstance(existing_qa, dict)
            and existing_qa.get("match_method") in retry_match_methods
        )
    required_fields = (
        "is_correct",
        "answer_is_correct",
        "evidence_pages_is_correct",
        "input_mode",
        "require_structured_output",
        "require_evidence_pages",
        "output_is_legal",
    )
    return not (
        isinstance(existing_qa, dict)
        and all(field in existing_qa for field in required_fields)
        and isinstance(existing_qa.get("is_correct"), bool)
        and isinstance(existing_qa.get("answer_is_correct"), bool)
        and isinstance(existing_qa.get("evidence_pages_is_correct"), bool)
    )


def judge_row_matches_inference(
    existing_qa: dict | None, inference_qa: dict
) -> bool:
    """Reject a cached Judge row produced for different inference bytes."""
    if not isinstance(existing_qa, dict) or not isinstance(inference_qa, dict):
        return False
    return all(
        existing_qa.get(field) == inference_qa.get(field)
        for field in JUDGE_INFERENCE_BINDING_FIELDS
    )


def should_process_judge_row(
    existing_qa: dict | None,
    inference_qa: dict,
    *,
    retry_errors_only: bool,
    retry_match_methods: set[str],
    expected_judge_protocol_fingerprint: str | None = None,
) -> bool:
    stale_judge_protocol = bool(
        expected_judge_protocol_fingerprint
        and (
            not isinstance(existing_qa, dict)
            or existing_qa.get("judge_protocol_fingerprint")
            != expected_judge_protocol_fingerprint
        )
    )
    return stale_judge_protocol or should_process_existing(
        existing_qa,
        retry_errors_only=retry_errors_only,
        retry_match_methods=retry_match_methods,
    ) or not judge_row_matches_inference(existing_qa, inference_qa)


def process_judge_paper(
    paper_id: str,
    paper_data: dict,
    gold_paper_data: dict,
    existing_paper: dict | None,
    *,
    judge_provider,
    args,
    retry_match_methods: set[str],
    checkpoint=None,
) -> dict:
    existing_paper = (
        existing_paper
        if isinstance(existing_paper, dict)
        else {"paper": paper_id, "QA": {}}
    )
    existing_paper.setdefault("QA", {})
    if "error" in paper_data:
        raise RuntimeError(f"Inference paper error: {paper_data['error']}")

    inference_qas = paper_data.get("QA", {})
    gold_qas = gold_paper_data.get("QA", {})
    if not isinstance(inference_qas, dict) or not isinstance(gold_qas, dict):
        raise ValueError(f"Malformed inference/gold paper: {paper_id}")
    if set(map(str, inference_qas)) != set(map(str, gold_qas)):
        raise ValueError(
            f"Inference/gold QA keys differ for paper {paper_id}: "
            f"inference={len(inference_qas)} gold={len(gold_qas)}"
        )

    qa_items = list(inference_qas.items())
    total_qa = len(qa_items)
    for qa_index, (qa_id, qa_data) in enumerate(qa_items, start=1):
        existing_qa = existing_paper["QA"].get(qa_id)
        gold_qa = gold_qas.get(qa_id)
        if not isinstance(gold_qa, dict):
            raise ValueError(f"Missing gold QA: {paper_id}/{qa_id}")
        gold_question = gold_qa.get("question")
        if not isinstance(gold_question, str) or not gold_question.strip():
            raise ValueError(f"Invalid gold question: {paper_id}/{qa_id}")
        gold_answer = canonical_gold_answer(
            gold_qa, item_id=f"{paper_id}/{qa_id}"
        )
        gold_reference_pages = canonical_gold_pages(
            gold_qa.get("evidence_pages", []),
            item_id=f"{paper_id}/{qa_id}",
        )
        if str(qa_data.get("question", "")) != gold_question:
            raise ValueError(f"Inference/gold question mismatch: {paper_id}/{qa_id}")
        if str(qa_data.get("correct_answer", "")) != gold_answer:
            raise ValueError(f"Inference/gold answer mismatch: {paper_id}/{qa_id}")
        embedded_reference_pages = canonical_gold_pages(
            qa_data.get("reference_evidence_pages", []),
            item_id=f"{paper_id}/{qa_id} inference reference",
        )
        if embedded_reference_pages != gold_reference_pages:
            raise ValueError(
                f"Inference/gold evidence mismatch: {paper_id}/{qa_id}"
            )
        protocol = protocol_from_inference_record(qa_data)
        needs_processing = should_process_judge_row(
            existing_qa,
            qa_data,
            retry_errors_only=args.retry_errors_only,
            retry_match_methods=retry_match_methods,
            expected_judge_protocol_fingerprint=getattr(
                args, "judge_protocol_fingerprint", None
            ),
        )
        if not needs_processing:
            continue

        qa_started_at = time.monotonic()
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"event=judge_qa_started worker={args.worker_id or args.shard_id} "
            f"paper={paper_id} qa={qa_id} "
            f"{progress_fields(qa_index, total_qa)} progress_scope=paper_qa",
            flush=True,
        )

        q_type = qa_type(gold_qa)
        ans = gold_answer
        out = qa_data.get("model_output", "")
        raw_out = qa_data.get("raw_model_output", out)
        recorded_normalizations = qa_data.get(
            "deterministic_normalizations", []
        )
        expected_raw_hash = hashlib.sha256(
            str(raw_out).encode("utf-8")
        ).hexdigest()
        if qa_data.get("raw_model_output_sha256") != expected_raw_hash:
            raise ValueError(
                f"Inference raw-output hash mismatch: {paper_id}/{qa_id}"
            )
        if protocol.input_mode == PDF_INPUT_MODE:
            try:
                canonical_out, actual_normalizations = (
                    canonicalize_pdf_output_for_storage(
                        raw_out,
                        allowed_pages=qa_data.get("shown_pdf_pages"),
                    )
                )
            except ValueError:
                if str(out) != str(raw_out) or recorded_normalizations:
                    raise ValueError(
                        "Invalid inference output was altered before "
                        f"Judge: {paper_id}/{qa_id}"
                    )
            else:
                if canonical_out != out:
                    raise ValueError(
                        "Inference canonical/raw output mismatch: "
                        f"{paper_id}/{qa_id}"
                    )
                if actual_normalizations != recorded_normalizations:
                    raise ValueError(
                        "Inference normalization audit mismatch: "
                        f"{paper_id}/{qa_id}"
                    )
        elif str(raw_out) != str(out) or recorded_normalizations:
            raise ValueError(
                "question_only inference must preserve its exact raw "
                f"answer without normalization: {paper_id}/{qa_id}"
            )
        require_evidence = protocol.require_evidence_pages
        require_structured = protocol.require_structured_output
        reference_pages = gold_reference_pages
        output_is_legal = True
        output_illegal_reason = None
        if require_structured:
            try:
                parsed_answer, predicted_pages = parse_canonical_pdf_output(
                    out, allowed_pages=qa_data.get("shown_pdf_pages")
                )
                output_parse_method = "canonical_json"
            except ValueError as exc:
                parsed_answer, predicted_pages = "", []
                output_parse_method = "invalid_pdf_output"
                output_is_legal = False
                output_illegal_reason = str(exc)
        else:
            parsed_answer, predicted_pages, output_parse_method = (
                str(out),
                [],
                "question_only_plain_answer",
            )
            if not parsed_answer.strip() or looks_like_error_output(parsed_answer):
                output_is_legal = False
                output_illegal_reason = "empty_or_error_question_only_output"
        evidence_pages_is_correct = (
            output_is_legal and predicted_pages == reference_pages
            if protocol.input_mode == PDF_INPUT_MODE
            else True
        )
        qa_judge = {
            "question": gold_question,
            "correct_answer": ans,
            "model_output": out,
            "raw_model_output": raw_out,
            "raw_model_output_sha256": qa_data.get(
                "raw_model_output_sha256"
            ),
            "deterministic_normalizations": recorded_normalizations,
            "parsed_answer": parsed_answer,
            "type": q_type,
            "require_structured_output": require_structured,
            "require_evidence_pages": require_evidence,
            "reference_evidence_pages": reference_pages,
            "predicted_evidence_pages": predicted_pages,
            "evidence_pages_is_correct": evidence_pages_is_correct,
            "output_parse_method": output_parse_method,
            "output_is_legal": output_is_legal,
            "output_illegal_reason": output_illegal_reason,
            "input_mode": protocol.input_mode,
            "publication_eligible": True,
            "protocol_validation": "publication_strict",
            "inference_binding_sha256": (
                judge_inference_binding_sha256(qa_data)
            ),
        }
        if getattr(args, "judge_protocol_fingerprint", None):
            qa_judge["judge_protocol_fingerprint"] = (
                args.judge_protocol_fingerprint
            )
        for metadata_field in (
            "answer_format",
            "answer_aliases",
            "answer_unit",
            "numeric_tolerance",
            "string_metric",
            "evidence_items",
            "evidence_hops",
            "evidence_span",
            "evidence_span_ratio",
            "modal_types",
            "generation_status",
            "question_type",
            "question_category",
            "page_input_policy",
            "prompt_style",
            "shown_pdf_pages",
            "total_pdf_pages",
            "max_pdf_pages",
            "requested_pdf_dpi",
            "rendered_pdf_dpi",
            "pdf_render_downscaled",
            "annotation_provenance",
            "review_status",
            "evaluated_model",
            "protocol_fingerprint",
            "qa_source_sha256",
            "pdf_corpus_sha256",
            "pdf_sha256",
        ):
            if metadata_field in gold_qa:
                qa_judge[metadata_field] = gold_qa[metadata_field]
            elif metadata_field in qa_data:
                qa_judge[metadata_field] = qa_data[metadata_field]

        if is_exact_unanswerable(ans):
            # The controlled abstention rule is global and takes precedence
            # over stale/accidental question-type metadata or option fields.
            answer_is_correct = (
                output_is_legal and parsed_answer == "Unanswerable"
            )
            qa_judge.update(
                {
                    "answer_is_correct": answer_is_correct,
                    "is_correct": answer_is_correct
                    and evidence_pages_is_correct,
                    "match_method": "exact_unanswerable",
                    "typed_score": None,
                    "judge_verdict": (
                        "CORRECT" if answer_is_correct else "INCORRECT"
                    ),
                }
            )
        else:
            typed_score = None
            if not output_is_legal:
                direct_res, method = False, "illegal_output"
            else:
                direct_res, method = direct_fill_match(ans, parsed_answer)
            if direct_res is not None:
                qa_judge.update(
                    {
                        "answer_is_correct": direct_res,
                        "is_correct": direct_res and evidence_pages_is_correct,
                        "match_method": method,
                        "typed_score": typed_score,
                        "judge_verdict": (
                            "CORRECT" if direct_res else "INCORRECT"
                        ),
                    }
                )
            elif judge_provider:
                qa_judge["judge_attempts"] = []
                for attempt in range(1, args.max_judge_retries + 1):
                    try:
                        verdict, judge_response = judge_fill(
                            judge_provider,
                            qa_judge["question"],
                            ans,
                            parsed_answer,
                            args.max_new_tokens,
                        )
                        qa_judge["judge_attempts"].append({"raw": judge_response})
                        answer_is_correct = verdict == "CORRECT"
                        qa_judge.update(
                            {
                                "judge_verdict": verdict,
                                "judge_response": judge_response,
                                "match_method": "llm_judge",
                                "answer_is_correct": answer_is_correct,
                                "is_correct": answer_is_correct
                                and evidence_pages_is_correct,
                            }
                        )
                        break
                    except Exception as exc:
                        qa_judge["judge_attempts"].append({"error_type": type(exc).__name__,
                            "raw": getattr(exc, "raw_response", None)})
                        if attempt < args.max_judge_retries:
                            time.sleep(2 ** (attempt - 1))
                else:
                    qa_judge.update(judge_verdict="ERROR", match_method="llm_judge_error",
                                    answer_is_correct=False, is_correct=False,
                                    technical_failure=True)
            else:
                qa_judge.update(judge_verdict="ERROR", match_method="llm_judge_error",
                                answer_is_correct=False, is_correct=False,
                                technical_failure=True,
                                judge_attempts=[{"error_type": "JudgeProviderUnavailable"}])

        existing_paper["QA"][qa_id] = qa_judge
        if checkpoint is not None:
            checkpoint(existing_paper)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"event=judge_qa_completed worker={args.worker_id or args.shard_id} "
            f"paper={paper_id} qa={qa_id} "
            f"{progress_fields(qa_index, total_qa)} progress_scope=paper_qa "
            f"duration_seconds={time.monotonic() - qa_started_at:.2f}",
            flush=True,
        )
    return existing_paper


def main():
    args = parse_args()
    gold_path = Path(args.qa_json)
    args.qa_json = str(gold_path)
    if args.retry_errors_only and args.retry_match_method:
        raise ValueError(
            "--retry-errors-only and --retry-match-method are mutually exclusive"
        )
    retry_match_methods = set(args.retry_match_method)
    configure_hf_environment(use_mirror=not args.disable_hf_mirror, cuda_visible_devices=args.gpu)
    
    # 针对多 shard 生成专属输出文件；单 shard 直接写目标文件，便于报告脚本读取。
    output_file = Path(args.output.replace(".json", f"_shard{args.shard_id}.json")) if args.num_shards > 1 else Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    inference_results = load_json(args.result_file, default={}) or {}
    gold_results = load_json(gold_path, default={}) or {}
    if not isinstance(inference_results, dict) or not isinstance(gold_results, dict):
        raise ValueError("Inference results and gold QA must both be JSON objects")
    inference_paper_ids = {
        paper_id for paper_id, _ in iter_paper_items(inference_results)
    }
    gold_paper_ids = {paper_id for paper_id, _ in iter_paper_items(gold_results)}
    if inference_paper_ids != gold_paper_ids:
        raise ValueError(
            "Inference/gold paper keys differ: "
            f"inference={len(inference_paper_ids)} gold={len(gold_paper_ids)}"
        )
    input_modes = {
        str(qa.get("input_mode", ""))
        for _, paper in iter_paper_items(inference_results)
        for qa in paper.get("QA", {}).values()
        if isinstance(qa, dict)
    }
    if len(input_modes) != 1:
        raise ValueError(f"Expected one input_mode in inference results, found {input_modes}")
    input_mode = next(iter(input_modes))
    validate_dataset_protocol(gold_results, input_mode)
    gold_sha256 = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    inference_fingerprints: set[str] = set()
    for paper_id, paper in iter_paper_items(inference_results):
        for qa_id, qa in paper.get("QA", {}).items():
            if not isinstance(qa, dict):
                continue
            fingerprint = str(qa.get("protocol_fingerprint", "")).strip()
            if not fingerprint:
                raise ValueError(
                    f"Missing protocol fingerprint: {paper_id}/{qa_id}"
                )
            if qa.get("qa_source_sha256") != gold_sha256:
                raise ValueError(
                    f"Inference/gold source hash mismatch: {paper_id}/{qa_id}"
                )
            if str(qa.get("input_mode", "")) == PDF_INPUT_MODE and (
                not str(qa.get("pdf_corpus_sha256", "")).strip()
                or not str(qa.get("pdf_sha256", "")).strip()
            ):
                raise ValueError(
                    f"Missing PDF hash lock: {paper_id}/{qa_id}"
                )
            if fingerprint:
                inference_fingerprints.add(fingerprint)
    if len(inference_fingerprints) > 1:
        raise ValueError("Inference result mixes protocol fingerprints")
    actual_inference_fingerprint = next(
        iter(inference_fingerprints), None
    )
    expected_inference_fingerprint = str(
        args.expected_inference_protocol_fingerprint or ""
    ).strip()
    if (
        expected_inference_fingerprint
        and actual_inference_fingerprint != expected_inference_fingerprint
    ):
        raise ValueError(
            "Inference protocol fingerprint does not match the parent "
            f"contract: expected {expected_inference_fingerprint}, found "
            f"{actual_inference_fingerprint}"
        )
    args.judge_protocol_fingerprint = None
    strict_judge_contract = build_judge_queue_contract(
            input_mode=input_mode,
            judge_runner_path=__file__,
            gold_source_sha256=gold_sha256,
            inference_protocol_fingerprint=actual_inference_fingerprint,
            judge_provider_identity=provider_runtime_identity(
                args.judge_provider_name,
                load_provider_specs(args.provider_config)[
                    args.judge_provider_name
                ],
            ),
            judge_config={
                "judge_provider_name": args.judge_provider_name,
                "max_new_tokens": args.max_new_tokens,
                "max_judge_retries": args.max_judge_retries,
            },
    )
    args.judge_protocol_fingerprint = strict_judge_contract[1][
        "judge_protocol_fingerprint"
    ]
    print(
        f"[judge-protocol] input_mode={input_mode} "
        f"gold_sha256={gold_sha256} papers={len(gold_paper_ids)}",
        flush=True,
    )
    judge_results = load_json(output_file, default={}) or {}
    
    # 🌟 核心：根据 shard_id 切分需要 judge 的论文
    all_papers = sorted(list(iter_paper_items(inference_results)), key=lambda x: x[0])
    my_papers = [p for i, p in enumerate(all_papers) if i % args.num_shards == args.shard_id]

    queue = None
    if args.work_queue_dir:
        required_fields, manifest_metadata = strict_judge_contract
        required_values, required_values_by_qa = (
            build_judge_queue_value_contract(
                inference_results,
                judge_metadata=manifest_metadata,
            )
        )
        queue = DurablePaperQueue(
            args.work_queue_dir,
            inference_results,
            required_qa_fields=required_fields,
            required_qa_field_values=required_values,
            required_qa_field_values_by_qa=required_values_by_qa,
            manifest_metadata=manifest_metadata,
        )
        queue.bootstrap([output_file])
        # Any worker can claim any paper; provider sizing below is global.
        my_papers = all_papers

    unresolved_fill = sum(
        1
        for paper_id, paper_data in my_papers
        for qa_id, qa in paper_data.get("QA", {}).items()
        if should_process_judge_row(
            (
                judge_results.get(paper_id, {})
                .get("QA", {})
                .get(qa_id)
            ),
            qa,
            retry_errors_only=args.retry_errors_only,
            retry_match_methods=retry_match_methods,
            expected_judge_protocol_fingerprint=(
                args.judge_protocol_fingerprint
            ),
        )
        and fill_needs_llm_judge(qa)
    )

    judge_provider = None
    if unresolved_fill > 0:
        print(
            f"[Shard {args.shard_id}] 加载裁判模型处理 "
            f"{unresolved_fill} 道简答题... "
            f"{progress_fields(0, max(1, unresolved_fill))} "
            "progress_scope=judge_model_load",
            flush=True,
        )
        judge_provider = create_provider(args.judge_provider_name, config_path=args.provider_config)
        print(
            f"[Shard {args.shard_id}] 裁判模型加载完成。 "
            f"{progress_fields(1, 1)} progress_scope=judge_model_load",
            flush=True,
        )

    if queue is not None:
        worker_id = args.worker_id or f"judge-gpu{args.gpu}-pid{os.getpid()}"
        initial_status = queue.status()
        print(
            f"[judge-queue-worker] event=worker_ready worker={worker_id} "
            f"{progress_fields(initial_status['completed_papers'], initial_status['total_papers'])} "
            "progress_scope=judge_papers",
            flush=True,
        )
        while not queue.is_complete():
            claim = queue.claim_next(worker_id)
            if claim is None:
                status = queue.status()
                quarantined = [
                    item
                    for item in status["failures"]
                    if item.get("quarantined")
                ]
                if (
                    status["unfinished_papers"]
                    and not status["active_claims"]
                    and quarantined
                    and not status.get("retryable_papers", 0)
                ):
                    raise RuntimeError(
                        f"{len(quarantined)} judge paper(s) quarantined"
                    )
                time.sleep(args.worker_idle_seconds)
                continue
            paper_data = inference_results[claim.paper_id]
            try:
                existing = queue.load_result(claim.paper_id)

                def checkpoint(result: dict) -> None:
                    queue.save_partial(claim.paper_id, result)
                    queue.heartbeat(claim)

                result = process_judge_paper(
                    claim.paper_id,
                    paper_data,
                    gold_results[claim.paper_id],
                    existing,
                    judge_provider=judge_provider,
                    args=args,
                    retry_match_methods=retry_match_methods,
                    checkpoint=checkpoint,
                )
                queue.finish(claim, result)
                status = queue.status()
                print(
                    f"[judge-queue-worker] event=paper_completed "
                    f"worker={worker_id} paper={claim.paper_id} "
                    f"{progress_fields(status['completed_papers'], status['total_papers'])} "
                    "progress_scope=judge_papers",
                    flush=True,
                )
            except KeyboardInterrupt:
                queue.release(claim)
                raise
            except BaseException as exc:
                queue.fail(claim, exc)
                raise
    else:
        total_papers = len(my_papers)
        for paper_index, (paper_id, paper_data) in enumerate(my_papers, start=1):
            existing_paper = judge_results.get(
                paper_id, {"paper": paper_id, "QA": {}}
            )
            result = process_judge_paper(
                paper_id,
                paper_data,
                gold_results[paper_id],
                existing_paper,
                judge_provider=judge_provider,
                args=args,
                retry_match_methods=retry_match_methods,
            )
            judge_results[paper_id] = result
            atomic_write_json(output_file, judge_results)
            print(
                f"[Shard {args.shard_id}] event=paper_completed "
                f"paper={paper_id} "
                f"{progress_fields(paper_index, total_papers)} "
                "progress_scope=judge_shard_papers",
                flush=True,
            )

    print(
        f"[Shard {args.shard_id}] 阅卷完成！ "
        f"{progress_fields(1, 1)} progress_scope=judge_worker",
        flush=True,
    )

if __name__ == "__main__":
    with managed_gpu_reservation("run_judge.py"):
        main()
