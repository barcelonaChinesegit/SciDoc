from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/generation/build_hard_cross_pdf_qa.py"
)
SPEC = importlib.util.spec_from_file_location("build_hard_cross", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def context() -> dict:
    return {
        "source_qas": [
            {
                "source_qa_id": "audited/QA1",
                "question": "What module produces embeddings?",
                "answer": "Encoder A produces embeddings.",
                "evidence_pages": [2, 12],
                "audit_decision": "KEEP",
            },
            {
                "source_qa_id": "audited/QA2",
                "question": "Which metric checks the downstream output?",
                "answer": "Metric B checks output quality.",
                "evidence_pages": [22, 23],
                "audit_decision": "KEEP",
            },
        ],
        "manifest": {
            "id": "xb_0001",
            "sources": [
                {"paper_id": "p1", "title": "One", "merged_start_page": 1, "merged_end_page": 10},
                {"paper_id": "p2", "title": "Two", "merged_start_page": 11, "merged_end_page": 20},
                {"paper_id": "p3", "title": "Three", "merged_start_page": 21, "merged_end_page": 30},
            ],
        },
        "bundle": {},
        "prior_bundle_summary": (
            "The papers share a coherent representation-validation pipeline."
        ),
    }


def valid_item() -> dict:
    return {
        "question": "Which pipeline dependency must hold for the output metric to validate the transferred representation?",
        "answer": (
            "Encoder A produces the representation consumed downstream. "
            "Metric B can validate output quality only after that "
            "representation is available."
        ),
        "derived_conclusion": (
            "Metric B can validate output quality only after that "
            "representation is available."
        ),
        "source_qa_ids": ["audited/QA1", "audited/QA2"],
        "reasoning_type": "adjacent_module_dependency",
        "relation": "The encoder output is the prerequisite input for downstream validation.",
        "intermediate_facts": [
            {"source_qa_id": "audited/QA1", "doc_numbers": [1, 2], "fact": "Encoder A produces the representation."},
            {"source_qa_id": "audited/QA2", "doc_numbers": [3], "fact": "Metric B validates output quality."},
            {"inference": "Validation follows representation production."},
        ],
        "source_necessity_tests": [
            {
                "source_qa_id": "audited/QA1",
                "missing_fact_if_removed": "the representation producer",
                "why_answer_is_impossible": "the pipeline input is unknown",
            },
            {
                "source_qa_id": "audited/QA2",
                "missing_fact_if_removed": "the downstream quality metric",
                "why_answer_is_impossible": "validation cannot be identified",
            },
        ],
        "quality_check": {flag: True for flag in MODULE.QUALITY_FLAGS},
    }


def test_prompt_hides_evidence_pages_but_keeps_document_membership() -> None:
    prompt = MODULE.build_messages("xb_0001", context(), 4, 0, True)[1]["content"]
    assert "evidence_pages" not in prompt
    assert "source_doc_numbers" in prompt
    assert "adjacent_module_dependency" in prompt
    assert "answer of every item must be at most 900 characters" in prompt
    assert "at or below 900 characters" in MODULE.SYSTEM_PROMPT
    assert "TARGET reasoning_type:" in prompt
    assert "do not return fewer merely for brevity" in prompt


def test_single_api_worker_argv_and_coordinator_assignment() -> None:
    result = MODULE.single_api_worker_argv(
        ["--target", "2400", "--api-workers=8", "--worker-id", "parent"], 3
    )
    assert result == [
        "--target",
        "2400",
        "--api-workers",
        "1",
        "--worker-id",
        "hard-cross-api-3",
    ]
    assert MODULE.is_aggregate_coordinator("local", "gpu-2")
    assert MODULE.is_aggregate_coordinator("gemini", "hard-cross-api-0")
    assert not MODULE.is_aggregate_coordinator("gemini", "hard-cross-api-1")
    assert MODULE.is_aggregate_coordinator("claude", "hard-cross-api-0")
    assert not MODULE.is_aggregate_coordinator("claude", "hard-cross-api-1")


def test_claude_generator_uses_qa_only_messages(monkeypatch, tmp_path) -> None:
    from pku_qa.workflows.review import review_cross_pdf_qa_api as review_module

    captured = {}

    class FakeResponse:
        def json(self):
            return {"content": [{"type": "text", "text": '{"items": []}'}]}

    def fake_post(url, headers, body, timeout, retries):
        captured.update(
            url=url,
            headers=headers,
            body=body,
            timeout=timeout,
            retries=retries,
        )
        return FakeResponse()

    monkeypatch.setattr(review_module, "load_key", lambda _path: "test-key")
    monkeypatch.setattr(review_module, "post_with_retries", fake_post)
    generator = MODULE.ClaudeGenerator(
        api_key_file=tmp_path / "missing.env",
        model="claude-sonnet-5",
        max_new_tokens=2000,
        temperature=0.1,
        timeout=30,
        retries=2,
    )

    output = generator.generate(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "audited QA facts"},
        ]
    )

    assert json.loads(output) == {"items": []}
    assert captured["body"]["system"] == "system rules"
    assert captured["body"]["messages"] == [
        {"role": "user", "content": "audited QA facts"}
    ]
    assert "image" not in json.dumps(captured["body"])


def test_cross_accepted_progress_uses_fallback(tmp_path) -> None:
    assert MODULE.accepted_progress(tmp_path, 9) == 9
    (tmp_path / "progress.json").write_text(
        '{"completed": 31}', encoding="utf-8"
    )
    assert MODULE.accepted_progress(tmp_path, 9) == 31


def test_target_reasoning_type_prefers_supported_weighted_relation() -> None:
    hints = [
        {
            "suggested_reasoning_types": [
                "component_hierarchy",
                "compatibility_judgment",
            ]
        }
    ]
    assert (
        MODULE.choose_target_reasoning_type("xb_0001", 0, hints)
        == "component_hierarchy"
    )
    assert (
        MODULE.choose_target_reasoning_type("xb_0008", 0, hints)
        == "compatibility_judgment"
    )


def test_validation_accepts_three_document_inference() -> None:
    item, errors = MODULE.validate_item(valid_item(), context(), 2, 5, set())
    assert not errors
    assert item["source_document_count"] == 3
    assert item["evidence_pages"] == [2, 12, 22, 23]


def test_three_document_only_gate_rejects_two_document_item() -> None:
    ctx = context()
    ctx["source_qas"][1]["evidence_pages"] = [12, 13]
    item, errors = MODULE.validate_item(
        valid_item(), ctx, 2, 5, set(), require_three_documents=True
    )
    assert item is None
    assert "requires_three_source_documents" in errors


def test_validation_rejects_source_answer_concatenation() -> None:
    raw = valid_item()
    raw["answer"] = "Encoder A produces embeddings; Metric B checks output quality."
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert item is None
    assert "answer_is_source_answer_concatenation" in errors


def test_validation_rejects_generic_contrast_but_allows_grounded_constraint() -> None:
    raw = valid_item()
    raw["question"] = (
        "How does the encoder bottleneck contrast with the output metric?"
    )
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert item is None
    assert "generic_cross_document_juxtaposition" in errors

    raw = valid_item()
    raw["question"] = (
        "If the target pipeline requires validation only after an encoder emits "
        "its representation, which dependency must hold?"
    )
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert "speculative_or_hypothetical_question" not in errors
    assert item is not None

    raw = valid_item()
    raw["answer"] = (
        "The learning rate is likely too aggressive and risks accelerating drift."
    )
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert item is None
    assert "unsupported_speculation_in_answer" in errors


def test_validation_allows_source_facts_plus_substantive_bridge() -> None:
    raw = valid_item()
    raw["answer"] = (
        "Encoder A produces embeddings and Metric B checks output quality. "
        "Together these facts establish an ordered production-to-validation "
        "dependency in which the representation must exist before downstream "
        "quality can be assessed."
    )
    raw["derived_conclusion"] = (
        "The representation must exist before downstream quality can be assessed."
    )
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert "answer_is_source_answer_concatenation" not in errors
    assert item is not None


def test_semantic_pair_requires_shared_topic_and_complementary_docs() -> None:
    ctx = context()
    ctx["source_qas"][0].update(
        {
            "question": "Which encoder representation feeds output validation?",
            "answer": "Encoder A produces the downstream representation.",
        }
    )
    ctx["source_qas"][1].update(
        {
            "question": "Which output metric validates the downstream representation?",
            "answer": "Metric B validates downstream representation quality.",
            "evidence_pages": [12, 22],
        }
    )
    hints = MODULE.semantic_source_pair_hints(ctx)
    assert len(hints) == 1
    assert hints[0]["source_qa_ids"] == ["audited/QA1", "audited/QA2"]
    assert hints[0]["union_doc_numbers"] == [1, 2, 3]
    prompt = MODULE.build_messages(
        "xb_0001", ctx, 2, 0, True, require_source_overlap=True
    )[1]["content"]
    assert "REQUIRED SEMANTIC SOURCE GROUPS" in prompt

    item, errors = MODULE.validate_item(
        valid_item(), ctx, 2, 5, set(), require_source_overlap=True
    )
    assert not errors
    assert item is not None

    unrelated = context()
    item, errors = MODULE.validate_item(
        valid_item(), unrelated, 2, 5, set(), require_source_overlap=True
    )
    assert item is None
    assert "source_qas_must_match_semantic_group" in errors


def test_semantic_pair_minimum_shared_terms_is_configurable() -> None:
    ctx = context()
    ctx["source_qas"][0].update(
        {
            "question": "Which downstream representation supports output validation?",
            "answer": "The downstream representation supports output validation.",
        }
    )
    ctx["source_qas"][1].update(
        {
            "question": "Which downstream representation enables output validation?",
            "answer": "The downstream representation enables output validation.",
        }
    )
    assert MODULE.semantic_source_pair_hints(ctx, min_shared_terms=2)
    assert not MODULE.semantic_source_pair_hints(ctx, min_shared_terms=20)

    prompt = MODULE.build_messages(
        "xb_0001",
        ctx,
        2,
        0,
        True,
        require_source_overlap=True,
        min_shared_terms=2,
    )[1]["content"]
    assert "REQUIRED SEMANTIC SOURCE GROUPS" in prompt


def test_relation_template_keeps_metric_label_compatibility_not_benchmark_overlap() -> None:
    ctx = context()
    ctx["source_qas"] = [
        {
            "source_qa_id": "audited/metric-categorical",
            "question": "Which agreement metric is used for categorical labels?",
            "answer": "Cohen kappa is used for discrete categorical labels.",
            "evidence_pages": [2],
        },
        {
            "source_qa_id": "audited/metric-ordinal",
            "question": "Which agreement metric is suitable for ordinal labels?",
            "answer": (
                "Cohen kappa is poorly suited for ordinal labels, so Pearson "
                "correlation is used."
            ),
            "evidence_pages": [22],
        },
    ]
    hints = MODULE.semantic_source_pair_hints(
        ctx, min_shared_terms=1, require_relation_template=True
    )
    assert len(hints) == 1
    assert "metric_reasoning" in hints[0]["suggested_reasoning_types"]
    assert "compatibility_judgment" in hints[0]["suggested_reasoning_types"]

    ctx["source_qas"] = [
        {
            "source_qa_id": "audited/bench-1",
            "question": "Which reasoning benchmarks are evaluated?",
            "answer": "The study evaluates AIME and GPQA-Diamond benchmarks.",
            "evidence_pages": [2],
        },
        {
            "source_qa_id": "audited/bench-2",
            "question": "Which reasoning benchmarks are evaluated?",
            "answer": "The study evaluates MMLU and GPQA-Diamond benchmarks.",
            "evidence_pages": [22],
        },
    ]
    assert not MODULE.semantic_source_pair_hints(
        ctx, min_shared_terms=1, require_relation_template=True
    )


def test_relation_template_accepts_rate_and_method_precondition() -> None:
    group = (
        {
            "question": "What convergence rate does the estimator achieve?",
            "answer": "The rate is O(1/sqrt(M)).",
        },
        {
            "question": "Which sampling method requires a Log-Sobolev condition?",
            "answer": "Langevin stochastic descent.",
        },
    )

    suggestions = MODULE.relation_template_suggestions(group, {"convergence"})

    assert "metric_reasoning" in suggestions
    assert "compatibility_judgment" in suggestions


def test_overlap_validation_uses_configured_threshold_and_relation_template() -> None:
    ctx = context()
    ctx["source_qas"][0].update(
        {
            "question": "Which encoder produces calibration?",
            "answer": "The encoder produces calibration.",
        }
    )
    ctx["source_qas"][1].update(
        {
            "question": "Which metric checks calibration?",
            "answer": "The output metric checks calibration.",
        }
    )
    item, errors = MODULE.validate_item(
        valid_item(),
        ctx,
        2,
        5,
        set(),
        require_source_overlap=True,
        min_shared_terms=2,
        require_relation_template=True,
    )
    assert item is None
    assert "source_qas_must_match_semantic_group" in errors


def test_load_prepared_contexts_bypasses_legacy_bundle_lookup(tmp_path) -> None:
    payload = {
        "contexts": {
            "hxb_0001": {
                "manifest": {
                    "id": "hxb_0001",
                    "sources": [
                        {"paper_id": "1", "merged_start_page": 1, "merged_end_page": 10},
                        {"paper_id": "2", "merged_start_page": 11, "merged_end_page": 20},
                        {"paper_id": "3", "merged_start_page": 21, "merged_end_page": 30},
                    ],
                },
                "source_qas": [
                    {"source_qa_id": "base/1/QA1", "evidence_pages": [2]},
                    {"source_qa_id": "base/2/QA1", "evidence_pages": [12]},
                    {"source_qa_id": "base/3/QA1", "evidence_pages": [22]},
                ],
                "prior_bundle_summary": "Shared ordinal evaluation anchors.",
            }
        }
    }
    path = tmp_path / "contexts.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    source = {"hxb_0001": {"primary_category": "Computer Science"}}
    contexts, manifest = MODULE.load_prepared_contexts(path, source)
    assert contexts["hxb_0001"]["bundle"] == source["hxb_0001"]
    assert manifest[0]["id"] == "hxb_0001"


def test_prior_bundle_coherence_rejects_negative_audit_summary() -> None:
    ctx = context()
    assert MODULE.prior_bundle_is_coherent(ctx)
    ctx["prior_bundle_summary"] = (
        "The papers are unrelated and lack a common scientific focus."
    )
    assert not MODULE.prior_bundle_is_coherent(ctx)


def test_validation_rejects_parallel_document_labeled_comparison() -> None:
    raw = valid_item()
    raw["question"] = "How do Doc 1 and Doc 2 differ in output validation?"
    raw["answer"] = "Doc 1 produces representations and Doc 2 checks them."
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert item is None
    assert "question_uses_document_labels" in errors
    assert "answer_uses_document_labels" in errors


def test_single_item_response_shape_is_deterministically_wrapped() -> None:
    response = valid_item()
    normalized, steps = MODULE.normalize_response_shape(response, "xb_0001")
    assert normalized == {"bundle_id": "xb_0001", "items": [response]}
    assert steps == ["wrapped_single_item_response"]


def test_document_label_repair_uses_actual_names() -> None:
    class Generator:
        def generate(self, messages):
            assert "Doc 1" in messages[-1]["content"]
            return (
                '{"question":"Why does Encoder A precede Metric B?",'
                '"answer":"Encoder A produces the representation before Metric B validates it.",'
                '"derived_conclusion":"Metric B depends on Encoder A output."}'
            )

    response = {
        "bundle_id": "xb_0001",
        "items": [
            {
                **valid_item(),
                "question": "Why does Doc 1 precede Doc 3?",
                "answer": "Doc 1 produces the input and Doc 3 validates it.",
            }
        ],
    }
    repaired, logs = MODULE.repair_document_labels(
        Generator(), response, context()
    )
    assert logs[0]["accepted"] is True
    assert "Doc 1" not in repaired["items"][0]["question"]
    assert "Encoder A" in repaired["items"][0]["answer"]


def test_cross_question_leakage_repair_hides_source_answers() -> None:
    class Generator:
        def generate(self, messages):
            assert "BANNED ANSWER PHRASES" in messages[-1]["content"]
            return (
                '{"question":"Which production-to-validation dependency must '
                'hold before the downstream metric can be applied?"}'
            )

    response = {"bundle_id": "xb_0001", "items": [valid_item()]}
    response["items"][0]["question"] = (
        "If Encoder A produces embeddings and Metric B checks output quality, "
        "which dependency holds?"
    )

    repaired, logs = MODULE.repair_leaked_cross_questions(
        Generator(), response, context()
    )

    assert logs[0]["accepted"] is True
    assert not MODULE.leaked_answer_phrases(repaired["items"][0], context())
    assert "production-to-validation" in repaired["items"][0]["question"]


def test_validation_rejects_inflectional_paraphrase_of_source_answer() -> None:
    raw = valid_item()
    raw["question"] = (
        "Given that the encoder is producing embeddings and the metric checks "
        "output quality, which dependency holds?"
    )
    item, errors = MODULE.validate_item(raw, context(), 2, 5, set())
    assert item is None
    assert "source_answer_leaked_in_question:audited/QA1" in errors
    assert "source_answer_leaked_in_question:audited/QA2" in errors


def test_validation_rejects_acronym_expansion_stitch() -> None:
    ctx = context()
    ctx["source_qas"][0]["question"] = "What tool is used in the equilibrium proof?"
    ctx["source_qas"][0]["answer"] = "Backward stochastic differential equation (BSDE)"
    ctx["source_qas"][1]["question"] = "What is the full name of BSDE?"
    ctx["source_qas"][1]["answer"] = "Backward Stochastic Differential Equations"
    raw = valid_item()
    raw["question"] = (
        "What framework is used in the equilibrium proof, and how is it formally defined?"
    )
    raw["answer"] = (
        "The proof uses BSDE, formally defined as Backward Stochastic Differential Equations."
    )
    raw["derived_conclusion"] = "The framework identity is BSDE."
    raw["relation"] = "Links the application to the acronym's full name and identity."
    item, errors = MODULE.validate_item(raw, ctx, 2, 5, set())
    assert item is None
    assert "parallel_subquestions" in errors
    assert "acronym_expansion_stitch" in errors


def test_cross_relation_self_review_can_remove_contrived_item() -> None:
    class Generator:
        def generate(self, messages):
            assert "invents a unified pipeline" in messages[-1]["content"]
            return '{"bundle_id":"xb_0001","items":[]}'

    response = {"bundle_id": "xb_0001", "items": [valid_item()]}
    reviewed, log = MODULE.self_review_cross_relations(
        Generator(), response, "xb_0001", context()
    )
    assert reviewed == {"bundle_id": "xb_0001", "items": []}
    assert log["input_item_count"] == 1
    assert log["output_item_count"] == 0


def test_semantic_group_can_link_three_single_document_facts() -> None:
    ctx = context()
    ctx["source_qas"] = [
        {
            "source_qa_id": "base/p1/QA1",
            "question": "Which retrieval module supplies grounded context?",
            "answer": "Dense retrieval supplies grounded context.",
            "evidence_pages": [2],
        },
        {
            "source_qa_id": "base/p2/QA1",
            "question": "Which routing module activates grounded retrieval?",
            "answer": "Conditional routing activates dense retrieval.",
            "evidence_pages": [12],
        },
        {
            "source_qa_id": "base/p3/QA1",
            "question": "Which grounded retrieval output receives verification?",
            "answer": "Claim verification checks grounded context.",
            "evidence_pages": [22],
        },
    ]
    hints = MODULE.semantic_source_pair_hints(ctx)
    triple = next(hint for hint in hints if len(hint["source_qa_ids"]) == 3)
    assert triple["union_doc_numbers"] == [1, 2, 3]
    assert set(triple["source_qa_ids"]) == {
        "base/p1/QA1",
        "base/p2/QA1",
        "base/p3/QA1",
    }


def test_semantic_group_can_link_four_facts_across_three_documents() -> None:
    ctx = context()
    ctx["source_qas"] = [
        {
            "source_qa_id": "base/p1/QA1",
            "question": "Which calibrated encoder emits a grounded representation?",
            "answer": "The calibrated encoder emits a grounded representation.",
            "evidence_pages": [2],
        },
        {
            "source_qa_id": "base/p2/QA1",
            "question": "Which calibrated grounded router consumes the encoder representation?",
            "answer": "The calibrated grounded router consumes the encoder representation.",
            "evidence_pages": [12],
        },
        {
            "source_qa_id": "base/p2/QA2",
            "question": "Which verified router produces a grounded decision?",
            "answer": "The verified router produces a grounded decision.",
            "evidence_pages": [13],
        },
        {
            "source_qa_id": "base/p3/QA1",
            "question": "Which verified decision receives grounded validation?",
            "answer": "The verified decision receives grounded validation.",
            "evidence_pages": [22],
        },
    ]
    hints = MODULE.semantic_source_pair_hints(ctx, min_shared_terms=2)
    four_fact = next(hint for hint in hints if len(hint["source_qa_ids"]) == 4)
    assert four_fact["union_doc_numbers"] == [1, 2, 3]

    raw = valid_item()
    raw["source_qa_ids"] = four_fact["source_qa_ids"]
    raw["intermediate_facts"] = [
        {
            "source_qa_id": source_id,
            "doc_numbers": [1 if "p1" in source_id else 2 if "p2" in source_id else 3],
            "fact": "A necessary connected chain fact.",
        }
        for source_id in four_fact["source_qa_ids"]
    ] + [{"inference": "The four facts form one connected three-document chain."}]
    raw["source_necessity_tests"] = [
        {
            "source_qa_id": source_id,
            "missing_fact_if_removed": "one necessary chain fact",
            "why_answer_is_impossible": "the connected chain would be broken",
        }
        for source_id in four_fact["source_qa_ids"]
    ]
    item, errors = MODULE.validate_item(
        raw, ctx, 2, 5, set(), require_source_overlap=True
    )
    assert "source_qas_must_match_semantic_group" not in errors
    assert item is not None


def test_audited_evidence_facts_keep_only_supported_keep_fix_rows() -> None:
    record = {
        "review": {
            "items": [
                {
                    "qa_id": "QA1",
                    "decision": "KEEP",
                    "support": [
                        {
                            "doc_number": 1,
                            "merged_page": 2,
                            "supported_fact": "Encoder A emits grounded context.",
                        },
                        {
                            "doc_number": 2,
                            "merged_page": 12,
                            "supported_fact": "Router B consumes grounded context.",
                        },
                    ],
                },
                {
                    "qa_id": "QA2",
                    "decision": "REJECT",
                    "support": [
                        {
                            "doc_number": 3,
                            "merged_page": 22,
                            "supported_fact": "This rejected row must not leak in.",
                        }
                    ],
                },
            ]
        }
    }
    facts = MODULE.audited_evidence_fact_qas(record)
    assert [row["source_qa_id"] for row in facts] == [
        "audited_fact/QA1/1",
        "audited_fact/QA1/2",
    ]
    assert [row["source_doc_number"] for row in facts] == [1, 2]
    assert all(row["audit_decision"] == "KEEP_EVIDENCE_FACT" for row in facts)


def test_one_specific_shared_term_can_link_audited_facts() -> None:
    ctx = context()
    ctx["source_qas"] = [
        {
            "source_qa_id": "audited_fact/QA1/1",
            "question": "fact",
            "answer": "Calibration is produced by the encoder.",
            "semantic_text": "Calibration is produced by the encoder.",
            "evidence_pages": [2],
        },
        {
            "source_qa_id": "audited_fact/QA1/2",
            "question": "fact",
            "answer": "Calibration is checked after routing.",
            "semantic_text": "Calibration is checked after routing.",
            "evidence_pages": [12],
        },
    ]
    hints = MODULE.semantic_source_pair_hints(ctx)
    assert len(hints) == 1
    assert hints[0]["shared_terms"] == ["calibration"]
