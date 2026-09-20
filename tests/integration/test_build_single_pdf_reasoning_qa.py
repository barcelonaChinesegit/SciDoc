from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/generation/build_single_pdf_reasoning_qa.py"
)
SPEC = importlib.util.spec_from_file_location("build_reasoning", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def paper() -> dict:
    return {
        "paper": "1",
        "primary_category": "Economics",
        "secondary_category": "Theory",
        "QA": {
            "QA1": {
                "question": "Which policy increases output?",
                "answer": "Policy A",
                "evidence_pages": [1],
                "question_type": "Literal",
                "question_category": "Result",
            },
            "QA2": {
                "question": "What follows from higher output?",
                "answer": "Lower prices",
                "evidence_pages": [2],
                "question_type": "Inferential",
                "question_category": "Result",
            },
            "QA3": {
                "question": "Which group is constrained?",
                "answer": "Group C",
                "evidence_pages": [3],
                "question_type": "Literal",
                "question_category": "Result",
            },
            "QA4": {
                "question": "What is the baseline method?",
                "answer": "Method D",
                "evidence_pages": [4],
                "question_type": "Literal",
                "question_category": "Method",
            },
        },
    }


def valid_item() -> dict:
    return {
        "question": "If the output-increasing policy is applied, what price effect follows?",
        "answer": "Lower prices",
        "source_qa_ids": ["QA1", "QA2"],
        "reasoning_type": "causal_chain",
        "relation": "The policy changes output, which determines prices.",
        "derivation": [
            {"source_qa_id": "QA1", "fact": "Policy A increases output."},
            {"source_qa_id": "QA2", "fact": "Higher output lowers prices."},
            {"inference": "Policy A therefore lowers prices."},
        ],
        "quality_check": {
            "requires_all_sources": True,
            "not_simple_concatenation": True,
            "answerable_from_source_qas": True,
            "answer_unique_and_concise": True,
        },
    }


def test_prompt_excludes_evidence_pages() -> None:
    messages = MODULE.build_messages("1", paper(), 3, 2, 4)
    prompt = messages[-1]["content"]
    assert '"evidence_pages": [' not in prompt
    assert "Which policy increases output?" in prompt
    assert "Policy A" in prompt
    assert all(qa["qa_id"] in prompt for qa in MODULE.compact_original_qas(paper()))


def test_sequence_ratio_upper_bound_preserves_exact_decision() -> None:
    samples = (
        ("the same generated question", "the same generated question", 0.88),
        ("a metric for model calibration", "a metric for model reliability", 0.88),
        ("short unrelated question", "a substantially different prompt", 0.88),
        ("boundary comparison alpha", "boundary comparison beta", 0.9),
    )
    for left, right, threshold in samples:
        expected = SequenceMatcher(None, left, right).ratio() >= threshold
        assert MODULE.sequence_ratio_at_least(left, right, threshold) is expected


def test_question_similarity_index_matches_exhaustive_sequence_matcher() -> None:
    prior_questions = (
        "the same generated question",
        "which metric measures calibration error",
        "what model component produces the hidden representation",
        "how many samples satisfy the stated filtering condition",
    )
    candidates = (
        "the same generated question",
        "which metric measures calibration errors",
        "what unrelated baseline is used",
        "how many samples satisfy a different filtering condition",
    )
    index = MODULE.QuestionSimilarityIndex()
    for question in prior_questions:
        index.add(question)
    for question in candidates:
        expected = any(
            SequenceMatcher(None, question, prior).ratio() >= 0.88
            for prior in prior_questions
        )
        assert index.is_near_duplicate(question, 0.88) is expected


def test_aggregate_interval_scales_for_large_candidate_pool() -> None:
    assert MODULE.aggregate_checkpoint_interval(500) == 100
    assert MODULE.aggregate_checkpoint_interval(2000) == 200
    assert MODULE.aggregate_checkpoint_interval(5000) == 500


def test_validate_response_reuses_supplied_semantic_hints() -> None:
    source = paper()
    item = valid_item()
    cached_original = MODULE.compact_original_qas(source)
    cached_hints = []
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        source,
        2,
        4,
        set(),
        original_qas=cached_original,
        semantic_hints=cached_hints,
    )
    assert accepted
    assert not issues


def test_prompt_restricts_reasoning_type_to_five_canonical_values() -> None:
    messages = MODULE.build_messages("1", paper(), 3, 2, 4, generation_round=7)
    prompt = "\n".join(message["content"] for message in messages)
    assert "generation_round=7" in prompt
    assert MODULE.REASONING_TYPES == {
        "causal_chain",
        "conditional_inference",
        "comparison",
        "constraint_intersection",
        "multi_step_calculation",
    }
    assert "reasoning_type must be exactly one of" in prompt


def test_validation_rejects_qa_ids_in_answer() -> None:
    item = valid_item()
    item["answer"] = "Policy A raises output (QA1), which leads to lower prices (QA2)."
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not accepted
    assert "answer_mentions_qa_id" in issues[0]["errors"]


def test_parenthetical_source_citations_are_removed_and_audited() -> None:
    response = {
        "paper_id": "1",
        "items": [
            {
                "question": "What follows?",
                "answer": "Policy A raises output (QA1), lowering prices (from QA2).",
                "derived_conclusion": "Policy A lowers prices (QA2).",
            }
        ],
    }
    normalized, changes = MODULE.strip_parenthetical_source_citations(response)
    item = normalized["items"][0]
    assert item["answer"] == "Policy A raises output, lowering prices."
    assert item["derived_conclusion"] == "Policy A lowers prices."
    assert len(changes) == 2


def test_hard_prompt_uses_id_allowlist_without_evidence_pages() -> None:
    messages = MODULE.build_messages(
        "1", paper(), 3, 2, 4, generation_round=1, hard_mode=True
    )
    prompt = messages[-1]["content"]
    assert "ALLOWED SOURCE QA COMBINATIONS" in prompt
    assert '"evidence_pages": [' not in prompt
    assert '["QA1", "QA4"]' in prompt


def test_hard_validation_requires_nonadjacent_pages_and_no_leakage() -> None:
    item = valid_item()
    item.update(
        {
            "question": "Which baseline follows after applying the output constraint?",
            "answer": "The applicable baseline is Method D because the constraint selects it.",
            "source_qa_ids": ["QA1", "QA4"],
            "reasoning_focus": "conditional_filtering",
            "intermediate_facts": ["The output constraint selects the baseline."],
            "source_necessity_tests": [
                {
                    "source_qa_id": "QA1",
                    "missing_fact_if_removed": "which policy changes output",
                    "why_answer_is_impossible": "the constraint cannot be applied",
                },
                {
                    "source_qa_id": "QA4",
                    "missing_fact_if_removed": "which method is the baseline",
                    "why_answer_is_impossible": "the selected baseline is unknown",
                },
            ],
            "derivation": [
                {"source_qa_id": "QA1", "fact": "Policy A raises output."},
                {"source_qa_id": "QA4", "fact": "Method D is the baseline."},
                {"inference": "The constraint selects Method D."},
            ],
        }
    )
    item["quality_check"].update(
        {flag: True for flag in MODULE.HARD_QUALITY_FLAGS}
    )
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
        hard_mode=True,
        min_evidence_span=3,
    )
    assert not issues
    assert accepted[0]["source_evidence_pages"] == [1, 4]
    assert accepted[0]["construction_key_evidence_pages"] == [1, 4]
    leaked = dict(item)
    leaked["question"] = "Is Method D the applicable baseline?"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [leaked]},
        "1",
        paper(),
        2,
        4,
        set(),
        hard_mode=True,
        min_evidence_span=3,
    )
    assert not accepted
    assert any(
        "paper_answer_leaked_in_question:QA4" in issue["errors"]
        for issue in issues
    )


def test_key_evidence_selection_maximizes_pairwise_distance() -> None:
    source = paper()
    source["QA"]["QA4"]["evidence_pages"] = [4, 9]
    assert MODULE.select_separated_pages(["QA1", "QA4"], source, 3) == [1, 9]


def test_leakage_repair_rewrites_only_the_question() -> None:
    item = valid_item()
    item["question"] = "If Policy A is used, what follows?"

    class Generator:
        def generate(self, messages):
            assert "Policy A" in messages[-1]["content"]
            return '{"question":"Which policy choice produces the downstream price effect?","answer":"The applicable choice raises output, which then causes prices to fall."}'

    response = {"paper_id": "1", "items": [item]}
    repaired, logs = MODULE.repair_leaked_questions(
        Generator(), response, paper()
    )
    assert logs[0]["accepted"] is True
    assert "prices to fall" in repaired["items"][0]["answer"]
    assert "Policy A" not in repaired["items"][0]["question"]


def test_semantic_relation_hints_require_topic_link_and_page_separation() -> None:
    chain_paper = paper()
    chain_paper["QA"]["QA1"].update(
        {
            "question": "Which encoder design improves calibration stability?",
            "answer": "A low-rank encoder",
        }
    )
    chain_paper["QA"]["QA2"].update(
        {
            "question": "Which metric diagnoses calibration stability?",
            "answer": "Expected calibration error",
            "evidence_pages": [5],
        }
    )
    hints = MODULE.semantic_chain_hints(chain_paper, min_span=3)
    assert any(hint["source_qa_ids"] == ["QA1", "QA2"] for hint in hints)
    prompt = MODULE.build_messages(
        "1",
        chain_paper,
        2,
        2,
        4,
        hard_mode=True,
        require_semantic_chain=True,
    )[-1]["content"]
    assert "SEMANTIC RELATION HINTS" in prompt
    assert '"source_qa_ids": ["QA1", "QA2"]' in prompt.replace("\n", " ")


def test_semantic_hints_are_batched_by_generation_round(monkeypatch) -> None:
    hints = [
        {
            "source_qa_ids": ["QA1", "QA4"],
            "relation_hint": f"relation_{index}",
        }
        for index in range(10)
    ]
    monkeypatch.setattr(MODULE, "semantic_chain_hints", lambda *_args, **_kwargs: hints)
    first = MODULE.build_messages(
        "1",
        paper(),
        2,
        2,
        4,
        generation_round=0,
        hard_mode=True,
        require_semantic_chain=True,
    )[-1]["content"]
    second = MODULE.build_messages(
        "1",
        paper(),
        2,
        2,
        4,
        generation_round=1,
        hard_mode=True,
        require_semantic_chain=True,
    )[-1]["content"]
    assert "relation_0" in first
    assert "relation_6" not in first
    assert "relation_6" in second
    assert "relation_0" in second  # The second batch wraps deterministically.


def test_gemini_generator_uses_json_api_contract(monkeypatch, tmp_path) -> None:
    from pku_qa.workflows.review import review_cross_pdf_qa_api as api

    captured = {}

    class Response:
        def json(self):
            return {"candidate": "payload"}

    def fake_post(url, headers, body, timeout, retries):
        captured.update(
            url=url,
            headers=headers,
            body=body,
            timeout=timeout,
            retries=retries,
        )
        return Response()

    monkeypatch.setattr(api, "load_key", lambda _path: "test-key")
    monkeypatch.setattr(api, "post_with_retries", fake_post)
    monkeypatch.setattr(api, "response_text_gemini", lambda _raw: '{"ok":true}')
    generator = MODULE.GeminiGenerator(
        api_key_file=tmp_path / ".env",
        model="gemini-2.5-flash",
        max_new_tokens=900,
        temperature=0.1,
        timeout=30,
        retries=2,
    )
    output = generator.generate(
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "build one item"},
        ]
    )
    assert output == '{"ok":true}'
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert captured["body"]["system_instruction"]["parts"][0]["text"] == "system rules"
    assert captured["headers"]["x-goog-api-key"] == "test-key"


def test_single_api_worker_argv_removes_parent_worker_flags() -> None:
    result = MODULE.single_api_worker_argv(
        [
            "--target",
            "2000",
            "--api-workers=4",
            "--worker-id",
            "parent",
            "--hard-mode",
        ],
        2,
    )
    assert result == [
        "--target",
        "2000",
        "--hard-mode",
        "--api-workers",
        "1",
        "--worker-id",
        "reasoning-api-2",
    ]


def test_only_primary_gemini_worker_coordinates_aggregate_refresh() -> None:
    assert MODULE.is_aggregate_coordinator("local", "gpu-2")
    assert MODULE.is_aggregate_coordinator("gemini", None)
    assert MODULE.is_aggregate_coordinator("gemini", "reasoning-api-0")
    assert not MODULE.is_aggregate_coordinator("gemini", "reasoning-api-1")


def test_accepted_progress_uses_durable_count_and_fallback(tmp_path) -> None:
    assert MODULE.accepted_progress(tmp_path, 17) == 17
    (tmp_path / "progress.json").write_text(
        '{"completed": 42}', encoding="utf-8"
    )
    assert MODULE.accepted_progress(tmp_path, 17) == 42


def test_generation_progress_heartbeat_preserves_durable_count(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(MODULE.time, "time", lambda: 1234.5)
    MODULE.write_generation_progress(
        tmp_path,
        target=5000,
        completed=3700,
        work_state="generating_checkpoints",
    )

    progress = MODULE.read_json(tmp_path / "progress.json")
    assert progress == {
        "status": "in_progress",
        "completed": 3700,
        "total": 5000,
        "percent": 74.0,
        "stage": "reasoning_qa_generation",
        "work_state": "generating_checkpoints",
        "updated_at": 1234.5,
    }


def test_parallel_api_parent_cleans_children_when_wait_is_interrupted(monkeypatch) -> None:
    processes = []

    class Process:
        def __init__(self, _command):
            self.terminated = False
            processes.append(self)

        def poll(self):
            return -15 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

        def wait(self, timeout=None):
            if timeout is None and not self.terminated:
                raise KeyboardInterrupt
            return -15

    monkeypatch.setattr(MODULE.subprocess, "Popen", Process)
    monkeypatch.setattr(MODULE.signal, "signal", lambda _signum, _handler: None)
    monkeypatch.setattr(MODULE.sys, "argv", ["generator.py", "--api-workers", "2"])
    try:
        MODULE.run_parallel_api_workers(2)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Interrupted parent wait must propagate")
    assert len(processes) == 2
    assert all(process.terminated for process in processes)


def test_semantic_relation_hints_prioritize_comparable_numeric_derivation() -> None:
    numeric_paper = paper()
    numeric_paper["QA"]["QA1"].update(
        {
            "question": "What is the server CPU load reduction percentage?",
            "answer": "80",
        }
    )
    numeric_paper["QA"]["QA2"].update(
        {
            "question": "What is the cache CPU load reduction percentage?",
            "answer": "63",
            "evidence_pages": [5],
        }
    )
    hints = MODULE.semantic_chain_hints(numeric_paper, min_span=3)
    assert hints[0]["source_qa_ids"] == ["QA1", "QA2"]
    assert hints[0]["relation_hint"] == "comparable_numeric_derivation"
    assert hints[0]["measurement_signatures"] == ["percentage"]
    assert hints[0]["measurement_subject_terms"] == ["cpu", "load"]
    assert hints[0]["numeric_values"] == {"QA1": 80.0, "QA2": 63.0}
    assert "difference_or_percentage_point_gap" in hints[0]["allowed_operations"]


def test_identity_hidden_sequential_chain_is_prioritized_and_validated() -> None:
    chain_paper = paper()
    chain_paper["QA"]["QA1"].update(
        {
            "question": "Which antibody has the higher isoelectric point?",
            "answer": "Ipilimumab",
        }
    )
    chain_paper["QA"]["QA2"].update(
        {
            "question": "How does high BMI affect uptake of Ipilimumab?",
            "answer": "Increases by 22%",
            "evidence_pages": [6],
        }
    )
    hints = MODULE.semantic_chain_hints(chain_paper, min_span=3)
    assert hints[0]["relation_hint"] == "identity_hidden_sequential_chain"
    assert hints[0]["source_qa_ids"] == ["QA1", "QA2"]

    item = valid_item()
    item.update(
        {
            "question": (
                "For the antibody with the higher isoelectric point, how does "
                "high BMI change its uptake?"
            ),
            "answer": (
                "The higher-isoelectric-point antibody is Ipilimumab, and high "
                "BMI increases its uptake by 22%."
            ),
            "derived_conclusion": (
                "Ipilimumab is the higher-isoelectric-point antibody whose "
                "uptake rises by 22% under high BMI."
            ),
            "source_qa_ids": ["QA1", "QA2"],
            "reasoning_focus": "sequential_reasoning",
            "intermediate_facts": [
                "The higher-isoelectric-point antibody must first be identified.",
                "The BMI-dependent uptake change then applies to that antibody.",
            ],
            "source_necessity_tests": [
                {
                    "source_qa_id": "QA1",
                    "missing_fact_if_removed": "the antibody identity",
                    "why_answer_is_impossible": "the unnamed antibody cannot be resolved",
                },
                {
                    "source_qa_id": "QA2",
                    "missing_fact_if_removed": "the uptake change",
                    "why_answer_is_impossible": "the downstream effect is unknown",
                },
            ],
            "derivation": [
                {"source_qa_id": "QA1", "fact": "Ipilimumab has the higher pI."},
                {"source_qa_id": "QA2", "fact": "High BMI raises its uptake by 22%."},
                {"inference": "The unnamed higher-pI antibody has that uptake change."},
            ],
        }
    )
    item["quality_check"].update(
        {flag: True for flag in MODULE.HARD_QUALITY_FLAGS}
    )
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        chain_paper,
        2,
        4,
        set(),
        hard_mode=True,
        min_evidence_span=3,
        require_semantic_chain=True,
    )
    assert not issues
    assert accepted[0]["derived_conclusion"].startswith("Ipilimumab")


def test_identity_chain_rejects_generic_new_method_name_lookup() -> None:
    chain_paper = paper()
    chain_paper["QA"]["QA1"].update(
        {
            "question": "What is the new solution concept proposed in the paper?",
            "answer": "Credible bargaining solution",
        }
    )
    chain_paper["QA"]["QA2"].update(
        {
            "question": "What matching works with the credible bargaining solution?",
            "answer": "Any optimal matching",
            "evidence_pages": [8],
        }
    )
    hints = MODULE.semantic_chain_hints(chain_paper, min_span=3)
    assert not any(
        hint.get("relation_hint") == "identity_hidden_sequential_chain"
        and hint["source_qa_ids"] == ["QA1", "QA2"]
        for hint in hints
    )


def test_semantic_relation_selection_prioritizes_numeric_papers() -> None:
    semantic_paper = paper()
    semantic_paper["QA"]["QA1"].update(
        {
            "question": "Which encoder design improves calibration stability?",
            "answer": "A low-rank encoder",
        }
    )
    semantic_paper["QA"]["QA2"].update(
        {
            "question": "Which metric diagnoses calibration stability?",
            "answer": "Expected calibration error",
            "evidence_pages": [5],
        }
    )
    numeric_paper = paper()
    numeric_paper["QA"]["QA1"].update(
        {
            "question": "What is the server CPU load reduction percentage?",
            "answer": "80",
        }
    )
    numeric_paper["QA"]["QA2"].update(
        {
            "question": "What is the cache CPU load reduction percentage?",
            "answer": "63",
            "evidence_pages": [5],
        }
    )
    selected = MODULE.select_papers(
        {"semantic": semantic_paper, "numeric": numeric_paper},
        2,
        123,
        hard_mode=True,
        min_evidence_span=3,
        require_semantic_chain=True,
    )
    assert selected[0] == "numeric"


def test_semantic_relation_does_not_mark_unrelated_counts_comparable() -> None:
    numeric_paper = paper()
    numeric_paper["QA"]["QA1"].update(
        {
            "question": "How many stages are in the water framework?",
            "answer": "4",
        }
    )
    numeric_paper["QA"]["QA2"].update(
        {
            "question": "How many treaties inform the water framework?",
            "answer": "14",
            "evidence_pages": [5],
        }
    )
    hints = MODULE.semantic_chain_hints(numeric_paper, min_span=3)
    assert hints
    assert hints[0]["relation_hint"] == "semantic_constraint_or_dependency"


def test_semantic_relation_does_not_compare_heterogeneous_percentages() -> None:
    numeric_paper = paper()
    numeric_paper["QA"]["QA1"].update(
        {
            "question": "What is the model memory reduction percentage?",
            "answer": "80",
        }
    )
    numeric_paper["QA"]["QA2"].update(
        {
            "question": "What is the training FLOPs reduction percentage?",
            "answer": "63",
            "evidence_pages": [5],
        }
    )
    hints = MODULE.semantic_chain_hints(numeric_paper, min_span=3)
    matching = [
        hint
        for hint in hints
        if hint["source_qa_ids"] == ["QA1", "QA2"]
    ]
    assert matching
    assert matching[0]["relation_hint"] == "semantic_constraint_or_dependency"


def test_semantic_relation_answer_must_include_new_conclusion() -> None:
    chain_paper = paper()
    chain_paper["QA"]["QA1"].update(
        {
            "question": "Which encoder design improves calibration stability?",
            "answer": "A low-rank encoder",
        }
    )
    chain_paper["QA"]["QA2"].update(
        {
            "question": "Which metric diagnoses calibration stability?",
            "answer": "Expected calibration error",
            "evidence_pages": [5],
        }
    )
    item = valid_item()
    item.update(
        {
            "question": "Which monitoring choice should accompany the dimension-reducing design to form a stable diagnosed pipeline?",
            "answer": (
                "The low-rank design should use expected-calibration-error "
                "monitoring, yielding a stable pipeline whose calibration "
                "quality is directly diagnosed."
            ),
            "derived_conclusion": (
                "A low-rank design paired with expected-calibration-error "
                "monitoring yields a stable diagnosed pipeline."
            ),
            "source_qa_ids": ["QA1", "QA2"],
            "reasoning_focus": "sequential_reasoning",
            "intermediate_facts": [
                "The design reduces rank while the metric diagnoses calibration."
            ],
            "source_necessity_tests": [
                {
                    "source_qa_id": "QA1",
                    "missing_fact_if_removed": "the stability-oriented design",
                    "why_answer_is_impossible": "the pipeline component is unknown",
                },
                {
                    "source_qa_id": "QA2",
                    "missing_fact_if_removed": "the calibration diagnostic",
                    "why_answer_is_impossible": "monitoring cannot be selected",
                },
            ],
        }
    )
    item["quality_check"].update(
        {flag: True for flag in MODULE.HARD_QUALITY_FLAGS}
    )
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        chain_paper,
        2,
        4,
        set(),
        hard_mode=True,
        min_evidence_span=3,
        require_semantic_chain=True,
    )
    assert not issues
    assert len(accepted) == 1

    item["derived_conclusion"] = "An unrelated unsupported conclusion."
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        chain_paper,
        2,
        4,
        set(),
        hard_mode=True,
        min_evidence_span=3,
        require_semantic_chain=True,
    )
    assert not accepted
    errors = issues[0]["errors"]
    assert "derived_conclusion_missing_from_answer" in errors


def test_semantic_relation_self_review_can_remove_unsupported_item() -> None:
    class Generator:
        def generate(self, messages):
            assert "changed-condition extrapolation" in messages[-1]["content"]
            return '{"paper_id":"1","items":[]}'

    response = {"paper_id": "1", "items": [valid_item()]}
    reviewed, log = MODULE.self_review_semantic_relations(
        Generator(), response, "1", paper()
    )
    assert reviewed == {"paper_id": "1", "items": []}
    assert log["accepted_response"] is True
    assert log["input_item_count"] == 1
    assert log["output_item_count"] == 0


def test_semantic_relation_self_review_normalizes_empty_object_to_rejection() -> None:
    class Generator:
        def generate(self, messages):
            return "{}"

    reviewed, log = MODULE.self_review_semantic_relations(
        Generator(), {"paper_id": "1", "items": [valid_item()]}, "1", paper()
    )
    assert reviewed == {"paper_id": "1", "items": []}
    assert log["normalized_empty_object_to_empty_items"] is True


def test_semantic_relation_self_review_accepts_empty_array_as_rejection() -> None:
    class Generator:
        def generate(self, messages):
            return "[]"

    reviewed, log = MODULE.self_review_semantic_relations(
        Generator(), {"paper_id": "1", "items": [valid_item()]}, "1", paper()
    )
    assert reviewed == {"paper_id": "1", "items": []}
    assert log["accepted_response"] is True


def test_validate_accepts_genuine_composition() -> None:
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [valid_item()]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not issues
    assert len(accepted) == 1
    assert accepted[0]["source_qa_ids"] == ["QA1", "QA2"]
    assert "evidence_pages" not in accepted[0]


def test_validate_normalizes_approved_reasoning_alias() -> None:
    item = valid_item()
    item["reasoning_type"] = "calculation"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not issues
    assert accepted[0]["reasoning_type"] == "multi_step_calculation"
    assert accepted[0]["reasoning_type_normalization"] == {
        "original": "calculation",
        "canonical": "multi_step_calculation",
        "rule": "approved_taxonomy_alias",
    }


def test_validate_normalizes_constraint_alias() -> None:
    item = valid_item()
    item["reasoning_type"] = "intersection_of_constraints"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not issues
    assert accepted[0]["reasoning_type"] == "constraint_intersection"


def test_validate_still_rejects_unapproved_reasoning_type() -> None:
    item = valid_item()
    item["reasoning_type"] = "attribute_binding"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not accepted
    assert any("invalid_reasoning_type" in issue["errors"] for issue in issues)


def test_alias_does_not_bypass_content_quality_gates() -> None:
    item = valid_item()
    item["reasoning_type"] = "calculation"
    item["answer"] = "Policy A; Lower prices"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not accepted
    assert any(
        "answer_is_source_answer_concatenation" in issue["errors"]
        for issue in issues
    )


def test_validate_rejects_answer_concatenation() -> None:
    item = valid_item()
    item["answer"] = "Policy A; Lower prices"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not accepted
    assert any(
        "answer_is_source_answer_concatenation" in issue["errors"]
        for issue in issues
    )


def test_validate_rejects_question_that_mentions_source_id() -> None:
    item = valid_item()
    item["question"] = "Using QA1 and QA2, what follows?"
    accepted, issues = MODULE.validate_response(
        {"paper_id": "1", "items": [item]},
        "1",
        paper(),
        2,
        4,
        set(),
    )
    assert not accepted
    assert any("question_mentions_qa_id" in issue["errors"] for issue in issues)


def test_build_dataset_stops_exactly_at_target() -> None:
    source = {"1": paper(), "2": {**paper(), "paper": "2"}}
    accepted = {
        "1": [valid_item(), {**valid_item(), "question": "Question two?"}],
        "2": [{**valid_item(), "question": "Question three?"}],
    }
    result = MODULE.build_dataset(source, accepted, 2, "Qwen3.6-27B")
    assert sum(len(record["QA"]) for record in result.values()) == 2
    assert list(result["1"]["QA"]) == ["RQA1", "RQA2"]


def test_build_dataset_stable_ids_do_not_depend_on_target() -> None:
    source = {"1": paper(), "2": {**paper(), "paper": "2"}}
    accepted = {
        "1": [valid_item(), {**valid_item(), "question": "Question two?"}],
        "2": [{**valid_item(), "question": "Question three?"}],
    }
    small = MODULE.build_dataset(
        source, accepted, 2, "Qwen3.6-27B", stable_qa_ids=True
    )
    large = MODULE.build_dataset(
        source, accepted, 3, "Qwen3.6-27B", stable_qa_ids=True
    )
    assert list(small["1"]["QA"]) == list(large["1"]["QA"])
    assert all(qa_id.startswith("RQA_") for qa_id in small["1"]["QA"])


def test_collect_existing_accumulates_generation_rounds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        first = {"paper_id": "1", "items": [valid_item()]}
        second_item = {**valid_item(), "question": "What follows for prices?"}
        second = {"paper_id": "1", "items": [second_item]}
        MODULE.atomic_json(
            raw_dir / "1.json",
            {"paper_id": "1", "parsed_response": first},
        )
        MODULE.atomic_json(
            raw_dir / "1__round1.json",
            {"paper_id": "1", "parsed_response": second},
        )
        accepted, issues = MODULE.collect_existing(
            raw_dir, {"1": paper()}, 2, 4, 4
        )
        assert not issues
        assert len(accepted["1"]) == 2


def test_collect_existing_can_use_pre_self_review_pool() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp)
        response = {"paper_id": "1", "items": [valid_item()]}
        MODULE.atomic_json(
            raw_dir / "1.json",
            {
                "paper_id": "1",
                "pre_self_review_response": response,
                "parsed_response": {"paper_id": "1", "items": []},
            },
        )
        accepted, issues = MODULE.collect_existing(
            raw_dir,
            {"1": paper()},
            2,
            4,
            4,
            local_self_review=False,
        )
        assert not issues
        assert len(accepted["1"]) == 1


def test_paper_lock_is_exclusive_and_recoverable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lock_dir = Path(tmp)
        first = MODULE.try_paper_lock(lock_dir, "paper-1")
        assert first is not None
        assert MODULE.try_paper_lock(lock_dir, "paper-1") is None
        MODULE.release_paper_lock(first)
        replacement = MODULE.try_paper_lock(lock_dir, "paper-1")
        assert replacement is not None
        MODULE.release_paper_lock(replacement)


def test_atomic_json_leaves_no_shared_temp_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "result.json"
        MODULE.atomic_json(target, {"ok": True})
        assert json.loads(target.read_text(encoding="utf-8")) == {
            "ok": True
        }
        assert not list(Path(tmp).glob("*.tmp"))


def test_mark_search_space_exhausted_publishes_actionable_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        MODULE.atomic_json(
            output_dir / "summary.json",
            {"status": "in_progress", "accepted_questions": 2457},
        )
        MODULE.mark_search_space_exhausted(output_dir, 2457, 3000, 7, 588)
        summary = json.loads(
            (output_dir / "summary.json").read_text(encoding="utf-8")
        )
        progress = json.loads(
            (output_dir / "progress.json").read_text(encoding="utf-8")
        )
        assert summary["status"] == "search_space_exhausted"
        assert summary["remaining_to_target"] == 543
        assert progress["exhausted_generation_rounds"] == 7
        assert progress["selected_papers"] == 588
