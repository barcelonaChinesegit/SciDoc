"""Hand-checked v4 semantics, source identity, cache/boundary and end-to-end parity."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import sxz_v4 as rules
from evaluation.prompts import build_prompt, PROMPT_PATH, PROMPT_HASH
from evaluation.runner import (cache_key, compare_summary, read_recorded_cache, required_decisions,
                               run, table2_metrics, table3_metrics, writable_output)
from evaluation.validation import ROOT


def test_kernel_is_bound_to_verbatim_source_segments():
    provenance = json.loads((PROMPT_PATH.parent / "provenance.json").read_text())
    text = Path(rules.__file__).read_text()
    lines = text.splitlines(keepends=True)
    nodes = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            nodes.update({target.id: node for target in node.targets if isinstance(target, ast.Name)})
    for name, binding in provenance["kernel_symbols"].items():
        node = nodes[name]
        segment = ''.join(lines[node.lineno - 1:node.end_lineno])
        assert hashlib.sha256(segment.encode()).hexdigest() == binding["sha256"], name
    assert PROMPT_PATH.read_text() == rules.JUDGE_RULES
    assert PROMPT_HASH == rules.JUDGE_RULES_SHA256


def test_full_prompt_not_just_rules_matches_original():
    prompt = build_prompt("What?", "42", "42.0", "reasoning_hard100")
    assert prompt == rules.build_judge_prompt("reasoning_hard100", "What?", "42", "42.0")
    assert "[Dataset]\nreasoning_hard100\n" in prompt
    assert "GOLD IS A REFERENCE, NOT A CHECKLIST" in prompt
    assert prompt.endswith("CORRECT\nPARTIAL\nWRONG\n")


@pytest.mark.parametrize("qa,answer,pages", [
    ({"answer_pre": '```json\n{"answer_pre":" A  B ","evidence_pages":["Page 2",3,3]}\n```'}, "A B", [2, 3]),
    ({"answer": "must not leak", "evidence_pages": [7]}, "", []),
    ({"answer_pre": "42", "evidence_pages_pre": [], "evidence_pages_pre_raw": [3]}, "42", [3]),
    ({"answer_pre": '{"answer_pre":"42","evidence_pages":[7]}', "evidence_pages_pre": []}, "42", []),
    ({"answer_pre": "  Unanswerable ", "evidence_pages_pre": [999]}, "Unanswerable", [999]),
])
def test_prediction_recovery_is_the_experiment_rule(qa, answer, pages):
    assert rules.get_pred_answer_from_qa(qa)[0] == answer
    assert rules.get_pred_pages_from_qa(qa)[0] == pages


def write_fixture(tmp_path):
    root = tmp_path / "results"
    directory = root / "Qwen3-VL-8B"
    directory.mkdir(parents=True)
    cache = {}
    for dataset in rules.FIXED_DENOMINATORS:
        records = {
            "1": {"question": "Q", "answer": "42", "evidence_pages": [2],
                  "answer_pre": "42", "evidence_pages_pre": [2]},
            "2": {"question": "Q", "answer": "42", "evidence_pages": [2],
                  "answer_pre": "partial answer", "evidence_pages_pre": [1, 2]},
            "3": {"question": "Q", "answer": "Unanswerable", "evidence_pages": [],
                  "answer_pre": "Unanswerable", "evidence_pages_pre": []},
            "4": {"question": "Q", "answer": "42", "evidence_pages": [2],
                  "answer_pre": "ERROR: OOM", "evidence_pages_pre": [2]},
        }
        for qa in list(records.values())[:3]:
            row = dict(dataset_id=dataset, question=qa["question"], gold_answer=qa["answer"],
                       pred_answer=qa["answer_pre"], answer_verdict="partial" if qa["answer_pre"] == "partial answer" else "correct",
                       judge_rules_sha256=rules.JUDGE_RULES_SHA256)
            row["judge_raw"] = row["answer_verdict"].upper()
            cache[cache_key(dataset, row["question"], row["gold_answer"], row["pred_answer"])] = row
        (directory / f"{dataset}.json").write_text(json.dumps({"paper": {"QA": records}}))
    path = tmp_path / "recorded.json"
    path.write_text(json.dumps(cache))
    return root, directory, path, cache


def test_end_to_end_offline_uses_embedded_gold_and_fixed_denominators(tmp_path):
    root, _, cache_path, _ = write_fixture(tmp_path)
    args = SimpleNamespace(output_dir=tmp_path / "out", results_dir=root, models=["Qwen3-VL-8B"],
                           judge_cache=cache_path, offline=True, subject_xlsx=None, reference_summary=None)
    report = run(args)
    assert report["execution"] == "recorded_cache_replay" and report["new_judge_calls"] == 0
    result = report["table2"]["Qwen3-VL-8B"]
    assert result["All"] == pytest.approx(100 * 10 / 2200)
    assert result["E-Precision"] == pytest.approx(100 * 17.5 / 2200)
    assert result["E-Recall"] == pytest.approx(100 * 20 / 2200)
    assert result["A-Pages"] == pytest.approx(20 / 2200)
    assert result["General"] == .1 and result["Unanswerable"] == .5
    assert not (args.output_dir / "judge_cache.json").exists()
    assert run(args)["table2"] == report["table2"]
    args.judge_cache.write_text('{}')
    with pytest.raises(ValueError, match="missing decisions"):
        run(args)


def test_original_script_and_portable_kernel_agree(tmp_path):
    source = ROOT / "sxz/evaluate_11models_5datasets_calibrated_fixeddenom_seed42_gpu2345_parallel_v4.py"
    if not source.exists():
        pytest.skip("Collaborator reference not distributed in clean clones; hash test remains mandatory")
    # Compile without importing the sxz path, so no __pycache__ can be generated there.
    original = {"__name__": "read_only_reference", "__file__": str(source)}
    exec(compile(source.read_text(), str(source), "exec"), original)
    _, directory, _, cache = write_fixture(tmp_path)
    for dataset in rules.FIXED_DENOMINATORS:
        args = ("Qwen3-VL-8B", dataset, directory / f"{dataset}.json", None, None, cache, tmp_path / "unused.json")
        with redirect_stdout(io.StringIO()):
            expected = original["evaluate_one_result"](*args)
            actual = rules.evaluate_one_result(*args)
        assert actual == expected


def test_recorded_cache_cannot_accept_changed_bindings_or_raw_verdicts(tmp_path):
    _, _, path, cache = write_fixture(tmp_path)
    assert read_recorded_cache(path) == cache
    key = next(iter(cache))
    cache[key]["pred_answer"] = "changed"
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="binding"):
        read_recorded_cache(path)
    cache[key]["pred_answer"] = "42"
    cache[key]["judge_raw"] = "WRONG"
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match="disagrees"):
        read_recorded_cache(path)


def test_missing_gold_evidence_is_not_empty_set_credit(tmp_path):
    _, directory, _, cache = write_fixture(tmp_path)
    path = directory / "ordinary1190.json"
    data = json.loads(path.read_text())
    del data["paper"]["QA"]["3"]["evidence_pages"]
    path.write_text(json.dumps(data))
    details, summary = rules.evaluate_one_result("m", "ordinary1190", path, None, None, cache, tmp_path / "unused")
    assert details[2]["E_F1"] == 0
    assert summary["Samples"] == 1200 and summary["Present Samples"] == 4


def test_subject_population_does_not_silently_import_outsiders():
    details = [dict(dataset="d", unit_id="p", qa_id="1", answer_correct=1),
               dict(dataset="d", unit_id="p", qa_id="2", answer_correct=1)]
    subjects = {("d", "p", "1"): {"discipline": "Physics"},
                ("d", "p", "3"): {"discipline": "Physics"}}
    result, audit = table3_metrics(details, subjects)
    assert result == {"All": 50, "Physics": 50}
    assert audit["correct_outside_subject_cohort"] == 1
    assert audit["missing_subject_predictions"] == [["d", "p", "3"]]


def test_protected_output_paths_include_symlink_alias(tmp_path):
    for path in (ROOT / "sxz/out.json", ROOT / "data/qa/out.json", ROOT / "models/out.json"):
        with pytest.raises(ValueError, match="Read-only"):
            writable_output(path)
    alias = tmp_path / "alias"
    alias.symlink_to(ROOT / "sxz", target_is_directory=True)
    with pytest.raises(ValueError, match="Read-only"):
        writable_output(alias / "out.json")
