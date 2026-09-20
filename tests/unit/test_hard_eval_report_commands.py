from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from run_hard_eval import cross_pdf_report_cmd, reasoning_qa_report_cmd, run_judge_stage


def _args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "qa_json": "data/qa/example.json",
        "pdf_dir": "data/pdfs",
        "extra_pdf_dir": [],
        "expected_qa_count": 100,
        "expected_qa_sha256": "qa-sha",
        "expected_pdf_corpus_sha256": "pdf-sha",
        "baseline_report": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cross_pdf_report_runs_as_package_module() -> None:
    command = cross_pdf_report_cmd(_args(), Path("output"), preflight_only=True)

    assert command[1:3] == [
        "-m",
        "pku_qa.workflows.reporting.report_cross_pdf_eval",
    ]


def test_reasoning_report_runs_as_package_module() -> None:
    command = reasoning_qa_report_cmd(_args(), Path("output"), preflight_only=True)

    assert command[1:3] == [
        "-m",
        "pku_qa.workflows.reporting.report_reasoning_qa_eval",
    ]


def test_single_gpu_judge_does_not_merge_missing_shard_zero(monkeypatch, tmp_path) -> None:
    base = tmp_path / "judge_4B.json"
    calls = []
    monkeypatch.setattr("run_hard_eval.show_nvidia_smi", lambda *_: None)
    monkeypatch.setattr("run_hard_eval.validate_judge_gpus", lambda *_: None)
    monkeypatch.setattr(
        "run_hard_eval.judge_cmd",
        lambda args, model, gpu, count, shard, output: [model],
    )
    monkeypatch.setattr("run_hard_eval.subprocess.Popen", lambda command, env: command)
    monkeypatch.setattr("run_hard_eval.wait_all", lambda processes: base.write_text('{"ok":1}\n'))
    monkeypatch.setattr("run_hard_eval.merge_shards", lambda *args: calls.append(args))
    args = Namespace(judge_gpus=["2"], allow_non_a800_judge=False)
    run_judge_stage(args, tmp_path, {})
    assert not calls
    assert base.read_text() == '{"ok":1}\n'
