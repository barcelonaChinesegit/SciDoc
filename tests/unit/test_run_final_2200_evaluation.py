from pathlib import Path

import pytest

from pku_qa.workflows.reporting.run_final_2200_evaluation import (
    COMPONENT_NAMES,
    build_commands,
)
from pku_qa.workflows.selection.sync_final_2200_manifest import build_manifest


def test_final_2200_commands_lock_all_four_manifest_inputs(tmp_path: Path) -> None:
    manifest = build_manifest()
    commands = build_commands(manifest, COMPONENT_NAMES, tmp_path, ["--dynamic-a800"])

    assert len(commands) == 4
    expected = {
        "ordinary": ("data/qa/7.final_2200/ordinary_qa.json", "1000"),
        "unanswerable": ("data/qa/7.final_2200/unanswerable_qa.json", "200"),
        "reasoning": ("data/qa/7.final_2200/reasoning_qa.json", "200"),
        "cross_pdf": ("data/qa/7.final_2200/cross_pdf_qa.json", "800"),
    }
    for name, command in zip(COMPONENT_NAMES, commands, strict=True):
        qa_index = command.index("--qa-json") + 1
        count_index = command.index("--expected-qa-count") + 1
        hash_index = command.index("--expected-qa-sha256") + 1
        output_index = command.index("--output-dir") + 1
        assert (command[qa_index], command[count_index]) == expected[name]
        assert len(command[hash_index]) == 64
        assert command[output_index] == str(tmp_path / name)
        assert command[-1] == "--dynamic-a800"


@pytest.mark.parametrize("option", ["--qa-json", "--input-mode=question_only"])
def test_final_2200_commands_reject_locked_overrides(
    tmp_path: Path, option: str
) -> None:
    with pytest.raises(ValueError, match="locks"):
        build_commands(build_manifest(), ["ordinary"], tmp_path, [option, "other"])
