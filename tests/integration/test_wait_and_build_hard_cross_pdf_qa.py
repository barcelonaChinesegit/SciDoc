from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src/pku_qa/workflows/generation/wait_and_build_hard_cross_pdf_qa.py"
)
SPEC = importlib.util.spec_from_file_location("wait_and_build_hard_cross", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_worker_command_propagates_precision_gates(tmp_path: Path) -> None:
    args = argparse.Namespace(
        source=tmp_path / "source.json",
        prepared_contexts=tmp_path / "prepared.json",
        base_source=tmp_path / "base.json",
        review_metadata_dir=tmp_path / "reviews",
        output_dir=tmp_path,
        target=400,
        generation_rounds=3,
        candidates_per_bundle=5,
        max_accepted_per_bundle=10,
        max_bundles=0,
        max_new_tokens=2200,
        min_three_doc_ratio=0.4,
        source_policy="audited_evidence_facts",
        min_shared_terms=2,
        require_source_overlap=True,
        require_relation_template=True,
        local_self_review=False,
    )
    command = MODULE.build_worker_command(args, "3", 2)
    assert command[command.index("--min-shared-terms") + 1] == "2"
    assert "--require-source-overlap" in command
    assert "--require-relation-template" in command
    assert "--no-local-self-review" in command
    assert command[command.index("--worker-id") + 1] == "hard-cross-gpu3-attempt2"
    assert command[command.index("--source") + 1].endswith("source.json")
    assert command[command.index("--prepared-contexts") + 1].endswith(
        "prepared.json"
    )
