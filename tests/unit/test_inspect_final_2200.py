import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pku_qa.workflows.operations import inspect_final_2200 as inspector


ROOT = Path(__file__).resolve().parents[2]


def test_fresh_clone_inspection_needs_no_site_packages_or_external_assets(tmp_path):
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    release = tmp_path / "data/qa/7.final_2200"
    release.mkdir(parents=True)
    for source in (ROOT / "data/qa/7.final_2200").glob("*.json"):
        shutil.copyfile(source, release / source.name)
    before = {p: p.read_bytes() for p in release.iterdir()}
    result = subprocess.run(
        [sys.executable, "-S", "-B", "-m", "pku_qa.workflows.operations.inspect_final_2200"],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
        check=True, capture_output=True, text=True,
    )
    summary = json.loads(result.stdout)
    assert summary["qa_count"] == 2200
    assert summary["evaluation_pdfs"] == 712
    assert summary["cross_pdf_source_document_counts"] == {"2": 621, "3": 179}
    assert summary["pdf_check"] == "not_requested"
    assert before == {p: p.read_bytes() for p in release.iterdir()}
    assert not (tmp_path / "data/pdfs").exists()
    assert not (tmp_path / "data/qa/1.base").exists()


def test_inspection_rejects_a_stale_manifest(tmp_path, monkeypatch):
    stale = tmp_path / "manifest.json"
    stale.write_text('{"qa_count": 2200}')
    monkeypatch.setattr(inspector, "MANIFEST_PATH", stale)
    with pytest.raises(ValueError, match="manifest does not match"):
        inspector.inspect_release()


def test_pdf_check_rejects_changed_asset_before_accepting_it(tmp_path, monkeypatch):
    asset_manifest = json.loads((ROOT / "data/pdf_assets_manifest.json").read_text())
    # Use the first actual release paper to make the corrupted PDF the first check.
    first = next(iter(json.loads((ROOT / "data/qa/7.final_2200/ordinary_qa.json").read_text())))
    asset = next(row for row in asset_manifest["assets"] if row["pdf_id"] == first)
    pdf_dir = tmp_path / "data/pdfs"
    pdf_dir.mkdir(parents=True)
    (tmp_path / "data/pdf_assets_manifest.json").write_text(json.dumps(asset_manifest))
    (pdf_dir / asset["filename"]).write_bytes(b"changed PDF")
    monkeypatch.setattr(inspector, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="PDF hash mismatch"):
        inspector.inspect_release(check_pdfs=True)
