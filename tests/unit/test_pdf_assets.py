from __future__ import annotations

import json
from collections import defaultdict

import pytest

from pku_qa.pdf_assets import PDF_ASSET_MANIFEST, PDF_ROOT, resolve_pdf_path


def test_flat_pdf_manifest_matches_directory_and_has_no_duplicate_hashes() -> None:
    manifest = json.loads(PDF_ASSET_MANIFEST.read_text(encoding="utf-8"))
    assets = manifest["assets"]

    assert manifest["layout"] == "flat"
    assert manifest["asset_count"] == len(assets) == 1717
    assert manifest["duplicate_copy_count"] == 229
    assert len({row["filename"] for row in assets}) == len(assets)
    assert len({row["sha256"] for row in assets}) == len(assets)
    assert {path.name for path in PDF_ROOT.glob("*.pdf")} == {
        row["filename"] for row in assets
    }
    assert not list(PDF_ROOT.glob("*/*.pdf"))


def test_legacy_pdf_ids_resolve_except_documented_collision() -> None:
    manifest = json.loads(PDF_ASSET_MANIFEST.read_text(encoding="utf-8"))
    aliases: dict[str, list[str]] = defaultdict(list)
    for asset in manifest["assets"]:
        for legacy_id in asset["legacy_ids"]:
            aliases[legacy_id].append(asset["filename"])

    collisions = {key: values for key, values in aliases.items() if len(values) > 1}
    assert collisions == {
        "hxb_0007": ["z_cross_hxb_0007.pdf", "z_cross_hxf_0016.pdf"]
    }
    assert resolve_pdf_path("1", [PDF_ROOT]).name == "paper_0001.pdf"
    assert resolve_pdf_path("xb_0024", [PDF_ROOT]).name == "z_cross_xb_0024.pdf"
    with pytest.raises(ValueError, match="Ambiguous legacy PDF ID"):
        resolve_pdf_path("hxb_0007", [PDF_ROOT])
    assert resolve_pdf_path(
        "hxb_0007",
        [PDF_ROOT / "qa_expansion/hard_cross_pdf_global_pilot_v4"],
    ).name == "z_cross_hxf_0016.pdf"


def test_pdf_manifest_orders_cross_pdf_assets_last() -> None:
    manifest = json.loads(PDF_ASSET_MANIFEST.read_text(encoding="utf-8"))
    rank = {"single_paper": 0, "source_paper": 1, "cross_pdf": 2}
    ranks = [rank[row["kind"]] for row in manifest["assets"]]
    assert ranks == sorted(ranks)
