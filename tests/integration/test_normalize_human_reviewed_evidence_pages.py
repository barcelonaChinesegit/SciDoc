import json
from pathlib import Path

from pypdf import PdfReader

from pku_qa.pdf_assets import resolve_pdf_path
from pku_qa.workflows.cleaning.normalize_human_reviewed_evidence_pages import (
    PAGE_MAPPINGS,
    UNRESOLVED,
    apply_mappings,
)


ROOT = Path(__file__).resolve().parents[2]


def test_all_resolved_mappings_fit_local_pdfs():
    dataset = json.loads(
        (ROOT / "data/qa/5.human_reviewed/rel__human_reviewed__authority__batch01__n983.json").read_text()
    )
    rows = apply_mappings(dataset)
    assert len(rows) == 22
    for row in rows:
        pdf = resolve_pdf_path(row["paper_id"], [ROOT / "data/pdfs"])
        assert min(row["physical_pdf_pages"]) >= 1
        assert max(row["physical_pdf_pages"]) <= len(PdfReader(str(pdf)).pages)


def test_only_known_pdf_mismatch_remains_unresolved():
    assert set(UNRESOLVED) == {("327", "QA6")}
