"""Reference revisions never change predictions or silently become input failures."""
import json
from dataclasses import replace

import pytest

from evaluation.reproduce import load_baseline, reference_revision_is_rescorable
from evaluation.validation import Gold

G = Gold('QA0001', 'What value?', '42', (2,), 'General', 'Physics', 'Optics', 'paper_0001', 3, '1', 'QA1')
ASSET = {'filename': 'paper_0001.pdf', 'legacy_paths': ['data/pdfs/papers/1.pdf'], 'sha256': 'frozen'}


def record():
    return {'question': G.question, 'answer': 'old reference', 'evidence_pages': [1],
            'gold_answer_sent': False, 'gold_evidence_pages_sent': False,
            'pdf_path_resolved': '/historical/project/data/pdfs/papers/1.pdf', 'pdf_total_pages': 3,
            'answer_pre_raw': '{"answer_pre":"42","evidence_pages":[2]}'}


@pytest.mark.parametrize('change', [
    {'question': 'Different question'}, {'gold_answer_sent': True}, {'gold_answer_sent': None},
    {'gold_evidence_pages_sent': 0}, {'pdf_path_resolved': '/other.pdf'},
    {'pdf_total_pages': 4}, {'pdf_total_pages': True}, {'pdf_sha256': 'different'},
    {'pdf_path_resolved': '../data/pdfs/papers/1.pdf'},
])
def test_unknown_or_changed_inputs_rejected(change):
    assert not reference_revision_is_rescorable({**record(), **change}, G, ASSET)


def test_reference_only_revision_allowed_without_modifying_gold(tmp_path):
    directory = tmp_path / 'Qwen3-VL-8B'
    directory.mkdir()
    source = directory / 'ordinary1190.json'
    q = record()
    source.write_text(json.dumps({'1': {'QA': {'QA1': q}}}))
    original = source.read_bytes()
    predictions, counts, _, _ = load_baseline(tmp_path, {'QA0001': G}, 'Qwen3-VL-8B',
                                              pdf_assets={'paper_0001': ASSET})
    p = predictions['QA0001']
    assert p.status == 'legal' and p.answer_pre == '42' and p.evidence_pages == [2]
    assert p.audit['historical_source']['reference_revision_rescorable'] is True
    assert counts['reference_only_revision_rescored'] == 1
    assert source.read_bytes() == original
    assert G.answer == '42' and G.evidence_pages == (2,)


def test_ref_revision_never_repairs_illegal_prediction(tmp_path):
    directory = tmp_path / 'Qwen3-VL-8B'
    directory.mkdir()
    q = {**record(), 'answer_pre_raw': '```json\n{"answer_pre":"42","evidence_pages":[2]}\n```'}
    (directory / 'ordinary1190.json').write_text(json.dumps({'1': {'QA': {'QA1': q}}}))
    predictions, _, _, _ = load_baseline(tmp_path, {'QA0001': G}, 'Qwen3-VL-8B', pdf_assets={'paper_0001': ASSET})
    assert predictions['QA0001'].status == 'illegal'


def test_unknown_asset_never_certifies_input():
    assert not reference_revision_is_rescorable(record(), G, None)


def test_current_pdf_path_and_matching_hash_are_accepted():
    q = {**record(), 'pdf_path_resolved': 'data/pdfs/paper_0001.pdf', 'pdf_sha256': 'frozen'}
    assert reference_revision_is_rescorable(q, G, ASSET)


def test_changed_question_is_retained_as_failure(tmp_path):
    directory = tmp_path / 'Qwen3-VL-8B'
    directory.mkdir()
    q = {**record(), 'question': 'Different question?'}
    (directory / 'ordinary1190.json').write_text(json.dumps({'1': {'QA': {'QA1': q}}}))
    predictions, counts, _, _ = load_baseline(tmp_path, {'QA0001': G}, 'Qwen3-VL-8B', pdf_assets={'paper_0001': ASSET})
    assert predictions['QA0001'].status == 'technical_failure'
    assert counts['unresolved_input_version_mismatch'] == 1
    assert predictions['QA0001'].audit['original_raw_output'] == q['answer_pre_raw']
