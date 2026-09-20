"""Lossless generation, strict ingestion and bounded crash-resume tests."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.validation import Gold, SubmissionError, apply_generation_audit, load_submission, validate_raw
from pku_qa.evaluation.run_inference import PDF_SYSTEM_PROMPT, generate_with_retries

RAW = ' \n{"answer_pre":"42","evidence_pages":[2]}\n'
ARGS = SimpleNamespace(max_qa_retries=3, max_new_tokens=512, require_evidence_pages=True, retry_backoff_seconds=0)


class Provider:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, messages, limit):
        self.calls.append((deepcopy(messages), limit))
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result


def generate(outputs, audit=None, checkpoint=None):
    audit = audit if audit is not None else {}
    provider = Provider(outputs)
    raw = generate_with_retries(provider, [{"role": "user", "content": "question"}], ARGS,
                                allowed_evidence_pages=[1, 2, 3], audit=audit, audit_checkpoint=checkpoint)
    return raw, audit, provider


def test_prompt_is_verbatim_paper_source():
    root = Path(__file__).resolve().parents[1]
    source = root / '论文/写作材料/图/附录/提示词方框_drawio/source_text/05_pdf_inference.txt'
    assert PDF_SYSTEM_PROMPT.encode('utf-8') == source.read_bytes()


def test_lossless_raw_retry_and_no_setting_changes(tmp_path):
    raw, audit, provider = generate(['```wrong```', RAW])
    assert raw == RAW
    assert audit['original_raw_output'] == '```wrong```'
    assert [a['status'] for a in audit['attempts']] == ['illegal', 'legal']
    assert [limit for _, limit in provider.calls] == [512, 512]
    assert provider.calls[1][0][-2]['content'][0]['text'] == '```wrong```'
    path = tmp_path / 'prediction.jsonl'
    path.write_text(json.dumps({'qa_id': 'QA0001', 'raw_model_output': raw, 'generation_audit': audit}))
    gold = Gold('QA0001', 'question', '42', (2,), 'General', 'Physics', 'Optics', 'paper', 3)
    p = load_submission(path, {'QA0001': gold})['QA0001']
    assert p.status == 'legal'
    assert p.audit['generation_audit'] == audit
    assert p.audit['retry_outputs'] == [RAW]


def test_technical_failure_is_retained():
    audit = {}
    with pytest.raises(RuntimeError) as exc:
        generate([ValueError('temporary')] * 3, audit)
    assert exc.value.generation_audit is audit
    p = apply_generation_audit(validate_raw('QA0001', '', 3), audit, 3)
    assert p.status == 'technical_failure'
    assert p.answer_pre is None
    assert len(audit['attempts']) == 3
    with pytest.raises(RuntimeError):
        generate([], audit)


def test_resume_after_checkpoint_interrupt_keeps_budget():
    audit = {}
    def stop(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        generate(['invalid'], audit, stop)
    first = deepcopy(audit['attempts'][0])
    raw, audit, provider = generate([RAW], audit)
    assert raw == RAW and len(provider.calls) == 1
    assert audit['attempts'][0] == first
    assert audit['attempts'][1]['correction_trigger'] == 'illegal'
    assert generate([], audit)[0] == RAW


def test_exhausted_illegal_resume_never_gets_extra_attempt():
    raw, audit, _ = generate(['invalid'] * 3)
    assert generate([], audit)[0] == raw
    assert len(audit['attempts']) == 3


@pytest.mark.parametrize('mutation', ['hash', 'bool_number', 'status', 'original', 'extra_after_legal'])
def test_tampered_attempts_fatal(mutation):
    raw, audit, _ = generate([RAW])
    if mutation == 'hash':
        audit['attempts'][0]['raw_output_sha256'] = 'bad'
    elif mutation == 'bool_number':
        audit['attempts'][0]['attempt'] = True
    elif mutation == 'status':
        audit['attempts'][0]['status'] = 'illegal'
    elif mutation == 'original':
        audit['original_raw_output'] = 'changed'
    else:
        audit['attempts'].append({**audit['attempts'][0], 'attempt': 2})
    with pytest.raises(SubmissionError):
        apply_generation_audit(validate_raw('QA0001', raw, 3), audit, 3)


def test_changed_retry_setting_rejected():
    _, audit, _ = generate([RAW])
    audit['attempts'][0]['max_new_tokens'] = 64
    with pytest.raises(ValueError, match='changed generation'):
        generate([], audit)


def test_question_only_resume_preserves_raw_text():
    args = SimpleNamespace(max_qa_retries=2, max_new_tokens=32, require_evidence_pages=False,
                           retry_backoff_seconds=0)
    audit = {}
    raw = generate_with_retries(Provider([' 42\n']), [], args, audit=audit)
    assert raw == ' 42\n'
    assert generate_with_retries(Provider([]), [], args, audit=audit) == raw


def test_resume_rejects_pages_outside_shown_subset():
    raw, audit, _ = generate([RAW])
    with pytest.raises(ValueError):
        generate_with_retries(Provider([]), [], ARGS, audit=audit, allowed_evidence_pages=[1, 3])
