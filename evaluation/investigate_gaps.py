#!/usr/bin/env python3
"""Attribute paper/current-release gaps without rescoring with historical decisions.

Historical flags are diagnostic counts only; never inserted in the official
semantic cache. Raw strings are classified but never repaired or normalized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.check_reproduction import answer_bounds, compare_bound
from evaluation.judge import Judge
from evaluation.metrics import score
from evaluation.reproduce import MODELS, load_baseline, write_csv
from evaluation.validation import ROOT, file_hash, load_gold


def raw_failure_category(prediction) -> str:
    if prediction.status != 'illegal':
        return prediction.status
    raw = prediction.audit.get('original_raw_output', '')
    errors = {e['type'] for e in prediction.errors}
    # Classification only: no extracted or stripped answer is ever scored.
    if 'invalid_json' in errors and raw.lstrip().startswith('```'):
        return 'markdown_fence'
    if 'invalid_json' in errors:
        return 'invalid_json_or_extra_text'
    for key in ('noncanonical_refusal', 'refusal_with_pages', 'answer_without_evidence',
                'page_out_of_range', 'page_nonpositive', 'wrong_fields', 'page_type', 'answer_type'):
        if key in errors:
            return key
    return 'other_contract_violation'


def historical_flags(path: Path) -> dict:
    import openpyxl
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    result = {}
    for sheet in workbook:
        rows = sheet.iter_rows(values_only=True)
        first, second = next(rows), next(rows)
        positions, current = {}, None
        for i, (group, field) in enumerate(zip(first, second)):
            if group in MODELS:
                current = group
            if current and field == 'Answer Correct':
                positions[current] = i
        unit, qa = second.index('Unit ID'), second.index('QA ID')
        for row in rows:
            key = (sheet.title, str(row[unit]), str(row[qa]))
            if key in result:
                raise ValueError(f'Duplicate workbook identity: {key}')
            result[key] = {m: row[pos] == 1 for m, pos in positions.items()}
    workbook.close()
    return result


def main() -> int:
    raise SystemExit(
        "This binary-paper diagnostic is retired after the sxz v4 scoring change. "
        "Saved reports remain historical evidence. Use python evaluation/evaluate.py --help "
        "for recorded-cache replay or a new local v4 Judge run."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT / 'sxz') or out.is_relative_to(ROOT / 'data/qa'):
        parser.error('Read-only input directory')
    out.mkdir(parents=True, exist_ok=True)
    gold, metadata = load_gold()
    paper = json.loads((ROOT / 'evaluation/paper_reference.json').read_text())
    if file_hash(ROOT / paper['source']) != paper['paper_sha256']:
        raise ValueError('Manuscript hash mismatch')
    workbook = ROOT / 'data/results/evaluations/detail_11models_5datasets.xlsx'
    flags = historical_flags(workbook)
    summary, attribution, witnesses, table2, table3 = [], [], [], [], []
    for directory, name in MODELS.items():
        predictions, counts, details, hashes = load_baseline(ROOT / 'data/results', gold, directory)
        report = score(gold, predictions, Judge({}))
        categories, historical_correct_by_status, sample_count = Counter(), Counter(), Counter()
        for qid, p in predictions.items():
            category = raw_failure_category(p)
            categories[category] += 1
            src = p.audit['historical_source']
            old = flags.get((Path(src['file']).stem, src['legacy_paper'], src['legacy_qa']), {}).get(directory)
            historical_correct_by_status[category] += old is True
            attribution.append({'model': name, 'qa_id': qid, 'task': gold[qid].task,
                'status': p.status, 'category': category, 'historical_correct': old,
                'reference_only_revision_rescored': src.get('reference_revision_rescorable', False),
                'differences': ';'.join(src['dataset_differences']), 'source': src['file'],
                'raw_sha256': p.audit.get('raw_output_sha256', '')})
            if category != 'legal' and sample_count[category] < 2:
                sample_count[category] += 1
                witnesses.append({'model': name, 'qa_id': qid, 'category': category, 'source': src,
                    'raw_prefix': p.audit.get('original_raw_output', '')[:700], 'errors': p.errors})
        all_bound = answer_bounds(report['rows'])
        reference = paper['table2'][name]['All']
        min_gap = max(0.0, reference-all_bound['maximum'], all_bound['minimum']-reference)
        summary.append({'model': name, 'paper_all': reference, 'strict_minimum': all_bound['minimum'],
                        'strict_maximum': all_bound['maximum'], 'minimum_gap_pp': min_gap,
                        'legal': report['legal'], 'illegal': report['illegal'], 'missing': report['missing'],
                        'unresolved_input_versions': counts['unresolved_input_version_mismatch'],
                        'reference_revisions_rescored': counts['reference_only_revision_rescored'],
                        'markdown_fences': categories['markdown_fence'],
                        'historically_correct_now_illegal': sum(v for k,v in historical_correct_by_status.items()
                            if k not in ('legal','technical_failure','missing')),
                        'historically_correct_unresolved_input': historical_correct_by_status['technical_failure']})
        for key, reference in paper['table2'][name].items():
            if key in report['answer_metrics']:
                subset = report['rows'] if key == 'All' else [r for r in report['rows'] if r['task'] == key]
                table2.append(compare_bound(name, key, reference, answer_bounds(subset)))
        for key, reference in paper['table3'][name].items():
            subset = report['rows'] if key == 'All' else [r for r in report['rows'] if r['discipline'] == key]
            table3.append(compare_bound(name, key, reference, answer_bounds(subset)))
    for filename, rows in [('model_gap_summary.csv',summary),('item_attribution.csv',attribution),
                           ('table2_bounds.csv',table2),('table3_bounds.csv',table3)]:
        write_csv(out/filename,rows)
    result = {'dataset_hashes': metadata['dataset_hashes'], 'historical_workbook_sha256':file_hash(workbook),
              'manuscript_sha256':paper['paper_sha256'], 'models':summary, 'witnesses':witnesses,
              'proven_table2_answer_mismatches':sum(r['status']=='PROVEN_MISMATCH'for r in table2),
              'proven_table3_mismatches':sum(r['status']=='PROVEN_MISMATCH'for r in table3),
              'interpretation':'Bounds use no guessed semantic decisions; historical correctness is used only for attribution, never as current Judge output.',
              'official_reproduction_verified':False}
    (out/'gap_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'models':summary,'proven_table2_answer_mismatches':result['proven_table2_answer_mismatches'],
                      'proven_table3_mismatches':result['proven_table3_mismatches']},ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
