#!/usr/bin/env python3
"""Rescore one complete historical baseline with the unchanged local paper Judge.

Retains all 2200 slots, raw-format failures and input mismatches. Batches only
execution; every decision uses the public Judge cache and strict label parser.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from evaluation import __version__
from evaluation.check_reproduction import answer_bounds, compare_bound
from evaluation.judge import Judge, object_hash
from evaluation.metrics import score
from evaluation.prompts import JUDGE_PROMPT, PROMPT_HASH
from evaluation.reproduce import MODELS, load_baseline, write_csv
from evaluation.validation import file_hash, load_gold
from scripts.smoke_local_evaluation import qa_hashes
from pku_qa.evaluation.eval_framework import atomic_write_json


def main() -> int:
    raise SystemExit(
        "This binary-paper diagnostic is retired after the sxz v4 scoring change. "
        "Saved reports remain historical evidence. Use python evaluation/evaluate.py --help "
        "for recorded-cache replay or a new local v4 Judge run."
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(MODELS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(ROOT / 'sxz') or out.is_relative_to(ROOT / 'data/qa'):
        parser.error('Read-only input directory cannot contain outputs')
    if args.batch_size < 1:
        parser.error('Batch size must be positive')
    out.mkdir(parents=True, exist_ok=True)
    before = qa_hashes()
    gold, metadata = load_gold()
    predictions, counts, details, sources = load_baseline(ROOT / 'data/results', gold, args.model)
    binding = {'model': args.model, 'source_hashes': sources, 'dataset_hashes': metadata['dataset_hashes'],
               'pdf_manifest_sha256': metadata['pdf_manifest_sha256'], 'judge_prompt_sha256': PROMPT_HASH,
               'code_hashes': {name: file_hash(ROOT / name) for name in [
                   'evaluation/reproduce.py', 'evaluation/metrics.py', 'evaluation/validation.py',
                   'evaluation/judge.py', 'scripts/rescore_baseline_local.py']}}
    manifest = out / 'input_binding.json'
    if manifest.exists() and json.loads(manifest.read_text()) != binding:
        raise ValueError('Input/code binding changed: use a fresh output directory')
    atomic_write_json(manifest, binding)
    paper = json.loads((ROOT / 'evaluation/paper_reference.json').read_text())
    if file_hash(ROOT / paper['source']) != paper['paper_sha256']:
        raise ValueError('Paper changed: recover the current reference tables')
    cache = out / 'judge_cache.jsonl'
    config_path = out / 'judge_config.json'
    started = time.monotonic()
    if args.offline:
        config = json.loads(config_path.read_text())
    else:
        uuid = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid',
                                       '--format=csv,noheader'], text=True).strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:
            raise ValueError('Expected one GPU UUID')
        os.environ.update(CUDA_VISIBLE_DEVICES=uuid, GPU_REQUIRE_NAME='A800', GPU_ALLOW_SHARED='0',
                          GPU_REPOSITORY_LOCK='1', GPU_COORDINATION_AUTO='1', GPU_WAIT_TIMEOUT_SECONDS='60')
        model_dir = ROOT / 'models/Qwen3.6-27B'
        print('Hashing Qwen3.6-27B; preparing all current-release slots', flush=True)
        artifacts = {p.name: file_hash(p) for p in sorted(model_dir.iterdir()) if p.is_file()}
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        generation = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=32)
        config = {'identity': {'model': 'Qwen/Qwen3.6-27B', 'checkpoint': 'sha256:' + object_hash(artifacts),
                              'revision': None, 'tokenizer_sha256': artifacts['tokenizer.json'],
                              'chat_template_sha256': artifacts['chat_template.jinja']},
                  'backend': 'local_transformers', 'generation': generation, 'seed': 42, 'max_attempts': 2,
                  'enable_thinking': False, 'batch_size': args.batch_size, 'padding_side': 'left',
                  'dtype': 'bfloat16', 'attention': 'sdpa', 'device_map': {'': 'cuda:0'},
                  'timeout_seconds': None, 'timeout_note': 'local synchronous token-bounded generation',
                  'purpose': 'new current-gold diagnostic; historical binary-Judge config unresolved',
                  'torch': torch.__version__, 'transformers': transformers.__version__, 'gpu_uuid': uuid,
                  'decode': {'skip_special_tokens': True, 'clean_up_tokenization_spaces': False}}
        with managed_gpu_reservation('ScienceDoc full-baseline semantic rescore', gpu_ids=[str(args.gpu)]):
            processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
            processor.tokenizer.padding_side = 'left'
            model = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True,
                trust_remote_code=False, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
                attn_implementation='sdpa').eval()
            if any(p.device.type != 'cuda' for p in model.parameters()):
                raise RuntimeError('CPU/disk offload forbidden')
            config['model_default_generation_config'] = model.generation_config.to_dict()
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError('Judge config changed: use a new output directory')
            atomic_write_json(config_path, config)
            atomic_write_json(out / 'checkpoint_artifacts.json', artifacts)
            first = defaultdict(deque)

            def batch(prompts):
                texts = [processor.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False,
                         add_generation_prompt=True, enable_thinking=False) for p in prompts]
                inputs = processor(text=texts, padding=True, return_tensors='pt').to('cuda:0')
                with torch.inference_mode():
                    tokens = model.generate(**inputs, **generation)[:, inputs.input_ids.shape[1]:]
                replies = processor.batch_decode(tokens, **config['decode'])
                with (out / 'raw_judge_calls.jsonl').open('a', encoding='utf-8') as f:
                    for prompt, raw, ids in zip(prompts, replies, tokens.tolist()):
                        f.write(json.dumps({'prompt_sha256': object_hash(prompt), 'raw': raw, 'token_ids': ids,
                                            'batch_size': len(prompts), 'padded_input_tokens': inputs.input_ids.shape[1]})+'\n')
                    f.flush()
                    os.fsync(f.fileno())
                return replies

            def generate(prompt):
                return first[prompt].popleft() if first[prompt] else batch([prompt])[0]

            judge = Judge(config, cache, generate)
            jobs = []
            cached = Judge(config, cache)
            for qid, p in predictions.items():
                if p.status == 'legal' and gold[qid].task != 'Unanswerable':
                    if cached.decide(gold[qid], p)['decision'] is None:
                        prompt = JUDGE_PROMPT.format(question=gold[qid].question, correct=gold[qid].answer,
                                                     model_answer=p.answer_pre)
                        jobs.append((qid, prompt))
            # Stable length ordering reduces padding; no input truncation.
            jobs.sort(key=lambda pair: (len(pair[1]), pair[0]))
            print(f'{len(jobs)} uncached semantic decisions; counts={dict(counts)}', flush=True)
            decisions = []
            for offset in range(0, len(jobs), args.batch_size):
                group = jobs[offset:offset+args.batch_size]
                prompts = [p for _, p in group]
                batch_started = time.monotonic()
                replies = batch(prompts)
                for prompt, reply in zip(prompts, replies):
                    first[prompt].append(reply)
                for qid, _ in group:
                    decision = judge.decide(gold[qid], predictions[qid])
                    decisions.append({'qa_id': qid, 'judge': decision})
                    if decision['decision'] is None:
                        # Persist the complete exhausted trace, not just successful cache entries.
                        with (out / 'judge_failures.jsonl').open('a', encoding='utf-8') as f:
                            f.write(json.dumps(decisions[-1])+'\n')
                print(f'Judged {min(offset+len(group),len(jobs))}/{len(jobs)}; batch {time.monotonic()-batch_started:.1f}s', flush=True)
            atomic_write_json(out / 'timing.json', {'seconds_including_checkpoint_hashing': time.monotonic()-started,
                              'new_judgments': len(jobs), 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(0)})
    report = score(gold, predictions, Judge(config, cache), detailed=True)
    report.update(metadata)
    report.update(evaluator_version=__version__, timestamp=datetime.now(timezone.utc).isoformat(),
                  scope='complete historical-baseline current-release diagnostic; no raw output repair',
                  judge_config_sha256=object_hash(config), prompt_sha256=PROMPT_HASH,
                  official_reproduction_verified=False, baseline_counts=dict(counts))
    atomic_write_json(out / ('offline_report.json' if args.offline else 'report.json'), report)
    table2, table3 = [], []
    name = MODELS[args.model]
    for key, reference in paper['table2'][name].items():
        if key in report['answer_metrics']:
            subset = report['rows'] if key == 'All' else [r for r in report['rows'] if r['task'] == key]
            table2.append(compare_bound(name, key, reference, answer_bounds(subset)))
        else:
            value = report['evidence_metrics'][key]
            table2.append({'model': name, 'metric': key, 'paper': reference, 'denominator': 2200,
                           'correct_known': None, 'pending_judge': None, 'minimum': None, 'maximum': None,
                           'exact': value, 'difference': None if value is None else value-reference,
                           'status': 'UNRESOLVED_ILLEGAL_EVIDENCE' if value is None else 'SCORED'})
    for key, reference in paper['table3'][name].items():
        subset = report['rows'] if key == 'All' else [r for r in report['rows'] if r['discipline'] == key]
        table3.append(compare_bound(name, key, reference, answer_bounds(subset)))
    write_csv(out / 'table2.csv', table2)
    write_csv(out / 'table3.csv', table3)
    if before != qa_hashes():
        raise RuntimeError('Immutable QA bytes changed')
    print(json.dumps({'answer': report['answer_metrics'], 'illegal': report['illegal'],
                      'missing': report['missing'], 'technical_failures': report['technical_failures']}, indent=2), flush=True)
    return 2 if any(r['status'] != 'DISPLAY_MATCH' for r in table2+table3) else 0


if __name__ == '__main__':
    raise SystemExit(main())
