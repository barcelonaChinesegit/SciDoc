#!/usr/bin/env python3
"""Check selected identical-input disagreements singly, plus positive/negative controls.

Diagnostic only: never changes baseline predictions, gold, cache or reported scores.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from evaluation.judge import Judge
from evaluation.prompts import PROMPT_HASH
from evaluation.validation import Prediction, file_hash, load_gold, validate_raw
from pku_qa.evaluation.eval_framework import atomic_write_json


def main():
    raise SystemExit(
        "This binary-paper diagnostic is retired after the sxz v4 scoring change. "
        "Saved reports remain historical evidence. Use python evaluation/evaluate.py --help "
        "for recorded-cache replay or a new local v4 Judge run."
    )
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attribution',type=Path,required=True)
    parser.add_argument('--rescore-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.is_relative_to(ROOT/'sxz') or out.is_relative_to(ROOT/'data/qa'):
        parser.error('Read-only output destination')
    out.mkdir(parents=True,exist_ok=False)
    gold,metadata=load_gold()
    attribution=json.loads(args.attribution.read_text())
    selected=[]
    for task in ('General','Reasoning','Multi-Document'):
        selected.extend(sorted((r for r in attribution['disagreements']if r['task']==task and r['identical_workbook_judge_inputs']),key=lambda r:r['qa_id'])[:4])
    config=json.loads((args.rescore_dir/'judge_config.json').read_text())
    artifacts=json.loads((args.rescore_dir/'checkpoint_artifacts.json').read_text())
    model_dir=ROOT/'models/Qwen3.6-27B'
    print('Verifying checkpoint identity before single-item checks',flush=True)
    if any(file_hash(model_dir/name)!=expected for name,expected in artifacts.items()):
        raise ValueError('Checkpoint changed')
    uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid,GPU_REQUIRE_NAME='A800',GPU_ALLOW_SHARED='0',GPU_REPOSITORY_LOCK='1',GPU_COORDINATION_AUTO='1',GPU_WAIT_TIMEOUT_SECONDS='60')
    import numpy as np
    import torch
    import transformers
    from transformers import AutoProcessor,AutoModelForImageTextToText
    from pku_qa.evaluation.gpu_reservation import managed_gpu_reservation
    if torch.__version__!=config['torch'] or transformers.__version__!=config['transformers']:
        raise ValueError('Runtime differs from baseline rescore')
    config={**config,'batch_size':1,'purpose':'paired single-item stability and controls; no score replacement','gpu_uuid':uuid}
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    rows=[]
    with managed_gpu_reservation('ScienceDoc Judge disagreement checks',gpu_ids=[str(args.gpu)]):
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        processor.tokenizer.padding_side='left'
        model=AutoModelForImageTextToText.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        if any(p.device.type!='cuda'for p in model.parameters()):raise RuntimeError('No offload allowed')
        if model.generation_config.to_dict()!=config['model_default_generation_config']:raise ValueError('Generation defaults changed')
        atomic_write_json(out/'judge_config.json',config)
        def generate(prompt):
            text=processor.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=processor(text=[text],padding=True,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():tokens=model.generate(**inputs,**config['generation'])[:,inputs.input_ids.shape[1]:]
            return processor.batch_decode(tokens,**config['decode'])[0]
        judge=Judge(config,out/'judge_cache.jsonl',generate)
        for item in selected:
            g=gold[item['qa_id']];p=Prediction(**item['current_prediction']);decision=judge.decide(g,p)
            rows.append({'qa_id':g.qa_id,'task':g.task,'kind':'paired_disagreement','batched':item['new_decision'],'single':decision['decision'],'stable':decision['decision']==item['new_decision'],'judge':decision})
            print(g.qa_id,item['new_decision'],'->',decision['decision'],flush=True)
        for task in ('General','Reasoning','Multi-Document'):
            item=next(r for r in selected if r['task']==task);g=gold[item['qa_id']]
            for kind,answer,expected in [('reference_control',g.answer,'CORRECT'),('unrelated_control','Purple elephants dance on the moon.','INCORRECT')]:
                p=validate_raw(g.qa_id,json.dumps({'answer_pre':answer,'evidence_pages':list(g.evidence_pages)}),g.page_count)
                decision=judge.decide(g,p)
                rows.append({'qa_id':g.qa_id,'task':g.task,'kind':kind,'expected':expected,'single':decision['decision'],'passed':decision['decision']==expected,'judge':decision})
                print(kind,g.qa_id,decision['decision'],flush=True)
    result={'selection':'first four identical-workbook-input old-correct/new-incorrect rows per task; not an unbiased error-rate estimate','rows':rows,'paired_count':len(selected),'paired_stable':sum(r.get('stable',False)for r in rows),'controls_passed':sum(r.get('passed',False)for r in rows),'controls':6,'dataset_hashes':metadata['dataset_hashes'],'prompt_sha256':PROMPT_HASH,'baseline_scores_modified':False}
    atomic_write_json(out/'summary.json',result)
    return 0 if result['paired_stable']==result['paired_count'] and result['controls_passed']==6 else 2


if __name__=='__main__':raise SystemExit(main())
