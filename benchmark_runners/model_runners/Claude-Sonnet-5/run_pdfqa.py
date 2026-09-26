#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, copy, io, json, os, sys, tempfile, time
from pathlib import Path
from typing import Any
import httpx, pypdfium2 as pdfium
from anthropic import Anthropic
from PIL import Image

REPO_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT)) if str(REPO_ROOT) not in sys.path else None
from settings.pdfqa_prompts import PROMPT_VERSION, PROMPTS

HERE=Path(__file__).resolve().parent
MODEL_NAME="Claude-Sonnet-5"; DEFAULT_MODEL="claude-sonnet-5"
DEFAULT_BASE_URL="https://api.aicodemirror.ai/api/claudecode"
PDF_ROOT=PDF_ROOT; RUN_VERSION="run-v1"

DATASETS={
 "ordinary":{"expected":1000,"data":Path(__file__).resolve().parents[3] / "data" / "qa" / "7.final_2200" / "ordinary_qa.json","prompt_key":"ordinary","prompt_id":"ordinary-v2-strict-min","max_tokens":1536,"policy":"ordinary","retries":3,"retry_base":5.0,"interval":0.0},
 "unanswerable":{"expected":200,"data":Path(__file__).resolve().parents[3] / "data" / "qa" / "7.final_2200" / "unanswerable_qa.json","prompt_key":"ordinary","prompt_id":"ordinary-v2-strict-min","max_tokens":1536,"policy":"ordinary","retries":3,"retry_base":5.0,"interval":0.0},
 "reasoning":{"expected":200,"data":Path(__file__).resolve().parents[3] / "data" / "qa" / "7.final_2200" / "reasoning_qa.json","prompt_key":"reasoning","prompt_id":"reasoning-v2-strict-min","max_tokens":2048,"policy":"profile","retries":4,"retry_base":2.0,"interval":0.2},
 "cross_pdf":{"expected":800,"data":Path(__file__).resolve().parents[3] / "data" / "qa" / "7.final_2200" / "cross_pdf_qa.json","prompt_key":"cross_document","prompt_id":"cross-document-v2-strict-min","max_tokens":4096,"policy":"profile","retries":4,"retry_base":2.0,"interval":0.2},
}
PROFILE=[(1584,68),(1440,68),(1440,62),(1280,60),(1120,56),(960,52),(896,48),(832,44)]
# Equivalent to the old ordinary adaptive path: lower Q to 42, then uniformly shrink long side by 0.85 down to 960.
ORDINARY_PROFILE=[(1584,68),(1584,60),(1584,52),(1584,44),(1584,42),(1346,42),(1144,42),(972,42),(960,42)]

class PayloadTooLarge(RuntimeError): pass

def system_prompt(ds): return PROMPTS[DATASETS[ds]["prompt_key"]][1]
def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))

def atomic_write(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=p.parent,prefix=p.name+".",suffix=".tmp",delete=False) as f:
        json.dump(obj,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno()); tmp=f.name
    os.replace(tmp,p)

def iter_qa(data):
    for uid,u in data.items():
        if not isinstance(u,dict) or not isinstance(u.get("QA"),dict): continue
        for qid,qa in u["QA"].items():
            if isinstance(qa,dict): yield str(uid),u,str(qid),qa

def merge_saved(src,saved):
    out=copy.deepcopy(src)
    for uid,u,qid,qa in iter_qa(out):
        ou=saved.get(uid,{}) if isinstance(saved,dict) else {}
        oqa=ou.get("QA",{}).get(qid,{}) if isinstance(ou,dict) else {}
        if isinstance(oqa,dict):
            for k,v in oqa.items():
                if k not in qa: qa[k]=copy.deepcopy(v)
    return out

def is_success(qa):
    return qa.get("status") in {"completed","normalized_completed"} and qa.get("parsed_valid") is True

def is_attempted(qa):
    return qa.get("status") in {"completed","normalized_completed","parse_failed"}

def pdf_hint(unit):
    keys=("pdf_path","pdf_path_resolved","merged_pdf_path","source_pdf_path","pdf_file","pdf_filename","file_name","filename")
    for k in keys:
        v=unit.get(k)
        if isinstance(v,str) and v.lower().strip().endswith(".pdf"): return v.strip()
    return None

def resolve_pdf(uid,unit):
    p=PDF_ROOT/f"{uid}.pdf"
    if p.is_file(): return p
    hint=pdf_hint(unit)
    if hint:
        for p in (Path(hint),PDF_ROOT/Path(hint).name):
            if p.is_file(): return p
    hits=list(PDF_ROOT.rglob(f"{uid}.pdf"))
    if len(hits)==1: return hits[0]
    if len(hits)>1: raise RuntimeError(f"ambiguous PDF for {uid}: {hits[:5]}")
    raise FileNotFoundError(f"PDF not found: {uid}")

def page_count(p):
    d=pdfium.PdfDocument(str(p))
    try: return len(d)
    finally: d.close()

def budgets(policy,n):
    if policy=="ordinary":
        initial=18.0 if n<=30 else 22.0 if n<=50 else 26.0
        vals=[initial,22.0,18.0,14.0]
        return [v for i,v in enumerate(vals) if 0<v<=initial and v not in vals[:i]]
    return [18.0,14.0] if n<=30 else [20.0,18.0,14.0] if n<=50 else [22.0,18.0,14.0]

def render_profile(pdf_path,n,dpi,long_side,q):
    d=pdfium.PdfDocument(str(pdf_path)); pages=[]; total=0
    try:
        for i in range(n):
            page=d[i]; bitmap=None; im=None
            try:
                bitmap=page.render(scale=dpi/72.0); im=bitmap.to_pil().convert("RGB")
                w,h=im.size; m=max(w,h)
                if m>long_side:
                    r=long_side/m; resized=im.resize((max(1,round(w*r)),max(1,round(h*r))),Image.Resampling.LANCZOS); im.close(); im=resized
                w,h=im.size; b=io.BytesIO(); im.save(b,format="JPEG",quality=q,optimize=False,progressive=False); raw=b.getvalue(); total+=len(raw)
                pages.append({"page_number":i+1,"width":w,"height":h,"binary_size":len(raw),"base64":base64.b64encode(raw).decode("ascii")})
            finally:
                try:
                    if im is not None: im.close()
                except: pass
                try:
                    if bitmap is not None: bitmap.close()
                except: pass
                try: page.close()
                except: pass
    finally: d.close()
    return {"pages":pages,"binary_payload_mb":total/1024**2,"base64_payload_mb_est":total*4/3/1024**2,"max_long_side":long_side,"jpeg_quality":q}

def compress(uid,pdf_path,n,dpi,policy,budget):
    profiles=ORDINARY_PROFILE if policy=="ordinary" else PROFILE; tried=[]
    for pi,(long_side,q) in enumerate(profiles):
        r=render_profile(pdf_path,n,dpi,long_side,q)
        tried.append({"profile_index":pi,"budget_mb":budget,"max_long_side":long_side,"jpeg_quality":q,"binary_payload_mb":round(r["binary_payload_mb"],4),"base64_payload_mb_est":round(r["base64_payload_mb_est"],4)})
        print(f"[Compress] {uid} budget={budget:.1f}MB profile={pi} long={long_side} q={q} binary={r['binary_payload_mb']:.3f}MB base64≈{r['base64_payload_mb_est']:.3f}MB",flush=True)
        if r["binary_payload_mb"]<=budget:
            r.update(selected_budget_mb=budget,selected_profile_index=pi,compression_trials=tried); return r
        r.clear()
    raise RuntimeError(f"{uid}: complete PDF cannot fit within {budget:.1f}MB")

def user_content(pages,question):
    c=[]
    for p in pages:
        n=p["page_number"]; c+=[{"type":"text","text":f"PDF_PAGE_{n}_START"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":p["base64"]}},{"type":"text","text":f"PDF_PAGE_{n}_END"}]
    c.append({"type":"text","text":f'Question:\n{question.strip()}\n\nReturn only one JSON object with keys "answer_pre" and "evidence_pages".'})
    return c

def extract_text(msg):
    return "\n".join(str(getattr(b,"text","")) for b in (getattr(msg,"content",[]) or []) if getattr(b,"type","")=="text" and getattr(b,"text","")).strip()

def usage_dict(u):
    if u is None:return {}
    if hasattr(u,"model_dump"):
        try:return u.model_dump()
        except:pass
    return {k:getattr(u,k) for k in ("input_tokens","output_tokens","cache_creation_input_tokens","cache_read_input_tokens") if hasattr(u,k)}

def parse_prediction(raw,page_count=None):
    original=str(raw or "").strip()
    if not original:
        return False,"",[],"empty"

    def validate(o):
        if not isinstance(o,dict):
            return False,"",[],"not_object"
        if set(o)!={"answer_pre","evidence_pages"}:
            return False,"",[],"wrong_keys"

        a=o["answer_pre"]
        pages=o["evidence_pages"]

        if not isinstance(a,str) or not a.strip():
            return False,"",[],"invalid_answer_pre"
        if not isinstance(pages,list):
            return False,"",[],"evidence_not_list"
        if not all(isinstance(x,int) and not isinstance(x,bool) for x in pages):
            return False,"",[],"evidence_not_integer_list"

        if page_count is not None:
            bad=[x for x in pages if x<1 or x>page_count]
            if bad:
                return False,"",[],"evidence_page_out_of_range:"+",".join(map(str,bad))

        if a.strip()=="Unanswerable" and pages:
            return False,"",[],"unanswerable_with_evidence"

        return True,a.strip(),pages,"ok"

    # 1. Entire model output is strict JSON.
    try:
        o=json.loads(original)
    except Exception:
        o=None

    if o is not None:
        ok,a,pages,reason=validate(o)
        if ok:
            return True,a,pages,"strict_ok"
        return False,"",[],reason

    # 2. Entire model output is one Markdown JSON fence.
    if original.startswith("```") and original.endswith("```"):
        ls=original.splitlines()
        if (
            len(ls)>=3
            and ls[0].strip().lower() in {"```","```json"}
            and ls[-1].strip()=="```"
        ):
            candidate="\n".join(ls[1:-1]).strip()
            try:
                o=json.loads(candidate)
            except Exception:
                o=None

            if o is not None:
                ok,a,pages,reason=validate(o)
                if ok:
                    return True,a,pages,"outer_markdown_fence_normalized"

    # 3. Deterministically extract exactly one already-valid JSON object.
    #    No regex, no backslash repair, no content modification.
    dec=json.JSONDecoder()
    candidates=[]

    for i,ch in enumerate(original):
        if ch!="{":
            continue

        try:
            o,n=dec.raw_decode(original[i:])
        except json.JSONDecodeError:
            continue

        ok,a,pages,reason=validate(o)
        if ok:
            candidates.append((i,i+n,a,pages))

    # de-duplicate exact spans
    unique=[]
    seen=set()
    for item in candidates:
        key=(item[0],item[1])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    if len(unique)==1:
        _,_,a,pages=unique[0]
        return True,a,pages,"single_json_object_extracted"

    if len(unique)>1:
        return False,"",[],f"ambiguous_multiple_json_objects:{len(unique)}"

    return False,"",[],"invalid_json:JSONDecodeError"

def payload_error(e):
    s=str(e).lower()
    return any(x in s for x in ("413","request too large","payload too large","request_too_large","content too large","maximum request","context length","context_length","too many tokens","message is too long"))

def call_once(client,args,ds,content):
    spec=DATASETS[ds]; kw={"model":args.model,"max_tokens":spec["max_tokens"],"system":system_prompt(ds),"messages":[{"role":"user","content":content}]}
    if args.disable_thinking:kw["thinking"]={"type":"disabled"}
    m=client.messages.create(**kw)
    return {"raw":extract_text(m),"usage":usage_dict(getattr(m,"usage",None)),"request_id":str(getattr(m,"id","")),"returned_model":str(getattr(m,"model","")),"stop_reason":str(getattr(m,"stop_reason",""))}

def call_retry(client,args,ds,content):
    spec=DATASETS[ds]; last=None
    for attempt in range(1,spec["retries"]+1):
        try:return call_once(client,args,ds,content),attempt
        except Exception as e:
            if payload_error(e):raise PayloadTooLarge(str(e)) from e
            last=e
            if attempt==spec["retries"]:break
            wait=spec["retry_base"]*2**(attempt-1); print(f"[API RETRY] {attempt}/{spec['retries']} {type(e).__name__}: {e} sleep={wait:.1f}s",flush=True); time.sleep(wait)
    raise last

def infer_one(args,client,ds,uid,qid,question,pdf_path,n,cache,start_idx):
    spec=DATASETS[ds]; bs=budgets(spec["policy"],n); history=[]; t=time.time()
    for bi in range(min(start_idx,len(bs)-1),len(bs)):
        budget=bs[bi]
        if budget not in cache:cache[budget]=compress(uid,pdf_path,n,args.pdf_dpi,spec["policy"],budget)
        comp=cache[budget]
        try:r,attempts=call_retry(client,args,ds,user_content(comp["pages"],question))
        except PayloadTooLarge as e:
            history.append({"budget_mb":budget,"error":str(e)}); print(f"[PAYLOAD FALLBACK] {ds}::{uid}::{qid} budget={budget} -> smaller",flush=True); continue
        ok,a,pages,reason=parse_prediction(r["raw"],n)
        status="normalized_completed" if ok and reason!="strict_ok" else "completed" if ok else "parse_failed"
        u=r["usage"]
        rec={"status":status,"parsed_valid":ok,"prediction_parse_status":"ok" if ok else reason,"prediction_parse_reason":reason,
             "answer_pre_raw":r["raw"],"answer_pre":a if ok else "","evidence_pages_pre":pages if ok else [],
             "model_name":MODEL_NAME,"model_id":args.model,"inference_backend":"anthropic_compatible_messages_api","dataset_id":ds,
             "prompt_id":spec["prompt_id"],"prompt_version":PROMPT_VERSION,"evidence_minimality_mode":"strict-minimum",
             "gold_answer_sent":False,"gold_evidence_pages_sent":False,"document_manifest_sent":False,"format_repair_used":False,"format_repair_raw_output":"",
             "reasoning_content_saved":False,"thinking_disabled":args.disable_thinking,"pdf_path_resolved":str(pdf_path),"pdf_total_pages":n,
             "pdf_pages_supplied":list(range(1,n+1)),"pdf_pages_supplied_count":n,"render_dpi":args.pdf_dpi,"render_format":"JPEG",
             "compression_policy":spec["policy"],"selected_budget_mb":budget,"selected_profile_index":comp["selected_profile_index"],
             "final_max_long_side":comp["max_long_side"],"final_jpeg_quality":comp["jpeg_quality"],"binary_payload_mb":round(comp["binary_payload_mb"],4),
             "base64_payload_mb_est":round(comp["base64_payload_mb_est"],4),"compression_trials":comp["compression_trials"],"payload_fallback_history":history,
             "max_output_tokens":spec["max_tokens"],"api_attempts":attempts,"api_request_id":r["request_id"],"api_returned_model":r["returned_model"],
             "api_stop_reason":r["stop_reason"],"usage":u,"input_tokens":int(u.get("input_tokens",0) or 0),"output_tokens":int(u.get("output_tokens",0) or 0),
             "elapsed_seconds":round(time.time()-t,4),"error_type":"","error_message":""}
        return rec,bi
    raise RuntimeError(f"all payload budgets rejected: {history}")

def error_record(args,ds,pdf,n,e,elapsed):
    s=DATASETS[ds]
    return {"status":"error","parsed_valid":False,"prediction_parse_status":"error","prediction_parse_reason":"","answer_pre_raw":"","answer_pre":"","evidence_pages_pre":[],
            "model_name":MODEL_NAME,"model_id":args.model,"inference_backend":"anthropic_compatible_messages_api","dataset_id":ds,"prompt_id":s["prompt_id"],
            "prompt_version":PROMPT_VERSION,"evidence_minimality_mode":"strict-minimum","gold_answer_sent":False,"gold_evidence_pages_sent":False,
            "document_manifest_sent":False,"format_repair_used":False,"format_repair_raw_output":"","reasoning_content_saved":False,"thinking_disabled":args.disable_thinking,
            "pdf_path_resolved":str(pdf) if pdf else "","pdf_total_pages":n,"pdf_pages_supplied":[],"pdf_pages_supplied_count":0,"render_dpi":args.pdf_dpi,
            "render_format":"JPEG","compression_policy":s["policy"],"max_output_tokens":s["max_tokens"],"elapsed_seconds":round(elapsed,4),
            "error_type":type(e).__name__,"error_message":str(e)}

def validate(ds):
    d=load_json(DATASETS[ds]["data"]); n=sum(1 for _ in iter_qa(d))
    if n!=DATASETS[ds]["expected"]:raise RuntimeError(f"{ds}: QA={n}, expected={DATASETS[ds]['expected']}")
    return d,n

def out_path(root,ds,dpi):return root/f"{ds}__{PROMPT_VERSION}__dpi{dpi}__{RUN_VERSION}.json"

def run_dataset(args,client,ds):
    src,total=validate(ds); spec=DATASETS[ds]; out=out_path(args.output_root,ds,args.pdf_dpi)
    result=merge_saved(src,load_json(out)) if out.is_file() else copy.deepcopy(src)
    pending=sum(1 for *_,qa in iter_qa(result) if args.force or not is_attempted(qa))
    print("="*100); print(f"Dataset={ds} QA={total} pending={pending} model={args.model} DPI={args.pdf_dpi} policy={spec['policy']} max_tokens={spec['max_tokens']}"); print(f"Data={spec['data']}\nPDF root={PDF_ROOT}\nOutput={out}\nThinking={'disabled' if args.disable_thinking else 'provider/model default'}\nWhole PDF=YES; gold answer/evidence=NEVER"); print("="*100)
    done=succ=pfail=err=0
    for uid,unit in list(result.items()):
        if not isinstance(unit,dict) or not isinstance(unit.get("QA"),dict):continue
        qmap=unit["QA"]; ids=[str(qid) for qid,qa in qmap.items() if isinstance(qa,dict) and (args.force or not is_attempted(qa))]
        if not ids:continue
        if args.max_new_qa>=0:
            remain=args.max_new_qa-done
            if remain<=0:break
            ids=ids[:remain]
        pdf=None;n=None
        try:pdf=resolve_pdf(str(uid),unit);n=page_count(pdf);print(f"\n[Paper] {uid} pages={n} pending={len(ids)} pdf={pdf}",flush=True)
        except Exception as e:
            for qid in ids:qmap[qid].update(error_record(args,ds,pdf,n,e,0));done+=1;err+=1;atomic_write(out,result)
            continue
        if args.dry_run:continue
        cache={}; preferred=0
        for qid in ids:
            qa=qmap[qid]; t=time.time()
            try:
                q=str(qa.get("question","")).strip()
                if not q:raise ValueError("empty question")
                rec,bi=infer_one(args,client,ds,str(uid),qid,q,pdf,n,cache,preferred);preferred=max(preferred,bi);qa.update(rec)
                if is_success(qa):succ+=1
                elif qa.get("status")=="parse_failed":pfail+=1
            except Exception as e:
                qa.update(error_record(args,ds,pdf,n,e,time.time()-t));err+=1;print(f"[QA ERROR] {ds}::{uid}::{qid} {type(e).__name__}: {e}",flush=True)
            atomic_write(out,result);done+=1
            print(f"[Done] {ds}::{uid}::{qid} status={qa.get('status')} parse={qa.get('prediction_parse_status')} evidence_n={len(qa.get('evidence_pages_pre',[]))} input_tokens={qa.get('input_tokens',0)} budget={qa.get('selected_budget_mb')} long={qa.get('final_max_long_side')} q={qa.get('final_jpeg_quality')} sec={qa.get('elapsed_seconds',0)}",flush=True)
            if spec["interval"]>0:time.sleep(spec["interval"])
            if args.max_new_qa>=0 and done>=args.max_new_qa:break
        if args.max_new_qa>=0 and done>=args.max_new_qa:break
    if not args.dry_run:atomic_write(out,result)
    attempted=sum(1 for *_,qa in iter_qa(result) if is_attempted(qa));success=sum(1 for *_,qa in iter_qa(result) if is_success(qa))
    print(f"\n[Dataset Finished] {ds} new={done} success_new={succ} parse_failed_new={pfail} error_new={err} all_success={success}/{total} all_attempted={attempted}/{total}")

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--datasets",nargs="+",choices=list(DATASETS),default=list(DATASETS));p.add_argument("--model",default=os.getenv("CLAUDE_MODEL",DEFAULT_MODEL));p.add_argument("--base-url",default=os.getenv("CLAUDE_BASE_URL",DEFAULT_BASE_URL))
    p.add_argument("--anthropic-api-key",default=None);p.add_argument("--output-root",type=Path,default=HERE/"output"/"raw_result");p.add_argument("--pdf-dpi",type=int,default=144);p.add_argument("--request-timeout",type=float,default=1800.0)
    p.add_argument("--max-new-qa",type=int,default=-1);p.add_argument("--force",action="store_true");p.add_argument("--dry-run",action="store_true");p.add_argument("--use-env-proxy",action="store_true");p.add_argument("--disable-thinking",action="store_true")
    return p.parse_args()

def main():
    args=parse_args();args.output_root.mkdir(parents=True,exist_ok=True)
    if args.dry_run:
        for ds in args.datasets:run_dataset(args,None,ds)
        return
    key=args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:raise RuntimeError("ANTHROPIC_API_KEY is not set")
    hc=httpx.Client(timeout=httpx.Timeout(args.request_timeout),trust_env=args.use_env_proxy)
    client=Anthropic(api_key=key,base_url=args.base_url,timeout=args.request_timeout,max_retries=0,http_client=hc)
    try:
        for ds in args.datasets:run_dataset(args,client,ds)
    finally:hc.close()

if __name__=="__main__":main()
