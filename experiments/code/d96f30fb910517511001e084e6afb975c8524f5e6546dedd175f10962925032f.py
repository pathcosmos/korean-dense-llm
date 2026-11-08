"""Content-bound fixed source probes, separate from publication full-coverage PPL."""
import copy
import json
import math
from pathlib import Path
from common.data_contracts import resolve,artifact_path,sha256_file,atomic_json,identity
from data.tokenize_shards import GROUPED_TOKEN_FORMAT


def select_probe(token_root,output,categories,*,seq_len=4096,windows_per_file=64,require_production=True):
    root=resolve(token_root);output=resolve(output)
    if type(seq_len) is not int or seq_len<1 or type(windows_per_file) is not int or windows_per_file<1:raise ValueError('Positive probe geometry required')
    if require_production and (seq_len!=4096 or windows_per_file!=64):raise ValueError('Production probe must match trainer4096/64 defaults')
    cp=root/'contract.json';rp=root/'report.json';contract=json.loads(cp.read_text());report=json.loads(rp.read_text());done=json.loads((root/'.done').read_text())
    if done!=dict(report_sha256=sha256_file(rp),contract_sha256=sha256_file(cp)):raise ValueError('Full token completion proof differs')
    if require_production and (contract['experimental'] or report['status']!='complete'):raise ValueError('Probe requires complete production tokens')
    if report['status'] not in ('complete','experimental_complete') or report['shards']!=report['planned_shards']:raise ValueError('Partial token output cannot certify training probe')
    for path,digest in contract['code'].items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Token producer implementation differs')
    evidence={artifact_path(p):sha256_file(p) for p in [cp,rp,root/'.done']};plans={};records={};logical=[];baseline=0
    for profile in report['files']:
        source=profile['source'];file=profile['path']
        if Path(source).name!=source or source in ('.','..') or (source,file) in records:raise ValueError('Invalid/duplicate probe source file')
        directory=root/source;mp=directory/'manifest.json'
        if source not in plans:
            plan=json.loads(mp.read_text());marker=json.loads((directory/'.done').read_text())
            if identity({k:v for k,v in plan.items() if k!='contract_id'})!=plan['contract_id'] or plan['source']!=source:raise ValueError('Source plan identity differs')
            if marker!=dict(contract_id=plan['contract_id'],shards=len(plan['inputs'])):raise ValueError('Source tokenization incomplete')
            if plan['format']!=GROUPED_TOKEN_FORMAT or plan['dtype']!='uint16' or plan['tokenizer_sha256']!=contract['tokenizer_sha256']:raise ValueError('Source token layout differs')
            plans[source]=plan;evidence[artifact_path(mp)]=sha256_file(mp);evidence[artifact_path(directory/'.done')]=sha256_file(directory/'.done')
        plan=plans[source]
        if sum(inp['path']==file for inp in plan['inputs'])!=1:raise ValueError('Native input missing from full source plan')
        marker=directory/(Path(file).stem+'.bin.done');r=json.loads(marker.read_text())
        if r['contract_id']!=plan['contract_id'] or r['marker']!=marker.name:raise ValueError('Native shard contract differs')
        if any(type(r[k]) is not int or r[k]<0 or r[k]!=profile[k] for k in ['tokens','holdout_tokens','train_docs','holdout_docs']):raise ValueError('Native source counts differ')
        expected=[Path(file).stem+'.bin']+([Path(file).stem+'.hold.bin'] if r['holdout_docs'] else [])
        if [o['name'] for o in r['outputs']]!=expected or r['holdout_file']!=(expected[1] if r['holdout_docs'] else None):raise ValueError('Native holdout names differ')
        if r['tokens']<r['train_docs'] or r['holdout_tokens']<r['holdout_docs'] or (not r['holdout_docs'] and r['holdout_tokens']):raise ValueError('Native tokens cannot cover documents')
        baseline+=min(windows_per_file,r['holdout_tokens']//(seq_len+1))
        records[source,file]=(r,marker);logical.append(dict(source=source,file=file,record=r))
    if identity(logical)!=report['logical_id'] or len(records)!=report['shards']:raise ValueError('Native full logical coverage differs')
    expected={(s,inp['path']) for s,p in plans.items() for inp in p['inputs']}
    if set(records)!=expected or set(plans)!=set(categories) or not set(categories.values())<={'ko','en','code','math'}:raise ValueError('Probe source/category coverage differs')
    if require_production and set(categories.values())!={'ko','en','code','math'}:raise ValueError('Production probe requires all categories')
    holdout=[];groups=[];offset=0
    for source in plans:
        candidates=[]
        for (s,file),(r,marker) in records.items():
            if s!=source:continue
            count=min(windows_per_file,r['holdout_tokens']//(seq_len+1))
            if count:candidates.append((count,r['holdout_tokens'],str(marker),r))
        # Prefer the smallest file that supplies64 windows. Otherwise combine the
        # few largest usable files; record any source-window count overshoot.
        candidates.sort(key=lambda x:(0,x[1],x[2]) if x[0]==windows_per_file else (1,-x[0],x[2]))
        start=offset;selected=[]
        for count,tokens,marker,r in candidates:
            path=root/source/r['holdout_file'];item=r['outputs'][1]
            if item['bytes']!=tokens*2 or not path.is_file() or path.stat().st_size!=item['bytes'] or sha256_file(path)!=item['sha256']:
                raise ValueError('Selected probe token content changed')
            evidence[artifact_path(path)]=item['sha256'];evidence[artifact_path(Path(marker))]=sha256_file(Path(marker))
            holdout.append(artifact_path(path));selected.append(artifact_path(path));offset+=count
            if offset-start>=windows_per_file:break
        if offset-start<windows_per_file:raise ValueError('Insufficient source probe windows:'+source)
        groups.append(dict(source=source,category=categories[source],start=start,count=offset-start,files=selected))
    result=dict(status='passed_probe_selection',production_input=require_production,seq_len=seq_len,windows_per_file=windows_per_file,
                tokenizer_sha256=contract['tokenizer_sha256'],holdout=holdout,groups=groups,total_windows=offset,target_tokens=offset*seq_len,
                baseline_all_file_windows=baseline,baseline_target_tokens=baseline*seq_len,evidence=evidence,
                code={f:sha256_file(resolve(f)) for f in ['train/probe.py','train/data.py','train/runtime.py']},
                scope=f'Fixed per-source training probes only, with{windows_per_file} nonoverlapping windows per file; tails/interwindow targets are excluded by HoldoutSet. Not uniform corpus sampling or full-coverage publication PPL.')
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'probe.json',result);atomic_json(output/'.done',dict(probe_sha256=sha256_file(output/'probe.json')))
    return result


def verify_probe(output,*,require_production=True):
    output=resolve(output);p=output/'probe.json';r=json.loads(p.read_text())
    if json.loads((output/'.done').read_text())!=dict(probe_sha256=sha256_file(p)):raise ValueError('Training probe proof differs')
    if r['status']!='passed_probe_selection' or (require_production and not r['production_input']):raise ValueError('Experimental probe cannot certify actual rehearsal')
    for path,digest in {**r['evidence'],**r['code']}.items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Training probe input/implementation changed')
    return r


def evaluate_sources(original,model,fwd,hold,device,probe):
    """Same FSDP-balanced evaluator per source; aggregate NLL by valid targets."""
    if hold is None or hold.wins is None or len(hold)!=probe['total_windows'] or hold.wins.shape[1]!=probe['seq_len']+1:raise ValueError('Training probe windows differ')
    total_nll=0.;total_tokens=0;forwards=0;by_source={};by_category={};offset=0
    for group in probe['groups']:
        source=group['source'];count=group['count']
        if type(count) is not int or count<1 or group['start']!=offset or source in by_source:raise ValueError('Invalid source probe coverage')
        subset=copy.copy(hold);subset.wins=hold.wins[offset:offset+count]
        if len(subset)!=count:raise ValueError('Incomplete source probe windows')
        r=original(model,fwd,subset,device)
        if r['valid_tokens']!=count*probe['seq_len'] or not math.isfinite(r['nll_sum']):raise ValueError('Invalid source probe targets/NLL')
        by_source[source]=r;category=by_category.setdefault(group['category'],dict(nll_sum=0.,valid_tokens=0))
        category['nll_sum']+=r['nll_sum'];category['valid_tokens']+=r['valid_tokens']
        total_nll+=r['nll_sum'];total_tokens+=r['valid_tokens'];forwards+=r['forwards_per_rank'];offset+=count
    if offset!=probe['total_windows'] or total_tokens<=0:raise ValueError('Incomplete source evaluation')
    for c in by_category.values():c['holdout_loss']=c['nll_sum']/c['valid_tokens']
    return dict(holdout_loss=total_nll/total_tokens,nll_sum=total_nll,valid_tokens=total_tokens,windows=offset,
                forwards_per_rank=forwards,by_source=by_source,by_category=by_category,probe_scope=probe['scope'])
