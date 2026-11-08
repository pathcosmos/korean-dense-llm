"""Read-only consumption of certified native pilot shards for diagnostic GPU tests.

This never manufactures a full-source completion marker. Inputs are immutable
whole-file pilot outputs; training contracts retain the explicit diagnostic scope.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,artifact_path,identity,sha256_file,atomic_json
from data.tokenize_shards import GROUPED_TOKEN_FORMAT


def build_pilot_mix(token_root,pilot_report,output,categories,weights,*,seq_len=4096,tokenizer_run=None,require_production=True):
    token_root=resolve(token_root);pilot_path=resolve(pilot_report);output=resolve(output)
    if type(seq_len) is not int or seq_len<1:raise ValueError('Positive sequence length required')
    report=json.loads(pilot_path.read_text());proof=json.loads(Path(str(pilot_path)+'.done').read_text())
    cp=token_root/'contract.json';contract=json.loads(cp.read_text())
    if proof['report_sha256']!=sha256_file(pilot_path) or proof['contract_sha256']!=sha256_file(cp):raise ValueError('Pilot completion proof differs')
    if report['status']!='pilot_complete' or report['report_path']!=artifact_path(pilot_path):raise ValueError('Expected native pilot report')
    if require_production:
        if contract['experimental'] or tokenizer_run is None:raise ValueError('Actual-token validation requires final tokenizer evidence')
        from scripts.run_final_tokens import verify_tokenizer_run
        trained,tc=verify_tokenizer_run(resolve(tokenizer_run),contract['source_sha256'],contract['split_sha256'])
        if trained['tokenizer_sha256']!=contract['tokenizer_sha256'] or categories!=tc['categories']:raise ValueError('Pilot final tokenizer/categories differ')
        if set(categories.values())!={'ko','en','code','math'}:raise ValueError('Require all four actual input categories')
    for file,digest in contract['code'].items():
        if sha256_file(resolve(file))!=digest:raise ValueError('Pilot producer code differs')
    if set(categories)!=set(weights) or not set(categories.values())<={'ko','en','code','math'}:raise ValueError('Pilot weight/category coverage differs')
    if any(type(w) not in (float,int) or not math.isfinite(w) or w<=0 for w in weights.values()):raise ValueError('Positive finite pilot weights required')
    denominator=sum(weights.values())
    if not math.isfinite(denominator):raise ValueError('Invalid pilot weight sum')
    sources={};holdout=[];evidence={artifact_path(p):sha256_file(p) for p in [cp,pilot_path,Path(str(pilot_path)+'.done')]};logical=[];common=None;seen=set();plans={}
    for profile in report['files']:
        source=profile['source'];file=profile['path']
        if source not in categories or Path(source).name!=source or source in ('.','..') or file in seen:raise ValueError('Unexpected or duplicate pilot input')
        seen.add(file);directory=token_root/source;mp=directory/'manifest.json'
        if source not in plans:
            plan=json.loads(mp.read_text())
            if identity({k:v for k,v in plan.items() if k!='contract_id'})!=plan['contract_id'] or plan['source']!=source:raise ValueError('Pilot source plan identity differs')
            plans[source]=plan;evidence[artifact_path(mp)]=sha256_file(mp)
        plan=plans[source]
        matches=[inp for inp in plan['inputs'] if inp['path']==file]
        if len(matches)!=1 or profile['input_bytes']!=matches[0]['bytes']:raise ValueError('Pilot input missing/different from full source plan')
        current={k:plan[k] for k in ['tokenizer_sha256','eos_id','dtype','format']}
        if current['format']!=GROUPED_TOKEN_FORMAT or current['dtype']!='uint16' or current['tokenizer_sha256']!=contract['tokenizer_sha256']:
            raise ValueError('Pilot token layout/tokenizer differs')
        if common is not None and common!=current:raise ValueError('Mixed pilot token contracts')
        common=current;name=Path(file).stem+'.bin';marker=directory/(name+'.done');r=json.loads(marker.read_text())
        if r['contract_id']!=plan['contract_id'] or r['marker']!=marker.name:raise ValueError('Pilot shard contract differs')
        for key in ['tokens','holdout_tokens','train_docs','holdout_docs']:
            if type(r[key]) is not int or r[key]<0 or r[key]!=profile[key]:raise ValueError('Pilot counts differ from certified report')
        if r['tokens']<r['train_docs'] or r['holdout_tokens']<r['holdout_docs']:raise ValueError('Pilot cannot cover documents/EOS')
        expected=[name]+([Path(file).stem+'.hold.bin'] if r['holdout_docs'] else [])
        if [o['name'] for o in r['outputs']]!=expected or r['holdout_file']!=(expected[1] if r['holdout_docs'] else None):raise ValueError('Pilot output names differ')
        if not r['holdout_docs'] and r['holdout_tokens']:raise ValueError('Unexpected pilot eval tokens')
        for i,item in enumerate(r['outputs']):
            path=directory/item['name'];count=r['tokens' if i==0 else 'holdout_tokens']
            if item['bytes']!=count*2 or not path.is_file() or path.stat().st_size!=item['bytes'] or sha256_file(path)!=item['sha256']:
                raise ValueError('Pilot token output missing/corrupt')
            evidence[artifact_path(path)]=item['sha256']
        evidence[artifact_path(marker)]=sha256_file(marker)
        if r['tokens']<seq_len+1 or r['holdout_tokens']<seq_len+1:raise ValueError('Pilot train/eval input too short')
        s=sources.setdefault(source,dict(weight=weights[source]/denominator,files=[],tokens=0,category=categories[source],contract_id=plan['contract_id']))
        s['files'].append(artifact_path(directory/name));s['tokens']+=r['tokens'];holdout.append(artifact_path(directory/expected[1]))
        logical.append(dict(source=source,file=file,record=r))
    if not seen or len(seen)!=report['shards'] or set(sources)!=set(categories):raise ValueError('Incomplete diagnostic pilot source coverage')
    if report['planned_shards']!=sum(len(p['inputs']) for p in plans.values()):raise ValueError('Diagnostic full plan coverage differs')
    if identity(logical)!=report['logical_id']:raise ValueError('Pilot logical contents differ')
    if sum(p['tokens']+p['holdout_tokens'] for p in report['files'])!=report['tokens']:raise ValueError('Pilot aggregate tokens differ')
    from train.data import SAMPLER_VERSION
    mix=dict(schema_version=2,sampler_version=SAMPLER_VERSION,seq_len=seq_len,**common,sources=sources,holdout=holdout,
             scope='diagnostic_pilot_only',full_corpus_complete=False,production_tokenizer=require_production,
             native_contract_sha256=sha256_file(cp),pilot_report=artifact_path(pilot_path),pilot_report_sha256=sha256_file(pilot_path))
    output.mkdir(parents=True,exist_ok=False);atomic_json(output/'mix.json',mix)
    metadata=dict(status='passed_diagnostic_input',scope='Certified whole-file native pilot only. Not a complete corpus mix or publication evaluation.',
                  experimental_input=not require_production,mix_sha256=sha256_file(output/'mix.json'),native_contract_sha256=sha256_file(cp),
                  pilot_report_sha256=sha256_file(pilot_path),files=evidence,code_sha256=sha256_file(Path(__file__)))
    atomic_json(output/'report.json',metadata);atomic_json(output/'.done',dict(mix_sha256=metadata['mix_sha256'],report_sha256=sha256_file(output/'report.json')))
    return mix


def verify_pilot_mix(output,*,require_production=True):
    output=resolve(output);proof=json.loads((output/'.done').read_text());r=json.loads((output/'report.json').read_text());mix=json.loads((output/'mix.json').read_text())
    if proof['mix_sha256']!=sha256_file(output/'mix.json') or proof['report_sha256']!=sha256_file(output/'report.json') or r['mix_sha256']!=proof['mix_sha256']:
        raise ValueError('Diagnostic input proof differs')
    if r['status']!='passed_diagnostic_input' or mix['scope']!='diagnostic_pilot_only' or mix['full_corpus_complete'] is not False:
        raise ValueError('Diagnostic input scope differs')
    if require_production and (r['experimental_input'] or not mix['production_tokenizer']):raise ValueError('Experimental pilot cannot certify actual final-tokenizer training')
    if r['code_sha256']!=sha256_file(Path(__file__)):raise ValueError('Diagnostic reader implementation changed')
    for path,digest in r['files'].items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Diagnostic source evidence changed')
    if sha256_file(resolve(mix['pilot_report']))!=r['pilot_report_sha256']:raise ValueError('Diagnostic pilot report changed')
    return mix


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['token-root','pilot-report','output','policy']:p.add_argument('--'+name,required=True)
    p.add_argument('--tokenizer-run');p.add_argument('--seq-len',type=int,default=4096);a=p.parse_args();policy=json.loads(resolve(a.policy).read_text())
    build_pilot_mix(a.token_root,a.pilot_report,a.output,policy['categories'],policy['weights'],seq_len=a.seq_len,tokenizer_run=a.tokenizer_run)
