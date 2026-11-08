"""Receipt-only storage planning. Estimates are not allocation authorization."""
import argparse,json,math,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file
GIB=2**30

def components(utf8_bytes,documents,largest_rows,policy,scenario):
    for key in ['bytes_per_token','split_bytes_per_document','text_uncertainty_factor']:
        if not isinstance(scenario[key],(float,int)) or isinstance(scenario[key],bool) or not math.isfinite(scenario[key]) or scenario[key]<=0:raise ValueError('Invalid scenario '+key)
    for key in ['parameter_count','checkpoint_bytes_per_parameter','checkpoint_overhead_factor','keep_checkpoints','split_journal_factor','other_future_gib','reserve_gib']:
        if type(policy[key]) not in (int,float) or not math.isfinite(policy[key]) or policy[key]<0:raise ValueError('Invalid policy '+key)
    if type(policy['keep_checkpoints']) is not int or policy['parameter_count']<=0:raise ValueError('Invalid checkpoint policy')
    tokens=math.ceil(utf8_bytes*scenario['text_uncertainty_factor']/scenario['bytes_per_token'])+documents # one EOS/document
    token_bytes=tokens*2
    split=math.ceil(documents*scenario['split_bytes_per_document'])
    checkpoint=math.ceil(policy['parameter_count']*policy['checkpoint_bytes_per_parameter']*policy['checkpoint_overhead_factor'])
    return dict(tokens_uint16=token_bytes,token_staging=math.ceil(token_bytes*largest_rows/max(documents,1)),split_database=split,
        split_journal=math.ceil(split*policy['split_journal_factor']),checkpoint_peak=(policy['keep_checkpoints']+1)*checkpoint,
        other_future=math.ceil(policy['other_future_gib']*GIB))

def guard(required_bytes,root,reserve_bytes):
    if type(required_bytes) is not int or required_bytes<0 or reserve_bytes<0:raise ValueError('Invalid allocation size')
    free=shutil.disk_usage(root).free
    if required_bytes+reserve_bytes>free:raise RuntimeError('Insufficient free space including reserve')
    return free

def plan(audit,pilot,policy,mount):
    audit=Path(audit);pilot=Path(pilot);contract=json.loads((audit/'contract.json').read_text());pc=json.loads((pilot/'contract.json').read_text())
    if contract['definition_id']!=pc['definition_id']:raise ValueError('Audit and pilot source definitions differ')
    full={};sampling={};evidence={}
    for directory,target,cid in [(audit,full,json.loads((audit/'state.json').read_text())['contract_id']),(pilot,sampling,json.loads((pilot/'report.json').read_text())['contract_id'])]:
        for f in (directory/'files').glob('*.json'):
            d=json.loads(f.read_text())
            if d['status']!='passed' or d['contract_id']!=cid:raise ValueError('Invalid receipt contract')
            target[d['path']]=d;evidence[str(f)]=sha256_file(f)
    totals={}
    for f in contract['files']:
        s=totals.setdefault(f['source'],dict(expected_rows=0,observed_rows=0,observed_utf8_bytes=0,pilot_rows=0,pilot_utf8_bytes=0));s['expected_rows']+=f['rows']
        if f['path'] in full:
            r=full[f['path']]
            if r['rows']!=f['rows']:raise ValueError('Receipt row mismatch')
            s['observed_rows']+=r['rows'];s['observed_utf8_bytes']+=r['utf8_bytes']
        if f['path'] in sampling:r=sampling[f['path']];s['pilot_rows']+=r['rows'];s['pilot_utf8_bytes']+=r['utf8_bytes']
    for s in totals.values():
        count=s['observed_rows'] or s['pilot_rows'];size=s['observed_utf8_bytes'] if s['observed_rows'] else s['pilot_utf8_bytes']
        if not count:raise ValueError('No source-specific observations')
        s['estimated_total_utf8_bytes']=s['observed_utf8_bytes']+math.ceil((s['expected_rows']-s['observed_rows'])*size/count)
    text=sum(s['estimated_total_utf8_bytes'] for s in totals.values());rows=sum(s['expected_rows'] for s in totals.values());largest=max(f['rows'] for f in contract['files'])
    device=Path(mount).stat().st_dev
    for target in [Path(PROJECT_ROOT)/'data/tokens',Path(PROJECT_ROOT)/'runs']:
        while not target.exists():target=target.parent
        if target.stat().st_dev!=device:raise ValueError('Outputs span filesystems; prepare separate mount budgets')
    free=shutil.disk_usage(mount).free;results={}
    for name,scenario in policy['scenarios'].items():
        parts=components(text,rows,largest,policy,scenario);need=sum(parts.values());results[name]=dict(assumptions=scenario,components_bytes=parts,new_peak_bytes=need,reserve_bytes=int(policy['reserve_gib']*GIB),fits_assumed_budget=need+policy['reserve_gib']*GIB<=free,remaining_after_peak_and_reserve_bytes=free-need-int(policy['reserve_gib']*GIB))
    return dict(status='planning_estimate_only',free_bytes_at_measurement=free,mount=str(mount),mount_device=Path(mount).stat().st_dev,policy=policy,sources=totals,scenarios=results,evidence_sha256=evidence,
        scope='Existing raw/clean/current artifacts already occupy disk and are not added again. Future outputs assumed new and coexisting on this filesystem. Source audit is partial; tokenizer efficiency and split DB density are assumptions. Do not treat small synthetic DB as corpus scale proof.',next_gate='Recompute after full audit and actual tokenizer/split pilots; call --required-new-gib for a fresh concrete allocation check before a producer starts. Producer integration is not automatic.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--audit');ap.add_argument('--pilot');ap.add_argument('--policy');ap.add_argument('--output');ap.add_argument('--mount',default=PROJECT_ROOT);ap.add_argument('--required-new-gib',type=float);ap.add_argument('--reserve-gib',type=float,default=200);a=ap.parse_args()
    if a.required_new_gib is not None:
        if not math.isfinite(a.required_new_gib) or a.required_new_gib<0 or not math.isfinite(a.reserve_gib) or a.reserve_gib<0:ap.error('Invalid allocation/reserve')
        try:free=guard(math.ceil(a.required_new_gib*GIB),a.mount,math.ceil(a.reserve_gib*GIB));print('ALLOCATION_CHECK_PASSED',free)
        except RuntimeError as e:print(str(e));sys.exit(2)
    else:
        if not all([a.audit,a.pilot,a.policy,a.output]):ap.error('Planning requires audit, pilot, policy and output')
        out=Path(a.output)
        if out.exists():raise ValueError('Use new report path')
        r=plan(a.audit,a.pilot,json.loads(Path(a.policy).read_text()),a.mount);r['policy_sha256']=sha256_file(Path(a.policy));r['code_sha256']=sha256_file(Path(__file__));atomic_json(out,r);atomic_json(Path(str(out)+'.done'),dict(sha256=sha256_file(out)));print(json.dumps({k:dict(new_gib=round(v['new_peak_bytes']/GIB,2),fits=v['fits_assumed_budget']) for k,v in r['scenarios'].items()}))
