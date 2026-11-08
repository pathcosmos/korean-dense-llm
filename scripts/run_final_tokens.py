"""Final-tokenizer native whole-file pilot, reviewed allocation, then resumable full run.

Review is an evidence-bound engineering decision, not user permission. No training,
deletion, overwrite of uncertified outputs, or external uploads occur here.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file,artifact_path

GIB=2**30
DEPENDENCIES=['scripts/run_final_tokens.py','data/parallel_tokenize.py','data/verified_split_context.py',
 'data/local_split_reader.py','data/tokenize_shards.py','data/split_manifest.py','data/build_mix.py',
 'common/data_contracts.py','common/source_inventory.py','train/data.py','scripts/hardware_profile.py',
 'scripts/run_local_split.py','scripts/watch_split_finalization.py']


def progress(root,phase,**fields):
    atomic_json(root/'worker_state.json',dict(phase=phase,unix=time.time(),pid=os.getpid(),**fields))


def verify_tokenizer_run(root,sources_sha,split_sha):
    """Bind actual final output, checker and efficiency results to one completed run."""
    root=Path(root);proof=json.loads((root/'.done').read_text());c=json.loads((root/'contract.json').read_text())
    r=json.loads((root/'report.json').read_text())
    if (proof['report_sha256']!=sha256_file(root/'report.json') or
            proof['contract_sha256']!=sha256_file(root/'contract.json')):raise ValueError('Tokenizer completion proof differs')
    if r['status']!='passed':raise ValueError('Final tokenizer requires review')
    if c['sources_sha256']!=sources_sha or c['canonical_split_sha256']!=split_sha:raise ValueError('Tokenizer source/split differs')
    for name,field in [('tokenizer_check.json','tokenizer_check_sha256'),('efficiency.json','efficiency_sha256'),
                       ('train_sample/manifest.json','train_sample_manifest_sha256'),('quality_inputs/manifest.json','quality_manifest_sha256'),
                       ('distribution.json','distribution_sha256'),('distribution_review.json','distribution_review_sha256')]:
        if sha256_file(root/name)!=r[field]:raise ValueError('Tokenizer evidence changed: '+name)
    check=json.loads((root/'tokenizer_check.json').read_text());eff=json.loads((root/'efficiency.json').read_text())
    if check['status']!='passed' or eff['status']!='passed':raise ValueError('Final tokenizer checks did not pass')
    if sha256_file(resolve(r['tokenizer']))!=r['tokenizer_sha256']:raise ValueError('Final tokenizer changed')
    return r,c


def pilot_projection(plan,pilot,uncertainty=1.25):
    """Source-specific projection, explicitly not a whole-corpus token count proof."""
    if not math.isfinite(uncertainty) or uncertainty<1:raise ValueError('Invalid uncertainty')
    expected={r['path']:r for r in plan['selected_files']};observed={r['path']:r for r in pilot['files']}
    if len(observed)!=len(pilot['files']) or set(expected)!=set(observed):raise ValueError('Pilot file coverage differs')
    if pilot['status']!='pilot_complete' or pilot['shards']!=len(expected):raise ValueError('Require completed whole-file pilot')
    if pilot['planned_shards']!=sum(s['files'] for s in plan['sources'].values()):raise ValueError('Pilot full plan coverage differs')
    totals={}
    for path,r in observed.items():
        e=expected[path]
        if r['source']!=e['source'] or r['input_bytes']!=e['bytes'] or r['train_docs']+r['holdout_docs']!=e['rows']:
            raise ValueError('Pilot source/document coverage differs')
        if r['utf8_bytes'] is None:
            if not r['reused']:raise ValueError('Missing fresh pilot UTF8 measurement')
        elif r['utf8_bytes']!=e['utf8_bytes']:raise ValueError('Pilot UTF8 bytes differ from full audit')
        for key in ['tokens','holdout_tokens','train_docs','holdout_docs']:
            if type(r[key]) is not int or r[key]<0:raise ValueError('Invalid pilot counts')
        if r['tokens']<r['train_docs'] or r['holdout_tokens']<r['holdout_docs']:raise ValueError('Pilot tokens cannot cover EOS')
        s=totals.setdefault(e['source'],dict(utf8_bytes=0,tokens=0,output_bytes=0,documents=0))
        s['utf8_bytes']+=e['utf8_bytes'];s['tokens']+=r['tokens']+r['holdout_tokens']
        s['output_bytes']+=2*(r['tokens']+r['holdout_tokens']);s['documents']+=e['rows']
    if set(totals)!=set(plan['sources']):raise ValueError('Missing pilot category/source')
    for source,s in totals.items():
        if not s['utf8_bytes'] or not s['tokens']:raise ValueError('Empty source pilot')
        s['projected_tokens']=math.ceil(plan['sources'][source]['utf8_bytes']*s['tokens']/s['utf8_bytes'])
        s['projected_output_bytes']=2*s['projected_tokens']
        s['upper_planning_output_bytes']=math.ceil(s['projected_output_bytes']*uncertainty)
    upper=sum(s['upper_planning_output_bytes'] for s in totals.values())
    written=sum(s['output_bytes'] for s in totals.values())
    if pilot['tokens']*2!=written:raise ValueError('Pilot aggregate tokens differ')
    return dict(status='measured_pilot_projection',sources=totals,uncertainty_factor=uncertainty,
                projected_output_bytes=sum(s['projected_output_bytes'] for s in totals.values()),
                upper_planning_output_bytes=upper,pilot_output_bytes=written,
                required_remaining_bytes=max(0,upper-written),
                scope='Source-specific whole-file pilot extrapolation with explicit margin; not a deterministic size bound. Live reserve/budget guards remain required.')


def verify_execution_review(root,pilot_path,projection,free_bytes):
    root=Path(root);r=json.loads((root/'execution_review.json').read_text())
    if (r.get('decision')!='proceed' or r.get('pilot_report_sha256')!=sha256_file(pilot_path) or
            r.get('projection_sha256')!=sha256_file(root/'projection.json') or
            r.get('contract_sha256')!=sha256_file(root/'contract.json')):raise ValueError('Execution review evidence differs')
    if r.get('workers') not in (16,24) or r.get('rayon_threads') not in (2,3) or r['workers']*r['rayon_threads']!=48:
        raise ValueError('Execution resource candidate was not measured')
    if not isinstance(r.get('reason'),str) or not r['reason'].strip():raise ValueError('Execution choice requires evidence/reason')
    for k in ['durable_budget_gib','future_checkpoint_gib']:
        if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) or r[k]<=0:raise ValueError('Invalid execution budget')
    if r['durable_budget_gib']*GIB<projection['upper_planning_output_bytes']:raise ValueError('Output budget below pilot projection')
    required=projection['required_remaining_bytes']+(200+r['future_checkpoint_gib'])*GIB
    if required>free_bytes:raise RuntimeError('Insufficient storage including future checkpoint budget and reserve')
    return r


def token_weights(f5_contract):
    targets={'ko':.5,'en':.3,'code':.1,'math':.1};categories=f5_contract['categories'];old=f5_contract['weights']
    if set(categories)!=set(old) or set(categories.values())!=set(targets):raise ValueError('Token mix requires all four categories')
    if any(type(w) not in (int,float) or not math.isfinite(w) or w<=0 for w in old.values()):raise ValueError('Invalid source weight')
    return {s:targets[c]*old[s]/sum(old[t] for t in old if categories[t]==c) for s,c in categories.items()}


def final_counts(context,out,categories,total_documents,minimum_holdout_tokens=50000000):
    counts={};out=Path(out)
    if set(categories)!=set(context.records):raise ValueError('Full token source categories differ')
    for source,records in context.records.items():
        totals=dict(train_docs=0,holdout_docs=0,tokens=0,holdout_tokens=0)
        for record in records:
            r=json.loads((out/source/(Path(record['path']).stem+'.bin.done')).read_text())
            expected=context.selection.counts(resolve(record['path']))
            if r['train_docs']!=expected.get('train',0) or r['holdout_docs']!=expected.get('eval',0):raise ValueError('Token file document coverage differs')
            for k in totals:totals[k]+=r[k]
        if not totals['tokens'] or totals['tokens']<totals['train_docs'] or totals['holdout_tokens']<totals['holdout_docs']:
            raise ValueError('Source tokens cannot cover documents')
        counts[source]=totals
    if sum(v['train_docs']+v['holdout_docs'] for v in counts.values())!=total_documents:raise ValueError('Full token document coverage differs')
    hold={cat:sum(v['holdout_tokens'] for s,v in counts.items() if categories[s]==cat) for cat in ['ko','en','code','math']}
    if min(hold['ko'],hold['en'])<minimum_holdout_tokens:raise ValueError('Actual ko/en holdout tokens below target')
    return counts,hold


def verify_mix_review(root,categories):
    root=Path(root);r=json.loads((root/'mix_review.json').read_text())
    if (r.get('decision')!='proceed' or r.get('counts_sha256')!=sha256_file(root/'counts.json') or
            r.get('contract_sha256')!=sha256_file(root/'contract.json')):raise ValueError('Mix review evidence differs')
    weights=r.get('weights',{})
    if set(weights)!=set(categories) or any(type(w) not in (float,int) or not math.isfinite(w) or w<=0 for w in weights.values()):
        raise ValueError('Mix review requires positive source weights')
    if not math.isclose(sum(weights.values()),1,rel_tol=0,abs_tol=1e-10):raise ValueError('Mix review weights must sum to1')
    for cat,target in dict(ko=.5,en=.3,code=.1,math=.1).items():
        if not math.isclose(sum(w for s,w in weights.items() if categories[s]==cat),target,rel_tol=0,abs_tol=1e-10):
            raise ValueError('Mix category policy differs')
    if not isinstance(r.get('reason'),str) or not r['reason'].strip():raise ValueError('Mix choice requires actual-token evidence/reason')
    return weights


def worker(a):
    from scripts.watch_split_finalization import activity,verify_completion
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import size_tree
    from data.local_split_reader import prepare_reader
    from data.verified_split_context import VerifiedSplitContext
    from data.parallel_tokenize import run
    from data.build_mix import build_mix
    root=resolve(a.run_dir);c=json.loads((root/'contract.json').read_text());f5=resolve(a.tokenizer_run)
    plan=json.loads(resolve(a.plan).read_text());sm=resolve(a.sources);canonical=resolve(a.split);out=resolve(a.output)
    started=time.perf_counter()
    if sha256_file(resolve(a.plan))!=c['plan_sha256']:raise ValueError('Pilot plan changed')
    if sha256_file(sm)!=c['sources_sha256'] or sha256_file(canonical)!=c['split_sha256']:raise ValueError('Token inputs changed')
    while True:
        state=json.loads((f5/'state.json').read_text())
        if state['phase']=='complete':break
        if state['phase'] in ('error','paused'):raise RuntimeError('Final tokenizer '+state['phase'])
        pidfile=resolve('logs')/(f5.name+'.pid')
        if not pidfile.exists() or activity(int(pidfile.read_text())) is None:
            if json.loads((f5/'state.json').read_text())['phase']=='complete':continue
            raise RuntimeError('Final tokenizer controller missing')
        progress(root,'await_final_tokenizer');time.sleep(30)
    trained,tc=verify_tokenizer_run(f5,c['sources_sha256'],c['split_sha256'])
    if sha256_file(sm)!=c['sources_sha256'] or sha256_file(canonical)!=c['split_sha256']:raise ValueError('Token inputs changed while waiting')
    f4=resolve(a.source_run);f4_report=verify_completion(f4,tc['f4_contract_sha256'],tc['total_documents'])
    if sha256_file(f4/'report.json')!=trained['f4_report_sha256']:raise ValueError('F4 proof differs from tokenizer input')
    hw=inspect();atomic_json(root/'launch_hardware.json',hw)
    if hw['memory']['conservative_headroom_gib']<512:raise RuntimeError('Insufficient memory for384GiB token job plus headroom')
    progress(root,'prepare_reader')
    local_manifest=json.loads((f5/'reader/manifest.json').read_text())
    local=resolve(local_manifest['database']['path'])
    reader=prepare_reader(canonical,root/'reader',local_database=local if local.exists() else None,scratch=a.scratch)
    progress(root,'verify_local_context')
    with VerifiedSplitContext(sm,canonical,reader_manifest=reader,hash_workers=32,cache_mib=tc['reader_cache_mib']) as ctx:
        actual={r['path']:r for rs in ctx.records.values() for r in rs}
        for e in plan['selected_files']:
            if actual[e['path']]['sha256']!=e['sha256']:raise ValueError('Pilot source differs from verified audit')
        progress(root,'whole_file_pilot')
        pilot=run(sm,canonical,trained['tokenizer'],out,workers=6,batch_size=1024,arrow_threads=1,
                  selected_files=[e['path'] for e in plan['selected_files']],verified_context=ctx)
        pp=resolve(pilot['report_path']);projection=pilot_projection(plan,pilot);atomic_json(root/'projection.json',projection)
        while not (root/'execution_review.json').exists():
            progress(root,'await_execution_review',pilot_report=artifact_path(pp),projection_sha256=sha256_file(root/'projection.json'));time.sleep(30)
        review=verify_execution_review(root,pp,projection,shutil.disk_usage(out).free)
        # Rayon is initialized only inside spawned encoding workers; the parent does not load it.
        os.environ['RAYON_NUM_THREADS']=str(review['rayon_threads'])
        progress(root,'full_tokenization',workers=review['workers'],durable_budget_gib=review['durable_budget_gib'])
        full=run(sm,canonical,trained['tokenizer'],out,workers=review['workers'],batch_size=1024,arrow_threads=1,verified_context=ctx)
        progress(root,'mix_counts_holdout',durable_budget_gib=review['durable_budget_gib'])
        counts,hold=final_counts(ctx,out,tc['categories'],tc['total_documents'])
        atomic_json(root/'counts.json',dict(source_counts=counts,category_holdout_tokens=hold,tokenizer_sha256=trained['tokenizer_sha256'],
                    full_report_sha256=sha256_file(out/'report.json'),
                    proposed_weights=token_weights(tc),proposal_scope='Category targets50/30/10/10; within-Korean byte weights are a proposal only. Review actual token counts/exposures before adoption.'))
        while not (root/'mix_review.json').exists():
            progress(root,'await_mix_review',counts_sha256=sha256_file(root/'counts.json'),durable_budget_gib=review['durable_budget_gib']);time.sleep(30)
        weights=verify_mix_review(root,tc['categories']);mix=build_mix(weights,out,out/'mix.json',seq_len=4096,target_tokens=10485760000)
        weights={s:v['weight'] for s,v in mix['sources'].items()}
        for f,h in c['code'].items():
            if sha256_file(resolve(f))!=h:raise ValueError('Token producer changed: '+f)
        report=dict(status='passed',scope='Full native v3 tokenization/mix/counts and storage; actual-token GPU validation and rehearsal remain pending.',
            tokenizer_run_sha256=sha256_file(f5/'report.json'),tokenizer_sha256=trained['tokenizer_sha256'],
            split_contract_id=ctx.selection.report['contract_id'],pilot_report=artifact_path(pp),pilot_report_sha256=sha256_file(pp),
            projection_sha256=sha256_file(root/'projection.json'),execution_review_sha256=sha256_file(root/'execution_review.json'),
            counts_sha256=sha256_file(root/'counts.json'),mix_review_sha256=sha256_file(root/'mix_review.json'),
            full_report_sha256=sha256_file(out/'report.json'),mix=artifact_path(out/'mix.json'),mix_sha256=sha256_file(out/'mix.json'),
            source_counts=counts,category_holdout_tokens=hold,token_weights=weights,
            actual_token_bytes=2*sum(v['tokens']+v['holdout_tokens'] for v in counts.values()),
            actual_output_tree_bytes=size_tree(out),
            expected_exposures={str(target):{s:target*weights[s]/v['tokens'] for s,v in counts.items()} for target in [10485760000,119537664000]},
            elapsed_seconds=time.perf_counter()-started,full=full)
        atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
        progress(root,'complete')


def supervise(a,root):
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    if not math.isfinite(a.max_seconds) or a.max_seconds<=0:raise ValueError('Invalid time budget')
    hw=inspect()
    scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=True)
    plan=json.loads(resolve(a.plan).read_text())
    if plan['status']!='prepared_not_launched' or plan['sources_sha256']!=sha256_file(resolve(a.sources)):raise ValueError('Source pilot plan differs')
    settings=dict(sources=artifact_path(resolve(a.sources)),split=artifact_path(resolve(a.split)),plan=artifact_path(resolve(a.plan)),
                  output=artifact_path(resolve(a.output)),tokenizer_run=artifact_path(resolve(a.tokenizer_run)),source_run=artifact_path(resolve(a.source_run)))
    c=dict(version=1,hardware=hw,settings=settings,plan_sha256=sha256_file(resolve(a.plan)),sources_sha256=sha256_file(resolve(a.sources)),split_sha256=sha256_file(resolve(a.split)),
           max_seconds=a.max_seconds,max_rss_gib=384,local_budget_gib=192,initial_durable_budget_gib=96,reserve_gib=200,
           code={f:sha256_file(resolve(f)) for f in DEPENDENCIES},gates='Passed final tokenizer/F4 proof -> pilot -> evidence-bound execution review -> full tokens/mix. No model training.')
    cp=root/'contract.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        if {k:v for k,v in old.items() if k!='hardware'}!={k:v for k,v in c.items() if k!='hardware'}:
            raise ValueError('Token controller contract changed; preserve outputs and use a new run')
        c=old
    else:
        if {p.name for p in root.iterdir()}!={'controller.lock'}:raise ValueError('Uncontracted controller output')
        atomic_json(cp,c)
    atomic_json(root/f'hardware_attempt_{time.time_ns()}.json',hw)
    for f,h in c['code'].items():
        d=root/'producer_code'/f;d.parent.mkdir(parents=True,exist_ok=True)
        if not d.exists():shutil.copyfile(resolve(f),d)
        if sha256_file(d)!=h:raise ValueError('Dependency changed during snapshot')
    env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='8',TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch),SQLITE_TMPDIR=str(scratch))
    cmd=['taskset','-c',','.join(map(str,hw['cpu']['affinity'][:64])),sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--run-dir',str(root),
         '--tokenizer-run',a.tokenizer_run,'--source-run',a.source_run,'--sources',a.sources,'--split',a.split,'--plan',a.plan,'--output',a.output,'--scratch',str(scratch)]
    started=time.monotonic();peak=0;active_budget=96
    old_review=root/'execution_review.json'
    if old_review.exists():
        prior=json.loads(old_review.read_text())
        if (prior.get('contract_sha256')==sha256_file(cp) and type(prior.get('durable_budget_gib')) in (int,float)
                and math.isfinite(prior['durable_budget_gib']) and prior['durable_budget_gib']>0):active_budget=prior['durable_budget_gib']
    with (root/'console.log').open('a') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while proc.poll() is None:
            ws=root/'worker_state.json';w=json.loads(ws.read_text()) if ws.exists() else dict(phase='launch')
            rss=process_tree_rss(proc.pid);peak=max(peak,rss);active_budget=w.get('durable_budget_gib',active_budget)
            atomic_json(root/'state.json',dict(phase='running',case=w['phase'],worker_pid=proc.pid,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=peak,worker=w))
            if rss>384*GIB or time.monotonic()-started>a.max_seconds:raise RuntimeError('Token time/RSS budget')
            for directory,limit in [(scratch,192),(resolve(a.output),active_budget),(root,4)]:
                if directory.exists() and (shutil.disk_usage(directory).free<200*GIB or size_tree(directory)>limit*GIB):raise RuntimeError('Token disk budget/reserve')
            time.sleep(2)
        if proc.returncode:raise RuntimeError('Token worker failed with exit '+str(proc.returncode))
        p=json.loads((root/'.done').read_text())
        if p['report_sha256']!=sha256_file(root/'report.json') or p['contract_sha256']!=sha256_file(root/'contract.json'):raise ValueError('Token completion proof differs')
        atomic_json(root/'state.json',dict(phase='complete',elapsed_seconds=time.monotonic()-started,peak_rss_bytes=peak))
    except BaseException as exc:
        if proc.poll() is None:stop(proc)
        atomic_json(root/'state.json',dict(phase='error',error=str(exc),peak_rss_bytes=peak));raise


def main(a):
    root=resolve(a.run_dir);root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        supervise(a,root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--tokenizer-run',default='runs/f5_tokenizer_v1');p.add_argument('--source-run',default='runs/f4_full_v2')
    p.add_argument('--sources',default='data/datasets/final_v1.ready.json');p.add_argument('--split',default='data/datasets/final_v2.split/manifest.json')
    p.add_argument('--plan',default='runs/f6_preflight_v1/plan.json');p.add_argument('--output',default='data/tokens/final_v1')
    p.add_argument('--scratch');p.add_argument('--max-seconds',type=float,default=72*3600)
    a=p.parse_args();worker(a) if a.worker else main(a)
