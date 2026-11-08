"""F4 read-only distribution staging -> gated final train-only tokenizer.
Fresh local DB/source validation and distribution can overlap F4 final verification;
sample extraction/training require both the completed F4 producer proof and an
explicit evidence-bound distribution review. One live context spans every step.
"""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file,identity,artifact_path

DEPENDENCIES=['scripts/run_final_tokenizer.py','data/local_split_reader.py','data/verified_split_context.py','data/split_distribution_native.py',
 'data/split_inputs.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py',
 'tokenizer/train_tokenizer.py','tokenizer/quality_inputs.py','tokenizer/check_tokenizer.py','scripts/watch_split_finalization.py',
 'scripts/hardware_profile.py','scripts/run_local_split.py']


def progress(root,phase,**fields):atomic_json(root/'worker_state.json',dict(phase=phase,unix=time.time(),pid=os.getpid(),**fields))


def wait_f4(root,source_run,expected_sha,total):
    from scripts.watch_split_finalization import verify_completion
    while True:
        state=json.loads((source_run/'state.json').read_text())
        if state['phase']=='complete':return verify_completion(source_run,expected_sha,total)
        if state['phase'] in ('error','paused'):raise RuntimeError('F4 producer '+state['phase']+': '+str(state.get('reason')))
        from scripts.watch_split_finalization import activity
        source_pid_file=resolve('logs')/(source_run.name+'.pid')
        if not source_pid_file.exists() or activity(int(source_pid_file.read_text())) is None:
            if json.loads((source_run/'state.json').read_text())['phase']=='complete':continue
            raise RuntimeError('F4 controller missing while awaiting final proof')
        progress(root,'await_f4_final_proof');time.sleep(30)


def wait_review(root,distribution):
    path=root/'distribution_review.json';expected=sha256_file(root/'distribution.json')
    while not path.exists():progress(root,'await_distribution_review',distribution_sha256=expected);time.sleep(30)
    r=json.loads(path.read_text())
    if r.get('distribution_sha256')!=expected or r.get('decision')!='proceed':raise ValueError('Distribution review does not authorize this input')
    return r


def byte_budgets(weights,total=3000000000):
    budgets={s:int(w*total) for s,w in weights.items()};left=total-sum(budgets.values())
    for source in sorted(weights,key=lambda s:(-(total*weights[s]-budgets[s]),s))[:left]:budgets[source]+=1
    if sum(budgets.values())!=total or any(n<1 for n in budgets.values()):raise ValueError('Invalid byte budget')
    return budgets


def efficiency(final,baseline,weights,policy):
    ratios={};fw=bw=0.
    for c in ['ko','en','code','math']:
        f=final['categories'][c];b=baseline['categories'][c]
        if f['utf8_bytes']!=b['utf8_bytes'] or f['documents']!=b['documents'] or not b['tokens']:raise ValueError('Efficiency comparison inputs differ')
        ratios[c]=f['tokens']/b['tokens'];fw+=weights[c]*f['tokens']/f['utf8_bytes'];bw+=weights[c]*b['tokens']/b['utf8_bytes']
    ratio=fw/bw
    return dict(status='passed' if max(ratios.values())<=policy['maximum_category_token_ratio'] and ratio<=policy['maximum_weighted_token_ratio'] else 'needs_review',
                category_token_ratios=ratios,weighted_tokens_per_byte_ratio=ratio,policy=policy,
                scope='Source-balanced independent eval documents, not corpus-wide tokenizer quality or model capability.')


def worker(a):
    from data.local_split_reader import prepare_reader
    from data.verified_split_context import VerifiedSplitContext
    from data.split_distribution_native import distribution
    from data.split_inputs import SplitInputs,prepare
    from tokenizer.quality_inputs import baseline_ids,exclusion_groups,prepare_checks
    from tokenizer.train_tokenizer import train_prepared
    from tokenizer.check_tokenizer import check
    root=resolve(a.run_dir);contract=json.loads((root/'contract.json').read_text());source_run=resolve(a.source_run)
    canonical=resolve(a.split);sm=resolve(a.sources);started=time.perf_counter()
    progress(root,'prepare_local_reader')
    source_plan=json.loads((source_run/'plan.json').read_text());builder=next(Path(source_plan['scratch']).glob('split-*/build.json'))
    existing=builder.parent/'split.sqlite'
    reader=prepare_reader(canonical,root/'reader',local_database=existing if existing.exists() else None,scratch=a.scratch)
    progress(root,'verify_local_context')
    with VerifiedSplitContext(sm,canonical,hash_workers=a.hash_workers,cache_mib=contract['reader_cache_mib'],reader_manifest=reader) as ctx:
        progress(root,'distribution')
        dist=distribution(ctx,sort_threads=8);atomic_json(root/'distribution.json',dist)
        atomic_json(root/'distribution.done',dict(sha256=sha256_file(root/'distribution.json')))
        f4=wait_f4(root,source_run,contract['f4_contract_sha256'],contract['total_documents'])
        review=wait_review(root,dist)
        inputs=SplitInputs(sm,canonical,verified_context=ctx)
        progress(root,'prepare_train_only_3gb')
        sample=prepare(sm,canonical,root/'train_sample',contract['byte_budgets'],mode='train',seed=0,max_chars=20000,verified_inputs=inputs)
        progress(root,'prepare_independent_checks')
        ids=root/'baseline_identities.jsonl'
        baseline_ids(contract['baseline']['frame'],contract['baseline']['frame_sha256'],ids)
        excluded=exclusion_groups(inputs,ids)
        checks=prepare_checks(inputs,root/'quality_inputs',contract['categories'],per_source=a.quality_docs,excluded_groups=excluded)
        progress(root,'train_raw_bpe_65536')
        trained=train_prepared(root/'train_sample/manifest.json',a.tokenizer,65536,'raw',verified_inputs=inputs)
        progress(root,'check_raw_special_ids_efficiency')
        texts=resolve(checks['samples']['path'])
        final=check(resolve(a.tokenizer),texts,root/'tokenizer_check.json',65536,'raw')
        baseline=check(resolve(contract['baseline']['tokenizer']),texts,root/'baseline_check.json',65536,'raw')
        if final['status']!='passed' or baseline['status']!='passed':raise ValueError('Raw/special/ID checker failed')
        measured=efficiency(final,baseline,contract['category_weights'],contract['efficiency_policy']);atomic_json(root/'efficiency.json',measured)
        ctx.assert_control_files()
        for f,h in contract['code'].items():
            if sha256_file(resolve(f))!=h:raise ValueError('F5 dependency changed: '+f)
        report=dict(status='passed' if measured['status']=='passed' else 'needs_review',scope='Final train-only3GB raw65536 tokenizer plus independent eval checks; full tokenization pending.',
            f4_report_sha256=sha256_file(source_run/'report.json'),distribution_sha256=sha256_file(root/'distribution.json'),distribution_review_sha256=sha256_file(root/'distribution_review.json'),
            train_sample_manifest_sha256=sha256_file(root/'train_sample/manifest.json'),train_sample_stats=sample['stats'],
            quality_manifest_sha256=sha256_file(root/'quality_inputs/manifest.json'),quality_stats=checks['stats'],
            tokenizer=artifact_path(resolve(a.tokenizer)),tokenizer_sha256=trained['tokenizer_sha256'],
            tokenizer_check_sha256=sha256_file(root/'tokenizer_check.json'),efficiency_sha256=sha256_file(root/'efficiency.json'),elapsed_seconds=time.perf_counter()-started,
            next='Actual whole-file tokenization pilot with this final tokenizer, then token counts/storage/mix validation.')
        atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
        progress(root,'complete')


def main(a):
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    root=resolve(a.run_dir);root.mkdir(parents=True,exist_ok=False);hw=inspect();scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=False)
    if hw['memory']['conservative_headroom_gib']<512:raise RuntimeError('Need concurrent-main future peak plus256GiB staging budget')
    canonical=resolve(a.split);stored=json.loads(canonical.read_text());db=resolve(stored['database']['path']);total=sum(f['rows'] for f in stored['contract']['files'])
    sm=json.loads(resolve(a.sources).read_text());policy_path=resolve('runs/f2_execution_v1/final_policy.json')
    if sha256_file(policy_path)!=sm['review']['policy_sha256']:raise ValueError('Approved F2 policy differs')
    decision=json.loads(policy_path.read_text())['checks']['tokenizer_byte_mix']
    if decision['status']!='passed' or decision['decision']['normalization']!='raw':raise ValueError('Need approved raw byte policy')
    weights=decision['decision']['weights']
    if set(weights)!=set(sm['sources']) or not math.isclose(sum(weights.values()),1):raise ValueError('Tokenizer source mix differs')
    baseline_root=resolve('runs/tokenizer_resources_v1');bp=json.loads((baseline_root/'.done').read_text())
    if sha256_file(baseline_root/'report.json')!=bp['report_sha256']:raise ValueError('Baseline report changed')
    br=json.loads((baseline_root/'report.json').read_text());bpe=json.loads((baseline_root/'bpe_16/result.json').read_text());bt=baseline_root/'bpe_16/experimental_tokenizer.json'
    if sha256_file(bt)!=bpe['tokenizer_sha256']:raise ValueError('Baseline tokenizer changed')
    categories={'wiki_ko':'ko','fineweb2_ko':'ko','hplt_ko':'ko','fineweb_edu':'en','github_code_v1':'code','owm':'math'}
    contract=dict(version=1,hardware=hw,code={f:sha256_file(resolve(f)) for f in DEPENDENCIES},sources_sha256=sha256_file(resolve(a.sources)),canonical_split_sha256=sha256_file(canonical),
        f4_contract_sha256=sha256_file(resolve(a.source_run)/'contract.json'),total_documents=total,byte_budgets=byte_budgets(weights),weights=weights,categories=categories,
        category_weights={c:sum(w for s,w in weights.items() if categories[s]==c) for c in set(categories.values())},
        reader_cache_mib=min(196608,max(8192,math.ceil(db.stat().st_size/1024**2*1.2))),
        memory_gib_before_f4_complete=256,memory_gib_after_f4_complete=384,max_seconds=a.max_seconds,local_budget_gib=192,durable_budget_gib=96,reserve_gib=200,
        baseline=dict(frame=artifact_path(baseline_root/'prepare/frame.json'),frame_sha256=br['frame_sha256'],tokenizer=artifact_path(bt),tokenizer_sha256=bpe['tokenizer_sha256']),
        efficiency_policy=dict(maximum_category_token_ratio=1.10,maximum_weighted_token_ratio=1.02,
            rationale='Predeclared engineering guardrails versus content-bound experimental64k baseline on eval groups excluded from both training sets. Not a model-quality guarantee; exceeding requires explicit investigation.'),
        gates='Only F4 read-only distribution stages may overlap source final validation. Train sample/BPE require F4 final proof and evidence-bound distribution review.')
    atomic_json(root/'contract.json',contract);atomic_json(root/'state.json',dict(phase='running',case='launch'))
    for f,h in contract['code'].items():
        destination=root/'producer_code'/f;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(resolve(f),destination)
        if sha256_file(destination)!=h:raise ValueError('Dependency changed during launch snapshot')
    env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='16',TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch),SQLITE_TMPDIR=str(scratch))
    cmd=['taskset','-c',','.join(map(str,hw['cpu']['affinity'][:64])),sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
         '--run-dir',str(root),'--sources',a.sources,'--split',a.split,'--tokenizer',a.tokenizer,'--source-run',a.source_run,'--hash-workers',str(a.hash_workers),'--quality-docs',str(a.quality_docs),'--scratch',str(scratch)]
    with (root/'console.log').open('w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.monotonic();peak=0
    try:
        while proc.poll() is None:
            rss=process_tree_rss(proc.pid);peak=max(peak,rss);source_state=json.loads((resolve(a.source_run)/'state.json').read_text())
            if source_state['phase'] in ('error','paused'):raise RuntimeError('F4 source '+source_state['phase'])
            limit=384 if source_state['phase']=='complete' else 256
            ws=root/'worker_state.json';w=json.loads(ws.read_text()) if ws.exists() else dict(phase='launch')
            atomic_json(root/'state.json',dict(phase='running',case=w['phase'],worker_pid=proc.pid,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak,rss_limit_gib=limit,worker=w))
            if rss>limit*2**30 or time.monotonic()-start>a.max_seconds:raise RuntimeError('F5 time/RSS budget')
            for directory,budget in [(scratch,192),(root,96)]:
                if shutil.disk_usage(directory).free<200*2**30 or size_tree(directory)>budget*2**30:raise RuntimeError('F5 disk budget/reserve')
            time.sleep(2)
        if proc.returncode:raise RuntimeError('F5 worker failed with exit '+str(proc.returncode))
        proof=json.loads((root/'.done').read_text())
        if sha256_file(root/'report.json')!=proof['report_sha256'] or sha256_file(root/'contract.json')!=proof['contract_sha256']:raise ValueError('F5 completion proof differs')
        atomic_json(root/'state.json',dict(phase='complete',elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak))
    except BaseException as exc:
        if proc.poll() is None:stop(proc)
        atomic_json(root/'state.json',dict(phase='error',error=str(exc),peak_rss_bytes=peak));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--sources',default='data/datasets/final_v1.ready.json');p.add_argument('--split',default='data/datasets/final_v2.split/manifest.json')
    p.add_argument('--tokenizer',default='tokenizer/candidate_v1/tokenizer.json');p.add_argument('--source-run',default='runs/f4_full_v2')
    p.add_argument('--hash-workers',type=int,default=32);p.add_argument('--quality-docs',type=int,default=1024);p.add_argument('--scratch');p.add_argument('--max-seconds',type=float,default=24*3600)
    a=p.parse_args();worker(a) if a.worker else main(a)
