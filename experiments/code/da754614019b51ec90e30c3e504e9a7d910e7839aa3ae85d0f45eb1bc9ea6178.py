"""Bounded 1B/3B FSDP8 correctness checks with certified final-tokenizer pilot inputs.

This is four-step diagnostic training, not 3B main pretraining. Large temporary
checkpoints stay on measured local storage; final checkpoints are published with
fresh per-file hashes. Production batch tuning/publication PPL/rehearsal are separate.
"""
import argparse
import copy
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
from common.data_contracts import resolve,artifact_path,atomic_json,sha256_file


def validation_geometry(settings=None):
    if settings is not None and not isinstance(settings,dict):raise ValueError('Model settings must be an object')
    result={}
    for model in ['1b','3b']:
        c=dict(micro_bsz=1,global_tokens=65536,lr=6e-4 if model=='1b' else 3e-4);overrides=(settings or {}).get(model,{})
        if not isinstance(overrides,dict) or set(overrides)-set(c):raise ValueError('Only microbatch/global tokens/peak LR can be selected')
        c.update(overrides);micro=c['micro_bsz'];tokens=c['global_tokens']
        if type(micro) is not int or not 1<=micro<=64 or type(tokens) is not int or not 1<=tokens<=2097152 or tokens%(8*4096*micro):
            raise ValueError('Invalid bounded actual validation batch geometry')
        if type(c['lr']) not in (int,float) or not math.isfinite(c['lr']) or not 0<c['lr']<=.001:raise ValueError('Invalid bounded peak LR')
        result[model]=c
    if set(settings or {})-set(result):raise ValueError('Unknown actual validation model')
    return result


def compare_trajectories(left,right):
    rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if '"loss"' in s]
    a,b=rows(left),rows(right)
    if len(a)!=4 or len(b)!=4:raise ValueError('Expected four actual-token updates')
    for step,(x,y) in enumerate(zip(a,b),1):
        if x['step']!=step or y['step']!=step or x['lr']!=y['lr']:raise ValueError('Actual-token resume schedule differs')
        for r in [x,y]:
            if any(not math.isfinite(r[k]) for k in ['loss','gnorm','tok_s','mfu']):raise ValueError('Non-finite actual-token training metrics')
        if abs(x['loss']-y['loss'])>1e-6:raise ValueError('Actual-token resume loss differs')
    return dict(loss_max_abs=max(abs(x['loss']-y['loss']) for x,y in zip(a,b)),updates=4,
                scope='Fixed eager four-step resume trajectory; logged tok_s includes prior eval/checkpoint time and is not a steady-state throughput benchmark.')


def publish_checkpoint(source,destination):
    """Atomic new publication, never overwrite an existing or interrupted checkpoint."""
    source=Path(source);destination=Path(destination)
    if not (source/'.complete').is_file():raise ValueError('Incomplete actual-token checkpoint')
    files=[p for p in source.rglob('*') if p.is_file()]
    if any(p.is_symlink() for p in source.rglob('*')):raise ValueError('Unexpected checkpoint symlink')
    records=[dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in sorted(files)]
    if sum(r['bytes'] for r in records)+200*2**30>shutil.disk_usage(destination.parent).free:raise RuntimeError('Checkpoint publication reserve')
    if destination.exists():raise ValueError('Refusing checkpoint publication overwrite')
    staging=destination.with_name(destination.name+'.publishing')
    if staging.exists():raise ValueError('Preserve interrupted publication; choose a new destination')
    staging.mkdir()
    for r in records:
        src=source/r['path'];dst=staging/r['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
        if dst.stat().st_size!=r['bytes'] or sha256_file(dst)!=r['sha256']:raise ValueError('Checkpoint copy hash differs')
        with dst.open('rb') as stream:os.fsync(stream.fileno())
    atomic_json(staging/'publication.json',dict(source=artifact_path(source),files=records))
    with (staging/'publication.json').open('rb') as stream:os.fsync(stream.fileno())
    for directory in sorted([staging]+[p for p in staging.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    staging.rename(destination)
    fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=artifact_path(destination),bytes=sum(r['bytes'] for r in records),publication_sha256=sha256_file(destination/'publication.json'))


def main(a):
    if a.compare_left:
        from scripts.validate_distributed import checkpoint_compare
        atomic_json(resolve(a.comparison_output),checkpoint_compare(resolve(a.compare_left),resolve(a.compare_right)))
        return
    if a.worker:
        from scripts.validate_distributed import worker
        return worker(a.worker,a.mode)
    if not a.run_dir or not a.diagnostic_input:raise ValueError('Require run directory and certified diagnostic input')
    from data.pilot_mix import verify_pilot_mix
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    mix=verify_pilot_mix(a.diagnostic_input);hw=inspect()
    if len(hw['gpus'])!=8 or hw['memory']['conservative_headroom_gib']<512:raise RuntimeError('Require8GPUs and concurrent-job future memory budget')
    if mix['seq_len']!=4096:raise ValueError('Actual full-context check requires4096-token windows')
    run=resolve(a.run_dir);run.mkdir(parents=True,exist_ok=False)
    scratch=Path(hw['storage']['scratch_dir'])/run.name;scratch.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(scratch).free<(512+200)*2**30 or shutil.disk_usage(run).free<(96+200)*2**30:raise RuntimeError('Actual validation storage budget')
    lock=resolve('runs/.gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute already active')
    cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity']
    import torch
    deps=['scripts/validate_actual_training.py','scripts/validate_distributed.py','data/pilot_mix.py','model/model.py','train/train.py','train/runtime.py','train/data.py','train/__init__.py']
    selected=validation_geometry(json.loads(resolve(a.model_settings).read_text()) if a.model_settings else None)
    contract=dict(version=1,hardware=hw,framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda),diagnostic_input=artifact_path(resolve(a.diagnostic_input)),mix_sha256=sha256_file(resolve(a.diagnostic_input)/'mix.json'),batch_geometry=selected,
                  code={f:sha256_file(resolve(f)) for f in deps},models=['1b','3b'],world=8,sequence=4096,steps=4,
                  compile=False,max_case_seconds=900,host_rss_gib=256,local_budget_gib=512,durable_budget_gib=96,reserve_gib=200,
                  cpu_affinity=cpus,checkpoint_tolerance=dict(atol=1e-6,rtol=1e-5),loss_resume_atol=1e-6,
                  scope='Actual final-tokenizer whole-file pilot; eager four-step finite/eval/checkpoint/RNG/resume checks only. Not production batch optimization or main training.')
    atomic_json(run/'contract.json',contract);(run/'pid').write_text(str(os.getpid()))
    results=[];case_results=[];started=time.monotonic();proc=None
    env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1',TMPDIR=str(scratch),TORCHINDUCTOR_COMPILE_THREADS='1')
    def launch(name,config,mode='normal',expected_error=None):
        nonlocal proc
        path=run/(name+'.json');atomic_json(path,config);start=time.monotonic();peak=0
        cmd=['taskset','-c',','.join(map(str,cpus))]
        if mode=='checkpoint_compare':
            cmd+=[sys.executable,str(Path(__file__).resolve()),'--compare-left',config['left'],'--compare-right',config['right'],'--comparison-output',config['output']]
        else:cmd+=['torchrun','--standalone','--nproc_per_node=8',str(Path(__file__).resolve()),'--worker',str(path),'--mode',mode]
        with (run/(name+'.log')).open('w') as stream:proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            peak=max(peak,process_tree_rss(os.getpid()));elapsed=time.monotonic()-start
            atomic_json(run/'state.json',dict(phase='running',case=name,worker_pid=proc.pid,elapsed_seconds=elapsed,peak_rss_bytes=peak))
            if elapsed>900 or peak>256*2**30:raise RuntimeError('Actual training case time/RSS budget')
            for directory,budget in [(scratch,512),(run,96)]:
                if size_tree(directory)>budget*2**30 or shutil.disk_usage(directory).free<200*2**30:raise RuntimeError('Actual training disk budget/reserve')
            time.sleep(2)
        text=(run/(name+'.log')).read_text()
        if expected_error is None:
            if proc.returncode:raise RuntimeError('Actual training failed:'+name)
        elif not proc.returncode or expected_error not in text:raise RuntimeError('Expected identity rejection missing:'+name)
        case_results.append(dict(case=name,returncode=proc.returncode,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak,expected_error=expected_error))
        atomic_json(run/'cases.json',case_results)
    try:
        for model in ['1b','3b']:
            local=scratch/model;local.mkdir();base=dict(model=model,vocab_size=65536,seq_len=4096,**selected[model],max_steps=4,warmup=1,
                  decay_frac=.5,compile=False,mix=str(resolve(a.diagnostic_input)/'mix.json'),holdout=mix['holdout'],
                  log_every=1,eval_every=2,ckpt_every=2,keep_ckpts=3,sampler_version=2)
            continuous=dict(base,out_dir=str(local/'continuous'));resumed=dict(base,out_dir=str(local/'resumed'))
            launch(model+'_continuous',continuous,'eval_suite')
            launch(model+'_stop_at2',dict(resumed,stop_after=2))
            partial=local/'resumed/ckpt_0000003';partial.mkdir();(partial/'partial').write_text('Deliberately incomplete diagnostic checkpoint')
            launch(model+'_resume',resumed)
            trajectory=compare_trajectories(local/'continuous/log.jsonl',local/'resumed/log.jsonl')
            comparison_path=run/(model+'_comparison.json')
            launch(model+'_checkpoint_compare',dict(left=str(local/'continuous/ckpt_0000004'),right=str(local/'resumed/ckpt_0000004'),output=str(comparison_path)),mode='checkpoint_compare')
            comparison=json.loads(comparison_path.read_text())
            changed=copy.deepcopy(mix);changed['tokenizer_sha256']='deliberately-wrong-diagnostic-identity'
            changed_path=run/(model+'_changed_mix.json');atomic_json(changed_path,changed)
            launch(model+'_reject_changed_tokenizer',dict(resumed,mix=str(changed_path)),expected_error='Training identity mismatch')
            for mode in ['nan_loss','nan_gradient']:
                out=local/mode;launch(model+'_'+mode,dict(base,out_dir=str(out),ckpt_every=1,eval_every=0),mode)
                faults=[json.loads((out/f'fault_rank{rank}.json').read_text()) for rank in range(8)]
                if not all(r['optimizer_updates']==1 for r in faults) or sorted(p.name for p in out.glob('ckpt_*'))!=['ckpt_0000001']:
                    raise ValueError('Actual-token non-finite failure was not synchronized before optimizer')
            published=publish_checkpoint(local/'resumed/ckpt_0000004',run/(model+'_validated_checkpoint'))
            eval_checks=[json.loads(s) for s in (local/'continuous/eval_checks.jsonl').read_text().splitlines()]
            if len(eval_checks)!=2 or any([r['count'] for r in suite]!=[1,7,9] for suite in eval_checks):raise ValueError('Actual uneven eval coverage differs')
            results.append(dict(model=model,trajectory=trajectory,checkpoint_comparison=comparison,published_checkpoint=published,uneven_evaluation=eval_checks,
                                local_checkpoint_directory=artifact_path(local)))
            atomic_json(run/'partial_results.json',results)
        verify_pilot_mix(a.diagnostic_input)
        for f,h in contract['code'].items():
            if sha256_file(resolve(f))!=h:raise ValueError('Actual validation implementation changed')
        report=dict(status='passed_bounded_actual_training',scope=contract['scope'],results=results,cases=case_results,elapsed_seconds=time.monotonic()-started,
                    actual_local_bytes=size_tree(scratch),actual_durable_bytes=size_tree(run),
                    remaining=['Actual full-mix input verification','Production micro-batch/accumulation performance sweep','Native ko/en full-coverage PPL and fixed generation','1B10B-token rehearsal','Final pretraining readiness review'])
        atomic_json(run/'report.json',report);atomic_json(run/'.done',dict(report_sha256=sha256_file(run/'report.json'),contract_sha256=sha256_file(run/'contract.json')))
        atomic_json(run/'state.json',dict(phase='complete'))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:stop(proc)
        atomic_json(run/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir');p.add_argument('--diagnostic-input');p.add_argument('--worker');p.add_argument('--mode',default='normal')
    p.add_argument('--compare-left');p.add_argument('--compare-right');p.add_argument('--comparison-output')
    p.add_argument('--model-settings',help='Selected1B/3B micro_bsz/global_tokens/lr JSON; steps remain bounded at4')
    main(p.parse_args())
