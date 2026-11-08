"""Bounded actual-token eager FSDP8 batch search before exact-geometry validation.

Inputs must be certified final-tokenizer whole-file pilot shards. This performs
eight updates per case with no checkpoint saving, and is not 3B main training.
Candidates remain conditional on exact geometry/LR/finite/eval/checkpoint/resume tests.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file,identity,artifact_path
from scripts.validate_actual_training import validation_geometry


def summarize(rows,*,steps=8,warmup_steps=2,global_tokens=2097152):
    rows=[r for r in rows if 'tok_s' in r]
    if [r.get('step') for r in rows]!=list(range(1,steps+1)) or not 0<=warmup_steps<steps:raise ValueError('Incomplete throughput trajectory')
    for r in rows:
        for k in ['loss','gnorm','tok_s','mfu','max_allocated_gib','max_reserved_gib','tokens_b']:
            if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]):raise ValueError('Non-finite throughput metrics')
        if r['tok_s']<=0 or r['max_reserved_gib']<r['max_allocated_gib'] or r['max_allocated_gib']<0:raise ValueError('Invalid time/memory metrics')
        if not math.isclose(r['tokens_b']*1e9,r['step']*global_tokens,abs_tol=1):raise ValueError('Measured token geometry differs')
    measured=rows[warmup_steps:];seconds=[global_tokens/r['tok_s'] for r in measured]
    return dict(aggregate_tok_s=global_tokens*len(measured)/sum(seconds),median_tok_s=statistics.median(r['tok_s'] for r in measured),
                timed_seconds=sum(seconds),timed_tokens=global_tokens*len(measured),timed_steps=len(measured),
                max_allocated_gib=max(r['max_allocated_gib'] for r in rows),max_reserved_gib=max(r['max_reserved_gib'] for r in rows),
                first_step_seconds=global_tokens/rows[0]['tok_s'],final_loss=rows[-1]['loss'])


def choose(cases,capacity_gib,*,memory_fraction=.90,near_fastest_fraction=.03):
    if not math.isfinite(capacity_gib) or capacity_gib<=0:raise ValueError('Invalid GPU capacity')
    valid=[r for r in cases if r['status']=='passed' and r['max_reserved_gib']<=capacity_gib*memory_fraction]
    if not valid:raise ValueError('No finite throughput candidate within GPU memory margin')
    fastest=max(r['aggregate_tok_s'] for r in valid)
    close=[r for r in valid if r['aggregate_tok_s']>=fastest*(1-near_fastest_fraction)]
    return min(close,key=lambda r:(r['max_reserved_gib'],r['micro_bsz']))


def main(a):
    from data.pilot_mix import verify_pilot_mix
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    mix=verify_pilot_mix(a.diagnostic_input);hw=inspect()
    if mix['seq_len']!=4096 or len(hw['gpus'])!=8 or hw['memory']['conservative_headroom_gib']<512:
        raise RuntimeError('Require actual4096-token pilot,8GPUs and concurrent future memory budget')
    run=resolve(a.run_dir);run.resolve().relative_to(resolve('runs').resolve());run.mkdir(parents=True,exist_ok=True)
    lock=(run/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu_lock=resolve('runs/.gpu_benchmark.lock').open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    occupied=lambda:subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied():raise RuntimeError('GPU compute already active')
    scratch=Path(hw['storage']['scratch_dir'])/run.name;scratch.mkdir(parents=True,exist_ok=True)
    capacity=min(float(gpu.rsplit(',',1)[-1]) for gpu in hw['gpus'])/1024
    cpus=hw['cpu']['affinity'][64:] or hw['cpu']['affinity']
    import torch
    framework=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                   driver_versions=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines())
    deps=['scripts/benchmark_actual_training.py','scripts/validate_actual_training.py','data/pilot_mix.py','train/train.py','train/runtime.py','train/data.py','model/model.py','common/data_contracts.py']
    stable=dict(version=1,mix_sha256=sha256_file(resolve(a.diagnostic_input)/'mix.json'),models={'1b':[4,8,16],'3b':[2,4,8]},
        global_tokens=2097152,steps=8,warmup_steps=2,compile=False,code={f:sha256_file(resolve(f)) for f in deps},
        gpu_identity=hw['gpus'],framework=framework,cpu_affinity=cpus,memory_capacity_gib=capacity,memory_fraction=.90,near_fastest_fraction=.03,
        repeat_max_relative_difference=.10,host_rss_gib=256,local_budget_gib=32,durable_budget_gib=8,reserve_gib=200,max_case_seconds=900,max_job_seconds=3600,
        scope='Final-tokenizer pilot actual eager8GPU throughput only; selected exact geometry requires full finite/eval/RNG/checkpoint/resume validation.')
    cp=run/'contract.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        if {k:v for k,v in old.items() if k!='hardware'}!=stable:raise ValueError('Benchmark identity changed; choose a new run')
        contract=old
    else:
        if {p.name for p in run.iterdir()}!={'controller.lock'}:raise ValueError('Uncontracted benchmark output')
        contract=dict(stable,hardware=hw);atomic_json(cp,contract)
    cid=identity(stable);(run/'pid').write_text(str(os.getpid()))
    atomic_json(run/f'hardware_{time.time_ns()}.json',hw)
    env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TMPDIR=str(scratch),TORCHINDUCTOR_COMPILE_THREADS='1',PYTHONUNBUFFERED='1')
    results=[];proc=None;job_started=time.monotonic()
    def case(model,micro,*,repeat=False):
        nonlocal proc
        name=f'{model}_b{micro}'+('_repeat' if repeat else '');out=run/name;out.mkdir(exist_ok=True)
        chosen=validation_geometry({model:dict(micro_bsz=micro,global_tokens=2097152)})[model]
        config=dict(model=model,vocab_size=65536,seq_len=4096,**chosen,max_steps=8,warmup=1,decay_frac=.15,compile=False,
                    mix=str(resolve(a.diagnostic_input)/'mix.json'),holdout=[],out_dir=str(out),log_every=1,eval_every=0,ckpt_every=0,
                    keep_ckpts=1,save_final=False,sampler_version=2)
        path=out/'config.json';result=out/'result.json';marker=out/'.done'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Benchmark case config differs')
        if marker.exists():
            proof=json.loads(marker.read_text())
            if proof['contract_id']!=cid:raise ValueError('Benchmark case identity differs')
            for f,k in [(result,'result_sha256'),(path,'config_sha256'),(out/'log.jsonl','log_sha256'),(out/'console.log','console_sha256')]:
                if sha256_file(f)!=proof[k]:raise ValueError('Benchmark case content changed')
            return json.loads(result.read_text())
        if occupied():raise RuntimeError('GPU compute appeared before benchmark case')
        if any(out.glob('ckpt_*')):raise ValueError('Unexpected checkpoint in throughput-only case')
        for old in [out/'log.jsonl',out/'console.log']:
            if old.exists():old.rename(old.with_name(old.name+f'.interrupted-{time.time_ns()}'))
        atomic_json(path,config);start=time.monotonic();peak=0
        cmd=['taskset','-c',','.join(map(str,cpus)),'torchrun','--standalone','--nproc_per_node=8','train/train.py','--config',str(path)]
        with (out/'console.log').open('w') as stream:proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            elapsed=time.monotonic()-start;peak=max(peak,process_tree_rss(os.getpid()))
            atomic_json(run/'state.json',dict(phase='running',case=name,worker_pid=proc.pid,elapsed_seconds=elapsed,peak_rss_bytes=peak))
            if elapsed>900 or time.monotonic()-job_started>3600 or peak>256*2**30:raise RuntimeError('Throughput case time/RSS budget')
            for directory,budget in [(scratch,32),(run,8)]:
                if size_tree(directory)>budget*2**30 or shutil.disk_usage(directory).free<200*2**30:raise RuntimeError('Throughput case disk budget')
            time.sleep(2)
        log=out/'log.jsonl'
        if not log.exists():log.touch()
        rows=[json.loads(s) for s in log.read_text().splitlines()]
        if proc.returncode:
            if 'CUDA out of memory' not in (out/'console.log').read_text():raise RuntimeError('Actual throughput case failed:'+name)
            r=dict(status='cuda_oom',model=model,micro_bsz=micro)
        else:r=dict(status='passed',model=model,micro_bsz=micro,**summarize(rows))
        r.update(wall_seconds=time.monotonic()-start,peak_host_rss_bytes=peak,repeat=repeat)
        atomic_json(result,r);atomic_json(marker,dict(contract_id=cid,result_sha256=sha256_file(result),config_sha256=sha256_file(path),log_sha256=sha256_file(log),console_sha256=sha256_file(out/'console.log')))
        return r
    try:
        selected={};stable_repeats=True
        for model,micros in stable['models'].items():
            candidates=[]
            for micro in micros:
                r=case(model,micro);results.append(r);candidates.append(r);atomic_json(run/'partial_results.json',results)
                if r['status']=='cuda_oom':break
            best=choose(candidates,capacity);again=case(model,best['micro_bsz'],repeat=True);results.append(again)
            if again['status']!='passed' or abs(again['aggregate_tok_s']/best['aggregate_tok_s']-1)>.10 or again['max_reserved_gib']>capacity*.90:stable_repeats=False
            selected[model]=dict(micro_bsz=best['micro_bsz'],global_tokens=2097152,lr=6e-4 if model=='1b' else 3e-4)
        verify_pilot_mix(a.diagnostic_input)
        for f,h in stable['code'].items():
            if sha256_file(resolve(f))!=h:raise ValueError('Throughput producer changed')
        atomic_json(run/'selected_model_settings.json',selected)
        report=dict(status='candidate_selected' if stable_repeats else 'needs_review',scope=stable['scope'],results=results,selected_model_settings=selected,
                    selected_model_settings_sha256=sha256_file(run/'selected_model_settings.json'),remaining='Actual exact-geometry/LR4step finite/eval/checkpoint/RNG/resume validation and native PPL are mandatory before rehearsal. Compile not adopted.')
        atomic_json(run/'report.json',report);atomic_json(run/'.done',dict(report_sha256=sha256_file(run/'report.json'),contract_sha256=sha256_file(cp)))
        atomic_json(run/'state.json',dict(phase='complete'))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:stop(proc)
        atomic_json(run/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--diagnostic-input',required=True)
    main(p.parse_args())
