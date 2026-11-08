"""Actual3B four-update monitoring integration, gated after rehearsal/native.

New checkpoints stay local and are preserved. It never starts main training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.run_native_evaluation import certified

GIB = 2**30


def seal_checkpoint(path, output, contract_id):
    path, output = Path(path), Path(output)
    meta = json.loads((path/'meta.json').read_text()); step = int(path.name.split('_')[1])
    if (step not in (2,4) or meta['step'] != step or meta['world_size'] != 8 or
            meta['tokens_per_step'] != 2097152 or meta['config']['model'] != '3b' or
            meta['config']['max_steps'] != 4 or meta['training_contract']['id'] != contract_id or
            not (path/'.complete').is_file() or not (path/'.metadata').is_file() or
            not all((path/f'rng_rank{rank}.pt').is_file() for rank in range(8))):
        raise ValueError('Actual bounded3B checkpoint identity differs')
    paths = sorted(path.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Checkpoint symlinks rejected')
    import torch.distributed.checkpoint as dcp
    storage=dcp.FileSystemReader(path).read_metadata().storage_data
    if not storage:
        raise ValueError('Checkpoint DCP payload map missing')
    for item in storage.values():
        payload=path/item.relative_path
        if (not payload.resolve().is_relative_to(path.resolve()) or not payload.is_file() or
                payload.stat().st_size<item.offset+item.length):
            raise ValueError('Checkpoint DCP payload missing or truncated')
    def record(p):
        before = p.stat()
        with p.open('rb') as stream:
            os.fsync(stream.fileno())
        digest = sha256_file(p); after = p.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino):
            raise ValueError('Checkpoint changed during verification')
        return dict(path=str(p.relative_to(path)),bytes=after.st_size,sha256=digest)
    with ThreadPoolExecutor(max_workers=8) as hashes:
        records = list(hashes.map(record,[p for p in paths if p.is_file()]))
    for directory in sorted([path]+[p for p in paths if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd = os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    report = dict(status='passed_actual_checkpoint_storage',checkpoint=artifact_path(path),step=step,
                  training_contract_id=contract_id,files=records,bytes=sum(r['bytes'] for r in records),
                  scope='Local bounded3B monitoring integration checkpoint; not a production checkpoint.')
    if output.exists():
        if json.loads(output.read_text()) != report:
            raise ValueError('Previously certified checkpoint changed')
    else:
        atomic_json(output,report)
    return report


def verify_generations(paths, schedule):
    observed = {}
    for path in paths:
        row = json.loads(Path(path).read_text())
        for item in row['generations']:
            key = (item['step'],row['rank'])
            previous = observed.setdefault(key,item['sha256'])
            if previous != item['sha256']:
                raise ValueError('Replayed generation decisions changed')
    if set(observed) != {(s,r) for s in schedule for r in range(8)}:
        raise ValueError('Generation step/rank coverage differs')
    result = {}
    for step in schedule:
        values = {observed[(step,rank)] for rank in range(8)}
        if len(values) != 1:
            raise ValueError('Generation rank decisions differ')
        result[str(step)] = values.pop()
    return result


def prepare_contract(a, root, scratch):
    from scripts.run_rehearsal import prerequisites
    from train.train import DEFAULT
    from train.probe import select_probe
    from train.parallel_contract import native_expected_inputs
    inputs = prerequisites(a)
    probe = select_probe(inputs['token_root'],root/'probe',inputs['categories'])
    config = dict(DEFAULT, model='3b',vocab_size=65536,seq_len=4096,**inputs['settings']['3b'],
                  max_steps=4,warmup=1000,compile=False,mix=artifact_path(inputs['mix_path']),holdout=probe['holdout'],
                  out_dir=str(scratch/'training'),ckpt_every=2,eval_every=2,log_every=1,keep_ckpts=5,
                  save_final=True,stop_after=None,sampler_version=2)
    for name, value in [('early.json',dict(config,stop_after=2)),('main.json',config)]:
        if (root/name).exists() and json.loads((root/name).read_text()) != value:
            raise ValueError('Bounded3B config changed')
        atomic_json(root/name,value)
    expected = native_expected_inputs(inputs['token_root'],inputs['mix'],probe['holdout'])
    ep = root/'expected_tokens.json'
    if ep.exists() and json.loads(ep.read_text()) != expected:
        raise ValueError('Bounded3B expected token inputs changed')
    atomic_json(ep,expected)
    code = {**inputs['code'],**{p:sha256_file(resolve(p)) for p in [
        'scripts/validate_monitored_training.py','train/monitored.py','train/rehearsal.py',
        'scripts/plan_main_training.py','scripts/run_rehearsal.py','train/parallel_contract.py',
        'train/probe.py','scripts/resource_admission.py','scripts/evaluate_final_rehearsal.py']}}
    evidence = {**inputs['evidence'],artifact_path(ep):sha256_file(ep)}
    return dict(version=1,purpose='bounded_validation',main_training_authorized=False,
                configurations={n:sha256_file(root/n) for n in ['early.json','main.json']},
                code=code,evidence=evidence,prompts=inputs['prompts'],tokenizer=inputs['tokenizer'],
                generation_steps=[2,4],scratch=str(scratch),rehearsal_run=a.rehearsal_run,
                final_native_run=a.final_native_run,host_rss_gib=256,local_budget_gib=96,
                durable_budget_gib=1,reserve_gib=200,max_seconds=172800,worker_max_seconds=3600,
                scope='Actual3B source/gen/checkpoint/resume monitoring integration only4updates; preserves all outputs locally. Main training remains unauthorized.')


def check_contract(c):
    for p,h in {**c['code'],**c['evidence']}.items():
        if sha256_file(resolve(p)) != h:
            raise ValueError('Bounded3B implementation/input changed: '+p)


def main(a):
    from scripts.resource_admission import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    from scripts.evaluate_final_rehearsal import rehearsal_observation,completed_rehearsal,acquire_idle_gpu
    from scripts.run_rehearsal import trace_review
    from train.rehearsal import aggregate_ledgers
    from train.probe import verify_probe
    from train.runtime import complete_checkpoints
    root = resolve(a.run_dir); root.relative_to(resolve('runs')); root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        hw = inspect(); scratch = Path(hw['storage']['scratch_dir'])/root.name
        scratch.mkdir(parents=True,exist_ok=True)
        c = prepare_contract(a,root,scratch)
        if (root/'contract.json').exists():
            if json.loads((root/'contract.json').read_text()) != c:
                raise ValueError('Bounded3B contract changed; preserve and use a new run path')
        else:
            atomic_json(root/'contract.json',c)
        if (root/'.done').exists(): return certified(root,'passed_actual_monitored_3b')
        if (root/'state.json').exists():
            previous = json.loads((root/'state.json').read_text())
            pid = previous.get('worker_pid')
            if pid and Path(f'/proc/{pid}/cmdline').exists():
                cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'train.monitored' in cmd and str(root).encode() in cmd:
                    raise RuntimeError('Existing monitored worker live; preserve before restart')
        (root/'pid').write_text(str(os.getpid())+'\n')
        for directory in ['ledgers','attempts','checkpoint_certificates']:
            (root/directory).mkdir(exist_ok=True)
        proc = None; start = time.monotonic(); peak = 0
        def progress(phase,**kw):
            atomic_json(root/'state.json',dict(phase='running',case=phase,elapsed_seconds=time.monotonic()-start,**kw))
        try:
            while True:
                observed = rehearsal_observation(a.rehearsal_run)
                final = resolve(a.final_native_run)
                if observed['complete'] and (final/'.done').exists(): break
                if (final/'state.json').exists():
                    phase=json.loads((final/'state.json').read_text()).get('phase')
                    if phase in ('error','failed','paused') or (phase=='complete' and not (final/'.done').exists()):
                        raise RuntimeError('Final native evaluation requires recovery')
                progress('await_rehearsal_and_final_native',step=observed['step'])
                if time.monotonic()-start > c['max_seconds']: raise RuntimeError('Bounded3B prerequisite wait budget')
                time.sleep(30)
            completed_rehearsal(a.rehearsal_run)
            certified(a.final_native_run,'passed_actual_trained_native_evaluation')
            check_contract(c); hw=inspect(); atomic_json(root/('hardware_'+str(time.time_ns())+'.json'),hw)
            if len(hw['gpus'])!=8 or hw['memory']['cache_aware_admission']['headroom_gib']<384:
                raise RuntimeError('Bounded3B measured memory/GPU headroom unavailable')
            if shutil.disk_usage(scratch).free < (96+200)*GIB or shutil.disk_usage(root).free < 201*GIB:
                raise RuntimeError('Bounded3B local/durable reserve unavailable')
            cpu = hw['cpu']['affinity'][64:] or hw['cpu']['affinity']
            env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='1',
                     TMPDIR=str(scratch),SQLITE_TMPDIR=str(scratch),PYTHONUNBUFFERED='1')
            with acquire_idle_gpu(progress):
                verified={}; training=scratch/'training'
                for path in complete_checkpoints(training):
                    proof=root/'checkpoint_certificates'/(path.name+'.json')
                    if proof.exists():
                        cid=json.loads(proof.read_text())['training_contract_id']
                        verified[path.name]=seal_checkpoint(path,proof,cid)
                    else:
                        archive=scratch/'unverified_checkpoints';archive.mkdir(exist_ok=True)
                        path.rename(archive/(path.name+'-'+str(time.time_ns())))
                for phase,end in [('early',2),('main',4)]:
                    checkpoints=complete_checkpoints(training)
                    latest=int(checkpoints[-1].name.split('_')[1]) if checkpoints else 0
                    if latest>=end: continue
                    attempt=phase+'-'+str(time.time_ns())
                    cmd=['taskset','-c',','.join(map(str,cpu)),sys.executable,'-m','torch.distributed.run',
                         '--standalone','--nproc_per_node=8','--module','train.monitored','--run-dir',str(root),
                         '--config',str(root/(phase+'.json')),'--attempt',attempt]
                    with (root/'attempts'/(attempt+'.log')).open('w') as log:
                        proc=subprocess.Popen(cmd,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    begun=time.monotonic()
                    while proc.poll() is None:
                        peak=max(peak,process_tree_rss(os.getpid()));progress(phase,worker_pid=proc.pid,peak_rss_bytes=peak)
                        if peak>256*GIB or time.monotonic()-begun>3600:raise RuntimeError('Bounded3B RSS/phase time budget')
                        if size_tree(scratch)>96*GIB or size_tree(root)>GIB or min(shutil.disk_usage(root).free,shutil.disk_usage(scratch).free)<200*GIB:
                            raise RuntimeError('Bounded3B disk budget')
                        time.sleep(2)
                    if proc.returncode: raise RuntimeError('Bounded3B worker failed; preserve outputs and inspect attempt log')
                    cid=json.loads((training/'training_contract.json').read_text())['id']
                    for path in complete_checkpoints(training):
                        if path.name not in verified:
                            verified[path.name]=seal_checkpoint(path,root/'checkpoint_certificates'/(path.name+'.json'),cid)
                if set(verified)!={'ckpt_0000002','ckpt_0000004'}:raise ValueError('Bounded3B retained checkpoint coverage differs')
                config=json.loads((root/'main.json').read_text());probe=verify_probe(root/'probe');paths=list((root/'ledgers').glob('*.json'))
                sources=json.loads(resolve(config['mix']).read_text())['sources']
                counts=aggregate_ledgers(paths,world=8,steps=4,seq_len=4096,global_tokens=2097152,sources=sources)
                generations=verify_generations(paths,[2,4]);curves=trace_review(training/'log.jsonl',config,probe,end_step=4)
                if generations!=counts['generation_sha256_by_step'] or generations!=curves['generation_sha256_by_step']:
                    raise ValueError('Bounded3B generation logs/rank receipts differ')
            check_contract(c)
            reports=[training/'log.jsonl',training/'training_contract.json',*paths,*sorted((root/'checkpoint_certificates').glob('*.json'))]
            report=dict(status='passed_actual_monitored_3b',counts=counts,curves=curves,
                        checkpoint_certificates=verified,elapsed_seconds=time.monotonic()-start,peak_host_rss_bytes=peak,
                        evidence={artifact_path(p):sha256_file(p) for p in reports},actual_local_bytes=size_tree(scratch),
                        actual_durable_bytes=size_tree(root),scope=c['scope'],
                        limitation='Monitoring hooks actual3B early2/resume4 verified; core exact model/optimizer/RNG resume is separate F7 evidence. No long3B convergence or main training performed.')
            atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
            atomic_json(root/'state.json',dict(phase='complete'))
        except BaseException as exc:
            if proc is not None and proc.poll() is None:stop(proc)
            atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--token-run',default='runs/f6_tokens_v1');p.add_argument('--batch-run',default='runs/actual_deterministic_batch_v1')
    p.add_argument('--validation-run',default='runs/actual_train_validation_v1');p.add_argument('--native-eval-run',default='runs/native_eval_validation_v1')
    p.add_argument('--rehearsal-run',default='runs/1b_rehearsal_final_v3');p.add_argument('--final-native-run',default='runs/rehearsal_final_native_v1')
    main(p.parse_args())
