"""Gated, resumable1B~10B-token rehearsal. Never launches3B main training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from common.data_contracts import resolve,artifact_path,atomic_json,sha256_file,identity
from scripts.run_native_evaluation import certified


def make_config(settings,mix,probe,output):
    from train.train import DEFAULT
    from scripts.validate_actual_training import validation_geometry
    selected=validation_geometry(settings)['1b']
    if selected['global_tokens']!=2097152 or mix['seq_len']!=4096 or probe['seq_len']!=4096:
        raise ValueError('Rehearsal requires validated4096/global2M geometry')
    if mix['tokenizer_sha256']!=probe['tokenizer_sha256'] or set(mix['sources'])!={r['source'] for r in probe['groups']}:
        raise ValueError('Rehearsal mix/probe source/tokenizer differs')
    return dict(DEFAULT,model='1b',vocab_size=65536,seq_len=4096,**selected,max_steps=5000,warmup=200,decay_frac=.15,
                compile=False,holdout=probe['holdout'],out_dir=str(output),eval_every=250,ckpt_every=500,log_every=10,
                keep_ckpts=5001,save_final=True,sampler_version=2,stop_after=None)


def prerequisites(a):
    f6,c6=certified(a.token_run,'passed');batch,bc=certified(a.batch_run,'candidate_selected')
    validation,vc=certified(a.validation_run,'passed_bounded_actual_training');native,nc=certified(a.native_eval_run,'passed_actual_native_evaluation')
    f5=resolve(c6['settings']['tokenizer_run']);trained,tc=certified(f5,'passed')
    if f6['tokenizer_run_sha256']!=sha256_file(f5/'report.json') or f6['tokenizer_sha256']!=trained['tokenizer_sha256']:
        raise ValueError('Rehearsal tokenizer chain differs')
    if not bc.get('deterministic_algorithms') or not vc.get('deterministic_algorithms') or vc['batch_geometry']!=batch['selected_model_settings']:
        raise ValueError('Exact deterministic selected batch was not validated')
    if native['validation_report_sha256']!=sha256_file(resolve(a.validation_run)/'report.json') or native['tokenizer_sha256']!=trained['tokenizer_sha256']:
        raise ValueError('Native evaluation checkpoint lineage differs')
    if any(f6['category_holdout_tokens'].get(k,0)<50000000 for k in ['ko','en']):raise ValueError('Full actual holdout size insufficient')
    mix_path=resolve(f6['mix']);mix=json.loads(mix_path.read_text())
    if sha256_file(mix_path)!=f6['mix_sha256'] or mix['tokenizer_sha256']!=trained['tokenizer_sha256'] or mix['sampler_version']!=2:
        raise ValueError('Full actual rehearsal mix differs')
    evidence={}
    for root in map(resolve,[a.token_run,a.batch_run,a.validation_run,a.native_eval_run,f5]):
        for name in ['report.json','contract.json','.done']:
            p=root/name;evidence[artifact_path(p)]=sha256_file(p)
    evidence[artifact_path(mix_path)]=sha256_file(mix_path)
    tokenizer=resolve(trained['tokenizer']);prompts=resolve(nc['evaluation']['prompts'])
    if sha256_file(tokenizer)!=trained['tokenizer_sha256'] or sha256_file(prompts)!=nc['evaluation']['prompts_sha256']:
        raise ValueError('Final tokenizer/fixed prompt content changed')
    for path in [tokenizer,prompts]:evidence[artifact_path(path)]=sha256_file(path)
    code={}
    for contract in [c6,bc,vc,nc,tc]:
        for path,digest in contract['code'].items():
            if path in code and code[path]!=digest:raise ValueError('Prerequisite implementation versions differ')
            code[path]=digest
    checkpoint=next(r['published_checkpoint'] for r in validation['results'] if r['model']=='1b')
    forecast=checkpoint['bytes']*13*1.1
    if forecast>176*2**30:raise RuntimeError('Thirteen retained/staging1B checkpoints exceed176GiB rehearsal budget')
    return dict(mix_path=mix_path,mix=mix,settings=vc['batch_geometry'],categories=tc['categories'],tokenizer=trained['tokenizer'],
                token_root=c6['settings']['output'],prompts=nc['evaluation']['prompts'],evidence=evidence,code=code,forecast_checkpoint_bytes=math.ceil(forecast))


def certify_checkpoint(path,output,training_id):
    path=Path(path);meta=json.loads((path/'meta.json').read_text());step=int(path.name.split('_')[1])
    if not (path/'.complete').is_file() or meta['step']!=step or meta['world_size']!=8 or meta['tokens_per_step']!=2097152:
        raise ValueError('Rehearsal checkpoint geometry/marker differs')
    if meta['training_contract']['id']!=training_id or meta['config']['model']!='1b':raise ValueError('Rehearsal checkpoint identity differs')
    if not (path/'.metadata').is_file() or not all((path/f'rng_rank{r}.pt').is_file() for r in range(8)):
        raise ValueError('Rehearsal checkpoint payload/RNG missing')
    if any(p.is_symlink() for p in path.rglob('*')):raise ValueError('Unexpected rehearsal checkpoint symlink')
    def record(p):
        with p.open('rb') as stream:os.fsync(stream.fileno())
        return dict(path=str(p.relative_to(path)),bytes=p.stat().st_size,sha256=sha256_file(p))
    files=[p for p in sorted(path.rglob('*')) if p.is_file()]
    with ThreadPoolExecutor(max_workers=min(8,len(files))) as hashes:records=list(hashes.map(record,files))
    for directory in sorted([path]+[p for p in path.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    result=dict(status='passed_checkpoint_storage',checkpoint=artifact_path(path),step=step,training_contract_id=training_id,files=records,
                bytes=sum(r['bytes'] for r in records),scope='Fresh complete DCP/RNG/meta content SHA and fsync; model/resume tensors validated separately in F7.')
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError('Existing checkpoint storage proof changed')
    else:atomic_json(output,result)
    return result


def recover_checkpoint_prefix(root,verified=None):
    """Reuse only certified checkpoints; preserve unverified complete outputs."""
    from train.runtime import complete_checkpoints
    moved=[]
    for path in complete_checkpoints(root/'training'):
        certificate=root/'checkpoint_certificates'/(path.name+'.json')
        if certificate.exists():
            prior=json.loads(certificate.read_text());result=certify_checkpoint(path,certificate,prior['training_contract_id'])
            if verified is not None:verified[path.name]=result
        else:
            archive=root/'unverified_checkpoints';archive.mkdir(exist_ok=True)
            destination=archive/(path.name+'-'+str(time.time_ns()));path.rename(destination)
            moved.append(dict(original=artifact_path(path),preserved=artifact_path(destination),reason='No completed content/fsync certificate after interruption; preserve and replay from preceding certified checkpoint.'))
    if moved:atomic_json(root/f'recovery_{time.time_ns()}.json',dict(moved=moved))
    return moved


def trace_review(path,config,probe,*,end_step=5000):
    from train.train import lr_at
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()];train={};evaluation={}
    for r in rows:
        step=r['step']
        if type(step) is not int or not 1<=step<=end_step:raise ValueError('Training trace step outside rehearsal')
        if 'loss' in r:
            for k in ['loss','gnorm','lr','tok_s','tokens_b']:
                if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]):raise ValueError('Nonfinite/incomplete rehearsal training trace')
            if r['tok_s']<=0 or not math.isclose(r['tokens_b']*1e9,step*config['global_tokens'],abs_tol=1):raise ValueError('Actual trained token geometry differs')
            if not math.isclose(r['lr'],lr_at(step-1,config),abs_tol=1e-12,rel_tol=1e-10):raise ValueError('Actual warmup/WSD learning rate differs')
            if step in train and any(not math.isclose(r[k],train[step][k],abs_tol=1e-6,rel_tol=1e-5 if k=='gnorm' else 0) for k in ['loss','gnorm','lr']):
                raise ValueError('Replayed rehearsal optimization trajectory changed')
            train[step]=r
        if 'holdout_loss' in r:
            if set(r['by_source'])!={g['source'] for g in probe['groups']} or set(r['by_category'])!={'ko','en','code','math'}:
                raise ValueError('Source/category curve coverage differs')
            nll=0.;tokens=0;categories={}
            for group in probe['groups']:
                v=r['by_source'][group['source']];count=group['count']*probe['seq_len']
                if v['valid_tokens']!=count or not math.isfinite(v['nll_sum']) or not math.isclose(v['holdout_loss'],v['nll_sum']/count,abs_tol=1e-10):
                    raise ValueError('Source curve target/NLL differs')
                nll+=v['nll_sum'];tokens+=count;c=categories.setdefault(group['category'],[0.,0]);c[0]+=v['nll_sum'];c[1]+=count
            if tokens!=r['valid_tokens'] or not math.isclose(r['holdout_loss'],nll/tokens,abs_tol=1e-10):raise ValueError('Aggregate probe is not token weighted')
            for name,(loss_sum,count) in categories.items():
                v=r['by_category'][name]
                if v['valid_tokens']!=count or not math.isclose(v['nll_sum'],loss_sum,abs_tol=1e-8) or not math.isclose(v['holdout_loss'],loss_sum/count,abs_tol=1e-10):
                    raise ValueError('Category curve is not token weighted')
            if step in evaluation and any(not math.isclose(r['by_source'][s]['holdout_loss'],evaluation[step]['by_source'][s]['holdout_loss'],abs_tol=1e-6) for s in r['by_source']):
                raise ValueError('Replayed source curve changed')
            evaluation[step]=r
    required_train=set(range(1,min(4,end_step)+1))|set(range(10,end_step+1,10));required_eval={s for s in [2,4] if s<=end_step}|set(range(250,end_step+1,250))
    if set(train)!=required_train or set(evaluation)!=required_eval:raise ValueError('Incomplete rehearsal training/evaluation trace')
    for step in [4,2500,5000]:
        if step>end_step:continue
        generated=evaluation[step].get('fixed_generations',[])
        if len(generated)!=6 or len({r['id'] for r in generated})!=6 or {r['category'] for r in generated}!={'ko','en'}:
            raise ValueError('Fixed rehearsal generation coverage differs')
    early=[v['loss'] for s,v in train.items() if 250<=s<=500];late=[v['loss'] for s,v in train.items() if 4500<=s<=5000]
    first=evaluation[min(evaluation)];last=evaluation[max(evaluation)]
    return dict(train_points=len(train),eval_points=len(evaluation),trained_tokens=end_step*config['global_tokens'],
                early_median_loss=statistics.median(early) if early else None,late_median_loss=statistics.median(late) if late else None,
                loss_decreased=(statistics.median(late)<statistics.median(early)) if early and late else None,
                first_source_losses={s:v['holdout_loss'] for s,v in first['by_source'].items()},last_source_losses={s:v['holdout_loss'] for s,v in last['by_source'].items()},
                first_category_losses={s:v['holdout_loss'] for s,v in first['by_category'].items()},last_category_losses={s:v['holdout_loss'] for s,v in last['by_category'].items()},
                max_gradient_norm=max(r['gnorm'] for r in train.values()),
                generation_sha256_by_step={str(s):identity(v['fixed_generations']) for s,v in evaluation.items() if 'fixed_generations' in v},
                scope='Actual effective update/source curves. Logged first interval after resume and intervals containing eval/save are not steady-state performance.')


def main(a):
    from scripts.resource_admission import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    from train.runtime import complete_checkpoints
    from train.probe import select_probe,verify_probe
    from train.parallel_contract import native_expected_inputs
    from train.rehearsal import aggregate_ledgers
    root=resolve(a.run_dir);root.resolve().relative_to(resolve('runs').resolve())
    if (root/'.done').exists():return certified(root,'passed_actual_rehearsal_training')
    hw=inspect()
    if len(hw['gpus'])!=8 or hw['memory']['cache_aware_admission']['headroom_gib']<384:raise RuntimeError('Rehearsal8GPU/memory headroom unavailable')
    inputs=prerequisites(a)
    if not root.exists():
        root.mkdir(parents=True);(root/'ledgers').mkdir();(root/'checkpoint_certificates').mkdir();(root/'attempts').mkdir()
        probe=select_probe(inputs['token_root'],root/'probe',inputs['categories'])
        atomic_json(root/'expected_tokens.json',native_expected_inputs(inputs['token_root'],inputs['mix'],probe['holdout']))
        config=make_config(inputs['settings'],inputs['mix'],probe,root/'training');config['mix']=artifact_path(inputs['mix_path'])
        early=dict(config,stop_after=4,eval_every=2,ckpt_every=2,log_every=1)
        atomic_json(root/'early.json',early);atomic_json(root/'main.json',config)
        evidence=dict(inputs['evidence'])
        for p in [root/'probe/probe.json',root/'probe/.done',root/'expected_tokens.json',root/'early.json',root/'main.json',resolve(inputs['prompts'])]:evidence[artifact_path(p)]=sha256_file(p)
        deps=['scripts/run_rehearsal.py','train/rehearsal.py','train/parallel_contract.py','train/deterministic_train.py','train/probe.py','train/train.py','train/runtime.py','train/data.py',
              'train/__init__.py','model/model.py','scripts/run_native_evaluation.py','scripts/validate_actual_training.py','scripts/hardware_profile.py','scripts/resource_admission.py','common/data_contracts.py']
        scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(parents=True,exist_ok=False)
        code=dict(inputs['code']);code.update({p:sha256_file(resolve(p)) for p in deps})
        contract=dict(version=1,hardware=hw,code=code,evidence=evidence,categories=inputs['categories'],
                      tokenizer=inputs['tokenizer'],prompts=inputs['prompts'],forecast_checkpoint_bytes=inputs['forecast_checkpoint_bytes'],
                      cpu_affinity=hw['cpu']['affinity'][64:] or hw['cpu']['affinity'],scratch=str(scratch),host_rss_gib=256,local_budget_gib=32,
                      durable_budget_gib=176,reserve_gib=200,max_seconds=172800,
                      scope='Actual1B983.64M parameter10,485,760,000-token rehearsal only;3B main training requires separate confirmation. Final native PPL and readiness review follow.')
        atomic_json(root/'contract.json',contract)
    contract=json.loads((root/'contract.json').read_text())
    for p,h in {**contract['code'],**contract['evidence']}.items():
        if sha256_file(resolve(p))!=h:raise ValueError('Rehearsal resume implementation/input differs:'+p)
    probe=verify_probe(root/'probe');config=json.loads((root/'main.json').read_text());scratch=Path(contract['scratch'])
    scratch.mkdir(parents=True,exist_ok=True)
    remaining=max(0,contract['forecast_checkpoint_bytes']-size_tree(root/'training'))
    if shutil.disk_usage(root).free<remaining+200*2**30:raise RuntimeError('Rehearsal retained/staging/reserve budget unavailable')
    control=(root/'controller.lock').open('a');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=resolve('runs/.gpu_benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU compute already active')
    (root/'pid').write_text(str(os.getpid()));atomic_json(root/f'hardware_{time.time_ns()}.json',hw)
    env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(scratch),PYTHONUNBUFFERED='1')
    cpus=','.join(map(str,contract['cpu_affinity']));start=time.monotonic();proc=None;peak=0;futures={};verified={};pool=ThreadPoolExecutor(max_workers=1)
    def inspect_checkpoints():
        cp=root/'training/training_contract.json'
        if not cp.exists():return
        training_id=json.loads(cp.read_text())['id']
        for p in complete_checkpoints(root/'training'):
            if p.name not in futures and p.name not in verified:futures[p.name]=pool.submit(certify_checkpoint,p,root/'checkpoint_certificates'/(p.name+'.json'),training_id)
        for name,future in list(futures.items()):
            if future.done():verified[name]=future.result();del futures[name]
    def phase(name,path):
        nonlocal proc,peak
        attempt=name+'-'+str(time.time_ns());cmd=['taskset','-c',cpus,'torchrun','--standalone','--nproc_per_node=8','--module','train.rehearsal','--run-dir',str(root),'--config',str(path),'--attempt',attempt]
        record=dict(phase=name,attempt=attempt,config_sha256=sha256_file(path),before_checkpoints=[p.name for p in complete_checkpoints(root/'training')]);atomic_json(root/'attempts'/(attempt+'.json'),record)
        with (root/'attempts'/(attempt+'.log')).open('w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            peak=max(peak,process_tree_rss(os.getpid()));elapsed=time.monotonic()-start;inspect_checkpoints()
            atomic_json(root/'state.json',dict(phase='running',case=name,attempt=attempt,worker_pid=proc.pid,elapsed_seconds=elapsed,peak_rss_bytes=peak,completed_checkpoints=len(verified)))
            if elapsed>172800 or peak>256*2**30:raise RuntimeError('Rehearsal time/RSS budget')
            for p,budget in [(root,176),(scratch,32)]:
                if size_tree(p)>budget*2**30 or shutil.disk_usage(p).free<200*2**30:raise RuntimeError('Rehearsal disk budget/reserve')
            time.sleep(2)
        record.update(returncode=proc.returncode);atomic_json(root/'attempts'/(attempt+'.json'),record)
        if proc.returncode:raise RuntimeError('Rehearsal phase failed:'+attempt)
        inspect_checkpoints()
    try:
        recover_checkpoint_prefix(root,verified)
        checkpoints=complete_checkpoints(root/'training');latest=int(checkpoints[-1].name.split('_')[1]) if checkpoints else 0
        if latest<4:phase('early',root/'early.json')
        if latest<5000:
            ep=root/'early_validation.json';ed=root/'early_validation.done'
            if ed.exists():
                if json.loads(ed.read_text())!=dict(report_sha256=sha256_file(ep)):raise ValueError('Early full-mix verification proof changed')
                early_result=json.loads(ep.read_text())
                if early_result['status']!='passed_actual_full_mix_probe' or early_result['counts']['contract_id']!=json.loads((root/'training/training_contract.json').read_text())['id']:
                    raise ValueError('Early full-mix verification identity changed')
            else:
                early_curves=trace_review(root/'training/log.jsonl',config,probe,end_step=4)
                early_counts=aggregate_ledgers(list((root/'ledgers').glob('*.json')),world=8,steps=4,seq_len=4096,global_tokens=2097152,sources=inputs['mix']['sources'])
                if early_counts['generation_sha256_by_step']!=early_curves['generation_sha256_by_step']:raise ValueError('Early generation log/rank receipts differ')
                atomic_json(ep,dict(status='passed_actual_full_mix_probe',counts=early_counts,
                            scope='Four real1B updates with raw full-mix source probes, all-rank fixed-generation agreement and preserved checkpoints. Not10B completion.'))
                atomic_json(ed,dict(report_sha256=sha256_file(ep)))
            phase('main',root/'main.json')
        inspect_checkpoints()
        for name,future in futures.items():verified[name]=future.result()
        expected={f'ckpt_{s:07d}' for s in [2,4,*range(500,5001,500)]}
        if {p.name for p in complete_checkpoints(root/'training')}!=expected:raise ValueError('Retained checkpoint coverage differs; never prune automatically')
        counts=aggregate_ledgers(list((root/'ledgers').glob('*.json')),world=8,steps=5000,seq_len=4096,global_tokens=2097152,sources=inputs['mix']['sources'])
        curves=trace_review(root/'training/log.jsonl',config,probe)
        if counts['generation_sha256_by_step']!=curves['generation_sha256_by_step']:raise ValueError('Fixed generation trace/rank receipts differ')
        for p,h in {**contract['code'],**contract['evidence']}.items():
            if sha256_file(resolve(p))!=h:raise ValueError('Rehearsal input/code changed')
        receipts={artifact_path(p):sha256_file(p) for p in [root/'training/log.jsonl',root/'training/training_contract.json',*sorted((root/'ledgers').glob('*.json')),*sorted((root/'checkpoint_certificates').glob('*.json')),*sorted(root.glob('*-input_hash.json'))]}
        report=dict(status='passed_actual_rehearsal_training' if curves['loss_decreased'] else 'needs_review',counts=counts,curves=curves,evidence=receipts,
                    final_checkpoint=artifact_path(root/'training/ckpt_0005000'),elapsed_seconds=time.monotonic()-start,peak_host_rss_bytes=peak,
                    actual_durable_bytes=size_tree(root),retained_checkpoints=sorted(expected),scope=contract['scope'],
                    remaining=['Review source/category and fixed-generation trajectories','Native final1B checkpoint full-target ko/en PPL','Final3B readiness report; do not start main training'])
        atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')));atomic_json(root/'state.json',dict(phase='complete'))
    except BaseException as exc:
        if proc is not None and proc.poll() is None:stop(proc)
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise
    finally:pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    p.add_argument('--token-run',default='runs/f6_tokens_v1');p.add_argument('--batch-run',default='runs/actual_deterministic_batch_v1')
    p.add_argument('--validation-run',default='runs/actual_train_validation_v1');p.add_argument('--native-eval-run',default='runs/native_eval_validation_v1')
    main(p.parse_args())
