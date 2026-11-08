"""Prepare late CPU input verification, then evaluate only the completed main3B.

Uses the unchanged native FP32 evaluator and fixed raw holdout. No training,
checkpoint mutation, pruning, official benchmark, SFT or upload. Preparation
and the tiny CPU tests are not actual final3B evaluation evidence.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file, identity
from scripts.run_native_evaluation import certified
from scripts.evaluate_final_rehearsal import check_bindings, acquire_idle_gpu

GIB=2**30


def process_live(pid, token, root):
    try:
        p=Path('/proc')/str(pid)
        return ((p/'stat').read_text().split(') ',1)[1].split()[0]!='Z' and
                token.encode() in (p/'cmdline').read_bytes() and str(root).encode() in (p/'cmdline').read_bytes())
    except FileNotFoundError:
        return False


def main_observation(root):
    root=resolve(root)
    if (root/'.done').exists():
        return dict(complete=True,ready=True,step=57000,scope='Completion marker triggers full audit; marker alone is not acceptance.')
    state=json.loads((root/'state.json').read_text())
    if state.get('phase') in ('error','failed','paused'):
        raise RuntimeError('Main training requires recovery; preserve evaluation preparation')
    pid=int((root/'pid').read_text());worker=state.get('worker_pid')
    # Controller CLI uses a relative run path, worker CLI an absolute path.
    if not process_live(pid,'supervise_main_training.py',Path(root.name)) or not worker or not process_live(worker,'train.monitored',root):
        if (root/'.done').exists():
            return dict(complete=True,ready=True,step=57000)
        raise RuntimeError('Actual main controller/worker handle missing; inspect, never restart from this evaluator')
    path=root/'training/log.jsonl'
    raw=path.read_bytes() if path.exists() else b''
    lines=raw.splitlines()
    if raw and not raw.endswith(b'\n'):
        lines=lines[:-1]
    rows=[json.loads(line) for line in lines if line.strip()]
    train=[r for r in rows if 'loss' in r]
    step=train[-1]['step'] if train else 0
    if type(step) is not int or not 0<=step<=57000:
        raise ValueError('Main observed step outside contract')
    return dict(complete=False,ready=step>=56500,step=step,
                scope='CPU prewarm begins at56500; GPU scoring requires complete57000 audit and idle GPU lock.')


def completed_main(root):
    from scripts.audit_main_training import audit_run
    audit=audit_run(root,57000)
    if (audit['validation_source'] or audit['counts']['committed_tokens']!=119537664000 or
            audit['counts']['successful_target_ranks']!=list(range(8))):
        raise ValueError('Final3B requires actual full8rank target coverage')
    checkpoint=resolve(root)/'training/ckpt_0057000'
    proof=json.loads(resolve(audit['checkpoint_certificate']).read_text())
    meta=json.loads((checkpoint/'meta.json').read_text())
    if meta['config']['model']!='3b' or meta['step']!=57000:
        raise ValueError('Final3B model/step identity differs')
    return checkpoint,proof,meta,audit


def verify_result(result, proof, tokenizer, expected_targets, reference, *, expected_step, expected_device):
    expected={r['path']:r['sha256'] for r in proof['files']}
    if len(expected)!=len(proof['files']) or result['contract']['checkpoint']!=expected:
        raise ValueError('Final evaluated checkpoint content differs')
    c=result['contract'];ref=reference['contract']
    if result['contract_id']!=identity(c) or result['status']!='passed' or result['checkpoint_step']!=expected_step:
        raise ValueError('Final result identity/step differs')
    if (c['input_provenance'].get('level')!='verified-eval-split' or c['input_provenance']!=ref['input_provenance'] or
            c['device']!=expected_device):
        raise ValueError('Final result provenance/device differs')
    for key in ['tokenizer_sha256','samples_sha256','prompts_sha256','seq_len','stride','max_new_tokens',
                'seed','temperature','template','append_eos_for_ppl','torch_version','code']:
        if c[key]!=ref[key]:
            raise ValueError('Fixed native evaluation policy differs: '+key)
    if set(result['languages'])!={'ko','en'} or set(expected_targets)!={'ko','en'}:
        raise ValueError('Both native languages required')
    def weighted(value,count):
        if type(value['valid_tokens']) is not int or count<1 or value['valid_tokens']!=count:
            raise ValueError('Full target coverage differs')
        nll=value['nll_sum'];mean=value['mean_nll']
        if not math.isfinite(nll) or nll<0 or not math.isfinite(mean) or not math.isclose(mean,nll/count,rel_tol=1e-12):
            raise ValueError('Native weighted NLL differs')
        overflow=mean>math.log(sys.float_info.max)
        if value['perplexity_overflow'] is not overflow or (overflow and value['perplexity'] is not None):
            raise ValueError('Native PPL overflow differs')
        if not overflow and (not isinstance(value['perplexity'],(int,float)) or not math.isclose(value['perplexity'],math.exp(mean),rel_tol=1e-12)):
            raise ValueError('Native PPL differs')
    for language,count in expected_targets.items():
        weighted(result['languages'][language],count)
    weighted(result['aggregate'],sum(expected_targets.values()))
    if not math.isclose(result['aggregate']['nll_sum'],sum(v['nll_sum'] for v in result['languages'].values()),rel_tol=1e-12):
        raise ValueError('Aggregate NLL sum differs')
    prompts=[(g['id'],g['category'],g['prompt']) for g in result['generations']]
    if (prompts!=[(g['id'],g['category'],g['prompt']) for g in reference['generations']] or
            len(prompts)!=6 or len({g[0] for g in prompts})!=6 or {g[1] for g in prompts}!={'ko','en'}):
        raise ValueError('Fixed6 prompt coverage differs')
    eos=tokenizer.token_to_id('<|eos|>')
    for generated in result['generations']:
        ids=generated['generated_ids']
        if (not 1<=len(ids)<=c['max_new_tokens'] or
                any(type(i) is not int or not 0<=i<tokenizer.get_vocab_size() for i in ids) or
                generated['stopped_at_eos'] is not (ids[-1]==eos) or eos in ids[:-1] or
                (ids[-1]!=eos and len(ids)!=c['max_new_tokens']) or
                generated['generated_text']!=tokenizer.decode(ids,skip_special_tokens=False)):
            raise ValueError('Native generation token/EOS/decode differs')
        if generated['prompt_ids']!=tokenizer.encode(generated['prompt'],add_special_tokens=False).ids:
            raise ValueError('Native prompt token IDs differ')
        for name,expected_value in [('seed',c['seed']),('temperature',c['temperature']),
                                    ('context_length',c['seq_len']),('max_new_tokens',c['max_new_tokens'])]:
            if generated[name]!=expected_value:
                raise ValueError('Native generation policy differs: '+name)


def make_contract(main_run,reference_run):
    main=resolve(main_run);c=json.loads((main/'contract.json').read_text())
    if not c.get('main_training_authorized') or not c.get('new_checkpoint_pruning_authorized'):
        raise ValueError('Main approval is required before final evaluation setup')
    config=json.loads((main/'main.json').read_text())
    if c['configurations']['main.json']!=sha256_file(main/'main.json') or config['model']!='3b' or config['max_steps']!=57000:
        raise ValueError('Unreviewed main evaluation target')
    ref,rc=certified(reference_run,'passed_actual_trained_native_evaluation')
    reference=resolve(reference_run)/ref['result']
    if sha256_file(reference)!=ref['result_sha256']:
        raise ValueError('Reference final1B native result changed')
    check_bindings(rc)
    old=json.loads(reference.read_text())
    if old['contract']['tokenizer_sha256']!=sha256_file(resolve(c['tokenizer'])):
        raise ValueError('Main/reference tokenizer differs')
    files=[main/'contract.json',main/'main.json',main/'authorization.json',
           resolve(reference_run)/'contract.json',resolve(reference_run)/'report.json',resolve(reference_run)/'.done',reference]
    code={**c['code'],**{p:sha256_file(resolve(p)) for p in
         ['scripts/evaluate_final_main.py','scripts/audit_main_training.py']}}
    inputs={**rc['inputs'],**{artifact_path(p):sha256_file(p) for p in files}}
    return dict(version=1,purpose='final_native_main_evaluation',main_run=artifact_path(main),
        evaluation=rc['evaluation'],reference_report=artifact_path(reference),tokenizer=c['tokenizer'],
        tokenizer_sha256=old['contract']['tokenizer_sha256'],samples=rc['samples'],holdout_manifest=rc['holdout_manifest'],
        expected_targets={k:v['valid_tokens'] for k,v in old['languages'].items()},code=code,inputs=inputs,framework=c['framework'],
        late_start_step=56500,final_step=57000,max_seconds=22*86400,worker_max_seconds=8*3600,
        host_rss_gib=256,local_budget_gib=8,durable_budget_gib=8,reserve_gib=200,
        cpu_affinity=list(range(64)),sqlite_cache_mib=131072,hash_workers=32,
        scope='Only completed3B57000 native FP32 fixed-holdout ko/en PPL and6generations. No model update, official benchmark or quality-success assertion.')


def admission(contract,root):
    from scripts.resource_admission import inspect
    from scripts.run_local_split import size_tree
    hw=inspect();main=resolve(contract['main_run'])
    mc=json.loads((main/'contract.json').read_text());p=mc['tiered_retention']
    if hw['gpus']!=mc['gpu_identity'] or hw['memory']['cache_aware_admission']['headroom_gib']<512:
        raise RuntimeError('Final eval256GiB plus main future peak headroom unavailable')
    nfs_remaining=max(0,p['nfs_budget_bytes']-size_tree(main))
    local_remaining=max(0,p['local_budget_bytes']-size_tree(Path(p['local_root'])))
    temp_remaining=max(0,p['temporary_budget_bytes']-size_tree(Path(p['temp_root'])))
    own_remaining=max(0,contract['durable_budget_gib']*GIB-size_tree(root))
    if shutil.disk_usage(root).free<nfs_remaining+own_remaining+p['other_nfs_future_bytes']+p['reserve_bytes']:
        raise RuntimeError('Final evaluation plus main NFS peak/reserve unavailable')
    scratch=Path(p['temp_root'])/'final_native'
    if shutil.disk_usage(Path(hw['storage']['scratch_dir'])).free<local_remaining+temp_remaining+p['reserve_bytes']:
        raise RuntimeError('Final evaluation/main local peak/reserve unavailable')
    if Path(p['temp_root'])!=Path(hw['storage']['scratch_dir'])/main.name:
        raise ValueError('Main reserved temporary path differs')
    return hw,scratch


def reject_live_orphan(root):
    path=root/'worker_state.json'
    if path.exists():
        pid=json.loads(path.read_text()).get('pid')
        if pid and process_live(pid,'evaluate_final_main.py',root):
            raise RuntimeError('Existing final3B evaluation worker still live; preserve before controller recovery')


def worker(root):
    root=resolve(root);c=json.loads((root/'contract.json').read_text());check_bindings(c)
    def notify(phase,**kwargs):
        atomic_json(root/'worker_state.json',dict(phase=phase,pid=os.getpid(),unix=time.time(),**kwargs))
    import torch
    import pyarrow as pa
    from tokenizers import Tokenizer
    from eval.prepare_holdout import local_verified_inputs
    from data.split_inputs import verify_prepared
    from eval.run import run,records
    if c['framework']!=dict(torch=str(torch.__version__),cuda=torch.version.cuda):
        raise ValueError('Final evaluator framework changed')
    pa.set_cpu_count(4);pa.set_io_thread_count(4)
    e=c['evaluation'];started=time.monotonic();notify('verify_fresh_local_eval_context')
    with local_verified_inputs(e['source_manifest'],e['canonical_split_manifest'],e['reader_manifest'],
                              cache_mib=c['sqlite_cache_mib'],hash_workers=c['hash_workers']) as inputs:
        prepared=verify_prepared(c['holdout_manifest'],'eval',c['samples'],verified_inputs=inputs)
        tokenizer=Tokenizer.from_file(str(resolve(c['tokenizer'])))
        samples=records(resolve(c['samples']))
        expected={k:sum(len(tokenizer.encode(r['text'],add_special_tokens=False).ids) for r in samples if r['category']==k) for k in ['ko','en']}
        if expected!=c['expected_targets']:
            raise ValueError('Fresh raw-text native target counts changed')
        atomic_json(root/'fresh_context.json',dict(seconds=time.monotonic()-started,prepared_contract_id=prepared['contract_id'],
                    expected_targets=expected,scope='Fresh physical verification in this live process; no persistent skip-validation cache.'))
        while not (resolve(c['main_run'])/'.done').exists():
            observation=main_observation(c['main_run']);notify('await_main_final_checkpoint',**observation);time.sleep(30)
        notify('audit_main_actual_target')
        checkpoint,proof,meta,audit=completed_main(c['main_run'])
        atomic_json(root/'final_training_audit.json',audit)
        if meta['training_contract']['mix']['tokenizer_sha256']!=c['tokenizer_sha256']:
            raise ValueError('Final main checkpoint tokenizer changed')
        check_bindings(c)
        with acquire_idle_gpu(notify):
            os.environ['CUDA_VISIBLE_DEVICES']='0';notify('native_final3b_evaluation');started=time.monotonic()
            result=run(checkpoint,resolve(c['tokenizer']),resolve(c['samples']),resolve(e['prompts']),root/'3b_final.json',
                seq_len=4096,stride=2048,max_new_tokens=64,seed=e['seed'],temperature=0,device='cuda',
                holdout_manifest=resolve(c['holdout_manifest']),verified_inputs=inputs)
            reference=json.loads(resolve(c['reference_report']).read_text())
            verify_result(result,proof,tokenizer,expected,reference,expected_step=57000,expected_device='cuda')
            seconds=time.monotonic()-started
    check_bindings(c)
    report=dict(status='passed_actual_final3b_native_evaluation',result='3b_final.json',result_sha256=sha256_file(root/'3b_final.json'),
        final_training_audit_sha256=sha256_file(root/'final_training_audit.json'),main_report_sha256=sha256_file(resolve(c['main_run'])/'report.json'),
        languages=result['languages'],aggregate=result['aggregate'],elapsed_evaluation_seconds=seconds,
        scope='Actual final3B native evaluation execution/coverage/content passed. Generated quality review,8GPU final restore and retention audit remain separate. Fixed1466document sample, not all holdout or official benchmarks.')
    atomic_json(root/'report.json',report)
    atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
    notify('complete')


def main(a):
    if a.worker:
        return worker(a.run_dir)
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    root=resolve(a.run_dir);root.relative_to(resolve('runs'))
    if root.is_relative_to(resolve(a.main_run)):
        raise ValueError('Evaluation must use an independent run')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=make_contract(a.main_run,a.reference_run)
        cp=root/'contract.json'
        if cp.exists() and json.loads(cp.read_text())!=c:
            raise ValueError('Final evaluator contract changed; use a new independent run')
        if not cp.exists():
            atomic_json(cp,c)
        if a.prepare:
            atomic_json(root/'prepared.json',dict(status='prepared_not_executed',contract_sha256=sha256_file(cp),
                command=[sys.executable,'scripts/evaluate_final_main.py','--run-dir',artifact_path(root),'--main-run',a.main_run,'--reference-run',a.reference_run,'--run'],
                scope='Preparation only; waits for main56500 before CPU work and audited57000 before GPU scoring.'))
            return
        if (root/'.done').exists():
            return certified(root,'passed_actual_final3b_native_evaluation')
        reject_live_orphan(root);check_bindings(c)
        (root/'pid').write_text(str(os.getpid())+'\n')
        started=time.monotonic();proc=None;peak=0;attempt=str(time.time_ns())
        try:
            while True:
                observation=main_observation(c['main_run'])
                if observation['ready']:
                    break
                atomic_json(root/'state.json',dict(phase='running',case='await_late_main',**observation))
                if time.monotonic()-started>c['max_seconds']:
                    raise RuntimeError('Final evaluation late-trigger time budget')
                time.sleep(30)
            hw,scratch=admission(c,root);atomic_json(root/('hardware_'+attempt+'.json'),hw)
            scratch.mkdir(parents=True,exist_ok=True)
            cpus=sorted(set(c['cpu_affinity'])&set(hw['cpu']['affinity']))
            if not cpus:
                raise RuntimeError('Final evaluation CPU affinity unavailable')
            env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='8',TMPDIR=str(scratch),SQLITE_TMPDIR=str(scratch),PYTHONUNBUFFERED='1')
            command=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),'--run-dir',str(root),'--worker']
            worker_start=time.monotonic()
            with (root/('console_'+attempt+'.log')).open('x') as log:
                proc=subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                peak=max(peak,process_tree_rss(os.getpid()))
                w=json.loads((root/'worker_state.json').read_text()) if (root/'worker_state.json').exists() else {}
                atomic_json(root/'state.json',dict(phase='running',case=w.get('phase','starting'),worker_pid=proc.pid,peak_rss_bytes=peak,elapsed_seconds=time.monotonic()-started))
                if peak>c['host_rss_gib']*GIB or time.monotonic()-worker_start>c['worker_max_seconds']:
                    raise RuntimeError('Final evaluator worker RSS/time budget')
                for path,budget in [(root,c['durable_budget_gib']),(scratch,c['local_budget_gib'])]:
                    if size_tree(path)>budget*GIB or shutil.disk_usage(path).free<c['reserve_gib']*GIB:
                        raise RuntimeError('Final evaluator disk/reserve budget')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError('Final3B evaluator worker failed; preserve evidence and inspect console')
            report,_=certified(root,'passed_actual_final3b_native_evaluation')
            if report['result_sha256']!=sha256_file(root/report['result']):
                raise ValueError('Final result content changed')
            atomic_json(root/'state.json',dict(phase='complete',peak_rss_bytes=peak))
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                stop(proc)
            atomic_json(root/'state.json',dict(phase='error',error=str(exc)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--main-run',default='runs/3b_full_v1')
    p.add_argument('--reference-run',default='runs/rehearsal_final_native_v1')
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    main(p.parse_args())
