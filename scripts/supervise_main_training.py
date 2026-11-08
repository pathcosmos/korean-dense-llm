"""Prepare an unauthorized3B bundle or supervise separately authorized training.

Supports latest NFS checkpoints with disposable local history, or legacy
local-latest retention. No self-authorization, cloud upload or corpus deletion.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve,artifact_path,atomic_json,sha256_file
from scripts.run_native_evaluation import certified

GIB=2**30


def require_authorization(contract):
    if contract.get('purpose')!='main_training' or contract.get('main_training_authorized') is not True:
        raise ValueError('Main training is not authorized; separate user confirmation required')
    if contract.get('new_checkpoint_pruning_authorized') is not True:
        raise ValueError('Local new-run checkpoint retention is not authorized')
    if contract.get('retention_policy') == 'nfs_latest_with_ephemeral_history':
        p = contract.get('tiered_retention', {})
        from scripts.tiered_checkpoint_retention import reviewed_policy
        if (not reviewed_policy(p) or
                not p.get('native_pruning_disabled') or p.get('resume_from_local') is not False or
                contract.get('archive_required_before_prune') is not True):
            raise ValueError('Unreviewed tiered checkpoint retention policy')
        if p.get('version') == 2:
            revision=contract.get('retention_revision',{})
            authorization=resolve(revision['authorization'])
            if sha256_file(authorization)!=revision['authorization_sha256']:
                raise ValueError('Retention revision authorization changed')
            approved=json.loads(authorization.read_text())
            if (approved.get('authorized') is not True or approved.get('nfs_keep')!=3 or
                    approved.get('local_max_keep')!=15 or not approved.get('adaptive_local_oldest_first') or
                    resolve(approved['management_run'])!=Path(p['nfs_run']) or approved['local_namespace']!=p['local_root']):
                raise ValueError('Retention revision scope is not authorized')
    elif contract.get('retention_policy')!='local_latest' or contract.get('archive_required_before_prune'):
        raise ValueError('This supervisor does not implement archive-ack retention')


def memory_violation(rss_bytes,pss_bytes,limit_bytes):
    if any(type(x) is not int or x<0 for x in [rss_bytes,pss_bytes,limit_bytes]) or limit_bytes<1 or pss_bytes>rss_bytes:
        raise ValueError('Invalid memory accounting')
    return pss_bytes>limit_bytes


def validated_copy_settings():
    """Bind the actually measured copy path, not merely the capacity proposal."""
    root=resolve('runs/tiered_retention_copy_v2')
    if not (root/'.done').exists():root=resolve('runs/tiered_retention_copy_v1')
    report=json.loads((root/'report.json').read_text())
    marker=json.loads((root/'.done').read_text())
    if (marker!={'report_sha256':sha256_file(root/'report.json')} or
            report['status']!='passed_actual_checkpoint_copy_and_cpu_state_restore' or
            report['cpu_state_exact'] is not True or report['all8_original_cpu_cuda_rng_exact'] is not True or
            report['cpu_restored_tensors']<1000 or report['selected_copy_workers'] not in (4,8)):
        raise ValueError('Actual tiered copy/CPU restore evidence missing or changed')
    for path,digest in json.loads((root/'code.json').read_text()).items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Measured copy implementation changed')
    for path,digest in report['evidence'].items():
        if sha256_file(resolve(path))!=digest:raise ValueError('Measured copy input evidence changed')
    return report['selected_copy_workers'],[root/'report.json',root/'.done',root/'code.json']


def validate_optimizer_map(metadata):
    """Reject a complete marker around model-only or partial AdamW state."""
    values=metadata.state_dict_metadata
    model={k[6:]:v for k,v in values.items() if k.startswith('model.')}
    if not model:raise ValueError('Main DCP model/optimizer state missing')
    for name,value in model.items():
        for field in ['exp_avg','exp_avg_sq','step']:
            key='optim.state.'+name+'.'+field
            item=values.get(key)
            if item is None or not hasattr(item,'size') or not hasattr(value,'size'):
                raise ValueError('Main DCP optimizer state missing: '+key)
            if (field!='step' and item.size!=value.size) or (field=='step' and math.prod(item.size)!=1):
                raise ValueError('Main DCP optimizer state shape differs: '+key)


def seal(path,output,training_id):
    import torch.distributed.checkpoint as dcp
    meta=json.loads((path/'meta.json').read_text());step=int(path.name.split('_')[1])
    if (not (path/'.complete').is_file() or meta['step']!=step or meta['world_size']!=8 or
            meta['tokens_per_step']!=2097152 or meta['training_contract']['id']!=training_id or
            meta['config']['model']!='3b' or meta['config']['max_steps']!=57000 or
            not all((path/f'rng_rank{r}.pt').is_file() for r in range(8))):
        raise ValueError('Main checkpoint identity differs')
    paths=sorted(path.rglob('*'))
    if any(p.is_symlink() for p in paths):raise ValueError('Main checkpoint symlink rejected')
    metadata=dcp.FileSystemReader(path).read_metadata();validate_optimizer_map(metadata)
    storage=metadata.storage_data
    if not storage:raise ValueError('Main DCP payload map missing')
    for item in storage.values():
        p=path/item.relative_path
        if not p.resolve().is_relative_to(path.resolve()) or not p.is_file() or p.stat().st_size<item.offset+item.length:
            raise ValueError('Main DCP payload missing or truncated')
    def record(p):
        before=p.stat()
        with p.open('rb') as f:os.fsync(f.fileno())
        digest=sha256_file(p);after=p.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino):
            raise ValueError('Main checkpoint changed while hashing')
        return dict(path=str(p.relative_to(path)),bytes=after.st_size,sha256=digest)
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(record,[p for p in paths if p.is_file()]))
    for directory in sorted([path]+[p for p in paths if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
        fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    result=dict(status='passed_main_checkpoint_storage',step=step,training_contract_id=training_id,
                checkpoint=artifact_path(path),files=records,bytes=sum(r['bytes'] for r in records))
    if output.exists():
        if json.loads(output.read_text())!=result:raise ValueError('Main checkpoint certificate changed')
    else:
        from scripts.tiered_checkpoint_retention import atomic_json as durable_json
        durable_json(output,result)
    return result


def prepare(a,root):
    from train.probe import select_probe
    from train.parallel_contract import native_expected_inputs
    review,rc=certified(a.review_run,'assembled_for_final_review')
    if review['concerns']:raise ValueError('Final review concerns need resolution before preparing main execution')
    proposal=resolve(a.review_run)/'main_proposal.json'
    if review['evidence'].get(artifact_path(proposal))!=sha256_file(proposal):raise ValueError('Reviewed proposal changed')
    plan=json.loads(proposal.read_text());config=dict(plan['candidate'],keep_ckpts=a.keep_checkpoints)
    history=getattr(a,'local_history',0)
    tiered=a.keep_checkpoints==10 and history==15
    if not tiered and (a.keep_checkpoints not in (2,3) or history):
        raise ValueError('Review latest10 NFS plus15 local history, or legacy latest2/3')
    # The native trainer must NEVER delete an older source before local ACK.
    if tiered:config['keep_ckpts']=config['max_steps']+1
    root.mkdir(parents=True,exist_ok=False)
    source=resolve(rc['sources']['rehearsal']);f8=json.loads((source/'contract.json').read_text())
    f6=json.loads(resolve('runs/f6_tokens_v1/report.json').read_text());c6=json.loads(resolve('runs/f6_tokens_v1/contract.json').read_text())
    f5c=json.loads(resolve('runs/f5_tokenizer_v1/contract.json').read_text())
    probe=select_probe(c6['settings']['output'],root/'probe',f5c['categories'])
    config.update(out_dir=artifact_path(root/'training'),holdout=probe['holdout'])
    atomic_json(root/'main.json',config)
    mix=json.loads(resolve(config['mix']).read_text())
    atomic_json(root/'expected_tokens.json',native_expected_inputs(c6['settings']['output'],mix,probe['holdout']))
    for directory in ['ledgers','attempts','checkpoint_certificates']:(root/directory).mkdir()
    monitored=resolve(rc['sources']['monitored']);mc=json.loads((monitored/'contract.json').read_text())
    framework=json.loads(resolve('runs/actual_train_validation_v1/contract.json').read_text())['framework']
    code={**mc['code'],**{p:sha256_file(resolve(p)) for p in ['scripts/supervise_main_training.py','scripts/process_accounting.py','scripts/tiered_checkpoint_retention.py']}}
    files=[proposal,resolve(a.review_run)/'report.json',resolve(a.review_run)/'contract.json',resolve(a.review_run)/'.done',root/'expected_tokens.json']
    code.update(rc['code'])
    baseline=plan['rolling2_forecast']['measured_checkpoint_bytes']
    durable=math.ceil((a.keep_checkpoints+1)*baseline*1.1)+GIB
    policy=None
    if tiered:
        from scripts.tiered_checkpoint_retention import make_policy
        from scripts.resource_admission import inspect
        hw=inspect();atomic_json(root/'retention_preparation_hardware.json',hw)
        # Assembly is gated by completed F8/native/monitored3B. Those outputs
        # already occupy measured free space, so do not reserve their old peaks
        # again. Fresh runtime admission still protects both devices' reserves.
        policy=make_policy(root,Path(hw['storage']['scratch_dir']),baseline,
                    other_nfs_future_bytes=0,run_id=uuid.uuid4().hex)
        policy['copy_workers'],copy_evidence=validated_copy_settings()
        files+=copy_evidence
        draft=getattr(a,'retention_plan',None)
        if draft:
            draft=resolve(draft);value=json.loads(draft.read_text())
            expected=value['policy']
            if (expected['nfs_run']!=str(root) or expected['nfs_keep']!=10 or expected['local_keep']!=15 or
                    expected['expected_checkpoint_bytes']!=baseline or expected['copy_workers']!=policy['copy_workers'] or
                    Path(expected['local_root'])!=Path(hw['storage']['scratch_dir'])/'checkpoint_history'/root.name/expected['run_id']):
                raise ValueError('Reviewed tiered draft differs from final execution settings')
            policy['run_id']=expected['run_id'];policy['local_root']=expected['local_root']
            files.append(draft)
        durable=policy['nfs_budget_bytes']
    contract=dict(version=1,purpose='main_training',main_training_authorized=False,new_checkpoint_pruning_authorized=False,
                  retention_policy='nfs_latest_with_ephemeral_history' if tiered else 'local_latest',
                  archive_required_before_prune=tiered,tiered_retention=policy,
                  configurations={'main.json':sha256_file(root/'main.json')},code=code,
                  evidence={**mc['evidence'],**{artifact_path(p):sha256_file(p) for p in files}},
                  prompts=mc['prompts'],tokenizer=mc['tokenizer'],generation_steps=[250,2500,5000,*range(10000,55001,5000),57000],
                  memory_metric='process_tree_pss',max_pss_gib=384,durable_budget_bytes=durable,
                  framework=framework,
                  gpu_identity=plan['hardware']['gpus'],
                  local_budget_gib=1024 if tiered else 32,reserve_gib=200,max_seconds=21*86400,
                  expected_checkpoint_bytes=baseline,scope='Separately authorized new3B run only. Local history is disposable; resume is NFS-only. No cloud archive or corpus deletion.')
    atomic_json(root/'contract.json',contract)
    instructions=dict(status='prepared_not_authorized',run_dir=artifact_path(root),config=artifact_path(root/'main.json'),
                  requires=['Final generated-sample/curve and operational review','Separate user confirmation to start3B and prune only new-run checkpoints'],
                  command_after_confirmation=[sys.executable,'scripts/supervise_main_training.py','--run-dir',artifact_path(root),'--run'],
                  recovery='Same frozen supervisor command; fresh verification of retained certificates before latest certified resume. Preserve unverified/incomplete outputs.',
                  retention=policy,
                  marker_scope='Preparation only,not a completed model or training authorization.')
    atomic_json(root/'prepared.json',instructions)
    return instructions


def run(root, adoption=None):
    c=json.loads((root/'contract.json').read_text());require_authorization(c)
    if (root/'state.json').exists():
        pid=json.loads((root/'state.json').read_text()).get('worker_pid')
        if pid:
            try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            except FileNotFoundError:cmd=b''
            if b'train.monitored' in cmd and str(root).encode() in cmd and adoption is None:
                raise RuntimeError('Existing main worker is live; inspect/adopt before restarting supervisor')
    from scripts.resource_admission import inspect
    from scripts.run_local_split import process_tree_rss,size_tree,stop
    from scripts.process_accounting import proportional_tree
    from scripts.evaluate_final_rehearsal import acquire_idle_gpu
    from train.runtime import complete_checkpoints
    from train.monitored import validate_configuration
    import torch
    adopted = None
    if adoption is not None:
        from scripts.adopt_main_worker import AdoptedWorker
        receipt=json.loads(resolve(adoption).read_text())
        if (receipt['main_run'] != artifact_path(root) or not receipt['authorized'] or
                receipt['contract_sha256'] != sha256_file(root/'contract.json')):
            raise ValueError('Unapproved supervisor handoff receipt')
        previous=receipt['previous_supervisor']
        try:
            from scripts.adopt_main_worker import stamp
            if stamp(previous['pid'])[0] == previous['start_ticks']:
                raise ValueError('Previous supervisor is still live')
        except (FileNotFoundError,ProcessLookupError):
            pass
        adopted=AdoptedWorker(receipt['worker']['pid'],root,receipt)
    if c['framework']!=dict(torch=str(torch.__version__),cuda=torch.version.cuda):
        raise ValueError('Main framework differs from actual validation; restore/revalidate before execution')
    for p,h in {**c['code'],**c['evidence']}.items():
        if sha256_file(resolve(p))!=h:raise ValueError('Main execution implementation/input changed: '+p)
    config=json.loads((root/'main.json').read_text());validate_configuration(config,c)
    if c['configurations']!={'main.json':sha256_file(root/'main.json')} or resolve(config['out_dir'])!=root/'training':
        raise ValueError('Main output/configuration differs')
    tiered=c['retention_policy']=='nfs_latest_with_ephemeral_history'
    manager=None;retention_cancel=threading.Event()
    if tiered:
        from scripts.tiered_checkpoint_retention import TieredRetention
        if config['keep_ckpts']!=config['max_steps']+1:
            raise ValueError('Native checkpoint pruning must be disabled for tiered retention')
        manager=TieredRetention(root,c['tiered_retention'],authorized=True,cancelled=retention_cancel.is_set)
    if (root/'.done').exists():return certified(root,'main_target_completed_review_pending')
    hw=inspect();atomic_json(root/('hardware_'+str(time.time_ns())+'.json'),hw)
    if len(hw['gpus'])!=8 or hw['gpus']!=c['gpu_identity'] or hw['memory']['cache_aware_admission']['headroom_gib']<512:
        raise RuntimeError('Main8GPU/PSS budget headroom unavailable')
    training=root/'training';remaining=max(0,c['durable_budget_bytes']-size_tree(root))
    if shutil.disk_usage(root).free<remaining+200*GIB:raise RuntimeError('Main storage peak/reserve unavailable')
    scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(parents=True,exist_ok=True)
    if (shutil.disk_usage(scratch).free<(c['local_budget_gib']+200)*GIB and
            not (tiered and c['tiered_retention'].get('adaptive_local_pruning'))):
        raise RuntimeError('Main local scratch reserve unavailable')
    if manager:
        if Path(c['tiered_retention']['temp_root'])!=scratch:
            raise ValueError('Tiered retention temporary path differs from measured scratch')
        with manager.locked():
            if c['tiered_retention'].get('adaptive_local_pruning'):
                from scripts.retention_capacity import history_growth
                incoming=history_growth(c['tiered_retention'],0)
                manager.make_local_room(incoming)
                manager.budgets(incoming_bytes=incoming)
            else:manager.budgets(startup=True)
    cpu=hw['cpu']['affinity'][64:] or hw['cpu']['affinity']
    env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='1',TMPDIR=str(scratch),SQLITE_TMPDIR=str(scratch),PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    (root/'pid').write_text(str(os.getpid())+'\n');proc=adopted;futures={};verified={};pool=ThreadPoolExecutor(max_workers=1)
    elapsed_before=receipt.get('elapsed_before_seconds',0) if adopted else 0
    start=time.monotonic()-elapsed_before;peak_rss=0;peak_pss=0;last_pss=-30;retention_future=None;retention_count=-1
    def progress(phase,**kw):
        if time.monotonic()-start>c['max_seconds']:raise RuntimeError('Main wait/elapsed budget')
        atomic_json(root/'state.json',dict(phase='running',case=phase,**kw))
    try:
        from scripts.adopt_main_worker import gpu_lock
        with (gpu_lock(adopted) if adopted else acquire_idle_gpu(progress)):
            for path in complete_checkpoints(training):
                certificate=root/'checkpoint_certificates'/(path.name+'.json')
                if certificate.exists():
                    prior=json.loads(certificate.read_text())
                    if adopted:
                        from scripts.tiered_checkpoint_retention import inventory
                        inventory(path,prior,hashes=True,workers=8);verified[path.name]=prior
                    else:
                        verified[path.name]=seal(path,certificate,prior['training_contract_id'])
                else:
                    if adopted:
                        raise ValueError('Uncertified complete checkpoint at handoff; preserve and inspect')
                    archive=root/'unverified_checkpoints';archive.mkdir(exist_ok=True)
                    path.rename(archive/(path.name+'-'+str(time.time_ns())))
            if size_tree(root)>c['durable_budget_bytes']:raise RuntimeError('Preserved recovery outputs exceed main budget; review before restart')
            if manager:manager.maintain()
            if adopted:
                proc=adopted
                atomic_json(root/'supervisor_adopted.json',dict(worker_pid=proc.pid,attempt=proc.attempt,receipt=adoption,unix=time.time()))
            else:
                attempt='main-'+str(time.time_ns())
                cmd=['taskset','-c',','.join(map(str,cpu)),sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                     '--module','train.monitored','--run-dir',str(root),'--config',str(root/'main.json'),'--attempt',attempt]
                with (root/'attempts'/(attempt+'.log')).open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                elapsed=time.monotonic()-start;rss=process_tree_rss(os.getpid())+(process_tree_rss(proc.pid) if adopted else 0);peak_rss=max(peak_rss,rss)
                if elapsed-last_pss>=30 or rss>c['max_pss_gib']*GIB:
                    observation=proportional_tree(os.getpid())
                    if adopted:
                        extra=proportional_tree(proc.pid)['totals_bytes']
                        for key in ['Rss','Pss']:observation['totals_bytes'][key]+=extra[key]
                    pss=observation['totals_bytes']['Pss'];last_pss=elapsed;peak_pss=max(peak_pss,pss)
                    if memory_violation(observation['totals_bytes']['Rss'],pss,c['max_pss_gib']*GIB):raise RuntimeError('Main proportional memory budget')
                if elapsed>c['max_seconds'] or size_tree(root)>c['durable_budget_bytes'] or size_tree(scratch)>c['local_budget_gib']*GIB or min(shutil.disk_usage(root).free,shutil.disk_usage(scratch).free)<200*GIB:
                    raise RuntimeError('Main elapsed/disk/reserve budget')
                if manager:
                    from scripts.retention_capacity import enforce_capacity
                    enforce_capacity(manager,busy=retention_future is not None and not retention_future.done())
                cp=training/'training_contract.json'
                if cp.exists():
                    cid=json.loads(cp.read_text())['id']
                    for path in complete_checkpoints(training):
                        if path.name not in verified and path.name not in futures:
                            futures[path.name]=pool.submit(seal,path,root/'checkpoint_certificates'/(path.name+'.json'),cid)
                for name,future in list(futures.items()):
                    if future.done():verified[name]=future.result();del futures[name]
                if retention_future is not None and retention_future.done():
                    retention_future.result();retention_future=None
                if manager and retention_future is None and not futures and retention_count!=len(verified):
                    retention_future=pool.submit(manager.maintain)
                    retention_count=len(verified)
                progress('main',worker_pid=proc.pid,elapsed_seconds=elapsed,peak_rss_bytes=peak_rss,peak_pss_bytes=peak_pss,
                         certified_checkpoints=len(list((root/'checkpoint_certificates').glob('*.json'))))
                time.sleep(2)
            if proc.returncode:raise RuntimeError('Main worker failed; preserve checkpoint prefixes and inspect attempt log')
            cid=json.loads((training/'training_contract.json').read_text())['id']
            for path in complete_checkpoints(training):
                if path.name in futures:verified[path.name]=futures.pop(path.name).result()
                elif path.name not in verified:verified[path.name]=seal(path,root/'checkpoint_certificates'/(path.name+'.json'),cid)
            for name,future in futures.items():verified[name]=future.result()
            if retention_future is not None:retention_future.result()
            retention=manager.maintain() if manager else None
            final=training/'ckpt_0057000'
            if final.name not in verified:raise ValueError('Main target final checkpoint missing')
            report=dict(status='main_target_completed_review_pending',final_checkpoint=artifact_path(final),trained_tokens=119537664000,
                        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak_rss,peak_pss_bytes=peak_pss,
                        certificate_receipts={p.name:sha256_file(p) for p in (root/'checkpoint_certificates').glob('*.json')},
                        retention=retention,
                        scope='Main target completed; final native/benchmark/quality review remains separate.')
            atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
            atomic_json(root/'state.json',dict(phase='complete'))
    except BaseException as exc:
        retention_cancel.set()
        if adopted is not None:adopted.stop_owned()
        elif proc is not None and proc.poll() is None:stop(proc)
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise
    finally:pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--review-run',default='runs/pretrain_final_review_v1');p.add_argument('--keep-checkpoints',type=int,default=10)
    p.add_argument('--local-history',type=int,default=15)
    p.add_argument('--retention-plan',help='Reviewed draft; preserves its exact local namespace after actual final gates close')
    p.add_argument('--adopt-receipt',help='Explicit approved PID/start-time-bound handoff; keep existing eight ranks')
    a=p.parse_args();root=resolve(a.run_dir);root.relative_to(resolve('runs'))
    if a.prepare:print(json.dumps(prepare(a,root)))
    else:
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(root,a.adopt_receipt)
