"""Bounded CPU-only final retention audit after actual main completion.

The approved main and both storage namespaces are read-only. A completion
marker triggers the worker; the worker must audit actual8rank57000 and freshly
hash every retained checkpoint. Interrupted evidence is preserved, never
automatically retried or repaired. This is not whole-Goal completion.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.evaluate_final_main import main_observation
from scripts.evaluate_final_rehearsal import check_bindings
from scripts.audit_checkpoint_retention import existing_ancestor
from scripts.run_final_resume_probe import supervise_command
from scripts.tiered_checkpoint_retention import size_tree
from train.final_resume_probe import file_stats

GIB = 2**30


def make_contract(main_run, native_run, resume_run):
    main, native, resume = map(resolve, [main_run, native_run, resume_run])
    mc = json.loads((main/'contract.json').read_text())
    from scripts.supervise_main_training import require_authorization
    require_authorization(mc)
    cfg = json.loads((main/'main.json').read_text())
    nc = json.loads((native/'contract.json').read_text())
    rc = json.loads((resume/'contract.json').read_text())
    plan = json.loads((resume/'plan.json').read_text())
    check_bindings(nc); check_bindings(rc)
    if (cfg['model'] != '3b' or cfg['max_steps'] != 57000 or cfg['global_tokens'] != 2097152 or
            mc['configurations']['main.json'] != sha256_file(main/'main.json') or
            nc['purpose'] != 'final_native_main_evaluation' or rc['purpose'] != 'final_resume_controller' or
            resolve(nc['main_run']) != main or resolve(plan['main_run']) != main or
            resolve(plan['native_run']) != native or resolve(rc['main_run']) != main):
        raise ValueError('Final retention dependencies/target differ')
    names = ['scripts/run_final_retention_audit.py', 'scripts/audit_checkpoint_retention.py',
             'scripts/audit_main_training.py', 'scripts/evaluate_final_main.py',
             'scripts/run_final_resume_probe.py', 'scripts/verify_final_resume_probe.py',
             'train/final_resume_probe.py', 'scripts/validate_tiered_retention_copy.py']
    inputs = [main/'contract.json', main/'main.json', main/'authorization.json',
              main/'training/training_contract.json', native/'contract.json',
              resume/'contract.json', resume/'plan.json', resume/'prepared.json']
    return dict(version=1, purpose='final_retention_audit_controller', main_run=artifact_path(main),
                native_run=artifact_path(native), resume_run=artifact_path(resume),
                code={**mc['code'], **{p:sha256_file(resolve(p)) for p in names}},
                inputs={**mc['evidence'], **{artifact_path(p):sha256_file(p) for p in inputs}},
                poll_seconds=30, wait_max_seconds=22*86400, worker_max_seconds=1200,
                max_pss_gib=16, durable_budget_gib=1, local_additional_budget_gib=0,
                cpu_affinity=list(range(32)), hash_workers=8,
                env=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', RAYON_NUM_THREADS='1'),
                source_contract_sha256=sha256_file(main/'contract.json'),
                scope='Fresh full SHA of actual final resident10+15, NFS-only resume and optional local loss. No main mutation, repair, pruning, GPU, upload or Goal completion.')


def validate_output(root, contract):
    root = resolve(root)
    if root == resolve('runs') or not root.is_relative_to(resolve('runs')):
        raise ValueError('Use an independent run under runs')
    mc = json.loads((resolve(contract['main_run'])/'contract.json').read_text())
    protected = [resolve(contract[k]) for k in ['main_run','native_run','resume_run']]
    protected += [Path(mc['tiered_retention'][k]) for k in ['local_root','temp_root']]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError('Audit output overlaps a protected source')
    if any(p.is_symlink() for p in [root,*root.parents]):
        raise ValueError('Audit output symlink rejected')


def capacity_check(contract, policy, *, nfs_free, local_free, main_used, history_used,
                   temp_used, own_used, native_used, resume_used, headroom_gib, combined_memory_gib):
    remaining = (max(0, policy['nfs_budget_bytes']-main_used) +
                 max(0, contract['durable_budget_gib']*GIB-own_used) +
                 max(0, 8*GIB-native_used) + max(0, GIB-resume_used))
    local_remaining = max(0, policy['local_budget_bytes']-history_used) + max(0, policy['temporary_budget_bytes']-temp_used)
    if headroom_gib < combined_memory_gib:
        raise RuntimeError('Combined main/native/resume/audit memory unavailable')
    if nfs_free < remaining+policy['reserve_bytes']+policy['other_nfs_future_bytes']:
        raise RuntimeError('Final retention NFS future/reserve budget unavailable')
    if local_free < local_remaining+policy['reserve_bytes']:
        raise RuntimeError('Final retention local history/temp/reserve unavailable')
    if (own_used > contract['durable_budget_gib']*GIB or main_used > policy['nfs_budget_bytes'] or
            history_used > policy['local_budget_bytes'] or temp_used > policy['temporary_budget_bytes']):
        raise RuntimeError('Preserved residues or current storage cap exceeded')
    return dict(combined_memory_gib=combined_memory_gib, nfs_remaining_bytes=remaining,
                local_remaining_bytes=local_remaining,
                nfs_after_budget_reserve_bytes=nfs_free-remaining-policy['reserve_bytes']-policy['other_nfs_future_bytes'],
                local_after_budget_reserve_bytes=local_free-local_remaining-policy['reserve_bytes'],
                audit_additional_local_bytes=0)


def admission(root, contract):
    from scripts.resource_admission import inspect
    hw = inspect(); main = resolve(contract['main_run'])
    mc = json.loads((main/'contract.json').read_text()); policy = mc['tiered_retention']
    nc = json.loads((resolve(contract['native_run'])/'contract.json').read_text())
    rc = json.loads((resolve(contract['resume_run'])/'contract.json').read_text())
    if (hw['gpus'] != mc['gpu_identity'] or nc['durable_budget_gib'] != 8 or
            rc['durable_budget_gib'] != 1 or nc['host_rss_gib'] != 256 or rc['worker_pss_gib'] != 256):
        raise RuntimeError('Reviewed hardware/dependency budgets changed')
    local = existing_ancestor(policy['local_root'])
    temp = existing_ancestor(policy['temp_root'])
    if local.stat().st_dev == root.stat().st_dev or temp.stat().st_dev != local.stat().st_dev:
        raise ValueError('NFS/local/temp filesystem identity differs')
    combined = mc['max_pss_gib']+nc['host_rss_gib']+rc['worker_pss_gib']+contract['max_pss_gib']
    result = capacity_check(contract, policy, nfs_free=shutil.disk_usage(root).free,
        local_free=shutil.disk_usage(local).free, main_used=size_tree(main),
        history_used=size_tree(Path(policy['local_root'])), temp_used=size_tree(Path(policy['temp_root'])),
        own_used=size_tree(root), native_used=size_tree(resolve(contract['native_run'])),
        resume_used=size_tree(resolve(contract['resume_run'])),
        headroom_gib=hw['memory']['cache_aware_admission']['headroom_gib'], combined_memory_gib=combined)
    return hw, result


def owned_workers(root):
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if ((proc/'stat').read_text().split(') ',1)[1].split()[0] != 'Z' and
                    any(a.endswith('run_final_retention_audit.py') for a in args) and '--worker' in args and
                    '--run-dir' in args and resolve(args[args.index('--run-dir')+1]) == resolve(root)):
                found.append(int(proc.name))
        except (OSError, ValueError, IndexError):
            pass
    return sorted(found)


def reject_previous(root):
    live = owned_workers(root)
    if live:
        raise RuntimeError('Actual orphan audit handles exist: '+str(live))
    if any((root/p).exists() for p in ['execution_started.json','worker_handle.json','worker_result.json','worker_console.log']):
        raise RuntimeError('Preserve prior audit attempt and review before a new namespace')


def require_result(result, contract):
    if (result['status'] != 'passed_final_retention_content' or result['production'] is not True or
            result['final'] is not True or result['full_payload_sha_refreshed'] is not True or
            result['source_run'] != contract['main_run'] or
            result['source_contract_sha256'] != contract['source_contract_sha256'] or
            result['audit_code_sha256'] != contract['code']['scripts/audit_checkpoint_retention.py'] or
            result['issues'] or result['residues'] or result['overlap']):
        raise ValueError('Actual final retention content verdict missing or differs')
    nfs = sorted(result['nfs']); local = sorted(result['local'])
    wanted = [f'ckpt_{s:07d}' for s in range(52500,57001,500)]
    if (nfs != wanted or result['nfs_resume_candidates'] != wanted or
            result['expected_nfs'] != wanted or len(local)>15 or set(nfs)&set(local) or
            result['distinct_checkpoints'] != len(nfs)+len(local) or
            len(result['certificate_receipts']) != 114):
        raise ValueError('Actual final10+15 coverage differs')
    if not result['local_history_optional_loss'] and len(local) != 15:
        raise ValueError('Unexplained missing local history')
    if ((result['local_namespace_missing'] and local) or
            (not result['local_namespace_missing'] and local != result['expected_local'])):
        raise ValueError('Local loss epoch coverage differs')
    for cp in [*result['nfs'].values(),*result['local'].values()]:
        if (cp['model_tensors'] != 255 or cp['state_tensors'] != 1020 or cp['rng_ranks'] != 8 or
                cp['payload_files'] != 8 or cp['full_payload_sha_refreshed'] is not True):
            raise ValueError('Full3B resume state coverage missing')
    audit = result['training_audit']
    if (audit['validation_source'] or audit['through_step'] != 57000 or
            audit['counts']['committed_tokens'] != 119537664000 or
            audit['counts']['successful_target_ranks'] != list(range(8))):
        raise ValueError('Actual8rank target audit missing')


def verify_current(result, contract):
    """Reuse fresh worker SHA only while every bound byte/stat/set is unchanged."""
    check_bindings(contract); require_result(result,contract)
    main = resolve(contract['main_run'])
    policy = json.loads((main/'contract.json').read_text())['tiered_retention']
    local = Path(policy['local_root'])
    paths = {str(main/'training'/n) for n in result['nfs']} | {str(local/'checkpoints'/n) for n in result['local']}
    if set(result['resident_file_stats']) != paths:
        raise ValueError('Resident proof path coverage differs')
    for name, stats in result['resident_file_stats'].items():
        path = Path(name)
        if any(p.is_symlink() for p in [path,*path.parents]):
            raise ValueError('Resident namespace became a symlink')
        if file_stats(Path(name)) != stats:
            raise ValueError('Resident checkpoint changed after fresh SHA: '+name)
    bindings = dict(result['metadata_evidence'])
    for name, digest in result['training_audit']['evidence'].items():
        if name in bindings and bindings[name] != digest:
            raise ValueError('Conflicting worker content proofs')
        bindings[name] = digest
    for name, digest in bindings.items():
        if sha256_file(resolve(name)) != digest:
            raise ValueError('Audited metadata/ledger changed after worker: '+name)
    if {p.name for p in (main/'training').iterdir()} != set(result['nfs']) | {'training_contract.json','log.jsonl'}:
        raise ValueError('NFS resident namespace changed after worker')
    if any((main/'retention_trash').iterdir()):
        raise ValueError('NFS retention residue appeared')
    certificates = {p.name:sha256_file(p) for p in (main/'checkpoint_certificates').iterdir()}
    if certificates != result['certificate_receipts']:
        raise ValueError('NFS certificate namespace changed')
    if result['local_namespace_missing']:
        if local.exists():
            raise ValueError('Local namespace reappeared after worker')
    else:
        if {p.name for p in local.iterdir()} != {'.retention_owner.json','checkpoints','certificates','staging','retention_trash'}:
            raise ValueError('Local namespace changed after worker')
        if ({p.name for p in (local/'checkpoints').iterdir()} != set(result['local']) or
                {p.name for p in (local/'certificates').iterdir()} != {n+'.json' for n in result['local']} or
                any((local/'staging').iterdir()) or any((local/'retention_trash').iterdir())):
            raise ValueError('Local resident/certificate/residue namespace changed')


def worker(root):
    contract = json.loads((root/'contract.json').read_text()); check_bindings(contract)
    validate_output(root,contract)
    if not (root/'execution_started.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Supervised CPU-only audit receipt/environment missing')
    from scripts.audit_checkpoint_retention import audit_run
    result = audit_run(contract['main_run'],final=True)
    verify_current(result,contract)
    atomic_json(root/'worker_result.json',result)


def verify_completed(root, contract):
    marker = json.loads((root/'.done').read_text())
    if marker != dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')):
        raise ValueError('Final audit completion binding differs')
    report = json.loads((root/'report.json').read_text()); verify_current(report,contract)
    return report


def controller(root, contract):
    if (root/'.done').exists():
        return verify_completed(root,contract)
    reject_previous(root); check_bindings(contract)
    (root/'pid').write_text(str(os.getpid())+'\n'); started = time.monotonic()
    try:
        while True:
            observation = main_observation(contract['main_run'])
            if observation['complete']:
                break
            atomic_json(root/'state.json',dict(phase='running',case='await_actual_main_final',main_step=observation['step']))
            if time.monotonic()-started > contract['wait_max_seconds']:
                raise RuntimeError('Final retention wait time budget')
            rss = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
            if size_tree(root)>GIB or rss>512*2**20:
                raise RuntimeError('Final retention waiting metadata/memory budget')
            time.sleep(contract['poll_seconds'])
        hw, capacity = admission(root,contract)
        atomic_json(root/'execution_hardware.json',hw); atomic_json(root/'execution_capacity.json',capacity)
        cpus = sorted(set(contract['cpu_affinity']) & set(hw['cpu']['affinity']))
        if not cpus:
            raise RuntimeError('Final retention CPU affinity unavailable')
        atomic_json(root/'execution_started.json',dict(controller_pid=os.getpid(),unix=time.time()))
        env = dict(os.environ,**contract['env'],PYTHONUNBUFFERED='1')
        command = ['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),'--run-dir',str(root),'--worker']
        began = time.monotonic(); measured = dict(peak_pss_bytes=0,peak_durable_bytes=0)
        policy = json.loads((resolve(contract['main_run'])/'contract.json').read_text())['tiered_retention']
        def guard(pid, elapsed, peak):
            from scripts.process_accounting import proportional_tree
            measured['peak_pss_bytes'] = max(measured['peak_pss_bytes'],proportional_tree(os.getpid())['totals_bytes']['Pss'],peak)
            measured['peak_durable_bytes'] = max(measured['peak_durable_bytes'],size_tree(root))
            if measured['peak_pss_bytes']>contract['max_pss_gib']*GIB or time.monotonic()-began>contract['worker_max_seconds']:
                raise RuntimeError('Final retention CPU audit memory/time budget')
            if (measured['peak_durable_bytes']>GIB or shutil.disk_usage(root).free<200*GIB or
                    shutil.disk_usage(existing_ancestor(policy['local_root'])).free<200*GIB):
                raise RuntimeError('Final retention proof storage/reserve budget')
            if elapsed-guard.last_admission>=30:
                _, current = admission(root,contract); atomic_json(root/'current_capacity.json',current)
                guard.last_admission = elapsed
            atomic_json(root/'state.json',dict(phase='running',case='fresh_full_retention_audit',worker_pid=pid,elapsed_seconds=time.monotonic()-began,**measured))
        guard.last_admission = -30
        metrics = supervise_command(command,root=root,env=env,guard=guard)
        if owned_workers(root):
            raise RuntimeError('Audit worker remains alive after supervision')
        result = json.loads((root/'worker_result.json').read_text()); verify_current(result,contract)
        guard(metrics['worker_pid'],time.monotonic()-began,metrics['peak_pss_bytes'])
        result['supervision'] = dict(**metrics,controller_and_worker_peak=measured,wall_seconds=time.monotonic()-began)
        result['goal_complete'] = False
        atomic_json(root/'report.json',result)
        atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
        atomic_json(root/'state.json',dict(phase='complete',case='actual_final_retention_verified',goal_complete=False,**measured))
    except BaseException as exc:
        atomic_json(root/'state.json',dict(phase='error',reason=type(exc).__name__+': '+str(exc),goal_complete=False))
        raise


def main(args):
    root = resolve(args.run_dir)
    if args.worker:
        return worker(root)
    if args.prepare:
        if root.exists():
            raise ValueError('Prepare a new independent audit namespace')
        c = make_contract(args.main_run,args.native_run,args.resume_run); validate_output(root,c)
        root.mkdir(parents=True)
        atomic_json(root/'contract.json',c)
        hw, capacity = admission(root,c)
        atomic_json(root/'preparation_hardware.json',hw); atomic_json(root/'preparation_capacity.json',capacity)
        atomic_json(root/'prepared.json',dict(status='prepared_not_executed',contract_sha256=sha256_file(root/'contract.json'),actual_final_retention_verified=False))
        return
    c = json.loads((root/'contract.json').read_text()); validate_output(root,c)
    prepared = json.loads((root/'prepared.json').read_text())
    if prepared['contract_sha256'] != sha256_file(root/'contract.json'):
        raise ValueError('Prepared audit contract changed')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def interrupted(_signum,_frame):
            raise InterruptedError('Audit controller interrupted; preserve outputs')
        previous = signal.signal(signal.SIGTERM,interrupted)
        try:
            return controller(root,c)
        finally:
            signal.signal(signal.SIGTERM,previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--main-run',default='runs/3b_full_v1')
    p.add_argument('--native-run',default='runs/3b_final_native_v3')
    p.add_argument('--resume-run',default='runs/3b_final_resume_v2')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--worker',action='store_true')
    main(p.parse_args())
