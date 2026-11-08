"""Wait for audited final3B, then supervise isolated8GPU resume verification.

The parent model/config/run is read-only. Preparation and wait states are not
final acceptance. Failed artifacts remain for review; no automatic retry,
pruning, external upload or main-training restart is implemented here.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file, identity
from train.final_resume_probe import (check_spec, file_stats, stat_record, metadata_for,
                                     OUTPUTS, MODES, verify_parent_metadata)
from scripts.evaluate_final_main import main_observation, completed_main
from scripts.evaluate_final_rehearsal import acquire_idle_gpu, check_bindings
from scripts.tiered_checkpoint_retention import size_tree, inventory

GIB = 2**30


def load_plan(root):
    root = resolve(root)
    plan = json.loads((root/'plan.json').read_text())
    marker = json.loads((root/'prepared.json').read_text())
    if (marker['plan_sha256'] != sha256_file(root/'plan.json') or marker['plan_id'] != identity(plan) or
            resolve(plan['run_dir']) != root or plan['purpose'] != 'isolated_final_resume_probe' or
            plan['parent_step'] != 57000 or plan['world_size'] != 8 or plan['updates'] != 2 or
            plan['main_target_modified'] is not False or plan['lr_policy'] != 'freeze_last_parent_update'):
        raise ValueError('Unreviewed isolated final resume plan')
    check_bindings(plan)
    main = resolve(plan['main_run'])
    mc = json.loads((main/'contract.json').read_text())
    if (root.is_relative_to(main) or main.is_relative_to(root) or not root.is_relative_to(resolve('runs')) or
            Path(plan['scratch']) != Path(mc['tiered_retention']['temp_root'])/root.name or
            any(p.is_symlink() for path in [root, Path(plan['scratch'])] for p in [path, *path.parents])):
        raise ValueError('Diagnostic run/scratch path differs or overlaps source')
    return plan


def make_contract(root):
    root = resolve(root); plan = load_plan(root)
    added = ['scripts/run_final_resume_probe.py', 'scripts/evaluate_final_main.py',
             'scripts/evaluate_final_rehearsal.py', 'scripts/process_accounting.py',
             'scripts/resource_admission.py', 'scripts/hardware_profile.py']
    return dict(version=1, purpose='final_resume_controller', plan_sha256=sha256_file(root/'plan.json'),
                code={**plan['code'], **{p:sha256_file(resolve(p)) for p in added}},
                inputs={**plan['inputs'], artifact_path(root/'plan.json'):sha256_file(root/'plan.json'),
                        artifact_path(root/'prepared.json'):sha256_file(root/'prepared.json')},
                poll_seconds=30, wait_max_seconds=22*86400, worker_max_seconds=plan['max_seconds'],
                worker_pss_gib=plan['max_pss_gib'], local_budget_gib=plan['local_budget_gib'],
                durable_budget_gib=plan['durable_budget_gib'], main_run=plan['main_run'],
                scope='Actual57000 audit, fresh parent SHA/input hashes,8GPU global lock, three isolated processes and full-state comparisons. No main update or pruning.')


def owned_processes(root):
    """Find live exact worker/spec argument handles, including orphan torchranks."""
    root = resolve(root); found = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            if (directory/'stat').read_text().split(') ', 1)[1].split()[0] == 'Z':
                continue
            args = (directory/'cmdline').read_bytes().decode(errors='replace').split('\0')
            owns = False
            if any('train.final_resume_probe' == arg for arg in args) and '--spec' in args:
                owns = resolve(args[args.index('--spec')+1]) == root/'spec.json'
            if '--worker' in args and '--run-dir' in args and any(arg.endswith('run_final_resume_probe.py') for arg in args):
                owns |= resolve(args[args.index('--run-dir')+1]) == root
            if owns:
                found.append(int(directory.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(found)


def reject_previous_execution(root):
    root = resolve(root)
    processes = owned_processes(root)
    if processes:
        raise RuntimeError('Existing actual diagnostic process handles: '+str(processes))
    plan = load_plan(root)
    if ((root/'execution_started.json').exists() or (root/'spec.json').exists() or
            (root/'worker_state.json').exists() or (root/'ledgers').exists() or Path(plan['scratch']).exists()):
        raise RuntimeError('Preserve previous diagnostic attempt; review before using a new execution namespace')


def budget_check(plan, policy, *, nfs_free, local_free, main_used, history_used, temp_used,
                 own_used, scratch_used, native_nfs_used, native_temp_used, headroom_gib):
    remaining_nfs = max(0, policy['nfs_budget_bytes']-main_used)
    own_nfs = max(0, plan['durable_budget_gib']*GIB-own_used)
    native_nfs = max(0, 8*GIB-native_nfs_used)
    history = max(0, policy['local_budget_bytes']-history_used)
    temporary = max(0, policy['temporary_budget_bytes']-temp_used)
    if nfs_free < remaining_nfs+own_nfs+native_nfs+policy['other_nfs_future_bytes']+policy['reserve_bytes']:
        raise RuntimeError('Diagnostic NFS budget/reserve unavailable')
    if local_free < history+temporary+policy['reserve_bytes']:
        raise RuntimeError('Diagnostic local history/temp/reserve unavailable')
    # The192GiB probe and8GiB native workspace are subdivisions of the existing
    # 1TiB temporary allowance. Account for other residues before admitting them.
    projected_temp = max(0, temp_used-scratch_used)+plan['local_budget_gib']*GIB+max(0, 8*GIB-native_temp_used)
    if (projected_temp > policy['temporary_budget_bytes'] or
            scratch_used > plan['local_budget_gib']*GIB or own_used > plan['durable_budget_gib']*GIB):
        raise RuntimeError('Diagnostic existing residues or temporary/durable cap exceeded')
    if headroom_gib < 384+256+plan['max_pss_gib']:
        raise RuntimeError('Combined main/native/probe memory admission unavailable')
    return dict(nfs_remaining_bytes=remaining_nfs+own_nfs+native_nfs,
                local_remaining_bytes=history+temporary, projected_temporary_bytes=projected_temp,
                nfs_after_budget_reserve_bytes=nfs_free-remaining_nfs-own_nfs-native_nfs-policy['other_nfs_future_bytes']-policy['reserve_bytes'],
                local_after_budget_reserve_bytes=local_free-history-temporary-policy['reserve_bytes'],
                combined_memory_gib=384+256+plan['max_pss_gib'])


def admission(root, plan):
    from scripts.resource_admission import inspect
    hw = inspect(); main = resolve(plan['main_run'])
    mc = json.loads((main/'contract.json').read_text()); policy = mc['tiered_retention']
    if hw['gpus'] != mc['gpu_identity'] or len(hw['gpus']) != 8:
        raise RuntimeError('Reviewed8GPU hardware changed')
    native = resolve('runs/3b_final_native_v2')
    scratch = Path(plan['scratch']); temp = Path(policy['temp_root'])
    # A worker has an explicit TMPDIR. hardware_profile derives its suggested
    # scratch from that environment, so compare real devices, not that suggestion.
    if temp.parent.stat().st_dev != hw['storage']['scratch_parent']['device'] or temp.parent.stat().st_dev == root.stat().st_dev:
        raise ValueError('Reviewed temporary storage is not the separate local filesystem')
    result = budget_check(plan, policy, nfs_free=shutil.disk_usage(root).free,
        local_free=shutil.disk_usage(temp.parent).free,
        main_used=size_tree(main), history_used=size_tree(Path(policy['local_root'])), temp_used=size_tree(temp),
        own_used=size_tree(root), scratch_used=size_tree(scratch), native_nfs_used=size_tree(native),
        native_temp_used=size_tree(temp/'final_native'), headroom_gib=hw['memory']['cache_aware_admission']['headroom_gib'])
    return hw, result


def fresh_inputs(plan, output):
    from train.parallel_contract import training_contract
    parent = json.loads(resolve(plan['parent_contract']).read_text())
    config = json.loads(resolve(plan['parent_config']).read_text())
    before = {p:stat_record(resolve(p)) for p in parent['files']}
    fresh = training_contract(config, 8, config['holdout'], workers=32,
                expected_files=parent['files'], observe=lambda metrics:atomic_json(output, metrics))
    for key, value in fresh.items():
        if key != 'id' and parent[key] != value:
            raise ValueError('Fresh native input/config/code identity differs: '+key)
    after = {p:stat_record(resolve(p)) for p in parent['files']}
    if before != after:
        raise ValueError('Training input changed across live fresh verification')
    return after


def materialize_spec(root, plan):
    checkpoint, proof, meta, audit = completed_main(plan['main_run'])
    parent = json.loads(resolve(plan['parent_contract']).read_text())
    config = json.loads(resolve(plan['parent_config']).read_text())
    verify_parent_metadata(meta, config, parent, plan['parent_training_contract_id'])
    if (root/'spec.json').exists():
        raise ValueError('Existing final execution spec is preserved')
    before = file_stats(checkpoint)
    inventory(checkpoint, proof, hashes=True, workers=8)
    if before != file_stats(checkpoint):
        raise ValueError('Final parent changed during fresh SHA verification')
    stats = fresh_inputs(plan, root/'fresh_input_hash.json')
    atomic_json(root/'final_training_audit.json', audit)
    proof_path = resolve(audit['checkpoint_certificate'])
    spec = dict(plan=plan, plan_id=identity(plan), parent_checkpoint=artifact_path(checkpoint),
                parent_proof=artifact_path(proof_path), parent_proof_sha256=sha256_file(proof_path),
                audit=artifact_path(root/'final_training_audit.json'), audit_sha256=sha256_file(root/'final_training_audit.json'),
                parent_stats=before, input_stats=stats)
    check_spec(spec)
    atomic_json(root/'spec.json', spec)
    return spec


def require_comparison(result):
    if (result.get('exact') is not True or result.get('tensors') != 1020 or result.get('model_tensors') != 255 or
            result.get('optimizer_tensors') != 765 or result.get('rng_ranks') != 8):
        raise ValueError('Actual3B1020tensor/8rank comparison is incomplete')


def collect_result(root, spec):
    from scripts.verify_final_resume_probe import verify_inventory, ledger_review
    plan, config, parent = check_spec(spec)
    scratch = Path(plan['scratch'])
    paths = sorted((root/'ledgers').glob('*.json'))
    wanted = {f'{m}-rank{r}.json' for m in MODES for r in range(8)}
    if {p.name for p in paths} != wanted:
        raise ValueError('Diagnostic24 ledger file coverage differs')
    counts = ledger_review([json.loads(p.read_text()) for p in paths], parent_step=57000, world=8,
        config=config, source_order=parent['source_order'], plan_id=identity(plan))
    checks = {name:json.loads((root/(name+'.json')).read_text()) for name in ['roundtrip_comparison','resume_comparison']}
    for result in checks.values():
        require_comparison(result)
    outputs = {'restored_parent':0, 'continuous_final':2, 'split_first':1, 'resumed_final':2}
    if {p.name for p in scratch.iterdir()} != set(outputs):
        raise ValueError('Unexpected diagnostic residue or missing checkpoint')
    for name, offset in outputs.items():
        proof = json.loads((root/'certificates'/(name+'.json')).read_text())
        if proof['metadata'] != metadata_for(spec, offset):
            raise ValueError('Diagnostic certificate lineage differs')
        verify_inventory(scratch/name, proof)
    proof_paths = [root/'spec.json', root/'final_training_audit.json', root/'fresh_input_hash.json',
                   resolve(spec['parent_proof']), *paths, *sorted((root/'certificates').glob('*.json')),
                   root/'roundtrip_comparison.json', root/'resume_comparison.json', root/'execution_timing.json']
    return dict(status='passed_actual_final3b_8gpu_resume', parent_checkpoint=spec['parent_checkpoint'],
        parent_step=57000, main_target_modified=False, counts=counts, comparisons=checks,
        local_checkpoint_bytes=size_tree(scratch), local_checkpoint_paths=[str(scratch/name) for name in outputs],
        evidence={artifact_path(p):sha256_file(p) for p in proof_paths},
        scope='Actual final3B full model/AdamW/8rank RNG roundtrip and continuous2 versus split1/restart1 exact. Diagnostic tokens excluded from main; native quality, official benchmarks and retention audit remain separate.')


def worker(root):
    import torch
    from scripts.verify_final_resume_probe import seal, compare_states
    root = resolve(root); contract = json.loads((root/'contract.json').read_text())
    check_bindings(contract); plan = load_plan(root)
    if plan['framework'] != dict(torch=str(torch.__version__), cuda=torch.version.cuda):
        raise ValueError('Final probe framework changed')
    if not (root/'execution_started.json').exists():
        raise ValueError('Supervised execution receipt missing')
    scratch = Path(plan['scratch']); started = time.monotonic(); phases = []
    def notify(case, **extra):
        atomic_json(root/'worker_state.json', dict(phase='running', case=case, pid=os.getpid(),
                    elapsed_seconds=time.monotonic()-started, **extra))
    notify('await_idle8gpu')
    with acquire_idle_gpu(notify):
        # Recheck measured reservations after any GPU lock wait.
        hw, budgets = admission(root, plan)
        atomic_json(root/'execution_hardware.json', hw); atomic_json(root/'execution_admission.json', budgets)
        notify('audit_fresh_parent_and_inputs')
        spec = materialize_spec(root, plan)
        scratch.mkdir(parents=True, exist_ok=False)
        for name in ['ledgers','certificates','attempts']:
            (root/name).mkdir(exist_ok=False)
        for mode in ['continuous', 'first', 'resumed']:
            check_bindings(contract); check_spec(spec)
            notify('actual8gpu_'+mode)
            command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                       '--module','train.final_resume_probe','--spec',str(root/'spec.json'),'--mode',mode]
            begin = time.monotonic()
            # Inherit this worker's dedicated process group so its supervisor
            # can terminate every owned rank on a fatal budget violation.
            with (root/'attempts'/(mode+'.log')).open('x') as stream:
                proc = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT)
            notify('actual8gpu_'+mode, torchrun_pid=proc.pid)
            rc = proc.wait()
            if rc:
                raise RuntimeError('Actual8GPU diagnostic failed in '+mode+'; preserve outputs')
            targets = [(OUTPUTS[mode], MODES[mode][1])]
            if mode == 'continuous':
                targets.insert(0, ('restored_parent',0))
            notify('certify_'+mode)
            for name, offset in targets:
                seal(scratch/name, metadata_for(spec,offset), root/'certificates'/(name+'.json'))
            phases.append(dict(mode=mode, elapsed_seconds=time.monotonic()-begin))
            check_spec(spec)
    # CPU comparisons can overlap final native scoring after releasing the GPUs.
    torch.set_num_threads(8)
    for label, left, right in [('roundtrip_comparison',resolve(spec['parent_checkpoint']),scratch/'restored_parent'),
                              ('resume_comparison',scratch/'continuous_final',scratch/'resumed_final')]:
        notify(label)
        result = compare_states(left,right,world=8); require_comparison(result)
        atomic_json(root/(label+'.json'),result)
    atomic_json(root/'execution_timing.json', dict(phases=phases, elapsed_seconds=time.monotonic()-started))
    check_bindings(contract)
    report = collect_result(root,spec)
    atomic_json(root/'worker_result.json',report)
    notify('worker_verified_pending_supervisor')


def terminate_group(proc):
    """Only the new session this controller itself started, never main workers."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait(); return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # Parent may exit before a stubborn torchrank; clear only this owned group.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def supervise_command(command, *, root, env, guard, interval=2):
    root = Path(root)
    with (root/'worker_console.log').open('x') as stream:
        proc = subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(root/'worker_handle.json',dict(pid=proc.pid,process_group=proc.pid,command=command))
    started = time.monotonic(); peak = 0
    from scripts.process_accounting import proportional_tree
    try:
        while proc.poll() is None:
            try:
                peak = max(peak, proportional_tree(proc.pid)['totals_bytes']['Pss'])
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if proc.poll() is None:
                    raise
            guard(proc.pid,time.monotonic()-started,peak)
            time.sleep(interval)
        # Final guards also run for fast exits; marker alone never passes.
        guard(proc.pid,time.monotonic()-started,peak)
        if proc.returncode:
            raise RuntimeError('Final resume worker exited '+str(proc.returncode)+'; evidence preserved')
        return dict(worker_pid=proc.pid,peak_pss_bytes=peak,elapsed_seconds=time.monotonic()-started,exit_code=proc.returncode)
    except BaseException:
        terminate_group(proc)
        raise


def verify_completed(root, contract):
    root = Path(root); check_bindings(contract)
    marker = json.loads((root/'.done').read_text())
    if marker != dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')):
        raise ValueError('Final resume completion proof differs')
    report = json.loads((root/'report.json').read_text())
    fresh = collect_result(root,json.loads((root/'spec.json').read_text()))
    if any(report.get(k) != v for k,v in fresh.items()):
        raise ValueError('Final resume report differs from current content proof')
    return report


def main(args):
    root = resolve(args.run_dir)
    if args.worker:
        return worker(root)
    plan = load_plan(root)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        contract = make_contract(root); cp = root/'contract.json'
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise ValueError('Controller binding changed; preserve old run and prepare a new namespace')
        if not cp.exists():
            atomic_json(cp,contract)
        if args.prepare:
            atomic_json(root/'controller_prepared.json',dict(status='prepared_not_executed',
                        contract_sha256=sha256_file(cp),command=[sys.executable,'-u',artifact_path(Path(__file__)),
                        '--run-dir',artifact_path(root),'--run']))
            return
        if (root/'.done').exists():
            return verify_completed(root,contract)
        reject_previous_execution(root); check_bindings(contract)
        (root/'pid').write_text(str(os.getpid())+'\n')
        started = time.monotonic()
        try:
            while True:
                observation = main_observation(plan['main_run'])
                if observation['complete']:
                    break
                atomic_json(root/'state.json',dict(phase='running',case='await_audited_main_final',
                            main_step=observation['step'],actual_final_resume_completed=False))
                if time.monotonic()-started > contract['wait_max_seconds']:
                    raise RuntimeError('Final resume wait budget exhausted; inspect live main before any recovery')
                time.sleep(contract['poll_seconds'])
            check_bindings(contract)
            hw, budgets = admission(root,plan)
            atomic_json(root/'launch_hardware.json',hw); atomic_json(root/'launch_admission.json',budgets)
            cpus = sorted(set(plan['cpu_affinity']) & set(hw['cpu']['affinity']))
            if not cpus:
                raise RuntimeError('Diagnostic CPU affinity unavailable')
            atomic_json(root/'execution_started.json',dict(controller_pid=os.getpid(),unix=time.time(),plan_id=identity(plan)))
            # TMPDIR is the already reserved parent directory; probe scratch is
            # created only after complete audit and fresh source verification.
            env = dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',RAYON_NUM_THREADS='1',
                       TMPDIR=str(Path(plan['scratch']).parent),PYTHONUNBUFFERED='1')
            command = ['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),
                       '--run-dir',str(root),'--worker']
            def guard(pid, elapsed, peak):
                state = json.loads((root/'worker_state.json').read_text()) if (root/'worker_state.json').exists() else {}
                atomic_json(root/'state.json',dict(phase='running',case=state.get('case','launching'),
                            worker_pid=pid,elapsed_seconds=elapsed,peak_pss_bytes=peak))
                if peak > contract['worker_pss_gib']*GIB or elapsed > contract['worker_max_seconds']:
                    raise RuntimeError('Final resume PSS/time budget exceeded')
                durable_bytes = size_tree(root); local_bytes = size_tree(Path(plan['scratch']))
                guard.storage_peak['durable'] = max(guard.storage_peak['durable'],durable_bytes)
                guard.storage_peak['local'] = max(guard.storage_peak['local'],local_bytes)
                if (durable_bytes > contract['durable_budget_gib']*GIB or local_bytes > contract['local_budget_gib']*GIB or
                        shutil.disk_usage(root).free < 200*GIB or shutil.disk_usage(Path(plan['scratch']).parent).free < 200*GIB):
                    raise RuntimeError('Final resume storage/reserve budget exceeded')
                if elapsed-guard.last_admission >= 30:
                    _, current = admission(root,plan)
                    atomic_json(root/'current_admission.json',current)
                    guard.last_admission = elapsed
            guard.last_admission = 0
            guard.storage_peak = dict(durable=0,local=0)
            metrics = supervise_command(command,root=root,env=env,guard=guard)
            metrics['sampled_peak_storage_bytes'] = guard.storage_peak
            if owned_processes(root):
                raise RuntimeError('Diagnostic ranks remain after coordinator exit')
            check_bindings(contract)
            report = json.loads((root/'worker_result.json').read_text())
            fresh = collect_result(root,json.loads((root/'spec.json').read_text()))
            if report != fresh:
                raise ValueError('Worker output differs from independently reread evidence')
            report['supervision'] = metrics
            atomic_json(root/'report.json',report)
            atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(cp)))
            verify_completed(root,contract)
            atomic_json(root/'state.json',dict(phase='complete',case='actual_final8gpu_resume_verified',**metrics))
        except BaseException as exc:
            atomic_json(root/'state.json',dict(phase='error',error=str(exc),actual_final_resume_completed=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--run',action='store_true'); mode.add_argument('--worker',action='store_true')
    main(parser.parse_args())
