"""Bounded actual3B NFS save/resume measurement; preserves all diagnostic files.

Reuses the frozen four-update monitored entry. Never launches a main-training
contract or prunes anything. Compares the same updates with the local run.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.run_native_evaluation import certified
from scripts.resource_admission import inspect
from scripts.run_local_split import process_tree_rss, size_tree, stop
from scripts.validate_monitored_training import check_contract, seal_checkpoint, verify_generations

GIB = 2**30


def compare_states(root):
    import torch
    from scripts.validate_tiered_retention_copy import cpu_state
    c = json.loads((root/'contract.json').read_text()); check_contract(c)
    torch.set_num_threads(8)
    start = time.monotonic()
    left_path = Path(c['reference_training'])/'ckpt_0000004'
    right_path = root/'training/ckpt_0000004'
    left, right = cpu_state(left_path), cpu_state(right_path)
    if left.keys() != right.keys():
        raise ValueError('Restored optimizer/model keys differ')
    count = 0
    for key, a in left.items():
        b = right[key]
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                raise ValueError('Local/NFS restored tensor differs: '+key)
            count += 1
        elif a != b:
            raise ValueError('Local/NFS optimizer metadata differs')
    for rank in range(8):
        a = torch.load(left_path/f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
        b = torch.load(right_path/f'rng_rank{rank}.pt', map_location='cpu', weights_only=True)
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError('Local/NFS rank RNG differs')
    atomic_json(root/'state_comparison.json', dict(exact=True, tensors=count,
                all8_rng_exact=True, elapsed_seconds=time.monotonic()-start))


def run(root):
    from scripts.evaluate_final_rehearsal import acquire_idle_gpu
    from scripts.validate_actual_training import compare_trajectories
    from scripts.run_rehearsal import trace_review
    from scripts.supervise_main_training import validate_optimizer_map
    from train.probe import verify_probe
    from train.rehearsal import aggregate_ledgers
    from train.runtime import complete_checkpoints
    import torch.distributed.checkpoint as dcp

    source = resolve('runs/monitored_3b_validation_v1')
    reference, original = certified(source, 'passed_actual_monitored_3b')
    for p, h in reference['evidence'].items():
        if sha256_file(resolve(p)) != h:
            raise ValueError('Reference actual evidence changed: '+p)
    root.relative_to(resolve('runs'))
    if root == source or root.name != 'nfs_checkpoint_validation_v1':
        raise ValueError('Use the dedicated diagnostic run; main paths rejected')
    root.mkdir(exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'.done').exists():
            return certified(root, 'passed_actual_nfs_save_resume_comparison')[0]
        # An interrupted attempt is preserved for explicit recovery, not replayed
        # over existing files by a second controller.
        if (root/'contract.json').exists():
            raise RuntimeError('Existing diagnostic attempt: inspect preserved state before recovery')
        hw = inspect()
        if not hw['storage']['project']['network'] or len(hw['gpus']) != 8:
            raise RuntimeError('Actual NFS and8GPU required')
        if hw['memory']['cache_aware_admission']['headroom_gib'] < 384:
            raise RuntimeError('Diagnostic memory headroom unavailable')
        # Reserve future main NFS peak, plus this retained96GiB diagnostic.
        if shutil.disk_usage(root).free < (437+96+200)*GIB:
            raise RuntimeError('NFS diagnostic would consume future main reserve')
        scratch = Path(hw['storage']['scratch_dir'])/root.name
        if hw['storage']['scratch_parent']['free_bytes'] < (635+1024+200+8)*GIB:
            raise RuntimeError('Local diagnostic would consume history/temp/reserve')
        scratch.mkdir(exist_ok=True)
        shutil.copytree(source/'probe', root/'probe')
        shutil.copy2(source/'expected_tokens.json', root/'expected_tokens.json')
        cfg = json.loads((source/'main.json').read_text())
        reference_training = cfg['out_dir']; cfg['out_dir'] = str(root/'training')
        for name, value in [('early.json', dict(cfg, stop_after=2)), ('main.json', cfg)]:
            atomic_json(root/name, value)
        for name in ['ledgers', 'attempts', 'checkpoint_certificates']:
            (root/name).mkdir()
        c = dict(original, configurations={n:sha256_file(root/n) for n in ['early.json','main.json']},
                 scratch=str(scratch), reference_training=reference_training,
                 scope='Four3B updates only, NFS2-save/resume4, no main authorization or deletion.')
        c['code'] = dict(original['code'], **{p:sha256_file(resolve(p)) for p in [
            'scripts/validate_nfs_checkpoint_timing.py','scripts/validate_tiered_retention_copy.py',
            'scripts/validate_actual_training.py','scripts/supervise_main_training.py']})
        c['evidence'] = dict(original['evidence'], **{artifact_path(p):sha256_file(p) for p in [
            source/'report.json',source/'contract.json',source/'.done',root/'expected_tokens.json']})
        atomic_json(root/'contract.json', c); atomic_json(root/'hardware.json', hw)
        (root/'pid').write_text(str(os.getpid()))
        env = dict(os.environ, OMP_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', RAYON_NUM_THREADS='1',
                   TMPDIR=str(scratch), SQLITE_TMPDIR=str(scratch), PYTHONUNBUFFERED='1')
        cpus = ','.join(map(str, hw['cpu']['affinity'][64:] or hw['cpu']['affinity']))
        start = time.monotonic(); peak = 0; proc = None; seals = []

        def progress(case, **kw):
            atomic_json(root/'state.json', dict(phase='running',case=case,elapsed_seconds=time.monotonic()-start,**kw))

        def launch(case, command):
            nonlocal proc, peak
            with (root/'attempts'/(case+'.log')).open('w') as log:
                proc = subprocess.Popen(command, env=env, cwd=resolve('.'), stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                peak = max(peak, process_tree_rss(os.getpid()))
                progress(case, worker_pid=proc.pid, peak_rss_bytes=peak)
                if time.monotonic()-start > 1200 or peak > 256*GIB:
                    raise RuntimeError('Diagnostic elapsed/RSS budget')
                if size_tree(root)>96*GIB or size_tree(scratch)>8*GIB or shutil.disk_usage(root).free<(437+200)*GIB:
                    raise RuntimeError('Diagnostic disk/future-main budget')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError('Diagnostic phase failed: '+case)

        try:
            with acquire_idle_gpu(progress):
                for phase, end in [('early',2), ('main',4)]:
                    launch(phase, ['taskset','-c',cpus,sys.executable,'-m','torch.distributed.run',
                        '--standalone','--nproc_per_node=8','--module','train.monitored',
                        '--run-dir',str(root),'--config',str(root/(phase+'.json')),'--attempt',phase])
                    checkpoint = root/'training'/f'ckpt_{end:07d}'
                    cid = json.loads((checkpoint/'meta.json').read_text())['training_contract']['id']
                    t = time.monotonic()
                    seal_checkpoint(checkpoint,root/'checkpoint_certificates'/(checkpoint.name+'.json'),cid)
                    validate_optimizer_map(dcp.FileSystemReader(checkpoint).read_metadata())
                    seals.append(dict(step=end,seconds=time.monotonic()-t))
            launch('compare_states', ['taskset','-c',cpus,sys.executable,str(Path(__file__).resolve()),
                                      '--run-dir',str(root),'--compare-states'])
            comparison = json.loads((root/'state_comparison.json').read_text())
            if comparison['tensors'] != 1020 or not comparison['exact']:
                raise ValueError('Actual1020tensor exact comparison missing')
            trajectory = compare_trajectories(Path(reference_training)/'log.jsonl',root/'training/log.jsonl')
            if trajectory['loss_max_abs'] != 0:
                raise ValueError('NFS trajectory differs from local reference')
            paths = list((root/'ledgers').glob('*.json'))
            counts = aggregate_ledgers(paths,world=8,steps=4,seq_len=4096,global_tokens=2097152,
                                       sources=json.loads(resolve(cfg['mix']).read_text())['sources'])
            generations = verify_generations(paths,[2,4])
            curves = trace_review(root/'training/log.jsonl',cfg,verify_probe(root/'probe'),end_step=4)
            if counts['source_tokens'] != reference['counts']['source_tokens'] or generations != reference['counts']['generation_sha256_by_step']:
                raise ValueError('NFS/local source or generation decisions differ')
            if {p.name for p in complete_checkpoints(root/'training')} != {'ckpt_0000002','ckpt_0000004'}:
                raise ValueError('Unexpected retained diagnostic checkpoint coverage')
            check_contract(c)
            evidence_paths = [root/'training/log.jsonl',root/'checkpoint_timing.jsonl',root/'state_comparison.json',
                              *paths,*sorted((root/'checkpoint_certificates').glob('*.json'))]
            report = dict(status='passed_actual_nfs_save_resume_comparison',counts=counts,curves=curves,
                comparison=comparison,trajectory=trajectory,certification_timings=seals,
                elapsed_seconds=time.monotonic()-start,peak_host_rss_bytes=peak,retained_bytes=size_tree(root),
                evidence={artifact_path(p):sha256_file(p) for p in evidence_paths},
                limitations=['Two early-step NFS saves, not long-run p95 or cold-server throughput.',
                    'DCP payload timings exclude later RNG/meta; certification measured separately.',
                    'No concurrent tiered history copy during this diagnostic; production contention remains possible.'])
            atomic_json(root/'report.json',report)
            atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(root/'contract.json')))
            atomic_json(root/'state.json',dict(phase='complete'))
            return report
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                stop(proc)
            atomic_json(root/'state.json',dict(phase='error',error=str(exc)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--compare-states',action='store_true')
    args = parser.parse_args(); root = resolve(args.run_dir)
    if args.compare_states:
        compare_states(root)
    else:
        print(json.dumps(run(root)))
