"""Overlap fresh CPU verification late in rehearsal, then score its final1B.

Never trains a model or prunes data. Existing native scoring is reused unchanged.
An independent controller supervises time/RSS/disk and polls locally every30s.
Restarting refreshes the physical source/split context, never a persistent skip.
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
from common.data_contracts import resolve, artifact_path, atomic_json, sha256_file
from scripts.run_native_evaluation import certified

GIB = 2**30


def check_bindings(contract):
    for path, digest in {**contract['code'], **contract['inputs']}.items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Final evaluation input/implementation changed: '+path)


def rehearsal_observation(root):
    root = resolve(root)
    if (root/'.done').exists():
        return dict(complete=True, step=5000, ready=True)
    state = json.loads((root/'state.json').read_text())
    if state.get('phase') in ('error', 'failed', 'paused'):
        raise RuntimeError('Rehearsal requires recovery: '+str(state))
    pid = int((root/'pid').read_text())
    try:
        os.kill(pid, 0)
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
    except (ProcessLookupError, FileNotFoundError) as exc:
        # Resolve the normal .done vs controller-exit publication race.
        if (root/'.done').exists():
            return dict(complete=True, step=5000, ready=True)
        raise RuntimeError('Rehearsal controller handle is missing') from exc
    if b'scripts/run_rehearsal.py' not in cmd or root.name.encode() not in cmd:
        raise RuntimeError('Rehearsal controller PID identity differs')
    rows = []
    path = root/'training/log.jsonl'
    lines = path.read_text().splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines)-1:
                continue  # A concurrently appended, incomplete final row only.
            raise
        if 'tok_s' in row and row['step'] >= 20:
            if not math.isfinite(row['tok_s']) or row['tok_s'] <= 0:
                raise ValueError('Invalid training throughput')
            rows.append(row)
    step = rows[-1]['step'] if rows else 0
    rate = statistics.median(r['tok_s'] for r in rows[-6:]) if rows else None
    eta = (5000-step)*2097152/rate if rate else None
    return dict(complete=False, step=step, recent_tokens_s=rate,
                remaining_compute_seconds=eta,
                ready=step >= 4000 or (eta is not None and eta <= 5400))


def completed_rehearsal(root):
    root = resolve(root)
    status = json.loads((root/'report.json').read_text())['status']
    if status not in ('passed_actual_rehearsal_training', 'needs_review'):
        raise ValueError('Rehearsal did not finish its actual training')
    report, contract = certified(root, status)
    if report['counts']['committed_tokens'] != 5000*2097152:
        raise ValueError('Full rehearsal token count differs')
    checkpoint = root/'training/ckpt_0005000'
    if resolve(report['final_checkpoint']) != checkpoint:
        raise ValueError('Final checkpoint path differs')
    for path, digest in report['evidence'].items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Rehearsal evidence changed: '+path)
    certificate = root/'checkpoint_certificates/ckpt_0005000.json'
    if report['evidence'].get(artifact_path(certificate)) != sha256_file(certificate):
        raise ValueError('Final checkpoint certificate is not content bound')
    proof = json.loads(certificate.read_text())
    meta = json.loads((checkpoint/'meta.json').read_text())
    if (proof['status'] != 'passed_checkpoint_storage' or proof['step'] != 5000 or
            resolve(proof['checkpoint']) != checkpoint or
            proof['training_contract_id'] != report['counts']['contract_id'] or
            meta['training_contract']['id'] != proof['training_contract_id'] or
            meta['step'] != 5000 or meta['config']['model'] != '1b' or
            meta['config']['max_steps'] != 5000 or not (checkpoint/'.complete').is_file()):
        raise ValueError('Final rehearsal checkpoint identity differs')
    return report, checkpoint, proof, meta


def verify_final_result(result, proof, tokenizer_sha256, expected_targets, reference):
    # Native run() freshly hashes all physical checkpoint files once. Reuse those
    # hashes to compare with the rehearsal certificate; avoid a second full pass.
    expected = {r['path']:r['sha256'] for r in proof['files']}
    if len(expected) != len(proof['files']) or result['contract']['checkpoint'] != expected:
        raise ValueError('Final evaluated checkpoint content differs')
    c = result['contract']
    if (result['status'] != 'passed' or result['checkpoint_step'] != 5000 or
            c['tokenizer_sha256'] != tokenizer_sha256 or
            c['input_provenance'].get('level') != 'verified-eval-split' or
            c['input_provenance'] != reference['contract']['input_provenance']):
        raise ValueError('Final evaluation provenance/step/tokenizer differs')
    for key in ('samples_sha256', 'prompts_sha256', 'seq_len', 'stride',
                'max_new_tokens', 'seed', 'temperature', 'template', 'append_eos_for_ppl'):
        if c[key] != reference['contract'][key]:
            raise ValueError('Fixed evaluation policy differs: '+key)
    if set(result['languages']) != {'ko', 'en'}:
        raise ValueError('Both native evaluation languages required')
    for category, count in expected_targets.items():
        r = result['languages'][category]
        if (count < 1 or r['valid_tokens'] != count or not math.isfinite(r['nll_sum']) or
                not math.isclose(r['mean_nll'], r['nll_sum']/count, rel_tol=1e-12)):
            raise ValueError('Native full-target coverage or weighted NLL differs')
    if [(r['id'], r['category'], r['prompt']) for r in result['generations']] != [
            (r['id'], r['category'], r['prompt']) for r in reference['generations']]:
        raise ValueError('Fixed generation prompt coverage differs')
    if (len(result['generations']) != 6 or len({r['id'] for r in result['generations']}) != 6 or
            {r['category'] for r in result['generations']} != {'ko', 'en'}):
        raise ValueError('Six unique bilingual generation prompts required')


def acquire_idle_gpu(notify):
    lock = resolve('runs/.gpu_benchmark.lock').open('a')
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            notify('await_gpu_lock'); time.sleep(30); continue
        occupied = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if not occupied:
            return lock
        fcntl.flock(lock, fcntl.LOCK_UN)
        notify('await_idle_gpu'); time.sleep(30)


def worker(root):
    root = resolve(root)
    contract = json.loads((root/'contract.json').read_text())
    check_bindings(contract)
    evaluation = contract['evaluation']
    def notify(phase, **kw):
        atomic_json(root/'worker_state.json', dict(phase=phase, unix=time.time(), pid=os.getpid(), **kw))
    import pyarrow as pa
    pa.set_cpu_count(4); pa.set_io_thread_count(4)
    from eval.prepare_holdout import local_verified_inputs
    from data.split_inputs import verify_prepared
    from eval.run import run, records
    from tokenizers import Tokenizer
    started = time.monotonic(); notify('verify_fresh_local_eval_context')
    with local_verified_inputs(evaluation['source_manifest'], evaluation['canonical_split_manifest'],
                               evaluation['reader_manifest'], cache_mib=131072, hash_workers=32) as inputs:
        prepared = verify_prepared(contract['holdout_manifest'], 'eval', contract['samples'], verified_inputs=inputs)
        tokenizer = Tokenizer.from_file(str(resolve(contract['tokenizer'])))
        expected = {k:sum(len(tokenizer.encode(r['text'], add_special_tokens=False).ids)
                         for r in records(resolve(contract['samples'])) if r['category'] == k) for k in ('ko', 'en')}
        if expected != contract['expected_targets']:
            raise ValueError('Raw document target counts differ from fixed native input')
        atomic_json(root/'fresh_context.json', dict(seconds=time.monotonic()-started,
                    prepared_contract_id=prepared['contract_id'], raw_reencoded_targets=expected,
                    scope='Fresh physical source/split validation in this live process; no persistent validation skip.'))
        while not (resolve(contract['rehearsal_run'])/'.done').exists():
            observation = rehearsal_observation(contract['rehearsal_run'])
            notify('await_rehearsal_final_checkpoint', **observation); time.sleep(30)
        rehearsal, checkpoint, proof, meta = completed_rehearsal(contract['rehearsal_run'])
        if meta['training_contract']['mix']['tokenizer_sha256'] != contract['tokenizer_sha256']:
            raise ValueError('Rehearsal final tokenizer differs')
        check_bindings(contract)
        with acquire_idle_gpu(notify):
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            notify('native_final1b_evaluation'); started = time.monotonic()
            result = run(checkpoint, resolve(contract['tokenizer']), resolve(contract['samples']),
                         resolve(evaluation['prompts']), root/'1b_final.json', seq_len=4096, stride=2048,
                         max_new_tokens=64, seed=evaluation['seed'], temperature=0, device='cuda',
                         holdout_manifest=resolve(contract['holdout_manifest']), verified_inputs=inputs)
            reference = json.loads(resolve(contract['reference_report']).read_text())
            verify_final_result(result, proof, contract['tokenizer_sha256'], expected, reference)
            seconds = time.monotonic()-started
    check_bindings(contract)
    report = dict(status='passed_actual_trained_native_evaluation',
                  result='1b_final.json', result_sha256=sha256_file(root/'1b_final.json'),
                  rehearsal_report_sha256=sha256_file(resolve(contract['rehearsal_run'])/'report.json'),
                  rehearsal_status=rehearsal['status'], languages=result['languages'], elapsed_evaluation_seconds=seconds,
                  scope='Actual final5000-step1B native ko/en full-target PPL and fixed6 base prompts. Execution/coverage passed; quality review remains separate. Official benchmarks not run.')
    atomic_json(root/'report.json', report)
    atomic_json(root/'.done', dict(report_sha256=sha256_file(root/'report.json'), contract_sha256=sha256_file(root/'contract.json')))
    notify('complete')


def make_contract(a):
    native, nc = certified(a.native_run, 'passed_actual_native_evaluation')
    f6, fc = certified(a.token_run, 'passed')
    root = resolve(a.rehearsal_run); rc = json.loads((root/'contract.json').read_text())
    for path, digest in rc['code'].items():
        if sha256_file(resolve(path)) != digest:
            raise ValueError('Active rehearsal code changed: '+path)
    reference = resolve(a.native_run)/'1b.json'
    if sha256_file(reference) != native['results']['1b']['report_sha256']:
        raise ValueError('Reference native evaluation content differs')
    old = json.loads(reference.read_text())
    token = resolve(fc['settings']['tokenizer_run'])/'report.json'
    trained = json.loads(token.read_text())
    if trained['tokenizer_sha256'] != f6['tokenizer_sha256'] or native['tokenizer_sha256'] != f6['tokenizer_sha256']:
        raise ValueError('Final tokenizer lineage differs')
    prepared = resolve(a.native_run)/'holdout/manifest.json'; samples = prepared.parent/'samples.jsonl'
    bindings = [root/'contract.json', resolve(a.native_run)/'report.json', resolve(a.native_run)/'.done',
                resolve(a.native_run)/'contract.json', reference, prepared, samples, token,
                resolve(a.token_run)/'report.json', resolve(a.token_run)/'.done',
                resolve(a.token_run)/'contract.json', resolve(trained['tokenizer'])]
    bindings += [resolve(nc['evaluation'][k]) for k in ('source_manifest', 'canonical_split_manifest', 'reader_manifest', 'prompts')]
    return dict(version=1, rehearsal_run=artifact_path(root), evaluation=nc['evaluation'],
                tokenizer=trained['tokenizer'], tokenizer_sha256=f6['tokenizer_sha256'],
                reference_report=artifact_path(reference), holdout_manifest=artifact_path(prepared), samples=artifact_path(samples),
                expected_targets={k:v['valid_tokens'] for k,v in old['languages'].items()},
                inputs={artifact_path(p):sha256_file(p) for p in bindings},
                code={**rc['code'], 'scripts/evaluate_final_rehearsal.py':sha256_file(Path(__file__))},
                host_rss_gib=256, local_budget_gib=8, durable_budget_gib=8, reserve_gib=200,
                max_seconds=172800, worker_max_seconds=21600, cpu_affinity=list(range(64)),
                scope='Only final1B native evaluation and CPU preparation; no training, deletion, overwrite or upload.')


def admission(contract, root):
    from scripts.resource_admission import inspect
    from scripts.run_local_split import size_tree
    hw = inspect()
    if len(hw['gpus']) != 8 or hw['memory']['cache_aware_admission']['headroom_gib'] < 512:
        raise RuntimeError('Insufficient measured headroom for256GiB eval plus active rehearsal future peak')
    run = resolve(contract['rehearsal_run']); rc = json.loads((run/'contract.json').read_text())
    remaining = max(0, rc['forecast_checkpoint_bytes']-size_tree(run/'training'))
    if shutil.disk_usage(root).free < remaining+(200+8)*GIB:
        raise RuntimeError('Final eval plus active rehearsal future checkpoint/disk reserve unavailable')
    return hw


def reject_live_orphan(root):
    path = root/'worker_state.json'
    if path.exists():
        pid = json.loads(path.read_text()).get('pid')
        if pid:
            try:
                cmd = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
            except FileNotFoundError:
                return
            if b'evaluate_final_rehearsal.py' in cmd and b'--worker' in cmd and str(root).encode() in cmd:
                raise RuntimeError('Existing final-evaluation worker is still live; preserve it before restarting controller')


def main(a):
    if a.worker:
        return worker(a.run_dir)
    from scripts.run_local_split import process_tree_rss, size_tree, stop
    root = resolve(a.run_dir); root.relative_to(resolve('runs'))
    root.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = make_contract(a)
        if (root/'contract.json').exists():
            if json.loads((root/'contract.json').read_text()) != contract:
                raise ValueError('Final eval contract changed; use a new run path')
        else:
            atomic_json(root/'contract.json', contract)
        if (root/'.done').exists():
            return certified(root, 'passed_actual_trained_native_evaluation')
        reject_live_orphan(root)
        (root/'pid').write_text(str(os.getpid())+'\n')
        started = time.monotonic(); proc = None; peak = 0; attempt = str(time.time_ns())
        try:
            while True:
                observation = rehearsal_observation(a.rehearsal_run)
                if observation['ready']:
                    break
                atomic_json(root/'state.json', dict(phase='running', case='await_late_rehearsal', **observation))
                if time.monotonic()-started > contract['max_seconds']:
                    raise RuntimeError('Final evaluation late-trigger time budget')
                time.sleep(30)
            hw = admission(contract, root)
            atomic_json(root/('hardware_'+attempt+'.json'), hw)
            cpus = sorted(set(contract['cpu_affinity']) & set(hw['cpu']['affinity']))
            if not cpus:
                raise RuntimeError('Final evaluation CPU affinity unavailable')
            scratch = Path(hw['storage']['scratch_dir'])/root.name
            scratch.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(scratch).free < (200+8)*GIB:
                raise RuntimeError('Final evaluation local disk reserve unavailable')
            env = dict(os.environ, OMP_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', RAYON_NUM_THREADS='8',
                       TMPDIR=str(scratch), SQLITE_TMPDIR=str(scratch), PYTHONUNBUFFERED='1')
            cmd = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-u', str(Path(__file__).resolve()),
                   '--run-dir', str(root), '--worker']
            worker_started = time.monotonic()
            with (root/('console_'+attempt+'.log')).open('w') as log:
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                peak = max(peak, process_tree_rss(os.getpid()))
                w = json.loads((root/'worker_state.json').read_text()) if (root/'worker_state.json').exists() else {}
                atomic_json(root/'state.json', dict(phase='running', case=w.get('phase', 'starting'),
                            worker_pid=proc.pid, peak_rss_bytes=peak, elapsed_seconds=time.monotonic()-started))
                if peak > contract['host_rss_gib']*GIB or time.monotonic()-worker_started > contract['worker_max_seconds']:
                    raise RuntimeError('Final evaluation worker RSS/time budget')
                for path in (root, scratch):
                    if size_tree(path) > 8*GIB or shutil.disk_usage(path).free < 200*GIB:
                        raise RuntimeError('Final evaluation disk budget')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError('Final evaluation worker failed; preserve outputs and inspect console')
            certified(root, 'passed_actual_trained_native_evaluation')
            atomic_json(root/'state.json', dict(phase='complete', peak_rss_bytes=peak))
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                stop(proc)
            atomic_json(root/'state.json', dict(phase='error', error=str(exc)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rehearsal-run', default='runs/1b_rehearsal_final_v3')
    parser.add_argument('--native-run', default='runs/native_eval_validation_v1')
    parser.add_argument('--token-run', default='runs/f6_tokens_v1')
    main(parser.parse_args())
