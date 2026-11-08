"""Bound a local split publication by elapsed time, RSS and per-mount space.

Limits pause the worker and preserve transaction/staging artifacts for same-command
resume. No corpus deletion and no implicit next-stage tokenizer/training launch.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve, atomic_json, sha256_file
from scripts.hardware_profile import inspect

GIB = 1024**3


def size_tree(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass
    return total


def process_tree_rss(pid):
    total = 0
    pending = [pid]
    while pending:
        child = pending.pop()
        try:
            # All descendants of this job only, not other CPU/GPU jobs.
            pending.extend(int(p) for p in Path(f'/proc/{child}/task/{child}/children').read_text().split())
            for line in Path(f'/proc/{child}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    total += int(line.split()[1]) * 1024
        except FileNotFoundError:
            pass
    return total


def violation(elapsed, rss, local_bytes, durable_bytes, local_free, durable_free, limits):
    checks = [(elapsed > limits['max_seconds'], 'time_budget'),
              (rss > limits['max_rss_gib']*GIB, 'rss_budget'),
              (local_bytes > limits['local_budget_gib']*GIB, 'local_disk_budget'),
              (durable_bytes > limits['durable_budget_gib']*GIB, 'durable_disk_budget'),
              (min(local_free, durable_free) < limits['reserve_gib']*GIB, 'disk_reserve')]
    return next((name for failed, name in checks if failed), None)


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run(args):
    limits = {k: getattr(args, k) for k in ['max_seconds', 'max_rss_gib', 'local_budget_gib',
                                          'durable_budget_gib', 'reserve_gib']}
    if any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Finite nonnegative budgets required')
    if min(args.max_seconds, args.max_rss_gib, args.local_budget_gib, args.durable_budget_gib) <= 0:
        raise ValueError('Positive execution budgets required')
    root = resolve(args.run_dir).resolve()
    out = resolve(args.output_dir).resolve()
    scratch = resolve(args.scratch_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hardware = inspect()
        if not hardware['presets']['launch_permitted']:
            raise RuntimeError('Insufficient fresh RAM headroom')
        attempt = len(list(root.glob('attempt_*.json'))) + 1
        atomic_json(root/f'attempt_{attempt:03d}.json', dict(limits=limits, hardware=hardware,
                    code_sha256=sha256_file(Path(__file__)), started_unix=time.time()))
        cmd = [sys.executable, 'scripts/run_with_resources.py', '--stage', 'sqlite', '--',
               sys.executable, 'scripts/publish_local_split.py']
        for key in ['sources_manifest', 'output_dir', 'run_dir', 'scratch_dir', 'seed',
                    'eval_fraction', 'sqlite_cache_mib', 'reserve_gib', 'local_budget_gib',
                    'durable_budget_gib']:
            cmd += ['--'+key.replace('_', '-'), str(getattr(args, key))]
        start = time.monotonic()
        peak_rss = peak_local = peak_durable = 0
        reason = None
        with (root/f'console_{attempt:03d}.log').open('w') as log:
            proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        try:
            while proc.poll() is None:
                elapsed = time.monotonic()-start
                rss = process_tree_rss(proc.pid)
                local = size_tree(scratch)
                durable = size_tree(out) + size_tree(out.with_name(out.name+'.publishing'))
                peak_rss, peak_local, peak_durable = max(rss, peak_rss), max(local, peak_local), max(durable, peak_durable)
                reason = violation(elapsed, rss, local, durable, shutil.disk_usage(scratch).free,
                                   shutil.disk_usage(out.parent).free, limits)
                state = dict(phase='running', attempt=attempt, pid=proc.pid, elapsed_seconds=elapsed,
                             peak_rss_bytes=peak_rss, peak_local_bytes=peak_local,
                             peak_durable_bytes=peak_durable)
                worker = root/'worker_state.json'
                if worker.exists():
                    state['worker'] = json.loads(worker.read_text())
                atomic_json(root/'state.json', state)
                if reason:
                    stop(proc)
                    break
                time.sleep(2)
        except BaseException:
            stop(proc)
            atomic_json(root/'state.json', dict(phase='paused', reason='controller_interrupted'))
            raise
        status = 'paused' if reason else 'complete' if proc.returncode == 0 else 'error'
        if status == 'complete':
            marker = json.loads((root/'.done').read_text())
            if (marker['report_sha256'] != sha256_file(root/'report.json') or
                    marker['contract_sha256'] != sha256_file(root/'contract.json')):
                status = 'error'
                reason = 'completion_content_mismatch'
        final = dict(phase=status, attempt=attempt, returncode=proc.returncode, reason=reason,
                     elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss,
                     peak_local_bytes=peak_local, peak_durable_bytes=peak_durable)
        atomic_json(root/f'result_{attempt:03d}.json', final)
        atomic_json(root/'state.json', final)
        print(json.dumps(final), flush=True)
        return 0 if status == 'complete' else 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['sources-manifest', 'output-dir', 'run-dir', 'scratch-dir']:
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--eval-fraction', type=float, default=.01)
    ap.add_argument('--sqlite-cache-mib', type=int, default=512)
    ap.add_argument('--reserve-gib', type=float, default=200)
    ap.add_argument('--local-budget-gib', type=float, default=300)
    ap.add_argument('--durable-budget-gib', type=float, default=160)
    ap.add_argument('--max-seconds', type=float, default=72*3600)
    ap.add_argument('--max-rss-gib', type=float, default=12)
    sys.exit(run(ap.parse_args()))
