"""Local state-change monitor. Writes events; does NOT send chat/push notifications.
Run: python -u scripts/watch_data_pipeline.py [--once] [--interval 30]
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT, DATA

ROOT = Path(PROJECT_ROOT)
LOG = ROOT / 'logs'


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except (FileNotFoundError, ProcessLookupError):
        return False


def tail(path):
    if not path.exists():
        return ''
    with path.open('rb') as f:
        f.seek(max(0, path.stat().st_size - 32768))
        return f.read().decode('utf-8', errors='replace')


def classify(complete, running, log, waiting=False):
    if complete:
        return 'complete'
    if any(x in log for x in ('Traceback (most recent call last)', 'BrokenProcessPool:', 'non-finite loss, abort')):
        return 'error'
    if not running:
        return 'unexpected_exit'
    return 'waiting' if waiting else 'running'


def verified_retired_payloads(root, paths):
    """Only a content-bound completed eviction can explain missing payloads."""
    from common.data_contracts import sha256_file
    run = Path(root) / 'runs/data_eviction_v1'
    try:
        marker = json.loads((run / '.done').read_text())
        report_path = run / 'eviction_receipt.json'
        if sha256_file(report_path) != marker['report_sha256']:
            return False
        report = json.loads(report_path.read_text())
        manifest_path = run / 'approved_manifest.json'
        if (report['status'] != 'completed_approved_payload_eviction'
                or sha256_file(manifest_path) != report['manifest_sha256']):
            return False
        approved = json.loads(manifest_path.read_text())['files']
        approved_paths = {Path(root) / e['path'] for e in approved}
        return bool(paths) and set(paths) <= approved_paths and all(
            not p.exists() and not p.is_symlink() for p in paths)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def registered_job(registration):
    """Small local heartbeat records; never reads corpus data on each poll."""
    from common.data_contracts import resolve
    state_path=resolve(registration['state_file']);pid_path=resolve(registration['pid_file'])
    pid=int(pid_path.read_text()) if pid_path.exists() else 0
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    phase=state.get('phase');complete=phase=='complete' and (state_path.parent/'.done').is_file()
    if phase in ('error','paused'):kind=phase
    elif phase=='complete' and not complete:kind='error'
    else:kind=classify(complete,alive(pid),tail(resolve(registration['log_file'])))
    if kind=='running' and state_path.exists() and time.time()-state_path.stat().st_mtime>1200:kind='stalled'
    return dict(state=kind,done=state.get('done',0),total=state.get('total',registration.get('total',0)),pid=pid,
                phase=phase,active_file=state.get('active_file'),processed_rows=state.get('processed_rows',0),run_id=registration['run_id'])


def snapshot():
    result = {}
    for source, stem in [('hplt_ko', 'clean_hplt_ko'), ('owm', 'clean_owm'),
                         ('fineweb_edu', 'clean_fineweb_edu_followup'),
                         ('github_code_v1', 'clean_github_code_v1_followup')]:
        manifest = Path(DATA) / 'raw' / source / 'download_manifest.json'
        names = json.loads(manifest.read_text())['files'] if manifest.exists() else []
        directory = Path(DATA) / 'clean' / source
        count = sum((directory / (Path(name).name + '.done')).exists() for name in names)
        pidfile = LOG / (stem + '.pid')
        pid = int(pidfile.read_text()) if pidfile.exists() else 0
        log = tail(LOG / (stem + '.log'))
        state = classify(bool(names) and count == len(names), alive(pid), log,
                         'START ' not in log and ('WAIT english' in log or 'WAIT code' in log))
        partials = list(directory.glob('*.tmp')) if directory.exists() else []
        ages = [time.time() - p.stat().st_mtime for p in partials if p.exists()]
        if state == 'running' and ages and min(ages) > 1200:
            state = 'stalled'
        result[source] = dict(state=state, done=count, total=len(names), pid=pid)
    raw = Path(DATA) / 'raw/github_code_v1'
    download_manifest = raw / 'download_manifest.json'
    if download_manifest.exists():
        names = json.loads(download_manifest.read_text())['files']
        count = sum((raw / n).exists() for n in names)
        pidfile = LOG / 'download_github_code_v1.pid'
        pid = int(pidfile.read_text()) if pidfile.exists() else 0
        log = tail(LOG / 'download_github_code_v1.log')
        complete = bool(names) and count == len(names) and 'done github_code_v1:' in log
        result['code_download'] = dict(state=classify(complete, alive(pid), log),
                                       done=count, total=len(names), pid=pid)
        if (not alive(pid) and 'done github_code_v1:' in log
                and verified_retired_payloads(ROOT, [raw / n for n in names])):
            result['code_download'].update(state='complete',
                payload_state='retired_after_verified_completion',
                historical_completed_files=len(names),
                evidence='runs/data_eviction_v1/eviction_receipt.json')
    directory = Path(DATA) / 'dedup/exact_v1'
    pidfile = LOG / 'dedup_followup.pid'
    pid = int(pidfile.read_text()) if pidfile.exists() else 0
    log = tail(LOG / 'dedup_followup.log')
    result['exact_dedup'] = dict(state=classify((directory / '.done').exists(), alive(pid), log,
                                  'START ' not in log and 'WAIT dedup' in log),
                                done=len(list(directory.glob('clean/*/*.done'))),
                                total=3 + 25 + 464 + 56 + 114, pid=pid)
    cross_pidfile = LOG / 'probe_code_math_overlap.pid'
    if cross_pidfile.exists():
        cross = Path(DATA) / 'samples/code_math_overlap_v1'
        pid = int(cross_pidfile.read_text())
        log = tail(LOG / 'probe_code_math_overlap.log')
        code_done = 64 if (cross / 'code_index.done').exists() else 0
        if not code_done:
            for line in log.splitlines():
                if line.startswith('CODE '):
                    code_done = int(line.split()[1].split('/')[0])
        count = code_done + len(list(cross.glob('*.parquet.json')))
        result['code_math_cross_check'] = dict(state=classify((cross / '.done').exists(), alive(pid), log),
                                               done=count, total=178, pid=pid)
    registry=LOG/'source_audit_job.json'
    if registry.exists():result['source_audit']=registered_job(json.loads(registry.read_text()))
    from scripts.job_registry import load_jobs
    result.update(load_jobs(LOG/'independent_jobs.json',alive,tail))
    return result


def transitions(previous, current):
    events = []
    for name, value in current.items():
        old = previous.get(name)
        if old is None or old['state'] != value['state']:
            events.append(dict(stage=name, event='state_changed', previous=old['state'] if old else None,
                               state=value['state'], done=value['done'], total=value['total']))
        elif value.get('run_id') != old.get('run_id') or value.get('case') != old.get('case'):
            events.append(dict(stage=name,event='job_changed',state=value['state'],run_id=value.get('run_id'),case=value.get('case')))
        elif value['state'] == 'running' and value['total']:
            before = old['done'] * 4 // value['total']
            now = value['done'] * 4 // value['total']
            if now > before:
                events.append(dict(stage=name, event='progress_milestone', state='running',
                                   done=value['done'], total=value['total']))
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=float, default=30)
    a = ap.parse_args()
    if a.interval < 1:
        ap.error('interval must be at least 1 second')
    LOG.mkdir(exist_ok=True)
    lock = (LOG / 'pipeline_watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statefile = LOG / 'pipeline_state.json'
    old = json.loads(statefile.read_text()) if statefile.exists() else {'stages': {}}
    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            stages = snapshot()
            events = transitions(old['stages'], stages)
            free = shutil.disk_usage(DATA).free // 2**30
            if free < 200 and old.get('disk_free_gib', 10000) >= 200:
                events.append(dict(event='low_disk', disk_free_gib=free))
            cg = Path('/sys/fs/cgroup/memory.events')
            memory = dict(line.split() for line in cg.read_text().splitlines()) if cg.exists() else {}
            oom = int(memory.get('oom_kill', 0))
            if oom > old.get('oom_kill', oom):
                events.append(dict(event='oom_kill', previous=old.get('oom_kill'), count=oom))
            with (LOG / 'pipeline_events.jsonl').open('a') as f:
                for event in events:
                    record = dict(utc=now, **event)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    print(json.dumps(record, ensure_ascii=False), flush=True)
            old = dict(utc=now, stages=stages, disk_free_gib=free, oom_kill=oom)
            tmp = statefile.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(old, ensure_ascii=False, indent=2))
            tmp.replace(statefile)
        except Exception as exc:
            # Surface monitor failure without dumping arbitrary log contents/secrets.
            print('MONITOR_ERROR', type(exc).__name__, flush=True)
            raise
        from scripts.job_registry import TERMINAL
        if a.once or all(v['state'] in TERMINAL for v in stages.values()):
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
