"""Execute only the two content-pinned payload lists explicitly approved by the user."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import stat
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json, sha256_file
from scripts.resource_admission import inspect

PINNED = {
    'candidate_manifest.json': '158aa0554d2cb3e3ce9308ab5ebb1edfd56a3a5c97208bba529b681aebbde86a',
    'additional_candidates_verified.json': '58d227d9941bfeafa6111809833468305a945ea9df903ed68aa556459e773c7a',
    'recovery_bundle.tar.gz': '5101bde4404b525e4bdd51e0a7a2af304b6b65033298eaf5880d03b59cd71677',
}


def checked(root, entry, content=False):
    rel = Path(entry['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe relative path')
    p = root / rel
    for ancestor in [p, *p.parents]:
        if ancestor == root: break
        if ancestor.is_symlink(): raise ValueError('Symlink refused: ' + str(rel))
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode): raise ValueError('Not a regular file')
    if (s.st_size, s.st_mtime_ns, s.st_ino) != (entry['bytes'], entry['mtime_ns'], entry['inode']):
        raise ValueError('File changed: ' + str(rel))
    if content:
        if sha256_file(p) != entry['sha256']: raise ValueError('Content changed: ' + str(rel))
        checked(root, entry)
    return p


def remove_checked(root, entry):
    p = checked(root, entry)
    fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        s = os.stat(p.name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISREG(s.st_mode) or (s.st_size, s.st_mtime_ns, s.st_ino) != (entry['bytes'], entry['mtime_ns'], entry['inode']):
            raise ValueError('File changed before unlink')
        os.unlink(p.name, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)


def self_test():
    with tempfile.TemporaryDirectory(prefix='eviction-guard-') as d:
        root = Path(d); p = root / 'a'; p.write_bytes(b'original')
        s = p.stat(); e = dict(path='a', bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                              inode=s.st_ino, sha256=sha256_file(p))
        checked(root, e, True)
        p.write_bytes(b'changed!'); os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns))
        try: checked(root, e, True)
        except ValueError: pass
        else: raise AssertionError('Same-size corruption accepted')
        p.unlink(); p.symlink_to(root / 'untouched')
        try: remove_checked(root, e)
        except ValueError: pass
        else: raise AssertionError('Symlink accepted')
        try: checked(root, dict(e, path='../outside'))
        except ValueError: pass
        else: raise AssertionError('Traversal accepted')
        p.unlink(); p.write_bytes(b'new-file')
        try: remove_checked(root, e)
        except ValueError: pass
        else: raise AssertionError('Replaced file accepted')
        s = p.stat(); e.update(bytes=s.st_size, inode=s.st_ino, mtime_ns=s.st_mtime_ns, sha256=sha256_file(p))
        sentinel = root / 'keep'; sentinel.write_bytes(b'keep')
        remove_checked(root, e)
        assert not p.exists() and sentinel.read_bytes() == b'keep'
    print('Guard checks passed: corruption, symlink, traversal, replacement, exact-file unlink.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    if a.self_test: self_test(); return
    if not a.execute: ap.error('Use --execute only for the user-approved pinned lists')
    root = Path(PROJECT_ROOT).resolve(); pack = root / 'docs/recovery/data_retention_v1'
    run = root / 'runs/data_eviction_v1'; run.mkdir(exist_ok=True)
    lock = (run / '.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, digest in PINNED.items():
        if sha256_file(pack / name) != digest: raise ValueError('Approved artifact changed: ' + name)
    for name, digest in json.loads((pack / 'CHECKSUMS.json').read_text()).items():
        if sha256_file(pack / name) != digest: raise ValueError('Recovery bundle evidence changed')
    entries = json.loads((pack / 'candidate_manifest.json').read_text())
    extra = json.loads((pack / 'additional_candidates_verified.json').read_text())['files']
    entries += extra
    assert len(entries) == 1394 and len({x['path'] for x in entries}) == 1394
    assert sum(x['bytes'] for x in entries) == 673858637616
    approved = dict(user_instruction='추가 후보까지 포함 삭제하자', artifacts=PINNED,
                    files=entries, scope='1394 listed payload files only; no directory or evidence deletion')
    if (run / 'approved_manifest.json').exists():
        assert json.loads((run / 'approved_manifest.json').read_text()) == approved
    else: atomic_json(run / 'approved_manifest.json', approved)
    if (run / '.done').exists():
        assert json.loads((run / '.done').read_text())['report_sha256'] == sha256_file(run / 'eviction_receipt.json')
        assert all(not os.path.lexists(root / e['path']) for e in entries)
        print('Previously completed eviction verified; no further deletion.'); return
    hardware = inspect(); atomic_json(run / 'hardware.json', hardware)
    assert hardware['memory']['cache_aware_admission']['headroom_gib'] >= 32
    cpus = sorted(os.sched_getaffinity(0))[-32:]; os.sched_setaffinity(0, cpus)
    os.environ.update(OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    atomic_json(run / 'resources.json', dict(cpu_affinity=cpus, hash_workers=32,
                max_rss_gib=32, time_limit_seconds=600, gpu_work=False,
                tuning='Reuse verified same-host 32-worker SHA; original data and outputs stay on NFS.'))
    (run / 'pid').write_text(str(os.getpid()))
    started = time.monotonic()
    def state(phase, **kw):
        if time.monotonic()-started > 600: raise TimeoutError('Eviction time budget')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 32*1024**2: raise MemoryError('RSS budget')
        atomic_json(run/'state.json', dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **kw))
        print(phase, json.dumps(kw), flush=True)
    paths = {str(root/e['path']) for e in entries}
    protected_roots = ['data/tokens/final_v1', 'data/dedup/exact_v1', 'data/clean/github_code_v1',
                       'data/datasets', 'tokenizer/candidate_v1', 'data/recovery', 'docs/recovery/data_retention_v1']
    protected = set()
    for name in protected_roots:
        protected.update(p for p in (root/name).rglob('*') if p.is_file())
    protected.update(p for p in (root/'data/raw').rglob('*') if p.is_file() and p.name.endswith('.json'))
    protected.update(p for p in (root/'data/clean').rglob('*') if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.done')))
    assert not paths & {str(p) for p in protected}
    def snapshot():
        return {str(p.relative_to(root)): [p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino]
                for p in sorted(protected)}
    before = snapshot()
    if (run/'protected_before.json').exists(): assert json.loads((run/'protected_before.json').read_text()) == before
    else: atomic_json(run/'protected_before.json', before)
    contracts = ['runs/1b_rehearsal_final_v3/contract.json','runs/rehearsal_final_native_v1/contract.json',
                 'runs/monitored_3b_validation_v1/contract.json','runs/pretrain_final_review_v1/contract.json']
    for name in contracts:
        if not (root/name).exists(): continue
        c=json.loads((root/name).read_text())
        for path,digest in c.get('code',{}).items(): assert sha256_file(root/path)==digest, path
        for section in ['inputs','evidence']:
            assert not paths & {str((root/p).resolve()) for p in c.get(section,{})}
    # Reconcile final ready receipts before unlink; this path has already passed candidate-unavailable testing.
    from data.quality_report import assess
    quality = assess('data/datasets/final_v1.candidate.json','data/dedup/exact_v1','runs/f2_execution_v1/final_policy.json')
    assert quality['readiness']=='passed'
    atomic_json(run/'quality_before.json', quality)
    journal = run/'journal.jsonl'
    history=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    intents={x['path'] for x in history if x['event']=='unlink_intent'}
    missing={e['path'] for e in entries if not os.path.lexists(root/e['path'])}
    assert missing <= intents, 'Unexpected missing file without prior durable unlink intent'
    remaining=[e for e in entries if e['path'] not in missing]
    state('hashing', total=len(remaining))
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i,f in enumerate(as_completed([pool.submit(checked,root,e,True) for e in remaining]),1):
            f.result()
            if i%200==0:state('hashing',done=i,total=len(remaining))
    # Scan after all hash handles have closed. Only known init/auth daemons may be uninspectable.
    opened=[]; denied=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            if any(x.decode(errors='replace').endswith('/'+n) or x.decode(errors='replace')==n
                   for x in argv for n in ['data/download.py','data/clean.py','data/clean_code.py','data/dedup.py']):
                raise RuntimeError('Upstream producer active')
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd).removesuffix(' (deleted)')
                    if target in paths:opened.append(dict(pid=int(proc.name),path=target,via='fd'))
                except FileNotFoundError:pass
            for line in (proc/'maps').read_text().splitlines():
                parts=line.split(None,5)
                if len(parts)==6 and parts[5] in paths:opened.append(dict(pid=int(proc.name),path=parts[5],via='mmap'))
        except PermissionError:
            comm=(proc/'comm').read_text().strip();denied.append(dict(pid=int(proc.name),comm=comm))
            if comm not in ['docker-init','ssh-agent']:raise RuntimeError('Uninspectable non-system process')
        except (FileNotFoundError,ProcessLookupError):pass
    assert not opened,opened
    atomic_json(run/'preunlink_check.json',dict(opened=opened,uninspectable_system_daemons=denied,
                content_verified_files=len(remaining),previous_intent_missing=len(missing),protected_files=len(protected)))
    free_before=shutil.disk_usage(root).free
    atomic_json(run/'execution_start.json',dict(utc=datetime.now(timezone.utc).isoformat(),free_bytes=free_before,
                training_log_tail=(root/'runs/1b_rehearsal_final_v3/training/log.jsonl').read_text().splitlines()[-1]))
    state('deleting',total=len(remaining))
    with journal.open('a') as stream:
        def record(event,e):
            stream.write(json.dumps(dict(event=event,path=e['path'],sha256=e['sha256'],utc=datetime.now(timezone.utc).isoformat()))+'\n')
            stream.flush();os.fsync(stream.fileno())
        for i,e in enumerate(remaining,1):
            checked(root,e)
            record('unlink_intent',e)
            remove_checked(root,e)
            record('unlinked',e)
            if i%200==0:state('deleting',done=i,total=len(remaining))
    assert all(not os.path.lexists(root/e['path']) for e in entries)
    assert snapshot()==before,'Protected artifact stat changed'
    after_quality=assess('data/datasets/final_v1.candidate.json','data/dedup/exact_v1','runs/f2_execution_v1/final_policy.json')
    assert after_quality==quality,'Final source receipt consumer changed'
    atomic_json(run/'quality_after.json',after_quality)
    free_after=shutil.disk_usage(root).free
    report=dict(status='completed_approved_payload_eviction',utc=datetime.now(timezone.utc).isoformat(),
                user_instruction=approved['user_instruction'],deleted_files=len(entries),
                logical_bytes=sum(e['bytes'] for e in entries),allocated_bytes=sum(e['allocated_bytes'] for e in entries),
                free_bytes_before=free_before,free_bytes_after=free_after,observed_free_delta_bytes=free_after-free_before,
                protected_file_count=len(protected),protected_stats_unchanged=True,source_receipt_consumer_unchanged=True,
                manifest_sha256=sha256_file(run/'approved_manifest.json'),journal_sha256=sha256_file(journal),
                elapsed_seconds=time.monotonic()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                code_sha256=sha256_file(Path(__file__)),training_log_tail=(root/'runs/1b_rehearsal_final_v3/training/log.jsonl').read_text().splitlines()[-1],
                scope='Listed payloads only. Directories, metadata, final source/tokens/split and checkpoints retained. No remote upload or main launch.')
    atomic_json(run/'eviction_receipt.json',report)
    atomic_json(run/'.done',dict(report_sha256=sha256_file(run/'eviction_receipt.json')))
    state('complete',done=len(entries),total=len(entries))
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
