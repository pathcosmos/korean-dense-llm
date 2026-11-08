"""Run-bound NFS latest checkpoints plus disposable local history.

No CLI and no implicit authorization. The separately authorized supervisor is
the only production caller. Publication renames are always within one device.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from common.data_contracts import sha256_file

GIB = 2**30
NAME = re.compile(r'ckpt_[0-9]{7}\Z')


def reviewed_policy(policy):
    legacy = policy.get('version') == 1 and policy.get('nfs_keep') == 10 and policy.get('local_keep') == 15
    adaptive = (policy.get('version') == 2 and policy.get('nfs_keep') == 3 and
                policy.get('local_keep') == 15 and policy.get('adaptive_local_pruning') is True)
    return legacy or adaptive


def size_tree(path):
    total = 0
    for p in Path(path).rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except FileNotFoundError:
            pass  # A concurrent publication can rename a whole subtree.
    return total


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    """A receipt is an ACK only after its bytes and parent are fsynced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    if path.is_symlink() or tmp.is_symlink():
        raise ValueError('Receipt symlink rejected')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
    fsync_dir(path.parent)


def exact_path(path):
    path = Path(path).absolute()
    if path != path.resolve():
        raise ValueError('Symlink or noncanonical retention path')
    return path


def records(proof):
    rows = proof['files']
    result = {}
    for row in rows:
        p = Path(row['path'])
        if (p.is_absolute() or '..' in p.parts or str(p) != row['path'] or
                row['path'] in result or type(row['bytes']) is not int or row['bytes'] < 0 or
                not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Unsafe checkpoint certificate file list')
        result[row['path']] = row
    required = {'.complete', '.metadata', 'meta.json', *[f'rng_rank{r}.pt' for r in range(8)]}
    if (not required <= result.keys() or not any(n.endswith('.distcp') for n in result) or
            sum(r['bytes'] for r in rows) != proof['bytes'] or len(result) != len(rows)):
        raise ValueError('Incomplete checkpoint certificate')
    return result


def inventory(path, proof, *, hashes=False, partial=False, workers=4):
    """Partial mode is only for a previously journaled deletion transaction."""
    path = exact_path(path)
    expected = records(proof)
    actual = {}
    for p in path.rglob('*'):
        if p.is_symlink() or (not p.is_dir() and not p.is_file()):
            raise ValueError('Unexpected checkpoint link/file type')
        if p.is_file():
            actual[str(p.relative_to(path))] = p
    if set(actual) - expected.keys() or (not partial and set(actual) != expected.keys()):
        raise ValueError('Checkpoint file list differs')
    def check(item):
        name, p = item
        before = p.stat()
        if before.st_size != expected[name]['bytes']:
            raise ValueError('Checkpoint size differs')
        if hashes and sha256_file(p) != expected[name]['sha256']:
            raise ValueError('Checkpoint SHA differs')
        after = p.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise ValueError('Checkpoint changed while checking')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check, actual.items()))
    if not partial:
        meta = json.loads((path / 'meta.json').read_text())
        if meta['step'] != proof['step'] or meta['training_contract']['id'] != proof['training_contract_id']:
            raise ValueError('Checkpoint metadata identity differs')
    return expected


def copy_payload(source, stage, proof, *, copy_workers=4, hash_workers=8, fault=None, cancelled=None):
    """Copy to a private stage, then verify destination against the source seal."""
    source, stage = exact_path(source), exact_path(stage)
    fault = fault or (lambda point: None)
    cancelled = cancelled or (lambda: False)
    def check_cancel():
        if cancelled():
            raise InterruptedError('Retention cancelled; preserve stage and NFS source')
    stage.mkdir(parents=True, exist_ok=True)
    expected = inventory(source, proof)
    existing = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
    if existing - expected.keys() or any(p.is_symlink() for p in stage.rglob('*')):
        raise ValueError('Unexpected staging leftovers; preserve and stop')
    start = time.monotonic()
    def copy(row):
        check_cancel()
        src, dst = source / row['path'], stage / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == row['bytes'] and sha256_file(dst) == row['sha256']:
            return
        before = src.stat()
        with src.open('rb') as inp, dst.open('wb') as out:
            while True:
                check_cancel()
                block = inp.read(8 * 1024 * 1024)
                if not block:break
                out.write(block)
            out.flush();os.fsync(out.fileno())
        after = src.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise ValueError('NFS source changed during history copy')
        fault('copied_file')
    with ThreadPoolExecutor(max_workers=copy_workers) as pool:
        list(pool.map(copy, expected.values()))
    copied = time.monotonic()
    fault('copied_payload');check_cancel()
    inventory(stage, proof, hashes=True, workers=hash_workers)
    for d in sorted([stage, *[p for p in stage.rglob('*') if p.is_dir()]], key=lambda p: len(p.parts), reverse=True):
        fsync_dir(d)
    fault('verified_copy');check_cancel()
    return dict(copy_seconds=copied-start, destination_verify_fsync_seconds=time.monotonic()-copied,
                total_seconds=time.monotonic()-start, bytes=proof['bytes'])


class TieredRetention:
    def __init__(self, root, policy, *, authorized=False, fault=None, cancelled=None):
        self.root = exact_path(root)
        self.policy = policy
        self.local = exact_path(policy['local_root'])
        self.training = self.root / 'training'
        self.authorized = authorized
        self.fault = fault or (lambda point: None)
        self.cancelled = cancelled or (lambda: False)
        if (policy['nfs_run'] != str(self.root) or not re.fullmatch('[0-9a-f]{32}', policy['run_id']) or
                not reviewed_policy(policy) or
                self.local == self.root or self.local.is_relative_to(self.root) or self.root.is_relative_to(self.local)):
            raise ValueError('Retention must bind exact independent run paths and reviewed policy')
        self.owner = dict(version=1, run_id=policy['run_id'], nfs_run=str(self.root), local_root=str(self.local))

    def event(self, event, **data):
        row = dict(unix=time.time(), event=event, run_id=self.policy['run_id'], **data)
        fd = os.open(self.root / 'retention_journal.jsonl', os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(fd, (json.dumps(row, sort_keys=True) + '\n').encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def bind(self, path):
        path = exact_path(path)
        path.mkdir(parents=True, exist_ok=True)
        marker = path / '.retention_owner.json'
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text()) != self.owner:
                raise ValueError('Retention owner differs')
        else:
            # Root has configs and logs; the local namespace must start empty.
            if path == self.local and any(path.iterdir()):
                raise ValueError('Unowned nonempty local history path')
            atomic_json(marker, self.owner)
            fsync_dir(path.parent)

    @contextmanager
    def locked(self):
        if not self.authorized:
            raise ValueError('Checkpoint retirement is not authorized')
        self.bind(self.root)
        with (self.root / 'retention.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            missing = not self.local.exists()
            self.bind(self.local)
            if self.root.stat().st_dev == self.local.stat().st_dev:
                raise ValueError('NFS/local history must be different filesystems')
            for base in [self.root, self.local]:
                for name in ['retention_trash'] if base == self.root else ['checkpoints', 'certificates', 'staging', 'retention_trash']:
                    p = exact_path(base / name)
                    p.mkdir(exist_ok=True)
            if missing:
                self.event('local_namespace_created_or_lost_history_not_reconstructed')
            yield

    def budgets(self, *, startup=False, incoming_bytes=0):
        p = self.policy
        nfs_used, local_used = size_tree(self.root), size_tree(self.local)
        temp = exact_path(p['temp_root'])
        if temp == self.local or temp.is_relative_to(self.local) or self.local.is_relative_to(temp):
            raise ValueError('Temporary and history namespaces overlap')
        if temp.exists() and temp.stat().st_dev != self.local.stat().st_dev:
            raise ValueError('Temporary allowance is not on the local history device')
        temp_used = size_tree(temp) if temp.exists() else 0
        temp_remaining = max(0, p['temporary_budget_bytes'] - temp_used)
        if temp_used > p['temporary_budget_bytes']:
            raise RuntimeError('Temporary working set exceeds shared local budget')
        if nfs_used > p['nfs_budget_bytes'] or local_used + incoming_bytes > p['local_budget_bytes']:
            raise RuntimeError('Retention budget exceeded; preserve leftovers and stop')
        nfs_remaining = max(0, p['nfs_budget_bytes'] - nfs_used) if startup else 0
        local_remaining = max(0, p['local_budget_bytes'] - local_used) if startup else incoming_bytes
        if shutil.disk_usage(self.root).free < p['reserve_bytes'] + p['other_nfs_future_bytes'] + nfs_remaining:
            raise RuntimeError('NFS future peak/reserve unavailable')
        # / and /tmp are one device; reserve this shared temporary allowance once.
        if shutil.disk_usage(self.local).free < p['reserve_bytes'] + temp_remaining + local_remaining:
            raise RuntimeError('Local temporary/history/reserve unavailable')
        return dict(nfs_used_bytes=nfs_used, local_used_bytes=local_used, temporary_used_bytes=temp_used,
                    nfs_free_bytes=shutil.disk_usage(self.root).free, local_free_bytes=shutil.disk_usage(self.local).free)

    def certificate(self, name):
        if not NAME.fullmatch(name):
            raise ValueError('Invalid checkpoint name')
        path = self.root / 'checkpoint_certificates' / (name + '.json')
        proof = json.loads(path.read_text())
        if (proof['status'] != 'passed_main_checkpoint_storage' or proof['step'] != int(name[5:]) or
                proof['checkpoint'] != self.policy['artifact_training_prefix'] + '/' + name):
            raise ValueError('Certificate bound to another run')
        cp = self.training / 'training_contract.json'
        if proof['training_contract_id'] != json.loads(cp.read_text())['id']:
            raise ValueError('Certificate bound to another training contract')
        records(proof)
        return proof

    def source_rows(self):
        complete, pending = [], []
        for path in sorted(self.training.glob('ckpt_*')):
            if path.is_symlink():
                raise ValueError('Checkpoint symlink rejected')
            if not NAME.fullmatch(path.name) or not (path / '.complete').exists():
                continue
            cert = self.root / 'checkpoint_certificates' / (path.name + '.json')
            if not cert.exists():
                pending.append(path.name)
                continue
            proof = self.certificate(path.name)
            inventory(path, proof)
            complete.append((path.name, proof))
        return complete, pending

    def local_rows(self):
        rows = []
        for path in sorted((self.local / 'checkpoints').iterdir()):
            if not NAME.fullmatch(path.name):
                raise ValueError('Unexpected local history path')
            cert = self.local / 'certificates' / (path.name + '.json')
            if not cert.exists():
                # An interrupted rename before certificate publication is
                # recoverable only while the matching NFS checkpoint survives.
                proof = self.certificate(path.name)
                inventory(path, proof, hashes=True)
                atomic_json(cert, proof)
            proof = json.loads(cert.read_text())
            if proof != self.certificate(path.name):
                raise ValueError('Local history certificate differs from NFS audit receipt')
            inventory(path, proof)
            rows.append((path.name, proof))
        return rows

    def publish_copy(self, name, proof):
        source = self.training / name
        final = self.local / 'checkpoints' / name
        cert = self.local / 'certificates' / (name + '.json')
        if final.exists():
            inventory(final, proof, hashes=True)
            if cert.exists() and json.loads(cert.read_text()) != proof:
                raise ValueError('Published local certificate differs')
            atomic_json(cert, proof)
            return
        stage = self.local / 'staging' / name
        self.make_local_room(max(0, proof['bytes'] - size_tree(stage)))
        stage.mkdir(exist_ok=True)
        self.budgets(incoming_bytes=max(0, proof['bytes'] - size_tree(stage)))
        self.event('copy_intent', checkpoint=name, bytes=proof['bytes'])
        timing=copy_payload(source,stage,proof,copy_workers=self.policy['copy_workers'],hash_workers=self.policy['hash_workers'],
                            fault=self.fault,cancelled=self.cancelled)
        stage.rename(final)  # Both paths are on the local filesystem.
        fsync_dir(final.parent)
        fsync_dir(stage.parent)
        self.fault('published_directory')
        atomic_json(cert, proof)
        self.event('local_published', checkpoint=name, certificate_sha256=sha256_file(cert),timing=timing)

    def retire(self, base, path, proof, tier):
        if self.cancelled():raise InterruptedError('Retention cancelled before retirement')
        # Rename on this SAME device first. A crash can leave a bounded trash
        # transaction, never a partially removed normal checkpoint candidate.
        tx = base / 'retention_trash' / path.name
        if tx.exists():
            raise ValueError('Unfinished retirement must be recovered first')
        tx.mkdir()
        fsync_dir(tx.parent)
        atomic_json(tx / 'intent.json', dict(owner=self.owner, tier=tier, checkpoint=path.name, proof=proof))
        self.event('retire_intent', checkpoint=path.name, tier=tier)
        self.fault('retire_intent')
        path.rename(tx / 'payload')
        fsync_dir(path.parent)
        fsync_dir(tx)
        self.fault('retired_directory')
        self.finish_retirement(base, tx, freshly_verified=(tier == 'nfs'))

    def finish_retirement(self, base, tx, *, freshly_verified=False):
        intent = json.loads((tx / 'intent.json').read_text())
        if (intent['owner'] != self.owner or not NAME.fullmatch(intent['checkpoint']) or
                intent['tier'] != ('nfs' if base == self.root else 'local')):
            raise ValueError('Unbound retirement transaction')
        proof = self.certificate(intent['checkpoint'])
        if proof != intent['proof']:
            raise ValueError('Retirement proof differs')
        payload = tx / 'payload'
        if payload.exists():
            inventory(payload, proof, hashes=not freshly_verified, partial=True, workers=self.policy['hash_workers'])
            if intent['tier'] == 'nfs':
                # Every restart verifies the durable latest10 and the local
                # source copy BEFORE completing a partially deleted old NFS copy.
                rows, _ = self.source_rows()
                if len(rows) < self.policy['nfs_keep'] or proof['step'] >= min(p['step'] for _, p in rows[-self.policy['nfs_keep']:]):
                    raise ValueError('Cannot retire a protected NFS checkpoint')
                inventory(self.local / 'checkpoints' / intent['checkpoint'], proof, hashes=not freshly_verified)
            for p in sorted(payload.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if self.cancelled():raise InterruptedError('Retention cancelled during retirement')
                if p.is_file():
                    p.unlink()
                    self.fault('deleted_file')
                elif p.is_dir():
                    p.rmdir()
            payload.rmdir()
        else:
            source = (self.training if base == self.root else self.local / 'checkpoints') / intent['checkpoint']
            if source.exists():
                # Crash after intent, before rename: leave source and retry.
                inventory(source, proof, hashes=True)
        self.event('retire_completed', checkpoint=intent['checkpoint'], tier=intent['tier'])
        if intent['tier'] == 'local' and not (self.local / 'checkpoints' / intent['checkpoint']).exists():
            cert = self.local / 'certificates' / (intent['checkpoint'] + '.json')
            if cert.exists():
                if json.loads(cert.read_text()) != proof:
                    raise ValueError('Local retired certificate changed')
                cert.unlink()
        (tx / 'intent.json').unlink()
        tx.rmdir()
        fsync_dir(tx.parent)

    def trim_local(self):
        rows, _ = self.source_rows()
        nfs_names = {n for n, _ in rows}
        unique = [(n, p) for n, p in self.local_rows() if n not in nfs_names]
        for name, proof in unique[:-self.policy['local_keep']]:
            self.retire(self.local, self.local / 'checkpoints' / name, proof, 'local')

    def make_local_room(self, incoming_bytes=0):
        """Evict only verified, oldest local-only history; never clean unknown leftovers."""
        if not self.policy.get('adaptive_local_pruning'):
            return
        p = self.policy
        if (size_tree(self.root) > p['nfs_budget_bytes'] or
                shutil.disk_usage(self.root).free < p['reserve_bytes'] + p['other_nfs_future_bytes']):
            raise RuntimeError('NFS budget unavailable; preserve local history')
        def enough():
            temp_used = size_tree(Path(p['temp_root']))
            if temp_used > p['temporary_budget_bytes']:
                raise RuntimeError('Temporary working set exceeds shared local budget')
            return (size_tree(self.local) + incoming_bytes <= p['local_budget_bytes'] and
                    shutil.disk_usage(self.local).free >= p['reserve_bytes'] +
                    max(0, p['temporary_budget_bytes'] - temp_used) + incoming_bytes)
        if enough():
            return
        rows, _ = self.source_rows()
        nfs_names = {name for name, _ in rows}
        for name, proof in self.local_rows():
            if name in nfs_names:
                continue
            self.retire(self.local, self.local / 'checkpoints' / name, proof, 'local')
            if enough():
                return
        raise RuntimeError('Local capacity unavailable after eligible oldest history; preserve NFS and leftovers')

    def maintain(self):
        with self.locked():
            if not self.policy.get('adaptive_local_pruning'):
                self.budgets()
            # A lost local copy cannot justify resuming a half-deleted NFS
            # retirement. Stop safely rather than manufacture a valid history.
            for base in [self.root, self.local]:
                for tx in sorted((base / 'retention_trash').iterdir()):
                    if not NAME.fullmatch(tx.name) or tx.is_symlink():
                        raise ValueError('Unexpected retirement leftovers')
                    self.finish_retirement(base, tx)
            self.make_local_room()
            self.budgets()
            rows, pending = self.source_rows()
            if not pending:
                for name, proof in rows[:-self.policy['nfs_keep']]:
                    self.publish_copy(name, proof)
                    latest, unsealed = self.source_rows()
                    if unsealed:
                        break
                    if name in {n for n, _ in latest[-self.policy['nfs_keep']:]}:
                        raise ValueError('Cannot retire latest NFS checkpoint')
                    self.retire(self.root, self.training / name, proof, 'nfs')
                    self.trim_local()  # Never accumulate several backlog copies.
            rows, pending = self.source_rows()
            local = self.local_rows()
            nfs_names = {n for n, _ in rows}
            # Remove only excess oldest local history; overlap is allowed only
            # while the corresponding NFS retirement has not yet committed.
            self.trim_local()
            local = self.local_rows()
            result = dict(nfs_checkpoints=[n for n, _ in rows], local_checkpoints=[n for n, _ in local],
                          distinct_checkpoints=len({n for n, _ in rows + local}), uncertified_nfs=pending,
                          resume_scope='NFS training directory only; local history is disposable', **self.budgets())
            atomic_json(self.root / 'retention_state.json', result)
            return result


def make_policy(root, scratch, checkpoint_bytes, *, other_nfs_future_bytes=0, run_id):
    root, scratch = exact_path(root), exact_path(scratch)
    from common.data_contracts import artifact_path
    return dict(version=1, run_id=run_id, nfs_run=str(root), nfs_keep=10, local_keep=15,
                artifact_training_prefix=artifact_path(root / 'training'),
                local_root=str(scratch / 'checkpoint_history' / root.name / run_id),
                temp_root=str(scratch / root.name), temporary_budget_bytes=1024 * GIB,
                reserve_bytes=200 * GIB, other_nfs_future_bytes=other_nfs_future_bytes,
                nfs_budget_bytes=math.ceil(11 * checkpoint_bytes * 1.1) + GIB,
                local_budget_bytes=math.ceil(16 * checkpoint_bytes * 1.1) + GIB,
                expected_checkpoint_bytes=checkpoint_bytes, copy_workers=4, hash_workers=8,
                native_pruning_disabled=True, local_loss_tolerated=True, resume_from_local=False)
