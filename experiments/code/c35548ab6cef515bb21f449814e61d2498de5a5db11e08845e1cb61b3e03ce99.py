"""Build resumably on local scratch and atomically publish a verified split directory.

Process loss resumes SQLite transactions. Scratch/container loss before a complete
durable staging copy rebuilds from immutable sources. No partial-DB backup is implied.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import resolve, artifact_path, identity, sha256_file
from data.split_manifest import build, SplitSelection

GIB = 1024**3
DEPENDENCIES = ['scripts/publish_local_split.py', 'data/split_manifest.py',
                'common/data_contracts.py', 'common/source_inventory.py',
                'scripts/run_local_split.py', 'scripts/hardware_profile.py',
                'scripts/run_with_resources.py']


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    sync_dir(path.parent)


def bind(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Publication contract changed; preserve output and choose a new run')
    else:
        durable_json(path, value)


def verify(directory, contract, cache):
    selector = SplitSelection(directory / 'manifest.json', sqlite_cache_mib=cache)
    try:
        report = selector.report
        c = report['contract']
        if (c['source_snapshot'] != contract['source_snapshot'] or
                c['seed'] != contract['seed'] or c['eval_fraction'] != contract['eval_fraction']):
            raise ValueError('Split input/policy differs from publication contract')
        if resolve(report['database']['path']).resolve() != (directory / 'split.sqlite').resolve():
            raise ValueError('Split DB must be inside the verified directory')
        return report
    finally:
        selector.close()


def publish(source_manifest, output_dir, run_dir, scratch_dir, *, seed=1234,
            eval_fraction=.01, sqlite_cache_mib=512, reserve_gib=200,
            local_budget_gib=300, durable_budget_gib=160, _checkpoint=None):
    if any(not math.isfinite(x) or x < 0 for x in [reserve_gib, local_budget_gib, durable_budget_gib]):
        raise ValueError('Finite nonnegative disk budgets required')
    source = resolve(source_manifest).resolve()
    out, run, scratch = [resolve(p).resolve() for p in [output_dir, run_dir, scratch_dir]]
    if any(a == b or a in b.parents or b in a.parents
           for a, b in [(out, run), (out, scratch), (run, scratch)]):
        raise ValueError('Output, run and scratch must be separate directories')
    run.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    contract = dict(version=1, source_manifest=artifact_path(source), source_sha256=sha256_file(source),
                    source_snapshot=identity(json.loads(source.read_text())), seed=seed,
                    eval_fraction=eval_fraction, sqlite_cache_mib=sqlite_cache_mib,
                    output=artifact_path(out), run=artifact_path(run),
                    code={p: sha256_file(Path(PROJECT_ROOT)/p) for p in DEPENDENCIES},
                    recovery='resume local transaction; rebuild from sources if local lost before durable staging')
    staging = out.with_name(out.name + '.publishing')
    started = time.monotonic()
    last = 0

    def progress(phase):
        nonlocal last
        if time.monotonic() - last > 2 or phase not in ('ingest', 'ingest_pending', 'assign', 'copy_chunk'):
            durable_json(run/'worker_state.json', dict(phase=phase, pid=os.getpid(),
                         elapsed_seconds=time.monotonic()-started, scratch=str(scratch)))
            last = time.monotonic()
        if _checkpoint:
            _checkpoint(phase)

    with (out.parent / (out.name + '.publish.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bind(run/'contract.json', contract)
        if out.exists():
            if not (out/'publication.json').exists() or json.loads((out/'publication.json').read_text()) != contract:
                raise ValueError('Existing output is not owned by this publication')
        else:
            if staging.exists():
                if not (staging/'publication.json').exists() or json.loads((staging/'publication.json').read_text()) != contract:
                    raise ValueError('Unrecognized staging output; preserve it')
            else:
                staging.mkdir()
                durable_json(staging/'publication.json', contract)
            # Budget remaining space on each mount; include simultaneous peaks if shared.
            local_need = local_budget_gib * GIB
            durable_need = durable_budget_gib * GIB
            if scratch.stat().st_dev == staging.stat().st_dev:
                if shutil.disk_usage(scratch).free < local_need + durable_need + reserve_gib*GIB:
                    raise RuntimeError('Insufficient shared-mount space including reserve')
            elif (shutil.disk_usage(scratch).free < local_need + reserve_gib*GIB or
                  shutil.disk_usage(staging).free < durable_need + reserve_gib*GIB):
                raise RuntimeError('Insufficient local or durable space including reserve')
            local = scratch / ('split-' + identity(contract)[:24])
            # Saved source report is immutable and sufficient to recover a verified copy
            # without the original local directory. Never trust a partial copy.
            saved = staging / 'source_report.json'
            db = staging / 'split.sqlite'
            if saved.exists() and db.exists():
                report = json.loads(saved.read_text())
                if sha256_file(db) != report['database']['sha256']:
                    raise ValueError('Durable staged DB changed; preserve it for diagnosis')
            else:
                progress('local_build')
                build(source, local, eval_fraction, seed, sqlite_cache_mib=sqlite_cache_mib,
                      _checkpoint=progress)
                report = verify(local, contract, sqlite_cache_mib)
                progress('local_verified')
                size = (local/'split.sqlite').stat().st_size
                if size > durable_need or size + reserve_gib*GIB > shutil.disk_usage(staging).free:
                    raise RuntimeError('Closed DB exceeds durable copy budget/reserve')
                # A previous interrupted copy may refer to a now-lost scratch path.
                # No durable DB exists here; replace only this run's derived receipt.
                if saved.exists() and json.loads(saved.read_text())['contract_id'] != report['contract_id']:
                    raise ValueError('Rebuilt split contract differs')
                durable_json(saved, report)
                progress('copy')
                partial = staging / 'split.sqlite.copying'
                with (local/'split.sqlite').open('rb') as src, partial.open('wb') as dst:
                    while chunk := src.read(16*1024**2):
                        dst.write(chunk)
                        progress('copy_chunk')
                    dst.flush()
                    os.fsync(dst.fileno())
                if sha256_file(partial) != report['database']['sha256']:
                    raise ValueError('Copied DB hash differs')
                progress('copy_verified')
                partial.replace(db)
                sync_dir(staging)
            # Validate content/coverage at the durable location before exposing any
            # consumer manifest. Reconstructing these tiny files after a crash is safe.
            staged = copy.deepcopy(report)
            staged['database']['path'] = artifact_path(db)
            durable_json(staging/'manifest.json', staged)
            durable_json(staging/'.done', dict(contract_id=staged['contract_id'],
                         manifest_sha256=sha256_file(staging/'manifest.json')))
            progress('verify_staging')
            verify(staging, contract, sqlite_cache_mib)
            progress('staging_verified')
            final = copy.deepcopy(report)
            final['database']['path'] = artifact_path(out/'split.sqlite')
            durable_json(staging/'manifest.json', final)
            durable_json(staging/'.done', dict(contract_id=final['contract_id'],
                         manifest_sha256=sha256_file(staging/'manifest.json')))
            progress('before_rename')
            staging.rename(out)
            sync_dir(out.parent)
            progress('renamed')
        progress('verify_published')
        report = verify(out, contract, sqlite_cache_mib)
        if sha256_file(source) != contract['source_sha256']:
            raise ValueError('Source manifest changed during publication')
        if any(sha256_file(Path(PROJECT_ROOT)/p) != h for p, h in contract['code'].items()):
            raise ValueError('Producer code changed during publication')
        result = dict(status='passed', manifest=artifact_path(out/'manifest.json'),
                      manifest_sha256=sha256_file(out/'manifest.json'), counts=report['counts'],
                      split_contract_id=report['contract_id'], database_sha256=report['database']['sha256'],
                      elapsed_seconds_current_attempt=time.monotonic()-started,
                      recovery_policy=contract['recovery'])
        progress('before_completion')
        durable_json(run/'report.json', result)
        durable_json(run/'.done', dict(report_sha256=sha256_file(run/'report.json'),
                     contract_sha256=sha256_file(run/'contract.json')))
        progress('complete')
        return result


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
    args = vars(ap.parse_args())
    args['source_manifest'] = args.pop('sources_manifest')
    print(json.dumps(publish(**args)))
