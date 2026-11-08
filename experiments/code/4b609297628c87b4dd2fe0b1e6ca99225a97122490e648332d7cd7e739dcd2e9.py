"""Read-only retention/recovery audit. Never deletes or overwrites corpus files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json, sha256_file
from common.source_inventory import load_audit
from scripts.resource_admission import inspect


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True)
    a = ap.parse_args()
    root = Path(PROJECT_ROOT).resolve()
    out = (root / a.run_dir).resolve()
    if not out.is_relative_to(root / 'runs'):
        raise ValueError('Audit output must be a separate runs directory')
    out.mkdir(parents=True, exist_ok=True)
    if (out / '.done').exists():
        marker = json.loads((out / '.done').read_text())
        if marker['report_sha256'] != sha256_file(out / 'report.json'):
            raise ValueError('Changed audit report')
        print('Completed read-only audit preserved; use a new run for a fresh audit.')
        return
    start = time.monotonic()
    hardware = inspect()
    atomic_json(out / 'hardware.json', hardware)
    if hardware['memory']['cache_aware_admission']['headroom_gib'] < 64:
        raise RuntimeError('Insufficient additional memory after reserve')
    scratch = Path(hardware['storage']['scratch_dir']) / out.name
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < 256 * 2**30:
        raise RuntimeError('Local reserve unavailable')
    os.environ.update(OMP_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', RAYON_NUM_THREADS='1',
                      HF_HUB_DISABLE_XET='1', HF_HUB_ENABLE_HF_TRANSFER='0',
                      HF_HUB_ETAG_TIMEOUT='30', HF_HUB_DOWNLOAD_TIMEOUT='30',
                      TMPDIR=str(scratch))
    cpus = sorted(os.sched_getaffinity(0))[-32:]
    os.sched_setaffinity(0, cpus)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    resources = dict(cpu_affinity=cpus, hash_workers=32, metadata_workers=6,
                     arrow_threads=2, download_workers=2, max_rss_gib=32,
                     local_output_cap_gib=16, max_seconds=1200,
                     scratch=str(scratch), gpu_work=False,
                     tuning_evidence='Reuse same-host 32-worker SHA evidence in HARDWARE_RESOURCES; no full-corpus regeneration.')
    atomic_json(out / 'resources.json', resources)
    (out / 'pid').write_text(str(os.getpid()))

    def state(phase, **kw):
        if time.monotonic() - start > 1200:
            raise TimeoutError('Audit time budget exceeded')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 32 * 1024**2:
            raise MemoryError('Audit RSS budget exceeded')
        atomic_json(out / 'state.json', dict(phase=phase, pid=os.getpid(),
                    utc=datetime.now(timezone.utc).isoformat(), **kw))
        print(phase, json.dumps(kw, ensure_ascii=False), flush=True)

    state('metadata')
    proposal = json.loads((root / 'runs/pretrain_readiness_v1/storage_cost_and_retention_proposal_v1.json').read_text())
    candidates = [dict(x, group=group) for group, desc in proposal['candidates'].items()
                  for x in desc['records']]
    paths = {x['path'] for x in candidates}
    if len(paths) != len(candidates):
        raise ValueError('Duplicate candidate path')
    inventory = json.loads((root / 'data/datasets/final_v1.ready.json').read_text())
    protected = {p for e in inventory['sources'].values() for p in e['files']}
    if paths & protected:
        raise ValueError('Candidate overlaps current source')
    audit = load_audit(inventory['audit_proof']['path'])
    if audit['content_inventory_id'] != inventory['content_inventory_id']:
        raise ValueError('Current ready/audit mismatch')
    manifests = {p.parent.name: json.loads(p.read_text())
                 for p in sorted((root / 'data/raw').glob('*/download_manifest.json'))}
    frozen = out / 'recovery_bundle'
    source_code = ['common/paths.py', 'data/download.py', 'data/clean.py',
                   'data/filters.py', 'data/clean_code.py', 'data/dedup.py', 'requirements.txt']
    preserve = [root / p for p in source_code]
    preserve += sorted((root / 'data/raw').glob('*/download_manifest.json'))
    preserve += [p for p in (root / 'data/clean').rglob('*')
                 if p.is_file() and (p.name.endswith('.json') or p.name.endswith('.done'))]
    preserve += sorted((root / 'data/recovery').rglob('*.json'))
    preserve += [root / 'data/dedup/exact_v1' / n for n in ['manifest.json', 'summary.json', '.done']]
    bundle = []
    for p in preserve:
        rel = p.relative_to(root); target = frozen / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        body = p.read_bytes()
        if target.exists() and target.read_bytes() != body:
            raise ValueError('Changed recovery evidence; choose a new run')
        if not target.exists():
            target.write_bytes(body)
        bundle.append(dict(path=str(rel), bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
    atomic_json(frozen / 'bundle_manifest.json', bundle)
    atomic_json(frozen / 'environment.json', dict(python=sys.version,
                packages={n: importlib.metadata.version(n) for n in
                          ['pyarrow', 'xxhash', 'huggingface_hub', 'requests', 'numpy', 'zstandard']}))

    def remote(source):
        m = manifests[source]
        api = HfApi(token=False)
        entries = api.get_paths_info(m['repo'], m['files'], repo_type='dataset', revision=m['revision'])
        by_path = {f.path: f for f in entries}
        if set(by_path) != set(m['files']):
            raise ValueError('Remote file coverage differs: ' + source)
        records = []
        for name in m['files']:
            f = by_path[name]
            if not f.lfs or not f.lfs.sha256 or f.lfs.size != f.size:
                raise ValueError('Missing remote LFS proof: ' + source)
            records.append(dict(source=source, path='data/raw/' + source + '/' + name,
                                remote_path=name, repo=m['repo'], revision=m['revision'],
                                bytes=f.size, sha256=f.lfs.sha256,
                                url='https://huggingface.co/datasets/' + m['repo'] + '/resolve/' + m['revision'] + '/' + name))
        atomic_json(out / 'remote' / (source + '.json'), records)
        return source, records

    remote_records = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(remote, s) for s in manifests]):
            source, records = future.result()
            remote_records[source] = records
            state('remote_metadata', source=source, files=len(records))
    remote_index = {x['path']: x for v in remote_records.values() for x in v}
    if set(remote_index) != {x['path'] for x in candidates if x['group'] == 'raw_payloads'}:
        raise ValueError('Raw candidate/remote coverage differs')
    atomic_json(out / 'remote_manifest.json', remote_index)

    def hashed(entry):
        p = root / entry['path']; before = p.stat()
        if p.is_symlink() or not p.is_file():
            raise ValueError('Nonregular candidate: ' + entry['path'])
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (entry['bytes'], entry['mtime_ns'], entry['inode']):
            raise ValueError('Candidate changed since prior inventory: ' + entry['path'])
        digest = sha256_file(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Candidate changed while reading')
        if entry['path'] in remote_index:
            remote_proof = remote_index[entry['path']]
            if digest != remote_proof['sha256'] or before.st_size != remote_proof['bytes']:
                raise ValueError('Raw differs from pinned remote: ' + entry['path'])
        return dict(entry, sha256=digest, remote_verified=entry['path'] in remote_index)

    state('hashing', total=len(candidates))
    hash_start = time.monotonic(); hashed_records = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for future in as_completed([pool.submit(hashed, e) for e in candidates]):
            hashed_records.append(future.result())
            if len(hashed_records) % 100 == 0:
                state('hashing', done=len(hashed_records), total=len(candidates))
    hashed_records.sort(key=lambda x: x['path'])
    atomic_json(out / 'candidate_manifest.json', hashed_records)
    state('hashes_verified', files=len(hashed_records), seconds=time.monotonic() - hash_start)

    # Public reads only. Full actual downloads to a fresh local directory, not Hub cache reuse.
    import requests
    download_sources = [s for s in manifests if s != 'github_code_pilot']
    selections = [min(remote_records[s], key=lambda x: x['bytes']) for s in download_sources]
    if sum(x['bytes'] for x in selections) > 8 * 2**30:
        raise ValueError('Download sample budget exceeded')

    def download(record):
        dst = scratch / 'download' / record['source'] / Path(record['remote_path']).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        t = time.monotonic()
        if dst.exists():
            if dst.stat().st_size != record['bytes'] or sha256_file(dst) != record['sha256']:
                raise ValueError('Changed sample file')
            return dict(record, downloaded_path=str(dst), reused=True)
        tmp = Path(str(dst) + '.partial')
        # Retain interrupted partials; a fresh attempt uses a distinct filename.
        if tmp.exists():
            tmp = Path(str(dst) + '.' + str(time.time_ns()) + '.partial')
        h = hashlib.sha256(); n = 0
        with requests.get(record['url'], stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with tmp.open('xb') as stream:
                for block in response.iter_content(4 * 1024**2):
                    n += len(block)
                    if n > record['bytes'] or time.monotonic() - start > 1200:
                        raise RuntimeError('Download budget exceeded')
                    stream.write(block); h.update(block)
                stream.flush(); os.fsync(stream.fileno())
        if n != record['bytes'] or h.hexdigest() != record['sha256']:
            raise ValueError('Downloaded payload verification failed')
        tmp.replace(dst)
        pq.ParquetFile(dst)
        return dict(record, downloaded_path=str(dst), seconds=time.monotonic()-t,
                    complete_download_sha256_verified=True, reused=False)

    downloads = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(download, r) for r in selections]):
            record = future.result(); downloads.append(record)
            atomic_json(out / 'downloads.json', downloads)
            state('sample_downloaded', source=record['source'], bytes=record['bytes'])

    # Exercise the exact existing cleaning functions on the initial 4096 raw rows.
    import importlib.util
    sys.path.insert(0, str(root / 'data'))
    spec = importlib.util.spec_from_file_location('retention_clean', root / 'data/clean.py')
    web = importlib.util.module_from_spec(spec); spec.loader.exec_module(web)
    from data import clean_code
    regen = scratch / 'regeneration'
    web.ROOT = str(regen / 'data'); clean_code.DATA = str(regen / 'data')
    reproduced = []
    for rec in downloads:
        source = rec['source']; original = Path(rec['downloaded_path'])
        pf = pq.ParquetFile(original)
        first = next(pf.iter_batches(batch_size=4096))
        prefix = regen / 'data/raw' / source / original.name
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if not prefix.exists():
            pq.write_table(pa.Table.from_batches([first]), prefix)
        if source == 'github_code_v1':
            clean_code.work((source, str(prefix)))
        else:
            lang = 'math' if source == 'owm' else ('en' if source == 'fineweb_edu' else 'ko')
            web.work((source, lang, str(prefix)))
        result = regen / 'data/clean' / source / original.name
        got = pq.read_table(result)
        if got.num_rows == 0:
            raise ValueError('Empty regeneration proof')
        prior = root / 'data/clean' / source / original.name
        chunks = []; total = 0
        for batch in pq.ParquetFile(prior).iter_batches(batch_size=4096):
            chunks.append(batch); total += batch.num_rows
            if total >= got.num_rows: break
        expected = pa.Table.from_batches(chunks).slice(0, got.num_rows)
        if not got.equals(expected):
            raise ValueError('Regenerated prefix differs: ' + source)
        reproduced.append(dict(source=source, raw_rows=first.num_rows, kept_rows=got.num_rows,
                               comparison='Exact all-column logical prefix equality',
                               scope='First4096 rows of one fully re-downloaded shard; not full-corpus regeneration'))
        atomic_json(out / 'regeneration.json', reproduced)
        state('sample_regenerated', source=source, kept_rows=got.num_rows)

    # Check currently open files and memory mappings; inability to inspect is explicit.
    absolute = {str((root / p).resolve()): p for p in paths}
    opened = []; denied = []; inspected = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd).removesuffix(' (deleted)')
                    if target in absolute: opened.append(dict(pid=int(proc.name), path=absolute[target], via='fd'))
                except FileNotFoundError: pass
            for line in (proc / 'maps').read_text().splitlines():
                part = line.split(None, 5)
                if len(part) == 6 and part[5] in absolute:
                    opened.append(dict(pid=int(proc.name), path=absolute[part[5]], via='mmap'))
            inspected += 1
        except PermissionError: denied.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError): pass
    groups = {}
    for group in proposal['candidates']:
        records = [x for x in hashed_records if x['group'] == group]
        groups[group] = dict(files=len(records), bytes=sum(x['bytes'] for x in records),
                             allocated_bytes=sum(x['allocated_bytes'] for x in records))
    report = dict(status='recovery_review_complete_not_deletion_authorization',
                  utc=datetime.now(timezone.utc).isoformat(), groups=groups,
                  pinned_remote_files=len(remote_index), actual_raw_hash_matches=len(remote_index),
                  sample_downloads=downloads, sample_regeneration=reproduced,
                  candidate_manifest_sha256=sha256_file(out / 'candidate_manifest.json'),
                  remote_manifest_sha256=sha256_file(out / 'remote_manifest.json'),
                  recovery_bundle_manifest_sha256=sha256_file(frozen / 'bundle_manifest.json'),
                  active_source_overlap=sorted(paths & protected),
                  open_candidate_handles=opened, inspected_processes=inspected,
                  process_inspection_denied=denied,
                  elapsed_seconds=time.monotonic()-start,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  deleted_files=0, main_training_started=False,
                  limits=['Current pinned remote availability is not a perpetual hosting guarantee.',
                          'All raw bytes match remote hashes, but only six shards were re-downloaded.',
                          'Cleaning prefix reproduction is not full-corpus or dedup rebuild proof.',
                          'Open-file scan is point-in-time; repeat candidate/consumer checks immediately before approved deletion.'])
    atomic_json(out / 'report.json', report)
    atomic_json(out / '.done', dict(report_sha256=sha256_file(out/'report.json'),
                scope='read_only_recovery_review_not_deletion_authorization'))
    state('complete', groups=groups, deleted_files=0)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Avoid signed URLs or credentials from HTTP exceptions in shared logs.
        print('AUDIT FAILED:', type(error).__name__, flush=True)
        raise SystemExit(1)
