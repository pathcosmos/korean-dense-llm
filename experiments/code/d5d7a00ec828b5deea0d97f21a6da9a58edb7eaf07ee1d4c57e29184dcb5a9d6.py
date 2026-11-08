"""Read final source footers and code provenance; never modify training inputs.
Run from project root: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python docs/data_provenance_v1/collect_local.py
This checks metadata/provenance columns. Existing content hashes are recorded, not re-certified.
"""
import collections
import hashlib
import gzip
import json
import os
from pathlib import Path
import resource
import sys
import time
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import PROJECT_ROOT
import pyarrow as pa
import pyarrow.parquet as pq

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
root = Path(PROJECT_ROOT)
out = root / 'docs/data_provenance_v1'
started = time.monotonic()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def check_budget():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 2**30:
        raise RuntimeError('8 GiB RSS budget exceeded')
    if time.monotonic() - started > 300:
        raise RuntimeError('300 second audit budget exceeded')

mp = root / 'data/datasets/final_v1.ready.json'
remote_path = root / 'docs/recovery/data_retention_v1/raw_remote_manifest.json'
m = json.loads(mp.read_text())
remote = json.loads(remote_path.read_text())
code_policy = json.loads((root / 'data/clean/github_code_v1/policy.json').read_text())['policy']
allowed = set(code_policy['allowed_license_metadata'])
inventory = []
source_summary = {}
repo_counts = collections.defaultdict(collections.Counter)
licenses = collections.Counter()
missing = collections.Counter()
metadata_columns = ['repo_name', 'path', 'license', 'source_row', 'content_sha256']
for key, v in m['content_inventory']['sources'].items():
    origin = v['origin']
    raw = json.loads((root / origin['raw_manifest']).read_text())
    assert raw['repo'] == origin['repo'] and raw['revision'] == origin['revision']
    by_name = {Path(f).name: f for f in raw['files']}
    assert len(by_name) == len(raw['files']) == len(v['files'])
    schemas = collections.Counter()
    rows = byte_count = 0
    for f in v['files']:
        check_budget()
        p = root / f['path']
        before = p.stat()
        pf = pq.ParquetFile(p)
        assert before.st_size == f['bytes'] and pf.metadata.num_rows == f['rows']
        cols = pf.schema_arrow.names
        schemas[tuple(cols)] += 1
        rf = by_name[p.name]
        raw_local = f'data/raw/{key}/{rf}'
        r = remote[raw_local]
        assert r['source'] == key and r['repo'] == origin['repo'] and r['revision'] == origin['revision'] and r['remote_path'] == rf
        inventory.append({'source': key, 'final': f, 'upstream_shard': r,
                          'mapping_scope': 'same-source same-basename shard lineage; web original row and URL absent',
                          'final_columns': cols})
        rows += f['rows']; byte_count += f['bytes']
        if key == 'github_code_v1':
            decoded = 0
            for batch in pf.iter_batches(batch_size=16384, columns=metadata_columns, use_threads=False):
                check_budget()
                for row in batch.to_pylist():
                    decoded += 1
                    for c in metadata_columns:
                        if row[c] is None or row[c] == '': missing[c] += 1
                    lic = row['license']
                    assert lic.lower() in allowed
                    assert isinstance(row['source_row'], int) and row['source_row'] >= 0
                    assert len(row['content_sha256']) == 64 and all(c in '0123456789abcdef' for c in row['content_sha256'])
                    licenses[lic] += 1
                    repo_counts[row['repo_name']][lic] += 1
            assert decoded == f['rows']
        after = p.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    source_summary[key] = {'origin': origin, 'files': len(v['files']), 'rows': rows, 'bytes': byte_count,
                           'schemas': [{'columns': list(s), 'files': n} for s, n in schemas.items()]}
repo_file = out / 'code_repository_labels.jsonl.gz'
with gzip.open(repo_file, 'wt', encoding='utf-8', compresslevel=6) as handle:
    for repo, counts in sorted(repo_counts.items()):
        handle.write(json.dumps({'repo_name': repo, 'repository_url': 'https://github.com/' + repo,
                                  'dataset_license_label_counts': dict(sorted(counts.items())),
                                  'original_git_commit_verified': False, 'license_notice_bundle_verified': False}, ensure_ascii=False) + '\n')
assert repo_file.stat().st_size < 128 * 2**20
result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'all 726 final file footers plus all code provenance columns; no text scan or new content hash scan',
          'input_manifest_sha256': sha(mp), 'content_inventory_id': m['content_inventory_id'],
          'raw_remote_manifest_sha256': sha(remote_path), 'collect_script_sha256': sha(Path(__file__)),
          'sources': source_summary, 'total_files': len(inventory), 'total_rows': sum(s['rows'] for s in source_summary.values()),
          'total_bytes': sum(s['bytes'] for s in source_summary.values()),
          'code': {'rows_scanned': sum(licenses.values()), 'license_labels': dict(sorted(licenses.items())),
                   'missing_fields': dict(missing), 'unique_repositories': len(repo_counts),
                   'repositories_with_multiple_labels': sum(len(v) > 1 for v in repo_counts.values()),
                   'repository_index': str(repo_file.relative_to(root)), 'repository_index_sha256': sha(repo_file),
                   'rights_verified': False, 'headers_and_license_texts_scanned': False},
          'resources': {'cpu_affinity': sorted(os.sched_getaffinity(0)), 'arrow_cpu_threads': 1, 'arrow_io_threads': 1,
                        'workers': 1, 'batch_rows': 16384, 'rss_budget_gib': 8, 'time_budget_seconds': 300,
                        'repository_index_cap_mib': 128, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        'elapsed_seconds': time.monotonic() - started}}
dump('file_inventory.json', {'scope': 'shard provenance, existing input content hashes, fresh footer validation', 'files': inventory})
dump('local_audit.json', result)
print(json.dumps({k: result[k] for k in ['total_files', 'total_rows', 'total_bytes', 'code', 'resources']}, ensure_ascii=False, indent=2))
