"""Redownload one pinned shard, verify SHA256 and every parquet column, preserve old files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source')
    ap.add_argument('filename')
    a = ap.parse_args()
    root = Path(DATA)
    raw = root / 'raw' / a.source
    manifest = json.loads((raw / 'download_manifest.json').read_text())
    if a.filename not in manifest['files']:
        ap.error('File is not part of the pinned manifest')
    repo, revision = manifest['repo'], manifest['revision']
    info = HfApi().get_paths_info(repo, [a.filename], repo_type='dataset', revision=revision)[0]
    expected = info.lfs.sha256
    report_dir = root / 'recovery' / a.source
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / (Path(a.filename).stem + '.verified.json')
    original = raw / a.filename
    if report.exists() and original.exists() and sha256(original) == expected:
        print('ALREADY VERIFIED', a.filename, flush=True)
        return
    print('DOWNLOAD', a.filename, 'expected_bytes', info.size, flush=True)
    candidate = Path(hf_hub_download(repo, a.filename, repo_type='dataset', revision=revision,
                                   local_dir=str(report_dir)))
    actual = sha256(candidate)
    if candidate.stat().st_size != info.size or actual != expected:
        raise RuntimeError('Replacement checksum mismatch; original left unchanged')
    print('SHA256 OK', actual, flush=True)
    rows = 0
    pf = pq.ParquetFile(candidate)
    for batch in pf.iter_batches(batch_size=4096):
        rows += batch.num_rows
    if rows != pf.metadata.num_rows:
        raise RuntimeError('Replacement row count mismatch; original left unchanged')
    print('ALL COLUMNS READ OK rows', rows, flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    quarantine = root / 'quarantine' / a.source / stamp
    quarantine.mkdir(parents=True)
    if original.exists():
        original.rename(quarantine / original.name)
    # A failed cleaning attempt must never be interpreted as completed output.
    clean = root / 'clean' / a.source / Path(a.filename).name
    for suffix in ('.tmp', '.stats.json.tmp'):
        partial = Path(str(clean) + suffix)
        if partial.exists():
            partial.rename(quarantine / partial.name)
    candidate.rename(original)
    result = dict(repo=repo, revision=revision, filename=a.filename, sha256=actual,
                  rows=rows, bytes=info.size, quarantine=str(quarantine.relative_to(root)),
                  verified_at=datetime.now(timezone.utc).isoformat())
    tmp = report.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(report)
    print('REPAIRED', a.filename, 'backup', result['quarantine'], flush=True)


if __name__ == '__main__':
    main()
