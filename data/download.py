"""Download pinned parquet shards with per-file progress and bounded retries.
Usage: python data/download.py NAME [MAX_FILES] [--workers 2] [--list-only]
Existing completed files are validated and preserved; partial HTTP transfers resume.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Standard HTTP supports timeouts and resuming. Override explicitly if needed.
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

SOURCES = {
    'fineweb2_ko': ('HuggingFaceFW/fineweb-2', 'data/kor_Hang/train/*'),
    'hplt_ko': ('HPLT/HPLT2.0_cleaned', 'kor_Hang/*'),
    'wiki_ko': ('wikimedia/wikipedia', '20231101.ko/*'),
    'fineweb_edu': ('HuggingFaceFW/fineweb-edu', 'sample/100BT/*'),
    'owm': ('open-web-math/open-web-math', 'data/*'),
    'github_code_pilot': ('codeparrot/github-code', 'data/*'),
    'github_code_v1': ('codeparrot/github-code', 'data/*'),
}


def select_entries(entries, limit=None, spread=False):
    if limit is None:
        if spread:
            raise ValueError('spread requires max_files')
        return entries
    if limit < 1:
        raise ValueError('max_files must be positive')
    if not spread or limit >= len(entries):
        return entries[:limit]
    if limit == 1:
        return entries[:1]
    return [entries[i * (len(entries) - 1) // (limit - 1)] for i in range(limit)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name', choices=SOURCES)
    ap.add_argument('max_files', nargs='?', type=int)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--list-only', action='store_true')
    ap.add_argument('--spread', action='store_true', help='Select evenly spaced shards instead of a prefix')
    ap.add_argument('--revision', help='Pin initial download to this source commit')
    ap.add_argument('--verify-sha256', action='store_true', help='Also hash existing files against Hub LFS SHA256')
    a = ap.parse_args()
    if a.workers < 1 or (a.max_files is not None and a.max_files < 1):
        ap.error('workers and max_files must be positive')
    repo, pattern = SOURCES[a.name]
    out = Path(DATA) / 'raw' / a.name
    mp = out / 'download_manifest.json'
    previous = json.loads(mp.read_text()) if mp.exists() else None
    api = HfApi()
    if previous and a.revision and previous['revision'] != a.revision:
        ap.error('revision conflicts with existing manifest')
    revision = previous['revision'] if previous else (a.revision or api.repo_info(repo, repo_type='dataset', timeout=30).sha)
    entries = sorted((f for f in api.list_repo_tree(repo, path_in_repo=pattern.split('*', 1)[0].rstrip('/'),
                                                            repo_type='dataset', revision=revision, recursive=True)
                      if getattr(f, 'size', None) is not None and
                      fnmatch.fnmatch(f.path, pattern) and f.path.endswith('.parquet')), key=lambda f: f.path)
    entries = select_entries(entries, a.max_files, a.spread)
    if not entries:
        raise RuntimeError(f'No parquet files matched {repo}/{pattern}')
    manifest = dict(repo=repo, revision=revision, files=[f.path for f in entries])
    if previous and previous != manifest:
        raise RuntimeError('Selection changed: preserve existing data and review before continuing')
    print(f'{a.name}: {len(entries)} shards, {sum(f.size for f in entries)/1e9:.2f} GB, revision={revision}', flush=True)
    if a.list_only:
        return
    out.mkdir(parents=True, exist_ok=True)
    if not previous:
        tmp = mp.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(mp)

    def verify_hash(entry, path):
        lfs = getattr(entry, 'lfs', None)
        if lfs is None:
            raise RuntimeError(f'No remote LFS SHA256 for {entry.path}')
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(16 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != lfs.sha256:
            raise RuntimeError(f'SHA256 mismatch; preserve and repair: {entry.path}')

    def fetch(entry):
        dst = out / entry.path
        if dst.exists():
            if dst.stat().st_size != entry.size:
                raise RuntimeError(f'Existing file size mismatch; will not overwrite: {entry.path}')
            if a.verify_sha256:
                verify_hash(entry, dst)
            rows = pq.ParquetFile(dst).metadata.num_rows
            print(f'EXISTS {entry.path} rows={rows}', flush=True)
            return
        for attempt in range(1, 4):
            try:
                print(f'START {entry.path} attempt={attempt}', flush=True)
                downloaded = hf_hub_download(repo, entry.path, repo_type='dataset', revision=revision, local_dir=str(out))
                if Path(downloaded).stat().st_size != entry.size:
                    raise RuntimeError('Downloaded file size mismatch')
                verify_hash(entry, downloaded)
                rows = pq.ParquetFile(downloaded).metadata.num_rows
                print(f'OK {entry.path} bytes={entry.size} rows={rows}', flush=True)
                return
            except Exception as exc:
                # Avoid logging signed URLs or authorization headers from exceptions.
                print(f'RETRY {entry.path} attempt={attempt} error={type(exc).__name__}', flush=True)
                if attempt == 3:
                    raise RuntimeError(f'Download failed: {entry.path} ({type(exc).__name__})') from None
                time.sleep(min(10, 2 ** attempt))

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(fetch, entries))
    print(f'done {a.name}: {len(entries)}/{len(entries)} verified', flush=True)


if __name__ == '__main__':
    main()
