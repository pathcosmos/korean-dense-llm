"""Continue authorized data preparation once prerequisites finish. Never launches training.
Usage: python scripts/preprocess_followup.py english | dedup | code
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT, DATA

ROOT = Path(PROJECT_ROOT)


def cleaned(source):
    manifest = Path(DATA) / 'raw' / source / 'download_manifest.json'
    if not manifest.exists():
        return False
    names = json.loads(manifest.read_text())['files']
    return bool(names) and all((Path(DATA) / 'clean' / source / (Path(n).name + '.done')).exists() for n in names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['english', 'dedup', 'code'])
    ap.add_argument('--timeout-hours', type=float, default=12)
    a = ap.parse_args()
    deadline = time.monotonic() + a.timeout_hours * 3600
    sources = ['wiki_ko', 'fineweb2_ko', 'hplt_ko', 'fineweb_edu', 'owm']
    while True:
        if a.stage in ('english', 'code'):
            source = 'fineweb_edu' if a.stage == 'english' else 'github_code_v1'
            log = ROOT / f'logs/download_{source}.log'
            ready = log.exists() and any(x.startswith(f'done {source}:') for x in log.read_text().splitlines())
        else:
            ready = all(cleaned(s) for s in sources)
        if ready:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Prerequisites did not finish; no downstream operation was started')
        print('WAIT', a.stage, 'prerequisites', flush=True)
        time.sleep(30)
    if a.stage == 'english':
        cmd = [sys.executable, '-u', 'data/clean.py', 'fineweb_edu', 'en', '4']
    elif a.stage == 'code':
        cmd = [sys.executable, '-u', 'data/clean_code.py', 'github_code_v1', '2']
    else:
        cmd = [sys.executable, '-u', 'data/dedup.py', '--sources', *sources, '--output', 'dedup/exact_v1']
    print('START', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print('DONE', a.stage, flush=True)


if __name__ == '__main__':
    main()
