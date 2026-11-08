"""Cross-shard/source exact dedup, case/whitespace preserving NFC SHA256.
Writes a NEW output tree; original clean files are never changed.
Usage: python data/dedup.py --sources wiki_ko fineweb2_ko hplt_ko --output dedup/ko_v1
Source order determines which duplicate survives. Inputs must be fully cleaned.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import unicodedata
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA
import pyarrow as pa
import pyarrow.parquet as pq


def fingerprint(text):
    # Preserve case, indentation, and whitespace: code/text meaning can depend on them.
    return hashlib.sha256(unicodedata.normalize('NFC', text).encode('utf-8')).digest()


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def deduplicate(inputs, out, policy):
    """inputs: ordered (source, parquet path) list; resumable per-shard transactions."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(policy=policy, inputs=[dict(source=s, path=str(p.resolve()),
                    size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for s, p in inputs])
    manifest = out / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise RuntimeError('Inputs/order/policy changed: choose a new output directory')
    else:
        atomic_json(manifest, signature)
    db = sqlite3.connect(out / 'index.sqlite', timeout=60)
    # WAL shared-memory coordination is unsuitable for network filesystems.
    db.execute('PRAGMA journal_mode=DELETE')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-1048576')  # up to 1 GiB, allocated on demand
    db.execute('CREATE TABLE IF NOT EXISTS seen (digest BLOB PRIMARY KEY) WITHOUT ROWID')
    db.execute('CREATE TABLE IF NOT EXISTS processed (name TEXT PRIMARY KEY, stats TEXT)')
    db.commit()
    totals = dict(input_docs=0, kept=0, exact_duplicates=0)
    schema = pa.schema([('text', pa.string()), ('h', pa.int64())])
    try:
        for i, (source, file) in enumerate(inputs, 1):
            key = source + '/' + file.name
            dst = out / 'clean' / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            record = db.execute('SELECT stats FROM processed WHERE name=?', (key,)).fetchone()
            if record:
                stats = json.loads(record[0])
                if not dst.exists() or pq.ParquetFile(dst).metadata.num_rows != stats['kept']:
                    raise RuntimeError('Committed output missing/invalid: ' + key)
                atomic_json(Path(str(dst) + '.done'), stats)
                print('SKIP', key, flush=True)
            else:
                # Crash between output rename and DB commit: preserve orphan and replay.
                if dst.exists():
                    orphan = out / 'orphans' / str(time.time_ns()) / key
                    orphan.parent.mkdir(parents=True, exist_ok=True)
                    dst.rename(orphan)
                stats = dict(input_docs=0, kept=0, exact_duplicates=0)
                db.execute('BEGIN IMMEDIATE')
                tmp = Path(str(dst) + '.tmp')
                try:
                    with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
                        for batch in pq.ParquetFile(file).iter_batches(batch_size=4096, columns=['text', 'h']):
                            texts, hashes = [], []
                            for text, old_h in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                                stats['input_docs'] += 1
                                if not isinstance(text, str):
                                    raise ValueError('Clean input text is not a string')
                                added = db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (fingerprint(text),)).rowcount
                                if added:
                                    texts.append(text); hashes.append(old_h); stats['kept'] += 1
                                else:
                                    stats['exact_duplicates'] += 1
                            if texts:
                                writer.write_table(pa.table({'text': texts, 'h': hashes}, schema=schema))
                    tmp.replace(dst)
                    db.execute('INSERT INTO processed VALUES (?,?)', (key, json.dumps(stats)))
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
                atomic_json(Path(str(dst) + '.done'), stats)
                print(f'[{i}/{len(inputs)}]', key, json.dumps(stats), flush=True)
            for name in totals:
                totals[name] += stats[name]
        atomic_json(out / 'summary.json', totals)
        atomic_json(out / '.done', totals)
        return totals
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--output', required=True, help='New output path, relative to DATA')
    a = ap.parse_args()
    inputs = []
    for source in a.sources:
        root = Path(DATA) / 'clean' / source
        manifest = Path(DATA) / 'raw' / source / 'download_manifest.json'
        expected = json.loads(manifest.read_text())['files']
        names = [Path(f).name for f in expected]
        if len(set(names)) != len(names):
            raise RuntimeError('Duplicate basenames in source')
        for name in names:
            file = root / name
            if not file.exists() or not Path(str(file) + '.done').exists():
                raise RuntimeError('Source not fully cleaned: ' + str(file))
            inputs.append((source, file))
    output = (Path(DATA) / a.output).resolve()
    clean_root = (Path(DATA) / 'clean').resolve()
    if output == clean_root or clean_root in output.parents or output in clean_root.parents:
        raise RuntimeError('Dedup output must be separate from original clean tree')
    print(deduplicate(inputs, output, 'nfc-sha256-case-and-whitespace-preserving-v1'), flush=True)


if __name__ == '__main__':
    main()
