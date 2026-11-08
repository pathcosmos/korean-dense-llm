"""Measure FW2/HPLT URL overlap using the SAME deterministic 1/N URL sample.
All shards are scanned, URL columns only. This measures URL overlap, not text dedup.
Usage: python data/probe_url_overlap.py [--one-in 100]
"""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit
import xxhash
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA


def canonical_url(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        p = urlsplit(value.strip())
        if p.scheme.lower() not in ('http', 'https') or not p.hostname:
            return None
        host = p.hostname.lower()
        if ':' in host:  # IPv6 literal
            host = '[' + host + ']'
        port = p.port
        scheme = p.scheme.lower()
        if port and (scheme, port) not in (('http', 80), ('https', 443)):
            host += ':' + str(port)
        # Keep scheme, path and query distinctions; remove only fragment/default port.
        return urlunsplit((scheme, host, p.path or '/', p.query, ''))
    except (ValueError, UnicodeError):
        return None


def scan(source, column, one_in, destination):
    files = sorted((Path(DATA) / 'raw' / source).rglob('*.parquet'))
    signature = [[str(f.relative_to(Path(DATA))), f.stat().st_size, f.stat().st_mtime_ns] for f in files]
    if not files:
        raise RuntimeError(f'No files for {source}')
    cache = destination / f'{source}.json'
    if cache.exists():
        data = json.loads(cache.read_text())
        if data['input_signature'] == signature and data['one_in'] == one_in:
            print('CACHED', source, flush=True)
            return set(data['hashes']), data['stats']
        raise RuntimeError('Cached inputs changed; use a separate --output directory')
    hashes = set()
    rows = invalid = selected_rows = 0
    for index, f in enumerate(files, 1):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16384, columns=[column]):
            for url in batch.column(0).to_pylist():
                rows += 1
                key = canonical_url(url)
                if key is None:
                    invalid += 1
                    continue
                h = xxhash.xxh64_intdigest(key.encode())
                if h % one_in == 0:
                    hashes.add(h)
                    selected_rows += 1
        if index == 1 or index % 10 == 0 or index == len(files):
            print(source, f'{index}/{len(files)}', 'rows', rows, 'sample_unique_urls', len(hashes), flush=True)
    stats = dict(files=len(files), rows=rows, invalid_urls=invalid, selected_rows=selected_rows, sample_unique_urls=len(hashes))
    tmp = cache.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(one_in=one_in, input_signature=signature, stats=stats, hashes=sorted(hashes))))
    tmp.replace(cache)
    return hashes, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--one-in', type=int, default=100)
    ap.add_argument('--output', default='samples/url_overlap_v1')
    a = ap.parse_args()
    if a.one_in < 1:
        ap.error('one-in must be positive')
    out = Path(DATA) / a.output
    out.mkdir(parents=True, exist_ok=True)
    left, ls = scan('fineweb2_ko', 'url', a.one_in, out)
    right, rs = scan('hplt_ko', 'u', a.one_in, out)
    common = len(left & right)
    result = dict(method='same xxh64(normalized_url) modulo sample across ALL raw shards',
                  one_in=a.one_in, fineweb2_ko=ls, hplt_ko=rs, intersection_sample_urls=common,
                  overlap_fraction_fw2=common / len(left) if left else None,
                  overlap_fraction_hplt=common / len(right) if right else None,
                  jaccard=common / len(left | right) if left | right else None,
                  limitations='URL sample estimates only; 64-bit hash collisions possible; not equivalent to text overlap; scheme/query preserved')
    tmp = out / 'summary.json.tmp'
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out / 'summary.json')
    (out / '.done').touch()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
