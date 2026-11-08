"""Restore pinned web-shard provenance into a new, text-free release sidecar.

Full raw-shard scanning detects ambiguous candidates. This establishes lineage,
not redistribution permission. Canonical split DB is opened read-only/immutable.
"""
import argparse
import hashlib
import json
import os
import resource
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from common.paths import PROJECT_ROOT
from data.filters import exact_hash, mask_pii, norm_ws


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def key(text, old_hash):
    return hashlib.sha256(text.encode('utf8')).hexdigest(), old_hash


def doc_id(inventory_id, source, final_shard, final_row):
    # JSON array encoding makes component boundaries unambiguous.
    data = json.dumps(['release-doc-id-v1', inventory_id, source, final_shard, final_row],
                      ensure_ascii=False, separators=(',', ':')).encode('utf8')
    return hashlib.sha256(data).hexdigest()


def match_candidates(final_rows, raw_rows):
    """Small reusable matching contract; retain every exact candidate."""
    wanted = {}
    for i, row in enumerate(final_rows):
        wanted.setdefault(key(row['text'], row['h']), []).append(i)
    candidates = [[] for _ in final_rows]
    for raw_row, row in enumerate(raw_rows):
        normalized = norm_ws(row['text'])
        h = exact_hash(normalized) - 2**63
        text = mask_pii(normalized)
        for i in wanted.get(key(text, h), []):
            if text == final_rows[i]['text']:
                candidates[i].append((raw_row, {k: v for k, v in row.items() if k != 'text'}))
    return candidates


def restore(record, raw_path, split_manifest, inventory_id, out, *, max_rss_gib=4):
    started = time.monotonic()
    root = Path(PROJECT_ROOT)
    final = root / record['final']['path']
    raw_path = Path(raw_path)
    raw_stat, final_stat = raw_path.stat(), final.stat()
    if out.exists():
        raise FileExistsError('Preserve existing run; select a new output directory')
    assert raw_stat.st_size == record['upstream_shard']['bytes']
    assert final_stat.st_size == record['final']['bytes']
    assert sha(raw_path) == record['upstream_shard']['sha256'], 'Raw SHA mismatch'
    assert sha(final) == record['final']['sha256'], 'Final SHA mismatch'
    source = record['source']
    if source == 'github_code_v1':
        raise ValueError('Code needs its separate raw UTF8 provenance policy')
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
    out.mkdir(parents=True)
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > max_rss_gib * 2**30:
            raise MemoryError('Provenance RSS budget exceeded')
    rows = pq.read_table(final, columns=['text', 'h'], use_threads=False).to_pylist()
    guard()
    assert len(rows) == record['final']['rows']
    raw = pq.ParquetFile(raw_path)
    def raw_rows():
        for batch in raw.iter_batches(batch_size=2048, use_threads=False):
            guard()
            yield from batch.to_pylist()
    matches = match_candidates(rows, raw_rows())
    guard()
    manifest = json.loads(Path(split_manifest).read_text())
    database = root / manifest['database']['path']
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        assignments = db.execute('SELECT row,content_sha256,repo,group_id,split FROM docs WHERE file=? ORDER BY row',
                                 (record['final']['path'],)).fetchall()
    finally:
        db.close()
    assert len(assignments) == len(rows), 'Canonical split coverage mismatch'
    fields = [pa.field(n, t) for n, t in [
        ('doc_id', pa.string()), ('source', pa.string()), ('original_final_file', pa.string()),
        ('final_row', pa.int64()), ('content_sha256', pa.string()), ('group_id', pa.string()),
        ('split', pa.string()), ('upstream_repo', pa.string()), ('upstream_revision', pa.string()),
        ('upstream_file', pa.string()), ('source_row', pa.int64()), ('source_url', pa.string()),
        ('source_title', pa.string()), ('candidate_count', pa.int32()), ('metadata_json', pa.string())]]
    schema = pa.schema(fields)
    tmp = out / 'provenance.parquet.building'
    logical = hashlib.sha256()
    missing = ambiguous = missing_url = 0
    counts = {}
    with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
        buffer = []
        for i, assignment in enumerate(assignments):
            row_num, body_sha, repo, group, split = assignment
            assert row_num == i and split in ('train', 'eval')
            assert body_sha == hashlib.sha256(rows[i]['text'].encode('utf8')).hexdigest(), 'Split text SHA mismatch'
            counts[split] = counts.get(split, 0) + 1
            did = doc_id(inventory_id, source, record['final']['path'], i)
            logical.update((json.dumps([did, i, body_sha, repo, group, split], separators=(',', ':'))+'\n').encode())
            candidates = matches[i]
            missing += not candidates
            ambiguous += len(candidates) > 1
            for raw_row, metadata in candidates:
                url = metadata.get('url', metadata.get('u'))
                missing_url += not url
                buffer.append(dict(doc_id=did, source=source, original_final_file=record['final']['path'],
                    final_row=i, content_sha256=body_sha, group_id=group, split=split,
                    upstream_repo=record['upstream_shard']['repo'], upstream_revision=record['upstream_shard']['revision'],
                    upstream_file=record['upstream_shard']['remote_path'], source_row=raw_row, source_url=url,
                    source_title=metadata.get('title'), candidate_count=len(candidates),
                    metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str)))
            if len(buffer) >= 8192:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
                guard()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    tmp.rename(out / 'provenance.parquet')
    for path, old in [(raw_path, raw_stat), (final, final_stat)]:
        now = path.stat()
        assert (old.st_size, old.st_mtime_ns, old.st_ino) == (now.st_size, now.st_mtime_ns, now.st_ino), 'Input changed'
    report = dict(status='lineage_verified' if not (missing or ambiguous or missing_url) else 'needs_lineage_review',
        source=source, final_file=record['final']['path'], final_rows=len(rows), raw_rows=raw.metadata.num_rows,
        matched_rows=len(rows)-missing, missing_rows=missing, ambiguous_rows=ambiguous,
        missing_candidate_urls=missing_url, counts=counts, logical_assignment_sha256=logical.hexdigest(),
        split_contract_id=manifest['contract_id'], raw_sha256_verified=record['upstream_shard']['sha256'],
        final_sha256_verified=record['final']['sha256'],
        sidecar_bytes=(out/'provenance.parquet').stat().st_size, sidecar_sha256=sha(out/'provenance.parquet'),
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu_affinity=sorted(os.sched_getaffinity(0)), arrow_threads=1, network_bytes=0,
        redistribution_rights_cleared=False, global_group_leakage_rechecked=False,
        matching_policy='normalized_then_masked_exact_bytes_plus_pre_mask_h_all_raw_candidates_v1')
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if report['status'] == 'lineage_verified':
        (out/'.done').write_text(json.dumps({'sidecar_sha256':report['sidecar_sha256']})+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--final-basename', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(PROJECT_ROOT)
    records = json.loads((root/'docs/data_provenance_v1/file_inventory.json').read_text())['files']
    found = [r for r in records if r['source']==args.source and Path(r['final']['path']).name==args.final_basename]
    assert len(found)==1
    registry = json.loads((root/'docs/data_provenance_v1/source_registry.json').read_text())
    print(json.dumps(restore(found[0], args.raw, root/'data/datasets/final_v2.split/manifest.json',
          registry['content_inventory_id'], root/args.output), ensure_ascii=False))
