"""Read-only cross-overlap: whole-document UTF8 equality and fenced-code matches.
Never edits clean data or the active exact_v1 SQLite index. Block matches preserve
case/indentation/Unicode, ignoring only outer CR/LF. They are diagnostic candidates.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA
from data.dedup import atomic_json
import pyarrow as pa
import pyarrow.parquet as pq

OPEN = re.compile(r'^[ ]{0,3}(`{3,}|~{3,})([^\r\n]*)\r?\n?$')


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


def block_key(text):
    return digest(text.strip('\r\n'))


def fenced_blocks(text):
    fence = None
    body = []
    for line in text.splitlines(keepends=True):
        if fence is None:
            match = OPEN.match(line)
            if match and not (match[1][0] == '`' and '`' in match[2]):
                fence = match[1]
                body = []
        elif re.fullmatch(r' {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}[ \t]*(?:\r?\n)?', line):
            yield ''.join(body)
            fence = None
            body = []
        else:
            body.append(line)
    # Unclosed fences are ignored, not guessed.


def run_probe(code_files, math_files, out, min_chars=80):
    out = Path(out);out.mkdir(parents=True, exist_ok=True)
    policy = dict(version='code-math-exact-v1', min_block_chars=min_chars,
                  whole_document='raw UTF8 SHA256', block='raw UTF8 SHA256, outer CR/LF only')
    inputs = dict(policy=policy, code=[(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in code_files],
                  math=[(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns) for f in math_files])
    # JSON converts tuples to lists, compare the same serialized representation.
    inputs = json.loads(json.dumps(inputs))
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != inputs:
        raise RuntimeError('Input snapshot or policy changed; use a new output directory')
    if not manifest.exists():
        atomic_json(manifest, inputs)
    index = out / 'code_index.parquet'
    index_done = out / 'code_index.done'
    raw_counts, block_counts = collections.Counter(), collections.Counter()
    if index_done.exists():
        print('RESUME CODE INDEX', flush=True)
        for b in pq.ParquetFile(index).iter_batches(batch_size=16384):
            raw_counts.update(b.column(0).to_pylist())
            block_counts.update(x for x in b.column(1).to_pylist() if x is not None)
    else:
        schema = pa.schema([('exact', pa.binary(32)), ('block', pa.binary(32))])
        tmp = Path(str(index) + '.tmp')
        with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
            for i, f in enumerate(code_files, 1):
                for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=['text', 'content_sha256']):
                    raws, blocks = [], []
                    for text, saved in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                        h = digest(text)
                        if h.hex() != saved:
                            raise RuntimeError('Stored code content hash mismatch: ' + str(f))
                        stripped = text.strip('\r\n')
                        key = block_key(text) if len(stripped) >= min_chars else None
                        raws.append(h);blocks.append(key)
                    raw_counts.update(raws)
                    block_counts.update(x for x in blocks if x is not None)
                    writer.write_table(pa.table({'exact': raws, 'block': blocks}, schema=schema))
                print('CODE', f'{i}/{len(code_files)}', 'docs', sum(raw_counts.values()), flush=True)
        tmp.replace(index)
        atomic_json(index_done, dict(rows=sum(raw_counts.values())))
    totals = collections.Counter();whole_matches = set();block_matches = set()
    for i, f in enumerate(math_files, 1):
        report_path = out / (f.name + '.json')
        if report_path.exists():
            result = json.loads(report_path.read_text())
        else:
            counts = collections.Counter({k: 0 for k in ['math_documents', 'whole_document_exact_matches',
                'whole_document_matching_pairs', 'fenced_blocks', 'eligible_fenced_blocks',
                'matching_fenced_blocks', 'block_matching_pairs', 'math_documents_with_code_block_match']})
            whole = set();blocks_matched = set();examples = []
            row = 0
            for batch in pq.ParquetFile(f).iter_batches(batch_size=1024, columns=['text']):
                for text in batch.column(0).to_pylist():
                    counts['math_documents'] += 1
                    h = digest(text)
                    if h in raw_counts:
                        counts['whole_document_exact_matches'] += 1
                        counts['whole_document_matching_pairs'] += raw_counts[h]
                        whole.add(h.hex())
                    doc_match = False
                    for code in fenced_blocks(text):
                        counts['fenced_blocks'] += 1
                        if len(code.strip('\r\n')) < min_chars:
                            continue
                        counts['eligible_fenced_blocks'] += 1
                        key = block_key(code)
                        if key in block_counts:
                            counts['matching_fenced_blocks'] += 1
                            counts['block_matching_pairs'] += block_counts[key]
                            blocks_matched.add(key.hex());doc_match = True
                            if len(examples) < 5:
                                examples.append(dict(math_file=f.name, math_row=row, block_sha256=key.hex()))
                    counts['math_documents_with_code_block_match'] += int(doc_match)
                    row += 1
            result = dict(counts=dict(counts), whole_hashes=sorted(whole), block_hashes=sorted(blocks_matched), examples=examples)
            atomic_json(report_path, result)
        totals.update(result['counts']);whole_matches.update(result['whole_hashes']);block_matches.update(result['block_hashes'])
        print('MATH', f'{i}/{len(math_files)}', json.dumps(dict(totals)), flush=True)
    summary = dict(policy=policy, code_documents=sum(raw_counts.values()), code_unique_exact_texts=len(raw_counts),
                   code_internal_duplicate_rows=sum(raw_counts.values())-len(raw_counts),
                   math_shards=len(math_files), counts=dict(totals),
                   math_whole_exact_fraction=totals['whole_document_exact_matches']/totals['math_documents'] if totals['math_documents'] else None,
                   math_code_block_match_fraction=totals['math_documents_with_code_block_match']/totals['math_documents'] if totals['math_documents'] else None,
                   unique_code_texts_matched_exactly=len(whole_matches),
                   unique_code_variants_matched_in_blocks=len(block_matches),
                   limitations='SHA256 equality; no approximate similarity or arbitrary inline snippets; block boundary-newline matches do not justify deleting whole math documents; math counts include source duplicates')
    atomic_json(out / 'summary.json', summary)
    atomic_json(out / '.done', dict(code_shards=len(code_files), math_shards=len(math_files)))
    print('DONE CROSS CHECK', json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='samples/code_math_overlap_v1')
    ap.add_argument('--min-block-chars', type=int, default=80)
    a = ap.parse_args()
    if a.min_block_chars < 1:
        ap.error('min-block-chars must be positive')
    files = []
    for source in ['github_code_v1', 'owm']:
        raw_manifest = json.loads((Path(DATA) / 'raw' / source / 'download_manifest.json').read_text())
        fs = [Path(DATA) / 'clean' / source / Path(x).name for x in raw_manifest['files']]
        if not all(f.exists() and Path(str(f)+'.done').exists() for f in fs):
            raise RuntimeError(source + ' clean snapshot is incomplete')
        files.append(fs)
    run_probe(files[0], files[1], Path(DATA) / a.output, a.min_block_chars)


if __name__ == '__main__':
    main()
