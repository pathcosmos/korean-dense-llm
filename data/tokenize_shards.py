"""Stream completed clean shards to uint16 tokens with content-based resume.
Usage: python data/tokenize_shards.py SOURCE [HOLDOUT_DOCS]
       [--sources-manifest PATH | --input-dir PATH] [--tokenizer PATH] [--output-dir PATH]
Holdout uses the final N documents of the last input shard, excluded from training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json, identity, resolve, sha256_file, source_files, artifact_path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


TOKEN_FORMAT = 'uint16-eos-v2'
GROUPED_TOKEN_FORMAT = 'uint16-eos-split-v3'


def uint16_ids(ids):
    values = np.asarray(ids)
    if values.size and (values.dtype.kind not in 'iu' or values.min() < 0 or values.max() > 65535):
        raise ValueError('Token IDs must be integers in 0..65535 before uint16 cast')
    return values.astype(np.uint16)


def file_record(path):
    return dict(path=artifact_path(path), bytes=Path(path).stat().st_size, sha256=sha256_file(path))


def commit_outputs(out, record, publish_done=True):
    """Recover a crash between data rename and .done using a content certificate."""
    for item in record['outputs']:
        final = out / item['name'];tmp = Path(str(final) + '.tmp')
        candidate = final if final.exists() else tmp
        if not candidate.exists() or candidate.stat().st_size != item['bytes'] or sha256_file(candidate) != item['sha256']:
            raise ValueError(f'Incomplete or corrupt token output: {candidate}')
        if candidate == tmp:
            tmp.replace(final)
    if publish_done:
        atomic_json(out / record['marker'], record)


def tokenize_source(source, tokenizer_path, files, output_dir, holdout_docs=0, batch_size=1024, snapshot=None, split_manifest=None):
    from data.split_manifest import SplitSelection
    selection = SplitSelection(split_manifest) if split_manifest else None
    try:
        return _tokenize_source(source, tokenizer_path, files, output_dir, holdout_docs, batch_size, snapshot, selection)
    finally:
        if selection is not None: selection.close()


def _tokenize_source(source, tokenizer_path, files, output_dir, holdout_docs, batch_size, snapshot, selection):
    if type(holdout_docs) is not int or type(batch_size) is not int or holdout_docs < 0 or batch_size < 1:
        raise ValueError('Invalid holdout/batch size')
    files = list(map(Path, files));out = Path(output_dir)
    if not files or len(set(f.stem for f in files)) != len(files):
        raise ValueError('Input shards must be nonempty with unique basenames')
    tok = Tokenizer.from_file(str(tokenizer_path))
    eos = tok.token_to_id('<|eos|>')
    if eos is None:
        raise ValueError('Tokenizer is missing EOS')
    uint16_ids(list(tok.get_vocab().values()))
    uint16_ids([eos])
    rows = [pq.ParquetFile(f).metadata.num_rows for f in files]
    if selection is not None and holdout_docs:
        raise ValueError('Cannot combine group split with last-N holdout')
    if holdout_docs > rows[-1]:
        raise ValueError('Holdout documents exceed final shard size')
    plan = dict(schema_version=2, source=source, format=GROUPED_TOKEN_FORMAT if selection else TOKEN_FORMAT, dtype='uint16',
                tokenizer_sha256=sha256_file(tokenizer_path), eos_id=eos, snapshot_id=snapshot,
                holdout_docs=holdout_docs, inputs=[file_record(f) for f in files])
    if selection is not None:
        selection.verify_source(source, plan['inputs'], snapshot)
        plan['split_contract_id'] = selection.report['contract_id']
    plan['contract_id'] = identity(plan)
    manifest = out / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != plan:
            raise ValueError('Tokenization contract changed: use a new output directory')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Existing output directory has no matching tokenization manifest')
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest, plan)
    totals = dict(tokens=0, holdout_tokens=0, train_docs=0, holdout_docs=0)
    for i, file in enumerate(files):
        name = file.stem + '.bin';dst = out / name
        marker = out / (name + '.done');pending = out / (name + '.pending.json')
        if marker.exists() or pending.exists():
            record = json.loads((marker if marker.exists() else pending).read_text())
            if record['contract_id'] != plan['contract_id']:
                raise ValueError('Shard completion contract mismatch')
            commit_outputs(out, record, publish_done=not marker.exists())
            print('REUSED', name, flush=True)
        else:
            hold = selection.counts(file).get('eval', 0) if selection else (holdout_docs if i == len(files)-1 else 0)
            train_docs = rows[i] - hold
            hold_name = file.stem + '.hold.bin' if selection else 'hold.bin'
            names = [name] + ([hold_name] if hold else [])
            assignments = iter(selection.rows(file)) if selection else None
            seen_splits = [0, 0]
            if any((out / n).exists() for n in names):
                raise ValueError('Uncertified token output exists; refusing overwrite')
            buffers = [open(out / (n + '.tmp'), 'wb') for n in names]
            hashes = [hashlib.sha256() for _ in names];counts = [0 for _ in names]
            offset = 0
            try:
                for batch in pq.ParquetFile(file).iter_batches(batch_size=batch_size, columns=['text']):
                    texts = batch.column(0).to_pylist()
                    if not all(isinstance(t, str) for t in texts):
                        raise ValueError('Input text must be strings')
                    encodings = tok.encode_batch(texts, add_special_tokens=False)
                    ids = [[], []]
                    for j, encoding in enumerate(encodings):
                        if assignments is not None:
                            row, label = next(assignments)
                            if row != offset + j: raise ValueError('Split row coverage mismatch')
                            split = int(label == 'eval')
                        else:
                            split = int(bool(hold) and offset + j >= train_docs)
                        seen_splits[split] += 1
                        ids[split].extend(encoding.ids);ids[split].append(eos)
                    for j in range(len(names)):
                        values = uint16_ids(ids[j]);data = values.tobytes()
                        buffers[j].write(data);hashes[j].update(data);counts[j] += len(values)
                    offset += len(texts)
            finally:
                for buffer in buffers:
                    buffer.close()
            if seen_splits != [train_docs, hold] or (assignments is not None and next(assignments, None) is not None):
                raise ValueError('Split document counts differ')
            if offset != rows[i]:
                raise ValueError('Input document count changed')
            record = dict(contract_id=plan['contract_id'], marker=marker.name, tokens=counts[0],
                          holdout_tokens=counts[1] if hold else 0, train_docs=train_docs, holdout_docs=hold,
                          holdout_file=hold_name if hold else None,
                          outputs=[dict(name=n, bytes=counts[j]*2, sha256=hashes[j].hexdigest()) for j,n in enumerate(names)])
            atomic_json(pending, record)
            commit_outputs(out, record)
            print(f'[{i+1}/{len(files)}]', name, 'tokens', counts[0], 'holdout_tokens', record['holdout_tokens'], flush=True)
        for key in totals:
            totals[key] += record[key]
    atomic_json(out / 'summary.json', dict(contract_id=plan['contract_id'], **totals))
    atomic_json(out / '.done', dict(contract_id=plan['contract_id'], shards=len(files)))
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source');ap.add_argument('holdout_docs', nargs='?', type=int, default=0)
    ap.add_argument('--sources-manifest');ap.add_argument('--input-dir')
    ap.add_argument('--tokenizer', default='tokenizer/tokenizer.json')
    ap.add_argument('--output-dir');ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--split-manifest')
    a = ap.parse_args()
    files, snapshot = source_files(a.source, a.sources_manifest, a.input_dir)
    print(tokenize_source(a.source, resolve(a.tokenizer), files, resolve(a.output_dir or f'data/tokens/{a.source}'),
                          a.holdout_docs, a.batch_size, snapshot, a.split_manifest), flush=True)


if __name__ == '__main__':
    main()
