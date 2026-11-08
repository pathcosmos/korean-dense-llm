"""Recover v3 document offsets from EOS with certified document-count guards.

The v3 producer appends one EOS per document. When EOS count equals the
certified document count, no additional literal EOS boundary is present.
Ambiguous files are rejected and require text/tokenizer replay instead.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def scan_boundaries(path,eos,documents,expected_sha,*,chunk_bytes=8*1024**2):
    path=Path(path)
    if path.stat().st_size%2:raise ValueError('Odd uint16 payload size')
    if chunk_bytes<2 or chunk_bytes%2:raise ValueError('Chunk must contain whole uint16 values')
    before=path.stat();h=hashlib.sha256();ends=[];tokens=0
    with path.open('rb') as f:
        while block:=f.read(chunk_bytes):
            h.update(block);values=np.frombuffer(block,dtype='<u2')
            positions=np.flatnonzero(values==eos).astype(np.uint64)+tokens+1
            if positions.size:ends.append(positions)
            tokens+=values.size
    assert h.hexdigest()==expected_sha,'Token content SHA mismatch'
    end=np.concatenate(ends) if ends else np.array([],dtype=np.uint64)
    if len(end)!=documents:
        raise ValueError('EOS/document count differs; literal special tokens require replay')
    if tokens and (not len(end) or end[-1]!=tokens):raise ValueError('Document has no terminal EOS')
    start=np.concatenate([np.array([0],dtype=np.uint64),end[:-1]]) if len(end) else end.copy()
    now=path.stat();assert(before.st_size,before.st_mtime_ns,before.st_ino)==(now.st_size,now.st_mtime_ns,now.st_ino)
    return start,end,{'bytes':now.st_size,'tokens':int(tokens),'documents':documents,'sha256':h.hexdigest()}


def export_offsets(assignment,train_path,eval_path,certificate,out,*,eos,tokenizer_sha):
    assignment=Path(assignment);out=Path(out)
    if out.exists():raise FileExistsError('Preserve previous offset export')
    c=json.loads(Path(certificate).read_text());by_name={r['name']:r for r in c['outputs']}
    ends={};starts={};proofs={}
    for split,path,count in [('train',Path(train_path),c['train_docs']),('eval',Path(eval_path),c['holdout_docs'])]:
        starts[split],ends[split],proofs[split]=scan_boundaries(path,eos,count,by_name[path.name]['sha256'])
    table=pq.read_table(assignment,columns=['row','split'],use_threads=False)
    rows=table.column('row').to_numpy();splits=table.column('split').to_pylist()
    assert np.array_equal(rows,np.arange(len(rows))), 'Assignment row coverage mismatch'
    start_out=np.empty(len(rows),dtype=np.uint64);length_out=np.empty(len(rows),dtype=np.uint64)
    for split in ['train','eval']:
        indexes=np.fromiter((i for i,s in enumerate(splits) if s==split),dtype=np.int64)
        assert len(indexes)==len(starts[split]),'Token/split document count mismatch'
        start_out[indexes]=starts[split];length_out[indexes]=ends[split]-starts[split]
    assert set(splits)<= {'train','eval'}
    schema=pa.schema([('row',pa.int64()),('split',pa.string()),('token_start',pa.uint64()),('token_length_including_eos',pa.uint64())],
        metadata={**(table.schema.metadata or {}),b'tokenizer_sha256':tokenizer_sha.encode(),b'eos_id':str(eos).encode(),
                  b'format':b'uint16-little-endian-eos-split-v3',b'train_file':Path(train_path).name.encode(),
                  b'eval_file':Path(eval_path).name.encode()})
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix('.parquet.building')
    if temporary.exists():raise FileExistsError('Preserve unfinished offset export')
    pq.write_table(pa.Table.from_arrays([pa.array(rows),pa.array(splits),pa.array(start_out),pa.array(length_out)],schema=schema),temporary,compression='zstd')
    check=pq.read_table(temporary,use_threads=False)
    assert np.array_equal(check.column('token_start').to_numpy(),start_out)
    assert np.array_equal(check.column('token_length_including_eos').to_numpy(),length_out)
    temporary.rename(out)
    with out.open('rb') as f:os.fsync(f.fileno())
    report={'status':'offsets_verified_against_certified_bins_and_assignments','rows':len(rows),'tokenizer_sha256':tokenizer_sha,
            'eos_id':eos,'files':proofs,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
            'producer_contract_id':c['contract_id'],'requires_one_appended_eos_per_document_contract':True,
            'text_reencoded':False,'dtype':'uint16','byte_order':'little'}
    marker=out.with_suffix('.json');tmp=marker.with_suffix('.json.tmp')
    with tmp.open('w') as f:f.write(json.dumps(report,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(marker)
    return report
