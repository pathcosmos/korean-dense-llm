"""Bounded fixed-frame diagnostic sampling from immutable completed Parquet files.
No ready promotion, full-file hash claim, full-corpus scan or source modification.
Read budgets charge the entire selected columns of a row group before decompression.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from common.data_contracts import resolve,artifact_path,identity,sha256_file,atomic_json
from data.split_manifest import text_hash


def load_samples(manifest):
    manifest=resolve(manifest);r=json.loads(manifest.read_text());marker=manifest.parent/'.done'
    if r.get('run_status')!='completed' or r.get('contract_id')!=identity(r['contract']) or not marker.exists() or json.loads(marker.read_text())!={'manifest_sha256':sha256_file(manifest)}:
        raise ValueError('Uncertified sample snapshot')
    path=resolve(r['samples']['path'])
    if sha256_file(path)!=r['samples']['sha256']:raise ValueError('Sample corpus changed')
    records=[];seen=set()
    with path.open() as stream:
        for line in stream:
            row=json.loads(line);key=(row['file'],row['row'])
            if key in seen or text_hash(row['text'])!=row['view_sha256']:raise ValueError('Duplicate/changed sample row')
            seen.add(key);records.append(row)
    if len(records)!=sum(s['documents'] for s in r['stats'].values()):raise ValueError('Sample count mismatch')
    return r,records


def sample(frame_path,output_dir,*,per_source=256,max_bytes=16*1024**2,max_read_bytes=128*1024**2,
           max_row_groups=32,max_chars=4096,max_files_per_source=4,row_groups_per_file=2,seed=1234,max_seconds=120):
    for value in [per_source,max_bytes,max_read_bytes,max_row_groups,max_chars,max_files_per_source,row_groups_per_file,max_seconds]:
        if type(value) is not int or value<1:raise ValueError('Require positive integer sampling limits')
    if type(seed) is not int or seed<0:raise ValueError('Invalid seed')
    frame_path=resolve(frame_path);frame=json.loads(frame_path.read_text())
    if frame.get('schema_version')!=1 or not frame.get('sources'):raise ValueError('Invalid sample frame')
    descriptors={};seen=set()
    for source,entry in frame['sources'].items():
        if entry.get('kind') not in ('web','code') or not entry.get('revision') or not entry.get('files'):raise ValueError('Frame requires kind/revision/files')
        descriptors[source]=[]
        for file in entry['files']:
            p=resolve(file);marker=Path(str(p)+'.done')
            if not p.is_file() or not marker.is_file():raise ValueError('Frame includes incomplete source')
            key=artifact_path(p)
            if key in seen:raise ValueError('Repeated frame file')
            seen.add(key);pf=pq.ParquetFile(p)
            descriptors[source].append(dict(path=key,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
                rows=pf.metadata.num_rows,completion_sha256=sha256_file(marker)))
    config=dict(per_source=per_source,max_bytes=max_bytes,max_read_bytes=max_read_bytes,max_row_groups=max_row_groups,
                max_chars=max_chars,max_files_per_source=max_files_per_source,row_groups_per_file=row_groups_per_file,seed=seed,max_seconds=max_seconds)
    contract=dict(version=2,frame_sha256=sha256_file(frame_path),frame=frame,files=descriptors,config=config,
                  code_sha256=sha256_file(Path(__file__)),full_source_hash_verified=False,
                  selection='Seeded file/row-group strata, capacity-aware row quotas and seeded row positions; bounded diagnostic frame only')
    cid=identity(contract);out=resolve(output_dir);report_path=out/'manifest.json'
    if report_path.exists():
        previous,_=load_samples(report_path)
        if previous['contract_id']!=cid:raise ValueError('Sample frame/config changed; use a new directory')
        return previous
    if out.exists() and any(out.iterdir()):raise ValueError('Partial sample exists; preserve and use a new directory')
    out.mkdir(parents=True,exist_ok=True);target=out/'samples.jsonl';tmp=out/'samples.jsonl.tmp'
    stats={};reads=groups=stored=0;limits=set();started=time.monotonic()
    with tmp.open('w') as stream:
        for source,entry in frame['sources'].items():
            rng=random.Random(identity([seed,source]));files=list(descriptors[source]);rng.shuffle(files);files=files[:max_files_per_source]
            specs=[]
            for desc in files:
                p=pq.ParquetFile(resolve(desc['path']));gs=list(range(p.num_row_groups));rng.shuffle(gs)
                for g in gs[:row_groups_per_file]:specs.append((desc,g))
            rng.shuffle(specs);count=used=read_rows=0
            capacities=[pq.ParquetFile(resolve(desc['path'])).metadata.row_group(g).num_rows for desc,g in specs]
            for spec_index,(desc,g) in enumerate(specs):
                if count>=per_source:break
                if time.monotonic()-started>=max_seconds:limits.add('time');break
                p=pq.ParquetFile(resolve(desc['path']));names=p.schema_arrow.names
                columns=['text']+([c for c in ['repo_name','content_sha256'] if c in names] if entry['kind']=='code' else [])
                if entry['kind']=='code' and 'repo_name' not in columns:raise ValueError('Code repository provenance missing')
                rg=p.metadata.row_group(g)
                charge=sum(rg.column(i).total_uncompressed_size for i in range(rg.num_columns) if rg.column(i).path_in_schema in columns)
                if groups>=max_row_groups:limits.add('row_groups');break
                if reads+charge>max_read_bytes:limits.add('read_bytes');continue
                reads+=charge;groups+=1
                quota=max(1,math.ceil((per_source-count)/max(1,len(specs)-spec_index)),per_source-count-sum(capacities[spec_index+1:]))
                chosen=sorted(rng.sample(range(rg.num_rows),min(quota,per_source-count,rg.num_rows)))
                selected=iter(chosen);next_row=next(selected,None);offset=sum(p.metadata.row_group(i).num_rows for i in range(g));position=0
                for batch in p.iter_batches(batch_size=256,row_groups=[g],columns=columns):
                    read_rows+=batch.num_rows
                    for item in batch.to_pylist():
                        if position==next_row:
                            text=item['text']
                            if not isinstance(text,str):raise ValueError('Non-string sample text')
                            digest=text_hash(text);view=text[:max_chars];size=len(view.encode())
                            if stored+size>max_bytes:limits.add('stored_bytes');next_row=None;break
                            if entry['kind']=='code' and item.get('content_sha256',digest)!=digest:raise ValueError('Code body checksum mismatch')
                            if text:
                                row=dict(source=source,kind=entry['kind'],file=desc['path'],row=offset+position,
                                    text=view,content_sha256=digest,view_sha256=text_hash(view),truncated=len(text)!=len(view),
                                    characters_total=len(text),repo=item.get('repo_name'))
                                stream.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1;used+=size;stored+=size
                            next_row=next(selected,None)
                        position+=1
                    if next_row is None:break
                    if time.monotonic()-started>=max_seconds:limits.add('time');break
                current=resolve(desc['path']).stat()
                if (current.st_size,current.st_mtime_ns)!=(desc['bytes'],desc['mtime_ns']):raise ValueError('Sampled source changed')
            stats[source]=dict(documents=count,stored_utf8_bytes=used,decoded_batch_rows=read_rows,target=per_source)
    tmp.replace(target)
    # Incomplete targets caused by an explicit bound are recorded as partial, not full coverage.
    status='complete_for_defined_frame' if all(s['documents']>=per_source for s in stats.values()) and not limits else 'partial'
    report=dict(run_status='completed',sample_status=status,contract_id=cid,contract=contract,stats=stats,
        limits_reached=sorted(limits),charged_row_group_bytes=reads,row_groups_read=groups,stored_utf8_bytes=stored,
        elapsed_seconds=time.monotonic()-started,samples=dict(path=artifact_path(target),sha256=sha256_file(target)),
        scope='Only selected completed files/row groups and stored views; no full-corpus representativeness or source content hash claim')
    atomic_json(report_path,report);atomic_json(out/'.done',dict(manifest_sha256=sha256_file(report_path)))
    return report


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--frame',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--per-source',type=int,default=256);a=ap.parse_args()
    print(json.dumps(sample(a.frame,a.output_dir,per_source=a.per_source)))
