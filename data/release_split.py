"""Export canonical assignments into resumable, file-scoped portable Parquet.

Never modifies the canonical SQLite DB. Content hashes and group/repo identity
are preserved; doc_id is derivable from the original shard and row contract.
"""
import argparse
import hashlib
import json
import os
import resource
import sqlite3
import struct
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from common.paths import PROJECT_ROOT
from data.release_provenance import sha


def logical_digest(rows):
    h=hashlib.sha256()
    for row,body,repo,group,split in rows:
        assert split in ('train','eval')
        body=bytes.fromhex(body) if isinstance(body,str) else body
        assert len(body)==32
        h.update(struct.pack('<Q',row)+body)
        for value in (repo,group):
            if value is None:h.update(struct.pack('<i',-1))
            else:
                b=value.encode('utf8');h.update(struct.pack('<i',len(b))+b)
        h.update(b'T' if split=='train' else b'E')
    return h.digest()


def export_one(db,record,dest,contract_id,*,batch_size=32768,compression='zstd'):
    dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('row',pa.int64()),('content_sha256',pa.binary(32)),('repo',pa.string()),
        ('group_id',pa.string()),('split',pa.string())],metadata={
        b'contract_id':contract_id.encode(),b'original_final_file':record['path'].encode(),
        b'source':record['source'].encode(),b'schema_version':b'portable-split-v1'})
    temporary=dest.with_suffix('.parquet.building')
    if temporary.exists():raise FileExistsError('Preserve unfinished output; inspect before retry')
    cursor=db.execute('SELECT row,content_sha256,repo,group_id,split FROM docs WHERE file=? ORDER BY row', (record['path'],))
    # Block digests are combined with boundaries; batch size is part of contract.
    logical=hashlib.sha256();counts={'train':0,'eval':0};next_row=0;blocks=0
    start=time.monotonic()
    with pq.ParquetWriter(temporary,schema,compression=compression) as writer:
        while batch:=cursor.fetchmany(batch_size):
            for row,body,repo,group,split in batch:
                if row!=next_row:raise ValueError('Canonical row gap or duplicate')
                if split not in counts:raise ValueError('Invalid split')
                if not group:raise ValueError('Missing group identity')
                counts[split]+=1;next_row+=1
            logical.update(struct.pack('<I',len(batch))+logical_digest(batch));blocks+=1
            columns=list(zip(*batch))
            writer.write_table(pa.Table.from_arrays([
                pa.array(columns[0],type=pa.int64()),
                pa.array([bytes.fromhex(s) for s in columns[1]],type=pa.binary(32)),
                pa.array(columns[2],type=pa.string()),pa.array(columns[3],type=pa.string()),
                pa.array(columns[4],type=pa.string())],schema=schema))
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*2**30:
                raise MemoryError('2 GiB export RSS budget exceeded')
    assert next_row==record['rows'],'Canonical row count mismatch'
    # Decode every outgoing assignment and recompute the same logical digest.
    verify=hashlib.sha256();verified=0
    for batch in pq.ParquetFile(temporary).iter_batches(batch_size=batch_size,use_threads=False):
        arrays=[c.to_pylist() for c in batch.columns]
        rows=list(zip(*arrays));verify.update(struct.pack('<I',len(rows))+logical_digest(rows));verified+=len(rows)
    assert verified==next_row and verify.digest()==logical.digest(),'Portable assignment differs'
    temporary.rename(dest)
    with dest.open('rb') as f:os.fsync(f.fileno())
    return {'original_final_file':record['path'],'source':record['source'],'rows':next_row,'counts':counts,
        'bytes':dest.stat().st_size,'sha256':sha(dest),'logical_digest':logical.hexdigest(),
        'logical_encoding':'block-length-le32+sha256(row-le64,body32,nullable-length-le32-utf8(repo,group),T/E)',
        'batch_size':batch_size,'compression':compression,'elapsed_seconds':time.monotonic()-start,
        'worker_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def export_job(args):
    database,record,dest,contract_id,batch_size,compression=args
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    db=sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    db.execute('PRAGMA cache_size=-131072')
    try:return export_one(db,record,dest,contract_id,batch_size=batch_size,compression=compression)
    finally:db.close()


def export(manifest,out,*,source=None,max_files=None,batch_size=32768,compression='zstd',workers=1):
    root=Path(PROJECT_ROOT);manifest=Path(manifest);m=json.loads(manifest.read_text())
    database=root/m['database']['path'];before=database.stat()
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-max(4,workers):])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    contract={'split_contract_id':m['contract_id'],'database_sha256_prior':m['database']['sha256'],
        'database_stat':{'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'inode':before.st_ino},
        'batch_size':batch_size,'compression':compression,'source':source,'max_files':max_files,'workers':workers}
    cp=out/'contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract,'Export inputs changed'
    else:cp.write_text(json.dumps(contract,indent=2)+'\n')
    records=[r for r in m['contract']['files'] if source is None or r['source']==source]
    if max_files is not None:records=records[:max_files]
    results=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={}
        for r in records:
            dest=out/r['source']/(Path(r['path']).stem+'.parquet');marker=dest.with_suffix('.json')
            if marker.exists():
                item=json.loads(marker.read_text());assert dest.stat().st_size==item['bytes'] and sha(dest)==item['sha256']
            else:
                args=(str(database),r,str(dest),m['contract_id'],batch_size,compression)
                pending[pool.submit(export_job,args)]=(r,marker)
                continue
            expected=m['per_file'][r['path']];assert item['counts']=={k:expected[k] for k in ['train','eval']}
            results.append(item)
        for future in as_completed(pending):
            r,marker=pending[future];item=future.result()
            expected=m['per_file'][r['path']];assert item['counts']=={k:expected[k] for k in ['train','eval']}
            tmp_marker=marker.with_suffix('.json.tmp')
            with tmp_marker.open('w') as f:
                f.write(json.dumps(item,indent=2)+'\n');f.flush();os.fsync(f.fileno())
            tmp_marker.replace(marker)
            results.append(item)
            if len(results)%10==0 or len(results)==len(records):
                print(json.dumps({'files_done':len(results),'files_target':len(records),'rows':sum(i['rows'] for i in results)}),flush=True)
    results.sort(key=lambda item:item['original_final_file'])
    now=database.stat();assert(before.st_size,before.st_mtime_ns,before.st_ino)==(now.st_size,now.st_mtime_ns,now.st_ino)
    report={'status':'portable_assignments_verified','scope':'whole_canonical' if source is None and max_files is None else 'selected_files',
        'files':results,'rows':sum(i['rows'] for i in results),'bytes':sum(i['bytes'] for i in results),
        'elapsed_seconds':time.monotonic()-start,'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'max_worker_peak_rss_bytes':max(i.get('worker_peak_rss_bytes',0) for i in results),'workers':workers,
        'fresh_global_leakage_check':False,'prior_canonical_proofs_reused':True,'contract':contract}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'.done').write_text(sha(out/'report.json')+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--source');p.add_argument('--max-files',type=int)
    p.add_argument('--batch-size',type=int,default=32768);p.add_argument('--compression',choices=['zstd','snappy'],default='zstd')
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    assert a.batch_size>0 and (a.max_files is None or a.max_files>0) and 1<=a.workers<=8
    root=Path(PROJECT_ROOT)
    report=export(root/'data/datasets/final_v2.split/manifest.json',root/a.output,source=a.source,max_files=a.max_files,batch_size=a.batch_size,compression=a.compression,workers=a.workers)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','contract')}))
