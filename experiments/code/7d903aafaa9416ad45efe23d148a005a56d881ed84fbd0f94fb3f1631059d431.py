"""Isolated shard scheduler with native v3 token layout and per-shard crash recovery.
One parent validates the closed split and sources; spawned workers share that invocation's
verified contract, not a persistent skip-validation marker. Production adoption is separate.
"""
import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,identity,sha256_file
from data.tokenize_shards import uint16_ids,commit_outputs,GROUPED_TOKEN_FORMAT
from data.verified_split_context import VerifiedSplitContext

_WORKER=None


def initialize(tokenizer,tokenizer_sha,database,arrow_threads,cache_mib):
    global _WORKER
    import pyarrow as pa
    from tokenizers import Tokenizer
    pa.set_cpu_count(arrow_threads);pa.set_io_thread_count(arrow_threads)
    if sha256_file(tokenizer)!=tokenizer_sha:raise ValueError('Tokenizer changed')
    tok=Tokenizer.from_file(tokenizer);eos=tok.token_to_id('<|eos|>')
    if eos is None:raise ValueError('Missing EOS')
    uint16_ids(list(tok.get_vocab().values()));uint16_ids([eos])
    db=sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)
    db.execute(f'PRAGMA cache_size={-cache_mib*1024}')
    _WORKER=(tok,eos,db)


def shard(task):
    import pyarrow.parquet as pq
    tok,eos,db=_WORKER;out=Path(task['output']);inp=task['input'];path=resolve(inp['path'])
    name=path.stem+'.bin';holdname=path.stem+'.hold.bin';counts=task['counts']
    train_docs=counts.get('train',0);eval_docs=counts.get('eval',0)
    names=[name]+([holdname] if eval_docs else []);marker=out/(name+'.done');pending=out/(name+'.pending.json')
    if marker.exists() or pending.exists():
        r=json.loads((marker if marker.exists() else pending).read_text())
        if (r['contract_id']!=task['contract_id'] or r['marker']!=marker.name or
                [x['name'] for x in r['outputs']]!=names or r['train_docs']!=train_docs or r['holdout_docs']!=eval_docs or
                r['holdout_file']!=(holdname if eval_docs else None) or r['outputs'][0]['bytes']!=2*r['tokens'] or
                (eval_docs and r['outputs'][1]['bytes']!=2*r['holdout_tokens'])):raise ValueError('Invalid resumed shard record')
        commit_outputs(out,r,publish_done=not marker.exists())
        return dict(record=r,reused=True,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if path.stat().st_size!=inp['bytes']:raise ValueError('Input size changed')
    for n in names:
        if (out/n).exists():raise ValueError('Uncertified output; refusing overwrite')
        tmp=out/(n+'.tmp')
        if tmp.exists():tmp.rename(out/(n+'.partial-'+uuid.uuid4().hex))
    streams=[(out/(n+'.tmp')).open('xb') for n in names];hashes=[hashlib.sha256() for n in names];tokens=[0]*len(names)
    cursor=iter(db.execute('SELECT row,split,content_sha256 FROM docs WHERE source=? AND file=? ORDER BY row',(task['source'],inp['path'])))
    offset=0;seen=[0,0]
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=task['batch'],columns=['text']):
            texts=batch.column(0).to_pylist()
            if not all(isinstance(t,str) for t in texts):raise ValueError('Non-string text')
            ids=[[],[]]
            for text,encoded in zip(texts,tok.encode_batch(texts,add_special_tokens=False)):
                row,label,body_sha=next(cursor)
                if row!=offset or label not in ('train','eval'):raise ValueError('Split row coverage mismatch')
                if hashlib.sha256(text.encode()).hexdigest()!=body_sha:raise ValueError('Source body changed after verification')
                which=int(label=='eval');ids[which].extend(encoded.ids);ids[which].append(eos);seen[which]+=1;offset+=1
            for j,stream in enumerate(streams):
                values=uint16_ids(ids[j]);body=values.astype('<u2',copy=False).tobytes()
                stream.write(body);hashes[j].update(body);tokens[j]+=len(values)
        if next(cursor,None) is not None or seen!=[train_docs,eval_docs]:raise ValueError('Split count mismatch')
        for stream in streams:stream.flush();os.fsync(stream.fileno())
    finally:
        for stream in streams:stream.close()
    r=dict(contract_id=task['contract_id'],marker=marker.name,tokens=tokens[0],holdout_tokens=tokens[1] if eval_docs else 0,
           train_docs=train_docs,holdout_docs=eval_docs,holdout_file=holdname if eval_docs else None,
           outputs=[dict(name=n,bytes=tokens[j]*2,sha256=hashes[j].hexdigest()) for j,n in enumerate(names)])
    atomic_json(pending,r)
    if task.get('_crash')=='after_pending':os._exit(73)
    commit_outputs(out,r)
    return dict(record=r,reused=False,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run(sources,split,tokenizer,output,*,workers=8,batch_size=1024,arrow_threads=1,hash_workers=16,experimental=False,_crash=None):
    if not 1<=workers<=64 or not 1<=batch_size<=65536 or not 1<=arrow_threads<=8:raise ValueError('Invalid resource settings')
    root=resolve(output);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with VerifiedSplitContext(sources,split,hash_workers=hash_workers) as context:
            tokenizer=resolve(tokenizer);tok_sha=sha256_file(tokenizer)
            from tokenizers import Tokenizer
            tok=Tokenizer.from_file(str(tokenizer));eos=tok.token_to_id('<|eos|>')
            if eos is None:raise ValueError('Missing EOS')
            uint16_ids(list(tok.get_vocab().values()));del tok
            if not experimental:
                proof=json.loads(Path(str(tokenizer)+'.done').read_text())
                if (not context.inventory.get('audit_proof') or proof.get('tokenizer_sha256')!=tok_sha or
                        proof.get('config',{}).get('split_contract_id')!=context.selection.report['contract_id']):
                    raise ValueError('Production requires audited input and a final train-only split-bound tokenizer')
            contract=dict(version=1,source_sha256=context.source_sha,split_sha256=context.split_sha,tokenizer_sha256=tok_sha,experimental=experimental,
                          code={f:sha256_file(resolve(f)) for f in ['data/parallel_tokenize.py','data/verified_split_context.py','data/tokenize_shards.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py']})
            cp=root/'contract.json'
            if cp.exists():
                if json.loads(cp.read_text())!=contract:raise ValueError('Parallel contract changed; use a new output')
            else:
                if set(p.name for p in root.iterdir())!={'.lock'}:raise ValueError('Uncontracted output directory')
                atomic_json(cp,contract)
            tasks=[];plans={}
            for source,records in context.records.items():
                if not source or Path(source).name!=source or source in ('.','..'):raise ValueError('Unsafe source name')
                out=root/source;out.mkdir(exist_ok=True)
                plan=dict(schema_version=2,source=source,format=GROUPED_TOKEN_FORMAT,dtype='uint16',tokenizer_sha256=tok_sha,
                          eos_id=eos,snapshot_id=context.snapshot,holdout_docs=0,inputs=records,split_contract_id=context.selection.report['contract_id'])
                plan['contract_id']=identity(plan);plans[source]=plan;mp=out/'manifest.json'
                if mp.exists() and json.loads(mp.read_text())!=plan:raise ValueError('Source contract changed')
                if not mp.exists():atomic_json(mp,plan)
                filenames=[]
                for inp in records:
                    counts=context.selection.counts(resolve(inp['path']));stem=Path(inp['path']).stem
                    filenames.extend([stem+'.bin']+([stem+'.hold.bin'] if counts.get('eval') else []))
                    tasks.append(dict(source=source,input=inp,counts=counts,output=str(out),contract_id=plan['contract_id'],batch=batch_size,_crash=_crash))
                if len(set(filenames))!=len(filenames):raise ValueError('Output filename collision')
            verified=time.perf_counter();database=resolve(context.selection.report['database']['path'])
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
                                     initargs=(str(tokenizer),tok_sha,str(database),arrow_threads,32)) as pool:
                results=list(pool.map(shard,tasks))
            context.assert_control_files()
            if sha256_file(database)!=context.selection.report['database']['sha256'] or sha256_file(tokenizer)!=tok_sha:
                raise ValueError('Closed DB/tokenizer changed during execution')
            for f,h in contract['code'].items():
                if sha256_file(resolve(f))!=h:raise ValueError('Producer changed during execution')
            for source,plan in plans.items():
                selected=[r['record'] for task,r in zip(tasks,results) if task['source']==source]
                totals={k:sum(r[k] for r in selected) for k in ['tokens','holdout_tokens','train_docs','holdout_docs']}
                atomic_json(root/source/'summary.json',dict(contract_id=plan['contract_id'],**totals))
                atomic_json(root/source/'.done',dict(contract_id=plan['contract_id'],shards=len(plan['inputs'])))
            logical=[dict(source=task['source'],file=task['input']['path'],record=r['record']) for task,r in zip(tasks,results)]
            report=dict(status='experimental_complete' if experimental else 'complete',workers=workers,batch_size=batch_size,arrow_threads=arrow_threads,
                        elapsed_seconds=time.perf_counter()-start,verification_seconds=verified-start,logical_id=identity(logical),
                        tokens=sum(r['record']['tokens']+r['record']['holdout_tokens'] for r in results),shards=len(tasks),
                        reused=sum(r['reused'] for r in results),max_worker_peak_rss_kib=max(r['peak_rss_kib'] for r in results),
                        scope='Native v3 bytes; production adoption requires final-input pilot. Hash/DB validation repeated on every new invocation.')
            atomic_json(root/'report.json',report);atomic_json(root/'.done',dict(report_sha256=sha256_file(root/'report.json'),contract_sha256=sha256_file(cp)))
            return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['sources','split','tokenizer','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--arrow-threads',type=int,default=1)
    p.add_argument('--experimental',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.sources,a.split,a.tokenizer,a.output,workers=a.workers,batch_size=a.batch_size,arrow_threads=a.arrow_threads,experimental=a.experimental)))
