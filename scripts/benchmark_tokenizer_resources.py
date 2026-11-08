"""Experimental-only BPE/thread and tokenization/worker sweep on a fixed real sample.
The sample is NOT train-only; all tokenizers here are forbidden as final candidates.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import resource
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file,identity


def prepare(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from common.source_inventory import load_audit,definition
    pa.set_cpu_count(2);pa.set_io_thread_count(2)
    out=resolve(a.output);local=resolve(a.scratch)/'samples';local.mkdir(parents=True,exist_ok=False)
    source=resolve(a.sources);inv=json.loads(source.read_text());proof=json.loads(Path(str(source)+'.done').read_text())
    if proof['sha256']!=sha256_file(source):raise ValueError('Ready input modified')
    audit=load_audit(inv['audit_proof']['path'])
    if (sha256_file(resolve(inv['audit_proof']['path']))!=inv['audit_proof']['sha256'] or
            definition(inv)!=audit['definition'] or inv['content_inventory']!=audit['content_inventory']):raise ValueError('Audit binding mismatch')
    weights={'fineweb2_ko':.50,'fineweb_edu':.35,'github_code_v1':.075,'owm':.075}
    paths=[local/f'part{i:03d}.jsonl' for i in range(32)];streams=[p.open('w') for p in paths]
    stats={};selected=[];index=0;started=time.monotonic()
    try:
        for name,weight in weights.items():
            target=int(a.mb*1024**2*weight);used=docs=0
            records=list(inv['content_inventory']['sources'][name]['files']);random.Random(31).shuffle(records)
            for f in records[:2]:
                path=resolve(f['path'])
                if path.stat().st_size!=f['bytes'] or sha256_file(path)!=f['sha256']:raise ValueError('Sample source changed')
                selected.append(dict(source=name,**f));pf=pq.ParquetFile(path)
                order=list(range(pf.num_row_groups));random.Random(41).shuffle(order)
                offsets=[];n=0
                for g in range(pf.num_row_groups):offsets.append(n);n+=pf.metadata.row_group(g).num_rows
                for rg in order:
                    row=offsets[rg]
                    for batch in pf.iter_batches(batch_size=256,row_groups=[rg],columns=['text']):
                        for body in batch.column(0).to_pylist():
                            text=body[:20000];size=len(text.encode())
                            if size:
                                item=dict(source=name,file=f['path'],row=row,text=text,truncated=len(text)!=len(body),
                                          body_sha256=hashlib.sha256(body.encode()).hexdigest(),text_sha256=hashlib.sha256(text.encode()).hexdigest())
                                streams[index%32].write(json.dumps(item,ensure_ascii=False)+'\n');index+=1;used+=size;docs+=1
                            row+=1
                            if used>=target:break
                        if used>=target:break
                    if used>=target:break
                if used>=target:break
            if used<target:raise ValueError('Sample budget not reached: '+name)
            stats[name]=dict(documents=docs,utf8_bytes=used,target_bytes=target)
    finally:
        for stream in streams:stream.close()
    report=dict(status='experimental_performance_only',source_sha256=sha256_file(source),seed=31,row_group_seed=41,
                max_chars=20000,statistics=stats,selected_files=selected,documents=index,
                shards=[dict(path=str(p),sha256=sha256_file(p)) for p in paths],elapsed_seconds=time.monotonic()-started,
                scope='Bounded shuffled prefix performance sample, not representative quality or train/eval-separated data. Never adopt its tokenizer for final training.')
    atomic_json(out/'frame.json',report)


def frame(path):
    r=json.loads(resolve(path).read_text())
    if r['status']!='experimental_performance_only':raise ValueError('Require experimental performance frame')
    for f in r['shards']:
        if sha256_file(resolve(f['path']))!=f['sha256']:raise ValueError('Sample changed')
    return r


def texts(paths):
    for p in paths:
        with resolve(p).open() as stream:
            for line in stream:yield json.loads(line)['text']


def train(a):
    from tokenizer.train_tokenizer import make_tokenizer,SPECIAL
    from tokenizers import trainers,pre_tokenizers
    r=frame(a.frame);out=resolve(a.output);tok=make_tokenizer('raw');start=time.perf_counter()
    tok.train_from_iterator(texts([f['path'] for f in r['shards']]),trainers.BpeTrainer(vocab_size=65536,special_tokens=SPECIAL,
              show_progress=False,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=2))
    elapsed=time.perf_counter()-start
    if tok.get_vocab_size()!=65536:raise ValueError('Experimental sample did not reach vocab65536')
    target=out/'experimental_tokenizer.json';tok.save(str(target))
    probe=list(texts([r['shards'][0]['path']]))[:256]
    for text in probe:
        encoded=tok.encode(text,add_special_tokens=False)
        if tok.decode(encoded.ids,skip_special_tokens=False)!=text:raise ValueError('Raw sample roundtrip failed')
    atomic_json(out/'result.json',dict(status='experimental_performance_only',rayon=int(os.environ['RAYON_NUM_THREADS']),
                train_seconds=elapsed,tokenizer_sha256=sha256_file(target),semantic_id=identity(json.loads(target.read_text())),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,probe_roundtrips=len(probe),vocab=tok.get_vocab_size()))


_TOKENIZER=None
def init_encoder(path):
    global _TOKENIZER
    from tokenizers import Tokenizer
    _TOKENIZER=Tokenizer.from_file(path)


def encode_part(task):
    from data.tokenize_shards import uint16_ids
    source,target=map(Path,task);eos=_TOKENIZER.token_to_id('<|eos|>');docs=tokens=0;pending=[]
    if eos is None:raise ValueError('Missing EOS')
    def batch(stream,items):
        ids=[]
        for encoded in _TOKENIZER.encode_batch(items,add_special_tokens=False):ids.extend(encoded.ids);ids.append(eos)
        values=uint16_ids(ids);stream.write(values.astype('<u2',copy=False).tobytes());return len(values)
    with source.open() as inp,target.open('wb') as dst:
        for line in inp:
            pending.append(json.loads(line)['text']);docs+=1
            if len(pending)>=256:tokens+=batch(dst,pending);pending=[]
        if pending:tokens+=batch(dst,pending)
    return dict(shard=source.name,documents=docs,tokens=tokens,sha256=sha256_file(target),bytes=target.stat().st_size,
                worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def encode(a):
    r=frame(a.frame);out=resolve(a.output);local=resolve(a.scratch)/out.name;local.mkdir(parents=True,exist_ok=False)
    tasks=[(f['path'],str(local/(Path(f['path']).stem+'.bin'))) for f in r['shards']]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_encoder,initargs=(str(resolve(a.tokenizer)),)) as pool:results=list(pool.map(encode_part,tasks))
    elapsed=time.perf_counter()-started
    logical=[{k:x[k] for k in ['shard','documents','tokens','sha256','bytes']} for x in results]
    atomic_json(out/'result.json',dict(status='experimental_performance_only',workers=a.workers,rayon_per_worker=int(os.environ['RAYON_NUM_THREADS']),
              elapsed_seconds=elapsed,documents=sum(x['documents'] for x in results),tokens=sum(x['tokens'] for x in results),
              tokens_per_second=sum(x['tokens'] for x in results)/elapsed,output_identity=identity(logical),shards=results))


def main(a):
    from scripts.hardware_profile import inspect
    from scripts.parallel_job_support import run_cases,certificate
    root=resolve(a.output);root.mkdir(parents=True,exist_ok=False);hw=inspect();cpus=hw['cpu']['affinity'][:64]
    if hw['memory']['conservative_headroom_gib']<384:raise RuntimeError('Insufficient concurrent-job RAM headroom')
    scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=False)
    files=['scripts/benchmark_tokenizer_resources.py','scripts/parallel_job_support.py','tokenizer/train_tokenizer.py',
           'data/tokenize_shards.py','common/data_contracts.py','common/source_inventory.py']
    contract=dict(code={p:sha256_file(resolve(p)) for p in files},hardware=hw,cpu_affinity=cpus,sample_mib=a.mb,
                  bpe_threads=[8,16,32,64],encoding=[dict(workers=w,rayon=48//w) for w in [4,8,16]],
                  max_case_seconds=900,max_case_rss_gib=128,scope='Performance preparation only; unsplit experimental sample/tokenizer, not final candidate approval.')
    atomic_json(root/'contract.json',contract)
    def launch(name,mode,extra,rayon=1):
        out=root/name;env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS=str(rayon),
                              TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch))
        cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,str(Path(__file__).resolve()),'--mode',mode,'--output',str(out),'--scratch',str(scratch)]+extra
        execution=run_cases(root,[dict(name=name,out=str(out),env=env,cmd=cmd)],max_seconds=900,max_rss_gib=128)
        return out,execution
    prepared,_=launch('prepare','prepare',['--sources',a.sources,'--mb',str(a.mb)])
    fp=prepared/'frame.json';bpe=[]
    for n in contract['bpe_threads']:
        out,execution=launch(f'bpe_{n}','train',['--frame',str(fp)],n)
        r=json.loads((out/'result.json').read_text());r['execution']=execution
        if bpe and r['semantic_id']!=bpe[0]['semantic_id']:raise ValueError('BPE semantic identity differs across threads')
        bpe.append(r);atomic_json(root/'bpe_partial.json',bpe)
    tok=root/'bpe_8/experimental_tokenizer.json';encoded=[]
    for repeat,order in enumerate([[4,8,16],[16,8,4]]):
        for w in order:
            out,execution=launch(f'encode_w{w}_r{repeat}','encode',['--frame',str(fp),'--workers',str(w),'--tokenizer',str(tok)],48//w)
            r=json.loads((out/'result.json').read_text());r['execution']=execution
            if encoded and r['output_identity']!=encoded[0]['output_identity']:raise ValueError('Token output differs across workers')
            encoded.append(r);atomic_json(root/'encode_partial.json',encoded)
    medians={str(w):statistics.median(r['elapsed_seconds'] for r in encoded if r['workers']==w) for w in [4,8,16]}
    report=dict(status='measurement_complete',scope=contract['scope'],bpe=bpe,encoding=encoded,
                fastest_measured_bpe_threads=min(bpe,key=lambda r:r['train_seconds'])['rayon'],
                encoding_median_seconds=medians,fastest_measured_encoding_workers=int(min(medians,key=medians.get)),
                frame_sha256=sha256_file(fp),production_adoption='pending')
    certificate(root,report,contract)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['prepare','train','encode'])
    p.add_argument('--sources',default='data/datasets/final_v1.ready.json');p.add_argument('--scratch');p.add_argument('--frame');p.add_argument('--tokenizer')
    p.add_argument('--workers',type=int);p.add_argument('--mb',type=int,default=128);a=p.parse_args()
    if not 16<=a.mb<=256:p.error('Bound sample to16..256MiB')
    {'prepare':prepare,'train':train,'encode':encode}.get(a.mode,main)(a)
