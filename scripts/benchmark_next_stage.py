"""Bounded experimental Parquet/split/native-token and larger BPE campaign.
Final F4, tokenizer, and token data are never consumed or changed by this campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file,identity


def prepare(a):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.benchmark_tokenizer_resources import prepare as sample,frame
    from data.split_manifest import build
    from scripts.report_split_distribution import report
    out=resolve(a.output);pa.set_cpu_count(2);pa.set_io_thread_count(2)
    sample(SimpleNamespace(output=str(out),scratch=a.scratch,sources='data/datasets/final_v1.ready.json',mb=512))
    fr=frame(out/'frame.json');buckets={};indices={}
    for f in fr['shards']:
        with resolve(f['path']).open() as stream:
            for line in stream:
                r=json.loads(line);name='perf_'+r['source'];n=indices.get(name,0);indices[name]=n+1
                buckets.setdefault((name,n%12),[]).append(r['text'])
    sources={};start=time.monotonic()
    for (name,index),texts in sorted(buckets.items()):
        d=out/'parquet'/name;d.mkdir(parents=True,exist_ok=True);path=d/f'{index:03d}.parquet'
        obj={'text':texts}
        if name=='perf_github_code_v1':obj['repo_name']=[f'perf/repo{index}_{n//8}' for n in range(len(texts))]
        pq.write_table(pa.table(obj),path,row_group_size=1024,compression='zstd');Path(str(path)+'.done').touch()
        entry=sources.setdefault(name,dict(files=[],text_policy='raw-utf8-code-v1' if 'github' in name else 'nfc-web-exact-v1',completion_marker=str(d/'complete')))
        entry['files'].append(str(path));(d/'complete').touch()
    sm=out/'sources.json';atomic_json(sm,dict(schema_version=1,status='ready',sources=sources,
        experimental_provenance=dict(frame_sha256=sha256_file(out/'frame.json'),scope='Repacked truncated performance texts; synthetic code repository labels; split does not represent final corpus.')))
    build(sm,out/'split',eval_fraction=.1,hash_workers=16,sqlite_cache_mib=512)
    report(sm,out/'split/manifest.json',out/'distribution.json')
    atomic_json(out/'result.json',dict(status='experimental_fixture_complete',documents=sum(indices.values()),shards=len(buckets),
        frame_sha256=sha256_file(out/'frame.json'),sources_sha256=sha256_file(sm),fixture_seconds=time.monotonic()-start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def publish_tree(source,target):
    """Fresh experimental publication: verified bytes, fsync, atomic directory rename.
    Completed staging can resume; incomplete staging is preserved for inspection.
    """
    from scripts.publish_local_split import sync_dir,durable_json
    source=Path(source);target=Path(target);stage=target.with_name(target.name+'.publishing')
    records=[dict(path=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in sorted(source.rglob('*')) if p.is_file() and p.name!='.lock']
    proof=dict(version=1,files=records)
    def check(directory):
        if json.loads((directory/'publication.json').read_text())!=proof:raise ValueError('Publication input changed')
        for item in records:
            p=directory/item['path']
            if p.stat().st_size!=item['bytes'] or sha256_file(p)!=item['sha256']:raise ValueError('Publication bytes differ')
    if target.exists():check(target);return
    if stage.exists():check(stage)
    else:
        stage.mkdir(parents=True)
        for item in records:
            src=source/item['path'];dst=stage/item['path'];dst.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256()
            with src.open('rb') as inp,dst.open('xb') as out:
                while block:=inp.read(8*1024**2):out.write(block);digest.update(block)
                out.flush();os.fsync(out.fileno())
            if digest.hexdigest()!=item['sha256']:raise ValueError('Publication copy mismatch')
        for d in sorted([p for p in stage.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):sync_dir(d)
        durable_json(stage/'publication.json',proof)
        check(stage)
    stage.rename(target);sync_dir(target.parent)


def encode(a):
    from data.parallel_tokenize import run
    out=resolve(a.output);frame=resolve(a.frame);start=time.perf_counter()
    target=out/'tokens' if a.storage=='nfs' else resolve(a.scratch)/out.name/'tokens'
    r=run(frame/'sources.json',frame/'split/manifest.json',a.tokenizer,target,
          workers=a.workers,batch_size=a.batch,arrow_threads=a.arrow,experimental=True)
    if a.storage=='local_publish':publish_tree(target,out/'tokens')
    r.update(storage=a.storage,end_to_end_seconds=time.perf_counter()-start,
             parent_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(out/'result.json',r)


def tree_bytes(path):return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def main(a):
    from scripts.hardware_profile import inspect
    from scripts.run_local_split import process_tree_rss,stop
    from scripts.parallel_job_support import observe_main,certificate
    root=resolve(a.output);root.mkdir(parents=True,exist_ok=False);hw=inspect();cpus=hw['cpu']['affinity'][:64]
    if hw['memory']['conservative_headroom_gib']<576:raise RuntimeError('Reserve main future RSS384GiB plus experiment192GiB')
    scratch=Path(hw['storage']['scratch_dir'])/root.name;scratch.mkdir(exist_ok=False)
    paths=['scripts/benchmark_next_stage.py','data/parallel_tokenize.py','data/verified_split_context.py','scripts/report_split_distribution.py',
           'scripts/benchmark_tokenizer_resources.py','scripts/parallel_job_support.py','scripts/publish_local_split.py','scripts/run_local_split.py',
           'scripts/hardware_profile.py','data/tokenize_shards.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py','tokenizer/train_tokenizer.py']
    contract=dict(code={p:sha256_file(resolve(p)) for p in paths},hardware=hw,cpu_affinity=cpus,max_rss_gib=192,
                  max_local_gib=64,max_durable_gib=64,reserve_gib=200,campaign_seconds=1200,
                  initial_workers=[8,16,24],batches=[256,1024,4096],bpe_threads=[16,32],scope='Experimental512MiB repacked Parquet/synthetic groups, not final train-only corpus. No production adoption.')
    atomic_json(root/'contract.json',contract);started=time.monotonic();deadline=started+1200;results=[]
    def launch(name,mode,extra,rayon=1):
        out=root/name;out.mkdir();before=observe_main();start=time.monotonic();peak=0
        env=dict(os.environ,OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS=str(rayon),TOKENIZERS_PARALLELISM='true',TMPDIR=str(scratch))
        cmd=['taskset','-c',','.join(map(str,cpus)),sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode,'--output',str(out),'--scratch',str(scratch)]+extra
        with (out/'console.log').open('w') as log:proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                rss=process_tree_rss(proc.pid);peak=max(peak,rss)
                atomic_json(root/'state.json',dict(phase='running',case=name,elapsed_seconds=time.monotonic()-start,campaign_seconds=time.monotonic()-started,peak_case_rss_bytes=peak,main=observe_main()))
                if time.monotonic()>deadline or rss>192*2**30:raise RuntimeError('Campaign time/RSS budget exceeded')
                for path in [root,scratch]:
                    if shutil.disk_usage(path).free<200*2**30 or tree_bytes(path)>64*2**30:raise RuntimeError('Storage budget exceeded')
                time.sleep(2)
            if proc.returncode:raise RuntimeError('Case failed: '+name)
        except BaseException:
            if proc.poll() is None:stop(proc)
            raise
        r=json.loads((out/'result.json').read_text())
        # Sampling can miss short-lived peaks; inspect worker and parent HWM too.
        for k in ['max_rss_kib','parent_peak_rss_kib','max_worker_peak_rss_kib']:
            if r.get(k,0)*1024>192*2**30:raise RuntimeError('Reported HWM exceeds RSS budget')
        for path in [root,scratch]:
            if tree_bytes(path)>64*2**30 or shutil.disk_usage(path).free<200*2**30:raise RuntimeError('Final storage budget exceeded')
        r.update(case=name,supervisor_seconds=time.monotonic()-start,sampled_group_peak_rss_bytes=peak,main_before=before,main_after=observe_main())
        results.append(r);atomic_json(root/'partial_results.json',results);return r
    try:
        launch('prepare','prepare',[]);fp=root/'prepare'
        # Use prior content-bound experimental tokenizer; BPE comparisons run after encoding.
        tp=resolve('runs/tokenizer_resources_v1/bpe_16/experimental_tokenizer.json')
        prior=json.loads(resolve('runs/tokenizer_resources_v1/bpe_16/result.json').read_text())
        if sha256_file(tp)!=prior['tokenizer_sha256']:raise ValueError('Experimental tokenizer changed')
        contract['experimental_tokenizer']=dict(path=str(tp),sha256=sha256_file(tp));atomic_json(root/'contract.json',contract)
        enc=[]
        def case(name,w,b=1024,arrow=1,storage='nfs'):
            r=launch(name,'encode',['--frame',str(fp),'--tokenizer',str(tp),'--workers',str(w),'--batch',str(b),'--arrow',str(arrow),'--storage',storage],max(1,48//w))
            if enc and r['logical_id']!=enc[0]['logical_id']:raise ValueError('Token output differs between configurations')
            enc.append(r);return r
        case('baseline_w1',1)
        choices=[case(f'workers_{w}',w) for w in [8,16,24]]
        best=min(choices,key=lambda x:x['end_to_end_seconds']);w=best['workers']
        choices+=[case(f'batch_{b}',w,b) for b in [256,4096]]
        best=min(choices,key=lambda x:x['end_to_end_seconds']);choices.append(case('arrow2',best['workers'],best['batch_size'],2))
        finalists=sorted(choices,key=lambda x:x['end_to_end_seconds'])[:2];repeats=[]
        for i,c in enumerate(reversed(finalists)):repeats.append(case(f'repeat_{i}',c['workers'],c['batch_size'],c['arrow_threads']))
        medians=[(statistics.median([c['end_to_end_seconds'],r['end_to_end_seconds']]),c) for c,r in zip(reversed(finalists),repeats)]
        winner=min(medians,key=lambda x:x[0])[1];w,b,ar=winner['workers'],winner['batch_size'],winner['arrow_threads']
        storage=[]
        for i,mode in enumerate(['nfs','local_publish','local_publish','nfs']):storage.append(case(f'storage_{i}_{mode}',w,b,ar,mode))
        bpe=[]
        for n in [16,32]:bpe.append(launch(f'bpe_{n}','bpe',['--frame',str(fp/'frame.json')],n))
        if bpe[0]['semantic_id']!=bpe[1]['semantic_id']:raise ValueError('BPE semantic identity differs')
        report=dict(status='measurement_complete',results=results,encoding_candidate={k:winner[k] for k in ['workers','batch_size','arrow_threads']},
                    storage_median_seconds={m:statistics.median(x['end_to_end_seconds'] for x in storage if x['storage']==m) for m in ['nfs','local_publish']},
                    bpe_candidate_threads=min(bpe,key=lambda x:x['train_seconds'])['rayon'],scope=contract['scope'],production_adoption='pending',
                    note='Adaptive small search, two measurements for finalists/storage; no global optimum or full-corpus speed guarantee.')
        certificate(root,report,contract)
    except BaseException as exc:
        atomic_json(root/'state.json',dict(phase='error',error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['prepare','encode','bpe']);p.add_argument('--scratch');p.add_argument('--frame');p.add_argument('--tokenizer')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--batch',type=int,default=1024);p.add_argument('--arrow',type=int,default=1);p.add_argument('--storage',choices=['nfs','local_publish'],default='nfs');a=p.parse_args()
    if a.mode=='bpe':
        from scripts.benchmark_tokenizer_resources import train
        train(a)
    else:{'prepare':prepare,'encode':encode}.get(a.mode,main)(a)
