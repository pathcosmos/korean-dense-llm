"""Same-input local SQLite cache sweep, including full validation and row digest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file
from data.split_manifest import build,SplitSelection
from scripts.hardware_profile import inspect
from scripts.run_local_split import process_tree_rss,stop,size_tree


def worker(a):
    import pyarrow as pa
    pa.set_cpu_count(4);pa.set_io_thread_count(4)
    case=resolve(a.scratch)/str(a.cache);case.mkdir(parents=True,exist_ok=True)
    out=resolve(a.output);last=0;phases={}
    started=time.monotonic()
    def progress(phase):
        nonlocal last
        phases.setdefault(phase,time.monotonic()-started)
        if time.monotonic()-last>=2:
            atomic_json(out/f'worker_{a.cache}.json',dict(phase=phase,elapsed_seconds=time.monotonic()-started))
            last=time.monotonic()
    report=build(a.sources,case/'split',.1,7,sqlite_cache_mib=a.cache,hash_workers=1,_checkpoint=progress)
    built=time.monotonic();s=SplitSelection(case/'split/manifest.json',sqlite_cache_mib=a.cache)
    verified=time.monotonic();h=hashlib.sha256();rows=0
    try:
        for row in s.db.execute('SELECT source,file,row,content_sha256,repo,group_id,split FROM docs ORDER BY file,row'):
            h.update(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode());h.update(b'\n');rows+=1
    finally:s.close()
    value=dict(cache_mib=a.cache,build_seconds=built-started,verify_seconds=verified-built,
               build_and_verify_seconds=verified-started,digest_seconds=time.monotonic()-verified,
               max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               logical_rows=rows,logical_sha256=h.hexdigest(),counts=report['counts'],
               database_bytes=(case/'split/split.sqlite').stat().st_size,
               manifest_sha256=sha256_file(case/'split/manifest.json'),phase_first_seconds=phases)
    atomic_json(out/f'case_{a.cache}.json',value)


def main(a):
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=False)
    scratch=resolve(a.scratch);scratch.mkdir(parents=True,exist_ok=False)
    hardware=inspect()
    files=['scripts/benchmark_split_resources.py','data/split_manifest.py','common/data_contracts.py',
           'common/source_inventory.py','scripts/run_local_split.py']
    contract=dict(source= str(resolve(a.sources)),source_sha256=sha256_file(resolve(a.sources)),
                  caches_mib=[512,8192,131072],hardware=hardware,code={p:sha256_file(resolve(p)) for p in files},
                  scope='Sequential same-input 3M local comparisons. Cache is an upper bound, not preallocated RAM. No full-corpus speed extrapolation.')
    atomic_json(out/'contract.json',contract)
    results=[]
    for cache in contract['caches_mib']:
        if shutil.disk_usage(scratch).free<220*1024**3:raise RuntimeError('Local space reserve')
        if inspect()['memory']['conservative_headroom_gib']<512:raise RuntimeError('Cache sweep memory headroom')
        with (out/f'console_{cache}.log').open('w') as log:
            env=dict(os.environ,OMP_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',SQLITE_TMPDIR=str(scratch),TMPDIR=str(scratch))
            p=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker','--cache',str(cache),
                                '--sources',a.sources,'--output',a.output,'--scratch',str(scratch)],
                               stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        started=time.monotonic();peak=0
        try:
            while p.poll() is None:
                elapsed=time.monotonic()-started;rss=process_tree_rss(p.pid);peak=max(peak,rss)
                atomic_json(out/'state.json',dict(phase='running',case=cache,pid=p.pid,elapsed_seconds=elapsed,peak_rss_bytes=peak))
                if elapsed>3600 or rss>512*1024**3 or size_tree(scratch/str(cache))>20*1024**3:
                    raise RuntimeError('Cache sweep resource limit')
                time.sleep(2)
            if p.returncode:raise RuntimeError('Cache case failed '+str(cache))
        except BaseException as exc:
            stop(p);atomic_json(out/'state.json',dict(phase='error',error=str(exc)));raise
        r=json.loads((out/f'case_{cache}.json').read_text())
        if results and (r['logical_rows'],r['logical_sha256'])!=(results[0]['logical_rows'],results[0]['logical_sha256']):
            raise ValueError('Logical split changed across cache configurations')
        results.append(r)
        atomic_json(out/'partial_results.json',results)
    assert all(sha256_file(resolve(p))==h for p,h in contract['code'].items())
    best=min(results,key=lambda r:r['build_and_verify_seconds'])
    report=dict(status='measurement_complete',cases=results,fastest_measured_cache_mib=best['cache_mib'],
                scope=contract['scope'],code_sha256=contract['code'],
                case_sha256={str(r['cache_mib']):sha256_file(out/f"case_{r['cache_mib']}.json") for r in results})
    atomic_json(out/'report.json',report);atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
    atomic_json(out/'state.json',dict(phase='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True);p.add_argument('--output',required=True);p.add_argument('--scratch',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--cache',type=int)
    a=p.parse_args();worker(a) if a.worker else main(a)
