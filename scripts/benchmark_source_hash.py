"""Bounded real-source SHA worker sweep; never drops the machine's page cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,sha256_file,atomic_json,identity


def main(a):
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=False)
    source=resolve(a.sources);inventory=json.loads(source.read_text())
    cert=json.loads(Path(str(source)+'.done').read_text())
    assert cert['sha256']==sha256_file(source)
    assert identity(inventory['content_inventory'])==cert['content_inventory_id']
    entries=list(inventory['content_inventory']['sources'].values())
    files=sorted((f for e in entries for f in e['files']),key=lambda f:f['bytes'],reverse=True)[:a.files]
    total=sum(f['bytes'] for f in files)
    contract=dict(source_sha256=cert['sha256'],files=files,order=a.workers,
                  code_sha256=sha256_file(Path(__file__)),
                  scope='Largest selected audited files, existing host cache, repeated worker configurations. Shared host with local SQLite sweep; not cold-storage or all-source throughput.')
    atomic_json(out/'contract.json',contract);results=[]
    def check(f):
        p=resolve(f['path']);actual=sha256_file(p)
        if p.stat().st_size!=f['bytes'] or actual!=f['sha256']:raise ValueError('Source content changed')
        return actual
    for i,workers in enumerate(contract['order']):
        atomic_json(out/'state.json',dict(phase='running',case=i,workers=workers))
        started=time.monotonic();cpu=time.process_time()
        with ThreadPoolExecutor(max_workers=workers) as pool:hashes=list(pool.map(check,files))
        seconds=time.monotonic()-started
        results.append(dict(workers=workers,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                            gib_per_second=total/2**30/seconds,all_hashes_verified=True,
                            hashes_identity=identity(hashes),cumulative_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        atomic_json(out/'partial_results.json',results)
    median={str(w):statistics.median(r['seconds'] for r in results if r['workers']==w) for w in sorted(set(a.workers))}
    report=dict(status='measurement_complete',selected_files=len(files),bytes_per_pass=total,cases=results,
                median_seconds=median,fastest_workers=int(min(median,key=median.get)),scope=contract['scope'])
    atomic_json(out/'report.json',report);atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
    atomic_json(out/'state.json',dict(phase='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',required=True);p.add_argument('--output',required=True)
    p.add_argument('--files',type=int,default=16);p.add_argument('--workers',type=int,nargs='+',default=[1,8,16,16,8,1])
    a=p.parse_args()
    if not 1<=a.files<=64 or any(not 1<=n<=32 for n in a.workers):p.error('Bound files to 1..64 and workers to 1..32')
    main(a)
