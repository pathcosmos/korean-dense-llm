"""Bounded comparison of indexed group-range distribution against source scans.
Only a completed independent fixture may be passed while full F4 is running.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file
from data.verified_split_context import VerifiedSplitContext,distribution as reference
from data.split_distribution import distribution


def main(a):
    from scripts.hardware_profile import inspect
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=False);hw=inspect()
    files=['scripts/benchmark_split_distribution.py','data/split_distribution.py','data/verified_split_context.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py']
    contract=dict(sources_sha256=sha256_file(resolve(a.sources)),split_sha256=sha256_file(resolve(a.split)),hardware=hw,
                  workers=a.workers,worker_cache_mib=64,parent_cache_mib=512,code={f:sha256_file(resolve(f)) for f in files},scope='Independent completed132k-row fixture only; full-corpus performance unproven.')
    atomic_json(out/'contract.json',contract)
    with VerifiedSplitContext(a.sources,a.split,hash_workers=16,cache_mib=512) as context:
        start=time.perf_counter();old=reference(context);oldtime=time.perf_counter()-start
        results=[]
        for workers in [1,a.workers]:
            r=distribution(context,workers=workers,worker_cache_mib=64)
            if r['counts']!=old['counts']:raise ValueError('Global counts differ')
            for name in old['sources']:
                if r['sources'][name]['splits']!=old['sources'][name]['splits']:raise ValueError('Source documents/groups differ')
                if r['sources'][name]['largest_components'][0]['documents']!=old['sources'][name]['largest_components'][0]['documents']:raise ValueError('Largest component differs')
            results.append(r)
    for f,h in contract['code'].items():
        if sha256_file(resolve(f))!=h:raise ValueError('Dependency changed')
    atomic_json(out/'report.json',dict(status='passed_bounded_distribution',reference_seconds=oldtime,results=results,scope=contract['scope'],production_adoption='pending'))
    atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
    print(json.dumps(dict(reference_seconds=oldtime,indexed_seconds=[r['elapsed_seconds'] for r in results],rows=results[0]['rows_scanned'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sources','split','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--workers',type=int,default=8);main(p.parse_args())
