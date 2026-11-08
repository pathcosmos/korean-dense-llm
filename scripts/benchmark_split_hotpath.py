"""Isolated hot-path candidate, same 3M inputs and 8GiB cache as the sweep."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file
from data.split_manifest_candidate import build,SplitSelection


def main(a):
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=False)
    local=resolve(a.scratch);local.mkdir(parents=True,exist_ok=False)
    files=['data/split_manifest_candidate.py','scripts/benchmark_split_hotpath.py','common/data_contracts.py','common/source_inventory.py']
    contract=dict(code={p:sha256_file(resolve(p)) for p in files},sources_sha256=sha256_file(resolve(a.sources)),cache_mib=8192,
                  scope='Avoid repeated identical NFC union/find only; raw/code grouping unchanged. Shared host comparison to cache sweep, not production adoption.')
    atomic_json(out/'contract.json',contract);start=time.monotonic();last=0;phases={}
    def progress(phase):
        nonlocal last
        phases.setdefault(phase,time.monotonic()-start)
        if time.monotonic()-last>2:
            atomic_json(out/'state.json',dict(phase='running',worker_phase=phase,elapsed_seconds=time.monotonic()-start))
            last=time.monotonic()
    r=build(a.sources,local/'split',.1,7,sqlite_cache_mib=8192,_checkpoint=progress)
    built=time.monotonic();s=SplitSelection(local/'split/manifest.json',sqlite_cache_mib=8192);verified=time.monotonic()
    h=hashlib.sha256();rows=0
    try:
        for row in s.db.execute('SELECT source,file,row,content_sha256,repo,group_id,split FROM docs ORDER BY file,row'):
            h.update(json.dumps(row,ensure_ascii=False,separators=(',',':')).encode());h.update(b'\n');rows+=1
    finally:s.close()
    assert all(sha256_file(resolve(p))==h for p,h in contract['code'].items())
    result=dict(status='measurement_complete',build_seconds=built-start,verify_seconds=verified-built,
                build_and_verify_seconds=verified-start,counts=r['counts'],logical_rows=rows,logical_sha256=h.hexdigest(),
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,phase_first_seconds=phases,
                manifest_sha256=sha256_file(local/'split/manifest.json'),scope=contract['scope'])
    atomic_json(out/'report.json',result);atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
    atomic_json(out/'state.json',dict(phase='complete'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',required=True);p.add_argument('--output',required=True);p.add_argument('--scratch',required=True)
    main(p.parse_args())
