"""Native TEMP vs source-scan distribution on completed, bounded fixtures."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,atomic_json,sha256_file
from data.verified_split_context import VerifiedSplitContext,distribution as reference
from data.split_distribution_native import distribution


def main(a):
    from scripts.hardware_profile import inspect
    out=resolve(a.output);out.mkdir(parents=True,exist_ok=False);sm=resolve(a.sources);split=resolve(a.split)
    stored=json.loads(split.read_text());rows=sum(f['rows'] for f in stored['contract']['files'])
    if rows>3000000 or resolve(stored['database']['path']).stat().st_size>8*1024**3:raise ValueError('Bounded fixture only')
    hw=inspect()
    contract=dict(sources_sha256=sha256_file(sm),split_sha256=sha256_file(split),rows=rows,hardware=hw,parent_cache_mib=8192,local_reader=a.local_reader,
        code={f:sha256_file(resolve(f)) for f in ['scripts/benchmark_distribution_native.py','data/split_distribution_native.py','data/verified_split_context.py','data/local_split_reader.py','data/split_manifest.py','common/data_contracts.py','common/source_inventory.py']})
    atomic_json(out/'contract.json',contract);start=time.perf_counter()
    reader=None
    if a.local_reader:
        from data.local_split_reader import prepare_reader
        reader=prepare_reader(split,out/'reader',scratch=Path(hw['storage']['scratch_dir'])/out.name)
    copied=time.perf_counter()
    with VerifiedSplitContext(sm,split,hash_workers=16,cache_mib=8192,reader_manifest=reader) as ctx:
        verified=time.perf_counter();t=time.perf_counter();old=reference(ctx);seconds=time.perf_counter()-t;results=[]
        for n in [1,8]:
            r=distribution(ctx,sort_threads=n)
            if r['counts']!=old['counts']:raise ValueError('Global counts differ')
            for name in old['sources']:
                if r['sources'][name]['splits']!=old['sources'][name]['splits']:raise ValueError('Source group/document counts differ')
                if r['sources'][name]['largest_components'][0]['documents']!=old['sources'][name]['largest_components'][0]['documents']:raise ValueError('Largest component differs')
            results.append(r)
    for f,h in contract['code'].items():
        if sha256_file(resolve(f))!=h:raise ValueError('Dependency changed')
    report=dict(status='passed_bounded_distribution',rows=rows,copy_seconds=copied-start,initial_validation_seconds=verified-copied,reference_seconds=seconds,results=results,
                total_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Warm-cache completed fixture; full128M-row behavior not established.')
    atomic_json(out/'report.json',report);atomic_json(out/'.done',dict(report_sha256=sha256_file(out/'report.json'),contract_sha256=sha256_file(out/'contract.json')))
    print(json.dumps({k:v for k,v in report.items() if k!='results'}));print('native_seconds',[r['elapsed_seconds'] for r in results])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['sources','split','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--local-reader',action='store_true')
    main(p.parse_args())
