"""Extract certified, untruncated eval-split ko/en JSONL. No tokenizer/EOS added."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data.split_inputs import prepare
from common.data_contracts import resolve


@contextmanager
def local_verified_inputs(sources,split,reader,*,cache_mib=None,hash_workers=32):
    """Fresh physical validation; logical source/split identities stay canonical."""
    from data.local_split_reader import closed_manifest
    from data.verified_split_context import VerifiedSplitContext
    from data.split_inputs import SplitInputs
    stored=closed_manifest(reader)
    if cache_mib is None:
        size=resolve(stored['database']['path']).stat().st_size
        cache_mib=min(196608,max(8192,math.ceil(size/1024**2*1.2)))
    with VerifiedSplitContext(sources,split,reader_manifest=reader,cache_mib=cache_mib,hash_workers=hash_workers) as ctx:
        inputs=SplitInputs(sources,split,verified_context=ctx)
        try:yield inputs
        finally:inputs.close()


def prepare_holdout(sources,split,output,budgets,categories,*,seed=0,reader_manifest=None,cache_mib=None,hash_workers=32):
    if reader_manifest is None:
        if cache_mib is not None:raise ValueError('Reader cache option requires a reader manifest')
        return prepare(sources,split,output,budgets,mode='eval',seed=seed,categories=categories)
    with local_verified_inputs(sources,split,reader_manifest,cache_mib=cache_mib,hash_workers=hash_workers) as inputs:
        return prepare(sources,split,output,budgets,mode='eval',seed=seed,categories=categories,verified_inputs=inputs)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--sources-manifest',required=True)
    ap.add_argument('--split-manifest',required=True);ap.add_argument('--source',action='append',required=True,help='source:ko|en:UTF8_bytes')
    ap.add_argument('--output-dir',required=True);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--reader-manifest',help='Certified local DB reader; original split remains the logical input')
    ap.add_argument('--sqlite-cache-mib',type=int);ap.add_argument('--hash-workers',type=int,default=32);a=ap.parse_args()
    entries=[x.rsplit(':',2) for x in a.source]
    if len({e[0] for e in entries})!=len(entries):ap.error('Repeated source')
    r=prepare_holdout(a.sources_manifest,a.split_manifest,a.output_dir,{s:int(b) for s,c,b in entries},{s:c for s,c,b in entries},seed=a.seed,
                      reader_manifest=a.reader_manifest,cache_mib=a.sqlite_cache_mib,hash_workers=a.hash_workers)
    print(json.dumps(dict(status=r['status'],stats=r['stats'])))
