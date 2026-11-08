"""Code/web equality join using certified audit fingerprints, never raw corpus text.
Pilot mode is explicit and cannot certify full-source coverage. No data is deleted.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from common.data_contracts import resolve,identity,sha256_file
from common.source_inventory import load_audit
from data.source_audit import FINGERPRINT_SCHEMA
from data.probe_text_similarity import certificate,publish


def run(audit_path,output,*,allow_pilot=False,max_code_rows=3000000,max_seconds=1800,max_rss_gib=8):
    if type(max_code_rows) is not int or max_code_rows<1 or max_seconds<=0 or max_rss_gib<=0:raise ValueError('Invalid join limits')
    audit_path=resolve(audit_path);audit=load_audit(audit_path,require_full=not allow_pilot);files=[]
    for source,e in audit['sources'].items():
        code=audit['definition']['sources'][source]['text_policy']=='raw-utf8-code-v1'
        for f in e['files']:
            fp=f['fingerprints'];path=resolve(fp['path'])
            if sha256_file(path)!=fp['sha256']:raise ValueError('Audit fingerprint content changed')
            p=pq.ParquetFile(path)
            if p.metadata.num_rows!=f['rows'] or p.schema_arrow!=FINGERPRINT_SCHEMA:raise ValueError('Fingerprint schema/coverage differs')
            files.append(dict(source=source,code=code,input_file=f['path'],rows=f['rows'],fingerprint=fp))
    expected_code=sum(f['rows'] for f in files if f['code']);expected_web=sum(f['rows'] for f in files if not f['code'])
    if not expected_code or not expected_web:raise ValueError('Require audited code and web rows')
    if expected_code>max_code_rows:raise ValueError('Code index exceeds configured row budget')
    contract=dict(version=1,audit_sha256=sha256_file(audit_path),audit_scope=audit['scope'],files=files,
        allow_pilot=allow_pilot,max_code_rows=max_code_rows,max_seconds=max_seconds,max_rss_gib=max_rss_gib,
        match_policy='whole raw SHA256; code-file block key vs audited closed web fences, outer CR/LF only, min80',
        code={p:sha256_file(resolve(p)) for p in ['data/probe_audit_overlap.py','data/probe_text_similarity.py']})
    cached=certificate(output,contract)
    if cached is not None:return cached
    started=time.monotonic();raw=defaultdict(list);blocks=defaultdict(list);examples=[];limits=set()
    code_rows=web_rows=whole_pairs=whole_docs=fence_pairs=fence_matches=fence_total=fence_docs=0
    per_source={f['source']:dict(web_rows=0,whole_pairs=0,fence_pairs=0) for f in files if not f['code']}
    def check():
        if time.monotonic()-started>=max_seconds:return 'time'
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>max_rss_gib*1024**2:return 'rss'
    def rows(f):
        expected=0
        for batch in pq.ParquetFile(resolve(f['fingerprint']['path'])).iter_batches(batch_size=2048,use_threads=False):
            for row in batch.to_pylist():
                if row['row']!=expected or not isinstance(row['raw_sha256'],bytes) or len(row['raw_sha256'])!=32:
                    raise ValueError('Fingerprint row identity/order invalid')
                expected+=1;yield row
        if expected!=f['rows']:raise ValueError('Fingerprint row coverage invalid')
    for f in (f for f in files if f['code']):
        for row in rows(f):
            if code_rows%2048==0:
                limit=check()
                if limit:limits.add(limit);break
            ref=[f['source'],f['input_file'],row['row']];raw[row['raw_sha256']].append(ref)
            if row['code_file_block_sha256'] is not None:blocks[row['code_file_block_sha256']].append(ref)
            code_rows+=1
        if limits:break
    if not limits:
        for f in (f for f in files if not f['code']):
            for row in rows(f):
                if web_rows%2048==0:
                    limit=check()
                    if limit:limits.add(limit);break
                web_rows+=1;per_source[f['source']]['web_rows']+=1
                found=raw.get(row['raw_sha256'],[]);n=len(found);whole_pairs+=n;whole_docs+=int(n>0);per_source[f['source']]['whole_pairs']+=n
                if found and len(examples)<20:examples.append(dict(kind='whole_raw',web_file=f['input_file'],web_row=row['row'],code_ref=found[0]))
                matched=False
                for key in row['web_block_sha256']:
                    if not isinstance(key,bytes) or len(key)!=32:raise ValueError('Invalid web block digest')
                    fence_total+=1;found=blocks.get(key,[]);n=len(found);fence_pairs+=n;fence_matches+=int(n>0);matched|=n>0
                    per_source[f['source']]['fence_pairs']+=n
                    if found and len(examples)<20:examples.append(dict(kind='closed_fence',web_file=f['input_file'],web_row=row['row'],code_ref=found[0]))
                fence_docs+=int(matched)
            if limits:break
    complete=not limits and code_rows==expected_code and web_rows==expected_web
    report=dict(status='completed_on_audited_scope' if complete else 'partial',contract_id=identity(contract),contract=contract,
        full_source_coverage=complete and audit['scope']=='full',audit_scope=audit['scope'],
        expected_code_rows=expected_code,expected_web_rows=expected_web,indexed_code_rows=code_rows,processed_web_rows=web_rows,
        whole_matching_pairs=whole_pairs,whole_matched_web_documents=whole_docs,eligible_web_fences=fence_total,
        matched_web_fences=fence_matches,fence_matching_pairs=fence_pairs,web_documents_with_fence_match=fence_docs,
        per_web_source=per_source,examples=examples,limits_reached=sorted(limits),elapsed_seconds=time.monotonic()-started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Equality only within certified audit scope; no inline/partial/approximate code analysis, deletion, or automatic readiness pass')
    return publish(output,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--allow-pilot',action='store_true');ap.add_argument('--max-code-rows',type=int,default=3000000)
    ap.add_argument('--max-seconds',type=int,default=1800);a=ap.parse_args()
    r=run(a.audit,a.output,allow_pilot=a.allow_pilot,max_code_rows=a.max_code_rows,max_seconds=a.max_seconds)
    print(json.dumps({k:r[k] for k in ['status','audit_scope','full_source_coverage','whole_matching_pairs','eligible_web_fences','fence_matching_pairs']}))
