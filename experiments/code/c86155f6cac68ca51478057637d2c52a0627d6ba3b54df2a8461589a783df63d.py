"""Bounded raw whole-file and fenced-block code/web equality diagnostics.
Full-source hashes are not claimed. Sampled original-body hashes distinguish whole
files from truncated views. Fences are searched only within stored web views.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,identity,sha256_file
from data.sample_snapshot import load_samples
from data.probe_code_math_overlap import fenced_blocks,block_key
from data.probe_text_similarity import certificate,publish


def run(samples_manifest,output,*,min_chars=80,max_blocks=100000,max_seconds=120):
    if any(type(v) is not int or v<1 for v in [min_chars,max_blocks,max_seconds]):raise ValueError('Invalid code-web limits')
    snapshot,records=load_samples(samples_manifest);code=[r for r in records if r['kind']=='code'];web=[r for r in records if r['kind']=='web']
    if not code or not web:raise ValueError('Require code and web samples')
    contract=dict(version=1,samples_manifest_sha256=sha256_file(resolve(samples_manifest)),min_chars=min_chars,max_blocks=max_blocks,
        max_seconds=max_seconds,normalization='whole raw UTF8; block comparison strips only outer CR/LF',
        code={p:sha256_file(resolve(p)) for p in ['data/probe_code_web_overlap.py','data/probe_code_math_overlap.py']})
    cached=certificate(output,contract)
    if cached is not None:return cached
    raw=defaultdict(list);block=defaultdict(list)
    for row in code:
        raw[row['content_sha256']].append(row)
        # A truncated code view cannot establish the CR/LF-stripped whole-file hash.
        if not row['truncated'] and len(row['text'].strip('\r\n'))>=min_chars:block[block_key(row['text'])].append(row)
    start=time.monotonic();whole_pairs=blocks=eligible=block_pairs=0;examples=[];limits=set();processed=0
    for row in web:
        if time.monotonic()-start>max_seconds:limits.add('time');break
        whole_pairs+=len(raw.get(row['content_sha256'],[]));processed+=1
        for body in fenced_blocks(row['text']):
            if blocks>=max_blocks:limits.add('blocks');break
            blocks+=1
            if len(body.strip('\r\n'))<min_chars:continue
            eligible+=1;found=block.get(block_key(body),[]);block_pairs+=len(found)
            if found and len(examples)<20:examples.append(dict(web_file=row['file'],web_row=row['row'],code_file=found[0]['file'],code_row=found[0]['row']))
        if limits:break
    report=dict(status='partial' if limits else 'completed_on_supplied_sample',contract=contract,contract_id=identity(contract),sampling_status=snapshot['sample_status'],
        sampled_code_files=len(code),sampled_web_documents=len(web),processed_web_documents=processed,
        whole_document_exact_pairs=whole_pairs,fenced_blocks=blocks,eligible_fenced_blocks=eligible,fenced_whole_code_pairs=block_pairs,
        truncated_code_views_excluded_from_block_index=sum(r['truncated'] for r in code),examples=examples,limits_reached=sorted(limits),
        elapsed_seconds=time.monotonic()-start,
        scope='Selected code/web sample only; no inline, arbitrary snippets, identifier changes or approximate code similarity; no automatic removal')
    return publish(output,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples-manifest',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    print(json.dumps(run(a.samples_manifest,a.output)))
