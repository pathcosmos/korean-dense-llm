"""Bounded cross-source web-view similarity diagnostics. No automatic data removal.
Candidates share at least two of the 16 lowest shingle hashes; actual Jaccard is
recomputed. Candidate generation can miss pairs; outputs are not corpus estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unicodedata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,identity,sha256_file,atomic_json
from data.sample_snapshot import load_samples


def certificate(output,contract):
    output=resolve(output);cid=identity(contract)
    if output.exists():
        old=json.loads(output.read_text());marker=Path(str(output)+'.done')
        if old.get('contract_id')!=cid:raise ValueError('Probe inputs/config changed; use a new report path')
        if not marker.exists() or json.loads(marker.read_text())!={'sha256':sha256_file(output),'contract_id':cid}:raise ValueError('Uncertified probe report')
        return old


def publish(output,report):
    output=resolve(output);atomic_json(output,report)
    atomic_json(Path(str(output)+'.done'),dict(sha256=sha256_file(output),contract_id=report['contract_id']))
    return report


def shingles(text,n=5):
    text=' '.join(unicodedata.normalize('NFC',text).split())
    return {hashlib.sha256(text[i:i+n].encode()).digest()[:8] for i in range(max(0,len(text)-n+1))}


def run(samples_manifest,output,*,threshold=.8,max_candidates=10000,max_seconds=120,max_view_chars=2048,_candidate_observer=None):
    import math
    if not math.isfinite(threshold) or not 0<threshold<=1 or any(type(v) is not int or v<1 for v in [max_candidates,max_seconds,max_view_chars]):raise ValueError('Invalid probe limits')
    snapshot,records=load_samples(samples_manifest);web=[r for r in records if r['kind']=='web']
    if len({r['source'] for r in web})<2:raise ValueError('Require two web sources')
    contract=dict(version=1,samples_manifest_sha256=sha256_file(resolve(samples_manifest)),threshold=threshold,
        max_candidates=max_candidates,max_seconds=max_seconds,max_view_chars=max_view_chars,n=5,signature=16,min_shared=2,
        normalization='web-view NFC and collapsed whitespace; case preserved',code_sha256=sha256_file(Path(__file__)))
    cached=certificate(output,contract)
    if cached is not None:return cached
    started=time.monotonic();index={};views=[];matches=[];examples=[];verified=0;limits=set();pairs={};empty=0
    for i,row in enumerate(web):
        if time.monotonic()-started>max_seconds:limits.add('time');break
        view=shingles(row['text'][:max_view_chars]);views.append(view)
        if not view:empty+=1;continue
        signature=sorted(view)[:16];votes={}
        for h in signature:
            for j in index.get(h,[]):
                if web[j]['source']!=row['source']:votes[j]=votes.get(j,0)+1
                if time.monotonic()-started>max_seconds:limits.add('time');break
            if limits:break
        if limits:break
        for j,count in sorted(votes.items()):
            if count<2:continue
            if verified>=max_candidates:limits.add('candidate_pairs');break
            verified+=1;other=views[j];similarity=len(view&other)/len(view|other)
            if _candidate_observer is not None:_candidate_observer(i,j,similarity)
            key=' | '.join(sorted([row['source'],web[j]['source']]))
            if key not in pairs:pairs[key]=dict(verified_pairs=0,matched_pairs=0)
            pairs[key]['verified_pairs']+=1
            if similarity>=threshold:
                matches.append((i,j));pairs[key]['matched_pairs']+=1
                if len(examples)<20:examples.append(dict(left=dict(file=web[j]['file'],row=web[j]['row']),right=dict(file=row['file'],row=row['row']),
                    view_jaccard=similarity,whole_document_raw_exact=row['content_sha256']==web[j]['content_sha256']))
        if limits:break
        for h in signature:index.setdefault(h,[]).append(i)
    report=dict(status='partial' if limits else 'completed_on_supplied_sample',contract_id=identity(contract),contract=contract,
        sampling_status=snapshot['sample_status'],sampled_web_documents=len(web),processed_web_documents=len(views),
        verified_candidate_pairs=verified,matched_view_pairs=len(matches),matched_sample_documents=len({x for p in matches for x in p}),
        empty_or_short_views=empty,source_pairs=pairs,examples=examples,limits_reached=sorted(limits),elapsed_seconds=time.monotonic()-started,
        scope='Only bounded normalized views and generated candidates; candidate recall unmeasured, no full-corpus duplicate rate or deletion decision')
    return publish(output,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples-manifest',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--threshold',type=float,default=.8);a=ap.parse_args()
    print(json.dumps(run(a.samples_manifest,a.output,threshold=a.threshold)))
