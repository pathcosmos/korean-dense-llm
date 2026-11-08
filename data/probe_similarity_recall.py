"""Exhaustive bounded reference vs the production web candidate generator.
Real sample and synthetic controls are reported separately. No-positive recall is
undefined (null), never 100%. Only stored samples are read, not raw source files.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import string
import sys
import time
import unicodedata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data_contracts import resolve,artifact_path,identity,sha256_file,atomic_json
from data.sample_snapshot import load_samples
from data.probe_text_similarity import run as production,certificate,publish


def reference_grams(text):
    text=' '.join(unicodedata.normalize('NFC',text).split())
    return {text[i:i+5] for i in range(max(0,len(text)-4))}


def controls(max_chars,seed=42):
    rng=random.Random(seed);result=[]
    for number in range(8):
        base=''.join(rng.choice(string.ascii_letters+string.digits+' 가나다라') for _ in range(max_chars-64))+' e\u0301'
        changed=list(base)
        for i in range(20,len(changed),80):changed[i]=rng.choice(string.ascii_lowercase)
        texts=[base,unicodedata.normalize('NFC',base),' '.join(base.split())+' tail suffix',''.join(changed),base.upper()]
        for variant,text in enumerate(texts):
            digest=hashlib.sha256(text.encode()).hexdigest()
            result.append(dict(source='control_left' if variant==0 else 'control_right',kind='web',
                file=f'fixture://similarity-controls/{number}/{variant}',row=0,text=text,content_sha256=digest,view_sha256=digest,
                truncated=False,characters_total=len(text),repo=None,study_origin='synthetic'))
    return result


def metrics(truth,candidates,scored):
    return dict(positive_pairs=len(truth),generated_candidate_pairs=len(candidates),
        candidate_true_positive_pairs=len(truth&candidates),scored_true_positive_pairs=len(truth&scored),
        missed_candidate_pairs=len(truth-candidates),false_positive_scored_pairs=len(scored-truth),
        candidate_recall=len(truth&candidates)/len(truth) if truth else None,
        scored_recall=len(truth&scored)/len(truth) if truth else None)


def run(samples_manifest,output_dir,*,max_real_per_source=32,max_view_chars=1024,max_candidates=20000,max_seconds=120,seed=42):
    if type(max_real_per_source) is not int or not 0<=max_real_per_source<=64 or not 128<=max_view_chars<=2048 or max_candidates<1 or max_seconds<1:
        raise ValueError('Invalid bounded recall-study limits')
    out=resolve(output_dir);snapshot,records=load_samples(samples_manifest)
    contract=dict(version=1,input_sha256=sha256_file(resolve(samples_manifest)),max_real_per_source=max_real_per_source,
        max_view_chars=max_view_chars,max_candidates=max_candidates,max_seconds=max_seconds,seed=seed,thresholds=[.8,.9,.95],
        controls='Eight seeded unrelated bases, four cross-source variants per base; separate real/synthetic metrics',
        reference='Unhashed NFC+whitespace-normalized 5-character string sets; complete cross-source pairs',
        code={p:sha256_file(resolve(p)) for p in ['data/probe_similarity_recall.py','data/probe_text_similarity.py']})
    summary=out/'summary.json';cached=certificate(summary,contract)
    if cached is not None:return cached
    if out.exists() and any(out.iterdir()):raise ValueError('Partial recall study exists; preserve and use a new directory')
    out.mkdir(parents=True,exist_ok=True);selected=[];counts={}
    for row in records:
        if row['kind']!='web' or counts.get(row['source'],0)>=max_real_per_source:continue
        counts[row['source']]=counts.get(row['source'],0)+1
        selected.append(dict(row,study_origin='real_sample'))
    selected+=controls(max_view_chars,seed)
    study=out/'study';study.mkdir();sample_file=study/'samples.jsonl'
    sample_file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    study_contract=dict(parent_contract=identity(contract),fixture_scope='Real stored rows plus explicitly synthetic controls')
    stat={}
    for row in selected:stat.setdefault(row['source'],dict(documents=0))['documents']+=1
    study_report=dict(run_status='completed',sample_status='bounded_recall_study',contract=study_contract,contract_id=identity(study_contract),
        stats=stat,samples=dict(path=artifact_path(sample_file),sha256=sha256_file(sample_file)))
    atomic_json(study/'manifest.json',study_report);atomic_json(study/'.done',dict(manifest_sha256=sha256_file(study/'manifest.json')))
    observed={}
    def observer(i,j,score):observed[tuple(sorted((i,j)))]=score
    result=production(study/'manifest.json',out/'production_candidates.json',max_candidates=max_candidates,max_seconds=max_seconds,
                      max_view_chars=max_view_chars,_candidate_observer=observer)
    started=time.monotonic();grams=[reference_grams(r['text'][:max_view_chars]) for r in selected]
    truth={t:set() for t in [.8,.9,.95]};all_pairs=set();origins={};reference_complete=True
    for i,left in enumerate(selected):
        for j in range(i):
            right=selected[j]
            if left['source']==right['source']:continue
            if time.monotonic()-started>max_seconds:reference_complete=False;break
            key=(j,i);all_pairs.add(key)
            origins[key]='real_only' if left['study_origin']==right['study_origin']=='real_sample' else 'synthetic_only' if left['study_origin']==right['study_origin']=='synthetic' else 'mixed'
            union=grams[i]|grams[j]
            score=len(grams[i]&grams[j])/len(union) if union else 0.
            for t in truth:
                if score>=t:truth[t].add(key)
        if not reference_complete:break
    thresholds={}
    for t,positives in truth.items():
        scored={p for p,s in observed.items() if s>=t};entry={}
        for group in ['all','real_only','synthetic_only','mixed']:
            frame=all_pairs if group=='all' else {p for p in all_pairs if origins[p]==group}
            entry[group]=metrics(positives&frame,set(observed)&frame,scored&frame)
        thresholds[str(t)]=entry
    report=dict(status='bounded_reference_complete' if reference_complete else 'partial_reference',contract=contract,contract_id=identity(contract),
        source_sample_status=snapshot['sample_status'],real_documents=sum(counts.values()),synthetic_documents=len(selected)-sum(counts.values()),
        cross_source_pairs=len(all_pairs),production_status=result['status'],production_candidate_pairs=len(observed),thresholds=thresholds,
        reference_complete=reference_complete,reference_seconds=time.monotonic()-started,
        scope='Only this selected stored frame and synthetic controls; no full-corpus recall/precision guarantee or policy promotion')
    return publish(summary,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples-manifest',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--max-real-per-source',type=int,default=32);ap.add_argument('--max-view-chars',type=int,default=1024)
    ap.add_argument('--max-candidates',type=int,default=20000);a=ap.parse_args()
    r=run(a.samples_manifest,a.output_dir,max_real_per_source=a.max_real_per_source,max_view_chars=a.max_view_chars,max_candidates=a.max_candidates)
    print(json.dumps(dict(status=r['status'],real_documents=r['real_documents'],synthetic_documents=r['synthetic_documents'],thresholds=r['thresholds'])))
