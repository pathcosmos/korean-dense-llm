"""Bounded experimental BPE comparison; never a final tokenizer or ready dataset.
Each --source is CATEGORY:PARQUET. Reads at most 256 rows/file, clips at 4096 chars.
First 96 distinct samples/category train, next 32 evaluate; fixed deterministic order.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from tokenizers import trainers,pre_tokenizers
from tokenizer.train_tokenizer import make_tokenizer,SPECIAL
from tokenizer.check_tokenizer import check,CATEGORIES
from common.data_contracts import resolve,atomic_json,sha256_file,identity,artifact_path


def run(sources,output,vocab=2048):
    out=resolve(output)
    if out.exists():raise ValueError('Use a new experiment directory; preserve previous evidence')
    if set(sources)!=set(CATEGORIES):raise ValueError('Require exactly ko/en/code/math sources')
    train=[];held=[];seen=set();provenance=[]
    for category in CATEGORIES:
        path=resolve(sources[category]);stat=path.stat();rows=[]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['text']):
            rows.extend(batch.column(0).to_pylist())
            if len(rows)>=256:break
        selected=[]
        for row,text in enumerate(rows[:256]):
            if not isinstance(text,str):raise ValueError('Non-string sample')
            text=text[:4096];sha=hashlib.sha256(text.encode()).hexdigest()
            if not text or sha in seen:continue
            seen.add(sha);selected.append(dict(category=category,text=text,source=artifact_path(path),row=row,text_sha256=sha))
            if len(selected)==128:break
        if len(selected)!=128:raise ValueError('Insufficient distinct bounded samples')
        if (path.stat().st_size,path.stat().st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):
            raise ValueError('Source changed during bounded sampling')
        train.extend(selected[:96]);held.extend(selected[96:])
        provenance.append(dict(category=category,path=artifact_path(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                               sampled_rows=[r['row'] for r in selected],source_full_hash=False))
    out.mkdir(parents=True)
    for name,items in [('train',train),('heldout',held)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    reports={}
    for policy in ['NFC','raw']:
        tok=make_tokenizer(policy)
        tok.train_from_iterator([r['text'] for r in train],trainers.BpeTrainer(vocab_size=vocab,
            special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=2,show_progress=False))
        path=out/(policy+'.json');tok.save(str(path))
        reports[policy]=check(path,out/'heldout.jsonl',out/(policy+'.report.json'),tok.get_vocab_size(),policy)
    comparison={c:dict(raw_tokens=reports['raw']['categories'][c]['tokens'],nfc_tokens=reports['NFC']['categories'][c]['tokens'],
        raw_token_increase_pct=100*(reports['raw']['categories'][c]['tokens']/reports['NFC']['categories'][c]['tokens']-1)) for c in CATEGORIES}
    report=dict(status='experiment_complete',scope='Small fixed prefix sample; not representative quality or final tokenizer acceptance',
                train_documents=len(train),heldout_documents=len(held),vocab=vocab,source_provenance=provenance,
                train_sha256=sha256_file(out/'train.jsonl'),heldout_sha256=sha256_file(out/'heldout.jsonl'),
                reports={p:r['contract_id'] for p,r in reports.items()},comparison=comparison,
                statuses={p:r['status'] for p,r in reports.items()},code_changes={p:r['code_raw_change_count'] for p,r in reports.items()})
    atomic_json(out/'summary.json',report)
    atomic_json(out/'.done',dict(summary_sha256=sha256_file(out/'summary.json')))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',action='append',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--vocab',type=int,default=2048);a=ap.parse_args()
    entries=[s.split(':',1) for s in a.source]
    if len(dict(entries))!=len(entries):ap.error('Duplicate categories')
    print(json.dumps(run(dict(entries),a.output,a.vocab),ensure_ascii=False))
