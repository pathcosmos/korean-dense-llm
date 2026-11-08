"""Content-verified train/eval row selection and bounded, certified JSONL inputs."""
import hashlib
import json
from pathlib import Path
import random
import pyarrow.parquet as pq
from common.data_contracts import source_files,resolve,artifact_path,identity,sha256_file,atomic_json
from data.split_manifest import SplitSelection,text_hash


class SplitInputs:
    def __init__(self,source_manifest,split_manifest):
        self.manifest=resolve(source_manifest);self.inventory=json.loads(self.manifest.read_text())
        self.selection=SplitSelection(split_manifest);self.files={}
        try:
            for source in self.inventory['sources']:
                paths,snapshot=source_files(source,self.manifest)
                records=[dict(path=artifact_path(p),bytes=p.stat().st_size,sha256=sha256_file(p)) for p in paths]
                self.selection.verify_source(source,records,snapshot);self.files[source]=paths
        except BaseException:self.close();raise
    def close(self):self.selection.close()
    def rows(self,source,split,seed=0):
        if split not in ('train','eval') or source not in self.files or type(seed) is not int or seed<0:
            raise ValueError('Invalid split/source/seed')
        rng=random.Random(identity([seed,source]));paths=list(self.files[source]);rng.shuffle(paths)
        for file in paths:
            p=pq.ParquetFile(file);groups=list(range(p.num_row_groups));rng.shuffle(groups)
            offsets=[];offset=0
            for g in range(p.num_row_groups):offsets.append(offset);offset+=p.metadata.row_group(g).num_rows
            for g in groups:
                start=offsets[g];end=start+p.metadata.row_group(g).num_rows
                cursor=self.selection.db.execute('''SELECT row,content_sha256,group_id,repo FROM docs
                    WHERE source=? AND file=? AND split=? AND row>=? AND row<? ORDER BY row''',
                    (source,artifact_path(file),split,start,end))
                selected=cursor.fetchone();row=start
                if selected is None:continue
                for batch in p.iter_batches(batch_size=1024,row_groups=[g],columns=['text']):
                    for text in batch.column(0).to_pylist():
                        if selected is not None and row==selected[0]:
                            if not isinstance(text,str) or text_hash(text)!=selected[1]:raise ValueError('Split/source body hash mismatch')
                            yield dict(source=source,file=artifact_path(file),row=row,text=text,content_sha256=selected[1],group_id=selected[2],repo=selected[3])
                            selected=cursor.fetchone()
                        row+=1
                    if selected is None:break
                if selected is not None:raise ValueError('Split row coverage mismatch')


def prepare(source_manifest,split_manifest,output_dir,budgets,*,mode,seed=0,max_chars=None,categories=None):
    if mode not in ('train','eval') or type(seed) is not int or seed<0 or not budgets or any(type(v) is not int or v<1 for v in budgets.values()):
        raise ValueError('Require positive byte budgets and valid mode/seed')
    if max_chars is not None and (type(max_chars) is not int or max_chars<1):raise ValueError('Invalid character limit')
    if mode=='eval' and (max_chars is not None or categories is None or set(categories)!=set(budgets) or set(categories.values())!={'ko','en'}):
        raise ValueError('Eval requires untruncated ko/en source mapping')
    out=resolve(output_dir);report_path=out/'manifest.json'
    inputs=SplitInputs(source_manifest,split_manifest)
    try:
        if set(budgets)-set(inputs.files):raise ValueError('Unknown budget source')
        contract=dict(version=1,source_manifest=artifact_path(inputs.manifest),source_manifest_sha256=sha256_file(inputs.manifest),
            source_snapshot_id=identity(inputs.inventory),split_manifest=artifact_path(resolve(split_manifest)),
            split_manifest_sha256=sha256_file(resolve(split_manifest)),split_contract_id=inputs.selection.report['contract_id'],
            mode=mode,seed=seed,budgets=budgets,max_chars=max_chars,categories=categories,
            budget_rule='whole-document stop after reaching per-source UTF8 budget; up to one document overshoot',
            selection_frame='seeded file/row-group order, selected split rows in source row order; not uniform corpus sampling',
            code_sha256=sha256_file(Path(__file__)))
        cid=identity(contract)
        if report_path.exists():
            previous=json.loads(report_path.read_text())
            if previous.get('contract_id')!=cid:raise ValueError('Prepared input contract changed; use a new directory')
            _certificate(report_path,mode)
            return previous
        if out.exists() and any(out.iterdir()):raise ValueError('Partial prepared inputs exist; preserve and use a new directory')
        out.mkdir(parents=True,exist_ok=True);path=out/'samples.jsonl';tmp=out/'samples.jsonl.tmp';stats={}
        with tmp.open('w') as stream:
            for source,budget in budgets.items():
                count=used=0
                iterator=inputs.rows(source,mode,seed)
                try:
                    for r in iterator:
                        text=r['text'] if max_chars is None else r['text'][:max_chars]
                        if not text:continue
                        r.update(text=text,text_sha256=text_hash(text),truncated=len(text)!=len(r['text']),split=mode)
                        if categories is not None:r['category']=categories[source]
                        stream.write(json.dumps(r,ensure_ascii=False)+'\n');used+=len(text.encode());count+=1
                        if used>=budget:break
                finally:iterator.close()
                if used<budget:raise ValueError(f'Insufficient {mode} sample bytes for {source}: {used} < {budget}')
                stats[source]=dict(documents=count,utf8_bytes=used,target_bytes=budget,overshoot_bytes=used-budget)
        tmp.replace(path)
        report=dict(status='prepared',contract_id=cid,contract=contract,stats=stats,
                    samples=dict(path=artifact_path(path),sha256=sha256_file(path)))
        atomic_json(report_path,report);atomic_json(out/'.done',dict(contract_id=cid,manifest_sha256=sha256_file(report_path)))
        return report
    finally:inputs.close()


def _certificate(manifest,mode):
    manifest=resolve(manifest);r=json.loads(manifest.read_text());marker=manifest.parent/'.done'
    if r.get('status')!='prepared' or r.get('contract_id')!=identity(r['contract']) or r['contract']['mode']!=mode:
        raise ValueError('Invalid prepared input contract')
    if not marker.exists() or json.loads(marker.read_text())!=dict(contract_id=r['contract_id'],manifest_sha256=sha256_file(manifest)):
        raise ValueError('Uncertified prepared inputs')
    if sha256_file(resolve(r['samples']['path']))!=r['samples']['sha256']:raise ValueError('Prepared samples content changed')
    return r


def verify_prepared(manifest,mode='eval',samples_path=None):
    r=_certificate(manifest,mode);c=r['contract'];source=resolve(c['source_manifest']);split=resolve(c['split_manifest'])
    if sha256_file(source)!=c['source_manifest_sha256'] or sha256_file(split)!=c['split_manifest_sha256']:
        raise ValueError('Prepared source/split input changed')
    if samples_path is not None and resolve(samples_path).resolve()!=resolve(r['samples']['path']).resolve():raise ValueError('Prepared samples path mismatch')
    inputs=SplitInputs(source,split)
    try:
        seen=set();stats={s:dict(documents=0,utf8_bytes=0) for s in c['budgets']};found=set()
        with resolve(r['samples']['path']).open() as stream:
            for line in stream:
                item=json.loads(line);source_name=item['source'];key=(item['file'],item['row'])
                if source_name not in stats or key in seen or type(item['row']) is not int or item['row']<0:raise ValueError('Invalid/duplicate prepared row')
                seen.add(key)
                stored=inputs.selection.db.execute('SELECT split,content_sha256,group_id,repo FROM docs WHERE source=? AND file=? AND row=?',
                                                   (source_name,*key)).fetchone()
                if stored!=(mode,item['content_sha256'],item['group_id'],item['repo']) or item.get('split')!=mode:
                    raise ValueError('Prepared row not in requested split')
                if not item['text'] or text_hash(item['text'])!=item['text_sha256']:raise ValueError('Prepared text hash mismatch')
                if mode=='eval':
                    if item.get('truncated') or item['text_sha256']!=item['content_sha256'] or item.get('category')!=c['categories'][source_name]:
                        raise ValueError('Eval text/category must match complete source document')
                    found.add(item['category'])
                stats[source_name]['documents']+=1;stats[source_name]['utf8_bytes']+=len(item['text'].encode())
        if mode=='eval' and found!={'ko','en'}:raise ValueError('Both eval languages required')
        for s,v in stats.items():
            if any(v[k]!=r['stats'][s][k] for k in v) or v['utf8_bytes']<c['budgets'][s]:raise ValueError('Prepared coverage/stats mismatch')
        return r
    finally:inputs.close()
