"""Create an immutable SQLite document/group split from a ready source inventory.
Raw exact-content groups span sources; NFC groups additionally connect web documents.
Code repository groups connect duplicate files across repositories. No near-dedup claim.
New builds resume committed batches only with an identical content/configuration contract.
Legacy partial databases remain untouched and require a new output path.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from common.data_contracts import source_files,resolve,artifact_path,identity,sha256_file,atomic_json


def text_hash(text):return hashlib.sha256(text.encode('utf8')).hexdigest()


def validate_db(db):
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Split DB integrity failure')
    if db.execute("SELECT 1 FROM docs WHERE split NOT IN ('train','eval') OR split IS NULL LIMIT 1").fetchone():
        raise ValueError('Unknown or missing split')
    for field in ['content_sha256','group_id']:
        if db.execute(f'SELECT 1 FROM docs GROUP BY {field} HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError(f'Train/eval leakage by {field}')
    if db.execute("SELECT 1 FROM docs WHERE repo IS NOT NULL GROUP BY repo HAVING COUNT(DISTINCT split)>1 LIMIT 1").fetchone():
        raise ValueError('Train/eval repository leakage')
    return {s:n for s,n in db.execute('SELECT split,COUNT(*) FROM docs GROUP BY split')}


MAX_CACHE_MIB=262144


def build(source_manifest,output_dir,eval_fraction=.01,seed=1234,*,_checkpoint=None,sqlite_cache_mib=64,hash_workers=1):
    """Resume only our version-2 builder with identical content/configuration.
    Each ingest/assignment batch commits progress and data in one SQLite transaction.
    _checkpoint is a fault-injection hook used only by bounded recovery tests.
    """
    if not math.isfinite(eval_fraction) or not 0<eval_fraction<1 or type(seed) is not int or seed<0:
        raise ValueError('Require 0 < eval_fraction < 1 and nonnegative integer seed')
    if type(sqlite_cache_mib) is not int or not 1<=sqlite_cache_mib<=MAX_CACHE_MIB:raise ValueError('SQLite cache must be 1..262144 MiB')
    if type(hash_workers) is not int or not 1<=hash_workers<=32:raise ValueError('Hash workers must be 1..32')
    source_manifest=resolve(source_manifest);out=resolve(output_dir)
    inventory=json.loads(source_manifest.read_text());files=[]
    for source,entry in inventory['sources'].items():
        if _checkpoint:_checkpoint('verify_source:'+source)
        paths,snapshot,bound=source_files(source,source_manifest,return_verified_records=True,verification_workers=hash_workers)
        if entry['text_policy'] not in ('raw-utf8-code-v1','nfc-web-exact-v1'):
            raise ValueError('Unsupported text policy')
        # Bound records were just verified against actual bytes in this call. Reuse
        # those SHA values, not an old successful audit flag or a metadata-only cache.
        verified={f['path']:f for f in bound} if bound else {}
        def record(p):
            path=artifact_path(p);prior=verified.get(path)
            rows=pq.ParquetFile(p).metadata.num_rows;size=p.stat().st_size
            if prior and (prior['bytes']!=size or prior['rows']!=rows):raise ValueError('Verified source metadata changed')
            return dict(source=source,path=path,bytes=size,sha256=prior['sha256'] if prior else sha256_file(p),
                        rows=rows,text_policy=entry['text_policy'])
        with ThreadPoolExecutor(max_workers=hash_workers) as pool:files.extend(pool.map(record,paths))
    if len({str(resolve(f['path']).resolve()) for f in files})!=len(files):raise ValueError('Duplicate input file')
    contract=dict(version=1,source_snapshot=identity(inventory),files=files,seed=seed,
                  eval_fraction=eval_fraction,group_policy='raw-exact+web-nfc+code-repository-components-v1')
    cid=identity(contract);report_path=out/'manifest.json';final=out/'split.sqlite'
    if report_path.exists() and (out/'.done').exists():
        selector=SplitSelection(report_path)
        try:
            if selector.report['contract_id']!=cid:raise ValueError('Split inputs/policy changed; use a new output directory')
            return selector.report
        finally:selector.close()
    expected_builder=dict(builder_version=2,contract_id=cid,contract=contract)
    builder=out/'build.json'
    if builder.exists():
        if json.loads(builder.read_text())!=expected_builder:
            raise ValueError('Partial split inputs/policy changed; use a new output directory')
    else:
        if out.exists() and any(out.iterdir()):raise ValueError('Unrecognized partial split; preserve it and use a new output directory')
        out.mkdir(parents=True,exist_ok=True);atomic_json(builder,expected_builder)
    database=out/'split.sqlite.building'
    if database.exists() and final.exists():raise ValueError('Ambiguous split databases')
    published=final.exists();db=sqlite3.connect(final if published else database)
    db.execute(f'PRAGMA cache_size={-sqlite_cache_mib*1024}')
    def notify(phase):
        if _checkpoint is not None:_checkpoint(phase)
    def get(key):return json.loads(db.execute('SELECT value FROM progress WHERE key=?',(key,)).fetchone()[0])
    def put(key,value):db.execute('INSERT OR REPLACE INTO progress VALUES (?,?)',(key,json.dumps(value)))
    def find(key):
        db.execute('INSERT OR IGNORE INTO nodes VALUES (?,?)',(key,key));trail=[];node=key
        while True:
            parent=db.execute('SELECT parent FROM nodes WHERE key=?',(node,)).fetchone()[0]
            if parent==node:break
            trail.append(node);node=parent
        db.executemany('UPDATE nodes SET parent=? WHERE key=?',[(node,k) for k in trail])
        return node
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:db.execute('UPDATE nodes SET parent=? WHERE key=?',(min(a,b),max(a,b)))
    try:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables and not published:
            db.executescript('''BEGIN IMMEDIATE;
                CREATE TABLE nodes (key TEXT PRIMARY KEY,parent TEXT NOT NULL);
                CREATE TABLE docs (source TEXT NOT NULL,file TEXT NOT NULL,row INTEGER NOT NULL,
                  content_sha256 TEXT NOT NULL,repo TEXT,group_id TEXT NOT NULL,split TEXT,
                  PRIMARY KEY(file,row));
                CREATE INDEX doc_group ON docs(group_id);
                CREATE TABLE assignments (original TEXT PRIMARY KEY,root TEXT,split TEXT);
                CREATE TABLE progress (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            ''')
            for k,v in dict(contract_id=cid,builder_version=2,phase='ingest',file_index=0,row=0,last='').items():put(k,v)
            db.commit()
        if get('contract_id')!=cid or get('builder_version')!=2:
            raise ValueError('Partial DB contract mismatch')
        if published and get('phase')!='complete':raise ValueError('Published DB is incomplete')
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Partial DB integrity failure')
        if get('phase')=='ingest':
            for index in range(get('file_index'),len(files)):
                f=files[index];p=pq.ParquetFile(resolve(f['path']));code=f['text_policy']=='raw-utf8-code-v1'
                if code and 'repo_name' not in p.schema_arrow.names:raise ValueError('Code repository provenance missing')
                columns=['text']+(['repo_name'] if code else []);row=0;resume_row=get('row')
                for batch in p.iter_batches(batch_size=1024,columns=columns):
                    end=row+batch.num_rows
                    if end<=resume_row:row=end;continue
                    for offset,item in enumerate(batch.to_pylist()):
                        current=row+offset
                        if current<resume_row:continue
                        text=item['text']
                        if not isinstance(text,str):raise ValueError('Text must be a string')
                        sha=text_hash(text);key='text:'+sha;repo=None
                        if code:
                            repo=item['repo_name']
                            if not isinstance(repo,str) or not repo.strip():raise ValueError('Missing code repository')
                            repo=repo.strip().lower();union(key,'repo:'+repo)
                        else:
                            normalized=unicodedata.normalize('NFC',text)
                            if normalized==text:find(key)
                            else:union(key,'text:'+text_hash(normalized))
                        db.execute('INSERT INTO docs VALUES (?,?,?,?,?,?,NULL)',(f['source'],f['path'],current,sha,repo,key))
                    row=end;put('row',row);notify('ingest_pending');db.commit();notify('ingest')
                if row!=f['rows'] or resume_row>row:raise ValueError('Source row count changed')
                put('file_index',index+1);put('row',0);db.commit()
            if not db.execute('SELECT 1 FROM docs LIMIT 1').fetchone():raise ValueError('No documents to split')
            put('phase','assign');db.commit()
        if get('phase')=='assign':
            last=get('last')
            while True:
                keys=db.execute('SELECT DISTINCT group_id FROM docs WHERE group_id>? ORDER BY group_id LIMIT 1000',(last,)).fetchall()
                if not keys:break
                for (key,) in keys:
                    root=find(key);draw=int(identity([seed,root]),16)/2**256
                    split='eval' if draw<eval_fraction else 'train'
                    db.execute('INSERT INTO assignments VALUES (?,?,?)',(key,root,split))
                last=keys[-1][0];put('last',last);db.commit();notify('assign')
            put('phase','finalize');db.commit()
        if get('phase')=='finalize':
            # Single transaction: a crash rolls back both reassignment and phase.
            db.execute('''UPDATE docs SET split=(SELECT split FROM assignments WHERE original=docs.group_id),
                          group_id=(SELECT root FROM assignments WHERE original=docs.group_id)''')
            put('phase','complete');notify('finalize_pending');db.commit();notify('finalize')
        if get('phase')!='complete':raise ValueError('Unknown partial build phase')
        counts=validate_db(db)
        per_file={f['path']:{s:n for s,n in db.execute('SELECT split,COUNT(*) FROM docs WHERE file=? GROUP BY split',(f['path'],))} for f in files}
        if sum(counts.values())!=sum(f['rows'] for f in files):raise ValueError('Source coverage mismatch')
        report=dict(status='validated',contract_id=cid,contract=contract,counts=counts,per_file=per_file,
                    scope='Exact content/web NFC/code repository grouping only; quality and near-duplicate policy remain separate')
    finally:db.close()
    if not published:database.replace(final)
    notify('publish_db')
    report['database']=dict(path=artifact_path(final),sha256=sha256_file(final))
    if report_path.exists() and json.loads(report_path.read_text())!=report:
        raise ValueError('Existing partial manifest differs; refusing overwrite')
    if not report_path.exists():atomic_json(report_path,report)
    notify('publish_report')
    atomic_json(out/'.done',dict(contract_id=cid,manifest_sha256=sha256_file(report_path)))
    return report

class SplitSelection:
    def __init__(self,manifest,sqlite_cache_mib=64):
        if type(sqlite_cache_mib) is not int or not 1<=sqlite_cache_mib<=MAX_CACHE_MIB:raise ValueError('SQLite cache must be 1..262144 MiB')
        self.path=resolve(manifest);self.report=json.loads(self.path.read_text());r=self.report
        marker=self.path.parent/'.done'
        if r.get('status')!='validated' or not marker.exists() or json.loads(marker.read_text())!=dict(contract_id=r['contract_id'],manifest_sha256=sha256_file(self.path)):
            raise ValueError('Incomplete or modified split manifest')
        if identity(r['contract'])!=r['contract_id']:raise ValueError('Split contract mismatch')
        path=resolve(r['database']['path'])
        if sha256_file(path)!=r['database']['sha256']:raise ValueError('Split DB content changed')
        self.db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        self.db.execute(f'PRAGMA cache_size={-sqlite_cache_mib*1024}')
        try:
            actual=validate_db(self.db)
            if actual!=r['counts']:raise ValueError('Split count mismatch')
            if sum(f['rows'] for f in r['contract']['files']) != sum(actual.values()):
                raise ValueError('Split total coverage mismatch')
            for f in r['contract']['files']:
                count,lo,hi=self.db.execute('SELECT COUNT(*),MIN(row),MAX(row) FROM docs WHERE file=? AND source=?',
                                            (f['path'],f['source'])).fetchone()
                if count!=f['rows'] or (count and (lo!=0 or hi!=count-1)):
                    raise ValueError('Split source row coverage mismatch')
                counts={s:n for s,n in self.db.execute('SELECT split,COUNT(*) FROM docs WHERE file=? GROUP BY split',(f['path'],))}
                if counts!=r['per_file'][f['path']]:raise ValueError('Split per-file count mismatch')
        except Exception:
            self.db.close();raise
    def close(self):self.db.close()
    def verify_source(self,source,inputs,snapshot):
        expected=[f for f in self.report['contract']['files'] if f['source']==source]
        fields=lambda f:{k:f[k] for k in ['path','bytes','sha256']}
        if [fields(f) for f in expected]!=inputs:raise ValueError('Split source files/content differ')
        if snapshot!=self.report['contract']['source_snapshot']:raise ValueError('Split source snapshot differs')
    def counts(self,path):return self.report['per_file'][artifact_path(path)]
    def rows(self,path):
        yield from self.db.execute('SELECT row,split FROM docs WHERE file=? ORDER BY row',(artifact_path(path),))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--sources-manifest',required=True)
    ap.add_argument('--output-dir',required=True);ap.add_argument('--eval-fraction',type=float,default=.01);ap.add_argument('--seed',type=int,default=1234)
    ap.add_argument('--sqlite-cache-mib',type=int,default=64)
    ap.add_argument('--hash-workers',type=int,default=1)
    a=ap.parse_args();r=build(a.sources_manifest,a.output_dir,a.eval_fraction,a.seed,sqlite_cache_mib=a.sqlite_cache_mib,hash_workers=a.hash_workers)
    print(json.dumps(dict(status=r['status'],counts=r['counts'],contract_id=r['contract_id'])))


if __name__=='__main__':main()
