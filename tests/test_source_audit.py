import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import atomic_json,identity,sha256_file,source_files
from common.source_inventory import definition,load_audit
from data.source_audit import run,AuditBudgetReached
from data.quality_report import assess,DECISIONS,POLICY
from scripts.finalize_source_manifest import finalize


def fixture(root):
    dedup=root/'dedup';web=dedup/'clean/web/a.parquet';web.parent.mkdir(parents=True)
    body='s = "e\u0301"\n'+'# preserve literal\n'*6
    texts=['plain web text','```python\n'+body+'```\n']
    pq.write_table(pa.table(dict(text=texts,h=[1,2])),web,row_group_size=1)
    counts=dict(input_docs=3,kept=2,exact_duplicates=1);atomic_json(Path(str(web)+'.done'),counts)
    atomic_json(dedup/'manifest.json',dict(policy=POLICY,inputs=[dict(source='web',path=str(root/'input/a.parquet'),size=1,mtime_ns=1)]))
    atomic_json(dedup/'.done',counts);atomic_json(dedup/'summary.json',counts)
    code=root/'code/a.parquet';code.parent.mkdir()
    pq.write_table(pa.table(dict(text=[body,body.replace('s =','S =')],h=[1,2],repo_name=['org/a','org/b'],path=['a.py','b.py'],
        license=['mit','mit'],language=['Python','Python'],source_row=[0,2],content_sha256=[__import__('hashlib').sha256(t.encode()).hexdigest() for t in [body,body.replace('s =','S =')]])),code,row_group_size=1)
    atomic_json(Path(str(code)+'.done'),dict(policy_hash='fixture-code-policy'))
    atomic_json(Path(str(code)+'.stats.json'),dict(counts=dict(kept=2,short=1)))
    atomic_json(code.parent/'.done',dict(shards=1,policy_hash='fixture-code-policy'))
    atomic_json(code.parent/'summary.json',dict(policy_hash='fixture-code-policy',counts=dict(kept=2,short=1)))
    sources=dict(web=dict(files=[str(web)],text_policy='nfc-web-exact-v1',origin=dict(revision='web-v1'),completion_marker=str(dedup/'.done')),
                 code=dict(files=[str(code)],text_policy='raw-utf8-code-v1',origin=dict(revision='code-v1'),completion_marker=str(code.parent/'.done')))
    manifest=root/'candidate.json';atomic_json(manifest,dict(schema_version=1,status='draft',sources=sources))
    return manifest,dedup,web,code


def policy(root,candidate,audit):
    proof=dict(path=str(audit),sha256=sha256_file(audit))
    checks={k:dict(status='passed',reason='Synthetic fixture scoped policy',decision=dict(action='keep',scope='fixture only'),evidence=[proof]) for k in DECISIONS}
    checks['holdout_policy']['decision'].update(action='group_split',seed=7,eval_fraction=.4,code_repository_grouping=True)
    checks['tokenizer_byte_mix']['decision'].update(action='byte_mix',weights=dict(web=1,code=1),normalization='raw')
    checks['content_audit']=dict(status='passed',reason='All fixture files decoded and hashed',evidence=[proof])
    path=root/'policy.json';atomic_json(path,dict(schema_version=1,source_snapshot_id=identity(json.loads(candidate.read_text())),checks=checks));return path


class SourceAuditTest(unittest.TestCase):
    def test_full_columns_fingerprints_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,dedup,web,code=fixture(root)
            r=run(candidate,root/'audit',min_free_gib=0,batch_size=1)
            self.assertEqual(r['rows_verified'],4);self.assertEqual(r['structural_status'],'passed')
            self.assertEqual(run(candidate,root/'audit',min_free_gib=0,batch_size=1),r)
            loaded=load_audit(root/'audit/report.json');self.assertEqual(loaded,r)
            web_fp=pq.read_table(r['sources']['web']['files'][0]['fingerprints']['path']).to_pylist()
            code_fp=pq.read_table(r['sources']['code']['files'][0]['fingerprints']['path']).to_pylist()
            self.assertEqual(web_fp[1]['web_block_sha256'],[code_fp[0]['code_file_block_sha256']])
            self.assertNotEqual(code_fp[0]['raw_sha256'],code_fp[1]['raw_sha256'])
            p=Path(r['sources']['web']['files'][0]['fingerprints']['path']);p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'artifact content changed'):run(candidate,root/'audit',min_free_gib=0,batch_size=1)

    def test_abrupt_crash_at_each_publish_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,*_=fixture(root)
            script='''import os,sys
from data.source_audit import run
def crash(phase):
    if phase==sys.argv[3]:os._exit(73)
run(sys.argv[1],sys.argv[2],min_free_gib=0,batch_size=1,_checkpoint=crash)
'''
            for phase in ['fingerprint_published','receipt_committed','report_published']:
                out=root/phase
                p=subprocess.run([sys.executable,'-c',script,str(candidate),str(out),phase],capture_output=True,text=True,timeout=30)
                self.assertEqual(p.returncode,73,p.stderr)
                r=run(candidate,out,min_free_gib=0,batch_size=1);self.assertEqual(r['rows_verified'],4)
                self.assertEqual(json.loads((out/'state.json').read_text())['phase'],'complete')

    def test_ready_proof_bridge_and_same_size_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,dedup,web,code=fixture(root)
            audit=root/'audit/report.json';r=run(candidate,audit.parent,min_free_gib=0);p=policy(root,candidate,audit)
            ready=finalize(candidate,audit,p,dedup,root/'ready.json')
            self.assertEqual(definition(ready),definition(json.loads(candidate.read_text())))
            self.assertEqual(ready['content_inventory_id'],r['content_inventory_id'])
            self.assertEqual(source_files('web',root/'ready.json')[0],[web])
            self.assertNotEqual(identity(ready),identity(json.loads(candidate.read_text())))
            original=web.read_bytes();stat=web.stat();mutated=original.replace(b'plain web text',b'other web text')
            self.assertNotEqual(mutated,original);self.assertEqual(len(original),len(mutated));web.write_bytes(mutated)
            os.utime(web,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'content changed'):source_files('web',root/'ready.json')
            with self.assertRaisesRegex(ValueError,'input/artifact content changed'):run(candidate,audit.parent,min_free_gib=0)

    def test_bad_code_provenance_and_pilot_not_global_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,dedup,web,code=fixture(root)
            pilot=root/'pilot/report.json';r=run(candidate,pilot.parent,pilot=True,min_free_gib=0)
            self.assertEqual(r['structural_status'],'passed_selected_files')
            with self.assertRaisesRegex(ValueError,'Full audit required'):load_audit(pilot)
            p=policy(root,candidate,pilot)
            with self.assertRaisesRegex(ValueError,'does not cover'):assess(candidate,dedup,p)
            table=pq.read_table(code).to_pydict();table['repo_name'][0]='';pq.write_table(pa.table(table),code)
            with self.assertRaisesRegex(ValueError,'provenance'):run(candidate,root/'bad',min_free_gib=0)
            self.assertEqual(json.loads((root/'bad/state.json').read_text())['phase'],'error')

    def test_pause_and_missing_ready_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,*_=fixture(root)
            with self.assertRaises(AuditBudgetReached):run(candidate,root/'paused',min_free_gib=0,max_seconds=1e-9)
            self.assertEqual(json.loads((root/'paused/state.json').read_text())['phase'],'paused')
            run(candidate,root/'paused',min_free_gib=0)
            raw=json.loads(candidate.read_text());raw.update(status='ready',content_inventory_id='fake')
            atomic_json(root/'fake.json',raw)
            with self.assertRaisesRegex(ValueError,'Incomplete audited'):source_files('web',root/'fake.json')

    def test_code_checksum_row_order_and_null_text_rejected(self):
        for mutation in ['checksum','row_order','null_text']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);candidate,dedup,web,code=fixture(root)
                values=pq.read_table(code).to_pydict()
                if mutation=='checksum':values['content_sha256'][0]='0'*64
                if mutation=='row_order':values['source_row']=[2,0]
                if mutation=='null_text':values['text'][0]=None
                pq.write_table(pa.table(values),code)
                with self.assertRaises(ValueError):run(candidate,root/'audit',min_free_gib=0)
