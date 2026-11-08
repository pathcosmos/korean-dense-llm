import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import identity,sha256_file,atomic_json,source_files
from data.quality_report import assess,save_report,DECISIONS,read_json,POLICY
from scripts.validate_source_manifest import validate


def fixture(root,complete=True):
    out=root/'dedup';web=out/'clean/web/a.parquet';web.parent.mkdir(parents=True)
    pq.write_table(pa.table({'text':['a','b']}),web)
    counts=dict(input_docs=3,kept=2,exact_duplicates=1)
    atomic_json(Path(str(web)+'.done'),counts)
    atomic_json(out/'manifest.json',dict(policy=POLICY,inputs=[dict(source='web',path=str(root/'input/a.parquet'),size=1,mtime_ns=1)]))
    if complete:atomic_json(out/'.done',counts);atomic_json(out/'summary.json',counts)
    manifest=root/'sources.json';atomic_json(manifest,dict(schema_version=1,status='draft',sources={
        'web':dict(text_policy='nfc-web-exact-v1',files=[str(web)],origin=dict(revision='r1'),completion_marker=str(out/'.done'))}))
    return manifest,out,web


def policy(root,manifest):
    audit=root/'audit.json';atomic_json(audit,validate(manifest,candidate=True))
    proof=dict(path=str(audit),sha256=sha256_file(audit))
    checks={k:dict(status='passed',reason='Fixture scoped decision',evidence=[proof],decision=dict(action='keep',scope='fixture only')) for k in DECISIONS}
    checks['holdout_policy']['decision'].update(action='group_split',seed=7,eval_fraction=.4,code_repository_grouping=True)
    checks['tokenizer_byte_mix']['decision'].update(action='byte_mix',weights={'web':1},normalization='raw')
    checks['content_audit']=dict(status='passed',reason='Full fixture read',evidence=[proof])
    path=root/'policy.json';atomic_json(path,dict(schema_version=1,source_snapshot_id=identity(read_json(manifest)),checks=checks));return path,audit


class ReadinessTest(unittest.TestCase):
    def test_pending_completed_assessment_does_not_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,out,web=fixture(root,False)
            r=assess(manifest,out);self.assertEqual(r['run_status'],'completed');self.assertEqual(r['readiness'],'pending')
            self.assertEqual(r['checks']['exact_completion']['status'],'pending')
            save_report(r,root/'report.json');self.assertEqual(read_json(root/'report.json.done')['readiness'],'pending')
            self.assertEqual(read_json(manifest)['status'],'draft')
            with self.assertRaisesRegex(ValueError,'draft'):source_files('web',manifest)
            self.assertEqual(r,assess(manifest,out))

    def test_full_fixture_evidence_and_content_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,out,web=fixture(root);p,audit=policy(root,manifest)
            r=assess(manifest,out,p);self.assertEqual(r['readiness'],'passed');self.assertEqual(r['source_counts']['web']['counts']['exact_duplicates'],1)
            save_report(r,root/'report.json');save_report(r,root/'report.json')
            data=read_json(p);data['checks']['code_cross_policy']['status']='failed';atomic_json(p,data)
            changed=assess(manifest,out,p);self.assertEqual(changed['readiness'],'failed')
            with self.assertRaisesRegex(ValueError,'differs'):save_report(changed,root/'report.json')
            audit.write_text(audit.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'Evidence content changed'):assess(manifest,out,p)

    def test_invalid_counts_inventory_policy_and_missing_evidence(self):
        for mutation in ['counts','negative','extra','snapshot','dedup_policy','missing_evidence','missing_shard','duplicate_json','audit_coverage','holdout_policy','byte_mix']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);manifest,out,web=fixture(root);p,audit=policy(root,manifest)
                if mutation=='counts':atomic_json(Path(str(web)+'.done'),dict(input_docs=4,kept=2,exact_duplicates=1))
                if mutation=='negative':atomic_json(Path(str(web)+'.done'),dict(input_docs=1,kept=2,exact_duplicates=-1))
                if mutation=='extra':Path(str(web.parent/'extra.parquet')+'.done').touch()
                if mutation=='snapshot':
                    d=read_json(manifest);d['sources']['web']['origin']['revision']='changed';atomic_json(manifest,d)
                if mutation=='dedup_policy':
                    d=read_json(out/'manifest.json');d['policy']='bad';atomic_json(out/'manifest.json',d)
                if mutation=='missing_evidence':
                    d=read_json(p);d['checks']['holdout_policy']['evidence']=[];atomic_json(p,d)
                if mutation=='missing_shard':Path(str(web)+'.done').unlink()
                if mutation=='duplicate_json':(out/'manifest.json').write_text('{"policy":"a","policy":"b"}')
                if mutation in ('holdout_policy','byte_mix'):
                    d=read_json(p);d['checks']['holdout_policy' if mutation=='holdout_policy' else 'tokenizer_byte_mix']['decision']={'action':'keep','scope':'incomplete'};atomic_json(p,d)
                if mutation=='audit_coverage':
                    d=read_json(audit);d['sources']['web']['files']=[];atomic_json(audit,d)
                    d=read_json(p)
                    for e in d['checks'].values():e['evidence'][0]['sha256']=sha256_file(audit)
                    atomic_json(p,d)
                with self.assertRaises(ValueError):assess(manifest,out,p)

    def test_code_policy_and_shared_parent_source_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,out,web=fixture(root)
            code=web.parent/'code.parquet';pq.write_table(pa.table(dict(text=['code'],repo_name=['org/repo'])),code)
            atomic_json(Path(str(code)+'.done'),dict(policy_hash='fixture-code-policy'))
            atomic_json(Path(str(code)+'.stats.json'),dict(counts=dict(kept=1,short=2)))
            atomic_json(code.parent/'.done',dict(shards=1,policy_hash='fixture-code-policy'))
            atomic_json(code.parent/'summary.json',dict(counts=dict(kept=1,short=2),policy_hash='fixture-code-policy'))
            d=read_json(manifest);d['sources']['code']=dict(files=[str(code)],origin=dict(revision='code-r1'),
                completion_marker=str(code.parent/'.done'),text_policy='raw-utf8-code-v1');atomic_json(manifest,d)
            r=assess(manifest,out);self.assertEqual(r['checks']['code_clean:code']['status'],'passed')
            self.assertEqual(r['source_counts']['code']['counts']['input_docs'],3)
            atomic_json(code.parent/'.done',dict(shards=1,policy_hash='changed'))
            with self.assertRaisesRegex(ValueError,'policy mismatch'):assess(manifest,out)
