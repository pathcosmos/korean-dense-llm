import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common.data_contracts import atomic_json,identity,sha256_file
from scripts.watch_split_finalization import verify_completion,phase,read_small


def fixture(root):
    run=root/'run';run.mkdir();out=root/'published';out.mkdir();source=root/'sources.json';source.write_text('{}')
    code=root/'producer.py';code.write_text('# fixed producer\n')
    contract=dict(output=str(out),source_manifest=str(source),source_sha256=sha256_file(source),source_snapshot='snapshot',seed=1234,eval_fraction=.01,code={str(code):sha256_file(code)})
    inner=dict(source_snapshot='snapshot',seed=1234,eval_fraction=.01,files=[dict(path='fixture',rows=3)])
    cid=identity(inner);database=out/'split.sqlite';database.write_bytes(b'closed publisher-owned DB fixture')
    manifest=dict(status='validated',contract=inner,contract_id=cid,counts=dict(train=2,eval=1),per_file=dict(fixture=dict(train=2,eval=1)),database=dict(path=str(database),sha256=sha256_file(database)))
    atomic_json(out/'manifest.json',manifest);atomic_json(out/'publication.json',contract)
    atomic_json(out/'.done',dict(contract_id=cid,manifest_sha256=sha256_file(out/'manifest.json')))
    report=dict(status='passed',manifest=str(out/'manifest.json'),manifest_sha256=sha256_file(out/'manifest.json'),counts=manifest['counts'],split_contract_id=cid,database_sha256=manifest['database']['sha256'],verification='Publisher validation fixture; observer is receipt-only.')
    atomic_json(run/'contract.json',contract);atomic_json(run/'report.json',report)
    atomic_json(run/'.done',dict(report_sha256=sha256_file(run/'report.json'),contract_sha256=sha256_file(run/'contract.json')))
    return run,out,sha256_file(run/'contract.json')


class FinalizationObserverTest(unittest.TestCase):
    def test_valid_receipts_without_rehashing_database(self):
        with tempfile.TemporaryDirectory() as d:
            run,out,digest=fixture(Path(d))
            with patch('scripts.watch_split_finalization.sha256_file',wraps=sha256_file) as hashed:
                result=verify_completion(run,digest,3)
            self.assertEqual(result['counts'],dict(train=2,eval=1))
            self.assertFalse(any(Path(c.args[0]).name=='split.sqlite' for c in hashed.call_args_list))

    def test_missing_marker_tampered_manifest_or_changed_producer_rejected(self):
        for variant in ['marker','manifest','code','row_total']:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                root=Path(d);run,out,digest=fixture(root)
                if variant=='marker':(out/'.done').unlink()
                if variant=='manifest':(out/'manifest.json').write_text((out/'manifest.json').read_text()+' ')
                if variant=='code':(root/'producer.py').write_text('# changed')
                with self.assertRaises((ValueError,FileNotFoundError)):verify_completion(run,digest,4 if variant=='row_total' else 3)

    def test_recertified_wrong_path_and_counts_rejected(self):
        for variant in ['path','counts','owner']:
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                run,out,digest=fixture(Path(d));m=json.loads((out/'manifest.json').read_text())
                if variant=='path':m['database']['path']=str(Path(d)/'other.sqlite')
                if variant=='counts':m['per_file']['fixture']['train']=1
                atomic_json(out/'manifest.json',m);atomic_json(out/'.done',dict(contract_id=m['contract_id'],manifest_sha256=sha256_file(out/'manifest.json')))
                r=json.loads((run/'report.json').read_text());r['manifest_sha256']=sha256_file(out/'manifest.json');atomic_json(run/'report.json',r)
                atomic_json(run/'.done',dict(report_sha256=sha256_file(run/'report.json'),contract_sha256=digest))
                if variant=='owner':atomic_json(out/'publication.json',dict(wrong=True))
                with self.assertRaises(ValueError):verify_completion(run,digest,3)

    def test_long_sql_phase_and_small_metadata_bound(self):
        self.assertEqual(phase('assign','finalize','assign'),'finalize')
        self.assertEqual(phase('finalize','complete','finalize'),'validate_local')
        self.assertEqual(phase('assign',None,'finalize'),'finalize')
        self.assertEqual(phase('verify_published','complete','copy'),'verify_published')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'large.json'
            with p.open('wb') as f:f.truncate(4*1024**2+1)
            with self.assertRaisesRegex(ValueError,'metadata'):read_small(p)
