import json
from pathlib import Path
import shutil
import tempfile
import unittest
from common.data_contracts import atomic_json,identity,sha256_file
from data.local_split_reader import prepare_reader
from data.verified_split_context import VerifiedSplitContext
from data.split_inputs import SplitInputs,prepare
from test_parallel_tokenize import fixture


class LocalSplitReaderTest(unittest.TestCase):
    def test_local_reader_keeps_canonical_sample_binding_and_exact_text(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,_=fixture(root)
            path=prepare_reader(split,root/'reader',scratch=root/'scratch')
            with VerifiedSplitContext(sm,split,reader_manifest=path) as ctx:
                inputs=SplitInputs(sm,split,verified_context=ctx)
                actual=prepare(sm,split,root/'local_sample',dict(web=120,code=120),mode='train',verified_inputs=inputs)
                self.assertEqual(actual['contract']['split_manifest'],str(split))
                self.assertEqual(actual['contract']['split_manifest_sha256'],sha256_file(split))
            expected=prepare(sm,split,root/'original_sample',dict(web=120,code=120),mode='train')
            self.assertEqual(actual['samples']['sha256'],expected['samples']['sha256'])
            self.assertEqual(prepare_reader(split,root/'reader',scratch=root/'scratch'),path)

    def test_same_size_corruption_and_recertified_policy_drift_rejected(self):
        for mode in ['content','policy','canonical_marker']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);sm,split,_=fixture(root);path=prepare_reader(split,root/'reader',scratch=root/'scratch')
                report=json.loads(path.read_text())
                if mode=='content':
                    p=Path(report['database']['path']);body=p.read_bytes();p.write_bytes(b'bad!'+body[4:])
                if mode=='policy':
                    report['contract']['seed']+=1;report['contract_id']=identity(report['contract']);atomic_json(path,report)
                    atomic_json(path.parent/'.done',dict(contract_id=report['contract_id'],manifest_sha256=sha256_file(path)))
                if mode=='canonical_marker':(split.parent/'.done').unlink()
                with self.assertRaises(ValueError):VerifiedSplitContext(sm,split,reader_manifest=path)

    def test_copy_resumes_verified_prefix_and_preserves_invalid_prefix(self):
        for correct in [True,False]:
            with self.subTest(correct=correct),tempfile.TemporaryDirectory() as d:
                root=Path(d);_,split,_=fixture(root);r=json.loads(split.read_text());scratch=root/'scratch';scratch.mkdir()
                partial=scratch/('reader-'+r['database']['sha256'][:24]+'.sqlite.copying')
                prefix=Path(r['database']['path']).read_bytes()[:1024];partial.write_bytes(prefix if correct else b'x'*len(prefix))
                if correct:
                    result=prepare_reader(split,root/'reader',scratch=scratch);local=Path(json.loads(result.read_text())['database']['path'])
                    self.assertEqual(sha256_file(local),r['database']['sha256'])
                else:
                    with self.assertRaisesRegex(ValueError,'prefix differs'):prepare_reader(split,root/'reader',scratch=scratch)
                    self.assertEqual(partial.read_bytes(),b'x'*len(prefix))
