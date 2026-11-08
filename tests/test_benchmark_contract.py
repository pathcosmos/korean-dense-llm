import json,tempfile,unittest
from pathlib import Path
from scripts.benchmark_contract import bind,certify,verify
class BenchmarkContractTest(unittest.TestCase):
    def test_reuse_rejects_contract_and_artifact_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);c=p/'contract.json';bind(c,{'code':'a'});bind(c,{'code':'a'})
            with self.assertRaises(ValueError):bind(c,{'code':'b'})
            r=p/'result.json';r.write_text('original');m=p/'.done';certify(m,[r],'id');verify(m,[r],'id')
            with self.assertRaises(ValueError):verify(m,[r],'different')
            r.write_text('modified')
            with self.assertRaises(ValueError):verify(m,[r],'id')
    def test_legacy_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=p/'.done';m.touch();r=p/'result.json';r.write_text('{}')
            with self.assertRaises(ValueError):verify(m,[r],'id')

    def test_event_accepts_string_project_root(self):
        from unittest.mock import patch
        from scripts.benchmark_gpu import event
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'logs').mkdir();run=p/'run';run.mkdir()
            with patch('scripts.benchmark_gpu.PROJECT_ROOT',tmp):event(run,'started')
            self.assertEqual(json.loads((run/'state.json').read_text())['event'],'started')
