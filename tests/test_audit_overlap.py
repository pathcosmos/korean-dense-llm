from pathlib import Path
import tempfile
import unittest
from data.source_audit import run as audit
from data.probe_audit_overlap import run
from test_source_audit import fixture


class AuditOverlapTest(unittest.TestCase):
    def test_fingerprint_join_positive_scope_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,*_=fixture(root);audit(candidate,root/'audit',min_free_gib=0)
            r=run(root/'audit/report.json',root/'overlap.json')
            self.assertEqual(r['fence_matching_pairs'],1);self.assertEqual(r['eligible_web_fences'],1)
            self.assertEqual(r['whole_matching_pairs'],0);self.assertTrue(r['full_source_coverage'])
            self.assertEqual(run(root/'audit/report.json',root/'overlap.json'),r)
            partial=run(root/'audit/report.json',root/'partial.json',max_seconds=1e-9)
            self.assertEqual(partial['status'],'partial');self.assertFalse(partial['full_source_coverage'])
            with self.assertRaisesRegex(ValueError,'row budget'):run(root/'audit/report.json',root/'cap.json',max_code_rows=1)

    def test_pilot_rejected_by_default_and_fingerprint_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,*_=fixture(root);r=audit(candidate,root/'pilot',pilot=True,min_free_gib=0)
            with self.assertRaisesRegex(ValueError,'Full audit required'):run(root/'pilot/report.json',root/'bad.json')
            result=run(root/'pilot/report.json',root/'pilot.json',allow_pilot=True)
            self.assertFalse(result['full_source_coverage'])
            f=Path(r['sources']['code']['files'][0]['fingerprints']['path']);f.write_bytes(f.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'fingerprint content changed'):run(root/'pilot/report.json',root/'changed.json',allow_pilot=True)
