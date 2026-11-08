from pathlib import Path
import tempfile
import unittest
from data.probe_similarity_recall import run
from test_text_similarity import fixture


class RecallTest(unittest.TestCase):
    def test_exhaustive_controls_and_real_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,manifest,*_=fixture(root)
            r=run(manifest,root/'study',max_real_per_source=4,max_view_chars=256)
            self.assertTrue(r['reference_complete']);self.assertEqual(r['real_documents'],8)
            self.assertEqual(r['synthetic_documents'],40)
            controls=r['thresholds']['0.8']['synthetic_only']
            self.assertGreater(controls['positive_pairs'],0);self.assertEqual(controls['candidate_recall'],1.)
            self.assertEqual(run(manifest,root/'study',max_real_per_source=4,max_view_chars=256),r)
            with self.assertRaisesRegex(ValueError,'config changed'):run(manifest,root/'study',max_real_per_source=3,max_view_chars=256)

    def test_candidate_cap_exposes_misses_and_zero_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,manifest,*_=fixture(root)
            r=run(manifest,root/'study',max_real_per_source=0,max_view_chars=256,max_candidates=1)
            self.assertEqual(r['production_status'],'partial')
            self.assertIsNone(r['thresholds']['0.8']['real_only']['candidate_recall'])
            m=r['thresholds']['0.8']['synthetic_only']
            self.assertGreater(m['missed_candidate_pairs'],0);self.assertLess(m['candidate_recall'],1.)
