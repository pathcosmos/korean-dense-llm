from pathlib import Path
import tempfile
import unittest
from data.verified_split_context import VerifiedSplitContext,distribution as reference
from data.split_distribution import distribution
from test_parallel_tokenize import fixture


class SplitDistributionTest(unittest.TestCase):
    def test_partitions_match_reference_counts_groups_and_largest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,split,_=fixture(root)
            with VerifiedSplitContext(sources,split) as context:
                original=reference(context)
                for workers in [1,3]:
                    r=distribution(context,workers=workers,worker_cache_mib=16)
                    self.assertEqual(r['counts'],original['counts']);self.assertEqual(r['rows_scanned'],160)
                    self.assertLess(r['global_groups'],160)
                    self.assertEqual(r['largest_components'][0]['documents'],original['largest_components'][0]['documents'])
                    for source in ['web','code']:
                        self.assertEqual(r['sources'][source]['splits'],original['sources'][source]['splits'])
                        self.assertEqual(r['sources'][source]['largest_components'][0]['documents'],original['sources'][source]['largest_components'][0]['documents'])
