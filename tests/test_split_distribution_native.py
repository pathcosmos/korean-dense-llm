from pathlib import Path
import tempfile
import unittest
from data.verified_split_context import VerifiedSplitContext,distribution as reference
from data.split_distribution_native import distribution
from test_parallel_tokenize import fixture


class NativeDistributionTest(unittest.TestCase):
    def test_native_counts_groups_and_no_persistent_db_change(self):
        from common.data_contracts import sha256_file,resolve
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,split,_=fixture(root)
            with VerifiedSplitContext(sources,split) as ctx:
                path=resolve(ctx.selection.report['database']['path']);before=sha256_file(path);baseline=reference(ctx)
                r=distribution(ctx,sort_threads=2)
                self.assertEqual(r['counts'],baseline['counts']);self.assertEqual(r['original_rows_materialized'],160)
                for source in baseline['sources']:
                    self.assertEqual(r['sources'][source]['splits'],baseline['sources'][source]['splits'])
                    self.assertEqual(r['sources'][source]['largest_components'][0]['documents'],baseline['sources'][source]['largest_components'][0]['documents'])
                self.assertEqual(sha256_file(path),before)
                self.assertFalse(ctx.selection.db.execute("SELECT name FROM sqlite_temp_master WHERE name LIKE 'sizes_%'").fetchall())
