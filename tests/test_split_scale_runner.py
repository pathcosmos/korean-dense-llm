import json
from pathlib import Path
import tempfile
import unittest
from scripts.measure_split import run,validate_sizes
from data.split_manifest import build,SplitSelection
from test_split_manifest import fixture


class SplitScaleRunnerTest(unittest.TestCase):
    def test_small_case_metrics_reuse_and_data_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'scale'
            r=run(out,[120],max_case_seconds=30,min_free_gib=0)
            self.assertEqual(r['status'],'bounded_synthetic_verified')
            c=r['cases'][0];self.assertEqual(sum(c['counts'].values()),120)
            self.assertGreater(c['database_bytes'],0);self.assertGreater(c['max_rss_kib'],0)
            self.assertIn('sampled_peak_journal_bytes',c)
            self.assertEqual(run(out,[120],max_case_seconds=30,min_free_gib=0),r)
            p=out/'120/web.parquet';p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaisesRegex(ValueError,'content changed'):run(out,[120],max_case_seconds=30,min_free_gib=0)

    def test_budget_pause_resume_and_million_configuration_only(self):
        from unittest.mock import patch
        validate_sizes([1000000]);validate_sizes([1000000,3000000])
        with self.assertRaises(ValueError):validate_sizes([3000001])
        with tempfile.TemporaryDirectory() as tmp:
            with patch('scripts.measure_split.audit_running',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'active source audit'):run(Path(tmp)/'blocked',[1000000])
            self.assertFalse((Path(tmp)/'blocked').exists())
            out=Path(tmp)/'scale'
            r=run(out,[120],max_case_seconds=1e-9,min_free_gib=0)
            self.assertEqual(r['status'],'paused');self.assertFalse((out/'.done').exists())
            r=run(out,[120],max_case_seconds=30,min_free_gib=0)
            self.assertEqual(r['status'],'bounded_synthetic_verified');self.assertEqual(r['cases'][0]['attempts'],2)
            with self.assertRaisesRegex(ValueError,'config changed'):run(out,[120],sqlite_cache_mib=32,min_free_gib=0)

    def test_cache_setting_preserves_logical_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,*_=fixture(root)
            a=build(manifest,root/'a',.4,7,sqlite_cache_mib=2)
            b=build(manifest,root/'b',.4,7,sqlite_cache_mib=64)
            self.assertEqual(a['contract_id'],b['contract_id'])
            left=SplitSelection(root/'a/manifest.json',sqlite_cache_mib=2);right=SplitSelection(root/'b/manifest.json',sqlite_cache_mib=64)
            try:self.assertEqual(left.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall(),right.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall())
            finally:left.close();right.close()
            with self.assertRaises(ValueError):build(manifest,root/'bad',sqlite_cache_mib=0)

    def test_fast_worker_final_memory_is_checked(self):
        import time
        from unittest.mock import patch
        original_sleep=time.sleep
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'scale'
            with patch('scripts.measure_split.time.sleep',side_effect=lambda _:original_sleep(2)):
                r=run(out,[120],max_case_seconds=30,max_rss_gib=.05,min_free_gib=0)
            self.assertEqual(r['status'],'paused');self.assertIn(r['pause_reason'],['rss','rss_at_completion'])
            self.assertFalse((out/'.done').exists())
