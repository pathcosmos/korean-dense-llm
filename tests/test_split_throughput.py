import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.data_contracts import sha256_file
from data.source_audit import run
from scripts.finalize_source_manifest import finalize
from data.split_manifest import build, SplitSelection
from test_source_audit import fixture, policy
from test_split_manifest import fixture as unbound_fixture


class SplitThroughputTest(unittest.TestCase):
    def test_parallel_bound_input_hashes_once_and_rejects_same_size_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate,dedup,web,code=fixture(root)
            audit=root/'audit/report.json';run(candidate,audit.parent,min_free_gib=0)
            finalize(candidate,audit,policy(root,candidate,audit),dedup,root/'ready.json')
            checked=[]
            def count(path):
                if Path(path).suffix=='.parquet':checked.append(Path(path))
                return sha256_file(path)
            with patch('common.source_inventory.sha256_file',side_effect=count), patch('data.split_manifest.sha256_file',side_effect=count):
                report=build(root/'ready.json',root/'split',.4,7,hash_workers=8,sqlite_cache_mib=131072)
            self.assertCountEqual(checked,[web,code])
            self.assertEqual(sum(report['counts'].values()),4)
            original=web.read_bytes();stat=web.stat();web.write_bytes(original.replace(b'plain web text',b'other web text'))
            os.utime(web,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            with self.assertRaisesRegex(ValueError,'content changed'):
                build(root/'ready.json',root/'changed',.4,7,hash_workers=8,sqlite_cache_mib=131072)

    def test_large_cache_and_workers_preserve_rows_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,*_=unbound_fixture(root)
            build(manifest,root/'baseline',.4,7,sqlite_cache_mib=64)
            def stop(phase):
                if phase=='ingest':raise RuntimeError('injected')
            with self.assertRaisesRegex(RuntimeError,'injected'):
                build(manifest,root/'large',.4,7,sqlite_cache_mib=131072,hash_workers=8,_checkpoint=stop)
            build(manifest,root/'large',.4,7,sqlite_cache_mib=131072,hash_workers=8)
            left=SplitSelection(root/'baseline/manifest.json')
            right=SplitSelection(root/'large/manifest.json',sqlite_cache_mib=131072)
            try:
                self.assertEqual(right.db.execute('PRAGMA cache_size').fetchone()[0],-131072*1024)
                self.assertEqual(left.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall(),right.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall())
            finally:left.close();right.close()
            for workers in [0,33,True]:
                with self.assertRaises(ValueError):build(manifest,root/'invalid',hash_workers=workers)
            for cache in [0,262145,True]:
                with self.assertRaises(ValueError):build(manifest,root/'invalid',sqlite_cache_mib=cache)
