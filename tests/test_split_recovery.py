import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from data.split_manifest import build,SplitSelection
from test_split_manifest import fixture


class SplitRecoveryTest(unittest.TestCase):
    def test_abrupt_exit_and_resume_at_every_transaction_and_publish_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,files=fixture(root)
            pq.write_table(pa.table({'text':['dup','e\u0301','é']+[f'row{i}' for i in range(2200)]}),files[0])
            baseline=build(manifest,root/'baseline',.4,7)
            def rows(path):
                s=SplitSelection(path/'manifest.json')
                try:return s.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall()
                finally:s.close()
            expected=rows(root/'baseline')
            script='''import os,sys
from data.split_manifest import build
def crash(phase):
    if phase==sys.argv[3]:os._exit(73)
build(sys.argv[1],sys.argv[2],.4,7,_checkpoint=crash)
'''
            for phase in ['ingest_pending','ingest','assign','finalize_pending','finalize','publish_db','publish_report']:
                with self.subTest(phase=phase):
                    out=root/phase
                    p=subprocess.run([sys.executable,'-c',script,str(manifest),str(out),phase],capture_output=True,text=True,timeout=30)
                    self.assertEqual(p.returncode,73,p.stderr);self.assertFalse((out/'.done').exists())
                    actual=build(manifest,out,.4,7)
                    self.assertEqual(actual['counts'],baseline['counts']);self.assertEqual(rows(out),expected)
                    self.assertEqual(build(manifest,out,.4,7),actual)

    def test_partial_contract_mutation_and_legacy_directory_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,files=fixture(root);out=root/'partial'
            def stop(phase):
                if phase=='ingest':raise RuntimeError('injected')
            with self.assertRaisesRegex(RuntimeError,'injected'):build(manifest,out,.4,7,_checkpoint=stop)
            with self.assertRaisesRegex(ValueError,'changed'):build(manifest,out,.4,8)
            pq.write_table(pa.table({'text':['changed']}),files[0])
            with self.assertRaisesRegex(ValueError,'changed'):build(manifest,out,.4,7)
            legacy=root/'legacy';legacy.mkdir();(legacy/'split.sqlite.building').touch()
            with self.assertRaisesRegex(ValueError,'Unrecognized'):build(manifest,legacy,.4,7)
