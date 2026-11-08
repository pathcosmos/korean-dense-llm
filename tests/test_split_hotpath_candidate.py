from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from data.split_manifest import build as baseline,SplitSelection
from data.split_manifest_candidate import build
from test_split_manifest import fixture


class SplitHotpathCandidateTest(unittest.TestCase):
    def test_raw_nfc_and_code_components_match_baseline_after_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,_,files=fixture(root)
            pq.write_table(pa.table({'text':['dup','e\u0301','é','  exact spacing  ','A','a']}),files[0])
            baseline(manifest,root/'base',.4,7)
            def fail(phase):
                if phase=='ingest':raise RuntimeError('injected')
            with self.assertRaisesRegex(RuntimeError,'injected'):build(manifest,root/'candidate',.4,7,_checkpoint=fail)
            build(manifest,root/'candidate',.4,7)
            a=SplitSelection(root/'base/manifest.json');b=SplitSelection(root/'candidate/manifest.json')
            try:
                self.assertEqual(a.report['contract_id'],b.report['contract_id'])
                self.assertEqual(a.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall(),b.db.execute('SELECT * FROM docs ORDER BY file,row').fetchall())
                self.assertEqual(a.db.execute('SELECT * FROM nodes ORDER BY key').fetchall(),b.db.execute('SELECT * FROM nodes ORDER BY key').fetchall())
                web=b.db.execute('SELECT row,content_sha256,group_id FROM docs WHERE file=? ORDER BY row',(str(files[0]),)).fetchall()
                self.assertNotEqual(web[1][1],web[2][1])  # raw decomposed/composed bytes remain different
                self.assertEqual(web[1][2],web[2][2])  # NFC grouping keeps them in one split
                self.assertNotEqual(web[4][2],web[5][2])  # A/a must not be normalized together
                self.assertEqual(b.db.execute("SELECT COUNT(DISTINCT group_id) FROM docs WHERE repo IN ('org/a','org/b')").fetchone()[0],1)
            finally:a.close();b.close()
