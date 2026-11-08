import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from data.dedup import deduplicate, fingerprint


class ExactDedupTest(unittest.TestCase):
    def test_global_duplicates_resume_and_input_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / 'a.parquet', root / 'b.parquet'
            pq.write_table(pa.table({'text': ['same', 'Case', 'same'], 'h': [1, 2, 3]}), a)
            pq.write_table(pa.table({'text': ['same', 'case', ' case'], 'h': [4, 5, 6]}), b)
            inputs = [('wiki', a), ('web', b)]
            out = root / 'output'
            expected = dict(input_docs=6, kept=4, exact_duplicates=2)
            self.assertEqual(deduplicate(inputs, out, 'test'), expected)
            output = out / 'clean/wiki/a.parquet'
            modified = output.stat().st_mtime_ns
            Path(str(output) + '.done').unlink()  # DB commit before marker crash
            self.assertEqual(deduplicate(inputs, out, 'test'), expected)
            self.assertEqual(output.stat().st_mtime_ns, modified)
            self.assertTrue(Path(str(output) + '.done').exists())
            with self.assertRaisesRegex(RuntimeError, 'Inputs/order/policy changed'):
                deduplicate(list(reversed(inputs)), out, 'test')

    def test_orphan_is_preserved_and_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = root / 'a.parquet';out = root / 'out'
            pq.write_table(pa.table({'text': ['hello'], 'h': [1]}), a)
            orphan = out / 'clean/src/a.parquet';orphan.parent.mkdir(parents=True)
            orphan.write_bytes(b'crash before db commit')
            self.assertEqual(deduplicate([('src', a)], out, 'test')['kept'], 1)
            self.assertEqual(len(list((out / 'orphans').rglob('*.parquet'))), 1)

    def test_failure_rolls_back_seen_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = root / 'a.parquet'; out = root / 'out'
            pq.write_table(pa.table({'text': ['valid', None], 'h': [1, 2]}), a)
            with self.assertRaisesRegex(ValueError, 'not a string'):
                deduplicate([('src', a)], out, 'test')
            with sqlite3.connect(out / 'index.sqlite') as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM seen').fetchone()[0], 0)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM processed').fetchone()[0], 0)
            self.assertFalse((out / '.done').exists())

    def test_semantic_whitespace_and_case_preserved(self):
        self.assertNotEqual(fingerprint('X'), fingerprint('x'))
        self.assertNotEqual(fingerprint('a b'), fingerprint('ab'))
        self.assertNotEqual(fingerprint('  code'), fingerprint('code'))
        self.assertEqual(fingerprint('é'), fingerprint('e\u0301'))
