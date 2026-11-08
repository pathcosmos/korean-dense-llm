import collections
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from data import clean


class CleanStreamingTest(unittest.TestCase):
    def test_dedup_across_batches_resume_and_protect_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.parquet'
            # Unique enough to pass the text filter, duplicate across 4096 boundary.
            text = ' '.join(f'한국어문서내용{i}' for i in range(100))
            values = [text] + ['짧음'] * 4095 + [text, None]
            pq.write_table(pa.table({'text': values}), source)
            with patch.object(clean, 'ROOT', tmp):
                name, (counts, chars) = clean.work(('fixture', 'ko', str(source)))
                self.assertEqual(counts, collections.Counter(kept=1, short=4095, exact_dup=1, null=1))
                out = Path(tmp) / 'clean/fixture/input.parquet'
                self.assertEqual(pq.read_table(out).column('text').to_pylist(), [text])
                before = out.stat().st_mtime_ns
                self.assertEqual(clean.work(('fixture', 'ko', str(source)))[1], (counts, chars))
                self.assertEqual(out.stat().st_mtime_ns, before)
                Path(str(out) + '.done').unlink()
                with self.assertRaisesRegex(RuntimeError, 'refusing overwrite'):
                    clean.work(('fixture', 'ko', str(source)))
