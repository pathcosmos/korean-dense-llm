import hashlib
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from data.probe_code_math_overlap import run_probe, fenced_blocks, block_key


class CrossOverlapTest(unittest.TestCase):
    def test_fences_and_semantics(self):
        source = '```python\nif x:\n    run()\n```\n~~~js\nlet X = 1;\n~~~\n'
        self.assertEqual(list(fenced_blocks(source)), ['if x:\n    run()\n', 'let X = 1;\n'])
        self.assertEqual(list(fenced_blocks('```py\nunclosed\n')), [])
        self.assertEqual(block_key('\nX = 1\n'), block_key('X = 1'))
        self.assertNotEqual(block_key(' X = 1'), block_key('X = 1'))
        self.assertNotEqual(block_key('X = 1'), block_key('x = 1'))
        self.assertNotEqual(block_key('é'), block_key('e\u0301'))

    def test_exact_and_block_counts_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);c=root/'c.parquet';m=root/'m.parquet';out=root/'result'
            code='if condition:\n    run()\n'
            pq.write_table(pa.table({'text':[code,code], 'content_sha256':[hashlib.sha256(code.encode()).hexdigest()]*2}),c)
            math=[code, 'Explanation\n```python\n'+code+'```\n', '```python\nif condition:\nrun()\n```\n']
            pq.write_table(pa.table({'text':math}),m)
            result=run_probe([c],[m],out,min_chars=10)
            self.assertEqual(result['code_internal_duplicate_rows'],1)
            self.assertEqual(result['counts']['whole_document_exact_matches'],1)
            self.assertEqual(result['counts']['matching_fenced_blocks'],1)
            self.assertEqual(result['counts']['math_documents_with_code_block_match'],1)
            before=(out/'code_index.parquet').stat().st_mtime_ns
            self.assertEqual(run_probe([c],[m],out,10),result)
            self.assertEqual((out/'code_index.parquet').stat().st_mtime_ns,before)
            with self.assertRaisesRegex(RuntimeError,'snapshot or policy changed'):
                run_probe([c],[m],out,20)

    def test_corrupted_saved_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=root/'c.parquet'
            pq.write_table(pa.table({'text':['hello'], 'content_sha256':['0'*64]}),c)
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                run_probe([c],[],root/'out',1)
