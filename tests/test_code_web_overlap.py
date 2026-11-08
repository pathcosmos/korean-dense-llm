from pathlib import Path
import tempfile
import unittest
from common.data_contracts import atomic_json
from data.sample_snapshot import sample
from data.probe_code_web_overlap import run
from test_text_similarity import fixture


class CodeWebTest(unittest.TestCase):
    def test_whole_file_fence_raw_semantics_and_truncated_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,manifest,snapshot,body=fixture(root)
            r=run(manifest,root/'code-web.json')
            self.assertEqual(r['whole_document_exact_pairs'],2)
            self.assertEqual(r['fenced_whole_code_pairs'],2)
            self.assertEqual(r['eligible_fenced_blocks'],2)
            self.assertEqual(run(manifest,root/'code-web.json'),r)
            truncated=sample(frame,root/'truncated',per_source=8,max_chars=20,max_read_bytes=1000000)
            r=run(root/'truncated/manifest.json',root/'short.json')
            self.assertGreater(r['truncated_code_views_excluded_from_block_index'],0)
            self.assertEqual(r['fenced_whole_code_pairs'],0)
            self.assertEqual(r['whole_document_exact_pairs'],2)

    def test_unicode_case_and_unclosed_fences_are_not_merged(self):
        import unicodedata
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);body='s = "e\u0301"\n'+'# raw literal preservation\n'*5
            code=root/'code.parquet';web=root/'web.parquet'
            pq.write_table(pa.table(dict(text=[body],repo_name=['org/r'])),code)
            texts=['```python\n'+body+'```\n','```python\n'+unicodedata.normalize('NFC',body)+'```\n',
                   '```python\n'+body.replace('s =','S =')+'```\n','```python\n'+body]
            pq.write_table(pa.table(dict(text=texts)),web)
            for p in [code,web]:Path(str(p)+'.done').touch()
            frame=root/'frame.json';atomic_json(frame,dict(schema_version=1,sources={
                'code':dict(kind='code',revision='fixture',files=[str(code)]),
                'web':dict(kind='web',revision='fixture',files=[str(web)])}))
            sample(frame,root/'sample',per_source=4,max_read_bytes=1000000)
            r=run(root/'sample/manifest.json',root/'result.json')
            self.assertEqual(r['fenced_whole_code_pairs'],1)
            self.assertEqual(r['fenced_blocks'],3)
