import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import atomic_json
from data.sample_snapshot import sample,load_samples
from data.probe_text_similarity import run,shingles


def fixture(root):
    body='def function(value):\n    result = value + 1\n    return result\n'+'# preserving case and indentation\n'*3
    shared='한국어 문서 sample text for deterministic overlap '*12
    variants=[shared,shared+'extra ending',shared.replace('sample','samples'),body,'```python\n'+body+'```\n']
    sources={}
    for name,kind in [('a','web'),('b','web'),('code','code')]:
        path=root/(name+'.parquet')
        texts=(variants+[f'{name} unrelated {i} '+str(i)*30 for i in range(3)]) if kind=='web' else [body,body.replace('    return','\treturn')]+[f'print({i})' for i in range(6)]
        columns=dict(text=texts)
        if kind=='code':columns['repo_name']=[f'org/{i}' for i in range(8)]
        pq.write_table(pa.table(columns),path,row_group_size=4);Path(str(path)+'.done').touch()
        sources[name]=dict(kind=kind,revision='fixture-v1',files=[str(path)])
    frame=root/'frame.json';atomic_json(frame,dict(schema_version=1,sources=sources))
    report=sample(frame,root/'sample',per_source=8,max_chars=4096,max_read_bytes=1000000)
    return frame,root/'sample/manifest.json',report,body


class TextSimilarityTest(unittest.TestCase):
    def test_fixed_sampling_and_positive_negative_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,manifest,snapshot,_=fixture(root)
            self.assertEqual(snapshot['sample_status'],'complete_for_defined_frame')
            other=sample(frame,root/'other',per_source=8,max_chars=4096,max_read_bytes=1000000)
            self.assertEqual(other['samples']['sha256'],snapshot['samples']['sha256'])
            self.assertEqual(sample(frame,root/'sample',per_source=8,max_chars=4096,max_read_bytes=1000000),snapshot)
            r=run(manifest,root/'web.json');self.assertGreater(r['matched_view_pairs'],0)
            self.assertEqual(r['status'],'completed_on_supplied_sample')
            self.assertEqual(run(manifest,root/'web.json'),r)
            self.assertEqual(shingles('Cafe\u0301  text'),shingles('Café text'))
            self.assertFalse(shingles('abcdefghi')&shingles('0123456789'))
            with self.assertRaisesRegex(ValueError,'config changed'):run(manifest,root/'web.json',threshold=.9)
            limited=run(manifest,root/'limited.json',max_candidates=1)
            self.assertEqual(limited['status'],'partial');self.assertIn('candidate_pairs',limited['limits_reached'])

    def test_read_storage_limits_and_source_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frame,manifest,snapshot,_=fixture(root)
            empty=sample(frame,root/'read_limit',per_source=8,max_read_bytes=1)
            self.assertEqual(empty['sample_status'],'partial');self.assertEqual(empty['row_groups_read'],0)
            bounded=sample(frame,root/'storage_limit',per_source=8,max_bytes=100,max_read_bytes=1000000)
            self.assertLessEqual(bounded['stored_utf8_bytes'],100);self.assertEqual(bounded['sample_status'],'partial')
            path=Path(snapshot['samples']['path']);original=path.read_text();path.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'corpus changed'):load_samples(manifest)
            path.write_text(original)
            pq.write_table(pa.table({'text':['changed']}),root/'a.parquet')
            with self.assertRaises(ValueError):sample(frame,root/'sample',per_source=8,max_chars=4096,max_read_bytes=1000000)
