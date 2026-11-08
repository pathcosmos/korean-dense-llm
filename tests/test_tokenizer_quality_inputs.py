import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from common.data_contracts import atomic_json,sha256_file
from data.split_manifest import build
from data.verified_split_context import VerifiedSplitContext
from data.split_inputs import SplitInputs,prepare
from tokenizer.quality_inputs import prepare_checks,verify_checks,exclusion_groups
from tokenizer.train_tokenizer import train_prepared
from tokenizer.check_tokenizer import check


def fixture(root):
    sources={}
    for name in ['ko','en','code','math']:
        p=root/(name+'.parquet');texts=[f'{name} document {i} 한글 e\u0301\n\treturn x+1 ∑ x²' for i in range(100)]
        table={'text':texts}
        if name=='code':table['repo_name']=[f'org/repo{i//4}' for i in range(100)]
        pq.write_table(pa.table(table),p,row_group_size=10);Path(str(p)+'.done').touch()
        sources[name]=dict(files=[str(p)],text_policy='raw-utf8-code-v1' if name=='code' else 'nfc-web-exact-v1',completion_marker=str(p)+'.done')
    manifest=root/'sources.json';atomic_json(manifest,dict(schema_version=1,status='ready',sources=sources));build(manifest,root/'split',eval_fraction=.3,seed=17)
    return manifest,root/'split/manifest.json'


class QualityInputsTest(unittest.TestCase):
    def test_independence_whole_docs_four_categories_and_native_checker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,split=fixture(root)
            with VerifiedSplitContext(sources,split) as ctx:
                inputs=SplitInputs(sources,split,verified_context=ctx)
                baseline=next(inputs.rows('code','eval'));ids=root/'ids.jsonl';ids.write_text(json.dumps(dict(source='code',file=baseline['file'],row=baseline['row'],body_sha256=baseline['content_sha256']))+'\n')
                atomic_json(str(ids)+'.done',dict(identities_sha256=sha256_file(ids),source_manifest_sha256=ctx.source_sha,documents=1))
                excluded=exclusion_groups(inputs,ids)
                prepared=prepare_checks(inputs,root/'checks',{s:s for s in inputs.files},per_source=3,excluded_groups=excluded)
                self.assertEqual(verify_checks(inputs,root/'checks/manifest.json',excluded_groups=excluded),prepared)
                rows=[json.loads(x) for x in Path(prepared['samples']['path']).read_text().splitlines()]
                self.assertEqual(len(rows),12);self.assertEqual({r['category'] for r in rows},{'ko','en','code','math'})
                self.assertTrue(all(r['group_id'] not in excluded and not r['truncated'] and r['split']=='eval' for r in rows))
                train=prepare(sources,split,root/'train',{s:80 for s in inputs.files},mode='train',verified_inputs=inputs)
                train_prepared(root/'train/manifest.json',root/'tok.json',272,verified_inputs=inputs)
                report=check(root/'tok.json',Path(prepared['samples']['path']),root/'check.json',272,'raw')
                self.assertEqual(report['status'],'passed')

    def test_bad_category_partial_budget_and_sample_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sources,split=fixture(root)
            with VerifiedSplitContext(sources,split) as ctx:
                inputs=SplitInputs(sources,split,verified_context=ctx)
                with self.assertRaisesRegex(ValueError,'four categories'):prepare_checks(inputs,root/'missing',dict(ko='ko'),per_source=1)
                with self.assertRaisesRegex(ValueError,'budget'):prepare_checks(inputs,root/'partial',{s:s for s in inputs.files},per_source=2,max_utf8_bytes=1)
                self.assertFalse((root/'partial/.done').exists())
                r=prepare_checks(inputs,root/'good',{s:s for s in inputs.files},per_source=2)
                Path(r['samples']['path']).write_text('changed')
                with self.assertRaisesRegex(ValueError,'certificate'):verify_checks(inputs,root/'good/manifest.json')
