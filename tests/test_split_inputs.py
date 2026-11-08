import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed.checkpoint as dcp
from common.data_contracts import atomic_json,identity,sha256_file
from data.split_manifest import build
from data.split_inputs import prepare,verify_prepared,SplitInputs
from tokenizer.train_tokenizer import train_prepared
from eval.run import run
from model.model import Transformer,get_config


def fixture(root):
    sources={}
    for name in ['ko','en','code']:
        path=root/(name+'.parquet')
        texts=[('한글 테스트' if name=='ko' else 'English document' if name=='en' else 'def f(x):\n    return x')+f' {name} {i} e\u0301' for i in range(80)]
        table={'text':texts}
        if name=='code':table['repo_name']=[f'org/repo{i//4}' for i in range(80)]
        pq.write_table(pa.table(table),path,row_group_size=10);Path(str(path)+'.done').touch()
        sources[name]=dict(files=[str(path)],text_policy='raw-utf8-code-v1' if name=='code' else 'nfc-web-exact-v1',completion_marker=str(path)+'.done')
    manifest=root/'sources.json';atomic_json(manifest,dict(schema_version=1,status='ready',sources=sources))
    build(manifest,root/'split',.4,7)
    return manifest,root/'split/manifest.json'


class SplitInputsTest(unittest.TestCase):
    def test_train_eval_separation_reproducibility_and_small_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,split=fixture(root)
            train=prepare(manifest,split,root/'train',dict(ko=120,en=120,code=120),mode='train',max_chars=20,seed=3)
            hold=prepare(manifest,split,root/'eval',dict(ko=120,en=120),mode='eval',categories=dict(ko='ko',en='en'),seed=3)
            self.assertEqual(verify_prepared(root/'train/manifest.json','train'),train)
            self.assertEqual(verify_prepared(root/'eval/manifest.json'),hold)
            t=[json.loads(x) for x in Path(train['samples']['path']).read_text().splitlines()]
            e=[json.loads(x) for x in Path(hold['samples']['path']).read_text().splitlines()]
            self.assertTrue(set(x['group_id'] for x in t).isdisjoint(x['group_id'] for x in e))
            self.assertTrue(all(x['split']=='train' and len(x['text'])<=20 for x in t))
            self.assertTrue(any(x['truncated'] for x in t))
            self.assertTrue(all(not x['truncated'] for x in e))
            other=prepare(manifest,split,root/'train2',dict(ko=120,en=120,code=120),mode='train',max_chars=20,seed=3)
            self.assertEqual(train['samples']['sha256'],other['samples']['sha256'])
            self.assertEqual(prepare(manifest,split,root/'train',dict(ko=120,en=120,code=120),mode='train',max_chars=20,seed=3),train)
            tok=train_prepared(root/'train/manifest.json',root/'tok.json',272)
            self.assertEqual(tok['config']['split_contract_id'],train['contract']['split_contract_id'])
            self.assertEqual(train_prepared(root/'train/manifest.json',root/'tok.json',272),tok)

    def test_bad_budgets_changed_sources_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,split=fixture(root)
            with self.assertRaisesRegex(ValueError,'Insufficient'):prepare(manifest,split,root/'short',dict(ko=1000000),mode='train')
            with self.assertRaisesRegex(ValueError,'Partial'):prepare(manifest,split,root/'short',dict(ko=1),mode='train')
            with self.assertRaisesRegex(ValueError,'ko/en'):prepare(manifest,split,root/'bad',dict(ko=20),mode='eval',categories=dict(ko='ko'))
            prepare(manifest,split,root/'train',dict(ko=120),mode='train')
            with self.assertRaisesRegex(ValueError,'contract changed'):prepare(manifest,split,root/'train',dict(ko=120),mode='train',seed=2)
            pq.write_table(pa.table({'text':['changed']}),root/'ko.parquet')
            with self.assertRaisesRegex(ValueError,'files/content'):SplitInputs(manifest,split)

    def test_holdout_mutation_membership_and_certificates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,split=fixture(root)
            r=prepare(manifest,split,root/'eval',dict(ko=120,en=120),mode='eval',categories=dict(ko='ko',en='en'))
            path=Path(r['samples']['path']);original=path.read_text();path.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'content changed'):verify_prepared(root/'eval/manifest.json')
            path.write_text(original)
            with self.assertRaisesRegex(ValueError,'path mismatch'):verify_prepared(root/'eval/manifest.json',samples_path=root/'wrong.jsonl')
            # Re-certify a deliberately wrong row to prove membership, not only checksums, is checked.
            rows=[json.loads(x) for x in original.splitlines()];inputs=SplitInputs(manifest,split)
            wrong=next(inputs.rows('ko','train'));inputs.close();rows[0].update(wrong)
            rows[0]['text_sha256']=rows[0]['content_sha256']
            path.write_text(''.join(json.dumps(x)+'\n' for x in rows));r['samples']['sha256']=sha256_file(path)
            atomic_json(root/'eval/manifest.json',r)
            atomic_json(root/'eval/.done',dict(contract_id=r['contract_id'],manifest_sha256=sha256_file(root/'eval/manifest.json')))
            with self.assertRaisesRegex(ValueError,'requested split'):verify_prepared(root/'eval/manifest.json')

    def test_native_eval_and_holdout_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest,split=fixture(root)
            p=subprocess.run([sys.executable,'eval/prepare_holdout.py','--sources-manifest',str(manifest),
                '--split-manifest',str(split),'--source','ko:ko:40','--source','en:en:40','--output-dir',str(root/'eval')],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            prepare(manifest,split,root/'train',dict(ko=120,en=120),mode='train',max_chars=20)
            train_prepared(root/'train/manifest.json',root/'tok.json',272)
            checkpoint=root/'checkpoint';model=Transformer(get_config('micro',vocab_size=272,max_seq_len=128))
            dcp.save({'model':model.state_dict()},checkpoint_id=checkpoint)
            atomic_json(checkpoint/'meta.json',dict(step=0,config=dict(model='micro',vocab_size=272,seq_len=128),
                training_contract=dict(mix=dict(tokenizer_sha256=sha256_file(root/'tok.json')))))
            (checkpoint/'.complete').touch();prompts=root/'prompts.jsonl'
            prompts.write_text('\n'.join(json.dumps(dict(id=c,category=c,text=t)) for c,t in [('ko','한글'),('en','English')]))
            r=run(checkpoint,root/'tok.json',root/'eval/samples.jsonl',prompts,root/'result.json',
                  seq_len=64,stride=32,max_new_tokens=1,holdout_manifest=root/'eval/manifest.json')
            self.assertEqual(r['contract']['input_provenance']['level'],'verified-eval-split')
