import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers
from common.data_contracts import source_files
from data.tokenize_shards import uint16_ids, tokenize_source
from data.build_mix import build_mix

CASES={x['id']:x for x in json.loads((Path(__file__).resolve().parents[1]/'docs/reviews/frankenstallm/acceptance_cases.json').read_text())['cases']}


class TokenContractTest(unittest.TestCase):
    def tokenizer(self,path,swap=False):
        c=CASES['T02']['inputs']['tokenizer_b' if swap else 'tokenizer_a']
        tok=Tokenizer(models.WordLevel(dict(c, **{'<|eos|>':3}),unk_token='<unk>'))
        tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(path));return path

    def fixture(self,root):
        f=root/'input.parquet';pq.write_table(pa.table({'text':['a b a b','b a b a','a a a a']}),f)
        Path(str(f)+'.done').touch();tok=self.tokenizer(root/'tok.json')
        return f,tok

    def test_T01_range_before_cast(self):
        c=CASES['T01']['inputs']
        self.assertEqual(uint16_ids(c['valid_ids']).tolist(),c['valid_ids'])
        for v in c['invalid_ids']:
            with self.assertRaises(ValueError):uint16_ids([v])
        with self.assertRaises(ValueError):uint16_ids([1.5])

    def test_split_resume_and_T02_cache_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f,tok=self.fixture(root);out=root/'tokens'
            totals=tokenize_source('a',tok,[f],out,holdout_docs=1,batch_size=2)
            self.assertEqual(totals,dict(tokens=10,holdout_tokens=5,train_docs=2,holdout_docs=1))
            self.assertEqual(np.fromfile(out/'input.bin',dtype=np.uint16).tolist(),[1,2,1,2,3,2,1,2,1,3])
            self.assertEqual(np.fromfile(out/'hold.bin',dtype=np.uint16).tolist(),[1,1,1,1,3])
            before=(out/'input.bin').stat().st_mtime_ns
            (out/'input.bin.done').unlink()  # crash after outputs, before completion marker
            self.assertEqual(tokenize_source('a',tok,[f],out,1,1),totals)
            self.assertEqual((out/'input.bin').stat().st_mtime_ns,before)
            self.tokenizer(tok,swap=True)
            with self.assertRaisesRegex(ValueError,'contract changed'):tokenize_source('a',tok,[f],out,1)

    def test_input_change_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f,tok=self.fixture(root);out=root/'tokens'
            tokenize_source('a',tok,[f],out)
            data=(out/'input.bin').read_bytes();(out/'input.bin').write_bytes(b'xx'+data[2:])
            with self.assertRaisesRegex(ValueError,'corrupt'):tokenize_source('a',tok,[f],out)
            with self.assertRaisesRegex(ValueError,'exceed'):tokenize_source('a',tok,[f],root/'other',4)

    def test_source_manifest_and_valid_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f,tok=self.fixture(root);token_root=root/'tokens'
            marker=root/'complete';marker.touch()
            manifest=root/'sources.json';manifest.write_text(json.dumps({'schema_version':1,'status':'ready','sources':{'a':dict(files=[str(f)],text_policy='test',completion_marker=str(marker))}}))
            files,snapshot=source_files('a',manifest)
            tokenize_source('a',tok,files,token_root/'a',1,snapshot=snapshot)
            with patch('data.build_mix.PROJECT_ROOT',str(root)):
                mix=build_mix({'a':1},token_root,root/'mix.json',seq_len=4,target_tokens=100)
            self.assertEqual(mix['sources']['a']['tokens'],10)
            self.assertEqual(mix['sources']['a']['expected_exposures'],10)
            self.assertEqual(mix['holdout'],['tokens/a/hold.bin'])
            draft=json.loads(manifest.read_text());draft['status']='draft';manifest.write_text(json.dumps(draft))
            with self.assertRaisesRegex(ValueError,'draft'):source_files('a',manifest)
            draft['status']='ready';manifest.write_text(json.dumps(draft))
            marker.unlink()
            with self.assertRaisesRegex(ValueError,'not complete'):source_files('a',manifest)

    def test_changed_input_content_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f,tok=self.fixture(root);out=root/'tokens'
            tokenize_source('a',tok,[f],out)
            pq.write_table(pa.table({'text':['b b b b','b a b a','a a a a']}),f)
            with self.assertRaisesRegex(ValueError,'contract changed'):
                tokenize_source('a',tok,[f],out)

    def test_tokenizer_sample_counts_actual_truncated_bytes(self):
        from tokenizer.train_tokenizer import sample_texts
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'sample.parquet'
            pq.write_table(pa.table({'text':['한'*25000]}),f)
            stats={};samples=list(sample_texts([f],1,stats=stats))
            self.assertEqual(samples,['한'*20000])
            self.assertEqual(stats['utf8_bytes'],len(samples[0].encode()))

    def test_different_tokenizer_mix_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f,tok=self.fixture(root);token_root=root/'tokens'
            tokenize_source('a',tok,[f],token_root/'a')
            self.tokenizer(tok,True);tokenize_source('b',tok,[f],token_root/'b')
            with patch('data.build_mix.PROJECT_ROOT',str(root)):
                with self.assertRaisesRegex(ValueError,'different tokenizer'):build_mix({'a':1,'b':1},token_root,root/'mix.json',4)
