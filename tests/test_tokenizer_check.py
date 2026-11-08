import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders,normalizers
from tokenizer.check_tokenizer import check, PROBES
from tokenizer.train_tokenizer import SPECIAL


def fixture(root):
    tok=Tokenizer(models.BPE());tok.normalizer=normalizers.NFC()
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator([t for _,t in PROBES],trainers.BpeTrainer(vocab_size=300,special_tokens=SPECIAL,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
    path=root/'tok.json';tok.save(str(path))
    samples=root/'samples.jsonl'
    samples.write_text('\n'.join(json.dumps(dict(category=c,text=t),ensure_ascii=False) for c,t in PROBES[:5]))
    return path,samples,tok.get_vocab_size()


class TokenizerCheckTest(unittest.TestCase):
    def test_multilingual_and_unicode_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tok,samples,n=fixture(root)
            result=check(tok,samples,root/'report.json',n)
            self.assertEqual(result['status'],'needs_review')
            self.assertEqual(result['errors'],[])
            self.assertGreater(result['code_raw_change_count'],0)
            self.assertFalse((root/'report.json.done').exists())
            self.assertTrue(all(x['documents']>0 and x['tokens']>0 for x in result['categories'].values()))
            self.assertEqual(check(tok,samples,root/'report.json',n),result)
            data=json.loads(tok.read_text());data['normalizer']=None;tok.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'changed'):check(tok,samples,root/'report.json',n)

    def test_missing_category_and_special_tokens_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tok,samples,n=fixture(root)
            samples.write_text(json.dumps(dict(category='ko',text='한글')))
            r=check(tok,samples,root/'missing.json',n)
            self.assertEqual(r['status'],'failed');self.assertIn('Missing category: en',r['errors'])
            data=json.loads(tok.read_text());data['added_tokens']=[];tok.write_text(json.dumps(data))
            r=check(tok,samples,root/'special.json',n)
            self.assertTrue(any('non-atomic' in x for x in r['errors']))

    def test_cli_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tok,samples,n=fixture(root)
            p=subprocess.run([sys.executable,'tokenizer/check_tokenizer.py','--tokenizer',str(tok),'--samples',str(samples),
                              '--output',str(root/'cli.json'),'--expected-vocab',str(n)],capture_output=True,text=True)
            self.assertEqual(p.returncode,2,p.stderr)
            samples.write_text(json.dumps(dict(category='invalid',text='test')))
            with self.assertRaisesRegex(ValueError,'Unknown category'):check(tok,samples,root/'bad.json',n)

    def test_raw_policy_preserves_unicode_and_rejects_mislabeled_normalizer(self):
        from tokenizer.train_tokenizer import make_tokenizer
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,samples,_=fixture(root)
            tok=make_tokenizer('raw')
            tok.train_from_iterator([t for _,t in PROBES],trainers.BpeTrainer(vocab_size=300,special_tokens=SPECIAL,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False))
            path=root/'raw.json';tok.save(str(path))
            r=check(path,samples,root/'raw-report.json',tok.get_vocab_size(),'raw')
            self.assertEqual(r['status'],'passed');self.assertEqual(r['code_raw_change_count'],0)
            self.assertTrue((root/'raw-report.json.done').exists())
            self.assertEqual(tok.decode(tok.encode('e\u0301 é 한글').ids,skip_special_tokens=False),'e\u0301 é 한글')
            r=check(path,samples,root/'wrong-policy.json',tok.get_vocab_size(),'NFC')
            self.assertEqual(r['status'],'failed')
