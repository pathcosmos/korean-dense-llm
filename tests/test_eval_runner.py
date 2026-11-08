import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
import torch.distributed.checkpoint as dcp
from tokenizers import Tokenizer,models,pre_tokenizers
from common.data_contracts import sha256_file
from eval.run import run,load_checkpoint
from model.model import Transformer,get_config


def fixture(root,missing_weight=False):
    tok=Tokenizer(models.WordLevel({'<unk>':0,'<|eos|>':1,'hello':2,'world':3,'안녕':4,'세계':5},unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace();tokenizer=root/'tok.json';tok.save(str(tokenizer))
    torch.manual_seed(0);model=Transformer(get_config('micro',vocab_size=6,max_seq_len=8));state=model.state_dict()
    if missing_weight:del state['blocks.0.attn.wq.weight']
    checkpoint=root/'checkpoint';dcp.save({'model':state},checkpoint_id=checkpoint)
    (checkpoint/'meta.json').write_text(json.dumps(dict(step=0,config=dict(model='micro',vocab_size=6,seq_len=8),
                    training_contract=dict(mix=dict(tokenizer_sha256=sha256_file(tokenizer))))))
    (checkpoint/'.complete').touch()
    samples=root/'samples.jsonl';samples.write_text('\n'.join(json.dumps(r) for r in [dict(category='ko',text='안녕 세계'),dict(category='en',text='hello world')]))
    prompts=root/'prompts.jsonl';prompts.write_text('\n'.join(json.dumps(r) for r in [dict(id='ko1',category='ko',text='안녕'),dict(id='en1',category='en',text='hello')]))
    return checkpoint,tokenizer,samples,prompts


class EvalRunnerTest(unittest.TestCase):
    def test_native_end_to_end_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root);output=root/'report.json'
            result=run(*args,output,seq_len=8,stride=2,max_new_tokens=2)
            self.assertEqual(result['status'],'passed');self.assertEqual(result['aggregate']['valid_tokens'],4)
            self.assertEqual(len(result['generations']),2)
            self.assertEqual(result,run(*args,output,seq_len=8,stride=2,max_new_tokens=2))
            p=subprocess.run([sys.executable,'eval/run.py','--checkpoint',str(args[0]),'--tokenizer',str(args[1]),
               '--samples',str(args[2]),'--prompts',str(args[3]),'--output',str(root/'cli.json'),
               '--seq-len','8','--stride','2','--max-new-tokens','2'],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            changed=json.loads(args[1].read_text());changed['model']['vocab']['hello']=3;changed['model']['vocab']['world']=2
            args[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'identity'):run(*args,output,seq_len=8,stride=2,max_new_tokens=2)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):run(*args,root/'changed.json',seq_len=8,stride=2,max_new_tokens=2)

    def test_missing_keys_incomplete_and_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=fixture(root,True)
            with self.assertRaisesRegex(ValueError,'keys differ'):load_checkpoint(args[0],sha256_file(args[1]))
            (args[0]/'.complete').unlink()
            with self.assertRaisesRegex(ValueError,'incomplete'):run(*args,root/'out.json')
            (args[0]/'.complete').touch();args[2].write_text(json.dumps(dict(category='ko',text='안녕')))
            with self.assertRaisesRegex(ValueError,'Both Korean'):run(*args,root/'out.json')
