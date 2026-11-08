import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from tokenizers import Tokenizer,models
from eval.multiple_choice import score_choices,adapt,run
from test_eval_runner import fixture


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('scores',torch.tensor([.1,.4,.6,.2]).log())
    def forward(self,ids):return self.scores.expand(*ids.shape,4)


class MultipleChoiceTest(unittest.TestCase):
    def test_scoring_mask_shift_sum_mean_and_model_state(self):
        tok=Tokenizer(models.BPE({'p':0,'a':1,'b':2,'x':3},[]));m=ConstantModel();m.train()
        sum_result=score_choices(m,tok,'pp',['a','bb'],scoring='sum',context_length=4)
        mean_result=score_choices(m,tok,'pp',['a','bb'],scoring='mean',context_length=4)
        self.assertTrue(m.training)
        self.assertEqual(sum_result['prediction'],0);self.assertEqual(mean_result['prediction'],1)
        for result in [sum_result,mean_result]:
            self.assertEqual([c['target_tokens'] for c in result['choices']],[1,2])
            self.assertAlmostEqual(result['choices'][0]['log_probability'],math.log(.4/1.3),places=6)
            self.assertAlmostEqual(result['choices'][1]['log_probability'],2*math.log(.6/1.3),places=6)
        tied=score_choices(m,tok,'pp',['a','a'],scoring='sum',context_length=4)
        self.assertEqual(tied['tied_indices'],[0,1]);self.assertEqual(tied['prediction'],0)

    def test_boundary_overflow_nonfinite_and_adapter_rejects(self):
        tok=Tokenizer(models.BPE({'p':0,'a':1,'b':2,'x':3,'pa':4},[('p','a')]))
        with self.assertRaisesRegex(ValueError,'boundary'):score_choices(ConstantModel(),tok,'p',['a','b'],scoring='sum',context_length=4)
        tok=Tokenizer(models.BPE({'p':0,'a':1,'b':2,'x':3},[]))
        with self.assertRaisesRegex(ValueError,'context'):score_choices(ConstantModel(),tok,'pp',['a','bb'],scoring='sum',context_length=2)
        m=ConstantModel();m.scores.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):score_choices(m,tok,'p',['a','b'],scoring='sum',context_length=4)
        task=dict(schema_version=1,name='fixture',revision='1',template='{question}\n{choices}\nAnswer:\n',target='label',scoring='sum')
        row=dict(id='x',question='문제',choices=['첫째','둘째'],answer=1)
        actual=adapt(row,task);self.assertEqual(actual['choices'],['A','B']);self.assertIn('B. 둘째',actual['prompt'])
        for changed in [dict(answer=True),dict(answer=2),dict(choices=['동일','동일'])]:
            with self.assertRaises(ValueError):adapt({**row,**changed},task)
        with self.assertRaisesRegex(ValueError,'display'):adapt(row,{**task,'template':'{question}'})

    def test_native_checkpoint_cli_and_task_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint,tok,_,_=fixture(root)
            task=root/'task.json';task.write_text(json.dumps(dict(schema_version=1,name='synthetic',revision='fixture-v1',
                template='{question} ',target='choice_text',scoring='sum')))
            examples=root/'examples.jsonl';examples.write_text('\n'.join(json.dumps(r) for r in [
                dict(id='en',question='hello',choices=['world','세계'],answer=0),
                dict(id='ko',question='안녕',choices=['world','세계'],answer=1)]))
            out=root/'report.json';r=run(checkpoint,tok,task,examples,out)
            self.assertEqual(r['examples'],2);self.assertEqual(r['accuracy'],r['correct']/2)
            self.assertEqual(r,run(checkpoint,tok,task,examples,out))
            command=[sys.executable,'eval/multiple_choice.py','--checkpoint',str(checkpoint),'--tokenizer',str(tok),
                '--task',str(task),'--examples',str(examples),'--output',str(root/'cli.json')]
            p=subprocess.run(command,capture_output=True,text=True,timeout=30);self.assertEqual(p.returncode,0,p.stderr)
            data=json.loads(task.read_text());data['scoring']='mean';task.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'identity'):run(checkpoint,tok,task,examples,out)
            examples.write_text(examples.read_text()+'\n'+examples.read_text().splitlines()[0])
            with self.assertRaisesRegex(ValueError,'unique'):run(checkpoint,tok,task,examples,root/'bad.json')
