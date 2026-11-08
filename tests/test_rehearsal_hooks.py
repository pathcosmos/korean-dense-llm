import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from train.data import MixSampler
from train.rehearsal import SourceLedger,observed_sampler,aggregate_ledgers,distributed_generate,step_and_record


class RehearsalHooksTest(unittest.TestCase):
    def test_real_dcp_state_initialization_is_not_a_committed_training_update(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict
        model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters(),lr=.001);ledger=SourceLedger(['a'])
        before=[p.detach().clone() for p in model.parameters()];original=torch.optim.AdamW.step
        def wrapped(opt,*args,**kwargs):return step_and_record(original,ledger,opt,*args,**kwargs)
        with patch.object(torch.optim.AdamW,'step',wrapped):
            get_state_dict(model,optimizer)
            self.assertEqual(ledger.optimizer_initializations,1);self.assertFalse(ledger.steps)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,model.parameters())))
            ledger.begin(0);ledger.record('a');model(torch.ones(1,2)).sum().backward();optimizer.step()
            self.assertEqual(ledger.steps,[dict(step=1,source_sequences={'a':1})]);self.assertEqual(ledger.optimizer_initializations,1)
            with self.assertRaisesRegex(ValueError,'Unexpected optimizer step'):optimizer.step()

    def test_observation_keeps_actual_sampler_bytes_and_rng_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources={}
            for i,name in enumerate(['a','b']):
                p=root/(name+'.bin');(np.arange(128,dtype=np.uint16)+i*128).tofile(p)
                sources[name]=dict(weight=.3 if i==0 else .7,files=[str(p)])
            mix=root/'mix.json';mix.write_text(json.dumps(dict(sampler_version=2,sources=sources)))
            ledger=SourceLedger(sources);plain=MixSampler(str(mix),8,42);observed=observed_sampler(MixSampler,ledger)(str(mix),8,42)
            for step in range(3):
                for micro in range(4):
                    a=plain.batch(step,0,micro,3);b=observed.batch(step,0,micro,3)
                    self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
                self.assertEqual(sum(ledger.pending.values()),12);ledger.commit()
            self.assertEqual([r['step'] for r in ledger.steps],[1,2,3]);self.assertFalse(ledger.pending)
            observed.batch(3,0,0,1)
            with self.assertRaisesRegex(ValueError,'Uncommitted'):observed.batch(4,0,0,1)
            # Sampled but not optimizer-committed work never enters the report.
            self.assertEqual(len(ledger.steps),3)

    def test_actual_counts_require_all_steps_ranks_and_consistent_replays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for rank in range(2):
                p=root/str(rank);paths.append(p)
                p.write_text(json.dumps(dict(rank=rank,contract_id='same',steps=[dict(step=i,source_sequences={'a':1,'b':2}) for i in [1,2,3]],generations=[])))
            result=aggregate_ledgers(paths,world=2,steps=3,seq_len=8,global_tokens=48,sources=['a','b'])
            self.assertEqual(result['source_tokens'],dict(a=48,b=96));self.assertEqual(result['replayed_tokens'],0)
            empty=root/'unstarted';empty.write_text(json.dumps(dict(rank=0,contract_id=None,steps=[],generations=[])))
            self.assertEqual(result,aggregate_ledgers(paths+[empty],world=2,steps=3,seq_len=8,global_tokens=48,sources=['a','b']))
            duplicated=aggregate_ledgers(paths+[paths[0]],world=2,steps=3,seq_len=8,global_tokens=48,sources=['a','b'])
            self.assertEqual(duplicated['committed_tokens'],144);self.assertEqual(duplicated['replayed_tokens'],72)
            with self.assertRaisesRegex(ValueError,'Incomplete'):aggregate_ledgers(paths[:1],world=2,steps=3,seq_len=8,global_tokens=48,sources=['a','b'])
            changed=json.loads(paths[0].read_text());changed['steps'][0]['source_sequences']={'a':2,'b':1};p=root/'changed';p.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'choices changed'):aggregate_ledgers(paths+[p],world=2,steps=3,seq_len=8,global_tokens=48,sources=['a','b'])

    def test_generation_uses_broadcast_eos_and_restores_model_mode(self):
        class Model(torch.nn.Module):
            def forward(self,x):return torch.zeros((1,x.shape[1],8))
        model=Model();decisions=iter([3,1]);inputs=[]
        def broadcast(token,src):self.assertEqual(src,0);token.fill_(next(decisions))
        def forward(x):inputs.append(x.tolist());return model(x)
        with patch('torch.distributed.broadcast',side_effect=broadcast),patch('torch.distributed.all_reduce'):
            result=distributed_generate(model,forward,[5,6,7],eos_id=1,context_length=2,max_new_tokens=4,device='cpu')
        self.assertEqual(result['generated_ids'],[3,1]);self.assertTrue(result['stopped_on_eos']);self.assertTrue(model.training)
        self.assertEqual(inputs,[[[6,7]],[[7,3]]])
        model.eval()
        with patch('torch.distributed.all_reduce'),self.assertRaisesRegex(FloatingPointError,'generation logits'):
            distributed_generate(model,lambda x:torch.full((1,1,8),float('nan')),[2],eos_id=1,context_length=2,max_new_tokens=1,device='cpu')
        self.assertFalse(model.training)

    def test_generation_receipts_require_every_rank_and_identical_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for rank in range(2):
                p=root/str(rank);paths.append(p)
                p.write_text(json.dumps(dict(rank=rank,contract_id='same',steps=[dict(step=i,source_sequences={'a':3}) for i in range(1,5)],
                                            generations=[dict(step=4,sha256='same-generated-tokens')])) )
            kwargs=dict(world=2,steps=4,seq_len=8,global_tokens=48,sources=['a'])
            self.assertEqual(aggregate_ledgers(paths,**kwargs)['generation_sha256_by_step'],{'4':'same-generated-tokens'})
            changed=json.loads(paths[1].read_text());changed['generations'][0]['sha256']='different';paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'agreement'):aggregate_ledgers(paths,**kwargs)
            changed['generations']=[];paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'coverage'):aggregate_ledgers(paths,**kwargs)


if __name__=='__main__':unittest.main()
