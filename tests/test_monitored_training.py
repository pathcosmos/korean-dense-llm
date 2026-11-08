import copy
import json
from pathlib import Path
import tempfile
import unittest

from common.data_contracts import atomic_json
from train.monitored import validate_configuration
from scripts.validate_monitored_training import seal_checkpoint, verify_generations


class MonitoredTrainingTest(unittest.TestCase):
    def configuration(self):
        return dict(model='3b',vocab_size=65536,seq_len=4096,micro_bsz=4,global_tokens=2097152,
                    max_steps=4,compile=False,sampler_version=2,lr=.0003,keep_ckpts=5,stop_after=None,
                    eval_every=2,ckpt_every=2)

    def test_bounded_execution_and_main_pruning_authorizations_are_separate(self):
        c=self.configuration();contract=dict(purpose='bounded_validation',generation_steps=[2,4])
        self.assertEqual(validate_configuration(c,contract)['accumulation'],16)
        for change in [dict(max_steps=57000),dict(keep_ckpts=2),dict(compile=True),dict(micro_bsz=8),dict(stop_after=3)]:
            with self.assertRaises(ValueError):validate_configuration(dict(c,**change),contract)
        main=dict(c,max_steps=57000,keep_ckpts=2)
        purpose=dict(purpose='main_training',generation_steps=[2,4])
        with self.assertRaisesRegex(ValueError,'user authorization'):validate_configuration(main,purpose)
        purpose['main_training_authorized']=True
        with self.assertRaisesRegex(ValueError,'pruning'):validate_configuration(main,purpose)
        purpose['new_checkpoint_pruning_authorized']=True
        self.assertEqual(validate_configuration(main,purpose)['target_tokens'],119537664000)
        with self.assertRaisesRegex(ValueError,'boundaries'):
            validate_configuration(c,dict(contract,generation_steps=[3]))

    def test_checkpoint_content_seal_preserves_and_detects_corruption(self):
        import torch
        import torch.distributed.checkpoint as dcp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ckpt=root/'ckpt_0000002'
            dcp.save({'model':{'weight':torch.arange(4.)}},checkpoint_id=ckpt)
            atomic_json(ckpt/'meta.json',dict(step=2,world_size=8,tokens_per_step=2097152,
                        config=dict(model='3b',max_steps=4),training_contract=dict(id='fixture')))
            (ckpt/'.complete').touch()
            for rank in range(8):(ckpt/f'rng_rank{rank}.pt').write_bytes(bytes([rank]))
            (ckpt/'payload').write_bytes(b'unchanged')
            proof=root/'certificate.json';r=seal_checkpoint(ckpt,proof,'fixture')
            self.assertEqual(r,seal_checkpoint(ckpt,proof,'fixture'))
            (ckpt/'payload').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'changed'):seal_checkpoint(ckpt,proof,'fixture')
            self.assertEqual((ckpt/'payload').read_bytes(),b'corrupted')
            payload=next(ckpt.glob('*.distcp'));payload.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'truncated'):seal_checkpoint(ckpt,proof,'fixture')
            (ckpt/'rng_rank7.pt').unlink()
            with self.assertRaisesRegex(ValueError,'identity'):seal_checkpoint(ckpt,proof,'fixture')

    def test_generation_requires_both_scheduled_steps_all8ranks_and_consistent_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for rank in range(8):
                p=root/str(rank);paths.append(p)
                atomic_json(p,dict(rank=rank,generations=[dict(step=s,sha256='step'+str(s)) for s in [2,4]]))
            self.assertEqual(verify_generations(paths+[paths[0]],[2,4]),dict({'2':'step2','4':'step4'}))
            with self.assertRaisesRegex(ValueError,'coverage'):verify_generations(paths[:-1],[2,4])
            d=json.loads(paths[-1].read_text());d['generations'][0]['sha256']='different';atomic_json(paths[-1],d)
            with self.assertRaisesRegex(ValueError,'decisions'):verify_generations(paths,[2,4])
            original=json.loads(paths[0].read_text());original['generations'][0]['sha256']='replay_changed'
            p=root/'replayed';atomic_json(p,original)
            with self.assertRaisesRegex(ValueError,'Replayed'):verify_generations(paths+[p],[2,4])


if __name__=='__main__':unittest.main()
