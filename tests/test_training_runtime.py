import copy
import json
from pathlib import Path
import tempfile
import unittest
from train.train import DEFAULT
from train.runtime import training_contract, validate_resume, evaluation_indices, complete_checkpoints


class TrainingRuntimeTest(unittest.TestCase):
    def test_balanced_evaluation_coverage(self):
        for count in [1, 7, 8, 9, 17]:
            batches = [evaluation_indices(count, 8, r) for r in range(8)]
            self.assertEqual(len({len(x) for x in batches}), 1)
            self.assertEqual(sorted(i for batch in batches for i, valid in batch if valid), list(range(count)))
        with self.assertRaises(ValueError): evaluation_indices(0, 8, 0)

    def test_training_identity_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root/'data.bin'; data.write_bytes(b'\x01\x00'*32)
            mix = root/'mix.json'
            payload = dict(tokenizer_sha256='first', sources={'s':dict(weight=1,files=[str(data)])})
            mix.write_text(json.dumps(payload))
            config = dict(DEFAULT, mix=str(mix))
            contract = training_contract(config,8,[])
            validate_resume({'training_contract':contract},contract)
            for key, value in [('seed',42),('max_steps',1001),('sampler_version',1),('global_tokens',4096)]:
                changed = training_contract(dict(config,**{key:value}),8,[])
                with self.assertRaises(ValueError): validate_resume({'training_contract':contract},changed)
            for changed in [training_contract(config,1,[]),training_contract(config,8,[str(data)])]:
                with self.assertRaises(ValueError): validate_resume({'training_contract':contract},changed)
            self.assertEqual(contract,training_contract(dict(config,stop_after=2,out_dir='elsewhere'),8,[]))
            payload['tokenizer_sha256']='second';mix.write_text(json.dumps(payload))
            with self.assertRaises(ValueError): validate_resume({'training_contract':contract},training_contract(config,8,[]))
            payload['tokenizer_sha256']='first';mix.write_text(json.dumps(payload))
            data.write_bytes(b'\x02\x00'*32)  # same size, changed content
            with self.assertRaises(ValueError): validate_resume({'training_contract':contract},training_contract(config,8,[]))
            with self.assertRaisesRegex(ValueError,'lacks training identity'): validate_resume({},contract)

    def test_source_order_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); f=p/'data.bin';f.write_bytes(b'\0\0'*32)
            m=p/'mix.json'; sources={name:dict(weight=.5,files=[str(f)]) for name in ['a','b']}
            m.write_text(json.dumps(dict(sources=sources)))
            c=dict(DEFAULT,mix=str(m)); a=training_contract(c,8,[])
            m.write_text(json.dumps(dict(sources=dict(reversed(list(sources.items()))))))
            self.assertNotEqual(a['id'],training_contract(c,8,[])['id'])

    def test_incomplete_checkpoint_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for step in [1,2,3]: (root/f'ckpt_{step:07d}').mkdir()
            (root/'ckpt_0000002'/'.complete').touch()
            self.assertEqual([p.name for p in complete_checkpoints(root)],['ckpt_0000002'])
