import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from train.data import MixSampler, validate_sampler_resume

CASES = {x['id']: x for x in json.loads((Path(__file__).resolve().parents[1] / 'docs/reviews/frankenstallm/acceptance_cases.json').read_text())['cases']}


class SamplerTest(unittest.TestCase):
    def make_mix(self, root, tokens, weight=1):
        file = root / 'tokens.bin';np.array(tokens, dtype=np.uint16).tofile(file)
        mix=root/'mix.json';mix.write_text(json.dumps({'sources':{'a':{'files':[str(file)],'weight':weight}}}))
        return mix, file

    def test_D01_single_window(self):
        c=CASES['D01']
        with tempfile.TemporaryDirectory() as tmp:
            mix,_=self.make_mix(Path(tmp), c['inputs']['tokens'])
            sampler=MixSampler(str(mix),c['inputs']['seq_len'])
            x,y=sampler.batch(0,0,0,1)
            self.assertEqual(x[0].tolist(), c['expected']['input_ids'])
            self.assertEqual(y[0].tolist(), c['expected']['targets'])
            self.assertEqual(sampler.window_counts[0].tolist(),[1])

    def test_D02_controlled_last_offset(self):
        c=CASES['D02']
        class LastRng:
            def choice(self,*args,**kwargs):return 0
            def integers(self,low,high):
                self_high=high
                assert low==0 and self_high==2
                return high-1
        with tempfile.TemporaryDirectory() as tmp:
            mix,_=self.make_mix(Path(tmp), c['inputs']['tokens'])
            sampler=MixSampler(str(mix),c['inputs']['seq_len'])
            with patch('train.data.np.random.default_rng',return_value=LastRng()):
                x,y=sampler.batch(0,0,0,1)
            self.assertEqual(x[0].tolist(),c['expected']['last_window_input'])
            self.assertEqual(y[0].tolist(),c['expected']['last_window_targets'])

    def test_invalid_shards_weights_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for weight in [0,-1,float('nan'),float('inf')]:
                mix,_=self.make_mix(root,list(range(20)),weight)
                with self.assertRaises(ValueError):MixSampler(str(mix),4)
            for tokens in [[],[1,2]]:
                mix,_=self.make_mix(root,tokens)
                with self.assertRaises(ValueError):MixSampler(str(mix),4)
            mix,file=self.make_mix(root,list(range(20)))
            sampler=MixSampler(str(mix),4)
            a=sampler.batch(20,1,2,8);b=sampler.batch(20,1,2,8)
            self.assertTrue(np.array_equal(a[0],b[0]))
            file.write_bytes(b'123')
            with self.assertRaises(ValueError):MixSampler(str(mix),4)

    def test_valid_window_weights_and_repeated_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mix,small=self.make_mix(root,list(range(5)))
            big=root/'big.bin';np.arange(10,dtype=np.uint16).tofile(big)
            mix.write_text(json.dumps({'sources':{'a':{'weight':1,'files':[str(small),str(big)]}}}))
            sampler=MixSampler(str(mix),4)
            self.assertEqual(sampler.window_counts[0].tolist(),[1,6])
            self.assertTrue(np.allclose(sampler.shard_probabilities[0],[1/7,6/7]))
            mix.write_text(json.dumps({'sources':{'a':{'weight':1,'files':[str(big),str(big)]}}}))
            with self.assertRaisesRegex(ValueError,'Repeated shard'):MixSampler(str(mix),4)

    def test_legacy_sequence_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            mix,_=self.make_mix(Path(tmp),list(range(20)))
            sampler=MixSampler(str(mix),4,seed=7,version=1)
            rng=np.random.default_rng([7,2,0,3]);expect=[]
            for _ in range(5):
                rng.choice(1,p=[1]);rng.choice(1,p=[1]);o=int(rng.integers(0,15));expect.append(list(range(o,o+4)))
            self.assertEqual(sampler.batch(2,0,3,5)[0].tolist(),expect)
        with self.assertRaisesRegex(ValueError,'version mismatch'):validate_sampler_resume({'config':{}},2)
        validate_sampler_resume({'config':{}},1)
        validate_sampler_resume({'sampler_version':2},2)
