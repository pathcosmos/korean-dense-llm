from pathlib import Path
import tempfile
import unittest
import torch
from data.parallel_tokenize import run
from train.data import HoldoutSet
from train.probe import select_probe,verify_probe,evaluate_sources
from test_parallel_tokenize import fixture


class TrainingProbeTest(unittest.TestCase):
    def test_actual_native_probe_bounded_immutable_and_experimental_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sm,split,tok=fixture(root)
            run(sm,split,tok,root/'tokens',workers=2,experimental=True)
            probe=select_probe(root/'tokens',root/'probe',dict(web='ko',code='en'),seq_len=8,windows_per_file=4,require_production=False)
            self.assertEqual(len(probe['holdout']),2);self.assertEqual(probe['total_windows'],8)
            self.assertGreater(probe['baseline_all_file_windows'],probe['total_windows'])
            self.assertEqual(len(HoldoutSet(probe['holdout'],8,n_windows=4)),8)
            self.assertEqual(verify_probe(root/'probe',require_production=False),probe)
            combined=select_probe(root/'tokens',root/'combined',dict(web='ko',code='en'),seq_len=8,windows_per_file=20,require_production=False)
            self.assertGreater(next(g['count'] for g in combined['groups'] if g['source']=='code'),20)
            self.assertEqual(len(HoldoutSet(combined['holdout'],8,n_windows=20)),combined['total_windows'])
            with self.assertRaisesRegex(ValueError,'Experimental'):verify_probe(root/'probe')
            path=Path(probe['holdout'][0]);path=path if path.is_absolute() else Path.cwd()/path
            with path.open('r+b') as stream:b=stream.read(1);stream.seek(0);stream.write(bytes([b[0]^1]))
            with self.assertRaisesRegex(ValueError,'changed'):verify_probe(root/'probe',require_production=False)

    def test_source_category_nll_weighting_and_complete_target_coverage(self):
        class Hold:
            def __init__(self):self.wins=torch.tensor([[1,0,0],[2,0,0],[3,0,0],[4,0,0]])
            def __len__(self):return len(self.wins)
        hold=Hold();before=hold.wins.clone();probe=dict(seq_len=2,total_windows=4,scope='test fixed probe',groups=[dict(source='a',category='ko',start=0,count=1),dict(source='b',category='en',start=1,count=3)])
        def original(model,fwd,subset,device):
            nll=float(subset.wins[:,0].sum())*2;n=len(subset)*2
            return dict(nll_sum=nll,valid_tokens=n,holdout_loss=nll/n,forwards_per_rank=len(subset))
        r=evaluate_sources(original,None,None,hold,None,probe)
        self.assertEqual(r['holdout_loss'],2.5);self.assertEqual(r['valid_tokens'],8)
        self.assertEqual(r['by_category']['ko']['holdout_loss'],1.);self.assertEqual(r['by_category']['en']['holdout_loss'],3.)
        self.assertTrue(torch.equal(hold.wins,before))
        merged=dict(probe,groups=[dict(g,category='ko') for g in probe['groups']])
        self.assertEqual(evaluate_sources(original,None,None,hold,None,merged)['by_category']['ko']['holdout_loss'],2.5)
        for bad in [dict(probe,total_windows=5),dict(probe,groups=[dict(source='a',category='ko',start=0,count=0)]),dict(probe,groups=[dict(source='a',category='ko',start=0,count=3)])]:
            with self.assertRaises(ValueError):evaluate_sources(original,None,None,hold,None,bad)
