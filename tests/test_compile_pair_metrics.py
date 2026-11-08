import unittest
import torch
from scripts.validate_compile_pair import metrics,combine,judge
class PairMetricTest(unittest.TestCase):
    def test_weighted_reference_and_zero(self):
        m=combine([metrics(torch.tensor([3.,4.]),torch.tensor([0.,4.])),metrics(torch.zeros(1),torch.zeros(1))])
        self.assertAlmostEqual(m['rel_l2'],.75);self.assertEqual(m['max_abs'],3.)
        self.assertEqual(combine([metrics(torch.zeros(2),torch.zeros(2))])['rel_l2'],0.)
    def test_nonfinite_and_failed_component_are_not_passed(self):
        good=combine([metrics(torch.ones(2),torch.ones(2))]);r=dict(logits=good,gradient=good,update=good,loss_abs=0.)
        self.assertTrue(judge(r));r['gradient']=combine([metrics(torch.tensor([float('nan')]),torch.ones(1))]);self.assertFalse(judge(r))
        r['gradient']=good;r['update']=dict(good,rel_l2=.11);self.assertFalse(judge(r))
