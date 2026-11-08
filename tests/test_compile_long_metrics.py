import unittest
import torch
from scripts.validate_compile_long import tensor_metric,summarize_metrics
from scripts.validate_compile_pair import judge


class LongCompileMetricTest(unittest.TestCase):
    def test_global_weights_and_nonfinite_reference(self):
        m=summarize_metrics([tensor_metric(torch.tensor([3.,4.]),torch.tensor([0.,4.])),tensor_metric(torch.zeros(2),torch.zeros(2))])
        self.assertEqual(m['rel_l2'],.75)
        self.assertFalse(summarize_metrics([tensor_metric(torch.ones(1),torch.tensor([float('nan')]))])['finite'])
        r=dict(logits=m,gradient=m,update=m,loss_abs=0.)
        self.assertFalse(judge(r))
