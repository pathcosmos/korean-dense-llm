import unittest
import torch
from scripts.diagnose_compile_precision import groups
from model.model import Transformer,get_config
class PrecisionGroupTest(unittest.TestCase):
    def test_tied_weights_once_and_norm_no_decay(self):
        m=Transformer(get_config('micro',vocab_size=32,tie_embeddings=True,max_seq_len=4));g=groups(m)
        ids=[id(p) for group in g for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in m.parameters()})
        self.assertTrue(all(p.ndim>=2 for p in g[0]['params']));self.assertTrue(all(p.ndim<2 for p in g[1]['params']));self.assertEqual(g[1]['weight_decay'],0.)
    def test_zero_gradient_decays_matrix_only(self):
        m=torch.nn.Linear(3,2);w=m.weight.detach().clone();b=m.bias.detach().clone()
        opt=torch.optim.AdamW(groups(m),lr=.01)
        for p in m.parameters():p.grad=torch.zeros_like(p)
        opt.step();torch.testing.assert_close(m.weight,w*.999);torch.testing.assert_close(m.bias,b,rtol=0,atol=0)
