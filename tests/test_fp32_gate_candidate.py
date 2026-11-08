import copy,unittest
import torch
from model.model import FeedForward,get_config
from scripts.probe_full_fp32_gate import fp32_gate
class FP32GateCandidateTest(unittest.TestCase):
    def test_fp32_forward_and_gradients_preserved(self):
        torch.manual_seed(3);m=FeedForward(get_config('micro'));n=copy.deepcopy(m);x=torch.randn(2,7,64)
        a=m(x);b=fp32_gate(n,x);torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().sum().backward();b.square().sum().backward()
        for p,q in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    def test_bf16_shape_dtype_and_finite_gradients(self):
        m=FeedForward(get_config('micro'));x=torch.randn(1,3,64)
        with torch.autocast('cpu',dtype=torch.bfloat16):y=fp32_gate(m,x)
        self.assertEqual(y.shape,x.shape);self.assertEqual(y.dtype,torch.bfloat16);y.float().square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
