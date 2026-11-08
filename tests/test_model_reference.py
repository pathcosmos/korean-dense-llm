"""Independent explicit attention equations: no production norm/RoPE/SDPA helpers."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from model.model import ModelConfig,Transformer,apply_rope,precompute_rope
from train.data import MixSampler


def reference(model,ids):
    c=model.c;hd=c.dim//c.n_heads;t=ids.shape[1]
    weights=dict(model.named_parameters())
    def linear(x,name):return x @ weights[name].T
    def norm(x,name):return x / torch.sqrt(torch.mean(x*x,dim=-1,keepdim=True)+c.norm_eps)*weights[name]
    def rotate(x):
        # Each half-pair is multiplied by its independently constructed 2x2 rotation.
        pairs=[]
        for j in range(hd//2):
            angles=torch.tensor([p*c.rope_theta**(-2*j/hd) for p in range(t)],dtype=x.dtype)
            a,b=x[...,j],x[...,j+hd//2]
            pairs.append((a*angles.cos()[None,:,None]-b*angles.sin()[None,:,None],
                          a*angles.sin()[None,:,None]+b*angles.cos()[None,:,None]))
        return torch.stack([p[0] for p in pairs]+[p[1] for p in pairs],dim=-1)
    x=weights['tok.weight'][ids]
    for i in range(c.n_layers):
        p=f'blocks.{i}.';h=norm(x,p+'attn_norm.weight')
        q=rotate(linear(h,p+'attn.wq.weight').reshape(*ids.shape,c.n_heads,hd))
        k=rotate(linear(h,p+'attn.wk.weight').reshape(*ids.shape,c.n_kv_heads,hd))
        v=linear(h,p+'attn.wv.weight').reshape(*ids.shape,c.n_kv_heads,hd)
        heads=[]
        for head in range(c.n_heads):
            kv=head//(c.n_heads//c.n_kv_heads);positions=[]
            for pos in range(t):
                scores=(q[:,pos,head,None,:]*k[:,:pos+1,kv,:]).sum(-1)/math.sqrt(hd)
                positions.append((scores.softmax(-1)[...,None]*v[:,:pos+1,kv,:]).sum(1))
            heads.append(torch.stack(positions,dim=1))
        x=x+linear(torch.stack(heads,dim=2).flatten(2),p+'attn.wo.weight')
        h=norm(x,p+'ffn_norm.weight');gate=linear(h,p+'ffn.w1.weight')
        x=x+linear(gate*gate.sigmoid()*linear(h,p+'ffn.w3.weight'),p+'ffn.w2.weight')
    return norm(x,'norm.weight') @ (weights['tok.weight'] if c.tie_embeddings else weights['out.weight']).T


class ModelReferenceTest(unittest.TestCase):
    def test_forward_loss_and_all_parameter_gradients(self):
        for tied in [False,True]:
            for kv in [1,2,4]:
                for length in [1,7]:
                    with self.subTest(tied=tied,kv=kv,length=length):
                        torch.manual_seed(42)
                        m=Transformer(ModelConfig(vocab_size=23,n_layers=2,dim=16,n_heads=4,n_kv_heads=kv,
                            ffn_dim=24,max_seq_len=8,tie_embeddings=tied));ref=copy.deepcopy(m)
                        ids=torch.randint(0,23,(2,length));targets=torch.randint(0,23,(2,length))
                        logits=m(ids);expected=reference(ref,ids)
                        torch.testing.assert_close(logits,expected,atol=1e-6,rtol=1e-5)
                        loss=m(ids,targets)
                        expected_loss=-expected.log_softmax(-1).gather(-1,targets[...,None]).mean()
                        torch.testing.assert_close(loss,expected_loss,atol=1e-6,rtol=1e-5)
                        loss.backward();expected_loss.backward()
                        for (name,p),(refname,q) in zip(m.named_parameters(),ref.named_parameters()):
                            self.assertEqual(name,refname)
                            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4,msg=name)

    def test_future_tokens_do_not_affect_prefix(self):
        torch.manual_seed(11)
        m=Transformer(ModelConfig(vocab_size=23,n_layers=2,dim=16,n_heads=4,n_kv_heads=2,ffn_dim=24,max_seq_len=8))
        ids=torch.tensor([[1,2,3,4,5,6,7]]);changed=ids.clone();changed[:,3:]=torch.tensor([8,9,10,11])
        torch.testing.assert_close(m(ids)[:,:3],m(changed)[:,:3],atol=0,rtol=0)
        torch.testing.assert_close(m(ids)[:,:3],m(ids[:,:3]),atol=1e-6,rtol=1e-5)

    def test_rope_known_quarter_turn_and_norm(self):
        x=torch.tensor([[[[1.,2.,3.,4.]],[[1.,2.,3.,4.]]]])
        cos=torch.tensor([[1.,1.],[0.,0.]]);sin=torch.tensor([[0.,0.],[1.,1.]])
        actual=apply_rope(x,cos,sin)
        torch.testing.assert_close(actual[:,1],torch.tensor([[[-3.,-4.,1.,2.]]]))
        cos,sin=precompute_rope(4,2,500000.)
        torch.testing.assert_close(apply_rope(x,cos,sin).square().sum(-1),x.square().sum(-1))

    def test_sampler_applies_exactly_one_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);tokens=np.array([1,5,3,8,2],dtype=np.uint16);tokens.tofile(p/'tokens.bin')
            (p/'mix.json').write_text(json.dumps({'sources':{'s':{'weight':1,'files':[str(p/'tokens.bin')]}}}))
            x,y=MixSampler(str(p/'mix.json'),4).batch(0,0,0,1)
            self.assertEqual(x.tolist(),[[1,5,3,8]]);self.assertEqual(y.tolist(),[[5,3,8,2]])
            m=Transformer(ModelConfig(vocab_size=11,n_layers=1,dim=16,n_heads=4,n_kv_heads=2,ffn_dim=24,max_seq_len=4))
            scores=m(x).log_softmax(-1)
            manual=-sum(scores[0,i,int(tokens[i+1])] for i in range(4))/4
            torch.testing.assert_close(m(x,y),manual)
