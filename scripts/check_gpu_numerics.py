"""Bounded CUDA bf16 attention/gradient and long RoPE reference checks."""
import argparse,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model.model import apply_rope,precompute_rope
from common.data_contracts import atomic_json

def run():
    torch.manual_seed(123);torch.backends.cuda.matmul.allow_tf32=False
    results=[]
    for t in [7,256]:
        q=torch.randn(1,4,t,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        k=torch.randn(1,2,t,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        a=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        qr,kr,vr=[x.detach().float().requires_grad_() for x in [q,k,v]]
        scores=qr@kr.repeat_interleave(2,1).transpose(-1,-2)/math.sqrt(32)
        scores=scores.masked_fill(torch.ones(t,t,device='cuda',dtype=torch.bool).triu(1),-torch.inf)
        b=scores.softmax(-1)@vr.repeat_interleave(2,1)
        grad=torch.randn_like(a);ga=torch.autograd.grad(a,[q,k,v],grad);gb=torch.autograd.grad(b,[qr,kr,vr],grad.float())
        torch.testing.assert_close(a.float(),b,atol=.02,rtol=.02)
        for x,y in zip(ga,gb):torch.testing.assert_close(x.float(),y,atol=.04,rtol=.04)
        results.append(dict(length=t,output_max_abs=(a.float()-b).abs().max().item(),gradient_max_abs=max((x.float()-y).abs().max().item() for x,y in zip(ga,gb))))
    t=4096;d=128;x=torch.randn(1,t,2,d,device='cuda',dtype=torch.bfloat16)
    cos,sin=precompute_rope(d,t,500000.,device='cuda');actual=apply_rope(x,cos,sin)
    theta=torch.arange(t,device='cuda',dtype=torch.float64)[:,None]*500000.**(-torch.arange(0,d,2,device='cuda',dtype=torch.float64)/d)
    z=x.double();u,w=z.chunk(2,-1);expected=torch.cat([u*theta.cos()[None,:,None]-w*theta.sin()[None,:,None],u*theta.sin()[None,:,None]+w*theta.cos()[None,:,None]],-1).float()
    torch.testing.assert_close(actual.float(),expected,atol=.035,rtol=.01)
    return dict(status='passed',attention=results,rope_length=t,rope_max_abs=(actual.float()-expected).abs().max().item(),scope='Bounded bf16 attention/gradients and long RoPE only; not full model equivalence',tolerances=dict(attention_atol=.02,attention_rtol=.02,gradient_atol=.04,gradient_rtol=.04,rope_atol=.035,rope_rtol=.01))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();r=run();atomic_json(Path(a.output),r);print(json.dumps(r))
