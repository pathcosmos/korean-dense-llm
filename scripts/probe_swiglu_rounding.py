"""Isolated SwiGLU rounding experiment; does not modify production modules."""
import argparse,sys,json,fcntl,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from common.paths import PROJECT_ROOT
from common.data_contracts import atomic_json,sha256_file
from model.model import FeedForward,get_config

@torch.compiler.disable
def materialized_silu(x):return F.silu(x)

def run(out):
    torch.set_num_threads(2);torch.manual_seed(711);torch.backends.cuda.matmul.allow_tf32=False
    m=FeedForward(get_config('3b')).cuda()
    with torch.no_grad():
        for n,p in m.named_parameters():p.normal_(0,.02/(math.sqrt(56) if n=='w2.weight' else 1))
    x=torch.randn(1,32,3072,device='cuda');target=torch.randn_like(x)
    def baseline(x):return m(x)
    def barrier(x):return m.w2(materialized_silu(m.w1(x))*m.w3(x))
    def fp32_gate(x):
        a,b=m.w1(x),m.w3(x)
        return m.w2((F.silu(a.float())*b.float()).to(a.dtype))
    def calc(fn,amp):
        m.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):y=fn(x);loss=(y.float()*target).mean()
        loss.backward();return y.detach().float(),[p.grad.detach().clone() for p in m.parameters()]
    def error(a,b):
        d=sum((u.double()-v.double()).square().sum() for u,v in zip(a,b));r=sum(v.double().square().sum() for v in b)
        return (d/r).sqrt().item()
    reference=calc(baseline,False);result={}
    for name,fn in [('baseline',baseline),('silu_materialization_barrier',barrier),('fp32_pointwise_gate',fp32_gate)]:
        eager=calc(fn,True);compiled=calc(torch.compile(fn),True)
        result[name]=dict(output_rel_l2=error([compiled[0]],[eager[0]]),gradient_rel_l2=error(compiled[1],eager[1]),eager_vs_fp32_output=error([eager[0]],[reference[0]]),compiled_vs_fp32_output=error([compiled[0]],[reference[0]]))
        del eager,compiled
    report=dict(status='diagnostic_complete',scope='One isolated 3B-shape FFN, random normalized-like input, one seed; no full-model/FSDP/performance adoption proof',results=result,code={p:sha256_file(Path(PROJECT_ROOT)/p) for p in ['scripts/probe_swiglu_rounding.py','model/model.py']})
    atomic_json(out/'report.json',report);atomic_json(out/'.done',dict(sha256=sha256_file(out/'report.json')));print(json.dumps(report))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(PROJECT_ROOT)/a.output;out.mkdir(parents=True,exist_ok=False)
    with (Path(PROJECT_ROOT)/'runs/.gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(out)
